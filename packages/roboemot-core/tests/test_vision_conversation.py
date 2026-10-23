"""SPEC 5, 8, 13: lighter vision while the robot thinks or speaks, and unknown ages retried.

Mock stages only: no models, no camera.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation.phase import ConversationPhase
from roboemot.runtime.ticks import conversation_vision_scale
from roboemot.vision import CIRCADIAN, CONVERSATION, Frame, RateGovernor
from test_vision import (
    CFG,
    CONSENT,
    FACE_MODEL,
    CountingEstimator,
    CountingExecutor,
    axis,
    crowd,
    entity_rig,
    face,
    image,
    rig,
)

DT = 1 / CFG.vision.rates.detection_hz  # one frame per detection period at full rate
SCALE = CFG.vision.rates.conversation_scale
RETRY_S = CFG.vision.attribute_retry_s


# ------------------------------------------------------------------ conversation scale


def test_defaults():
    assert SCALE == 0.5 and RETRY_S == 5.0


def test_phase_sets_the_scale_while_thinking_or_speaking_only():
    speaking = {"on": False}
    phase = ConversationPhase(lambda now: speaking["on"])
    assert conversation_vision_scale(CFG, None) == 1.0
    assert conversation_vision_scale(CFG, phase) == 1.0  # idle
    phase.set_listening(True)
    assert conversation_vision_scale(CFG, phase) == 1.0  # the face attributes the speech
    phase.begin_thinking()
    assert conversation_vision_scale(CFG, phase) == SCALE
    speaking["on"] = True
    phase.end_thinking()
    assert conversation_vision_scale(CFG, phase) == SCALE
    speaking["on"] = False
    phase.set_listening(False)
    assert conversation_vision_scale(CFG, phase) == 1.0


def test_conversation_scale_multiplies_the_other_scales_and_is_reported(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [])
    pipe.governor.set_scale(CIRCADIAN, 0.5)
    pipe.governor.set_scale(CONVERSATION, SCALE)
    stats = pipe.stats()
    assert stats["rate_scale"] == pytest.approx(0.5 * SCALE)
    assert stats["rate_scales"] == {CIRCADIAN: 0.5, CONVERSATION: SCALE}
    assert pipe.governor.period("detection") == pytest.approx(
        1 / (CFG.vision.rates.detection_hz * 0.5 * SCALE)
    )


def test_detection_rate_drops_while_thinking_and_comes_back_when_idle(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [])
    frames = 80
    for i in range(frames):
        pipe.process(Frame(image(i), i * DT))
    full = pipe.face_detector.calls
    pipe.governor.set_scale(CONVERSATION, SCALE)
    for i in range(frames):
        pipe.process(Frame(image(i), (frames + i) * DT))
    thinking = pipe.face_detector.calls - full
    pipe.governor.set_scale(CONVERSATION, 1.0)
    for i in range(frames):
        pipe.process(Frame(image(i), (2 * frames + i) * DT))
    idle = pipe.face_detector.calls - full - thinking
    assert full == frames and idle == frames
    assert thinking == pytest.approx(frames * SCALE, abs=1)


def test_stretch_divides_by_the_scale_and_zero_is_off():
    g = RateGovernor(CFG.vision.rates)
    assert g.stretch(0.25) == 0.25
    g.set_scale(CONVERSATION, 0.5)
    assert g.stretch(0.25) == pytest.approx(0.5)
    assert g.stretch(0.0) == 0.0
    g.set_scale(CIRCADIAN, 0.0)
    assert math.isinf(g.stretch(0.25)) and math.isinf(g.stretch(0.0))


def burst_stamps(pipe, emb, frames: int) -> list[float]:
    clock = {"t": 0.0}
    emb.stamp_of = lambda: clock["t"]
    for i in range(frames):
        clock["t"] = i * DT
        pipe.process(Frame(image(i), clock["t"]))
    return emb.stamps[: CFG.vision.embedding_burst]


def test_the_face_burst_gap_is_stretched_by_the_governor_scale(tmp_path):
    _, _, pipe, emb, _ = rig(tmp_path, lambda i: [face(100)])
    pipe.governor.set_scale(CONVERSATION, SCALE)
    stamps = burst_stamps(pipe, emb, 80)
    gap = CFG.vision.burst_min_gap_s / SCALE
    assert len(stamps) == CFG.vision.embedding_burst
    assert np.diff(stamps).min() >= gap - 1e-9
    assert np.diff(stamps).max() <= gap + 2 * DT + 1e-9


def test_the_face_burst_keeps_its_gap_at_full_rate(tmp_path):
    _, _, pipe, emb, _ = rig(tmp_path, lambda i: [face(100)])
    stamps = burst_stamps(pipe, emb, 80)
    gap = CFG.vision.burst_min_gap_s
    assert np.diff(stamps).min() >= gap - 1e-9
    assert np.diff(stamps).max() <= gap + DT + 1e-9


def test_the_entity_burst_gap_is_stretched_by_the_governor_scale(tmp_path):
    rates = CFG.vision.rates.model_copy(update={"objects_hz": 4.0, "appearance_hz": 0.05})
    pipe, app = entity_rig(tmp_path, crowd(1), rates=rates, entity_burst_min_gap_s=1.0)
    pipe.governor.set_scale(CONVERSATION, SCALE)
    stamps = []
    for i in range(48):
        before = len(app.boxes)
        pipe.process(Frame(image(i), i * 0.25))
        if len(app.boxes) > before:
            stamps.append(i * 0.25)
    assert len(stamps) >= 3
    assert all(b - a >= 1.0 / SCALE - 1e-9 for a, b in zip(stamps, stamps[1:], strict=False))


def test_a_conversation_scale_that_would_end_tracks_is_rejected():
    lost = CFG.vision.tracker.lost_s
    hz = CFG.vision.rates.detection_hz
    with pytest.raises(ValidationError, match="conversation_scale"):
        RoboEmotConfig(vision={"rates": {"conversation_scale": 1.0 / (hz * lost)}})
    with pytest.raises(ValidationError):
        RoboEmotConfig(vision={"rates": {"conversation_scale": 0.0}})
    RoboEmotConfig(vision={"rates": {"conversation_scale": 2.0 / (hz * lost)}})


# -------------------------------------------------------------------- age retries


class SwitchableEstimator(CountingEstimator):
    """Unknown (below the score floor) until ``confident`` is called."""

    def __init__(self, age="adult"):
        super().__init__(age=age, conf=CFG.vision.attribute_score_min - 0.01)
        self.age = age

    def confident(self):
        self.sample = CountingEstimator(age=self.age).sample


def run_jobs(pipe, ex, got, start: int, frames: int) -> list[tuple[float, int]]:
    """Process ``frames`` frames; return (stamp, jobs so far) per frame and check that every
    published percept is a possible minor while the age is unknown."""
    out = []
    for i in range(start, start + frames):
        pipe.process(Frame(image(i), i * DT))
        out.append((i * DT, ex.jobs))
        if got["percept.face"] and got["percept.face"][-1].payload.age_group == "unknown":
            assert got["percept.face"][-1].payload.possible_minor
    return out


def test_an_unknown_age_is_retried_after_the_interval_not_before(tmp_path):
    est, ex = SwitchableEstimator(), CountingExecutor()
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    seen = run_jobs(pipe, ex, got, 0, int(3 * RETRY_S / DT))
    first = next(t for t, jobs in seen if jobs == 1)
    second = next(t for t, jobs in seen if jobs == 2)
    absorbed = first + DT  # the inline job's result is polled at the next frame
    assert second >= absorbed + RETRY_S - 1e-9
    assert second <= absorbed + RETRY_S + DT + 1e-9
    assert pipe.stats()["attributes"]["retries"] == ex.jobs - 1 >= 2
    assert got["percept.face"][-1].payload.possible_minor


def test_a_confident_result_stops_the_retries(tmp_path):
    est, ex = SwitchableEstimator(), CountingExecutor()
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    frames = int(RETRY_S / DT) + 4
    run_jobs(pipe, ex, got, 0, frames)
    assert ex.jobs == 2
    est.confident()
    more = int(RETRY_S / DT) + 4
    run_jobs(pipe, ex, got, frames, more)
    assert ex.jobs == 3
    last = got["percept.face"][-1].payload
    assert (last.age_group, last.possible_minor) == ("adult", False)
    run_jobs(pipe, ex, got, frames + more, int(4 * RETRY_S / DT))
    assert ex.jobs == 3 and pipe.stats()["attributes"]["retries"] == 2


def test_a_confident_minor_is_not_retried(tmp_path):
    est, ex = CountingEstimator(age="child"), CountingExecutor()
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    run_jobs(pipe, ex, got, 0, int(4 * RETRY_S / DT))
    assert ex.jobs == 1 and pipe.stats()["attributes"]["retries"] == 0
    assert got["percept.face"][-1].payload.possible_minor


def test_a_declared_age_is_not_retried(tmp_path):
    est, ex = SwitchableEstimator(), CountingExecutor()
    reg, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    # Age declared, gender not: the job still runs, its unknown age is never retried.
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT, age_group="adult")
    run_jobs(pipe, ex, got, 0, int(4 * RETRY_S / DT))
    assert ex.jobs == 1 and pipe.stats()["attributes"]["retries"] == 0
    last = got["percept.face"][-1].payload
    assert (last.age_group, last.age_source) == ("adult", "declared")


def test_retry_interval_range():
    with pytest.raises(ValidationError):
        RoboEmotConfig(vision={"attribute_retry_s": 0.0})
    assert RoboEmotConfig(vision={"attribute_retry_s": 30}).vision.attribute_retry_s == 30
