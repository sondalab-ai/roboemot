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
from roboemot.vision import CIRCADIAN, CONVERSATION, AttributeSample, Frame, RateGovernor
from test_vision import (
    CFG,
    CONSENT,
    FACE_MODEL,
    THREE_CROPS,
    CountingEstimator,
    CountingExecutor,
    DeferredExecutor,
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
    assert conversation_vision_scale(CFG, phase) == 1.0  # still talking while the stream runs
    phase.set_listening(False)
    assert conversation_vision_scale(CFG, phase) == SCALE
    speaking["on"] = True
    phase.end_thinking()
    assert conversation_vision_scale(CFG, phase) == SCALE
    phase.set_listening(True)
    assert conversation_vision_scale(CFG, phase) == 1.0  # a barge-in while speaking
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


def test_stretch_divides_by_one_source_only_and_zero_is_off():
    g = RateGovernor(CFG.vision.rates)
    assert g.stretch(0.25, CONVERSATION) == 0.25
    g.set_scale(CONVERSATION, 0.5)
    g.set_scale(CIRCADIAN, 0.5)
    assert g.stretch(0.25, CONVERSATION) == pytest.approx(0.5)  # circadian not applied
    assert g.stretch(0.0, CONVERSATION) == 0.0
    g.set_scale(CONVERSATION, 0.0)
    assert math.isinf(g.stretch(0.25, CONVERSATION))
    assert math.isinf(g.stretch(0.0, CONVERSATION))


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


def identity_stamp(tmp_path, circadian: float, conversation: float, miss=()) -> float:
    """Capture stamp of ``person.recognized`` for a track that starts at 0 under the given
    scales; frames listed in ``miss`` carry no face (a missed vote)."""
    reg, _, pipe, _, got = rig(tmp_path, lambda i: [] if i in miss else [face(100)])
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT, age_group="adult")
    pipe.governor.set_scale(CIRCADIAN, circadian)
    pipe.governor.set_scale(CONVERSATION, conversation)
    for i in range(80):
        pipe.process(Frame(image(i), i * DT))
    rec = got["person.recognized"]
    assert len(rec) == 1
    return rec[0].stamp


DROWSY = CFG.circadian.states["drowsy"].vision_rate_scale
FLOOR = 0.35


@pytest.mark.parametrize(
    ("circadian", "conversation", "expected"),
    [
        (1.0, SCALE, 1.25),  # awake, speaking: votes at 0.25, 0.75, 1.25 s
        (DROWSY, SCALE, 1.5),  # drowsy, speaking: detection at 2 Hz, the burst at every frame
        (1.0, FLOOR, 1.875),  # detection at 2.8 Hz: frames of a 8 Hz camera are 0.125 s apart
        (DROWSY, FLOOR, 2.25),
    ],
)
def test_identity_decision_while_speaking(tmp_path, circadian, conversation, expected):
    """SPEC 5, 13: the 2.5 s identity budget holds while awake or drowsy and speaking."""
    assert identity_stamp(tmp_path, circadian, conversation) == pytest.approx(expected)


def test_identity_decision_while_speaking_survives_one_missed_vote(tmp_path):
    """Awake and drowsy at the default scale, and awake at the floor: one missed vote (the
    face gone from the frame of the deciding vote) still fits 2.5 s."""
    for n, (circadian, conversation) in enumerate(((1.0, SCALE), (DROWSY, SCALE), (1.0, FLOOR))):
        clean = identity_stamp(tmp_path / f"c{n}", circadian, conversation)
        miss = {round(clean / DT)}
        stamp = identity_stamp(tmp_path / f"m{n}", circadian, conversation, miss=miss)
        assert clean < stamp <= 2.5


def test_the_circadian_scale_does_not_stretch_the_burst(tmp_path):
    _, _, pipe, emb, _ = rig(tmp_path, lambda i: [face(100)])
    pipe.governor.set_scale(CIRCADIAN, DROWSY)
    stamps = burst_stamps(pipe, emb, 80)
    period = 1 / (CFG.vision.rates.detection_hz * DROWSY)
    assert CFG.vision.burst_min_gap_s <= period  # the burst runs at every detection frame
    assert np.diff(stamps) == pytest.approx([period] * (len(stamps) - 1))


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


def test_conversation_scale_floor():
    with pytest.raises(ValidationError):
        RoboEmotConfig(vision={"rates": {"conversation_scale": FLOOR - 0.01}})
    RoboEmotConfig(vision={"rates": {"conversation_scale": FLOOR}})
    # Low detection rates are no longer rejected: the face tracker lifetime follows them.
    RoboEmotConfig(vision={"rates": {"detection_hz": 1.0}})


def test_face_tracks_survive_a_detector_slower_than_lost_s(tmp_path):
    """SPEC 5, 13: at the resting scale detection runs every 2.5 s, longer than tracker.lost_s
    (1.5 s); the lifetime follows face_lost_periods detection periods, so one face stays one
    track instead of a new track (and a new burst and age job) at every detection."""
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)])
    resting = CFG.circadian.states["resting"].vision_rate_scale
    pipe.governor.set_scale(CIRCADIAN, resting)
    period = 1 / (CFG.vision.rates.detection_hz * resting)
    assert period > CFG.vision.tracker.lost_s
    for i in range(int(8 * period / DT)):
        pipe.process(Frame(image(i), i * DT))
    tracks = pipe.stats()["tracks"]
    assert tracks["created"] == 1 and tracks["ended"] == 0


def test_face_lifetime_is_kept_until_the_next_face_frame_after_the_rate_goes_up(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)] if i in (0, 10) or i >= 30 else [])
    pipe.governor.set_scale(CIRCADIAN, 0.1)  # detection period 1.25 s, lifetime 3.75 s
    for i in range(3):
        pipe.process(Frame(image(i), i * DT))
    pipe.process(Frame(image(10), 1.25))  # second detection: the track is confirmed
    pipe.governor.set_scale(CIRCADIAN, 1.0)
    pipe.process(Frame(image(30), 3.0))  # 1.75 s later: past lost_s, inside the held lifetime
    assert pipe.stats()["tracks"]["created"] == 1 and pipe.stats()["tracks"]["ended"] == 0


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
        pipe.process(Frame(image(i % 256), i * DT))
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


ROUND = int(RETRY_S / DT) + 4  # frames from one job to the next retry's result, and a margin


def test_a_retried_adult_needs_a_second_agreeing_result_then_retries_stop(tmp_path):
    """SPEC 5: retrying until a crop clears the floor selects for confident errors, so an
    adult reached by a retry counts only when the next confident result agrees."""
    est, ex = SwitchableEstimator(), CountingExecutor()
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    run_jobs(pipe, ex, got, 0, ROUND)
    assert ex.jobs == 2
    est.confident()
    run_jobs(pipe, ex, got, ROUND, ROUND)
    assert ex.jobs == 3
    last = got["percept.face"][-1].payload
    assert (last.age_group, last.possible_minor) == ("unknown", True)  # held, not yet used
    run_jobs(pipe, ex, got, 2 * ROUND, ROUND)
    assert ex.jobs == 4
    last = got["percept.face"][-1].payload
    assert (last.age_group, last.possible_minor) == ("adult", False)
    run_jobs(pipe, ex, got, 3 * ROUND, int(4 * RETRY_S / DT))
    assert ex.jobs == 4 and pipe.stats()["attributes"]["retries"] == 3


class SequenceEstimator(CountingEstimator):
    """One (age, confidence) per crop, in order; the last one repeats."""

    def __init__(self, *samples):
        super().__init__()
        self.samples = [AttributeSample(age, conf, "female", conf) for age, conf in samples]

    def estimate(self, crop):
        self.crops.append(crop.shape)
        return self.samples[min(len(self.crops), len(self.samples)) - 1]


WEAK = CFG.vision.attribute_score_min - 0.01
STRONG = 0.95


def run_until(pipe, ex, got, start: int, jobs: int, after: int = 2) -> int:
    """Process frames until ``jobs`` jobs were sent, then ``after`` more (the result is
    absorbed at the next frame); return the next frame index."""
    i = start
    while ex.jobs < jobs:
        run_jobs(pipe, ex, got, i, 1)
        i += 1
        assert i < start + 4 * ROUND, "the job never came"
    run_jobs(pipe, ex, got, i, after)
    return i + after


def test_an_unknown_between_two_retried_adults_clears_the_candidate(tmp_path):
    est = SequenceEstimator(
        ("adult", WEAK), ("adult", STRONG), ("adult", WEAK), ("adult", STRONG), ("adult", STRONG)
    )
    ex = CountingExecutor()
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    i = run_until(pipe, ex, got, 0, 4)
    assert got["percept.face"][-1].payload.possible_minor  # adult, unknown, adult: not agreed
    run_until(pipe, ex, got, i, 5)
    assert not got["percept.face"][-1].payload.possible_minor


def test_a_retried_minor_counts_at_once(tmp_path):
    est = SequenceEstimator(("adult", WEAK), ("child", STRONG))
    ex = CountingExecutor()
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    run_jobs(pipe, ex, got, 0, 4 * ROUND)
    assert ex.jobs == 2
    last = got["percept.face"][-1].payload
    assert (last.age_group, last.age_source) == ("child", "estimated")


def test_no_retry_while_the_robot_thinks_or_speaks(tmp_path):
    """SPEC 5, 13: a due retry waits for the conversation scale to come back to 1, then runs
    at the first face frame; ``retry_at`` is a capture stamp, never stretched (scale 0 trap)."""
    est, ex = SwitchableEstimator(), CountingExecutor()
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    run_jobs(pipe, ex, got, 0, 4)
    assert ex.jobs == 1
    pipe.governor.set_scale(CONVERSATION, SCALE)
    run_jobs(pipe, ex, got, 4, 3 * ROUND)
    assert ex.jobs == 1
    pipe.governor.set_scale(CONVERSATION, 0.0)  # vision off: nothing runs, nothing is lost
    run_jobs(pipe, ex, got, 4 + 3 * ROUND, 8)
    pipe.governor.set_scale(CONVERSATION, 1.0)
    start = 12 + 3 * ROUND
    seen = run_jobs(pipe, ex, got, start, 4)
    assert ex.jobs == 2
    assert seen[0][1] == 2  # the first face frame back at full rate sends it
    assert got["percept.face"][-1].payload.possible_minor


def test_report_counts_tracks_that_ended_unknown_and_first_estimates_only(tmp_path):
    """SPEC 12A: ``ended_unknown`` counts tracks, ``time_to_estimate_ms`` the first job of
    each track only (a retry 5 s later would hide the first-estimate latency)."""
    est, ex = SwitchableEstimator(), CountingExecutor()
    present = 3 * ROUND
    _, _, pipe, _, got = rig(
        tmp_path, lambda i: [face(100)] if i < present else [], estimator=est, executor=ex
    )
    run_jobs(pipe, ex, got, 0, present + 40)
    attributes = pipe.stats()["attributes"]
    assert ex.jobs >= 3 and attributes["retries"] == ex.jobs - 1
    assert attributes["ended_unknown"] == 1
    assert attributes["time_to_estimate_ms"]["count"] == 1
    assert attributes["outcomes"]["unknown"] == ex.jobs


def test_a_confident_track_does_not_count_as_ended_unknown(tmp_path):
    est, ex = CountingEstimator(), CountingExecutor()
    _, _, pipe, _, got = rig(
        tmp_path, lambda i: [face(100)] if i < 20 else [], estimator=est, executor=ex
    )
    run_jobs(pipe, ex, got, 0, 60)
    assert pipe.stats()["tracks"]["ended"] == 1
    assert pipe.stats()["attributes"]["ended_unknown"] == 0


def test_a_retry_result_for_an_ended_track_is_dropped(tmp_path):
    est, ex = SwitchableEstimator(), DeferredExecutor()
    _, _, pipe, _, got = rig(
        tmp_path, lambda i: [face(100)] if i < ROUND else [], estimator=est, executor=ex
    )
    for i in range(3):
        pipe.process(Frame(image(i), i * DT))
    ex.run_all()  # first job: unknown
    for i in range(3, ROUND):
        pipe.process(Frame(image(i), i * DT))
    assert len(ex.waiting) == 1  # the retry job, still waiting
    for i in range(ROUND, ROUND + 40):  # the face is gone: the track ends
        pipe.process(Frame(image(i), i * DT))
    assert pipe._faces == {}
    ex.run_all()
    pipe.process(Frame(image(0), (ROUND + 40) * DT))
    attributes = pipe.stats()["attributes"]
    assert attributes["dropped_track_gone"] == 1 and attributes["retries"] == 1
    assert attributes["ended_unknown"] == 1


def test_forgetting_a_declared_age_retries_the_hidden_unknown(tmp_path):
    """Review N1: a declared age hid an unknown estimate (no retry), then the person is
    forgotten: the track is a stranger with an unknown age again and is retried."""
    est, ex = SwitchableEstimator(), CountingExecutor()
    reg, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    person = reg.enroll(
        "Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT, age_group="adult"
    )
    run_jobs(pipe, ex, got, 0, 2 * ROUND)
    assert ex.jobs == 1
    reg.forget(person.person_id)
    run_jobs(pipe, ex, got, 2 * ROUND, 4)
    assert ex.jobs == 2 and pipe.stats()["attributes"]["retries"] == 1
    assert got["percept.face"][-1].payload.possible_minor


def test_forgetting_a_fully_declared_person_runs_the_skipped_job(tmp_path):
    est, ex = CountingEstimator(), CountingExecutor()
    # Three crops 0.3 s apart: the job would go after recognition (0.625 s), so it is skipped.
    reg, _, pipe, _, got = rig(
        tmp_path, lambda i: [face(100)], estimator=est, executor=ex, vision=THREE_CROPS
    )
    person = reg.enroll(
        "Ada",
        np.stack([axis(0)] * 5),
        FACE_MODEL,
        consent=CONSENT,
        age_group="adult",
        gender="female",
    )
    run_jobs(pipe, ex, got, 0, 2 * ROUND)
    assert ex.jobs == 0  # skipped: declared age and gender
    reg.forget(person.person_id)
    run_jobs(pipe, ex, got, 2 * ROUND, 12)
    assert ex.jobs == 1


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
