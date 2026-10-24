"""SPEC 5, 8, 13: lighter vision while the robot thinks or speaks.

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
    axis,
    crowd,
    entity_rig,
    face,
    image,
    rig,
)

DT = 1 / CFG.vision.rates.detection_hz  # one frame per detection period at full rate
SCALE = CFG.vision.rates.conversation_scale


# ------------------------------------------------------------------ conversation scale


def test_defaults():
    assert SCALE == 0.5


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
FLOOR = 0.5  # the lowest conversation_scale allowed, also the default


@pytest.mark.parametrize(
    ("circadian", "conversation", "expected"),
    [
        (1.0, SCALE, 1.25),  # awake, speaking: votes at 0.25, 0.75, 1.25 s
        (DROWSY, SCALE, 1.5),  # drowsy, speaking: detection at 2 Hz, the burst at every frame
    ],
)
def test_identity_decision_while_speaking(tmp_path, circadian, conversation, expected):
    """SPEC 5, 13: the 2.5 s identity budget holds while awake or drowsy and speaking."""
    assert identity_stamp(tmp_path, circadian, conversation) == pytest.approx(expected)


def test_identity_decision_while_speaking_survives_one_missed_vote(tmp_path):
    """Awake and drowsy at the floor (the default): one missed vote (the face gone from the
    frame of the deciding vote) still fits 2.5 s."""
    assert FLOOR == SCALE
    for n, (circadian, conversation) in enumerate(((1.0, FLOOR), (DROWSY, FLOOR))):
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
