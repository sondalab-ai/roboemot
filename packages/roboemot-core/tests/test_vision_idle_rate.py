"""SPEC 5, 13: a lower face detection rate while no face is in view (owner decision 2026-11-11).

Mock stages only: no models, no camera. Time is the capture stamp of each frame.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision import CIRCADIAN, Frame, RateGovernor
from test_vision import CFG, face, image, rig

RATES = CFG.vision.rates
FULL = RATES.detection_hz
IDLE = RATES.detection_idle_hz
AFTER = RATES.detection_idle_after_s
CAMERA_DT = 1 / 40  # a 40 fps camera: every period in these tests is a whole number of frames


class StampDetector:
    """Faces as a function of the capture stamp; records the stamp of every run."""

    def __init__(self, faces_at):
        self.faces_at = faces_at
        self.now = 0.0
        self.stamps: list[float] = []

    def detect(self, img):
        self.stamps.append(self.now)
        return self.faces_at(self.now)


def stream(pipe, start: float, end: float, dt: float = CAMERA_DT) -> None:
    """Feed frames stamped ``start``, ``start + dt``, ... up to ``end`` (excluded)."""
    det = pipe.face_detector
    i = 0
    while (stamp := start + i * dt) < end - 1e-9:
        det.now = stamp
        pipe.process(Frame(image(i % 256), stamp))
        i += 1


def idle_rig(tmp_path, faces_at=lambda t: [], vision=None):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [], vision=vision)
    pipe.face_detector = StampDetector(faces_at)
    return pipe


def runs_between(det: StampDetector, a: float, b: float) -> int:
    return sum(a <= s < b for s in det.stamps)


def vision_with(**rates):
    return RoboEmotConfig(vision={"rates": rates}).vision


# ----------------------------------------------------------------------------- config


def test_defaults():
    assert IDLE == 4.0
    assert AFTER == 10.0
    assert IDLE <= FULL


def test_an_idle_rate_above_the_detection_rate_has_no_effect():
    """An old config with detection_hz below the idle default still loads (no validator): the
    idle rate only ever lowers the rate."""
    low = RoboEmotConfig(vision={"rates": {"detection_hz": 3.0}}).vision.rates
    assert low.detection_idle_hz == IDLE > low.detection_hz
    g = RateGovernor(low)
    assert g.period("detection") == pytest.approx(1 / 3.0)
    g.set_idle("detection", True)
    assert g.idle("detection")
    assert g.period("detection") == pytest.approx(1 / 3.0)
    g.set_scale(CIRCADIAN, 0.5)
    assert g.period("detection") == pytest.approx(1 / 1.5)


def test_detection_below_the_idle_default_runs_at_its_own_rate_in_both_states(tmp_path):
    low = 2.5  # below the idle default; 0.4 s is a whole number of camera frames
    pipe = idle_rig(tmp_path, vision=vision_with(detection_hz=low))
    stream(pipe, 0.0, AFTER + 20)
    det = pipe.face_detector
    assert pipe.governor.idle("detection")
    assert runs_between(det, 0, AFTER) == pytest.approx(low * AFTER, abs=1)
    assert runs_between(det, AFTER + 10, AFTER + 20) == pytest.approx(low * 10, abs=1)


def test_idle_after_range():
    for bad in (-1, 601):
        with pytest.raises(ValidationError):
            RoboEmotConfig(vision={"rates": {"detection_idle_after_s": bad}})
    for good in (0, 600):
        RoboEmotConfig(vision={"rates": {"detection_idle_after_s": good}})


def test_both_fields_are_hot_reloadable_like_detection_hz():
    fields = type(RATES).model_fields
    restart = {
        name: fields[name].json_schema_extra["restart"]
        for name in ("detection_hz", "detection_idle_hz", "detection_idle_after_s")
    }
    assert restart == dict.fromkeys(restart, False)


# --------------------------------------------------------------------------- governor


def test_governor_idle_period_is_scaled_like_every_rate():
    g = RateGovernor(RATES)
    assert g.period("detection") == pytest.approx(1 / FULL)
    g.set_idle("detection", True)
    assert g.idle("detection")
    assert g.period("detection") == pytest.approx(1 / IDLE)
    g.set_scale(CIRCADIAN, 0.5)
    assert g.period("detection") == pytest.approx(1 / (IDLE * 0.5))
    g.set_idle("detection", False)
    assert g.period("detection") == pytest.approx(1 / (FULL * 0.5))


def test_governor_ignores_idle_for_a_stage_without_an_idle_rate():
    g = RateGovernor(RATES)
    g.set_idle("objects", True)
    assert not g.idle("objects")
    assert g.period("objects") == pytest.approx(1 / RATES.objects_hz)


def test_governor_reset_clears_idle():
    g = RateGovernor(RATES)
    g.set_idle("detection", True)
    g.reset()
    assert g.period("detection") == pytest.approx(1 / FULL)


# --------------------------------------------------------------------------- pipeline


def test_an_empty_room_drops_to_the_idle_rate_after_the_gap(tmp_path):
    pipe = idle_rig(tmp_path)
    stream(pipe, 0.0, AFTER + 20)
    det = pipe.face_detector
    assert runs_between(det, 0, AFTER) == pytest.approx(FULL * AFTER, abs=1)
    assert runs_between(det, AFTER + 10, AFTER + 20) == pytest.approx(IDLE * 10, abs=1)
    assert pipe.governor.idle("detection")
    assert pipe.stats()["detection_idle"] == {
        "idle": True,
        "no_face_s": pytest.approx(AFTER + 20 - CAMERA_DT, abs=1e-6),
    }


def test_one_face_detection_returns_the_full_rate_at_once(tmp_path):
    arrive = AFTER + 5.0
    pipe = idle_rig(tmp_path, lambda t: [face(100)] if t >= arrive else [])
    stream(pipe, 0.0, arrive + 2.0)
    det = pipe.face_detector
    assert runs_between(det, AFTER + 1, arrive) == pytest.approx(IDLE * (arrive - AFTER - 1), abs=1)
    first = min(s for s in det.stamps if s >= arrive)
    assert first - arrive <= 1 / IDLE + CAMERA_DT  # worst case: one idle period
    after = [s for s in det.stamps if s >= first]
    gaps = [b - a for a, b in zip(after, after[1:], strict=False)]
    assert max(gaps) <= 1 / FULL + CAMERA_DT  # the next run is a full-rate period later
    assert not pipe.governor.idle("detection")
    assert pipe.stats()["detection_idle"] == {"idle": False, "no_face_s": None}


def test_a_single_face_detection_restarts_the_gap(tmp_path):
    """A tentative track that is dropped at the next miss still counts as a face: the
    full rate lasts detection_idle_after_s from when the room is empty again."""
    blip = AFTER + 5.0
    pipe = idle_rig(tmp_path, lambda t: [face(100)] if blip <= t < blip + 0.3 else [])
    stream(pipe, 0.0, blip + AFTER + 10)
    det = pipe.face_detector
    assert runs_between(det, blip + 1, blip + AFTER) == pytest.approx(FULL * (AFTER - 1), abs=1)
    assert runs_between(det, blip + AFTER + 2, blip + AFTER + 10) == pytest.approx(IDLE * 8, abs=1)


def test_idle_equal_to_the_detection_rate_keeps_the_old_behaviour(tmp_path):
    pipe = idle_rig(tmp_path, vision=vision_with(detection_idle_hz=FULL))
    stream(pipe, 0.0, AFTER + 20)
    assert len(pipe.face_detector.stamps) == pytest.approx(FULL * (AFTER + 20), abs=1)


def test_the_idle_rate_follows_the_circadian_scale(tmp_path):
    pipe = idle_rig(tmp_path)
    pipe.governor.set_scale(CIRCADIAN, 0.5)
    stream(pipe, 0.0, AFTER + 20)
    det = pipe.face_detector
    assert runs_between(det, 0, AFTER) == pytest.approx(FULL * 0.5 * AFTER, abs=1)
    assert runs_between(det, AFTER + 10, AFTER + 20) == pytest.approx(IDLE * 0.5 * 10, abs=1)


def test_the_gap_counts_while_vision_is_off(tmp_path):
    """Capture time, not detections: a paused governor (scale 0) still ages the gap."""
    pipe = idle_rig(tmp_path)
    pipe.governor.set_scale(CIRCADIAN, 0.0)
    stream(pipe, 0.0, AFTER + 1)
    pipe.governor.set_scale(CIRCADIAN, 1.0)
    stream(pipe, AFTER + 1, AFTER + 11)
    assert runs_between(pipe.face_detector, AFTER + 1, AFTER + 11) == pytest.approx(
        IDLE * 10, abs=1
    )


def test_hot_reload_changes_the_idle_rate_and_gap(tmp_path):
    pipe = idle_rig(tmp_path)
    stream(pipe, 0.0, AFTER + 5)
    pipe.apply_config(vision_with(detection_idle_hz=2.0, detection_idle_after_s=60.0), CFG.identity)
    stream(pipe, AFTER + 5, AFTER + 15)  # 15 s without a face is under the new 60 s gap
    assert runs_between(pipe.face_detector, AFTER + 6, AFTER + 15) == pytest.approx(FULL * 9, abs=1)
    pipe.apply_config(vision_with(detection_idle_hz=2.0, detection_idle_after_s=0.0), CFG.identity)
    stream(pipe, AFTER + 15, AFTER + 25)
    assert runs_between(pipe.face_detector, AFTER + 16, AFTER + 25) == pytest.approx(2.0 * 9, abs=1)


def test_reset_restarts_the_gap_and_replays_are_deterministic(tmp_path):
    arrive = AFTER + 3.0
    pipe = idle_rig(tmp_path, lambda t: [face(100)] if arrive <= t < arrive + 1 else [])
    stream(pipe, 0.0, 2 * AFTER + 20)
    first = list(pipe.face_detector.stamps)
    assert runs_between(pipe.face_detector, AFTER + 1, arrive) == pytest.approx(
        IDLE * 2, abs=1
    )  # idle before the face, full after it, idle again once it is gone
    assert runs_between(pipe.face_detector, 2 * AFTER + 10, 2 * AFTER + 20) == pytest.approx(
        IDLE * 10, abs=1
    )
    pipe.reset()
    pipe.face_detector.stamps.clear()
    stream(pipe, 0.0, 2 * AFTER + 20)
    assert pipe.face_detector.stamps == first
