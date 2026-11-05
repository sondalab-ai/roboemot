"""SPEC 5, 13: detect-then-track of the shown object (``vision.object_track``).

Mock stages and a fake single-object tracker; 10 Hz frames, so during a look the object
detector (``objects_active_hz``, 3 Hz) takes about every third frame and the tracker
(``object_track_hz``, 5 Hz) runs on frames between. The scene-change gate and the crop are off.
"""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision import Detection, Frame, models
from roboemot.vision.object_follow import box_iou
from test_vision import CFG, ObjectMock, image, rig

DT = 0.1
CUP = Detection((140, 100, 180, 140), 0.8, "cup")
MOVED = Detection((220, 100, 260, 140), 0.8, "cup")  # 80 px right: no overlap with CUP
TRACK = CFG.vision.object_track
CACHE = models.default_dir()  # read at import: the autouse fixture moves the data folder


class FakeTracker:
    """A single-object tracker that returns ``box`` with ``score``; records its starts."""

    def __init__(self, score: float = 0.7):
        self.score = score
        self.box: tuple[float, float, float, float] | None = None
        self.inits: list[tuple[float, float, float, float]] = []
        self.updates = 0
        self.error: Exception | None = None

    def init(self, img, box):
        self.inits.append(tuple(box))
        self.box = tuple(box)

    def update(self, img):
        if self.error is not None:
            raise self.error
        self.updates += 1
        assert self.box is not None
        return self.box, self.score


def track_rig(tmp_path, *, tracker=None, unavailable=None, **update):
    vision = CFG.vision.model_copy(
        update={
            "object_gate": CFG.vision.object_gate.model_copy(update={"enabled": False}),
            "object_crop": CFG.vision.object_crop.model_copy(update={"enabled": False}),
            **update,
        }
    )
    objs = ObjectMock([CUP])
    tracker = FakeTracker() if tracker is None and unavailable is None else tracker
    _, _, pipe, _, got = rig(
        tmp_path,
        lambda i: [],
        vision=vision,
        object_detector=objs,
        object_tracker=tracker,
        object_track_unavailable=unavailable,
    )
    pipe.governor.set_active("objects", True)  # a look at a shown thing
    return pipe, objs, tracker, got["percept.entity"]


def run(pipe, start: int, stop: int) -> list:
    """Frames ``start`` .. ``stop - 1`` at ``DT``; the reports."""
    return [pipe.process(Frame(image(i), i * DT)) for i in range(start, stop)]


def report(pipe) -> dict:
    return pipe.stats()["object_track"]


def tracker_stamps(reports, seen) -> list[float]:
    """Stamps of the observations published on frames the object detector did not take."""
    detector = {r.stamp for r in reports if r.objects_detected}
    return [m.stamp for m in seen if m.stamp not in detector]


def test_the_shown_track_seeds_the_tracker_on_the_detector_box(tmp_path):
    pipe, _, tracker, _ = track_rig(tmp_path)
    run(pipe, 0, 5)  # two object runs confirm the cup
    (track_id,) = pipe._entities
    assert pipe.follower.target == track_id
    assert tracker.inits == [CUP.box]
    assert report(pipe)["seeds"] == 1 and report(pipe)["state"] == "following"


def test_no_look_no_tracker(tmp_path):
    pipe, _, tracker, _ = track_rig(tmp_path)
    pipe.governor.set_active("objects", False)
    run(pipe, 0, 40)
    assert tracker.inits == [] and tracker.updates == 0
    assert report(pipe)["state"] == "idle"


def test_a_detector_run_on_the_tracked_box_confirms_it_and_the_tracker_publishes_between(
    tmp_path,
):
    pipe, _, tracker, seen = track_rig(tmp_path)
    reports = run(pipe, 0, 30)
    r = report(pipe)
    assert r["seeds"] == 1 and r["reseeds"] == 0 and r["confirms"] >= 5
    assert r["updates"] == tracker.updates > 0
    between = tracker_stamps(reports, seen)
    assert len(between) == tracker.updates  # one observation per tracker update
    assert all(r.objects_detected is False for r in reports if r.stamp in between)
    period = 1 / CFG.vision.rates.object_track_hz
    assert min(b - a for a, b in zip(between, between[1:], strict=False)) >= period - 1e-9
    timing = pipe.stats()["stage_timing"]["object_track"]
    assert timing["runs"] == tracker.updates


def test_a_detector_miss_is_bridged_and_the_track_follows_the_thing(tmp_path):
    """The cup moves 80 px while the detector misses it: the tracker keeps the world entry
    fresh, and the track follows the tracker's box so the next detection matches it."""
    pipe, objs, tracker, seen = track_rig(tmp_path)
    run(pipe, 0, 5)
    (track_id,) = pipe._entities
    objs.dets = []
    tracker.box = MOVED.box
    reports = run(pipe, 5, 20)  # 1.5 s, under confirm_max_s
    assert report(pipe)["bridged"] >= 3 and report(pipe)["lost"]["timeout"] == 0
    assert tracker_stamps(reports, seen)  # published while the detector missed it
    assert seen[-1].stamp == pytest.approx(19 * DT, abs=DT)
    objs.dets = [MOVED]
    run(pipe, 20, 25)
    assert list(pipe._entities) == [track_id]  # no new track for the moved cup
    assert report(pipe)["confirms"] >= 1 and pipe.follower.target == track_id


def test_without_the_tracker_the_moved_thing_starts_a_new_track(tmp_path):
    """The baseline the bridge fixes: a constant-velocity track left at the old place."""
    pipe, objs, _, _ = track_rig(tmp_path, unavailable="ModelError: vittrack: missing")
    run(pipe, 0, 5)
    objs.dets = []
    run(pipe, 5, 20)
    objs.dets = [MOVED]
    run(pipe, 20, 25)
    assert len(pipe._entities) == 2
    assert report(pipe) | {} == {
        "state": "unavailable",
        "reason": "ModelError: vittrack: missing",
        "seeds": 0,
        "reseeds": 0,
        "updates": 0,
        "confirms": 0,
        "bridged": 0,
        "lost": report(pipe)["lost"],
        "score": None,
    }


def test_the_follow_ends_when_the_detector_has_not_confirmed_it_for_too_long(tmp_path):
    pipe, objs, tracker, seen = track_rig(tmp_path)
    run(pipe, 0, 5)
    last_confirm = 4 * DT  # the seed run
    objs.dets = []
    stop = round((last_confirm + TRACK.confirm_max_s) / DT) + 5
    run(pipe, 5, stop)
    assert report(pipe)["lost"]["timeout"] == 1 and pipe.follower.target is None
    assert seen[-1].stamp <= last_confirm + TRACK.confirm_max_s + 1e-9
    updates = tracker.updates
    run(pipe, stop, stop + 10)
    assert tracker.updates == updates  # nothing to follow until the detector sees it


def test_a_low_tracker_score_ends_the_follow_and_publishes_nothing(tmp_path):
    pipe, objs, tracker, seen = track_rig(tmp_path)
    run(pipe, 0, 5)
    objs.dets = []
    tracker.score = TRACK.score_min / 2
    tracker.box = MOVED.box  # a drifting box: never published
    count = len(seen)
    run(pipe, 5, 15)
    assert report(pipe)["lost"]["score"] == 1 and tracker.updates == 1
    assert len(seen) == count
    assert report(pipe)["score"] == round(TRACK.score_min / 2, 2)


def test_the_tracker_starts_again_when_the_detector_box_is_elsewhere(tmp_path):
    """The tracker slid 16 px off the cup: the detector still matches the track, and the
    tracker starts again on the detector's box."""
    pipe, _, tracker, _ = track_rig(tmp_path)
    run(pipe, 0, 5)
    (track_id,) = pipe._entities
    slid = (156.0, 100.0, 196.0, 140.0)
    assert box_iou(slid, CUP.box) < TRACK.reseed_iou
    tracker.box = slid
    run(pipe, 5, 10)
    assert report(pipe)["reseeds"] >= 1 and list(pipe._entities) == [track_id]
    assert tracker.inits[-1] == CUP.box and tracker.box == CUP.box


def test_a_tracker_that_drifted_away_moves_to_the_cups_new_track(tmp_path):
    """The tracker drifted onto the background and pulled the old track with it, so the
    detector's cup starts a new track: once that one is confirmed and shown, the tracker
    starts on it; a thing of another class never takes the follow over."""
    pipe, objs, tracker, _ = track_rig(tmp_path, object_categories=["cup", "couch"])
    run(pipe, 0, 5)
    (old,) = pipe._entities
    tracker.box = (40.0, 20.0, 80.0, 60.0)
    objs.dets = [Detection((100, 60, 220, 180), 0.8, "couch")]  # near the centre
    run(pipe, 5, 15)
    assert pipe.follower.target == old and report(pipe)["bridged"] >= 2
    objs.dets = [CUP]
    run(pipe, 15, 25)
    assert pipe.follower.target not in (None, old)
    assert pipe._entities[pipe.follower.target].kind == "object"
    assert report(pipe)["reseeds"] == 1 and tracker.inits[-1] == CUP.box


def test_the_follow_ends_with_the_look_and_with_the_switch(tmp_path):
    pipe, _, tracker, _ = track_rig(tmp_path)
    run(pipe, 0, 5)
    pipe.governor.set_active("objects", False)
    run(pipe, 5, 6)
    assert report(pipe)["lost"]["episode_end"] == 1 and pipe.follower.target is None
    pipe.governor.set_active("objects", True)
    run(pipe, 6, 10)
    assert pipe.follower.target is not None
    off = pipe.cfg.model_copy(update={"object_track": TRACK.model_copy(update={"enabled": False})})
    pipe.apply_config(off, pipe.identity)
    updates = tracker.updates
    run(pipe, 10, 20)
    assert report(pipe)["lost"]["disabled"] == 1 and report(pipe)["state"] == "off"
    assert tracker.updates == updates and len(tracker.inits) == 2


def test_a_backend_error_ends_the_follow_and_is_counted(tmp_path, caplog):
    pipe, _, tracker, _ = track_rig(tmp_path)
    run(pipe, 0, 5)
    tracker.error = RuntimeError("frame data")
    run(pipe, 5, 7)
    assert report(pipe)["lost"]["error"] == 1
    assert "frame data" not in caplog.text and "RuntimeError" in caplog.text


def test_a_reset_and_a_sleep_end_the_follow(tmp_path):
    pipe, _, _, _ = track_rig(tmp_path)
    run(pipe, 0, 5)
    pipe.clear_overlay()
    assert report(pipe)["lost"]["asleep"] == 1
    run(pipe, 5, 10)
    pipe.reset()
    assert report(pipe)["lost"]["reset"] == 1 and pipe.follower.target is None


def test_the_tracker_settings_are_validated():
    with pytest.raises(ValueError):
        RoboEmotConfig.model_validate({"vision": {"object_track": {"score_min": 1.5}}})
    with pytest.raises(ValueError):
        RoboEmotConfig.model_validate({"vision": {"rates": {"object_track_hz": 0}}})
    on = RoboEmotConfig().vision.object_track
    assert on.enabled and 0 < on.score_min < 1 and on.confirm_max_s > 0


# ------------------------------------------------------------------ real VitTrack model


def textured(seed: int, shape: tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (*shape, 3), dtype=np.uint8)


def scene(x: int, y: int, patch: np.ndarray | None) -> np.ndarray:
    """A dim noise background with ``patch`` (or nothing) at ``(x, y)``."""
    img = textured(0, (240, 320)) // 4
    if patch is not None:
        h, w = patch.shape[:2]
        img[y : y + h, x : x + w] = patch
    return img


@pytest.mark.models
def test_real_vittrack_follows_a_moving_patch_and_scores_its_loss():
    from roboemot.vision import onnx

    root = CACHE
    spec = models.load_manifest()[onnx.VITTRACK_NAME]
    if models.status(spec, root) != "ok":
        pytest.skip(f"vittrack missing in {root} (roboemot models fetch vittrack)")
    onnx.configure(CFG.vision.threads)
    tracker = onnx.VitTracker(models.check_file(root, spec.files[0]))
    patch = textured(1, (40, 32))
    x, y = 60, 100
    tracker.init(scene(x, y, patch), (x, y, x + 32, y + 40))
    for _ in range(30):
        x += 4
        box, score = tracker.update(scene(x, y, patch))
        assert score >= TRACK.score_min
        assert box_iou(box, (x, y, x + 32, y + 40)) > 0.5
    _, score = tracker.update(scene(x, y, None))  # the thing is gone
    assert score < TRACK.score_min
