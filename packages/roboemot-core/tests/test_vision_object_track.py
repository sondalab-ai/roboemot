"""SPEC 5, 13: detect-then-track of the shown object (``vision.object_track``).

Mock stages and a fake single-object tracker; 10 Hz frames, so during a look the object
detector (``objects_active_hz``, 3 Hz) takes about every third frame and the tracker
(``object_track_hz``, 5 Hz) runs on frames between. The scene-change gate and the crop are off.
"""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig, TrackerConfig
from roboemot.vision import CIRCADIAN, Detection, Frame, Tracker, models
from roboemot.vision.governor import CONVERSATION
from roboemot.vision.object_follow import box_iou
from test_vision import CFG, H, ObjectMock, W, image, rig

DT = 0.1
CUP = Detection((140, 100, 180, 140), 0.8, "cup")
MOVED = Detection((220, 100, 260, 140), 0.8, "cup")  # 80 px right: no overlap with CUP
TRACK = CFG.vision.object_track
CACHE = models.default_dir()  # read at import: the autouse fixture moves the data folder


class FakeTracker:
    """A single-object tracker that returns ``box`` with ``score``; records its starts.

    ``box_of`` (frame index -> box) replaces ``box`` when set; ``grow`` scales the box about
    its centre at every update; ``hook`` runs inside ``update`` (a race from another thread);
    ``init_error`` makes ``init`` raise."""

    def __init__(self, score: float = 0.7):
        self.score = score
        self.box: tuple[float, float, float, float] | None = None
        self.inits: list[tuple[float, float, float, float]] = []
        self.updates = 0
        self.update_frames: list[int] = []
        self.error: Exception | None = None
        self.init_error: Exception | None = None
        self.box_of = None
        self.grow = 1.0
        self.hook = None

    def init(self, img, box):
        if self.init_error is not None:
            raise self.init_error
        self.inits.append(tuple(box))
        self.box = tuple(box)

    def update(self, img):
        if self.error is not None:
            raise self.error
        self.updates += 1
        self.update_frames.append(int(img[0, 0, 0]))
        if self.hook is not None:
            self.hook()
        if self.box_of is not None:
            self.box = self.box_of(int(img[0, 0, 0]))
        assert self.box is not None
        x1, y1, x2, y2 = self.box
        cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
        g = self.grow
        self.box = (cx - g * hw, cy - g * hh, cx + g * hw, cy + g * hh)
        return self.box, self.score


def track_rig(tmp_path, *, tracker=None, unavailable=None, objects=None, **update):
    vision = CFG.vision.model_copy(
        update={
            "object_gate": CFG.vision.object_gate.model_copy(update={"enabled": False}),
            "object_crop": CFG.vision.object_crop.model_copy(update={"enabled": False}),
            **update,
        }
    )
    objs = ObjectMock([CUP]) if objects is None else objects
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


# ------------------------------------------------------------------ review of PR 197


def test_the_init_on_the_detector_frame_is_timed(tmp_path):
    pipe, _, tracker, _ = track_rig(tmp_path)
    run(pipe, 0, 12)
    timing = pipe.stats()["stage_timing"]
    assert timing["object_track_init"]["runs"] == len(tracker.inits) >= 1
    assert timing["object_track"]["runs"] == tracker.updates


def moving_box(i: int) -> tuple[float, float, float, float]:
    """A cup still until the seed (frame 4), then 16 px per frame (160 px/s) to the right."""
    x = 60.0 + 16.0 * max(0, i - 4)
    return (x, 100.0, x + 40.0, 140.0)


class MovingCup:
    """The object detector of a cup moving as ``moving_box`` (frame index from the image)."""

    def detect(self, img):
        return [Detection(moving_box(int(img[0, 0, 0])), 0.8, "cup")]


def test_a_steadily_moving_cup_is_confirmed_not_reseeded(tmp_path):
    """The tracker box is up to one tracker period older than the detector frame; at 160 px/s
    that alone gave an IoU of 0.43 with the detector box and a needless re-seed."""
    tracker = FakeTracker()
    tracker.box_of = moving_box
    pipe, _, _, _ = track_rig(tmp_path, tracker=tracker, objects=MovingCup())
    run(pipe, 0, 14)
    r = report(pipe)
    assert r["seeds"] == 1 and r["confirms"] >= 2 and r["reseeds"] == 0
    assert len(tracker.inits) == 1


def test_a_box_that_grows_past_the_area_ratio_is_dropped_not_published(tmp_path):
    """A drifting ViT box can grow to the whole frame at a high score (0.93 measured)."""
    tracker = FakeTracker(score=0.93)
    pipe, objs, _, seen = track_rig(tmp_path, tracker=tracker)
    run(pipe, 0, 5)
    objs.dets = []
    tracker.grow = 1.6  # area x2.56 per update
    count = len(seen)
    run(pipe, 5, 15)
    assert report(pipe)["lost"]["box"] == 1 and pipe.follower.target is None
    published = seen[count:]
    assert len(published) == 1  # the first grown box (x2.56) passes, the next (x6.6) does not
    assert TRACK.box_area_ratio_max < 1.6**4


def test_a_box_whose_centre_leaves_the_frame_is_dropped(tmp_path):
    pipe, objs, tracker, seen = track_rig(tmp_path)
    run(pipe, 0, 5)
    objs.dets = []
    tracker.box = (-60.0, 100.0, -20.0, 140.0)  # centre left of the image
    count = len(seen)
    run(pipe, 5, 10)
    assert report(pipe)["lost"]["box"] == 1 and len(seen) == count


def test_a_tracker_sighting_publishes_the_lower_of_the_two_scores(tmp_path):
    tracker = FakeTracker(score=0.65)
    pipe, _, _, seen = track_rig(tmp_path, tracker=tracker)
    reports = run(pipe, 0, 12)
    between = set(tracker_stamps(reports, seen))
    assert between
    for m in seen:
        want = min(CUP.score, 0.65) if m.stamp in between else CUP.score
        assert m.payload.confidence == pytest.approx(want)


def test_a_stop_from_another_thread_during_an_update_publishes_nothing(tmp_path):
    """The app thread's clear_overlay (the robot fell asleep) races with the vision thread's
    step: the box the backend returns after the stop is dropped."""
    pipe, objs, tracker, seen = track_rig(tmp_path)
    run(pipe, 0, 5)
    objs.dets = []
    tracker.hook = pipe.clear_overlay
    count = len(seen)
    run(pipe, 5, 7)
    assert report(pipe)["lost"]["asleep"] == 1 and len(seen) == count
    assert pipe.follower.target is None


def test_a_zero_rate_ends_the_follow(tmp_path):
    pipe, _, tracker, _ = track_rig(tmp_path)
    run(pipe, 0, 5)
    pipe.governor.set_scale(CIRCADIAN, 0.0)
    run(pipe, 5, 6)
    assert report(pipe)["lost"]["paused"] == 1 and pipe.follower.target is None
    updates = tracker.updates
    run(pipe, 6, 12)
    assert tracker.updates == updates


BOWL = Detection((142, 102, 182, 142), 0.8, "bowl")  # the held cup, read as a bowl


def test_a_new_track_of_another_class_on_the_tracker_box_takes_the_follow(tmp_path):
    """The detector reads the held cup as a bowl: a new track, overlapping the tracker box."""
    pipe, objs, tracker, _ = track_rig(tmp_path, object_categories=["cup", "bowl"])
    run(pipe, 0, 5)
    (old,) = pipe._entities
    objs.dets = [BOWL]
    run(pipe, 5, 15)
    target = pipe.follower.target
    assert target not in (None, old) and pipe._entities[target].kind == "object"
    assert report(pipe)["reseeds"] == 1 and tracker.inits[-1] == BOWL.box


def test_the_tracker_runs_at_its_scaled_rate_and_never_on_a_detector_frame(tmp_path):
    pipe, _, tracker, _ = track_rig(tmp_path)
    pipe.governor.set_scale(CONVERSATION, 0.5)
    reports = run(pipe, 0, 40)
    detector = {round(r.stamp / DT) for r in reports if r.objects_detected}
    assert tracker.update_frames and not detector & set(tracker.update_frames)
    period = 1 / (CFG.vision.rates.object_track_hz * 0.5)
    assert (np.diff(tracker.update_frames) * DT).min() >= period - 1e-9


def test_an_init_that_raises_ends_the_follow_and_is_counted(tmp_path, caplog):
    tracker = FakeTracker()
    tracker.init_error = RuntimeError("pixels")
    pipe, _, _, _ = track_rig(tmp_path, tracker=tracker)
    run(pipe, 0, 12)
    r = report(pipe)
    assert r["seeds"] == 0 and r["lost"]["error"] >= 1 and pipe.follower.target is None
    assert "pixels" not in caplog.text and "RuntimeError" in caplog.text


LEFT_CUP = Detection((40, 100, 80, 140), 0.8, "cup")
RIGHT_CUP = Detection((240, 100, 280, 140), 0.8, "cup")


def bearing(pipe, det):
    x1, y1, x2, y2 = det.box
    return pipe._camera(W, H).bearing((x1 + x2) / 2, (y1 + y2) / 2)


def test_attention_moving_to_another_thing_switches_the_follow(tmp_path):
    objs = ObjectMock([LEFT_CUP, RIGHT_CUP])
    pipe, _, tracker, _ = track_rig(tmp_path, objects=objs)
    pipe.governor.set_active("objects", True, bearing(pipe, LEFT_CUP))
    run(pipe, 0, 5)
    assert tracker.inits == [LEFT_CUP.box]
    pipe.governor.set_active("objects", True, bearing(pipe, RIGHT_CUP))
    run(pipe, 5, 10)
    assert report(pipe)["lost"]["switched"] == 1 and tracker.inits[-1] == RIGHT_CUP.box


def test_after_a_drift_the_published_bearing_is_back_on_the_cup_soon(tmp_path):
    """The tracker locks onto the background at score 0.7 while the detector keeps seeing the
    cup: the cup gets a new track, and the follow moves to it once that one is confirmed."""
    pipe, _, tracker, seen = track_rig(tmp_path)
    run(pipe, 0, 5)
    cup_az = bearing(pipe, CUP)[0]
    drift = (40.0, 20.0, 80.0, 60.0)
    tracker.box_of = lambda i: drift if len(tracker.inits) == 1 else CUP.box  # until re-seeded
    start = 5 * DT
    reseeded = None
    for i in range(5, 25):
        pipe.process(Frame(image(i), i * DT))
        if reseeded is None and report(pipe)["reseeds"] == 1:
            reseeded = i * DT
    assert reseeded is not None
    period = 1 / CFG.vision.rates.objects_active_hz
    # The new track needs object_confirm_hits detector runs, then the next run re-seeds.
    assert reseeded - start <= (CFG.vision.object_confirm_hits + 1) * period + DT
    after = [m for m in seen if m.stamp > reseeded]
    assert after and all(abs(m.payload.azimuth_rad - cup_az) < 1e-6 for m in after)


def test_a_hot_reload_mid_follow_applies_at_once(tmp_path):
    pipe, objs, tracker, _ = track_rig(tmp_path)
    run(pipe, 0, 5)
    objs.dets = []
    faster = pipe.cfg.model_copy(
        update={
            "object_track": pipe.cfg.object_track.model_copy(update={"confirm_max_s": 0.5}),
            "rates": pipe.cfg.rates.model_copy(update={"object_track_hz": 10.0}),
        }
    )
    pipe.apply_config(faster, pipe.identity)
    first = tracker.updates
    run(pipe, 5, 8)
    assert tracker.updates - first == 3  # every frame the detector does not take
    run(pipe, 8, 12)
    assert report(pipe)["lost"]["timeout"] == 1  # 0.5 s after the seed at 0.4 s


def test_a_seed_box_outside_the_image_is_clipped_into_it():
    pytest.importorskip("cv2")
    from roboemot.vision import onnx

    calls = []

    class Inner:
        def init(self, image, rect):
            calls.append(rect)

    vit = object.__new__(onnx.VitTracker)
    vit._tracker = Inner()
    vit.init(np.zeros((H, W, 3), np.uint8), (400.0, -50.0, 500.0, -10.0))  # right, above
    ((x, y, w, h),) = calls
    assert 0 <= x < W and 0 <= y < H and w >= 1 and h >= 1
    assert x + w <= W and y + h <= H


# ------------------------------------------------------------------ Tracker.follow


def follow_rig():
    tracker = Tracker(TrackerConfig(confirm_hits=2, lost_s=1.0))
    det = Detection((100, 100, 140, 140), 0.8, "cup")
    tracker.update([det], 0.0)
    (track,) = tracker.update([det], 0.5).visible
    return tracker, track, det


def test_follow_moves_the_box_but_keeps_the_detectors_fields():
    tracker, track, det = follow_rig()
    hits, score, detection = track.hits, track.score, track.detection
    assert tracker.follow(track.track_id, (130.0, 100.0, 170.0, 140.0), 0.7) is track
    assert (track.hits, track.score, track.detection) == (hits, score, detection)
    assert track.box[0] > det.box[0] and track.last_stamp == 0.7


def test_follow_refuses_an_unknown_or_unconfirmed_track():
    tracker, track, _ = follow_rig()
    assert tracker.follow(track.track_id + 99, (0.0, 0.0, 10.0, 10.0), 0.7) is None
    tentative = Detection((10, 10, 30, 30), 0.8, "cup")
    tracker.update([Detection((100, 100, 140, 140), 0.8, "cup"), tentative], 0.8)
    young = max(t.track_id for t in tracker._tracks)
    assert young != track.track_id
    assert tracker.follow(young, tentative.box, 0.9) is None


def test_follow_keeps_the_track_alive_and_a_later_detection_matches_it():
    tracker, track, _ = follow_rig()
    stamp, box = 0.5, (100.0, 100.0, 140.0, 140.0)
    for _ in range(10):  # 2 s of followed sightings against lost_s 1 s, moving right
        stamp += 0.2
        box = (box[0] + 8, box[1], box[2] + 8, box[3])
        assert tracker.follow(track.track_id, box, stamp) is track
    assert tracker.expire(stamp) == []
    later = Detection((box[0] + 8, 100, box[2] + 8, 140), 0.8, "cup")
    result = tracker.update([later], stamp + 0.2)
    assert [t.track_id for t in result.visible] == [track.track_id]


def test_follow_twice_at_one_stamp_moves_no_further_than_the_box():
    tracker, track, _ = follow_rig()
    box = (110.0, 100.0, 150.0, 140.0)
    tracker.follow(track.track_id, box, 0.7)
    first = track.box
    tracker.follow(track.track_id, box, 0.7)
    assert track.last_stamp == 0.7
    assert abs(track.box[0] - box[0]) <= abs(first[0] - box[0])  # closer, never past it
    assert track.box[0] <= box[0]


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
