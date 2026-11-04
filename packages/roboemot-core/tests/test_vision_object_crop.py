"""Object detection on a crop around the faces in view (SPEC 5, 5A, 13): the crop geometry,
the box mapping back to frame pixels, edge-cut detections, the full-frame schedule, the switch,
the tracker's crop region and the debug counts. Mock stages only."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.config.schema import (
    ObjectCropConfig,
    RoboEmotConfig,
    ShownObjectConfig,
    TrackerConfig,
)
from roboemot.vision import Detection, Frame, Tracker
from roboemot.vision.object_crop import Crop, CropSchedule, crop_for
from roboemot.world import FrameSpec
from test_vision import CFG, H, W, rig

CROP = ObjectCropConfig()
SHOWN = ShownObjectConfig()
FW, FH = 640, 480  # a camera frame larger than the detector input


def face_box(cx: float, cy: float, h: float) -> tuple[float, float, float, float]:
    return (cx - h / 2, cy - h / 2, cx + h / 2, cy + h / 2)


# ------------------------------------------------------------------------- config


def test_defaults_units_and_ranges():
    c = ObjectCropConfig()
    assert c.enabled is True
    assert c.margin_faces == 0.5 and c.min_side_share == 0.4 and c.full_frame_share == 0.5
    assert c.full_interval_s == 2.0 and c.edge_margin_share == 0.01
    assert ObjectCropConfig.model_fields["full_interval_s"].json_schema_extra["unit"] == "s"
    assert RoboEmotConfig().vision.object_crop == c
    for bad in ({"min_side_share": 0}, {"full_frame_share": 1.5}, {"edge_margin_share": 0.5}):
        with pytest.raises(ValidationError):
            ObjectCropConfig(**bad)


def test_full_interval_must_stay_below_the_object_track_lifetime():
    with pytest.raises(ValidationError, match="full_interval_s"):
        RoboEmotConfig.model_validate(
            {"vision": {"object_lost_s": 2.0, "object_crop": {"full_interval_s": 2.0}}}
        )


# ----------------------------------------------------------------------- geometry


def test_crop_is_the_square_shown_region_of_a_far_face_with_its_margin():
    # A 60 px face at (320, 160): region 1.5 h to each side, 0.5 h above, 2.5 h below the
    # centre (3 h x 3 h), plus 0.5 h on every side: a 240 px square from (200, 100).
    crop = crop_for([face_box(320, 160, 60)], FW, FH, CROP, SHOWN)
    assert crop is not None
    assert (crop.x0, crop.y0, crop.x1, crop.y1) == (200, 100, 440, 340)


def test_crop_follows_the_shown_region_fields():
    shown = SHOWN.model_copy(update={"region_below_faces": 1.5})
    crop = crop_for([face_box(320, 160, 60)], FW, FH, CROP, shown)
    assert crop is not None
    # 3 h wide, 2 h tall plus margins: 240 x 180, made square on the wider side around the
    # region centre (y 100 .. 280, centre 190).
    assert crop.x1 - crop.x0 == crop.y1 - crop.y0 == 240
    assert (crop.y0 + crop.y1) / 2 == pytest.approx(190, abs=1)


def test_a_tiny_face_gets_the_minimum_crop_side():
    crop = crop_for([face_box(320, 200, 20)], FW, FH, CROP, SHOWN)
    assert crop is not None
    side = crop.x1 - crop.x0
    assert side == crop.y1 - crop.y0 == math.ceil(CROP.min_side_share * FH)


def test_the_crop_is_shifted_inside_the_frame_not_cut():
    crop = crop_for([face_box(30, 20, 60)], FW, FH, CROP, SHOWN)
    assert crop is not None
    assert crop.x0 == 0 and crop.y0 == 0
    assert crop.x1 - crop.x0 == crop.y1 - crop.y0 == 240
    crop = crop_for([face_box(630, 470, 60)], FW, FH, CROP, SHOWN)
    assert crop is not None
    assert crop.x1 == FW and crop.y1 == FH and crop.x1 - crop.x0 == 240


def test_a_close_face_runs_the_full_frame():
    # A 120 px face: a 480 px square, 75% of the frame area, above full_frame_share.
    assert crop_for([face_box(320, 200, 120)], FW, FH, CROP, SHOWN) is None


def test_full_frame_share_sets_the_fallback():
    loose = CROP.model_copy(update={"full_frame_share": 1.0})
    crop = crop_for([face_box(320, 200, 120)], FW, FH, loose, SHOWN)
    assert crop is not None and crop.y1 - crop.y0 == FH  # capped by the shorter side


def test_several_faces_share_one_crop_covering_all_their_regions():
    crop = crop_for([face_box(250, 150, 40), face_box(390, 150, 40)], FW, FH, CROP, SHOWN)
    assert crop is not None
    # Each region spans 4 h = 160 px around its face (with the margin); together x 170 .. 470.
    assert crop.x0 <= 170 and crop.x1 >= 470
    assert crop.x1 - crop.x0 == crop.y1 - crop.y0


def test_two_faces_far_apart_run_the_full_frame():
    assert crop_for([face_box(80, 150, 40), face_box(560, 150, 40)], FW, FH, CROP, SHOWN) is None


def test_no_face_no_crop():
    assert crop_for([], FW, FH, CROP, SHOWN) is None


# -------------------------------------------------------------------- box mapping


def test_boxes_map_back_to_frame_pixels_with_the_offset():
    crop = Crop(200, 100, 440, 340, FW, FH, edge_px=2.4)
    dets = crop.to_frame([Detection((40.0, 60.0, 100.0, 150.0), 0.7, "cup")])
    assert [d.box for d in dets] == [(240.0, 160.0, 300.0, 250.0)]
    assert dets[0].score == 0.7 and dets[0].label == "cup"


def test_view_is_the_crop_of_the_frame():
    img = np.arange(FH * FW, dtype=np.int64).reshape(FH, FW)
    crop = Crop(200, 100, 440, 340, FW, FH, edge_px=0)
    view = crop.view(img)
    assert view.shape == (240, 240) and view[0, 0] == img[100, 200]


def test_a_detection_cut_by_an_inner_crop_edge_is_dropped():
    crop = Crop(200, 100, 440, 340, FW, FH, edge_px=2.4)
    cut = [
        Detection((0.0, 50.0, 60.0, 120.0), 0.8, "chair"),  # left edge
        Detection((50.0, 1.0, 90.0, 60.0), 0.8, "chair"),  # top edge, within the tolerance
        Detection((180.0, 50.0, 240.0, 120.0), 0.8, "chair"),  # right edge
        Detection((50.0, 200.0, 90.0, 239.0), 0.8, "chair"),  # bottom edge
    ]
    whole = Detection((50.0, 50.0, 90.0, 120.0), 0.8, "cup")
    assert [d.label for d in crop.to_frame([*cut, whole])] == ["cup"]


def test_a_detection_on_the_frame_border_is_kept():
    crop = Crop(0, 0, 240, 240, FW, FH, edge_px=2.4)  # left and top edges are the frame's
    dets = crop.to_frame([Detection((0.0, 0.0, 40.0, 60.0), 0.8, "cup")])
    assert [d.box for d in dets] == [(0.0, 0.0, 40.0, 60.0)]


# ----------------------------------------------------------------------- schedule


def schedule(**update) -> CropSchedule:
    return CropSchedule(CROP.model_copy(update=update))


def choose(s: CropSchedule, stamp: float, reason: str | None = "motion", faces=None):
    faces = [face_box(320, 160, 60)] if faces is None else faces
    return s.choose(stamp, reason, faces, FW, FH, SHOWN)


def test_first_run_is_full_then_crops_until_the_full_interval():
    s = schedule()
    assert choose(s, 0.0) is None
    assert choose(s, 1.0) is not None
    assert choose(s, 1.9) is not None
    assert choose(s, 2.0) is None  # full_interval_s since the last full run
    assert choose(s, 3.0) is not None
    snap = s.snapshot()
    assert (snap["full"], snap["crop"]) == (2, 3)
    assert snap["full_by"]["interval"] == 2


@pytest.mark.parametrize("reason", ["floor", "settled"])
def test_floor_and_settled_gate_runs_are_full(reason):
    s = schedule()
    choose(s, 0.0)
    assert choose(s, 0.5, reason) is None
    assert s.snapshot()["full_by"]["gate"] == 1


@pytest.mark.parametrize("reason", ["motion", "live_track", "episode", "no_pose", None])
def test_other_gate_runs_and_the_gate_off_crop(reason):
    s = schedule()
    choose(s, 0.0)
    assert choose(s, 0.5, reason) is not None


def test_no_face_and_a_large_crop_run_full():
    s = schedule()
    choose(s, 0.0)
    assert choose(s, 0.5, faces=[]) is None
    assert choose(s, 0.6, faces=[face_box(320, 200, 120)]) is None
    by = s.snapshot()["full_by"]
    assert by["no_face"] == 1 and by["large"] == 1


def test_the_switch_turns_the_crop_off():
    s = schedule(enabled=False)
    assert all(choose(s, t) is None for t in (0.0, 0.5, 1.0))
    snap = s.snapshot()
    assert snap["enabled"] is False and snap["crop"] == 0 and snap["full_by"]["off"] == 3


def test_reset_forgets_counts_and_the_last_full_run():
    s = schedule()
    choose(s, 0.0)
    choose(s, 0.5)
    s.reset()
    assert (s.snapshot()["full"], s.snapshot()["crop"]) == (0, 0)
    assert choose(s, 0.6) is None


# ------------------------------------------------------------------ tracker region


def obj(x: float, label: str = "chair", size: float = 40, score: float = 0.8) -> Detection:
    return Detection((x, 100, x + size, 100 + size), score, label)


def object_tracker() -> Tracker:
    return Tracker(TrackerConfig(high_score=0.4, low_score=0.4, confirm_hits=2, lost_s=5.0))


def test_a_track_outside_the_crop_region_is_neither_missed_nor_dropped():
    t = object_tracker()
    t.update([obj(10)], 0.0)  # tentative chair far from the crop
    region = (200.0, -math.inf, 440.0, math.inf)
    res = t.update([obj(300, "cup")], 1.0, region)
    assert [a.label for a in res.aside] == ["chair"]
    res = t.update([obj(10), obj(300, "cup")], 2.0)  # a full run: the chair confirms
    assert {v.label for v in res.visible} == {"chair", "cup"}


def test_without_a_region_an_unmatched_tentative_track_is_dropped():
    t = object_tracker()
    t.update([obj(10)], 0.0)
    t.update([obj(300, "cup")], 1.0)
    res = t.update([obj(10), obj(300, "cup")], 2.0)
    assert {v.label for v in res.visible} == {"cup"}  # the chair started over


def test_a_track_partly_outside_the_region_is_aside():
    t = object_tracker()
    t.update([obj(190)], 0.0)
    res = t.update([], 1.0, (200.0, -math.inf, 440.0, math.inf))
    assert len(res.aside) == 1


# ------------------------------------------------------------------------ pipeline

SPEC = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",))


class CropAwareDetector:
    """Returns ``things`` (frame boxes) that lie inside the image it is given, in its pixels;
    the frame's position is found from a marker grid written into the image."""

    def __init__(self, things):
        self.things = things
        self.shapes: list[tuple[int, int]] = []

    def detect(self, img):
        self.shapes.append(img.shape[:2])
        y0, x0 = int(img[0, 0, 1]), int(img[0, 0, 2])
        h, w = img.shape[:2]
        out = []
        for d in self.things:
            x1, y1, x2, y2 = d.box
            if x1 >= x0 and y1 >= y0 and x2 <= x0 + w and y2 <= y0 + h:
                out.append(Detection((x1 - x0, y1 - y0, x2 - x0, y2 - y0), d.score, d.label))
        return out


def marked_image() -> np.ndarray:
    """A frame whose pixel (y, x) holds y in channel 1 and x (mod 256) in channel 2."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, 1] = np.arange(H, dtype=np.uint8)[:, None]
    img[:, :, 2] = (np.arange(W) % 256).astype(np.uint8)[None, :]
    return img


def face_script(i):
    # A 24 px face near the top centre of the 320 x 240 frame, every frame.
    return [Detection((148, 30, 172, 54), 0.9, landmarks=np.zeros((5, 2)))]


def crop_rig(tmp_path, things, vision=None):
    det = CropAwareDetector(things)
    _, _, pipe, _, got = rig(tmp_path, face_script, vision=vision, object_detector=det)
    return pipe, det, got


def feed(pipe, n: int, dt: float = 0.125, start: float = 0.0):
    img = marked_image()
    for i in range(n):
        pipe.process(Frame(img, start + i * dt))


def test_pipeline_crops_around_the_face_and_maps_boxes_back(tmp_path):
    cup = Detection((150, 90, 170, 110), 0.8, "cup")  # held below the face
    pipe, det, got = crop_rig(tmp_path, [cup])
    feed(pipe, 24)  # 3 s at 8 Hz: object runs at 0, 1, 2 s
    assert det.shapes[0] == (H, W)  # first run: full frame
    assert det.shapes[1][0] == det.shapes[1][1] < H  # then a square crop
    kinds = {m.payload.category for m in got["percept.entity"] if m.payload.kind != "person"}
    assert kinds == {"cup"}
    overlay = pipe.object_overlay()
    assert len(overlay) == 1
    assert overlay[0][0] == pytest.approx(cup.box, abs=1.0)
    snap = pipe.stats()["object_input"]
    assert snap["crop"] >= 1 and snap["full"] >= 1


def test_pipeline_keeps_a_thing_outside_the_crop_tracked_and_drawn(tmp_path):
    chair = Detection((5, 150, 45, 230), 0.8, "chair")  # bottom left, far from the face
    cfg = CFG.vision.model_copy(
        update={
            "object_crop": CFG.vision.object_crop.model_copy(update={"full_interval_s": 2.5}),
            "object_gate": CFG.vision.object_gate.model_copy(update={"enabled": False}),
        }
    )
    pipe, det, _ = crop_rig(tmp_path, [chair], vision=cfg)
    feed(pipe, 8 * 6)  # runs at 0..5 s: full, crop, crop, full, crop, crop
    full = [s == (H, W) for s in det.shapes]
    assert full == [True, False, False, True, False, False]
    overlay = pipe.object_overlay()
    assert [o[1] for o in overlay] == ["chair"]
    ids = {t.track_id for t in pipe.object_tracker._tracks}
    assert len(ids) == 1  # one chair track throughout


def test_pipeline_switch_off_runs_the_full_frame(tmp_path):
    cfg = CFG.vision.model_copy(
        update={"object_crop": CFG.vision.object_crop.model_copy(update={"enabled": False})}
    )
    pipe, det, _ = crop_rig(tmp_path, [], vision=cfg)
    feed(pipe, 24)
    assert det.shapes and all(s == (H, W) for s in det.shapes)
    snap = pipe.stats()["object_input"]
    assert snap["crop"] == 0 and snap["full"] == len(det.shapes)


def test_pipeline_runs_the_full_frame_on_the_gate_floor(tmp_path):
    # With a head pose the gate's first run is a floor run: full frame, counted as "gate".
    cup = Detection((150, 90, 170, 110), 0.8, "cup")
    det = CropAwareDetector([cup])
    _, _, pipe, _, _ = rig(tmp_path, face_script, object_detector=det, frame_spec=SPEC)
    img = marked_image()
    joint = {"head.yaw": 0.0, "head.pitch": 0.0}
    for i in range(16):
        pipe.process(Frame(img, i * 0.125, joint))
    snap = pipe.stats()["object_input"]
    assert det.shapes[0] == (H, W) and snap["full_by"]["gate"] == 1
    assert det.shapes[1] != (H, W)  # the tentative cup holds the gate: a live_track crop run
    assert pipe.object_gate.last_reason == "live_track"


def test_stale_face_boxes_do_not_crop(tmp_path):
    pipe, _, _ = crop_rig(tmp_path, [])
    feed(pipe, 8)
    assert pipe._visible_face_boxes(0.875)  # the face of the last face run
    lost = pipe.face_tracker.cfg.lost_s
    assert pipe._visible_face_boxes(0.875 + lost + 0.1) == []


def test_hot_reload_takes_the_crop_switch_and_the_shown_region(tmp_path):
    pipe, _, _ = crop_rig(tmp_path, [])
    off = CFG.vision.model_copy(
        update={"object_crop": CFG.vision.object_crop.model_copy(update={"enabled": False})}
    )
    shown = SHOWN.model_copy(update={"region_below_faces": 1.0})
    pipe.apply_config(off, CFG.identity, shown=shown)
    assert pipe.object_crop.cfg.enabled is False and pipe.shown is shown


def test_pipeline_without_faces_runs_the_full_frame(tmp_path):
    det = CropAwareDetector([])
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [], object_detector=det)
    feed(pipe, 24)
    assert det.shapes and all(s == (H, W) for s in det.shapes)
