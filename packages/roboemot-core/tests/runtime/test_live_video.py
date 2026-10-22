"""SPEC 4A, 11A, 12: the Live tab's video preview. One JPEG from the frame the vision pump already
captured; encoded only on request from a newer frame; one encode shared by every viewer; viewers
tracked by a window; the frame and JPEG dropped when viewing stops, the flag goes off or the robot
sleeps; nothing in logs or reports."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.runtime import live_video
from roboemot.runtime.live_video import LiveVideo, ascii_label
from roboemot.vision.runner import LatestFrameRunner
from roboemot.vision.types import Frame
from roboemot.world.types import Entry

SENTINEL_BGR = (13, 77, 201)
WINDOW_S = 6.0


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Module:
    name = "dummy"

    def process(self, frame):
        pass


def config(**live):
    base = {"enabled": True, "video": True, "video_fps": 2.0, "viewer_window_s": WINDOW_S}
    return RoboEmotConfig(webui={"live": {**base, **live}})


def image(width=640, height=480):
    return np.full((height, width, 3), SENTINEL_BGR, np.uint8)


class World:
    def __init__(self, identity):
        self.entry = Entry(
            entry_id="e1", kind="person", category="face", identity=identity, azimuth_rad=0.0,
            elevation_rad=0.0, distance_m=1.0, uncertainty_rad=0.1, confidence=0.9,
            first_seen=0.0, last_seen=1.0,
        )  # fmt: skip

    def get(self, entry_id):
        return self.entry


class Rig:
    """A LiveVideo over a real runner, a fake pipeline overlay and a settable clock."""

    def __init__(self, faces=(), names=None, target_identity=None, **live):
        self.clock = Clock()
        self.runner = LatestFrameRunner(Module(), clock=self.clock)
        self.service = SimpleNamespace(runner=self.runner, paused=False)
        people = {
            k: SimpleNamespace(display_name=v, status="active") for k, v in (names or {}).items()
        }
        self.faces = list(faces)
        registry = SimpleNamespace(get_person=people.__getitem__)
        pipeline = SimpleNamespace(
            face_overlay=lambda: self.faces, matcher=SimpleNamespace(registry=registry)
        )
        self.host = SimpleNamespace(
            perception=SimpleNamespace(service=self.service, pipeline=pipeline)
        )
        target = None if target_identity is None else SimpleNamespace(entry_id="e1")
        self.p = SimpleNamespace(
            sleep=SimpleNamespace(asleep=False),
            selector=SimpleNamespace(target=target),
            world=World(target_identity),
        )
        self.video = LiveVideo(self.host, self.p, config(**live), clock=self.clock)
        self.encodes = 0
        real = self.video._encode

        def counting(*args):
            self.encodes += 1
            return real(*args)

        self.video._encode = counting

    def submit(self, img=None):
        self.runner.submit(Frame(image() if img is None else img, self.clock()))

    def apply(self, **live):
        self.video.apply_config(SimpleNamespace(config=config(**live)))


@pytest.fixture
def rig():
    return Rig()


def test_nothing_is_kept_or_encoded_with_no_viewer(rig):
    rig.submit()
    rig.submit()
    assert rig.runner.latest() is None
    assert rig.encodes == 0


def test_the_first_request_starts_watching_and_the_next_frame_is_served(rig):
    assert rig.video.frame("10.0.0.5") == (None, "video_no_frame")
    assert rig.video.viewers()["active"] is False  # nothing was sent yet
    rig.submit()
    jpeg, code = rig.video.frame("10.0.0.5")
    assert code is None and jpeg[:2] == b"\xff\xd8"
    assert rig.video.viewers() == {"active": True, "count": 1, "poll_ms": 3000}
    assert rig.encodes == 1


def test_two_viewers_share_one_encode_and_a_newer_frame_waits_for_the_rate(rig):
    rig.video.frame("a")
    rig.submit()
    first, _ = rig.video.frame("a")
    second, _ = rig.video.frame("b")
    assert first is second and rig.encodes == 1
    rig.clock.now += 0.1
    rig.submit()  # newer, but 1 / video_fps = 0.5 s has not passed
    assert rig.video.frame("a")[0] is first and rig.encodes == 1
    rig.clock.now += 0.5
    third, _ = rig.video.frame("b")
    assert third is not first and rig.encodes == 2
    assert rig.video.viewers()["count"] == 2


def test_the_same_frame_is_not_encoded_twice(rig):
    rig.video.frame("a")
    rig.submit()
    rig.video.frame("a")
    rig.clock.now += 2.0
    rig.video.frame("a")
    assert rig.encodes == 1


def test_codes_for_every_state_without_a_picture():
    assert Rig(enabled=False).video.frame("a") == (None, "live_disabled")
    assert Rig(video=False).video.frame("a") == (None, "video_off")
    no_service = Rig()
    no_service.host.perception.service = None
    assert no_service.video.frame("a") == (None, "video_no_vision")
    no_cv2 = Rig()
    no_cv2.video._load_cv2 = lambda: None
    assert no_cv2.video.frame("a") == (None, "video_no_vision")
    for rig in (Rig(), Rig()):
        rig.submit()
    asleep = Rig()
    asleep.p.sleep.asleep = True
    assert asleep.video.frame("a") == (None, "video_asleep")
    paused = Rig()
    paused.service.paused = True
    assert paused.video.frame("a") == (None, "video_asleep")


def test_asleep_never_resumes_or_touches_the_pump():
    rig = Rig()
    rig.service.resume = lambda: pytest.fail("the preview resumed the pump")
    rig.service.pause = lambda: pytest.fail("the preview paused the pump")
    rig.service.source = None  # a read would raise
    rig.service.paused = True
    for _ in range(3):
        assert rig.video.frame("a") == (None, "video_asleep")
    rig.service.paused = False
    rig.p.sleep.asleep = True
    assert rig.video.frame("a") == (None, "video_asleep")


def test_the_viewer_window_starts_and_stops_and_the_banner_flag_follows(rig, caplog):
    caplog.set_level(logging.INFO, logger=live_video.__name__)
    rig.video.frame("10.0.0.5")
    rig.submit()
    rig.video.frame("10.0.0.5")
    rig.clock.now += WINDOW_S - 0.5
    assert rig.video.viewers()["active"] is True
    rig.video.frame("10.0.0.6")  # a second client; the first lapses first
    rig.clock.now += 1.0
    assert rig.video.viewers() == {"active": True, "count": 1, "poll_ms": 3000}
    rig.clock.now += WINDOW_S
    assert rig.video.viewers() == {"active": False, "count": 0, "poll_ms": 3000}
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [
        "live video view started (10.0.0.5)",
        "live video view started (10.0.0.6)",
        "live video view stopped (10.0.0.5)",
        "live video view stopped (10.0.0.6)",
    ]
    assert rig.video.viewers()["active"] is False and len(caplog.records) == 4  # once each


def test_the_frame_and_jpeg_are_dropped_when_viewing_stops(rig):
    rig.video.frame("a")
    rig.submit()
    rig.video.frame("a")
    assert rig.runner.latest() is not None and rig.video._jpeg is not None
    rig.clock.now += WINDOW_S + 1
    rig.video.viewers()
    assert rig.runner.latest() is None and rig.video._jpeg is None


def test_the_frame_and_jpeg_are_dropped_when_the_flag_goes_off(rig):
    rig.video.frame("a")
    rig.submit()
    rig.video.frame("a")
    rig.apply(video=False)
    assert rig.runner.latest() is None and rig.video._jpeg is None
    assert rig.video.frame("a") == (None, "video_off")
    assert rig.video.viewers()["active"] is False
    rig.apply(video=True, enabled=True)
    rig.apply(enabled=False)
    assert rig.video.frame("a") == (None, "live_disabled")


def test_the_frame_and_jpeg_are_dropped_when_the_robot_sleeps(rig):
    rig.video.frame("a")
    rig.submit()
    rig.video.frame("a")
    rig.p.sleep.asleep = True
    assert rig.video.frame("a") == (None, "video_asleep")
    assert rig.runner.latest() is None and rig.video._jpeg is None
    rig.p.sleep.asleep = False
    assert rig.video.viewers()["active"] is False
    rig.submit()
    assert rig.runner.latest() is None  # nobody watches until the next request


def test_drop_is_the_bus_and_shutdown_hook(rig):
    rig.video.frame("a")
    rig.submit()
    rig.video.frame("a")
    rig.video.drop()
    assert rig.video._jpeg is None and rig.runner.latest() is None
    assert rig.video.viewers()["active"] is False


def decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_the_picture_is_scaled_to_the_max_width_with_the_quality_setting():
    small = Rig(video_max_width=160)
    small.video.frame("a")
    small.submit()
    jpeg, _ = small.video.frame("a")
    assert decode(jpeg).shape == (120, 160, 3)
    narrow = Rig(video_max_width=1280)
    narrow.video.frame("a")
    narrow.submit(image(200, 100))
    assert decode(narrow.video.frame("a")[0]).shape == (100, 200, 3)  # never scaled up
    low, high = Rig(video_jpeg_quality=20), Rig(video_jpeg_quality=95)
    noisy = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    sizes = []
    for r in (low, high):
        r.video.frame("a")
        r.submit(noisy)
        sizes.append(len(r.video.frame("a")[0]))
    assert sizes[0] < sizes[1]


def drawn(rig):
    rig.apply(video_max_width=640)  # scale 1: the box coordinates are pixels of the picture
    rig.video.frame("a")
    rig.submit(np.zeros((480, 640, 3), np.uint8))
    return decode(rig.video.frame("a")[0])


def green_pixels(img):
    return int(((img[:, :, 1] > 150) & (img[:, :, 0] < 120) & (img[:, :, 2] < 120)).sum())


def yellow_pixels(img):
    return int(((img[:, :, 2] > 180) & (img[:, :, 1] > 150) & (img[:, :, 0] < 90)).sum())


def test_face_boxes_are_drawn_and_the_gaze_target_is_highlighted():
    boxes = [((20.0, 40.0, 140.0, 200.0), "mia"), ((300.0, 60.0, 420.0, 220.0), "leo")]
    plain = drawn(Rig(faces=boxes, names={"mia": "Mia", "leo": "Leo"}))
    assert green_pixels(plain) > 0 and yellow_pixels(plain) == 0
    gaze = drawn(Rig(faces=boxes, names={"mia": "Mia", "leo": "Leo"}, target_identity="mia"))
    assert yellow_pixels(gaze) > 0 and green_pixels(gaze) > 0
    assert gaze[120, 20:24].max() > 0  # the left edge of the first box (scale 1)
    empty = drawn(Rig())
    assert int(empty.max()) <= 10  # JPEG noise only


def test_a_name_label_only_for_a_confirmed_person():
    faces = [((100.0, 100.0, 200.0, 220.0), "mia")]
    named = drawn(Rig(faces=faces, names={"mia": "Mia"}))
    stranger = drawn(Rig(faces=[((100.0, 100.0, 200.0, 220.0), None)]))
    above = lambda img: int(img[60:92, 104:170].sum())  # noqa: E731
    assert above(named) > 0 and above(stranger) == 0
    pending = Rig(faces=faces)
    pending.host.perception.pipeline.matcher.registry = SimpleNamespace(
        get_person=lambda _id: SimpleNamespace(display_name="Secret", status="pending")
    )
    assert above(drawn(pending)) == 0


def test_a_lone_unknown_face_is_the_target_when_the_target_has_no_identity():
    rig = Rig(faces=[((100.0, 100.0, 200.0, 220.0), None)], target_identity="unknown")
    assert yellow_pixels(drawn(rig)) > 0
    two = Rig(
        faces=[((10.0, 10.0, 90.0, 90.0), None), ((300.0, 10.0, 380.0, 90.0), None)],
        target_identity="unknown",
    )
    assert yellow_pixels(drawn(two)) == 0  # ambiguous: no guess


def test_label_text_is_folded_to_what_the_font_can_draw():
    assert ascii_label("Zoë") == "Zoe"
    assert ascii_label("李") == "?"
    assert len(ascii_label("x" * 100)) == live_video.LABEL_MAX_CHARS


def test_logs_and_the_vision_report_hold_no_image_bytes(caplog):
    caplog.set_level(logging.DEBUG)
    rig = Rig()
    rig.video.frame("10.0.0.9")
    rig.submit()
    jpeg, _ = rig.video.frame("10.0.0.9")
    rig.video.drop()
    text = "\n".join(r.getMessage() + repr(r.args) for r in caplog.records)
    assert "10.0.0.9" in text
    assert jpeg[:16].decode("latin-1") not in text and repr(jpeg[:16]) not in text
    assert "\\xff\\xd8" not in text and str(SENTINEL_BGR) not in text

    runner = rig.runner
    runner.watch(WINDOW_S)
    runner.submit(Frame(image(), 1.0))
    assert runner.latest() is not None
    report = json.dumps(
        {"frames": {"processed": runner.processed, "dropped": runner.dropped, **runner.rates()}}
    )
    assert "image" not in report and "ffd8" not in report.lower()
