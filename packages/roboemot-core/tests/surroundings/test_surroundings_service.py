"""The surroundings service: the due picture, the views of a photo sweep, the joined picture,
descriptions, failures (SPEC 8, Surroundings snapshots; designs 2026-11-24, 2026-11-25)."""

from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np
from roboemot.bus import Bus
from roboemot.circadian.model import CircadianState
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.router import ImagesUnsupported, NoRouteAvailable
from roboemot.surroundings.image import MASK_BGR
from roboemot.surroundings.service import Surroundings
from roboemot.vision import Frame
from roboemot.world import FrameSpec

SPEC = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",))
AZIMUTHS = (-90.0, -30.0, 30.0, 90.0)  # Reachy's planner at 180 degrees (design, facts)
H, W = 120, 160  # each view: small enough to be neither shrunk nor scaled in the tile


def eye(**kw) -> LLMRoute:
    return LLMRoute(name="eye", base_url="https://api.example.com/v1", model="m", images=True, **kw)


def config(images: bool = True, **surroundings) -> RoboEmotConfig:
    surroundings.setdefault("min_sharpness", 0.0)
    surroundings.setdefault("candidates", 1)
    route = eye() if images else eye().model_copy(update={"images": False})
    return RoboEmotConfig(llm_routes=[route], surroundings=surroundings)


class FakeRouter:
    def __init__(self, reply="a living room", error=None) -> None:
        self.reply, self.error, self.calls = reply, error, []
        self.offline = False
        self.usable = True

    def eligible(self):
        return [eye()] if self.usable else []

    async def describe_image(self, route, prompt, jpeg, max_tokens, first_token_s):
        self.calls.append((route.name, prompt, jpeg))
        if self.error is not None:
            raise self.error
        return self.reply


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Scan:
    """The scan controller's seams: the dwelt view and the planned view count."""

    def __init__(self) -> None:
        self.current: tuple[int, float] | None = None
        self.planned = 0

    def view(self):
        return self.current

    def views_planned(self) -> int:
        return self.planned


def board(mark: int = 0, cell: int = 8) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    gray = (((ys // cell) + (xs // cell)) % 2 * 255).astype(np.uint8)
    img = np.stack([gray] * 3, axis=-1)
    img[H - 2 :, W - 2 :] = mark  # identifies the view in the joined picture
    return img


class Rig:
    def __init__(self, cfg=None, router=None, gate=None, vision=True, first=False) -> None:
        self.cfg = cfg or config()
        self.router = router or FakeRouter()
        self.clock = Clock()
        self.gate_reason = gate
        self.events: list[tuple] = []
        self.encoded: list[np.ndarray] = []
        self.day = dt.date(2026, 11, 25)
        self.scan = Scan()

        def encode(image, quality):
            self.encoded.append(image)
            return b"\xff\xd8" + image.tobytes()[:64]  # no cv2 in CI

        self.s = Surroundings(
            lambda: self.cfg,
            lambda: self.router,
            gate=lambda: self.gate_reason,
            scan_view=self.scan.view,
            views_planned=self.scan.views_planned,
            vision=vision,
            frame_spec=SPEC,
            clock=self.clock,
            encode=encode,
            today=lambda: self.day,
        )
        self.s.events = lambda kind, reason, **f: self.events.append((kind, reason, f))
        self.bus = Bus()
        self.s.attach(self.bus)
        if not first:
            self.s._want_first = False

    def frame(self, image=None, yaw_deg: float = 0.0, faces=()):
        img = board() if image is None else image
        joint = {"head.yaw": math.radians(yaw_deg), "head.pitch": 0.0}
        self.s.tap(Frame(img, self.clock.now, joint), None if faces is None else list(faces))
        self.clock.now += 0.1

    def dwell(self, index: int, azimuth_deg: float, frames: int = 12, **kw) -> None:
        self.scan.current = (index, math.radians(azimuth_deg))
        for _ in range(frames):
            self.frame(yaw_deg=azimuth_deg, **kw)
        self.scan.current = None

    def sweep(self, azimuths=AZIMUTHS, marks=None, faces=None, done: bool = True) -> None:
        """A photo sweep over ``azimuths`` (in visit order); ``faces[i]`` the boxes of view i."""
        self.scan.planned = len(azimuths)
        for i, az in enumerate(azimuths):
            mark = marks[i] if marks else 10 * (i + 1)
            box = faces[i] if faces else ()
            self.dwell(i, az, image=board(mark), faces=box)
        if done:
            self.finish()

    def finish(self, photo: bool = True) -> None:
        self.bus.publish("world.scan_finished", {"photo": photo}, self.clock.now)
        self.s.step()

    def due(self) -> bool:
        self.s.step()  # the service thread refreshes the due state
        return self.s.picture_due()

    def picture(self) -> None:
        """A due picture, swept and sent."""
        self.bus.publish("robot.awake", None, self.clock.now)
        self.sweep()

    def later(self) -> None:
        self.clock.now += self.cfg.surroundings.room.min_interval_s

    def reasons(self):
        return [r for k, r, _ in self.events if k == "surroundings"]


def circadian(state: str, reason=None) -> CircadianState:
    return CircadianState(12.0, 0.0, 0.0, 0.0, 0.0, state, None, reason)


def view_at(joined: np.ndarray, k: int, sep: int) -> np.ndarray:
    return joined[:, k * (W + sep) : k * (W + sep) + W]


# ---- the due picture


def test_switch_off_makes_nothing_due():
    rig = Rig(cfg=config(images=False), first=True)
    assert rig.s.trigger() == "off"
    assert not rig.due() and not rig.s.wanted()
    assert rig.s.state()["state"] == "off"


def test_start_with_the_switch_on_makes_a_first_picture_due():
    rig = Rig(first=True)
    assert rig.due()


def test_the_first_picture_waits_for_a_quiet_moment_without_counting_refusals():
    rig = Rig(cfg=config(images=False), first=True)
    rig.cfg = config()
    rig.s.apply_config(rig.cfg)
    rig.gate_reason = "busy"
    assert not rig.due() and rig.reasons() == []
    rig.gate_reason = None
    rig.clock.now += rig.cfg.surroundings.room.check_interval_s
    assert rig.due()


def test_wake_and_the_end_of_a_rest_make_a_picture_due():
    rig = Rig()
    rig.bus.publish("robot.awake", None, rig.clock.now)
    assert rig.due()
    rig = Rig()
    rig.bus.publish("circadian.state", circadian("resting", "quiet_hours"), rig.clock.now)
    assert not rig.due()
    rig.bus.publish("circadian.state", circadian("awake"), rig.clock.now)
    assert rig.due()


def test_a_refused_trigger_is_counted_and_nothing_turns():
    for reason in ("asleep", "music", "busy"):
        rig = Rig(gate=reason)
        assert rig.s.trigger() == reason
        assert not rig.due() and reason in rig.reasons()
    rig = Rig()
    rig.router.offline = True
    assert rig.s.trigger() == "offline" and not rig.due()
    rig = Rig(vision=False)
    assert rig.s.trigger() == "no_vision"
    rig = Rig()
    rig.router.usable = False
    assert rig.s.trigger() == "route_unavailable"


def test_picture_due_follows_the_live_refusals():
    rig = Rig()
    rig.s.trigger()
    assert rig.due()
    rig.gate_reason = "busy"  # a conversation started: the head does not turn for it
    assert not rig.due()
    rig.gate_reason = None
    assert rig.due()


def test_no_second_picture_within_min_interval():
    rig = Rig()
    rig.picture()
    assert len(rig.router.calls) == 1 and not rig.due()
    assert rig.s.trigger() == "interval"
    rig.later()
    assert rig.s.trigger() is None and rig.due()


# ---- the photo sweep and the joined picture


def test_a_photo_sweep_sends_one_picture_with_the_views_left_to_right():
    rig = Rig(router=FakeRouter("a study: shelves left, desk centre, window right"))
    rig.s.trigger()
    rig.sweep(AZIMUTHS, marks=[11, 22, 33, 44])
    assert len(rig.router.calls) == 1
    joined = rig.encoded[0]
    sep = rig.cfg.surroundings.room.separator_px
    assert joined.shape == (H, 4 * W + 3 * sep, 3)
    # positive yaw turns left: the view at +90 degrees is leftmost
    marks = [int(view_at(joined, k, sep)[-1, -1, 0]) for k in range(4)]
    assert marks == [44, 33, 22, 11]
    assert (joined[:, W : W + sep] == 0).all()
    prompt = rig.router.calls[0][1]
    fov = rig.cfg.world.camera_fov_h_deg
    assert "4 " in prompt and f"{round(min(360.0, 180.0 + fov))} degrees" in prompt
    assert rig.s.room(rig.clock.now).startswith("a study")
    assert not rig.due() and "described" in rig.reasons()
    last = rig.s.state()["last"]
    assert last["views"] == 4 and last["outcome"] == "described"


def test_the_order_does_not_depend_on_the_visit_direction():
    rig = Rig()
    rig.s.trigger()
    rig.sweep(tuple(reversed(AZIMUTHS)), marks=[44, 33, 22, 11])
    sep = rig.cfg.surroundings.room.separator_px
    marks = [int(view_at(rig.encoded[0], k, sep)[-1, -1, 0]) for k in range(4)]
    assert marks == [44, 33, 22, 11]


def test_a_missing_view_sends_nothing_and_the_picture_stays_due():
    rig = Rig()
    rig.s.trigger()
    rig.scan.planned = 4
    for i, az in enumerate(AZIMUTHS[:3]):  # the last waypoint timed out
        rig.dwell(i, az)
    rig.finish()
    assert rig.router.calls == [] and "views_missing" in rig.reasons()
    assert rig.due()


def test_a_view_without_a_still_frame_is_missing():
    rig = Rig()
    rig.s.trigger()
    rig.scan.planned = 1
    rig.frame(yaw_deg=0.0)  # the first pose only anchors the head
    rig.scan.current = (0, 0.0)
    for i in range(1, 13):
        rig.frame(yaw_deg=i * 5.0)  # the head never settles
    rig.scan.current = None
    rig.finish()
    assert rig.router.calls == [] and "views_missing" in rig.reasons()


def test_a_view_without_a_face_run_is_missing():
    rig = Rig()
    rig.s.trigger()
    rig.sweep(AZIMUTHS[:2], faces=[(), None])  # no face detection ran on view 1
    assert rig.router.calls == [] and "views_missing" in rig.reasons()


def test_faces_of_one_view_never_mask_another_and_mask_their_own_frames():
    rig = Rig()
    rig.s.trigger()
    face = (0.0, 0.0, 40.0, 40.0)
    rig.scan.planned = 2
    rig.scan.current = (0, math.radians(30.0))
    rig.frame(yaw_deg=30.0, faces=[face])  # found on an early frame of view 0 only
    for _ in range(11):
        rig.frame(yaw_deg=30.0, faces=[])
    rig.scan.current = None
    rig.dwell(1, -30.0, faces=[])
    rig.finish()
    sep = rig.cfg.surroundings.room.separator_px
    left, right = view_at(rig.encoded[0], 0, sep), view_at(rig.encoded[0], 1, sep)
    assert (left[0:40, 0:40] == MASK_BGR).all()  # the chosen later frame is masked too
    assert not (right[0:40, 0:40] == MASK_BGR).all()  # the box never crosses to view 1


def test_a_normal_sweep_end_sends_nothing():
    rig = Rig()
    rig.s.trigger()
    rig.sweep(done=False)
    rig.finish(photo=False)
    assert rig.router.calls == [] and rig.due()


def test_frames_outside_a_view_feed_no_view():
    rig = Rig()
    rig.s.trigger()
    rig.scan.planned = 1
    for _ in range(12):
        rig.frame()  # travelling: no view named
    rig.finish()
    assert rig.router.calls == [] and "views_missing" in rig.reasons()


def test_a_conversation_at_send_time_does_not_stop_the_picture():
    rig = Rig()
    rig.s.trigger()
    rig.sweep(done=False)
    rig.gate_reason = "busy"
    rig.finish()
    assert len(rig.router.calls) == 1


def test_sleep_or_music_at_send_time_drops_the_picture_and_keeps_it_due():
    for reason in ("asleep", "music"):
        rig = Rig()
        rig.s.trigger()
        rig.sweep(done=False)
        rig.gate_reason = reason
        rig.finish()
        assert rig.router.calls == [] and reason in rig.reasons()
        rig.gate_reason = None
        assert rig.due()


def test_going_to_sleep_or_rest_drops_the_views_and_the_kept_picture():
    for topic, payload in (
        ("robot.asleep", None),
        ("circadian.state", circadian("resting", "quiet_hours")),
    ):
        rig = Rig()
        rig.picture()
        assert rig.s.jpeg() is not None
        rig.later()
        rig.s.trigger()
        rig.sweep(done=False)
        rig.bus.publish(topic, payload, rig.clock.now)
        assert rig.s.jpeg() is None
        rig.finish()
        assert len(rig.router.calls) == 1 and "views_missing" in rig.reasons()


# ---- the scene change, per view


def test_a_scene_change_at_a_stored_view_makes_a_picture_due():
    rig = Rig()
    rig.picture()
    rig.later()
    frames = int(rig.cfg.surroundings.room.check_interval_s / 0.1) + 8
    for _ in range(frames):
        rig.frame(image=board(30), yaw_deg=30.0)  # the same view as sent
    assert not rig.due()
    changed = 255 - board(30)  # every pixel flips
    for _ in range(frames):
        rig.frame(image=changed, yaw_deg=60.0)  # no stored view within view_match_deg
    assert not rig.due()
    for _ in range(frames):
        rig.frame(image=changed, yaw_deg=-89.0)  # within view_match_deg of -90
    assert rig.due()


def test_the_view_match_wraps_around_the_back():
    rig = Rig()
    rig.s.trigger()
    rig.sweep((180.0,))
    rig.later()
    frames = int(rig.cfg.surroundings.room.check_interval_s / 0.1) + 8
    for _ in range(frames):
        rig.frame(image=255 - board(10), yaw_deg=-178.0)
    assert rig.due()


# ---- failures, caps, privacy


def test_room_text_expires_after_max_age():
    rig = Rig(router=FakeRouter("a kitchen"))
    rig.picture()
    assert rig.s.room(rig.clock.now) == "a kitchen"
    later = rig.clock.now + rig.cfg.surroundings.room.max_age_s + 1
    assert rig.s.room(later) is None


def test_unsupported_stops_pictures_until_the_route_changes():
    rig = Rig(router=FakeRouter(error=ImagesUnsupported("http_400")))
    rig.picture()
    assert rig.s.state()["state"] == "unsupported"
    assert rig.s.state()["unsupported"] == "http_400"
    rig.later()
    assert rig.s.trigger() == "unsupported"
    rig.s.apply_config(rig.cfg)  # an unrelated reload keeps the mark
    assert rig.s.trigger() == "unsupported"
    rig.cfg = RoboEmotConfig(
        llm_routes=[eye().model_copy(update={"model": "m2"})],
        surroundings=rig.cfg.surroundings,
    )
    rig.s.apply_config(rig.cfg)
    assert rig.s.trigger() is None


def test_max_failures_pauses_for_pause_s():
    rig = Rig(cfg=config(max_failures=2, room={"min_interval_s": 60}),
              router=FakeRouter(error=RuntimeError("boom")))  # fmt: skip
    for _ in range(2):
        rig.picture()
        rig.later()
    assert rig.s.trigger() == "paused"
    rig.clock.now += rig.cfg.surroundings.pause_s
    assert rig.s.trigger() is None


def test_a_route_error_at_send_time_is_counted_as_unavailable():
    rig = Rig(router=FakeRouter(error=NoRouteAvailable("eye")))
    rig.picture()
    assert "route_unavailable" in rig.reasons()


def test_switch_off_mid_request_drops_reply_and_jpeg():
    rig = Rig()
    rig.s.trigger()
    rig.sweep(done=False)

    class TurnOff(FakeRouter):
        async def describe_image(inner, route, prompt, jpeg, max_tokens, first_token_s):
            rig.cfg = config(images=False)
            rig.s.apply_config(rig.cfg)
            return "a living room"

    rig.router = TurnOff()
    rig.finish()
    assert rig.s.room(rig.clock.now) is None
    assert rig.s.jpeg() is None
    assert not rig.due()


def test_state_and_logs_hold_no_text_and_no_image_bytes(caplog):
    rig = Rig(router=FakeRouter("Marcello's study"))
    with caplog.at_level(logging.DEBUG):
        rig.picture()
    state = repr(rig.s.state())
    assert "study" not in state and "Marcello" not in state
    assert "Marcello" not in caplog.text
    assert all("Marcello" not in repr(f) for _, _, f in rig.events)
    assert rig.s.live()["described"]["room"] == "Marcello's study"


def test_the_state_shows_offline_and_route_unavailable():
    rig = Rig()
    rig.router.offline = True
    assert rig.s.state()["state"] == "offline"
    rig.router.offline = False
    rig.router.usable = False
    assert rig.s.state()["state"] == "route_unavailable"
    rig.router.usable = True
    assert rig.s.state()["state"] == "on"


def test_the_daily_cap_stops_pictures_until_the_next_day():
    rig = Rig(cfg=config(max_per_day=2))
    rig.picture()
    rig.later()
    rig.picture()
    assert len(rig.router.calls) == 2
    rig.later()
    assert rig.s.trigger() == "daily_limit"
    assert rig.s.state()["state"] == "daily_limit"
    rig.day = rig.day + dt.timedelta(days=1)
    assert rig.s.trigger() is None
    assert rig.s.state()["state"] == "on"


def test_zero_means_no_cap():
    rig = Rig(cfg=config(max_per_day=0))
    for _ in range(3):
        rig.picture()
        rig.later()
    assert len(rig.router.calls) == 3


def test_a_429_stops_pictures_until_the_next_day():
    from roboemot.llm.router import RateLimited

    rig = Rig(router=FakeRouter(error=RateLimited("http_429")))
    rig.picture()
    assert "daily_limit" in rig.reasons()
    rig.later()
    assert rig.s.trigger() == "daily_limit"
    rig.day = rig.day + dt.timedelta(days=1)
    assert rig.s.trigger() is None


def test_the_last_picture_carries_its_outcome_and_a_failure_its_reason():
    rig = Rig(router=FakeRouter(error=TimeoutError()))
    rig.picture()
    last = rig.s.state()["last"]
    assert (last["outcome"], last["detail"]) == ("failed", "timeout")


def test_an_empty_reply_is_failed_empty():
    rig = Rig(router=FakeRouter("   "))
    rig.picture()
    last = rig.s.state()["last"]
    assert (last["outcome"], last["detail"]) == ("failed", "empty")


def test_the_history_lists_the_last_pictures_newest_first_with_their_texts():
    rig = Rig(cfg=config(history_size=2), router=FakeRouter("a kitchen"))
    rig.picture()
    rig.later()
    rig.router.reply = "a study"
    rig.picture()
    rig.later()
    rig.router.error = TimeoutError()
    rig.picture()
    history = rig.s.live()["history"]
    assert [(h["outcome"], h["text"]) for h in history] == [
        ("failed", None),
        ("described", "a study"),
    ]
    assert all(h["views"] == 4 and h["age_s"] >= 0 for h in history)


def test_the_history_is_dropped_when_the_switch_goes_off():
    rig = Rig(router=FakeRouter("Marcello's desk"))
    rig.picture()
    assert "Marcello" not in repr(rig.s.state())
    assert rig.s.live()["history"][0]["text"] == "Marcello's desk"
    rig.cfg = config(images=False)
    rig.s.apply_config(rig.cfg)
    assert rig.s.live()["history"] == []


def test_an_http_error_is_failed_with_its_status():
    import httpx
    from openai import APIStatusError

    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    error = APIStatusError("busy", response=httpx.Response(503, request=request), body=None)
    rig = Rig(router=FakeRouter(error=error))
    rig.picture()
    assert rig.s.state()["last"]["detail"] == "http_503"
