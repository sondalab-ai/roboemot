"""The surroundings service: the due picture, the views of a photo sweep, the joined picture,
descriptions, failures (SPEC 8, Surroundings snapshots; designs 2026-11-24, 2026-11-25)."""

from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np
import pytest
from roboemot.bus import Bus
from roboemot.circadian.model import CircadianState
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.router import ImagesUnsupported, NoRouteAvailable, RateLimited
from roboemot.surroundings.image import MASK_BGR
from roboemot.surroundings.service import Surroundings
from roboemot.vision import Frame
from roboemot.world import FrameSpec
from roboemot.world.scan import ScanView

SPEC = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",))
AZIMUTHS = (-90.0, -30.0, 30.0, 90.0)  # Reachy's planner at 180 degrees (design, facts)
H, W = 120, 160  # each view: small enough to be neither shrunk nor scaled in the tile


def eye(name: str = "eye", **kw) -> LLMRoute:
    kw.setdefault("images", True)
    kw.setdefault("model", "m")
    return LLMRoute(name=name, base_url="https://api.example.com/v1", **kw)


def config(images: bool = True, routes=None, **surroundings) -> RoboEmotConfig:
    surroundings.setdefault("min_sharpness", 0.0)
    surroundings.setdefault("candidates", 1)
    if routes is None:
        routes = [eye(images=images)]
    return RoboEmotConfig(llm_routes=routes, surroundings=surroundings)


def two(**surroundings) -> RoboEmotConfig:
    """Two image routes: ``free`` first, ``paid`` second."""
    return config(routes=[eye("free"), eye("paid")], **surroundings)


class FakeRouter:
    """``error`` fails every route; ``errors`` maps a route name to its own error. ``usable``
    false (or a name in ``unusable``) leaves a route out of ``eligible``."""

    def __init__(self, reply="a living room", error=None, errors=None) -> None:
        self.reply, self.error, self.calls = reply, error, []
        self.errors: dict[str, BaseException] = dict(errors or {})
        self.offline = False
        self.usable = True
        self.unusable: set[str] = set()
        self.cfg: RoboEmotConfig | None = None  # the rig's config, for eligible

    def eligible(self):
        routes = self.cfg.llm_routes if self.cfg is not None else [eye()]
        return [r for r in routes if self.usable and r.name not in self.unusable]

    async def describe_image(self, route, prompt, jpeg, max_tokens, first_token_s):
        self.calls.append((route.name, prompt, jpeg))
        error = self.errors.get(route.name, self.error)
        if error is not None:
            raise error
        return self.reply


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Scan:
    """The scan controller's seams: the dwelt view and the planned view count."""

    def __init__(self) -> None:
        self.current: ScanView | None = None
        self.planned = 0
        self.sweep = 1  # the photo sweep's id

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
            self._router,
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

    def _router(self) -> FakeRouter:
        self.router.cfg = self.cfg
        return self.router

    def apply(self, cfg: RoboEmotConfig) -> None:
        self.cfg = cfg
        self.s.apply_config(cfg)

    def routes(self) -> list[str]:
        """The route of each describe request, in order."""
        return [name for name, _, _ in self.router.calls]

    def frame(self, image=None, yaw_deg: float = 0.0, faces=()):
        img = board() if image is None else image
        joint = {"head.yaw": math.radians(yaw_deg), "head.pitch": 0.0}
        self.s.tap(Frame(img, self.clock.now, joint), None if faces is None else list(faces))
        self.clock.now += 0.1

    def dwell(self, index: int, azimuth_deg: float, frames: int = 12, **kw) -> None:
        self.at(index, math.radians(azimuth_deg))
        for _ in range(frames):
            self.frame(yaw_deg=azimuth_deg, **kw)
        self.scan.current = None

    def at(self, index: int, azimuth_rad: float) -> None:
        """The head dwells at view ``index`` of the current photo sweep."""
        self.scan.current = ScanView(self.scan.sweep, index, azimuth_rad)

    def sweep(self, azimuths=AZIMUTHS, marks=None, faces=None, done: bool = True) -> None:
        """A new photo sweep over ``azimuths`` (in visit order); ``faces[i]`` view i's boxes."""
        self.scan.sweep += 1
        self.scan.planned = len(azimuths)
        for i, az in enumerate(azimuths):
            mark = marks[i] if marks else 10 * (i + 1)
            box = faces[i] if faces else ()
            self.dwell(i, az, image=board(mark), faces=box)
        if done:
            self.finish()

    def finish(self, photo: bool = True) -> None:
        payload = {"photo": photo, "sweep": self.scan.sweep}
        self.bus.publish("world.scan_finished", payload, self.clock.now)
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
    rig.at(0, 0.0)
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


def test_faces_of_one_view_never_mask_another():
    rig = Rig()
    rig.s.trigger()
    face = (0.0, 0.0, 40.0, 40.0)
    rig.sweep((30.0, -30.0), faces=[[face], []])
    sep = rig.cfg.surroundings.room.separator_px
    left, right = view_at(rig.encoded[0], 0, sep), view_at(rig.encoded[0], 1, sep)
    assert (left[0:40, 0:40] == MASK_BGR).all()
    assert not (right[0:40, 0:40] == MASK_BGR).all()  # the box never crosses to view 1


# ---- security review 2026-11-25: a frame is chosen only with its own face run


def test_a_still_frame_without_a_face_run_is_never_chosen():
    rig = Rig()
    rig.s.trigger()
    rig.scan.planned = 1
    rig.frame(yaw_deg=0.0)  # the first pose only anchors the head
    rig.at(0, 0.0)
    for _ in range(8):
        rig.frame(image=board(11), faces=None)  # still, sharp, but no face run of its own
    rig.frame(image=board(22), faces=[])  # still, with its own face run
    rig.scan.current = None
    rig.finish()
    assert len(rig.encoded) == 1
    assert int(view_at(rig.encoded[0], 0, 0)[-1, -1, 0]) == 22


def test_a_chosen_frame_is_masked_with_its_own_boxes_only():
    rig = Rig()
    rig.s.trigger()
    rig.scan.planned = 1
    rig.frame(yaw_deg=0.0)
    rig.at(0, 0.0)
    own, other = (0.0, 0.0, 40.0, 40.0), (100.0, 60.0, 140.0, 100.0)
    rig.frame(image=board(22), faces=[own])  # chosen (candidates=1): masked with its own box
    for _ in range(5):
        rig.frame(image=board(33), faces=[other])  # later frames: their boxes stay theirs
    rig.scan.current = None
    rig.finish()
    view = view_at(rig.encoded[0], 0, 0)
    assert int(view[-1, -1, 0]) == 22
    assert (view[0:40, 0:40] == MASK_BGR).all()
    assert not (view[60:100, 100:140] == MASK_BGR).all()


def test_views_of_a_sweep_cancelled_without_a_sleep_event_never_join_the_next():
    rig = Rig()
    rig.s.trigger()
    rig.sweep(AZIMUTHS[:2], done=False)  # views 0 and 1, then cancelled (no event at all)
    rig.scan.sweep += 1  # the next photo sweep
    rig.scan.planned = 4
    for i in (2, 3):
        rig.dwell(i, AZIMUTHS[i])
    rig.finish()
    assert rig.router.calls == [] and "views_missing" in rig.reasons()


def test_the_end_of_another_sweep_takes_no_views():
    rig = Rig()
    rig.s.trigger()
    rig.sweep(done=False)
    rig.bus.publish("world.scan_finished", {"photo": True, "sweep": rig.scan.sweep - 1}, 0.0)
    rig.s.step()
    assert rig.router.calls == []


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


def test_a_400_skips_the_only_route_for_the_day_not_forever():
    rig = Rig(router=FakeRouter(error=ImagesUnsupported("http_400")))
    rig.picture()
    assert "unsupported" in rig.reasons()
    state = rig.s.state()
    assert state["state"] == "daily_limit"  # every image route is skipped today
    assert state["routes"] == [
        {"route": "eye", "skipped": True, "reason": "http_400", "paused_s": 0.0}
    ]
    rig.later()
    assert not rig.due()  # owed, but not sent again today
    rig.s.apply_config(rig.cfg)  # an unrelated reload keeps the mark
    assert not rig.due()
    rig.router.error = None
    rig.day = rig.day + dt.timedelta(days=1)
    assert rig.due()  # a top-up works the next day
    rig.sweep()
    assert rig.s.room(rig.clock.now) == "a living room"


def test_a_change_of_the_routes_model_clears_its_skip_at_once():
    rig = Rig(router=FakeRouter(error=ImagesUnsupported("http_415")))
    rig.picture()
    rig.later()
    assert not rig.due()
    rig.apply(config(routes=[eye(model="m2")]))
    assert rig.due()


def test_max_failures_pauses_for_pause_s():
    rig = Rig(cfg=config(max_failures=2, room={"min_interval_s": 60}),
              router=FakeRouter(error=RuntimeError("boom")))  # fmt: skip
    for _ in range(2):
        rig.picture()
        rig.later()
    assert not rig.due() and rig.s.state()["state"] == "paused"
    rig.clock.now += rig.cfg.surroundings.pause_s
    assert rig.due()  # the failed picture is still owed


def test_a_route_error_at_send_time_is_counted_as_unavailable_and_never_skips_it():
    rig = Rig(router=FakeRouter(error=NoRouteAvailable("eye")))
    rig.picture()
    assert "route_unavailable" in rig.reasons()
    assert rig.s.state()["routes"][0] == {
        "route": "eye",
        "skipped": False,
        "reason": None,
        "paused_s": 0.0,
    }  # passed over for that picture only (owner decision 2026-11-25)
    rig.router.error = None
    rig.later()
    assert rig.due()  # the same day: usable again, so the picture goes out


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
    assert not rig.due()  # still owed, but the head does not turn for it today
    assert rig.s.state()["state"] == "daily_limit"
    rig.day = rig.day + dt.timedelta(days=1)
    assert rig.due()  # due again the next day, with no new trigger


# ---- a failed describe leaves the picture due (review 2026-11-25)


def test_a_failed_describe_leaves_the_picture_due_after_the_interval():
    for error in (TimeoutError(), RuntimeError("boom")):
        rig = Rig(router=FakeRouter(error=error))
        rig.picture()
        assert len(rig.router.calls) == 1
        assert not rig.due()  # no back-to-back sweep: the interval gates it
        rig.clock.now += rig.cfg.surroundings.room.min_interval_s / 2
        assert not rig.due()
        rig.later()
        assert rig.due(), type(error).__name__


def test_an_empty_reply_leaves_the_picture_due():
    rig = Rig(router=FakeRouter("   "))
    rig.picture()
    rig.later()
    assert rig.due()


def test_a_described_picture_is_no_longer_due():
    rig = Rig()
    rig.picture()
    rig.later()
    assert not rig.due()


def test_the_first_picture_survives_a_failure():
    rig = Rig(router=FakeRouter(error=TimeoutError()), first=True)
    rig.s.step()
    rig.sweep()
    assert len(rig.router.calls) == 1
    rig.later()
    rig.router.error = None
    assert rig.due()
    rig.sweep()
    assert rig.s.room(rig.clock.now) == "a living room"


def test_sleep_during_a_describe_leaves_the_picture_due():
    rig = Rig()
    rig.s.trigger()
    rig.sweep(done=False)

    class Sleepy(FakeRouter):
        async def describe_image(inner, route, prompt, jpeg, max_tokens, first_token_s):
            rig.bus.publish("robot.asleep", None, rig.clock.now)
            return "a living room"

    rig.router = Sleepy()
    rig.finish()
    assert rig.s.room(rig.clock.now) is None  # the reply is ignored
    rig.later()
    assert rig.due()


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


# ---- several image routes: priority and fallback (owner decision 2026-11-25)


def test_a_429_sends_the_same_picture_to_the_next_route_and_skips_the_first_today():
    rig = Rig(cfg=two(), router=FakeRouter(errors={"free": RateLimited("http_429")}))
    rig.picture()
    assert rig.routes() == ["free", "paid"]
    assert rig.router.calls[0][2] == rig.router.calls[1][2]  # the same JPEG, no new sweep
    assert rig.s.room(rig.clock.now) == "a living room"
    assert rig.reasons() == ["described"]
    state = rig.s.state()
    assert state["state"] == "on" and state["route"] == "paid"
    assert [(r["route"], r["skipped"], r["reason"]) for r in state["routes"]] == [
        ("free", True, "http_429"),
        ("paid", False, None),
    ]
    rig.later()
    rig.s.trigger()
    rig.sweep()
    assert rig.routes()[2:] == ["paid"]  # the first stays skipped for the rest of the day
    rig.later()
    rig.day = rig.day + dt.timedelta(days=1)
    rig.router.errors = {}
    rig.s.trigger()
    rig.sweep()
    assert rig.routes()[3:] == ["free"]  # used again the next day


def test_a_400_falls_back_and_the_route_is_used_again_the_next_day():
    rig = Rig(cfg=two(), router=FakeRouter(errors={"free": ImagesUnsupported("http_400")}))
    rig.picture()
    assert rig.routes() == ["free", "paid"]
    assert rig.s.state()["routes"][0]["reason"] == "http_400"
    rig.later()
    rig.day = rig.day + dt.timedelta(days=1)
    rig.router.errors = {}
    rig.s.trigger()
    rig.sweep()
    assert rig.routes()[2:] == ["free"]  # not unsupported forever: a top-up works


def test_a_timeout_stays_on_the_route_and_counts_a_failure_there():
    rig = Rig(cfg=two(max_failures=2), router=FakeRouter(errors={"free": TimeoutError()}))
    rig.picture()
    assert rig.routes() == ["free"]  # no fallback on a timeout
    assert rig.s.state()["last"]["detail"] == "timeout"
    assert rig.s.room(rig.clock.now) is None
    rig.later()
    assert rig.due()
    rig.sweep()
    assert rig.routes() == ["free", "free"]
    routes = rig.s.state()["routes"]
    assert routes[0]["paused_s"] > 0 and routes[1]["paused_s"] == 0  # paused per route
    rig.later()
    assert rig.due() and rig.s.state()["state"] == "on"
    rig.sweep()
    assert rig.routes()[2:] == ["paid"]  # the paused route's next picture goes to the next


def test_every_route_refusing_leaves_the_picture_due_with_the_last_outcome():
    errors = {"free": RateLimited("http_429"), "paid": ImagesUnsupported("http_400")}
    rig = Rig(cfg=two(), router=FakeRouter(errors=errors))
    rig.picture()
    assert rig.routes() == ["free", "paid"]
    assert rig.reasons() == ["unsupported"]  # one outcome per picture: the last route's
    last = rig.s.state()["last"]
    assert (last["outcome"], last["detail"], last["route"]) == ("unsupported", "http_400", "paid")
    assert rig.s.state()["state"] == "daily_limit"
    rig.later()
    assert not rig.due()  # still owed; the head does not turn for it today
    rig.router.errors = {}
    rig.day = rig.day + dt.timedelta(days=1)
    assert rig.due()


def test_the_daily_cap_counts_the_requests_to_every_route():
    rig = Rig(cfg=two(max_per_day=2), router=FakeRouter(errors={"free": RateLimited("http_429")}))
    rig.picture()
    assert rig.routes() == ["free", "paid"]
    rig.later()
    assert rig.s.trigger() == "daily_limit"


def test_the_cap_stops_the_fallback_mid_picture():
    errors = {"free": RateLimited("http_429")}
    rig = Rig(cfg=two(max_per_day=1), router=FakeRouter(errors=errors))
    rig.picture()
    assert rig.routes() == ["free"]
    assert rig.s.state()["last"]["outcome"] == "daily_limit"


def test_an_unusable_route_at_send_time_sends_nothing_and_falls_back():
    errors = {"free": NoRouteAvailable("free")}
    rig = Rig(cfg=two(max_per_day=2), router=FakeRouter(errors=errors))
    rig.picture()
    assert rig.routes() == ["free", "paid"]  # free raised before sending: not counted
    assert rig.s.room(rig.clock.now) == "a living room"
    assert rig.s._sent == 1  # the fallback counts once
    assert not rig.s.state()["routes"][0]["skipped"]  # this picture only
    rig.router.errors = {}
    rig.later()
    rig.s.trigger()
    rig.sweep()
    assert rig.routes()[2:] == ["free"]  # the next picture tries it first again
    rig.later()
    assert rig.s.trigger() == "daily_limit"  # two requests left the robot: the cap


def test_a_paused_route_is_passed_over_and_the_next_picture_goes_to_the_next():
    rig = Rig(cfg=two(max_failures=1), router=FakeRouter(errors={"free": TimeoutError()}))
    rig.picture()
    assert rig.routes() == ["free"]
    routes = rig.s.state()["routes"]
    assert routes[0]["paused_s"] > 0 and routes[1]["paused_s"] == 0
    assert rig.s.state()["route"] == "paid"
    rig.later()
    assert rig.due()
    rig.sweep()
    assert rig.routes() == ["free", "paid"]
    assert rig.s.room(rig.clock.now) == "a living room"


CHANGES = {
    "model": [eye("free", model="m2"), eye("paid")],
    "key_reference": [eye("free", api_key="${secret:other}"), eye("paid")],
    "switched_off": [eye("free", images=False), eye("paid")],
    "removed": [eye("paid")],
}


@pytest.mark.parametrize("change", sorted(CHANGES))
def test_a_reply_from_a_route_changed_during_its_request_is_dropped(change):
    rig = Rig(cfg=two())
    rig.s.trigger()
    rig.sweep(done=False)
    changed = config(routes=CHANGES[change])

    class Change(FakeRouter):
        async def describe_image(inner, route, prompt, jpeg, max_tokens, first_token_s):
            inner.calls.append((route.name, prompt, jpeg))
            rig.apply(changed)
            raise RateLimited("http_429")

    rig.router = Change()
    rig.finish()
    assert rig.routes() == ["free"]  # dropped: no fallback for this picture
    assert rig.s.room(rig.clock.now) is None and rig.s.live()["history"] == []
    assert rig.s.jpeg() is None and rig.s.state()["last"] is None
    assert all(not r["skipped"] for r in rig.s.state()["routes"])  # no mark
    assert rig.reasons() == []
    rig.later()
    assert rig.due()  # the picture stays due


def test_a_text_reply_from_a_changed_route_is_dropped():
    rig = Rig(cfg=two())
    rig.s.trigger()
    rig.sweep(done=False)

    class Change(FakeRouter):
        async def describe_image(inner, route, prompt, jpeg, max_tokens, first_token_s):
            inner.calls.append((route.name, prompt, jpeg))
            rig.apply(config(routes=[eye("free", model="m2"), eye("paid")]))
            return "a kitchen"

    rig.router = Change()
    rig.finish()
    assert rig.s.room(rig.clock.now) is None


@pytest.mark.parametrize("error", [RateLimited("http_429"), TimeoutError()], ids=["429", "timeout"])
def test_a_stale_reply_after_off_and_on_sets_no_mark(error):
    rig = Rig(cfg=two(max_failures=1))

    class OffOn(FakeRouter):
        async def describe_image(inner, route, prompt, jpeg, max_tokens, first_token_s):
            inner.calls.append((route.name, prompt, jpeg))
            rig.apply(config(routes=[eye("free", images=False), eye("paid", images=False)]))
            rig.apply(two(max_failures=1))
            raise error

    rig.router = OffOn()
    rig.s.trigger()
    rig.sweep()
    assert rig.routes() == ["free"]
    routes = rig.s.state()["routes"]
    assert all(not r["skipped"] and r["paused_s"] == 0 for r in routes)


def test_marks_of_routes_no_longer_seeing_images_are_pruned():
    rig = Rig(cfg=two(), router=FakeRouter(errors={"free": RateLimited("http_429")}))
    rig.picture()
    assert set(rig.s._marks) == {"free", "paid"}
    rig.apply(config(routes=[eye("paid")]))
    assert set(rig.s._marks) == {"paid"}


def test_a_route_turned_off_mid_request_never_gets_the_picture():
    rig = Rig(cfg=two())
    rig.s.trigger()
    rig.sweep(done=False)

    class TurnOffPaid(FakeRouter):
        async def describe_image(inner, route, prompt, jpeg, max_tokens, first_token_s):
            inner.calls.append((route.name, prompt, jpeg))
            rig.apply(config(routes=[eye("free"), eye("paid", images=False)]))
            raise RateLimited("http_429")

    rig.router = TurnOffPaid()
    rig.finish()
    assert rig.routes() == ["free"]
    assert rig.s.state()["last"]["outcome"] == "daily_limit"


def test_adding_reordering_or_removing_image_routes_keeps_the_room_and_history():
    rig = Rig(router=FakeRouter("a kitchen"))
    rig.picture()
    rig.apply(config(routes=[eye(), eye("paid")]))
    assert rig.s.room(rig.clock.now) == "a kitchen" and len(rig.s.live()["history"]) == 1
    assert not rig.s._want_first  # no new first picture owed
    rig.apply(config(routes=[eye("paid"), eye()]))
    rig.apply(config(routes=[eye("paid"), eye(images=False)]))
    assert rig.s.room(rig.clock.now) == "a kitchen" and len(rig.s.live()["history"]) == 1
    assert rig.s.state()["route"] == "paid"
    rig.apply(config(routes=[eye("paid", images=False), eye(images=False)]))
    assert rig.s.room(rig.clock.now) is None and rig.s.live()["history"] == []
    rig.apply(config(routes=[eye("paid"), eye(images=False)]))
    assert rig.s._want_first  # off to on: a first picture is owed
    rig.later()
    assert rig.due()


def test_history_and_last_carry_the_route_and_the_state_holds_no_text_or_key():
    routes = [eye("free", api_key="sk-free-secret"), eye("paid", api_key="sk-paid-secret")]
    router = FakeRouter("Marcello's study", errors={"free": RateLimited("http_429")})
    rig = Rig(cfg=config(routes=routes), router=router)
    rig.picture()
    assert rig.s.state()["last"]["route"] == "paid"
    assert rig.s.live()["history"][0]["route"] == "paid"
    state = repr(rig.s.state())
    assert "sk-" not in state and "study" not in state and "Marcello" not in state
    shape = rig.s.state()  # the roboemot.surroundings report section (SPEC 12A)
    assert "unsupported" not in shape
    assert [set(r) for r in shape["routes"]] == [{"route", "skipped", "reason", "paused_s"}] * 2
