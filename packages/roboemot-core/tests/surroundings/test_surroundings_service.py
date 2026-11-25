"""The surroundings service: triggers, gate, attempts, descriptions, failures (SPEC 8,
Surroundings snapshots; design 2026-11-24)."""

from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np
from roboemot.bus import Bus
from roboemot.circadian.model import CircadianState
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.router import ImagesUnsupported, NoRouteAvailable
from roboemot.surroundings.service import Surroundings
from roboemot.vision import Frame
from roboemot.world import FrameSpec

SPEC = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",))


def eye(**kw) -> LLMRoute:
    return LLMRoute(name="eye", base_url="https://api.example.com/v1", model="m", images=True, **kw)


def config(images: bool = True, **surroundings) -> RoboEmotConfig:
    surroundings.setdefault("min_sharpness", 0.0)
    surroundings.setdefault("candidates", 1)
    route = eye() if images else eye().model_copy(update={"images": False})
    return RoboEmotConfig(llm_routes=[route], surroundings=surroundings)


class FakeRouter:
    def __init__(self, reply="a silver laptop", error=None) -> None:
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


class Rig:
    def __init__(self, cfg=None, router=None, gate=None, speed=None, vision=True) -> None:
        self.cfg = cfg or config()
        self.router = router or FakeRouter()
        self.clock = Clock()
        self.gate_reason = gate
        self.speeds: dict[str, float] = speed or {}
        self.events: list[tuple] = []
        self.day = dt.date(2026, 11, 25)
        self.s = Surroundings(
            lambda: self.cfg,
            lambda: self.router,
            gate=lambda kind: self.gate_reason,
            speed=self.speeds.get,
            vision=vision,
            frame_spec=SPEC,
            clock=self.clock,
            encode=lambda image, quality: b"\xff\xd8" + image.tobytes()[:64],  # no cv2 in CI
            today=lambda: self.day,
        )
        self.s.events = lambda kind, reason, **f: self.events.append((kind, reason, f))

    def frame(self, image=None, yaw_deg: float = 0.0, faces=(), shown=None):
        img = board() if image is None else image
        joint = {"head.yaw": math.radians(yaw_deg), "head.pitch": 0.0}
        self.s.tap(Frame(img, self.clock.now, joint), list(faces), shown)
        self.clock.now += 0.1

    def settle(self, frames: int = 8, **kw):
        for _ in range(frames):
            self.frame(**kw)

    def reasons(self):
        return [r for k, r, _ in self.events if k == "surroundings"]


def circadian(state: str, reason=None) -> CircadianState:
    return CircadianState(12.0, 0.0, 0.0, 0.0, 0.0, state, None, reason)


def board(cell: int = 8, h: int = 120, w: int = 160) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    gray = (((ys // cell) + (xs // cell)) % 2 * 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


THING = (40.0, 40.0, 100.0, 100.0)


def test_switch_off_starts_nothing():
    rig = Rig(cfg=config(images=False))
    assert rig.s.request("shown", "e1", "laptop") == "off"
    assert not rig.s.wanted()
    assert rig.s.state()["state"] == "off"


def test_a_shown_request_ends_described_with_the_sharp_crop_sent():
    rig = Rig()
    assert rig.s.request("shown", "e1", "laptop") is None
    assert rig.s.wanted()
    rig.settle(shown=THING)
    rig.s.step()
    assert rig.s.description("e1") == "a silver laptop"
    assert rig.s.description("e2") is None
    ((name, prompt, jpeg),) = rig.router.calls
    assert name == "eye" and "laptop" in prompt and jpeg[:2] == b"\xff\xd8"
    assert rig.s.jpeg() == jpeg
    assert "described" in rig.reasons()
    assert rig.s.state()["last"]["kind"] == "shown"


def test_wait_description_returns_the_text_or_none_after_the_timeout():
    rig = Rig()
    rig.s.request("shown", "e1", "laptop")
    assert rig.s.wait_description("e1", 0.0) is None
    rig.settle(shown=THING)
    rig.s.step()
    assert rig.s.wait_description("e1", 0.0) == "a silver laptop"


def test_room_on_wake_then_not_again_within_min_interval():
    rig = Rig(router=FakeRouter("a living room, evening light"))
    bus = Bus()
    rig.s.attach(bus)
    bus.publish("robot.awake", None, rig.clock.now)
    rig.settle()
    rig.s.step()
    assert rig.s.room(rig.clock.now) == "a living room, evening light"
    assert rig.s.request("room") == "interval"
    rig.clock.now += rig.cfg.surroundings.room.min_interval_s
    assert rig.s.request("room") is None


def test_leaving_a_rest_requests_a_room_snapshot():
    rig = Rig()
    bus = Bus()
    rig.s.attach(bus)
    bus.publish("circadian.state", circadian("resting", "quiet_hours"), rig.clock.now)
    assert rig.s.state()["attempt"] is None
    bus.publish("circadian.state", circadian("awake"), rig.clock.now)
    assert rig.s.state()["attempt"] == "room"


def test_scan_finished_requests_a_room_snapshot():
    rig = Rig()
    bus = Bus()
    rig.s.attach(bus)
    bus.publish("world.scan_finished", None, rig.clock.now)
    assert rig.s.wanted()


def test_room_text_expires_after_max_age():
    rig = Rig(router=FakeRouter("a kitchen"))
    rig.s.request("room")
    rig.settle()
    rig.s.step()
    assert rig.s.room(rig.clock.now) == "a kitchen"
    later = rig.clock.now + rig.cfg.surroundings.room.max_age_s + 1
    assert rig.s.room(later) is None


def test_a_large_scene_change_with_the_head_still_requests_a_room_snapshot():
    rig = Rig()
    rig.s.request("room")
    rig.settle()
    rig.s.step()
    rig.clock.now += rig.cfg.surroundings.room.min_interval_s
    rig.settle(frames=int(rig.cfg.surroundings.room.check_interval_s / 0.1) + 2, image=board())
    rig.s.step()
    assert len(rig.router.calls) == 1  # the same view: no new snapshot
    changed = np.zeros_like(board())
    rig.settle(frames=int(rig.cfg.surroundings.room.check_interval_s / 0.1) + 2, image=changed)
    rig.s.step()
    assert len(rig.router.calls) == 2


def test_head_moving_ends_the_attempt_with_its_reason():
    rig = Rig(cfg=config(shown={"wait_s": 1.0}))
    rig.frame(yaw_deg=0.0)  # the first pose only anchors the head
    rig.s.request("shown", "e1", "laptop")
    for i in range(1, 16):
        rig.frame(yaw_deg=i * 5.0, shown=THING)
    rig.s.step()
    assert rig.s.description("e1") is None
    assert "head_moving" in rig.reasons()
    assert rig.router.calls == []


def test_object_moving_ends_the_attempt_with_its_reason():
    rig = Rig(cfg=config(shown={"wait_s": 1.0}), speed={"e1": 40.0})
    rig.s.request("shown", "e1", "laptop")
    rig.settle(frames=15, shown=THING)
    rig.s.step()
    assert "object_moving" in rig.reasons()


def test_unsupported_stops_attempts_until_the_route_changes():
    rig = Rig(router=FakeRouter(error=ImagesUnsupported("http_400")))
    rig.s.request("shown", "e1", "laptop")
    rig.settle(shown=THING)
    rig.s.step()
    assert rig.s.state()["state"] == "unsupported"
    assert rig.s.state()["unsupported"] == "http_400"
    assert rig.s.request("shown", "e2", "cup") == "unsupported"
    rig.s.apply_config(rig.cfg)  # an unrelated reload keeps the mark
    assert rig.s.request("shown", "e2", "cup") == "unsupported"
    rig.cfg = RoboEmotConfig(
        llm_routes=[eye().model_copy(update={"model": "m2"})],
        surroundings=rig.cfg.surroundings,
    )
    rig.s.apply_config(rig.cfg)
    assert rig.s.request("shown", "e2", "cup") is None


def test_max_failures_pauses_for_pause_s():
    rig = Rig(cfg=config(max_failures=2, room={"min_interval_s": 60}),
              router=FakeRouter(error=RuntimeError("boom")))  # fmt: skip
    for _ in range(2):
        assert rig.s.request("room") is None
        rig.settle()
        rig.s.step()
        rig.clock.now += 60
    assert rig.s.request("room") == "paused"
    rig.clock.now += rig.cfg.surroundings.pause_s
    assert rig.s.request("room") is None


def test_gate_reasons_asleep_music_busy_offline():
    for reason in ("asleep", "music", "busy"):
        assert Rig(gate=reason).s.request("room") == reason
    rig = Rig()
    rig.router.offline = True
    assert rig.s.request("shown", "e1", "cup") == "offline"


def test_no_vision_is_reported_and_never_raises():
    rig = Rig(vision=False)
    assert rig.s.request("shown", "e1", "cup") == "no_vision"
    assert rig.s.state()["state"] == "no_vision"


def test_route_unavailable_when_key_missing_or_budget_spent():
    rig = Rig()
    rig.router.usable = False
    assert rig.s.request("room") == "route_unavailable"


def test_a_route_error_at_request_time_is_counted_as_unavailable():
    rig = Rig(router=FakeRouter(error=NoRouteAvailable("eye")))
    rig.s.request("room")
    rig.settle()
    rig.s.step()
    assert "route_unavailable" in rig.reasons()


def test_second_shown_replaces_first_and_old_text_never_labels_new_entry():
    rig = Rig()
    rig.s.request("shown", "e1", "laptop")
    assert rig.s.request("shown", "e2", "cup") is None
    rig.settle(shown=THING)
    rig.s.step()
    assert rig.s.description("e2") == "a silver laptop"
    assert rig.s.description("e1") is None
    assert "cup" in rig.router.calls[0][1]


def test_switch_off_mid_request_drops_reply_and_jpeg():
    rig = Rig()
    rig.s.request("shown", "e1", "laptop")
    rig.settle(shown=THING)

    class TurnOff(FakeRouter):
        async def describe_image(inner, route, prompt, jpeg, max_tokens, first_token_s):
            rig.cfg = config(images=False)
            rig.s.apply_config(rig.cfg)
            return "a silver laptop"

    rig.router = TurnOff()
    rig.s.step()
    assert rig.s.description("e1") is None
    assert rig.s.jpeg() is None
    assert not rig.s.wanted()


def test_state_and_logs_hold_no_text_and_no_image_bytes(caplog):
    rig = Rig(router=FakeRouter("Marcello's silver laptop"))
    with caplog.at_level(logging.DEBUG):
        rig.s.request("shown", "e1", "laptop")
        rig.settle(shown=THING)
        rig.s.step()
    state = repr(rig.s.state())
    assert "laptop" not in state and "Marcello" not in state and "e1" not in state
    assert "laptop" not in caplog.text and "Marcello" not in caplog.text
    assert all("laptop" not in repr(f) for _, _, f in rig.events)
    assert rig.s.live()["described"]["shown"] == "Marcello's silver laptop"


def test_cancel_closes_the_open_shown_attempt():
    rig = Rig()
    rig.s.request("shown", "e1", "laptop")
    rig.s.cancel("e2")
    assert rig.s.wanted()
    rig.s.cancel("e1")
    assert rig.s.state()["attempt"] is None


# ---- the gate holds for the whole attempt (review I2)


def test_a_gate_that_closes_during_the_attempt_sends_nothing():
    rig = Rig()
    rig.s.request("room")
    rig.gate_reason = "asleep"
    rig.settle()
    rig.s.step()
    assert rig.router.calls == []
    assert "asleep" in rig.reasons()


def test_the_gate_is_checked_again_right_before_sending():
    rig = Rig()
    rig.s.request("room")
    rig.settle()  # a frame was chosen
    rig.gate_reason = "music"
    rig.s.step()
    assert rig.router.calls == []
    assert "music" in rig.reasons()


def test_going_to_sleep_or_rest_drops_the_attempt_and_the_kept_snapshot():
    for topic, payload in (
        ("robot.asleep", None),
        ("circadian.state", circadian("resting", "quiet_hours")),
    ):
        rig = Rig()
        bus = Bus()
        rig.s.attach(bus)
        rig.s.request("room")
        rig.settle()
        rig.s.step()
        assert rig.s.jpeg() is not None
        rig.clock.now += rig.cfg.surroundings.room.min_interval_s
        rig.s.request("room")
        bus.publish(topic, payload, rig.clock.now)
        assert rig.s.jpeg() is None and rig.s.state()["attempt"] is None


def test_the_state_shows_offline_and_route_unavailable():
    rig = Rig()
    rig.router.offline = True
    assert rig.s.state()["state"] == "offline"
    rig.router.offline = False
    rig.router.usable = False
    assert rig.s.state()["state"] == "route_unavailable"
    rig.router.usable = True
    assert rig.s.state()["state"] == "on"


# ---- a first room snapshot at switch-on and at start (owner, 2026-11-25)


def test_start_with_the_switch_on_takes_a_first_room_snapshot():
    rig = Rig(router=FakeRouter("a study with a desk"))
    rig.settle()
    rig.s.step()
    assert rig.s.room(rig.clock.now) == "a study with a desk"


def test_switch_on_waits_for_a_quiet_moment_without_counting_refusals():
    rig = Rig(cfg=config(images=False), router=FakeRouter("a kitchen"))
    rig.cfg = config()
    rig.s.apply_config(rig.cfg)
    rig.gate_reason = "busy"
    rig.settle(frames=int(rig.cfg.surroundings.room.check_interval_s / 0.1) * 3)
    rig.s.step()
    assert rig.router.calls == [] and rig.reasons() == []
    rig.gate_reason = None
    rig.settle(frames=int(rig.cfg.surroundings.room.check_interval_s / 0.1) + 2)
    rig.s.step()
    assert rig.s.room(rig.clock.now) == "a kitchen"


def test_the_first_room_snapshot_is_taken_once():
    rig = Rig()
    rig.settle()
    rig.s.step()
    rig.settle(frames=int(rig.cfg.surroundings.room.check_interval_s / 0.1) + 2)
    rig.s.step()
    assert len(rig.router.calls) == 1


# ---- the head holds still for a snapshot (owner, 2026-11-25)


def test_holding_lasts_hold_s_after_an_attempt_opens():
    rig = Rig(cfg=config(hold_s=2.0))
    rig.s._want_first = False
    assert not rig.s.holding(rig.clock.now)
    rig.s.request("room")
    start = rig.clock.now
    assert rig.s.holding(start) and rig.s.holding(start + 1.9)
    assert not rig.s.holding(start + 2.1)


def test_no_hold_with_hold_s_zero_or_after_the_attempt_ends():
    rig = Rig(cfg=config(hold_s=0.0))
    rig.s.request("room")
    assert not rig.s.holding(rig.clock.now)
    rig = Rig(cfg=config(hold_s=5.0))
    rig.s.request("room")
    rig.settle()  # the attempt picked its frame and closed
    assert not rig.s.holding(rig.clock.now)


# ---- the daily cap (owner, 2026-11-25: Google's free tier allows ~20 requests a day)


def describe_room(rig) -> None:
    rig.clock.now += rig.cfg.surroundings.room.min_interval_s
    rig.s.request("room")
    rig.settle()
    rig.s.step()


def test_the_daily_cap_stops_snapshots_until_the_next_day():
    rig = Rig(cfg=config(max_per_day=2, room={"max_per_day": 10}))
    rig.s._want_first = False
    describe_room(rig)
    describe_room(rig)
    assert len(rig.router.calls) == 2
    assert rig.s.request("shown", "e1", "cup") == "daily_limit"
    assert rig.s.state()["state"] == "daily_limit"
    rig.day = rig.day + dt.timedelta(days=1)
    assert rig.s.request("shown", "e1", "cup") is None
    assert rig.s.state()["state"] == "on"


def test_room_snapshots_have_their_own_smaller_cap():
    rig = Rig(cfg=config(max_per_day=10, room={"max_per_day": 1}))
    rig.s._want_first = False
    describe_room(rig)
    rig.clock.now += rig.cfg.surroundings.room.min_interval_s
    assert rig.s.request("room") == "daily_limit"
    assert rig.s.request("shown", "e1", "cup") is None  # shown things keep their share


def test_zero_means_no_cap():
    rig = Rig(cfg=config(max_per_day=0, room={"max_per_day": 0}))
    rig.s._want_first = False
    for _ in range(3):
        describe_room(rig)
    assert len(rig.router.calls) == 3


def test_a_429_stops_snapshots_until_the_next_day():
    from roboemot.llm.router import RateLimited

    rig = Rig(router=FakeRouter(error=RateLimited("http_429")))
    rig.s._want_first = False
    describe_room(rig)
    assert "daily_limit" in rig.reasons()
    assert rig.s.request("shown", "e1", "cup") == "daily_limit"
    rig.day = rig.day + dt.timedelta(days=1)
    assert rig.s.request("shown", "e1", "cup") is None


# ---- the last snapshot's outcome and the history (owner, 2026-11-25)


def test_the_last_snapshot_carries_its_outcome_and_a_failure_its_reason():
    rig = Rig(router=FakeRouter(error=TimeoutError()))
    rig.s._want_first = False
    describe_room(rig)
    last = rig.s.state()["last"]
    assert (last["outcome"], last["detail"]) == ("failed", "timeout")
    ok = Rig(router=FakeRouter("a kitchen"))
    ok.s._want_first = False
    describe_room(ok)
    assert ok.s.state()["last"]["outcome"] == "described"


def test_an_empty_reply_is_failed_empty():
    rig = Rig(router=FakeRouter("   "))
    rig.s._want_first = False
    describe_room(rig)
    assert (rig.s.state()["last"]["outcome"], rig.s.state()["last"]["detail"]) == (
        "failed",
        "empty",
    )


def test_the_history_lists_the_last_snapshots_newest_first_with_their_texts():
    rig = Rig(cfg=config(history_size=2, room={"max_per_day": 0}), router=FakeRouter("a kitchen"))
    rig.s._want_first = False
    describe_room(rig)
    rig.router.reply = "a study"
    describe_room(rig)
    rig.router.error = TimeoutError()
    describe_room(rig)
    history = rig.s.live()["history"]
    assert [(h["outcome"], h["text"]) for h in history] == [
        ("failed", None),
        ("described", "a study"),
    ]
    assert all(h["kind"] == "room" and h["age_s"] >= 0 for h in history)


def test_the_history_holds_no_text_in_the_state_and_is_dropped_when_the_switch_goes_off():
    rig = Rig(router=FakeRouter("Marcello's desk"))
    rig.s._want_first = False
    describe_room(rig)
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
    rig.s._want_first = False
    describe_room(rig)
    assert rig.s.state()["last"]["detail"] == "http_503"
