"""The surroundings service: triggers, gate, attempts, descriptions, failures (SPEC 8,
Surroundings snapshots; design 2026-11-24)."""

from __future__ import annotations

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

    async def describe_image(self, route, prompt, jpeg, max_tokens):
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
        self.s = Surroundings(
            lambda: self.cfg,
            lambda: self.router,
            gate=lambda kind: self.gate_reason,
            speed=self.speeds.get,
            vision=vision,
            frame_spec=SPEC,
            clock=self.clock,
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
    assert not rig.s.wanted()
    bus.publish("circadian.state", circadian("awake"), rig.clock.now)
    assert rig.s.wanted()


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
        async def describe_image(inner, route, prompt, jpeg, max_tokens):
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
    assert rig.s.live()["text"]["shown"] == "Marcello's silver laptop"
