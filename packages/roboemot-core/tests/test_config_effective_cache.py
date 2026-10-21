"""SPEC 11A: ``ConfigStack.effective()`` is cached until a change or a TTL lapse invalidates it."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import RoboEmotConfig

REPO = Path(__file__).resolve().parents[3]
PRESETS = REPO / "configs" / "presets"


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_two_calls_without_a_change_return_one_object_and_validate_once(monkeypatch):
    stack = ConfigStack()
    calls: list[int] = []
    real = RoboEmotConfig.model_validate.__func__
    monkeypatch.setattr(
        RoboEmotConfig,
        "model_validate",
        classmethod(lambda cls, *a, **k: calls.append(1) or real(cls, *a, **k)),
    )
    stack.app = {"affect": {"half_life_s": 12.0}}  # drops the cache the constructor filled
    first = stack.effective()
    assert stack.effective() is first and stack.get("affect.half_life_s") == 12.0
    assert len(calls) == 1


def test_runtime_set_and_clear_give_a_new_object():
    stack = ConfigStack()
    a = stack.effective()
    stack.set("voice.expressivity", 0.2)
    b = stack.effective()
    assert b is not a and b.voice.expressivity == 0.2 and stack.effective() is b
    stack.clear("voice.expressivity")
    c = stack.effective()
    assert c is not b and c.voice.expressivity == a.voice.expressivity


def test_set_many_and_preview_do_not_leave_a_stale_value():
    stack = ConfigStack()
    a = stack.effective()
    stack.set_many({"voice.expressivity": 0.25})
    assert stack.effective().voice.expressivity == 0.25
    assert stack.preview({"voice.expressivity": 0.9}).voice.expressivity == 0.9
    assert stack.effective().voice.expressivity == 0.25 and stack.effective() is not a


def test_a_refused_change_leaves_the_served_config_intact():
    stack = ConfigStack()
    a = stack.effective()
    with pytest.raises(ConfigError):
        stack.set("voice.expressivity", 9.0)
    assert stack.effective().voice.expressivity == a.voice.expressivity


@pytest.mark.parametrize("layer", ["app", "derived"])
def test_layer_writes_invalidate(layer, tmp_path):
    stack = ConfigStack(audit_path=None)
    a = stack.effective()
    stack.set("affect.half_life_s", 33.0, layer=layer)
    b = stack.effective()
    assert b is not a and b.affect.half_life_s == 33.0
    if layer == "derived":
        stack.clear("affect.half_life_s", layer="derived")
        assert stack.effective().affect.half_life_s == a.affect.half_life_s


def test_rebinding_a_layer_invalidates():
    stack = ConfigStack()
    a = stack.effective()
    stack.defaults = {"affect": {"half_life_s": 41.0}}
    assert stack.effective().affect.half_life_s == 41.0
    stack.preset = {"affect": {"half_life_s": 42.0}}
    assert stack.effective().affect.half_life_s == 42.0
    assert stack.effective() is not a


def test_apply_preset_invalidates():
    stack = ConfigStack.from_files(preset="default", preset_dirs=[PRESETS])
    a = stack.effective()
    stack.apply_preset("calm")
    b = stack.effective()
    assert b is not a and b.personality == "calm"


def test_manifest_behaviour_and_channels_invalidate():
    stack = ConfigStack()
    a = stack.effective()
    stack.set_manifest_behaviour({"motion": {"app_tick_hz": 7.0}}, ["head_yaw", "head_pitch"])
    b = stack.effective()
    assert b is not a and b.motion.app_tick_hz == 7.0
    stack.defaults = {"circadian_motion": {"rest_pose_deg": {"head_yaw": 1.0}}}
    assert stack.effective().circadian_motion.rest_pose_deg == {"head_yaw": 1.0}
    stack.channels = frozenset({"nothing"})  # a channels change is re-checked, not served stale
    with pytest.raises(ConfigError, match="unknown channel"):
        stack.effective()


def test_reload_invalidates(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text("affect:\n  half_life_s: 10\n")
    stack = ConfigStack.from_files(app=path)
    a = stack.effective()
    path.write_text("affect:\n  half_life_s: 11\n")
    import os

    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
    assert stack.poll() is not None
    b = stack.effective()
    assert b is not a and b.affect.half_life_s == 11


def test_a_ttl_expires_by_clock_without_any_write():
    clock = Clock()
    stack = ConfigStack(clock=clock)
    base = stack.effective().voice.expressivity
    stack.set("voice.expressivity", 0.2, ttl_s=10)
    held = stack.effective()
    assert held.voice.expressivity == 0.2 and stack.effective() is held
    clock.t += 9.9
    assert stack.effective() is held
    clock.t += 0.2
    assert stack.effective().voice.expressivity == base


def test_the_earliest_of_several_ttls_bounds_the_cache():
    clock = Clock()
    stack = ConfigStack(clock=clock)
    stack.set("voice.expressivity", 0.2, ttl_s=100)
    stack.set("affect.half_life_s", 33.0, ttl_s=10)
    assert stack.effective().affect.half_life_s == 33.0
    clock.t += 11
    cfg = stack.effective()
    assert cfg.affect.half_life_s != 33.0 and cfg.voice.expressivity == 0.2
    clock.t += 100
    assert stack.effective().voice.expressivity != 0.2


def test_poll_after_expiry_reports_the_change():
    clock = Clock()
    stack = ConfigStack(clock=clock)
    stack.set("voice.expressivity", 0.2, ttl_s=10)
    stack.effective()
    clock.t += 11
    change = stack.poll()
    assert change is not None and "voice.expressivity" in change.changed_keys


def test_a_person_call_is_never_cached_and_never_leaks():
    stack = ConfigStack()
    base = stack.effective()
    person = {"voice": {"expressivity": 0.9}}
    p1, p2 = stack.effective(person), stack.effective(person)
    assert p1.voice.expressivity == 0.9 and p1 is not p2
    assert stack.effective() is base and base.voice.expressivity != 0.9
    assert stack.get("voice.expressivity") == base.voice.expressivity


def test_an_error_is_not_cached():
    stack = ConfigStack()
    stack.channels = frozenset({"nothing"})
    stack.defaults = {"circadian_motion": {"rest_pose_deg": {"head_yaw": 1.0}}}
    with pytest.raises(ConfigError):
        stack.effective()
    with pytest.raises(ConfigError):
        stack.effective()
    stack.channels = None
    assert stack.effective().circadian_motion.rest_pose_deg == {"head_yaw": 1.0}


def test_readers_never_see_a_value_older_than_a_returned_write():
    stack = ConfigStack()
    stop = threading.Event()
    floor = {"v": 0.0}
    bad: list[tuple[float, float]] = []

    def read() -> None:
        while not stop.is_set():
            written = floor["v"]
            seen = stack.effective().voice.expressivity
            if seen < written:
                bad.append((written, seen))

    readers = [threading.Thread(target=read) for _ in range(4)]
    for t in readers:
        t.start()
    try:
        for i in range(1, 60):
            v = i / 100
            stack.set("voice.expressivity", v)
            floor["v"] = v
            assert stack.effective().voice.expressivity >= v
    finally:
        stop.set()
        for t in readers:
            t.join()
    assert not bad
