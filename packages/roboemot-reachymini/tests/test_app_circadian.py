"""The circadian runner wired into the app (SPEC 7A, 10, 12A), against a mocked SDK client."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from roboemot.circadian.motion import FX_SOURCE, REST_SOURCE
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.motion import MotionLoop, Priority
from roboemot.vision import Frame
from roboemot_reachymini import app as app_module
from test_app_perception import (  # noqa: F401  (data is a fixture)
    AppWithFakes,
    FakeDetector,
    data,
    new_app,
    start,
    wait_for,
)

START_HOUR = 21.5
SPEED = 1800.0  # one virtual hour in two seconds


class AcceleratedClock:
    """Virtual time running SPEED times faster than real time from START_HOUR."""

    def __init__(self) -> None:
        self.origin = time.monotonic()

    def monotonic(self) -> float:
        return self.origin + (time.monotonic() - self.origin) * SPEED

    def hour(self) -> float:
        return (START_HOUR + (self.monotonic() - self.origin) / 3600) % 24


class DarkSource:
    """A camera in a dark room: a new, nearly black frame on every read."""

    reads = 0

    def read(self):
        type(self).reads += 1
        img = np.full((120, 160, 3), 4, np.uint8)
        img[0, :4, :] = type(self).reads % 250  # a different frame each time
        return Frame(img, time.monotonic(), None)


class DarkApp(AppWithFakes):
    source_class = DarkSource

    def make_circadian_clock(self):
        return AcceleratedClock()


class DarkNoVisionApp(DarkApp):
    """Vision models fail to load: only the idle light reader sees the room."""

    def load_vision_stages(self, cfg, profile):
        raise RuntimeError("no models")


@pytest.fixture()
def fed(monkeypatch):
    log: list = []
    real = MotionLoop.feed

    def feed(self, cmd):
        log.append(cmd)
        real(self, cmd)

    monkeypatch.setattr(MotionLoop, "feed", feed)
    return log


def section(app) -> dict:
    return app.diagnostics.report()["sections"]["roboemot.circadian"]["data"]


def test_a_dark_room_late_in_the_evening_puts_the_robot_to_rest(data, fed, monkeypatch):  # noqa: F811
    monkeypatch.setattr(FakeDetector, "empty", True)  # nobody around
    app = new_app(DarkApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.diagnostics is not None)
        assert wait_for(lambda: section(app).get("state") == "resting", timeout=30)
        report = section(app)
        assert report["light_level"] == "dark" and report["light"]["samples"] > 0
        assert report["light_source"] == "camera" and report["enabled"] is True
        assert {"phase_h", "sleepiness", "hold", "present"} <= set(report)
        assert app.circadian is app.circadian_runner.model
        # the arousal baseline, the amplitude hook and the vision governor follow the state
        states = RoboEmotConfig().circadian.states
        assert app.affect_runner.engine.baseline_arousal == states["resting"].arousal_baseline
        assert app.expression.circadian_factor() == states["resting"].expression_scale
        assert wait_for(lambda: app.perception.pipeline.governor.scale == 0.05)
        # rest posture: IDLE priority, head lowered, antennas down; idle motion stopped
        assert wait_for(lambda: any(c.source == REST_SOURCE for c in fed))
        rest = next(c for c in reversed(fed) if c.source == REST_SOURCE)
        assert rest.priority == Priority.IDLE and rest.targets["head.pitch"] > 0
        fed.clear()
        # held, not released: the posture keeps being fed, and nothing else moves the head
        assert wait_for(lambda: sum(c.source == REST_SOURCE for c in fed) >= 5)
        assert not any(c.source in ("idle", "idle-additive") for c in fed)
        # the app API: wake the robot; it stretches and the posture is released
        seen: list[str] = []  # the wake hold is 300 virtual s, a fraction of a real second
        app.bus.subscribe("circadian.state", lambda m: seen.append(m.payload.state))
        app.wake()
        assert wait_for(lambda: any(c.source == FX_SOURCE for c in fed))
        assert wait_for(lambda: "awake" in seen)
        app.hold("resting", 600)  # minutes of virtual time: 20 s real
        assert wait_for(lambda: section(app)["state"] == "resting")
        assert section(app)["hold"]["state"] == "resting"
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()


def test_without_vision_the_idle_reader_senses_the_light(data, monkeypatch):  # noqa: F811
    monkeypatch.setattr(DarkSource, "reads", 0)
    app = new_app(DarkNoVisionApp)
    (data / "app.yaml").write_text(
        (data / "app.yaml").read_text() + "circadian:\n  idle_sample_interval_s: 1\n"
        "  idle_poll_s: 0.05\n"
    )
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.diagnostics is not None)
        assert app.perception.pipeline is None
        assert wait_for(lambda: section(app)["light"]["samples"] >= 3)
        assert wait_for(lambda: section(app).get("light_level") == "dark")
        assert DarkSource.reads >= 3
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()


def test_clock_only_never_reads_the_camera(data, fed, monkeypatch):  # noqa: F811
    monkeypatch.setattr(DarkSource, "reads", 0)
    (data / "app.yaml").write_text(
        (data / "app.yaml").read_text() + "circadian:\n  light_source: clock_only\n"
    )
    app = new_app(DarkNoVisionApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.diagnostics is not None)
        assert wait_for(lambda: any(c.source == "idle" for c in fed))  # a full app tick has run
        assert section(app).get("state") is not None
        assert app.light_reader is not None and not app.light_reader.running
        assert DarkSource.reads == 0
        assert section(app)["light"]["samples"] == 0
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_disabled_circadian_leaves_the_robot_awake(data, fed):  # noqa: F811
    (data / "app.yaml").write_text(
        (data / "app.yaml").read_text() + "circadian:\n  enabled: false\n"
    )
    app = new_app(DarkNoVisionApp)  # an accelerated clock in quiet hours would force rest
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.diagnostics is not None)
        assert wait_for(lambda: any(c.source == "idle" for c in fed))  # a full app tick has run
        assert not app.light_reader.running
        assert app.circadian is None
        assert section(app)["enabled"] is False and section(app)["state"] == "awake"
        assert not any(c.source == REST_SOURCE for c in fed)
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_the_hold_and_wake_calls_need_a_running_app():
    app = new_app()
    with pytest.raises(RuntimeError):
        app.hold("awake", 5)
    with pytest.raises(RuntimeError):
        app.wake()


def test_circadian_settings_reload_without_a_restart(data):  # noqa: F811
    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.circadian_runner is not None and app.diagnostics is not None)
        assert app.circadian_runner.cfg.circadian.yawn_interval_s == 120.0
        assert app.soothe_runner is None  # voice is off: no speech output, no runner (SPEC 7B)
        (data / "app.yaml").write_text(
            (data / "app.yaml").read_text()
            + "circadian:\n  yawn_interval_s: 45\nlight:\n  sample_interval_s: 3\n"
        )
        assert wait_for(lambda: app.circadian_runner.cfg.circadian.yawn_interval_s == 45.0)
        runner = app.circadian_runner
        assert runner.sampler.cfg.sample_interval_s == 3.0
        assert runner.motion.cfg.yawn_interval_s == 45.0
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def fx(cfg, name):
    return cfg.circadian.states[name]


def crossings(cfg, name, seconds=240.0, dt=0.05):
    """Zero crossings and peak of idle breathing in a state, with the app's phase clock."""
    state = fx(cfg, name)
    t = phase = peak = 0.0
    prev = 0.0
    n = 0
    while t < seconds:
        phase += dt / state.idle_period_scale
        pitch = app_module.idle_command(cfg, t, state.idle_motion_scale, phase).targets[
            "head.pitch"
        ]
        n += prev * pitch < 0
        peak = max(peak, abs(pitch))
        prev = pitch
        t += dt
    return n, peak


def test_drowsy_idle_breathing_is_slower_and_smaller():
    cfg = RoboEmotConfig()
    awake_n, awake_peak = crossings(cfg, "awake")
    drowsy_n, drowsy_peak = crossings(cfg, "drowsy")
    assert drowsy_n < awake_n
    assert drowsy_n == pytest.approx(awake_n / fx(cfg, "drowsy").idle_period_scale, rel=0.1)
    assert drowsy_peak == pytest.approx(awake_peak * fx(cfg, "drowsy").idle_motion_scale, rel=0.05)
    assert awake_peak == pytest.approx(cfg.motion.idle.breathing_amplitude, rel=0.05)


def test_the_breathing_phase_does_not_jump_when_the_pace_changes():
    cfg = RoboEmotConfig()
    t, phase, last = 0.0, 0.0, None
    for scale in (1.0,) * 200 + (1.5,) * 200:
        phase += 0.05 / scale
        pitch = app_module.idle_command(cfg, t, 1.0, phase).targets["head.pitch"]
        if last is not None:
            assert abs(pitch - last) < 0.2 * cfg.motion.idle.breathing_amplitude
        last, t = pitch, t + 0.05


def test_resting_feeds_no_idle_motion_and_clears_it():
    cfg = RoboEmotConfig()
    manifest = load_manifest(
        app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
    )
    loop = MotionLoop(
        manifest,
        MockPoseDriver(home={n: d.home for n, d in manifest.dofs.items()}),
        cfg,
        rate_hz=50,
        clock=lambda: 0.0,
    )
    loop.set_mode("active")
    app_module.feed_idle(loop, cfg, 1.0, False)
    assert "idle" in {c.source for c in loop.core.arbiter.active(1.0)}
    app_module.feed_idle(loop, cfg, 1.1, False, 0.0)  # circadian resting: scale 0
    assert not {c.source for c in loop.core.arbiter.active(1.1)} & {"idle", "idle-additive"}
    app_module.feed_idle(loop, cfg, 1.2, True, 0.0)
    assert not {c.source for c in loop.core.arbiter.active(1.2)} & {"idle", "idle-additive"}
    assert math.isfinite(app_module.idle_command(cfg, 1.0, 0.6).targets["head.pitch"])


def test_the_hold_and_wake_calls_refuse_after_the_app_stopped(data):  # noqa: F811
    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    assert wait_for(lambda: app.circadian_runner is not None)
    stop.set()
    thread.join(15)
    assert not errors
    with pytest.raises(RuntimeError, match="not running"):
        app.hold("awake", 5)
    with pytest.raises(RuntimeError, match="not running"):
        app.wake()


REAL_CLOCK = app_module.RoboEmotApp.make_circadian_clock  # before the conftest fixture patches it


def test_the_default_clock_reads_the_local_hour(data, monkeypatch):  # noqa: F811
    """The shipped path: ``SystemClock`` with only the wall-clock hour stubbed (23:10, quiet
    hours) puts the robot to rest, with nothing else faked."""
    import datetime as dt

    from roboemot.circadian import runner as runner_module

    class Late(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 2, 23, 10, 0)

    monkeypatch.setattr(app_module.RoboEmotApp, "make_circadian_clock", REAL_CLOCK)
    monkeypatch.setattr(runner_module, "datetime", Late)
    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.diagnostics is not None)
        assert wait_for(lambda: section(app).get("state") == "resting")
        assert section(app)["hour"] == pytest.approx(23 + 10 / 60, abs=0.01)
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_the_stretch_follows_the_register_amplitude_and_the_minor_cap(data, fed):  # noqa: F811
    """Nobody is in view: the fail-closed ``unknown`` register (0.7x) and the minor caps are on,
    so the 25 degree antenna stretch peaks at 0.7 x 25 degrees."""
    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.circadian_runner is not None and app.diagnostics is not None)
        app.hold("resting", 5)
        assert wait_for(lambda: section(app)["state"] == "resting")
        app.hold("awake", 5)
        assert wait_for(lambda: any(c.source == FX_SOURCE for c in fed))
        assert wait_for(lambda: section(app)["waking"] is False, timeout=10)
        peak = max(c.targets["antenna_right"] for c in fed if c.source == FX_SOURCE)
        assert peak == pytest.approx(math.radians(25) * 0.7, rel=0.05)
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_the_app_feeds_the_affect_intensity_to_the_circadian_motion(data):  # noqa: F811
    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.circadian_runner is not None and app.affect_runner is not None)
        motion = app.circadian_runner.motion
        app.affect_runner.engine.impulse(time.monotonic(), 0.6, 0.6, "joy")
        expected = app.affect_runner.engine.state(time.monotonic()).intensity
        assert expected > 0
        assert motion.affect_intensity() == pytest.approx(expected, abs=0.05)
    finally:
        stop.set()
        thread.join(15)
    assert not errors
