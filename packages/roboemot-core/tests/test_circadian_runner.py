"""SPEC 7A: light sampling, the circadian runner, its motion and the 24-hour replay."""

from __future__ import annotations

import io
import json
import math
import random
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from roboemot.affect.runner import AffectRunner
from roboemot.bus import Bus
from roboemot.circadian import (
    CircadianMotion,
    CircadianRunner,
    IdleLightReader,
    LightSampler,
)
from roboemot.circadian.motion import FX_SOURCE, REST_SOURCE
from roboemot.circadian.replay import (
    LightSample,
    RecordingSink,
    ReplayClock,
    format_line,
    parse_line,
    read_log,
    replay,
    synthetic_day,
    write_log,
)
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.manifest import ManifestError, load_manifest
from roboemot.motion import Priority
from roboemot.vision import LockedSource
from roboemot.vision.types import Frame

MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
)
DAY_S = 24 * 3600


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST, allow_unmeasured_dynamics=True)


@pytest.fixture(scope="module")
def cfg(manifest) -> RoboEmotConfig:
    stack = ConfigStack()
    stack.set_manifest_behaviour(manifest.model.behaviour, manifest.channels)
    return stack.effective()


class FakeClock(ReplayClock):
    def advance(self, dt: float) -> None:
        self.t += dt
        self._hour = (self._hour + dt / 3600) % 24


def grey(level: float, shape=(48, 64, 3)) -> np.ndarray:
    return np.full(shape, level * 255.0).astype(np.uint8)


def sampler(cfg, clock=None) -> tuple[LightSampler, FakeClock]:
    clock = clock or FakeClock()
    return LightSampler(cfg.light, cfg.circadian, clock.monotonic), clock


# ----------------------------------------------------------------------------- sampler


def ndarrays_in(obj, seen=None, depth=0) -> list:
    """Every ndarray reachable from the object's attributes."""
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 6:
        return []
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return [obj]
    found = []
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)) or type(obj).__name__ == "deque":
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())
    elif hasattr(obj, "__slots__"):
        children = [getattr(obj, n, None) for n in obj.__slots__]
    else:
        children = []
    for child in children:
        found += ndarrays_in(child, seen, depth + 1)
    return found


def test_the_sampler_keeps_scalars_only_and_the_ring_is_bounded(cfg):
    s, clock = sampler(cfg)
    rng = np.random.default_rng(0)
    # Three days of one frame per minute: the ring stays at 24 hours.
    for _ in range(3 * 24 * 60):
        frame = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
        assert s.offer(frame, force=True)
        clock.advance(60.0)
    assert ndarrays_in(s) == []
    assert len(s.history()) <= 24 * 60
    assert s._ring.maxlen == 24 * 60
    assert all(isinstance(v, (int, float)) for m in s.history() for v in m)
    assert s.samples == 3 * 24 * 60
    # The minute values are means of the samples taken in that minute.
    assert s.history()[-1].minute > s.history()[0].minute


def test_a_frame_is_measured_at_most_once_per_sample_interval(cfg):
    s, clock = sampler(cfg)
    assert s.offer(grey(0.6)) is True
    assert s.offer(grey(0.6)) is False  # not due
    clock.advance(cfg.light.sample_interval_s - 0.1)
    assert s.offer(grey(0.6)) is False
    clock.advance(0.2)
    assert s.offer(grey(0.6)) is True
    assert s.samples == 2


def test_the_frame_is_read_as_bgr(cfg):
    s, _ = sampler(cfg)
    red = np.zeros((8, 8, 3), np.uint8)
    red[..., 2] = 255  # BGR: pure red
    s.offer(red)
    assert s.history()[-1].luma == pytest.approx(0.299, abs=1e-3)
    s2, _ = sampler(cfg)
    s2.offer(red, bgr=False)  # as RGB it is pure blue
    assert s2.history()[-1].luma == pytest.approx(0.114, abs=1e-3)


def test_clock_only_never_samples(cfg):
    off = cfg.model_copy(
        update={"circadian": cfg.circadian.model_copy(update={"light_source": "clock_only"})}
    )
    s, _ = sampler(off)
    assert s.enabled is False
    assert s.offer(grey(0.6), force=True) is False and s.due() is False and s.idle_due() is False
    reads = []

    class Camera:
        def read(self):
            reads.append(1)
            return Frame(grey(0.6), 0.0, None)

    assert IdleLightReader(s, Camera()).step() is False and reads == []


def test_the_level_needs_confirmation_and_goes_stale(cfg):
    s, clock = sampler(cfg)
    assert s.level() is None
    s.offer_stats(0.5, 0.0)
    clock.advance(1)
    s.offer_stats(0.5, 0.0)
    assert s.level() is None  # agreed for 1 s only (light.confirm_s is 60)
    clock.advance(cfg.light.confirm_s)
    s.offer_stats(0.5, 0.0)
    assert s.level() == "bright"
    clock.advance(cfg.light.stale_s + 1)
    assert s.level() is None  # the latest sample is too old: the clock rules alone


def test_idle_sampling_reads_one_frame_per_interval(cfg):
    s, clock = sampler(cfg)
    reads = []

    class Camera:
        def read(self):
            reads.append(clock.t)
            return Frame(grey(0.6), clock.t, None)

    reader = IdleLightReader(s, Camera())
    interval = cfg.circadian.idle_sample_interval_s
    assert reader.step() is True  # nothing sampled yet
    for _ in range(int(interval) - 1):
        clock.advance(1.0)
        assert reader.step() is False
    assert len(reads) == 1
    clock.advance(1.0)
    assert reader.step() is True and len(reads) == 2
    assert reads[1] - reads[0] == pytest.approx(interval)


def test_a_vision_sample_postpones_the_idle_read(cfg):
    s, clock = sampler(cfg)
    reads = []

    class Camera:
        def read(self):
            reads.append(1)
            return Frame(grey(0.6), 0.0, None)

    reader = IdleLightReader(s, Camera())
    for _ in range(10):  # vision runs: a frame every sample interval
        s.offer(grey(0.6))
        clock.advance(cfg.light.sample_interval_s)
        assert reader.step() is False
    assert reads == []


def test_a_camera_without_frames_is_retried_once_per_interval(cfg):
    s, clock = sampler(cfg)
    calls = []

    class Camera:
        def read(self):
            calls.append(1)

    reader = IdleLightReader(s, Camera())
    reader.step()
    clock.advance(1.0)
    reader.step()
    assert len(calls) == 1  # not every poll
    clock.advance(cfg.circadian.idle_sample_interval_s)
    reader.step()
    assert len(calls) == 2


def test_an_unreadable_frame_or_camera_does_not_raise(cfg):
    s, clock = sampler(cfg)

    class Broken:
        def read(self):
            raise OSError("camera gone")

    class OneDim:
        def read(self):
            return Frame(np.zeros(5, np.uint8), 0.0, None)

    assert IdleLightReader(s, Broken()).step() is False
    clock.advance(cfg.circadian.idle_sample_interval_s)
    assert IdleLightReader(s, OneDim()).step() is False


# ----------------------------------------------------------------------------- the runner


def runner_for(cfg, manifest=None, *, hour=12.0, affect=None, **kw):
    clock = FakeClock()
    clock.set(0.0, hour)
    r = CircadianRunner(
        cfg,
        Bus(),
        affect or RecordingSink(),
        clock=clock,
        dofs=manifest.dofs if manifest else None,
        **kw,
        body_yaw=None,
    )
    return r, clock


def run_for(r, clock, seconds, step=10.0):
    end = clock.t + seconds
    while clock.t < end:
        clock.advance(step)
        r.tick(clock.t)


def feed_light(r, clock, luma, noise=0.0):
    """Samples long enough for the level to be confirmed (light.confirm_s)."""
    step = r.cfg.light.sample_interval_s
    for _ in range(math.ceil(r.cfg.light.confirm_s / step) + 1):
        r.sampler.offer_stats(luma, noise)
        clock.advance(step)


def test_light_changes_raise_events_only_after_a_confirmed_level(cfg):
    sink = RecordingSink()
    r, clock = runner_for(cfg, affect=sink)
    feed_light(r, clock, 0.03)  # first level: dark, no event
    r.tick(clock.t)
    run_for(r, clock, 60)
    assert sink.events == []
    feed_light(r, clock, 0.6)
    run_for(r, clock, 20)
    assert [e for _, e in sink.events] == ["light.on"]
    feed_light(r, clock, 0.03)
    run_for(r, clock, 20)
    assert [e for _, e in sink.events] == ["light.on", "light.off"]


def test_light_noise_around_a_threshold_does_not_flap(cfg):
    sink = RecordingSink()
    r, clock = runner_for(cfg, affect=sink)
    feed_light(r, clock, 0.03)
    run_for(r, clock, 20)
    rng = random.Random(1)
    edge = cfg.light.dark_luma
    for _ in range(400):  # a flickering lamp right at the dark edge
        r.sampler.offer_stats(max(0.0, edge + rng.gauss(0, 0.012)), 0.0)
        clock.advance(cfg.light.sample_interval_s)
        r.tick(clock.t)
    assert len(sink.events) <= 2  # at most one on/off pair, nothing like one per sample


def test_dim_to_bright_raises_no_event(cfg):
    sink = RecordingSink()
    r, clock = runner_for(cfg, affect=sink)
    feed_light(r, clock, 0.25)
    run_for(r, clock, 20)
    feed_light(r, clock, 0.6)
    run_for(r, clock, 20)
    assert sink.events == []


def test_the_runner_publishes_on_change_and_periodically(cfg):
    bus = Bus()
    got = []
    bus.subscribe("circadian.state", lambda m: got.append((m.stamp, m.payload.state)))
    clock = FakeClock()
    clock.set(0.0, 12.0)
    r = CircadianRunner(cfg, bus, RecordingSink(), clock=clock, body_yaw=None)
    run_for(r, clock, 10)
    assert len(got) == 1  # the first state
    run_for(r, clock, cfg.circadian.publish_period_s * 3, step=10)
    assert 3 <= len(got) <= 5  # periodic, not per update
    n = len(got)
    r.hold("resting", 5)
    run_for(r, clock, 10)
    assert len(got) == n + 1 and got[-1][1] == "resting"  # a change is published at once


def test_hold_expires_and_wake_does_not_reset_process_c(cfg):
    r, clock = runner_for(cfg, hour=23.0)  # quiet hours: resting
    run_for(r, clock, 30, step=10)
    assert r.model.state == "resting"
    shift, c_before = r.model.phase_shift_h, r.model.process_c(r.model.t_h)
    r.hold("awake", 30)
    run_for(r, clock, 20, step=10)
    assert r.model.state == "awake"
    run_for(r, clock, 29 * 60, step=10)
    assert r.model.state == "awake"
    run_for(r, clock, 2 * 60, step=10)  # the hold is over: the quiet hours rule again
    assert r.model.state == "resting"
    assert r.model.phase_shift_h == shift  # C is untouched by a hold
    assert r.model.process_c(r.model.t_h) != c_before  # it moved with the clock only
    r.wake()
    run_for(r, clock, 20, step=10)
    assert r.model.state == "awake" and r.model.phase_shift_h == shift


def test_awake_stops_sleep_pressure_from_decaying(cfg):
    a, ca = runner_for(cfg, hour=23.0)
    b, cb = runner_for(cfg, hour=23.0)
    run_for(a, ca, 10)
    run_for(b, cb, 10)
    b.hold("awake", 90)
    run_for(a, ca, 3600)
    run_for(b, cb, 3600)
    assert a.model.state == "resting" and b.model.state == "awake"
    assert b.model.s > a.model.s  # a rested robot decays S, a woken one does not


def test_hold_arguments_are_validated(cfg):
    r, _ = runner_for(cfg)
    for bad in (("sleeping", 5), ("awake", 0), ("awake", float("nan")), ("awake", 1e9)):
        with pytest.raises(ValueError):
            r.hold(*bad)


def test_disabled_runner_is_inert(cfg):
    off = cfg.model_copy(update={"circadian": cfg.circadian.model_copy(update={"enabled": False})})
    sink = RecordingSink()
    r, clock = runner_for(off, affect=sink, hour=23.0)
    run_for(r, clock, 600)
    assert r.last is None and r.model.state == "awake" and sink.effects == []
    assert r.amplitude_factor() == 1.0 and r.commands(clock.t) == []


def test_presence_is_the_weak_second_cue(cfg):
    bus = Bus()
    clock = FakeClock()
    clock.set(0.0, 23.0)
    r = CircadianRunner(cfg, bus, RecordingSink(), clock=clock, body_yaw=None)
    run_for(r, clock, 60)
    assert r.model.state == "resting"
    bus.publish("percept.face", object(), clock.t)
    run_for(r, clock, 20)
    assert r.present() and r.model.state != "resting"
    run_for(r, clock, cfg.circadian.presence_window_s + 20)
    assert not r.present() and r.model.state == "resting"
    r2 = CircadianRunner(
        cfg, bus, RecordingSink(), clock=clock, engaged=lambda: True, body_yaw=None
    )
    assert r2.present()
    bus.publish("attention.target", None, clock.t)  # a cleared target is not presence
    r3 = CircadianRunner(cfg, Bus(), None, clock=clock, body_yaw=None)
    r3.bus.publish("attention.target", None, 0.0)
    assert not r3.present()


def test_presence_window_follows_the_detection_period(cfg):
    clock = FakeClock()
    r = CircadianRunner(cfg, Bus(), None, clock=clock, detection_period=lambda: 60.0, body_yaw=None)
    r.note_presence()
    clock.advance(cfg.circadian.presence_window_s + 30)
    assert r.present()  # three slow detection periods outlast the base window
    clock.advance(cfg.circadian.presence_periods * 60.0)
    assert not r.present()


def test_the_arousal_baseline_follows_the_state_through_the_affect_runner(cfg):
    bus = Bus()
    affect = AffectRunner(cfg.affect, cfg.appraisal, bus, 0.0)
    clock = FakeClock()
    clock.set(0.0, 12.0)
    r = CircadianRunner(cfg, bus, affect, clock=clock, body_yaw=None)
    run_for(r, clock, 10)
    states = cfg.circadian.states
    assert affect.engine.baseline_arousal == states["awake"].arousal_baseline
    r.hold("drowsy", 10)
    run_for(r, clock, 10)
    assert affect.engine.baseline_arousal == states["drowsy"].arousal_baseline
    r.hold("resting", 10)
    run_for(r, clock, 10)
    assert affect.engine.baseline_arousal == states["resting"].arousal_baseline
    affect.close()
    r.close()


def test_the_expression_amplitude_factor_follows_the_state(cfg):
    r, clock = runner_for(cfg)
    run_for(r, clock, 10)
    assert r.amplitude_factor() == cfg.circadian.states["awake"].expression_scale == 1.0
    r.hold("resting", 5)
    run_for(r, clock, 10)
    assert r.amplitude_factor() == cfg.circadian.states["resting"].expression_scale
    assert 0 < r.amplitude_factor() < 1  # resting still expresses, only smaller


def test_the_snapshot_has_numbers_and_names_only(cfg):
    r, clock = runner_for(cfg)
    feed_light(r, clock, 0.6)
    run_for(r, clock, 20)
    snap = r.snapshot()
    assert snap["state"] == "awake" and snap["light_level"] == "bright"
    assert {"phase_h", "sleepiness", "light", "hold", "enabled"} <= set(snap)
    assert ndarrays_in(snap) == []
    json.dumps(snap)


def test_hot_reload_of_the_thresholds(cfg):
    r, clock = runner_for(cfg)
    new = cfg.model_copy(
        update={"circadian": cfg.circadian.model_copy(update={"update_period_s": 100.0})}
    )
    r.apply_config(new)
    assert r.model.cfg.update_period_s == 100.0 and r.sampler.circadian.update_period_s == 100.0


# ----------------------------------------------------------------------------- motion


def drive(motion, seconds, state, step=0.05, start=0.0, attending=False):
    out = []
    t = start
    while t < start + seconds:
        out.append((t, motion.commands(t, state, attending)))
        t += step
    return out


def test_drowsy_yawns_now_and_then_additive_at_expression_priority(cfg, manifest):
    m = CircadianMotion(
        cfg.circadian,
        cfg.circadian_motion,
        manifest.dofs,
        rng=random.Random(3),
        body_yaw="body_yaw",
    )
    run = drive(m, 1200, "drowsy")
    yawns = [t for (t, cmds), (_, prev) in zip(run[1:], run, strict=False) if cmds and not prev]
    interval = cfg.circadian.yawn_interval_s
    assert 5 <= len(yawns) <= 20
    gaps = np.diff(yawns)
    assert (gaps >= interval * (1 - cfg.circadian.yawn_jitter) - 1).all()
    assert gaps.std() > 1.0  # jittered, not regular
    cmd = next(c for _, cmds in run for c in cmds)
    assert cmd.source == FX_SOURCE and cmd.additive and cmd.priority == Priority.EXPRESSION
    assert cmd.expires is not None and "antenna_right" in cmd.targets
    assert "body_yaw" not in cmd.targets


def test_no_yawn_while_awake_resting_or_attending(cfg, manifest):
    m = CircadianMotion(
        cfg.circadian,
        cfg.circadian_motion,
        manifest.dofs,
        rng=random.Random(3),
        body_yaw="body_yaw",
    )
    for state, attending in (("awake", False), ("resting", False), ("drowsy", True)):
        run = drive(m, 600, state, attending=attending)
        assert not any(c.source == FX_SOURCE for _, cmds in run for c in cmds), state


def test_resting_commands_the_rest_posture_at_idle_priority(cfg, manifest):
    m = CircadianMotion(cfg.circadian, cfg.circadian_motion, manifest.dofs, body_yaw="body_yaw")
    cmds = m.commands(1.0, "resting")
    assert [c.source for c in cmds] == [REST_SOURCE]
    rest = cmds[0]
    assert rest.priority == Priority.IDLE and not rest.additive and rest.expires is not None
    assert rest.targets["head.pitch"] > 0  # lowered (positive pitch is down)
    assert rest.targets["antenna_right"] < 0 < rest.targets["antenna_left"]  # drooped
    assert set(rest.targets) <= set(manifest.dofs)
    # attention owns the head: the posture is withheld so it can win
    assert m.commands(1.1, "resting", attending=True) == []
    # and nothing outside resting
    assert m.commands(1.2, "awake") == [] and m.commands(1.3, "drowsy") == []


def test_waking_plays_a_three_second_stretch(cfg, manifest):
    m = CircadianMotion(cfg.circadian, cfg.circadian_motion, manifest.dofs, body_yaw="body_yaw")
    assert m.start_waking(10.0)
    run = drive(m, 4.0, "drowsy", start=10.0)
    active = [t for t, cmds in run if cmds]
    assert active[0] == pytest.approx(10.0) and 2.9 <= active[-1] - active[0] <= 3.05
    peak = max(c.targets["antenna_right"] for _, cmds in run for c in cmds)
    assert peak > 0 and m.playing == ""


def test_primitives_stay_inside_the_manifest_range(cfg, manifest):
    m = CircadianMotion(cfg.circadian, cfg.circadian_motion, manifest.dofs, body_yaw="body_yaw")
    m.start_waking(0.0)
    cmds = [c for _, cs in drive(m, 3.0, "drowsy") for c in cs]
    assert cmds
    for c in cmds:
        for name, v in c.targets.items():
            d = manifest.dofs[name]
            assert d.min - d.home <= v <= d.max - d.home


def test_a_robot_without_primitives_just_skips_them(cfg, manifest):
    bare = cfg.circadian_motion.model_copy(
        update={"yawn": None, "stretch": None, "rest_pose_deg": {}}
    )
    m = CircadianMotion(cfg.circadian, bare, manifest.dofs, body_yaw="body_yaw")
    assert m.start_waking(0.0) is False
    assert drive(m, 600, "drowsy") and m.commands(1.0, "resting") == []


def test_waking_follows_lights_on_and_the_wake_call(cfg, manifest):
    sink = RecordingSink()
    r, clock = runner_for(cfg, manifest, hour=3.0, affect=sink)  # the quiet hours
    feed_light(r, clock, 0.03)
    run_for(r, clock, 120)
    assert r.model.state == "resting"
    clock.set(clock.t, 8.0)  # morning
    feed_light(r, clock, 0.6)  # lights on
    r.tick(clock.t)
    run_for(r, clock, 20)
    assert r.model.state != "resting"
    assert ("light.on") in [e for _, e in sink.events]
    assert r.motion.playing == "stretch"
    assert "circadian.woken_at_night" not in [e for _, e in sink.events]  # morning, not night
    # woken in the quiet hours by the app: a stretch and the (mild) appraisal event
    r2, c2 = runner_for(cfg, manifest, hour=3.0, affect=sink)
    run_for(r2, c2, 120)
    r2.wake()
    run_for(r2, c2, 10)
    assert r2.motion.playing == "stretch"
    assert [e for _, e in sink.events].count("circadian.woken_at_night") == 1
    assert r2.snapshot()["waking"] is True


def test_being_woken_at_night_dips_valence_briefly_and_never_refuses(cfg):
    bus = Bus()
    affect = AffectRunner(cfg.affect, cfg.appraisal, bus, 0.0)
    clock = FakeClock()
    clock.set(0.0, 3.0)
    r = CircadianRunner(cfg, bus, affect, clock=clock, body_yaw=None)
    run_for(r, clock, 120)
    r.wake()
    run_for(r, clock, 10)
    assert affect.engine.valence < 0
    affect.engine.update(clock.t + 10 * cfg.affect.half_life_s)
    assert affect.engine.valence > -0.05  # brief
    affect.close()
    r.close()


def test_light_off_makes_the_robot_slightly_sleepy(cfg):
    from roboemot.affect import AffectEngine

    e = AffectEngine(cfg.affect, cfg.appraisal)
    assert e.on_event("light.off", 0.0)
    assert e.state(0.1).label == "sleepy" and e.arousal < 0


# ----------------------------------------------------------------------------- replay


def test_replay_of_a_synthetic_day_goes_awake_drowsy_resting_awake(cfg):
    samples = synthetic_day()
    assert len(samples) == 24 * 60
    sink = RecordingSink()
    started = time.perf_counter()
    result = replay(samples, cfg, sink=sink)
    wall = time.perf_counter() - started
    assert wall < 30.0  # generous: it takes milliseconds, this only catches a hang
    assert result.states() == ["awake", "drowsy", "resting", "drowsy", "awake"]
    drowsy = result.first_hour("drowsy")
    resting = result.first_hour("resting")
    morning = result.first_hour("awake", after_h=12)
    assert 21.0 <= drowsy <= 21.5  # SPEC 7A: about 21:15
    assert resting == pytest.approx(22.5, abs=0.1)  # quiet hours start
    assert 7.0 <= morning <= 7.1  # SPEC 7A: 07:00 to 07:05
    events = [e for _, e in result.events]
    assert events == ["light.off", "light.on"]
    off_hour = 8 + result.events[0][0] / 3600
    on_hour = 8 + result.events[1][0] / 3600
    assert 22.0 <= off_hour <= 22.5  # lights off at 22:15 plus the confirmation
    assert 6.58 <= on_hour % 24 <= 7.08  # SPEC 7A: about 06:50
    # the baseline moved with the states
    assert [fx.arousal_baseline for _, fx in sink.effects] == [
        cfg.circadian.states[name].arousal_baseline for name in result.states()
    ]


def test_quiet_hours_force_resting_in_the_replay(cfg):
    result = replay(synthetic_day(), cfg)
    quiet = [p for p in result.timeline if p.hour >= 22.75 or p.hour < 6.75]
    assert quiet and {p.state for p in quiet} == {"resting"}
    noon = [p for p in result.timeline if 11 <= p.hour < 15]
    assert {p.state for p in noon} == {"awake"}


def test_people_around_at_night_keep_the_robot_awake_and_wake_it(cfg):
    samples = synthetic_day(presence=[(23.0, 23.5)])
    result = replay(samples, cfg)
    events = [e for _, e in result.events]
    assert "circadian.woken_at_night" in events
    during = [p.state for p in result.timeline if 23.1 <= p.hour < 23.45]
    assert during and "resting" not in during
    after = [p.state for p in result.timeline if p.hour >= 23.8 or p.hour < 6]
    assert set(after) == {"resting"}


def test_clock_only_ignores_the_light(cfg):
    clock_only = cfg.model_copy(
        update={"circadian": cfg.circadian.model_copy(update={"light_source": "clock_only"})}
    )
    result = replay(synthetic_day(), clock_only)
    assert result.events == [] and "resting" in result.states()
    assert all(p.light is None for p in result.timeline)


def test_a_late_household_shifts_the_phase_across_days(cfg):
    day = synthetic_day(hours=72)
    late = [
        LightSample(
            s.when,
            0.25 if 22.25 <= s.when.hour + s.when.minute / 60 < 24 or s.when.hour < 1 else s.luma,
            s.noise,
        )
        for s in day
    ]
    shifted = replay(late, cfg).final.phase_shift_h
    assert abs(shifted) <= cfg.circadian.max_shift_h_per_day * 3 + 1e-9


# ----------------------------------------------------------------------------- log format


def test_the_light_log_round_trips_through_jsonl(tmp_path):
    samples = synthetic_day(hours=2)
    path = tmp_path / "day.jsonl"
    write_log(samples, path)
    again = read_log(path)
    assert len(again) == len(samples)
    assert again[0].when == samples[0].when and again[5].luma == pytest.approx(
        samples[5].luma, abs=1e-4
    )
    first = json.loads(path.read_text().splitlines()[0])
    assert set(first) == {"t", "luma", "noise", "exposure_s", "gain", "presence"}


def test_the_log_accepts_iso_epoch_and_optional_fields():
    s = parse_line('{"t": "2026-03-02T07:05:00+01:00", "luma": 0.4}')
    assert (s.when.hour, s.when.minute) == (7, 5) and s.noise == 0.0 and s.presence is False
    e = parse_line(
        '{"t": 1772433900, "luma": 0.4, "exposure_s": 0.01, "gain": 4, "presence": true}'
    )
    assert e.exposure_s == 0.01 and e.gain == 4.0 and e.presence is True
    assert parse_line(format_line(e)).luma == e.luma
    lines = io.StringIO('\n{"t": "2026-03-02T07:05:00", "luma": 0.4}\n\n')
    assert len(read_log(lines)) == 1


@pytest.mark.parametrize(
    "line",
    [
        '{"luma": 0.4}',
        '{"t": "2026-03-02T07:05:00"}',
        "not json SECRET",
        '{"t": "SECRET", "luma": 0.4}',
        '{"t": "2026-03-02T07:05:00", "luma": 7}',
        '{"t": 1e400, "luma": 0.4}',
        '{"t": 1e30, "luma": 0.4}',
        '{"t": "2026-03-02T07:05:00", "luma": NaN}',
        '{"t": "2026-03-02T07:05:00", "luma": 0.4, "noise": NaN}',
        '{"t": "2026-03-02T07:05:00", "luma": 0.4, "noise": -0.1}',
        '{"t": "2026-03-02T07:05:00", "luma": 0.4, "exposure_s": Infinity}',
        '{"t": "2026-03-02T07:05:00", "luma": 0.4, "gain": -1}',
        '{"t": "2026-03-02T07:05:00", "luma": 0.4, "gain": "SECRET"}',
        '{"t": "2026-03-02T07:05:00", "luma": 0.4, "presence": "false"}',
        '[1, 2, "SECRET"]',
    ],
)
def test_a_bad_log_line_is_rejected_without_echoing_it(line):
    with pytest.raises(ValueError) as err:
        parse_line(line, 7)
    assert "line 7" in str(err.value)
    assert "SECRET" not in str(err.value) and line not in str(err.value)
    # no chained exception either: its message could carry the content
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_the_line_number_counts_blank_lines():
    lines = ['{"t": "2026-03-02T07:05:00", "luma": 0.4}', "", "garbage"]
    with pytest.raises(ValueError, match="line 3"):
        read_log(lines)


def test_a_log_out_of_order_is_rejected():
    lines = [
        '{"t": "2026-03-02T07:06:00", "luma": 0.4}',
        '{"t": "2026-03-02T07:05:00", "luma": 0.4}',
    ]
    with pytest.raises(ValueError, match="order"):
        read_log(lines)


def test_circadian_config_validation_of_the_new_fields():
    stack = ConfigStack(app={"circadian_motion": {"rest_pose_deg": {"wheels": 1}}})
    with pytest.raises(ConfigError, match="unknown channel 'wheels'"):
        stack.set_manifest_behaviour({}, {"head.yaw"})
    with pytest.raises(ConfigError):
        ConfigStack(app={"circadian": {"yawn_jitter": 2.0}})
    assert {f.name for f in fields(LightSample)} >= {"when", "luma", "noise", "presence"}


def test_sensor_noise_around_its_threshold_in_a_dim_room_does_not_flap(cfg):
    """Auto-exposure case: a dim room whose image noise hovers at the dark-noise threshold."""
    sink = RecordingSink()
    r, clock = runner_for(cfg, affect=sink)
    feed_light(r, clock, 0.25)
    run_for(r, clock, 20)
    rng = random.Random(2)
    edge = cfg.light.noise_dark_threshold
    for _ in range(600):
        r.sampler.offer_stats(0.25, max(0.0, edge + rng.gauss(0, 0.008)))
        clock.advance(cfg.light.sample_interval_s)
        r.tick(clock.t)
    assert len(sink.events) <= 2


# ----------------------------------------------------------------------------- review fixes


@pytest.mark.parametrize("period", [10.0, 600.0])
def test_wake_counts_from_now_not_from_the_last_update(cfg, period):
    c = cfg.circadian.model_copy(update={"update_period_s": period, "wake_hold_s": 10.0})
    r, clock = runner_for(cfg.model_copy(update={"circadian": c}), hour=23.0)
    r.tick(clock.t)  # the first update; the next is due `period` later
    clock.advance(min(period, 10.0) - 1.0)  # the model's clock lags real time by this much
    called = clock.t
    r.wake()
    assert r.snapshot()["hold"]["left_s"] == pytest.approx(10.0, abs=0.01)
    while clock.t < called + 9.9:
        clock.advance(0.1)
        r.tick(clock.t)
    assert r.model.state == "awake"  # still held 9.9 s after the call
    clock.advance(20.0 + period)
    r.tick(clock.t)
    assert r.model.state == "resting"  # and over afterwards


def test_a_hold_reports_the_time_left_from_now(cfg):
    r, clock = runner_for(cfg, hour=12.0)
    r.tick(clock.t)
    clock.advance(cfg.circadian.update_period_s - 1)
    r.hold("drowsy", 1)
    assert r.snapshot()["hold"]["left_s"] == pytest.approx(60.0, abs=0.01)


def test_a_level_does_not_revive_after_a_gap_longer_than_stale_s(cfg):
    s, clock = sampler(cfg)
    s.offer_stats(0.03, 0.0)
    clock.advance(cfg.light.confirm_s)
    s.offer_stats(0.03, 0.0)
    assert s.level() == "dark"
    clock.advance(cfg.light.stale_s + 1)
    assert s.level() is None
    s.offer_stats(0.03, 0.0)  # a single sample after the gap
    assert s.level() is None  # not the old confirmed level; the streak starts anew
    clock.advance(cfg.light.confirm_s / 2)
    s.offer_stats(0.03, 0.0)
    assert s.level() is None
    clock.advance(cfg.light.confirm_s / 2)
    s.offer_stats(0.03, 0.0)
    assert s.level() == "dark"


def test_a_streak_does_not_span_a_gap(cfg):
    s, clock = sampler(cfg)
    s.offer_stats(0.6, 0.0)
    clock.advance(cfg.light.stale_s + 1)
    s.offer_stats(0.6, 0.0)  # long after the streak began, but across a gap
    assert s.level() is None


def test_disabling_mid_resting_hands_the_robot_back_awake(cfg, manifest):
    bus = Bus()
    affect = AffectRunner(cfg.affect, cfg.appraisal, bus, 0.0)
    r, clock = runner_for(cfg, manifest, hour=23.0, affect=affect)
    run_for(r, clock, 60)
    assert r.model.state == "resting" and r.commands(clock.t)
    assert affect.engine.baseline_arousal == cfg.circadian.states["resting"].arousal_baseline
    off = cfg.model_copy(update={"circadian": cfg.circadian.model_copy(update={"enabled": False})})
    r.apply_config(off)
    assert affect.engine.baseline_arousal == cfg.circadian.states["awake"].arousal_baseline
    assert r.model.state == "awake" and r.commands(clock.t) == [] and r.last is None
    assert r.motion.playing == "" and r.amplitude_factor() == 1.0
    r.apply_config(cfg)  # and back on: the rhythm resumes from the clock
    run_for(r, clock, 30)
    assert r.model.state == "resting"
    assert affect.engine.baseline_arousal == cfg.circadian.states["resting"].arousal_baseline
    affect.close()
    r.close()


def test_disabling_stops_a_stretch_in_progress(cfg, manifest):
    r, clock = runner_for(cfg, manifest, hour=3.0)
    run_for(r, clock, 60)
    r.wake()
    run_for(r, clock, 10)
    assert r.motion.playing == "stretch"
    off = cfg.model_copy(update={"circadian": cfg.circadian.model_copy(update={"enabled": False})})
    r.apply_config(off)
    assert r.motion.playing == "" and r.commands(clock.t) == []


def test_stretch_follows_the_register_and_the_ceiling(cfg, manifest):
    def stretch(amplitude, ceiling=2.0):
        m = CircadianMotion(cfg.circadian, cfg.circadian_motion, manifest.dofs, body_yaw="body_yaw")
        m.set_register(amplitude, ceiling)
        m.start_waking(0.0)
        cmds = [c for _, cs in drive(m, 3.0, "drowsy") for c in cs]
        return max(c.targets["antenna_right"] for c in cmds)

    full = stretch(1.0)
    assert stretch(0.7) == pytest.approx(full * 0.7, rel=0.01)
    assert stretch(1.2) == pytest.approx(full * 1.2, rel=0.01)
    assert stretch(1.2, ceiling=0.5) == pytest.approx(full * 0.5, rel=0.01)  # capped


def test_a_register_that_tightens_mid_play_scales_the_primitive_down_only(cfg, manifest):
    m = CircadianMotion(cfg.circadian, cfg.circadian_motion, manifest.dofs, body_yaw="body_yaw")
    m.set_register(1.0, 2.0)
    m.start_waking(0.0)
    mid = m.commands(1.0, "drowsy")[0].targets["antenna_right"]
    m.set_register(0.5, 2.0)  # a stricter register: someone with overrides walks in
    assert m.commands(1.0, "drowsy")[0].targets["antenna_right"] == pytest.approx(
        mid * 0.5, rel=0.01
    )
    m.set_register(1.2, 2.0)  # looser again: never above where it started
    assert m.commands(1.0, "drowsy")[0].targets["antenna_right"] == pytest.approx(mid, rel=0.01)


def test_a_quiet_hours_hold_is_not_a_night_wake_but_a_person_is(cfg, manifest):
    sink = RecordingSink()
    r, clock = runner_for(cfg, manifest, hour=23.0, affect=sink)
    run_for(r, clock, 60)
    r.hold("awake", 5)
    run_for(r, clock, 20)
    assert r.model.state == "awake"
    assert "circadian.woken_at_night" not in [e for _, e in sink.events]
    assert r.motion.playing == "stretch"  # a hold still wakes it with a stretch
    r2, c2 = runner_for(cfg, manifest, hour=23.0, affect=sink)
    run_for(r2, c2, 60)
    r2.note_presence()
    run_for(r2, c2, 20)
    assert [e for _, e in sink.events] == ["circadian.woken_at_night"]


def test_light_alone_does_not_wake_the_robot_in_the_quiet_hours(cfg):
    sink = RecordingSink()
    r, clock = runner_for(cfg, hour=23.0, affect=sink)
    feed_light(r, clock, 0.03)
    run_for(r, clock, 60)
    feed_light(r, clock, 0.6)
    run_for(r, clock, 60)
    assert "light.on" in [e for _, e in sink.events]
    assert r.model.state == "resting"  # SPEC 7A: only a person or a wake call wakes it at night
    assert "circadian.woken_at_night" not in [e for _, e in sink.events]


def test_people_flapping_at_night_replay_the_stretch_only_after_the_cooldown(cfg, manifest):
    r, clock = runner_for(cfg, manifest, hour=23.0)
    run_for(r, clock, 60)
    plays = []

    def one_visit():
        r.note_presence()
        run_for(r, clock, 20)
        plays.append(r.motion.playing == "stretch")
        for _ in range(100):  # let any stretch play out
            clock.advance(0.05)
            r.commands(clock.t)
        run_for(r, clock, cfg.circadian.presence_window_s + 30)
        assert r.model.state == "resting"

    one_visit()
    one_visit()  # well inside the cooldown
    assert plays == [True, False]
    run_for(r, clock, cfg.circadian.stretch_cooldown_s)
    one_visit()
    assert plays[-1] is True


def test_a_rest_pose_the_robot_cannot_hold_fails_at_manifest_load():
    import copy

    import yaml

    raw = yaml.safe_load(MANIFEST.read_text())
    load_manifest(copy.deepcopy(raw), allow_unmeasured_dynamics=True)
    for edit in (
        lambda m: m.__setitem__("rest_pose_deg", {"head.pitch": 80}),
        lambda m: m.__setitem__("rest_pose_deg", {"wheels": 1}),
        lambda m: m["yawn"]["channels"].__setitem__("tail", [[0, 0], [1, 5], [3, 0]]),
        lambda m: m.__setitem__("stretch", {"channels": {"head.pitch": [[0, 1], [1, 0]]}}),
    ):
        bad = copy.deepcopy(raw)
        edit(bad["behaviour"]["circadian_motion"])
        with pytest.raises(ManifestError, match="circadian_motion"):
            load_manifest(bad, allow_unmeasured_dynamics=True)


def test_the_frame_source_is_never_read_by_two_threads_at_once():
    import threading

    inside = []
    overlaps = []

    class Camera:
        def camera_model(self, width, height):
            return ("intrinsics", width, height)

        def read(self):
            inside.append(1)
            if len(inside) > 1:
                overlaps.append(1)
            time.sleep(0.002)
            inside.pop()

    shared = LockedSource(Camera())
    threads = [
        threading.Thread(target=lambda: [shared.read() for _ in range(25)]) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == [] and shared.camera_model(4, 3) == ("intrinsics", 4, 3)
    with pytest.raises(AttributeError):
        shared.nothing  # noqa: B018


def test_speech_is_presence_and_the_wake_word_wakes(cfg):
    from types import SimpleNamespace

    bus = Bus()
    clock = FakeClock()
    clock.set(0.0, 23.0)
    r = CircadianRunner(cfg, bus, RecordingSink(), clock=clock, body_yaw=None)
    run_for(r, clock, 60)
    assert r.model.state == "resting" and not r.present()
    bus.publish("percept.utterance", SimpleNamespace(addressed=True, wake_word=False), clock.t)
    assert r.present()
    run_for(r, clock, cfg.circadian.presence_window_s + 20)
    assert r.model.state == "resting"
    bus.publish("percept.utterance", SimpleNamespace(addressed=True, wake_word=True), clock.t)
    run_for(r, clock, 20)
    assert r.model.state == "awake"


def test_unaddressed_speech_neither_shows_presence_nor_wakes(cfg):
    from types import SimpleNamespace

    bus = Bus()
    clock = FakeClock()
    clock.set(0.0, 23.0)
    r = CircadianRunner(cfg, bus, RecordingSink(), clock=clock, body_yaw=None)
    run_for(r, clock, 60)
    tv = SimpleNamespace(addressed=False, wake_word=True)  # even a mis-heard wake word
    bus.publish("percept.utterance", tv, clock.t)
    run_for(r, clock, 20)
    assert not r.present() and r.model.state == "resting"
