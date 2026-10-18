"""SPEC 5A, 4A: Reachy Mini scan planner, sound-direction source, entry point, app wiring."""

from __future__ import annotations

import math
import random
from importlib.metadata import entry_points
from unittest.mock import MagicMock

import pytest
from roboemot.attention import SOURCE as ATTENTION
from roboemot.attention import AttentionSelector, GazeController
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.motion import MotionLoop
from roboemot.world import (
    Observation,
    ScanController,
    WorldModel,
    angular_distance,
    feed_world,
    load_world_plugin,
)
from roboemot.world.scan import SOURCE as SCAN
from roboemot_reachymini import app as app_module
from roboemot_reachymini.gaze import FRAME, GAZE
from roboemot_reachymini.world import DoASource, ReachyScanPlanner, doa_azimuth, plugin

DEG = math.radians(1)
CFG = RoboEmotConfig()
MANIFEST = load_manifest(
    app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
)
FOV = math.radians(CFG.world.camera_fov_h_deg)
HOME = {n: d.home for n, d in MANIFEST.dofs.items()}


def planner(cfg: RoboEmotConfig = CFG) -> ReachyScanPlanner:
    return ReachyScanPlanner(MANIFEST.dofs, MANIFEST.constraints, cfg)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def gap_to_nearest_view(az: float, wps) -> float:
    """Angular distance from ``az`` to the nearest waypoint's camera axis."""
    return min(abs(wrap(az - w.targets[GAZE.head_yaw])) for w in wps)


# ---- planner


def test_full_sweep_covers_the_whole_circle_with_overlap():
    wps = planner().plan(HOME, FOV, amplitude=1.0, dwell_s=1.0)
    half = FOV / 2
    for deg in range(-180, 180):
        assert gap_to_nearest_view(deg * DEG, wps) <= half, deg
    heads = sorted(w.targets[GAZE.head_yaw] for w in wps)
    overlap = math.radians(CFG.world.scan.overlap_deg)
    assert all(b - a <= FOV - overlap + 1e-9 for a, b in zip(heads, heads[1:], strict=False))
    wrap_gap = 2 * math.pi - (heads[-1] - heads[0])  # across the rear
    assert wrap_gap <= FOV - overlap + 1e-9


def test_waypoints_stay_inside_manifest_limits_and_margin():
    margin = math.radians(CFG.safety.constraint_margin_deg)
    reach = math.radians(65) - margin
    for w in planner().plan(HOME, FOV, amplitude=1.0, dwell_s=1.0):
        head, body = w.targets[GAZE.head_yaw], w.targets[GAZE.body_yaw]
        assert MANIFEST.dofs[GAZE.head_yaw].min <= head <= MANIFEST.dofs[GAZE.head_yaw].max
        assert MANIFEST.dofs[GAZE.body_yaw].min <= body <= MANIFEST.dofs[GAZE.body_yaw].max
        assert abs(head - body) <= reach + 1e-9
        assert set(w.targets) == {GAZE.head_yaw, GAZE.body_yaw}  # pitch stays with idle


def test_the_head_reaches_every_waypoint_bearing():
    for w in planner().plan(HOME, FOV, amplitude=1.0, dwell_s=1.0):
        assert w.targets[GAZE.head_yaw] == pytest.approx(w.azimuth_rad, abs=1e-9)


def test_path_is_monotonic_and_never_crosses_the_rear_seam():
    wps = planner().plan(HOME, FOV, amplitude=1.0, dwell_s=1.0)
    heads = [w.targets[GAZE.head_yaw] for w in wps]
    steps = [b - a for a, b in zip(heads, heads[1:], strict=False)]
    assert all(s > 0 for s in steps) or all(s < 0 for s in steps)
    assert all(abs(s) < math.pi / 2 for s in steps)  # no jump across the rear
    bodies = [w.targets[GAZE.body_yaw] for w in wps]
    assert max(abs(b - a) for a, b in zip(bodies, bodies[1:], strict=False)) < math.pi / 2


def test_path_starts_at_the_end_nearest_the_head_and_alternates_on_a_tie():
    p = planner()
    left = p.plan({**HOME, GAZE.head_yaw: 2.5}, FOV, amplitude=1.0, dwell_s=1.0)
    right = p.plan({**HOME, GAZE.head_yaw: -2.5}, FOV, amplitude=1.0, dwell_s=1.0)
    assert left[0].azimuth_rad > 0 > right[0].azimuth_rad
    first = p.plan(HOME, FOV, amplitude=1.0, dwell_s=1.0)[0].azimuth_rad
    second = p.plan(HOME, FOV, amplitude=1.0, dwell_s=1.0)[0].azimuth_rad
    assert first * second < 0


def test_amplitude_limits_the_span_around_the_front():
    wps = planner().plan(HOME, FOV, amplitude=0.5, dwell_s=1.0)
    heads = [w.targets[GAZE.head_yaw] for w in wps]
    assert min(heads) == pytest.approx(-math.pi / 2) and max(heads) == pytest.approx(math.pi / 2)
    assert gap_to_nearest_view(0.0, wps) <= FOV / 2


def test_dwell_is_carried_by_every_waypoint():
    assert {w.dwell_s for w in planner().plan(HOME, FOV, amplitude=1.0, dwell_s=1.7)} == {1.7}


def test_a_larger_margin_shrinks_the_head_lead_but_keeps_the_limits():
    cfg = RoboEmotConfig.model_validate({"safety": {"constraint_margin_deg": 20.0}})
    reach = math.radians(65 - 20)
    for w in planner(cfg).plan(HOME, FOV, amplitude=1.0, dwell_s=1.0):
        assert abs(w.targets[GAZE.head_yaw] - w.targets[GAZE.body_yaw]) <= reach + 1e-9


def test_glance_points_the_head_at_the_bearing():
    w = planner().glance(HOME, 2.0, 0.0, dwell_s=1.0)
    assert w.targets[GAZE.head_yaw] == pytest.approx(2.0)
    behind = planner().glance(HOME, math.pi, 0.0, dwell_s=1.0)
    assert behind.targets[GAZE.head_yaw] == pytest.approx(math.pi)
    assert abs(behind.targets[GAZE.body_yaw]) <= math.radians(160)


# ---- sound direction


@pytest.mark.parametrize(
    ("sdk_angle_deg", "azimuth_deg"), [(0, 90), (90, 0), (180, -90), (45, 45), (135, -45)]
)
def test_doa_mapping(sdk_angle_deg, azimuth_deg):
    """SDK: 0 rad is left, pi/2 front, pi right; REP-103 azimuth is positive left."""
    assert doa_azimuth(sdk_angle_deg * DEG) == pytest.approx(azimuth_deg * DEG)


class FakeMini:
    def __init__(self, readings):
        self.readings = list(readings)
        self.media = MagicMock()
        self.media.get_DoA.side_effect = self._next

    def _next(self):
        item = self.readings.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def doa_source(readings, state=None, stamp=12.5):
    mini = FakeMini(readings)
    src = DoASource(mini, CFG.world.scan, lambda: state or {"head.yaw": 0.4}, lambda: stamp)
    got = []
    return src, got, lambda o, js: got.append((o, js))


def test_doa_source_emits_a_bearing_only_unknown_observation():
    src, got, sink = doa_source([(math.pi / 2 + 0.3, True)])
    assert src.poll_once(sink)
    ((obs, js),) = got
    assert obs.kind == "unknown" and obs.source == "doa" and obs.stamp == 12.5
    assert obs.azimuth_rad == pytest.approx(-0.3) and obs.elevation_rad == 0.0
    assert obs.confidence == CFG.world.scan.doa_confidence
    assert js == {"head.yaw": 0.4}


def test_doa_source_ignores_silence_missing_array_and_failures(caplog):
    src, got, sink = doa_source(
        [(1.0, False), None, RuntimeError("usb"), RuntimeError("usb"), (1.0, True)]
    )
    assert [src.poll_once(sink) for _ in range(5)] == [False, False, False, False, True]
    assert caplog.text.count("doa: read: RuntimeError: usb") == 1  # logged once, not per poll


def test_doa_error_log_resets_after_a_good_read(caplog):
    src, got, sink = doa_source([RuntimeError("usb"), (1.0, False), RuntimeError("usb")])
    for _ in range(3):
        src.poll_once(sink)
    assert caplog.text.count("doa: read: RuntimeError: usb") == 2


def test_a_raising_subscriber_is_logged_and_the_thread_survives(caplog):
    mini = MagicMock()
    mini.media.get_DoA.return_value = (math.pi / 2, True)
    cfg = CFG.world.scan.model_copy(update={"doa_poll_hz": 50.0})
    src = DoASource(mini, cfg, lambda: {})
    calls = []

    def sink(o, js):
        calls.append(o)
        raise RuntimeError("subscriber boom")

    src.start(sink)
    for _ in range(300):
        if len(calls) >= 3:
            break
        src._stop.wait(0.01)
    alive = src._thread.is_alive()
    src.stop()
    assert alive and len(calls) >= 3
    assert caplog.text.count("doa: subscriber: RuntimeError: subscriber boom") == 1


def test_start_twice_runs_one_thread():
    mini = MagicMock()
    mini.media.get_DoA.return_value = None
    src = DoASource(mini, CFG.world.scan, lambda: {})
    src.start(lambda o, js: None)
    first = src._thread
    src.start(lambda o, js: None)
    assert src._thread is first
    src.stop()
    assert src._thread is None


def test_stop_keeps_the_handle_while_the_thread_is_stuck_in_the_sdk(monkeypatch):
    import threading

    from roboemot_reachymini import world as world_module

    monkeypatch.setattr(world_module, "JOIN_TIMEOUT_S", 0.05)
    release, entered = threading.Event(), threading.Event()
    mini = MagicMock()

    def blocked():
        entered.set()
        release.wait(5)

    mini.media.get_DoA.side_effect = blocked
    src = DoASource(mini, CFG.world.scan, lambda: {})
    src.start(lambda o, js: None)
    assert entered.wait(2)
    src.stop()
    stuck = src._thread
    assert stuck is not None and stuck.is_alive()
    src.start(lambda o, js: None)  # no second thread beside the stuck one
    assert src._thread is stuck
    release.set()
    src._stop.set()
    stuck.join(2)


def test_mute_ignores_the_microphones_until_the_stamp_passes():
    now = [10.0]
    mini = FakeMini([(1.0, True), (1.0, True)])
    src = DoASource(mini, CFG.world.scan, lambda: {}, lambda: now[0])
    got = []
    src.mute(12.0)
    assert not src.poll_once(lambda o, js: got.append(o))
    now[0] = 12.5
    assert src.poll_once(lambda o, js: got.append(o)) and len(got) == 1


def test_doa_confirms_an_existing_entry_without_creating_one():
    from roboemot.bus import Bus

    bus = Bus()
    world = WorldModel(CFG.world, FRAME, bus)
    feed_world(bus, world)
    world.observe(Observation("object", 0.0, "t", 0.0, 0.0, "cup"))
    (before,) = world.entries
    src, _, _ = doa_source([(math.pi / 2, True)], state={"head.yaw": 0.0}, stamp=1.0)
    src.poll_once(lambda o, js: bus.publish("percept.entity", o, o.stamp, joint_state=js))
    (after,) = world.entries
    assert after.confidence > before.confidence and "doa" in after.sources


def test_doa_source_skips_without_a_joint_state():
    mini = FakeMini([(1.0, True)])

    def no_state():
        raise ConnectionError("lost")

    src = DoASource(mini, CFG.world.scan, no_state)
    assert not src.poll_once(lambda o, js: pytest.fail("must not publish"))


def test_doa_source_thread_starts_and_stops():
    mini = MagicMock()
    mini.media.get_DoA.return_value = (math.pi / 2, True)
    cfg = CFG.world.scan.model_copy(update={"doa_poll_hz": 50.0})
    src = DoASource(mini, cfg, lambda: {})
    got = []
    src.start(lambda o, js: got.append(o))
    for _ in range(200):
        if got:
            break
        src._stop.wait(0.01)
    src.stop()
    assert got and src._thread is None


def test_a_sound_confirms_an_entry_or_asks_for_a_glance_but_never_creates_one():
    from roboemot.bus import Bus

    bus, world = Bus(), None
    world = WorldModel(CFG.world, FRAME, bus)
    pending = []
    feed_world(bus, world, lambda o, r: pending.append(r.unexplained_bearing))
    src, _, _ = doa_source([(math.pi / 2 - 1.0, True)])
    src.poll_once(lambda o, js: bus.publish("percept.entity", o, o.stamp, joint_state=js))
    assert world.entries == []
    # the head was at yaw 0.4: sensor azimuth 1.0 is base azimuth 1.4
    assert pending[0] == (pytest.approx(1.4), pytest.approx(0.0, abs=1e-9))


# ---- entry point


def test_world_entry_point_is_registered_for_the_manifest_id():
    (ep,) = [e for e in entry_points(group="roboemot.world") if e.name == "reachy-mini-wireless"]
    assert ep.load() is plugin


def test_load_world_plugin_builds_planner_and_sound_source():
    robot = MagicMock()
    loaded = load_world_plugin(MANIFEST, CFG, robot, dict)
    assert isinstance(loaded.planner, ReachyScanPlanner)
    assert [s.name for s in loaded.sources] == ["doa"]
    robot.set_automatic_body_yaw.assert_not_called()  # no second driver is built


# ---- app wiring


def rig(*, scan_cfg: dict | None = None, attention: dict | None = None):
    cfg = RoboEmotConfig.model_validate(
        {
            "world": {"scan": {"idle_interval_s": 20.0, "jitter_s": 5.0, **(scan_cfg or {})}},
            "attention": attention or {},
        }
    )
    driver = MockPoseDriver(home=dict(HOME))
    clock = [0.0]
    loop = MotionLoop(MANIFEST, driver, cfg, rate_hz=50, clock=lambda: clock[0])
    loop.set_mode("active")
    world = WorldModel(cfg.world, FRAME)
    selector = AttentionSelector(cfg.attention)
    gaze = GazeController(cfg.attention, GAZE, MANIFEST.dofs, MANIFEST.constraints)
    scan = ScanController(cfg.world.scan, planner(cfg), MANIFEST.dofs, random.Random(4))
    return cfg, loop, driver, clock, world, selector, gaze, scan


def run_tick(cfg, loop, clock, world, selector, gaze, scan, *, engaged_hint=None):
    """One app cycle: attention first, then scan, then a motion tick (as in ``run``)."""
    now = clock[0]
    attending = app_module.attention_tick(now, world, selector, gaze, loop)
    ctx = app_module.scan_context(now, cfg, selector, hour=12.0)
    scanning = app_module.scan_tick(ctx, scan, loop, cfg)
    loop.cycle(now, 0.02)
    clock[0] += 0.02
    return attending, scanning


def test_idle_robot_sweeps_the_whole_circle_and_returns_to_idle():
    cfg, loop, driver, clock, world, selector, gaze, scan = rig()
    seen_yaw, scanned = [], False
    for _ in range(int(120 / 0.02)):
        _, scanning = run_tick(cfg, loop, clock, world, selector, gaze, scan)
        scanned |= scanning
        seen_yaw.append(driver.read_state()["head.yaw"])
        if scanned and not scanning:
            break
    assert scanned and not scan.active
    pose = loop.pose()
    assert abs(pose[GAZE.head_yaw]) < 0.15 and abs(pose[GAZE.body_yaw]) < 0.15  # glided home
    assert max(seen_yaw) - min(seen_yaw) > 5.0  # the full circle, not a front arc
    assert max(abs(y) for y in seen_yaw) > 3.0  # and it looked straight behind
    assert SCAN not in {c.source for c in loop.core.arbiter.active(clock[0])}


def test_a_person_appearing_mid_sweep_interrupts_it_within_one_tick():
    cfg, loop, driver, clock, world, selector, gaze, scan = rig()
    while not scan.active and clock[0] < 60:
        run_tick(cfg, loop, clock, world, selector, gaze, scan)
    assert scan.active
    for _ in range(100):  # well into the sweep
        run_tick(cfg, loop, clock, world, selector, gaze, scan)
    assert scan.active
    sources = {c.source for c in loop.core.arbiter.active(clock[0])}
    assert SCAN in sources
    world.observe(Observation("person", clock[0], "test", math.radians(20), 0.0, "face"))
    attending, scanning = run_tick(cfg, loop, clock, world, selector, gaze, scan)
    assert attending and not scanning and not scan.active
    sources = {c.source for c in loop.core.arbiter.active(clock[0])}
    assert SCAN not in sources and ATTENTION in sources


def test_an_unexplained_sound_makes_a_glance_toward_it():
    cfg, loop, driver, clock, world, selector, gaze, scan = rig()
    scan.policy.note_unexplained(math.radians(100), 0.0, clock[0])
    peak = 0.0
    for _ in range(int(8 / 0.02)):
        run_tick(cfg, loop, clock, world, selector, gaze, scan)
        peak = max(peak, driver.read_state()["head.yaw"])
    assert peak == pytest.approx(math.radians(100), abs=math.radians(5))


def test_scan_tick_failure_is_logged_and_falls_back_to_idle(caplog):
    cfg, loop, driver, clock, world, selector, gaze, scan = rig()
    scan.planner.plan = MagicMock(side_effect=RuntimeError("planner broke"))
    for _ in range(int(40 / 0.02)):  # past the idle interval, so the policy asks for a sweep
        run_tick(cfg, loop, clock, world, selector, gaze, scan)
    assert "scan: RuntimeError: planner broke" in caplog.text
    assert not scan.active
    assert SCAN not in {c.source for c in loop.core.arbiter.active(clock[0])}


@pytest.mark.parametrize(
    ("hour", "quiet"),
    [(2.0, True), (6 + 59 / 60, True), (7.0, False), (22 + 29 / 60, False), (22.5, True)],
)
def test_quiet_hours_boundaries_in_the_app_context(hour, quiet):
    cfg, loop, driver, clock, world, selector, gaze, scan = rig()
    ctx = app_module.scan_context(0.0, cfg, selector, hour=hour)
    assert ctx.quiet_hours is quiet and ctx.asleep is quiet


def test_quiet_hours_do_not_apply_when_the_circadian_model_is_disabled():
    cfg = RoboEmotConfig.model_validate({"circadian": {"enabled": False}})
    selector = AttentionSelector(cfg.attention)
    assert not app_module.scan_context(0.0, cfg, selector, hour=2.0).quiet_hours


def test_resting_circadian_state_stops_scanning_in_the_app_context():
    from roboemot.circadian import CircadianModel

    cfg, loop, driver, clock, world, selector, gaze, scan = rig()
    model = CircadianModel(cfg.circadian)
    model.state = "resting"
    ctx = app_module.scan_context(0.0, cfg, selector, hour=12.0, circadian=model)
    assert ctx.asleep


def test_object_attention_holds_a_due_scan_then_the_sweep_hides_it():
    """SPEC 5A: a moving object target holds a due scan off for at most
    ``attention.object_max_hold_s``, then the sweep runs and hides the attention command."""
    cfg, loop, driver, clock, world, selector, gaze, scan = rig(attention={"motion_min_dps": 0.1})
    scanned, attention_hidden, attended_before = False, True, False
    started_at = None
    for _ in range(int(90 / 0.02)):
        now = clock[0]
        world.observe(Observation("object", now, "t", math.radians(10 + 3 * now % 40), 0.0, "cup"))
        attending, scanning = run_tick(cfg, loop, clock, world, selector, gaze, scan)
        sources = {c.source for c in loop.core.arbiter.active(clock[0])}
        if scanning:
            started_at = clock[0] if not scanned else started_at
            scanned = True
            attention_hidden &= ATTENTION not in sources
        elif not scanned:
            attended_before |= attending
        if scanned and not scanning:
            break
    assert attended_before  # the object really was the attention target before the sweep
    assert scanned and attention_hidden
    assert not selector.engaged
    # "watch, then scan": due after ~20 s, held off for attention.object_max_hold_s (30 s)
    assert started_at >= cfg.attention.object_max_hold_s + cfg.world.scan.idle_interval_s - 5.0


def test_an_interrupted_sweep_resumes_fresh_after_the_target_lost_delay():
    cfg, loop, driver, clock, world, selector, gaze, scan = rig(
        scan_cfg={"target_lost_delay_s": 10.0, "target_lost_jitter_s": 2.0}
    )
    while not scan.active and clock[0] < 60:
        run_tick(cfg, loop, clock, world, selector, gaze, scan)
    for _ in range(50):
        run_tick(cfg, loop, clock, world, selector, gaze, scan)
    interrupted_at = clock[0]
    world.observe(Observation("person", clock[0], "t", 0.3, 0.0, "face"))
    for _ in range(int(8 / 0.02)):  # the person stays: attention holds, nothing sweeps
        world.observe(Observation("person", clock[0], "t", 0.3, 0.0, "face"))
        _, scanning = run_tick(cfg, loop, clock, world, selector, gaze, scan)
        assert not scanning
    left_at = clock[0]
    started = None
    for _ in range(int(120 / 0.02)):
        _, scanning = run_tick(cfg, loop, clock, world, selector, gaze, scan)
        if scanning:
            started = clock[0]
            break
    lost, jitter = cfg.world.scan.target_lost_delay_s, cfg.world.scan.target_lost_jitter_s
    # the delay runs from the tick attention lets go, which is never before the person left
    assert started is not None and started - left_at >= lost - jitter
    assert started - left_at < cfg.world.scan.idle_interval_s - cfg.world.scan.jitter_s
    assert started > interrupted_at


def test_angular_helper_sanity():
    assert angular_distance(0.0, 0.0, math.pi, 0.0) == pytest.approx(math.pi)


def test_doa_source_is_gated_on_every_poll_by_the_speaking_query():
    speaking = [True]
    src, got, sink = doa_source([(1.0, True), (1.0, True)])
    src.speaking = lambda now: speaking[0]
    assert not src.poll_once(sink) and got == []
    assert len(src._mini.readings) == 2  # the microphones were not even read
    speaking[0] = False
    assert src.poll_once(sink) and len(got) == 1
