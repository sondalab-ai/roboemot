"""SPEC 5A Scanning: policy triggers and exclusions, sweep runner, controller, plugin loader."""

from __future__ import annotations

import math
import random
from unittest.mock import patch

import pytest
from roboemot.bus import Bus
from roboemot.config.schema import PersonaConfig, WorldScanConfig
from roboemot.manifest import Dof
from roboemot.motion import MotionCore, Priority
from roboemot.persona import PersonaEffects
from roboemot.persona import effects as persona_effects
from roboemot.world import (
    Observation,
    ScanContext,
    ScanController,
    ScanPolicy,
    ScanWaypoint,
    SweepRunner,
    WorldModel,
    WorldPlugin,
    feed_world,
    load_world_plugin,
)

CFG = WorldScanConfig(
    idle_interval_s=60.0,
    jitter_s=10.0,
    target_lost_delay_s=10.0,
    target_lost_jitter_s=2.0,
    dwell_s=1.0,
    glance_cooldown_s=8.0,
)
DEG = math.radians(1)
DOFS = {
    "head.yaw": Dof("head.yaw", "head", -math.pi, math.pi, 10.0, 100.0),
    "body_yaw": Dof("body_yaw", "body", -2.8, 2.8, 10.0, 100.0),
}


def ctx(now: float, **kw) -> ScanContext:
    return ScanContext(now=now, engaged=kw.pop("engaged", False), **kw)


def first_decision(policy: ScanPolicy, start: float, **kw):
    """Step the policy every second from ``start``; the (time, decision) of the first one."""
    for i in range(0, 400):
        d = policy.update(ctx(start + i, **kw))
        if d is not None:
            return start + i, d
    return None


# ---- triggers


def test_idle_sweep_comes_after_interval_with_jitter():
    times = []
    for seed in range(6):
        t, d = first_decision(ScanPolicy(CFG, random.Random(seed)), 0.0)
        assert d.kind == "sweep" and d.reason == "idle"
        assert 50.0 <= t <= 70.0  # idle_interval_s +- jitter_s
        times.append(t)
    assert len(set(times)) > 1  # the jitter does vary the time


def test_seeded_policy_is_replayable():
    a = first_decision(ScanPolicy(CFG, random.Random(7)), 0.0)
    b = first_decision(ScanPolicy(CFG, random.Random(7)), 0.0)
    assert a == b


def test_no_second_decision_until_finished_then_clock_restarts():
    p = ScanPolicy(CFG, random.Random(1))
    t, _ = first_decision(p, 0.0)
    assert all(p.update(ctx(t + i)) is None for i in range(1, 200))
    p.finished()
    t2, d2 = first_decision(p, t + 200)
    assert d2.reason == "idle" and t2 - (t + 200) >= 50.0


def test_wake_up_from_rest_sweeps_at_once():
    p = ScanPolicy(CFG, random.Random(1))
    for i in range(5):
        assert p.update(ctx(float(i), circadian="resting")) is None
    d = p.update(ctx(5.0, circadian="awake"))
    assert d is not None and d.kind == "sweep" and d.reason == "wakeup"


def test_wake_up_at_the_end_of_quiet_hours():
    p = ScanPolicy(CFG, random.Random(1))
    assert p.update(ctx(0.0, quiet_hours=True)) is None
    d = p.update(ctx(1.0, quiet_hours=False))
    assert d is not None and d.reason == "wakeup"


def test_wake_up_while_engaged_does_not_sweep_afterwards():
    p = ScanPolicy(CFG, random.Random(1))
    p.update(ctx(0.0, circadian="resting"))
    assert p.update(ctx(1.0, engaged=True)) is None
    assert p.update(ctx(2.0)) is None  # the person was the look-around


def test_glance_toward_an_unexplained_bearing():
    p = ScanPolicy(CFG, random.Random(1))
    p.note_unexplained(1.2, 0.1, 5.0)
    d = p.update(ctx(5.5))
    assert d.kind == "glance" and d.reason == "unexplained"
    assert (d.azimuth_rad, d.elevation_rad) == (1.2, 0.1)
    assert not d.engaged


def test_glance_cooldown_and_age():
    p = ScanPolicy(CFG, random.Random(1))
    p.note_unexplained(1.0, 0.0, 0.0)
    assert p.update(ctx(0.1)).kind == "glance"
    p.finished()
    p.note_unexplained(-1.0, 0.0, 1.0)
    assert p.update(ctx(1.5)) is None  # inside glance_cooldown_s
    p.note_unexplained(-1.0, 0.0, 9.0)
    assert p.update(ctx(9.5)).kind == "glance"
    p.finished()
    p.note_unexplained(1.0, 0.0, 20.0)
    assert p.update(ctx(20.0 + CFG.glance_max_age_s + 1.0)) is None  # too old


def test_a_sweep_drops_the_pending_glance():
    p = ScanPolicy(CFG, random.Random(1))
    p.update(ctx(0.0, circadian="resting"))
    p.note_unexplained(1.0, 0.0, 1.0)
    d = p.update(ctx(1.0))
    assert d.kind == "sweep"  # wake-up first; the sweep passes that bearing anyway
    p.finished()
    assert p.update(ctx(2.0)) is None


# ---- never conditions


def test_never_a_sweep_while_engaged():
    p = ScanPolicy(CFG, random.Random(1))
    assert all(p.update(ctx(float(i), engaged=True)) is None for i in range(500))


def engage(p: ScanPolicy, until: int) -> None:
    for i in range(until):
        assert p.update(ctx(float(i), engaged=True)) is None


def test_sweep_starts_soon_after_the_person_target_is_lost():
    times = []
    for seed in range(8):
        p = ScanPolicy(CFG, random.Random(seed))
        engage(p, 100)
        t, d = first_decision(p, 100.0)
        assert d.kind == "sweep" and d.reason == "idle"
        assert 8.0 <= t - 100.0 <= 12.0  # target_lost_delay_s +- target_lost_jitter_s
        times.append(t)
    assert len(set(times)) > 1


def test_lost_delay_scales_with_idle_period_and_is_not_the_idle_interval():
    slow = PersonaEffects(idle_period_x=2.0)
    p = ScanPolicy(CFG, random.Random(3))
    engage(p, 10)
    t, _ = first_decision(p, 10.0, effects=slow)
    assert 16.0 <= t - 10.0 <= 24.0


def test_lost_delay_applies_once_then_the_idle_interval_returns():
    p = ScanPolicy(CFG, random.Random(1))
    engage(p, 10)
    t, _ = first_decision(p, 10.0)
    p.finished()
    t2, _ = first_decision(p, t + 1)
    assert t2 - (t + 1) >= 50.0


def test_person_returning_cancels_the_pending_lost_sweep():
    p = ScanPolicy(CFG, random.Random(1))
    engage(p, 10)
    assert all(p.update(ctx(10.0 + i)) is None for i in range(5))  # waiting, not yet due
    assert p.update(ctx(15.0, engaged=True)) is None
    assert p.update(ctx(30.0, engaged=True)) is None  # would have been due by now
    t, _ = first_decision(p, 31.0)  # the delay restarts from the next loss
    assert 8.0 <= t - 31.0 <= 12.0


def test_speaker_hint_cancels_a_running_lost_sweep():
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    for i in range(10):
        ctrl.tick(ctx(float(i), engaged=True), pose, 1.0)
    out = run_controller(ctrl, 10.0, 14.0, pose)
    first = next(t for t, c in out if c is not None)
    assert first < 25.0  # sweeping well before idle_interval_s
    assert ctrl.active
    assert ctrl.tick(ctx(24.0, engaged=True), pose, 1.0) is None and not ctrl.active


@pytest.mark.parametrize("kw", [{"circadian": "resting"}, {"quiet_hours": True}])
def test_lost_sweep_does_not_run_while_resting_or_in_quiet_hours(kw):
    p = ScanPolicy(CFG, random.Random(1))
    engage(p, 10)
    assert all(p.update(ctx(10.0 + i, **kw)) is None for i in range(300))
    # waking later is a wake-up sweep, and the lost-target delay is not carried over
    t, d = first_decision(p, 400.0)
    assert d.reason == "wakeup" and t == 400.0
    p.finished()
    t2, d2 = first_decision(p, 401.0)
    assert t2 - 401.0 >= 50.0


def test_lost_delay_stays_within_jitter_bounds():
    for seed in range(50):
        p = ScanPolicy(CFG, random.Random(seed))
        engage(p, 5)
        t, _ = first_decision(p, 5.0)
        assert 8.0 <= t - 5.0 <= 12.0


@pytest.mark.parametrize("kw", [{"circadian": "resting"}, {"quiet_hours": True}])
def test_never_while_resting_or_in_quiet_hours(kw):
    p = ScanPolicy(CFG, random.Random(1))
    p.note_unexplained(1.0, 0.0, 0.0)
    assert all(p.update(ctx(float(i), **kw)) is None for i in range(500))
    assert all(p.update(ctx(float(i), engaged=True, **kw)) is None for i in range(500, 600))


def test_drowsy_still_scans():
    assert first_decision(ScanPolicy(CFG, random.Random(1)), 0.0, circadian="drowsy")


def test_one_glance_per_engagement_toward_a_sound():
    p = ScanPolicy(CFG, random.Random(1))
    p.note_unexplained(2.0, 0.0, 0.0)
    d = p.update(ctx(0.1, engaged=True))
    assert d is not None and d.kind == "glance" and d.engaged
    p.finished()
    p.note_unexplained(-2.0, 0.0, 20.0)  # past the cooldown, same engagement
    assert p.update(ctx(20.1, engaged=True)) is None
    p.update(ctx(30.0))  # engagement ends ...
    p.note_unexplained(-2.0, 0.0, 40.0)  # ... and a new one may glance again
    d2 = p.update(ctx(40.0, engaged=True))
    assert d2 is not None and d2.engaged
    assert p.update(ctx(40.0, engaged=True)) is None  # already running: no second decision


def test_a_newer_bearing_set_during_a_glance_decision_is_kept():
    """The pending slot is cleared only if it still holds the bearing that was acted on."""
    p = ScanPolicy(CFG, random.Random(1))
    p.note_unexplained(1.0, 0.0, 0.0)
    seen = p._pending
    p.note_unexplained(-1.0, 0.0, 0.1)  # the source's thread wins the race
    p._drop_pending(seen)
    assert p._pending == (-1.0, 0.0, 0.1)


def test_a_stale_bearing_is_dropped_without_a_glance():
    p = ScanPolicy(CFG, random.Random(1))
    p.note_unexplained(1.0, 0.0, 0.0)
    assert p.update(ctx(CFG.glance_max_age_s + 0.5)) is None
    assert p._pending is None


HOLD = 30.0


def hold_ctx(now: float, key: str | None = "cat-1", **kw) -> ScanContext:
    return ctx(now, object_target=key, object_max_hold_s=HOLD, **kw)


def due_time(p: ScanPolicy, **kw) -> float:
    """When the policy would sweep with no object target (the idle clock is drawn at t=0)."""
    return first_decision(p, 0.0, **kw)[0]


def test_an_object_target_holds_a_due_scan_off_for_less_than_the_max():
    p = ScanPolicy(CFG, random.Random(1))
    due = due_time(ScanPolicy(CFG, random.Random(1)))
    t = 0.0
    while t < due + HOLD - 1.0:
        assert p.update(hold_ctx(t)) is None, t
        t += 1.0


def test_an_object_held_past_the_max_lets_the_sweep_run():
    p = ScanPolicy(CFG, random.Random(1))
    due = due_time(ScanPolicy(CFG, random.Random(1)))
    t, d = 0.0, None
    while d is None and t < due + 3 * HOLD:
        d = p.update(hold_ctx(t))
        t += 1.0
    assert d is not None and d.kind == "sweep"
    assert HOLD <= (t - 1.0) - due <= HOLD + 1.0  # held off about max_hold after it was due


def test_the_hold_timer_survives_a_switch_between_objects_and_resets_on_loss():
    p = ScanPolicy(CFG, random.Random(1))
    due = due_time(ScanPolicy(CFG, random.Random(1)))
    t = 0.0
    while t < due + 20.0:
        p.update(hold_ctx(t, "cat-1"))
        t += 1.0
    assert p.update(hold_ctx(t, "dog-2")) is None  # switched: the same hold carries on
    t += 15.0
    d = p.update(hold_ctx(t, "dog-2"))
    assert d is not None  # 20 + 15 s since due: past the max in total
    p2 = ScanPolicy(CFG, random.Random(1))
    t = 0.0
    while t < due + 20.0:
        p2.update(hold_ctx(t))
        t += 1.0
    # the target is lost: nothing holds a due scan any more, so it runs at once
    assert p2.update(hold_ctx(t, None)) is not None


def test_a_person_blocks_indefinitely_whatever_the_object_hold():
    p = ScanPolicy(CFG, random.Random(1))
    assert all(
        p.update(ctx(float(i), engaged=True, object_target="cat-1", object_max_hold_s=HOLD)) is None
        for i in range(500)
    )


def test_a_wake_up_sweep_and_a_glance_are_held_like_an_idle_sweep():
    p = ScanPolicy(CFG, random.Random(1))
    p.update(ctx(0.0, circadian="resting"))
    assert p.update(hold_ctx(1.0)) is None  # wake-up due, object held
    assert p.update(hold_ctx(1.0 + HOLD)).reason == "wakeup"
    q = ScanPolicy(CFG, random.Random(1))
    q.note_unexplained(1.0, 0.0, 0.0)
    assert q.update(hold_ctx(0.5)) is None  # held glance; it is stale by the time the hold ends
    assert q.update(hold_ctx(0.5 + HOLD)) is None


# ---- energy and minors


def test_energy_scales_pace_amplitude_and_interval():
    low = persona_effects(PersonaConfig(traits={"energy": 0.0}))
    high = persona_effects(PersonaConfig(traits={"energy": 1.0}))
    mid = persona_effects(PersonaConfig())
    t_low, d_low = first_decision(ScanPolicy(CFG, random.Random(3)), 0.0, effects=low)
    t_mid, d_mid = first_decision(ScanPolicy(CFG, random.Random(3)), 0.0, effects=mid)
    t_high, d_high = first_decision(ScanPolicy(CFG, random.Random(3)), 0.0, effects=high)
    assert t_low > t_mid > t_high  # a calm robot looks around less often
    assert d_low.pace_x < d_mid.pace_x == 1.0 < d_high.pace_x
    assert d_low.dwell_s > d_mid.dwell_s > d_high.dwell_s
    assert d_low.amplitude == pytest.approx(low.motion_amplitude_x)
    assert d_mid.amplitude == 1.0  # default energy covers the whole circle
    assert d_high.amplitude == 1.0  # capped: never more than the full circle


def test_amplitude_has_a_floor():
    tiny = PersonaEffects(motion_amplitude_x=0.01)
    _, d = first_decision(ScanPolicy(CFG, random.Random(1)), 0.0, effects=tiny)
    assert d.amplitude == CFG.min_amplitude


def test_minor_present_slows_the_pace():
    _, d = first_decision(ScanPolicy(CFG, random.Random(1)), 0.0, minor_factor=0.6)
    assert d.pace_x == pytest.approx(0.6)
    assert d.dwell_s == pytest.approx(CFG.dwell_s / 0.6)


# ---- runner


def wp(az: float, yaw: float, dwell: float = 1.0) -> ScanWaypoint:
    return ScanWaypoint(az, {"head.yaw": yaw, "body_yaw": 0.0}, dwell)


def drive(runner: SweepRunner, t0: float, pose: dict, seconds: float, dt: float = 0.02):
    """Run the runner with a perfect follower (the pose is the last command)."""
    cmds, t = [], t0
    while t < t0 + seconds and runner.active:
        cmd = runner.step(t, pose)
        if cmd is None:
            break
        pose.update(cmd.targets)
        cmds.append((t, cmd))
        t += dt
    return cmds, t


def test_runner_visits_waypoints_in_order_and_dwells():
    r = SweepRunner(DOFS, CFG)
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    r.start(0.0, pose, [wp(0.5, 0.5), wp(1.0, 1.0)])
    cmds, t = drive(r, 0.0, pose, 30.0)
    assert not r.active
    yaws = [c.targets["head.yaw"] for _, c in cmds]
    assert yaws == sorted(yaws) and yaws[0] < 0.5 and yaws[-1] == 1.0
    # time at the first waypoint (goal reached) is at least the dwell
    at_first = [tt for tt, c in cmds if c.targets["head.yaw"] == 0.5]
    assert at_first[-1] - at_first[0] >= 1.0 - 0.05
    slew = math.radians(CFG.speed_dps)
    assert t >= 1.0 / slew + 2.0  # travel at the configured speed plus two dwells


def test_runner_slew_rate_follows_pace():
    fast, slow = SweepRunner(DOFS, CFG), SweepRunner(DOFS, CFG)
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    fast.start(0.0, dict(pose), [wp(1.0, 1.0)], pace_x=1.0)
    slow.start(0.0, dict(pose), [wp(1.0, 1.0)], pace_x=0.5)
    for r in (fast, slow):
        r.step(0.0, pose)
    f, s = fast.step(0.2, pose), slow.step(0.2, pose)
    assert f.targets["head.yaw"] == pytest.approx(2 * s.targets["head.yaw"])


def test_runner_waits_for_the_pose_then_gives_up_on_timeout():
    cfg = CFG.model_copy(update={"waypoint_timeout_s": 3.0})
    r = SweepRunner(DOFS, cfg)
    stuck = {"head.yaw": 0.0, "body_yaw": 0.0}  # the robot never moves
    r.start(0.0, stuck, [wp(0.5, 0.5), wp(1.0, 1.0)])
    t = 0.0
    while t < 3.5:  # slew done after ~0.7 s, but the pose never arrives: no dwell countdown
        r.step(t, stuck)
        t += 0.05
    assert r._i == 0
    while t < 4.5:
        r.step(t, stuck)
        t += 0.05
    assert r._i == 1  # timed out


def test_runner_judges_arrival_against_the_minor_cap():
    r = SweepRunner(DOFS, CFG)
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    r.start(0.0, pose, [wp(1.0, 1.0, dwell=0.0)], cap=0.5)
    t = 0.0
    while r.active and t < 20.0:
        cmd = r.step(t, pose)
        if cmd:
            pose["head.yaw"] = 0.5 * cmd.targets["head.yaw"]  # the arbiter halves the amplitude
        t += 0.05
    assert not r.active and t < 10.0  # arrived, no timeout


def test_cancel_stops_at_once():
    r = SweepRunner(DOFS, CFG)
    r.start(0.0, {}, [wp(1.0, 1.0)])
    r.cancel()
    assert not r.active and r.step(0.1, {}) is None


def test_low_pace_and_long_dwell_are_not_cut_by_the_waypoint_timeout():
    cfg = CFG.model_copy(update={"waypoint_timeout_s": 3.0})
    r = SweepRunner(DOFS, cfg)
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    r.start(0.0, pose, [wp(0.5, 0.5, dwell=6.0), wp(1.0, 1.0, dwell=6.0)], pace_x=0.2)
    cmds, t = drive(r, 0.0, pose, 200.0)
    assert not r.active
    first = [tt for tt, c in cmds if c.targets["head.yaw"] == 0.5]
    assert first[-1] - first[0] >= 6.0 - 0.05  # the full dwell, though longer than the timeout
    assert cmds[-1][1].targets["head.yaw"] == 1.0  # the travel (> 3 s at this pace) finished


# ---- controller


class FakePlanner:
    def __init__(self) -> None:
        self.plans: list[tuple] = []

    def plan(self, pose, fov_h_rad, *, amplitude, dwell_s):
        self.plans.append((amplitude, dwell_s, fov_h_rad))
        return [wp(0.3, 0.3, dwell_s), wp(-0.3, -0.3, dwell_s)]

    def glance(self, pose, azimuth_rad, elevation_rad, *, dwell_s):
        return wp(azimuth_rad, azimuth_rad, dwell_s)

    def apply_config(self, cfg) -> None:
        pass


def run_controller(ctrl: ScanController, t0: float, seconds: float, pose: dict, **kw):
    t, out = t0, []
    while t < t0 + seconds:
        cmd = ctrl.tick(ctx(t, **kw), pose, 1.0)
        if cmd is not None:
            pose.update(cmd.targets)
        out.append((t, cmd))
        t += 0.05
    return out


def test_controller_sweeps_when_idle_then_goes_quiet():
    planner = FakePlanner()
    ctrl = ScanController(CFG, planner, DOFS, random.Random(2))
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    out = run_controller(ctrl, 0.0, 200.0, pose)
    sweeping = [t for t, c in out if c is not None]
    assert sweeping and sweeping[0] >= 50.0
    assert all(c.priority == Priority.IDLE and c.source == "scan" for _, c in out if c)
    assert planner.plans and planner.plans[0][0] == 1.0


def test_sweep_ends_with_a_glide_home_at_scan_pace():
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    out = run_controller(ctrl, 0.0, 120.0, pose)
    first = next(i for i, (_, c) in enumerate(out) if c is not None)
    end = next(i for i in range(first, len(out)) if out[i][1] is None)
    cmds = [c for _, c in out[first:end]]  # the first sweep only
    one_step = math.radians(CFG.speed_dps) * 0.05
    assert abs(cmds[-1].targets["head.yaw"]) <= one_step  # glided home, not cut off midway
    yaws = [c.targets["head.yaw"] for c in cmds]
    step = max(abs(b - a) for a, b in zip(yaws, yaws[1:], strict=False))
    assert step <= math.radians(CFG.speed_dps) * 0.25 + 1e-9  # never a snap


def test_interrupted_sweep_waits_the_lost_delay_and_restarts_from_the_pose():
    planner = FakePlanner()
    ctrl = ScanController(CFG, planner, DOFS, random.Random(2))
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    run_controller(ctrl, 0.0, 70.0, pose)
    assert ctrl.active and len(planner.plans) == 1
    pose = {"head.yaw": 0.9, "body_yaw": 0.2}
    assert ctrl.tick(ctx(70.0, engaged=True), pose, 1.0) is None and not ctrl.active
    out = run_controller(ctrl, 70.05, 40.0, pose, engaged=True)
    assert all(c is None for _, c in out)
    out = run_controller(ctrl, 110.0, 20.0, dict(pose))
    first = next(t for t, c in out if c is not None)
    assert CFG.target_lost_delay_s - CFG.target_lost_jitter_s <= first - 110.0 <= 12.0  # lost delay
    assert len(planner.plans) == 2


def test_person_arriving_mid_sweep_cancels_it_after_an_object_hold_expired():
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    t, started = 0.0, None
    while started is None and t < 300.0:
        cmd = ctrl.tick(hold_ctx(t), pose, 1.0)
        if cmd is not None:
            started, _ = t, pose.update(cmd.targets)
        t += 0.05
    assert started is not None and started >= HOLD  # the sweep ran only after the hold
    assert ctrl.tick(hold_ctx(t, engaged=True, key=None), pose, 1.0) is None
    assert not ctrl.active


def test_controller_cancels_a_sweep_the_tick_a_person_engages():
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    out = run_controller(ctrl, 0.0, 70.0, pose)
    assert ctrl.active
    cmd = ctrl.tick(ctx(70.0, engaged=True), pose, 1.0)
    assert cmd is None and not ctrl.active
    assert out[-1][1] is not None  # it was commanding up to the tick before


def test_controller_glance_while_engaged_shows_over_attention_priority():
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    ctrl.policy.note_unexplained(1.5, 0.0, 0.0)
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    out = run_controller(ctrl, 0.0, 10.0, pose, engaged=True)
    cmds = [c for _, c in out if c]
    assert cmds and all(c.priority == Priority.EXPRESSION for c in cmds)
    assert cmds[-1].targets["head.yaw"] == pytest.approx(1.5)
    assert out[-1][1] is None  # the glance ended on its own
    assert not ctrl.active


def test_engaged_glance_expires_and_attention_resumes():
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    ctrl.policy.note_unexplained(1.5, 0.0, 0.0)
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    out = run_controller(ctrl, 0.0, 15.0, pose, engaged=True)
    last = max(t for t, c in out if c is not None)
    assert last < 12.0  # shorter than the run: it ended by itself
    assert all(c is None for t, c in out if t > last)
    assert not ctrl.active
    assert ctrl.policy.update(ctx(20.0, engaged=True)) is None  # no repeat in this engagement


def test_controller_stops_when_the_robot_goes_to_rest():
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    run_controller(ctrl, 0.0, 70.0, pose)
    assert ctrl.active
    assert ctrl.tick(ctx(70.0, circadian="resting"), pose, 1.0) is None


def test_sweep_stays_inside_the_minor_cap_in_the_motion_core():
    """With a minor present the arbiter scales the amplitude around home; the sweep follows."""
    core = MotionCore(
        DOFS,
        [],
        crossfade_s=0.0,
        watchdog_s=100.0,
        minor_factor=0.5,
        watchdog_return_factor=1.0,
    )
    core.minor_present = True
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    t, peak = 0.0, 0.0
    pose = dict(core.limiter.pos)
    while t < 120.0:
        cmd = ctrl.tick(ctx(t, minor_factor=0.5), pose, 1.0)
        if cmd is not None:
            core.feed(cmd)
        pose = core.tick(t, 0.05)
        peak = max(peak, abs(pose["head.yaw"]))
        t += 0.05
    assert 0.1 < peak <= 0.5 * 0.3 + 1e-9  # waypoints are +-0.3 rad, halved by the cap


def test_controller_reset_drops_the_sweep():
    ctrl = ScanController(CFG, FakePlanner(), DOFS, random.Random(2))
    pose = {"head.yaw": 0.0, "body_yaw": 0.0}
    run_controller(ctrl, 0.0, 70.0, pose)
    ctrl.reset()
    assert not ctrl.active


# ---- world feed and plugin loading


def test_feed_world_reports_unexplained_bearings():
    from roboemot.config.schema import WorldConfig

    bus, seen = Bus(), []
    world = WorldModel(WorldConfig(), None, bus)
    feed_world(bus, world, lambda obs, res: seen.append(res.unexplained_bearing))
    bus.publish("percept.entity", Observation("unknown", 1.0, "doa", 0.8, 0.0), 1.0)
    assert seen == [(pytest.approx(0.8), pytest.approx(0.0))]


def test_load_world_plugin_uses_the_entry_point_of_the_robot_id():
    from types import SimpleNamespace

    manifest = SimpleNamespace(robot=SimpleNamespace(id="acme"))
    expected = WorldPlugin()

    class EP:
        name = "acme"

        @staticmethod
        def load():
            return lambda m, c, r, js: expected if r == "handle" and js is dict else None

    with patch("roboemot.world.sources.entry_points", return_value=[EP]):
        assert load_world_plugin(manifest, None, "handle", dict) is expected
        other = SimpleNamespace(robot=SimpleNamespace(id="other"))
        assert load_world_plugin(other, None, "handle", dict) == WorldPlugin()


def test_a_broken_world_plugin_is_logged_not_raised(caplog):
    from types import SimpleNamespace

    class EP:
        name = "acme"

        @staticmethod
        def load():
            raise ImportError("no module")

    manifest = SimpleNamespace(robot=SimpleNamespace(id="acme"))
    with patch("roboemot.world.sources.entry_points", return_value=[EP]):
        assert load_world_plugin(manifest, None, None, dict) == WorldPlugin()
    assert "ImportError: no module" in caplog.text


def test_scan_config_rejects_jitter_beyond_the_interval():
    with pytest.raises(ValueError):
        WorldScanConfig(idle_interval_s=10.0, jitter_s=10.0)


def test_target_lost_jitter_may_not_exceed_the_delay():
    WorldScanConfig(target_lost_delay_s=3.0, target_lost_jitter_s=3.0)
    with pytest.raises(ValueError, match="target_lost_jitter_s"):
        WorldScanConfig(target_lost_delay_s=3.0, target_lost_jitter_s=3.5)
