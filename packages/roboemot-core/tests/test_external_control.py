"""The motion loop yields when something else controls the robot (SPEC 4A, 12A): the measured
pose stops following the commands (the platform runs a move of its own and drops our targets, or
a hand moves the head). No write while that lasts and during the grace after the pose is still;
a motors-off seen then is never written over."""

from __future__ import annotations

import bisect
import math
import random

import pytest
from roboemot.config import ConfigStack
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import Constraint, Dof, Manifest, ManifestModel, RobotInfo
from roboemot.motion import Command, MotionLoop, Priority

DOFS = {
    "head.pitch": Dof("head.pitch", "head", -0.7, 0.7, 2.0, 15.0),
    "head.yaw": Dof("head.yaw", "head", -math.pi, math.pi, 3.0, 20.0),
    "head.z": Dof("head.z", "head", -0.05, 0.05, 0.2, 2.0, kind="translation"),
}
CFG = ConfigStack().effective()
GRACE_S = CFG.motion.platform_move_grace_s
DT = 0.02
TAU_S = 0.1  # the simulated servos' first-order tracking lag
NOW = [0.0]


def manifest() -> Manifest:
    model = ManifestModel(robot=RobotInfo(id="test"))
    return Manifest(
        model, DOFS, [Constraint(type="yaw_delta", a="head.yaw", b="head.pitch", max_deg=180)]
    )


class ServoDriver(MockPoseDriver):
    """A robot whose measured pose lags the last accepted target by ``TAU_S``. While ``external``
    is set the platform owns it: targets are dropped and the pose follows ``goal`` at ``speed``."""

    external = False
    goal: dict[str, float] = {}
    speed = 1.0

    def setup(self) -> None:
        self.pose = dict(self.home)
        self.target = dict(self.home)
        self.accepted: list[float] = []

    def write_pose_targets(self, t, values):
        super().write_pose_targets(t, values)
        if not self.external:
            self.target.update(values)  # a group not written keeps its target
            self.accepted.append(t)

    jitter = 0.0  # +- one encoder quantum on every reading
    stale_ticks = 0  # a state cache that refreshes only every this many readings
    reads = 0
    cache: dict[str, float] | None = None

    def read_state(self):
        self.reads += 1
        base = self.pose
        if self.stale_ticks:
            if self.cache is None or self.reads % self.stale_ticks == 0:
                self.cache = dict(self.pose)
            base = self.cache
        sign = 1 if self.reads % 2 else -1
        return {k: v + sign * self.jitter for k, v in base.items()}

    def advance(self, dt: float) -> None:
        for k in self.pose:
            if self.external:
                step = self.speed * dt
                d = self.goal.get(k, self.pose[k]) - self.pose[k]
                self.pose[k] += max(-step, min(step, d))
                self.target[k] = self.pose[k]  # the platform leaves the targets where it ended
            else:
                self.pose[k] += (self.target[k] - self.pose[k]) * (1 - math.exp(-dt / TAU_S))

    def motors_enabled(self):
        return self.mode == "active"


def make() -> tuple[MotionLoop, ServoDriver]:
    NOW[0] = 0.0
    driver = ServoDriver(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    driver.setup()
    loop = MotionLoop(manifest(), driver, CFG, rate_hz=50, clock=lambda: NOW[0])
    loop.set_mode("active")
    return loop, driver


def drive(loop, driver, t, seconds, yaw=0.3):
    for _ in range(round(seconds / DT)):
        loop.feed(Command("app", Priority.COMMANDED, {"head.yaw": yaw}, t))
        NOW[0] = t
        loop.cycle(t, DT)
        driver.advance(DT)
        t += DT
    NOW[0] = t
    return t


def test_default_grace_outlasts_the_platforms_two_second_wait_before_motors_off():
    assert GRACE_S > 2.0


def test_our_own_lagging_motion_never_triggers_it():
    loop, driver = make()
    t = 0.0
    for yaw in (0.3, -0.5, 1.2, -1.0, 0.0):  # hard reversals at the limiter's speed limit
        t = drive(loop, driver, t, 1.5, yaw=yaw)
    assert not loop.external and loop.external_events == 0
    assert len(driver.accepted) == len(driver.writes)


def test_a_platform_move_is_noticed_and_nothing_is_written_during_it_or_the_grace():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal, driver.speed = True, {"head.pitch": 0.5, "head.yaw": -1.0}, 1.0
    t = drive(loop, driver, t, 0.5)  # a fraction of a second: the confirmation
    assert loop.external and loop.snapshot()["external"] is True
    n = len(driver.writes)
    t = drive(loop, driver, t, 1.6)  # the move runs on, then the pose sits still
    assert driver.pose["head.yaw"] == pytest.approx(-1.0)
    t = drive(loop, driver, t, GRACE_S - 1.5)
    assert len(driver.writes) == n and loop.external
    driver.external = False  # the platform lets go; the pose has been still for the grace
    t = drive(loop, driver, t, 1.5)
    assert not loop.external and len(driver.writes) > n


def test_motors_off_in_the_grace_means_asleep_with_zero_writes():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal = True, {"head.pitch": 0.5, "head.yaw": -1.0}
    t = drive(loop, driver, t, 2.5)
    n = len(driver.writes)
    driver.mode = "limp"  # the platform disables the motors 2 s after the move
    t = drive(loop, driver, t, 0.2)
    assert loop.motors_off and loop.snapshot()["asleep"] is True and not loop.external
    t = drive(loop, driver, t, GRACE_S + 3.0)
    assert len(driver.writes) == n
    driver.mode, driver.external = "active", False  # woken
    drive(loop, driver, t, 3.0)
    assert len(driver.writes) > n and not loop.motors_off


def test_grace_end_with_motors_on_resumes_from_the_measured_pose_without_a_jump():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal = True, {"head.yaw": -0.8}
    t = drive(loop, driver, t, 2.0)
    n = len(driver.writes)
    driver.external = False  # the move is over, motors stay on
    t = drive(loop, driver, t, GRACE_S + 0.5)
    resumed = [v["head.yaw"] for _, v in driver.writes[n:]]
    vmax = DOFS["head.yaw"].max_vel
    assert resumed and resumed[0] == pytest.approx(-0.8, abs=vmax * DT * 2)
    assert max(abs(b - a) for a, b in zip(resumed, resumed[1:], strict=False)) <= vmax * DT + 1e-9
    assert driver.modes == ["active"]  # never turned the motors on


def test_a_hand_pushing_the_head_yields_and_control_returns_after_it_settles():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.pose["head.yaw"] = 0.3 + 0.5  # shoved well past the tolerance, held
    driver.target = dict(driver.pose)
    driver.external, driver.goal = True, {"head.yaw": 0.8}
    t = drive(loop, driver, t, 0.6)
    assert loop.external
    driver.external = False
    t = drive(loop, driver, t, GRACE_S + 0.5)
    assert not loop.external


def test_a_move_that_is_seen_again_in_the_grace_keeps_it_going():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal = True, {"head.pitch": 0.3}
    t = drive(loop, driver, t, 1.0)
    t = drive(loop, driver, t, GRACE_S - 1.0)  # nearly out of the grace
    driver.goal = {"head.pitch": -0.3}  # a second move (go-to-sleep is two)
    n = len(driver.writes)
    t = drive(loop, driver, t, 1.5)
    assert loop.external and len(driver.writes) == n


def _move_then_settle(loop, driver, t):
    t = drive(loop, driver, t, 1.0)
    driver.external, driver.goal = True, {"head.yaw": -0.8}
    t = drive(loop, driver, t, 2.0)
    assert loop.external
    driver.external = False
    return t


def test_encoder_jitter_does_not_keep_a_settled_robot_from_resuming():
    loop, driver = make()
    driver.jitter = 0.0015  # one quantum, alternating: 0.15 rad/s tick to tick
    t = _move_then_settle(loop, driver, 0.0)
    drive(loop, driver, t, GRACE_S + 1.0)
    assert not loop.external


def test_a_state_cache_slower_than_the_loop_does_not_keep_a_settled_robot_from_resuming():
    loop, driver = make()
    driver.stale_ticks = 7  # the same value for several ticks, then a step
    driver.jitter = 0.0015
    t = _move_then_settle(loop, driver, 0.0)
    drive(loop, driver, t, GRACE_S + 1.0)
    assert not loop.external


def test_a_real_drift_beyond_the_band_restarts_the_count():
    loop, driver = make()
    t = _move_then_settle(loop, driver, 0.0)
    t = drive(loop, driver, t, GRACE_S - 1.0)
    assert loop.external
    driver.pose["head.yaw"] -= 3 * CFG.motion.external.still_band_rad  # pushed again
    driver.target = dict(driver.pose)
    t = drive(loop, driver, t, 2.0)  # the first count would have ended by now
    assert loop.external
    drive(loop, driver, t, GRACE_S)
    assert not loop.external


def _calibrated_after_an_event() -> tuple[MotionLoop, dict]:
    """A loop that saw jitter, then one external move: peaks and a last event are set."""
    loop, driver = make()
    driver.jitter = 0.001
    t = drive(loop, driver, 0.0, 2.0)
    snap = loop.snapshot()
    assert 0.0005 < snap["external_peak_err_rad"] < CFG.motion.external.tolerance_rad
    assert snap["external_peak_err_latest_rad"] >= snap["external_peak_err_rad"]
    assert snap["external_peak_diverged_s"] == 0.0 and snap["external_last_event"] is None
    assert snap["external_peak_err_m"] < 0.01
    driver.jitter = 0.0
    driver.external, driver.goal = True, {"head.yaw": -0.8}
    t = drive(loop, driver, t, 2.0)
    snap = loop.snapshot()
    event = snap["external_last_event"]
    assert event["joint"] == "head.yaw" and event["tolerance"] == CFG.motion.external.tolerance_rad
    assert event["error"] > event["tolerance"]
    assert snap["external_peak_diverged_samples"] >= math.ceil(CFG.motion.external.confirm_s * 50)
    assert snap["external_peak_speed_rad_s"] > 0.5  # the move at 1 rad/s, seen while yielding
    return loop, snap


def test_an_unrelated_reload_keeps_the_peaks_and_the_last_event():
    loop, before = _calibrated_after_an_event()
    unrelated = CFG.model_copy(
        update={"motion": CFG.motion.model_copy(update={"platform_move_grace_s": GRACE_S + 1})}
    )
    loop.apply_config(unrelated)
    loop.apply_config(CFG)
    assert loop.snapshot() == before


def test_a_new_external_tolerance_resets_the_peaks_and_keeps_the_last_event():
    loop, before = _calibrated_after_an_event()
    external = CFG.motion.external.model_copy(update={"tolerance_rad": 0.2})
    loop.apply_config(
        CFG.model_copy(update={"motion": CFG.motion.model_copy(update={"external": external})})
    )
    snap = loop.snapshot()
    assert snap["external_last_event"] == before["external_last_event"] is not None
    assert snap["external_events"] == before["external_events"] == 1
    assert snap["external_peak_err_rad"] == 0.0 and snap["external_peak_err_latest_rad"] == 0.0
    assert snap["external_peak_diverged_s"] == 0.0 and snap["external_peak_speed_rad_s"] == 0.0


def test_the_latest_command_error_exceeds_the_windowed_one_under_servo_lag():
    loop, driver = make()
    t = 0.0
    for yaw in (0.8, -0.8, 0.8):
        t = drive(loop, driver, t, 1.5, yaw=yaw)
    snap = loop.snapshot()
    assert snap["external_peak_err_latest_rad"] > snap["external_peak_err_rad"]
    assert snap["external_events"] == 0


# ---- review R1 (B1): old state, late ticks, repeated samples (SPEC 12A, External control)


class DelayedServo(ServoDriver):
    """A state cache that runs ``age_s`` behind the robot (a backlogged receive thread): every
    reading is the newest sample at least that old. Each ``advance`` is one new sample, and
    ``state_seq`` numbers the sample a reading returns."""

    age_s = 0.0

    def setup(self) -> None:
        super().setup()
        self.clock = 0.0
        self.history: list[tuple[float, dict[str, float]]] = [(0.0, dict(self.pose))]

    def _sample(self) -> tuple[int, dict[str, float]]:
        cut = self.clock - self.age_s + 1e-9
        best = max(0, bisect.bisect_right(self.history, cut, key=lambda h: h[0]) - 1)
        return best, self.history[best][1]

    def read_state(self):
        return dict(self._sample()[1])

    def state_seq(self):
        return self._sample()[0]

    def advance(self, dt: float) -> None:
        super().advance(dt)
        self.clock += dt
        self.history.append((self.clock, dict(self.pose)))


def drive_irregular(loop, driver, t, seconds, yaw, stall_every=None, stall_s=0.0):
    """``drive`` with a stall of ``stall_s`` after every ``stall_every`` ticks (an overloaded CPU):
    the robot moves on during the stall, the loop sees one late tick."""
    for n in range(1, round(seconds / DT) + 1):
        loop.feed(Command("app", Priority.COMMANDED, {"head.yaw": yaw}, t))
        NOW[0] = t
        loop.cycle(t, DT)
        driver.advance(DT)
        t += DT
        if stall_every and n % stall_every == 0:
            driver.advance(stall_s)
            t += stall_s
    NOW[0] = t
    return t


def make_with(driver: ServoDriver, man: Manifest | None = None, cfg=CFG) -> MotionLoop:
    NOW[0] = 0.0
    driver.setup()
    loop = MotionLoop(man or manifest(), driver, cfg, rate_hz=50, clock=lambda: NOW[0])
    loop.set_mode("active")
    return loop


def solo(*groups: str):
    """The config with ``motion.external.solo_groups`` set: those groups may yield alone."""
    ext = CFG.motion.external.model_copy(update={"solo_groups": list(groups)})
    return CFG.model_copy(update={"motion": CFG.motion.model_copy(update={"external": ext})})


@pytest.mark.parametrize("age_s", [0.35, 0.38, 0.5])
@pytest.mark.parametrize("stalls", [None, (7, 0.21), (13, 0.21)])
def test_old_state_through_hard_reversals_never_triggers_it(age_s, stalls):
    """Review R1 B1: the reversal sequence with a state 0.35-0.5 s old, with and without 0.21 s
    stalls, yielded 1-3 times; a lagging robot stays behind on the commanded path."""
    driver = DelayedServo(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    driver.age_s = age_s
    loop = make_with(driver)
    every, stall = stalls or (None, 0.0)
    t = 0.0
    for yaw in (0.3, -0.5, 1.2, -1.0, 0.0):
        t = drive_irregular(loop, driver, t, 1.5, yaw, every, stall)
    assert loop.external_events == 0 and not loop.external


def test_a_stall_does_not_bridge_the_confirmation():
    """Review R1 B1: one divergent tick, a stall of ``confirm_s``, one more divergent tick used
    to confirm on two samples. A late tick is no evidence: the count starts again."""
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.pose["head.yaw"] += 0.6  # shoved and held
    driver.target = dict(driver.pose)
    driver.external, driver.goal = True, {"head.yaw": driver.pose["head.yaw"]}
    t = drive(loop, driver, t, DT)
    t += CFG.motion.external.confirm_s  # the loop thread stalls
    drive(loop, driver, t, DT)
    assert not loop.external


def test_overrun_ticks_during_fast_gaze_moves_never_trigger_it():
    driver = DelayedServo(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    loop = make_with(driver)
    t = 0.0
    for yaw in (1.2, -1.2, 1.2, -1.2):
        t = drive_irregular(loop, driver, t, 1.0, yaw, 6, DT * 1.5)
    assert loop.external_events == 0


class SlowCache(ServoDriver):
    """A state cache that takes a new sample every fifth tick, and says so in ``state_seq``."""

    every = 5

    def setup(self) -> None:
        super().setup()
        self.samples = 0
        self.cache = dict(self.pose)

    def advance(self, dt: float) -> None:
        super().advance(dt)
        self.samples += 1
        if self.samples % self.every == 0:
            self.cache = dict(self.pose)

    def read_state(self):
        return dict(self.cache)

    def state_seq(self):
        return self.samples // self.every


def test_a_repeated_sample_does_not_count_towards_the_confirmation():
    """Only a new sample is evidence (review R1 B1b): a cache that refreshes every fifth tick
    needs five times as many ticks to confirm a held push."""
    driver = SlowCache(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    loop = make_with(driver)
    t = drive(loop, driver, 0.0, 1.0)
    driver.pose["head.yaw"] += 0.6
    driver.target = dict(driver.pose)
    driver.external, driver.goal = True, {"head.yaw": driver.pose["head.yaw"]}
    t = drive(loop, driver, t, 2.5 * CFG.motion.external.confirm_s)
    assert not loop.external
    drive(loop, driver, t, 5 * CFG.motion.external.confirm_s)
    assert loop.external


def test_a_push_ahead_of_the_motion_is_detected():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0, yaw=0.0)
    driver.external, driver.goal, driver.speed = True, {"head.yaw": 1.5}, 3.0
    drive(loop, driver, t, 0.6, yaw=1.0)  # commands go to 1.0, a hand takes the head past it
    assert loop.external


def test_a_push_against_the_motion_is_detected():
    """Behind where the commands came from is never lag: the path started from a still pose."""
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0, yaw=0.0)
    driver.external, driver.goal, driver.speed = True, {"head.yaw": -0.6}, 2.0
    drive(loop, driver, t, 1.0, yaw=1.0)
    assert loop.external


def test_a_pose_behind_moving_commands_counts_once_they_settle():
    """Behind the commands, on the side the path came from, may be lag however far (an old
    state cache): it counts only once the commands hold still."""
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0, yaw=-1.5)
    t = drive(loop, driver, t, 0.3, yaw=1.5)  # on the way up at full speed
    driver.external, driver.goal, driver.speed = True, {"head.yaw": -1.0}, 4.0
    t = drive(loop, driver, t, 0.3, yaw=1.5)  # pulled back, the commands still climbing
    assert not loop.external
    drive(loop, driver, t, 1.5, yaw=1.5)  # the commands settle at 1.5
    assert loop.external


def test_recent_events_keep_the_joint_group_and_tick_gap_and_survive_a_reload():
    loop, driver = make()
    t = 0.0
    for goal in (-0.8, 0.8):
        t = drive(loop, driver, t, 1.0)
        driver.external, driver.goal = True, {"head.yaw": goal}
        t = drive(loop, driver, t, 1.5)
        driver.external = False
        t = drive(loop, driver, t, GRACE_S + 0.5)
        assert not loop.external
    snap = loop.snapshot()
    events = snap["external_recent_events"]
    assert len(events) == 2 and snap["external_last_event"] == events[-1]
    for event in events:
        assert event["joint"] == "head.yaw" and event["group"] == "head"
        assert event["samples"] >= math.ceil(CFG.motion.external.confirm_s * 50)
        assert event["max_gap_s"] == pytest.approx(DT) and event["overruns"] == 0
    loop.apply_config(
        CFG.model_copy(update={"motion": CFG.motion.model_copy(update={"app_tick_hz": 10.0})})
    )
    assert loop.snapshot()["external_recent_events"] == events


# ---- review R1 (S3): an antenna alone yields only the antennas


GROUPED = {
    "head.yaw": Dof("head.yaw", "head", -math.pi, math.pi, 3.0, 20.0),
    "body_yaw": Dof("body_yaw", "body_yaw", -2.8, 2.8, 2.0, 10.0),
    "antenna_left": Dof("antenna_left", "antennas", -3.0, 3.0, 6.0, 40.0),
    "antenna_right": Dof("antenna_right", "antennas", -3.0, 3.0, 6.0, 40.0),
}


def grouped_manifest() -> Manifest:
    model = ManifestModel(robot=RobotInfo(id="test"))
    return Manifest(
        model, GROUPED, [Constraint(type="yaw_delta", a="head.yaw", b="body_yaw", max_deg=65)]
    )


def test_an_antenna_held_by_a_hand_yields_the_antennas_only_when_they_may_yield_alone():
    driver = ServoDriver(home=dict.fromkeys(GROUPED, 0.0))
    loop = make_with(driver, grouped_manifest(), solo("antennas"))
    t = drive(loop, driver, 0.0, 1.0)
    driver.pose["antenna_left"] = 1.0  # pinched and held
    driver.target = dict(driver.pose)
    driver.external, driver.goal = True, {"antenna_left": 1.0}
    held = {k: driver.pose[k] for k in ("head.yaw", "body_yaw")}
    t = drive(loop, driver, t, 0.5)
    assert loop.external and loop.snapshot()["external_groups"] == ["antennas"]
    n = len(driver.writes)
    t = drive(loop, driver, t, 1.0, yaw=0.6)
    after = driver.writes[n:]
    assert after and all("antenna_left" not in v and "antenna_right" not in v for _, v in after)
    assert after[-1][1]["head.yaw"] > held["head.yaw"] + 0.2  # the head keeps moving
    driver.external = False
    t = drive(loop, driver, t, GRACE_S + 0.5, yaw=0.6)
    assert not loop.external
    resumed = [v["antenna_left"] for _, v in driver.writes if "antenna_left" in v][-1]
    assert resumed == pytest.approx(0.0, abs=0.05)  # back to its command, under the limits


def test_head_and_body_yield_together_since_a_constraint_couples_them():
    driver = ServoDriver(home=dict.fromkeys(GROUPED, 0.0))
    loop = make_with(driver, grouped_manifest(), solo("head", "body_yaw"))
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal = True, {"body_yaw": -1.0}
    drive(loop, driver, t, 0.6)
    assert loop.snapshot()["external_groups"] == ["body_yaw+head"]


# ---- review R1 (S4): the motor status is the platform's 1 Hz status


class StatusServo(ServoDriver):
    """Reports the motor mode as the platform's status thread does: refreshed every
    ``status_period_s``, so a change is seen up to that late. Counts the reads."""

    status_period_s = 1.0
    checks = 0

    def setup(self) -> None:
        super().setup()
        self.reported = self.mode == "active"
        self.next_status = 0.0

    def motors_enabled(self):
        self.checks += 1
        if NOW[0] >= self.next_status:
            self.reported = self.mode == "active"
            self.next_status = NOW[0] + self.status_period_s
        return self.reported


def test_motors_off_reported_a_status_period_late_is_still_seen_before_any_write():
    """The platform switches the motors off 2.0 s after its sleep move and its status says so
    up to one period later: the grace must cover both (SPEC 4A, 12A)."""
    driver = StatusServo(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    loop = make_with(driver)
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal, driver.speed = True, {"head.pitch": 0.5, "head.yaw": -1.0}, 1.0
    t = drive(loop, driver, t, 1.0)  # the move reaches its goal now
    n = len(driver.writes)
    t = drive(loop, driver, t, 2.0)  # the platform's wait
    driver.mode = "limp"
    drive(loop, driver, t, GRACE_S + 1.0)
    assert len(driver.writes) == n and loop.motors_off


def test_the_motor_status_is_read_at_its_own_period_while_yielding():
    driver = StatusServo(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    loop = make_with(driver)
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal = True, {"head.yaw": -1.0}
    t = drive(loop, driver, t, 1.0)
    assert loop.external
    before = driver.checks
    drive(loop, driver, t, 2.0)
    assert driver.checks - before <= math.ceil(2.0 / driver.status_period_s) + 1


# ---- review R1 (S5): the watchdog shows in the report


def test_snapshot_reports_the_watchdog():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 0.5)
    for _ in range(round(2 * CFG.safety.watchdog_ms / 1000 / DT)):  # the app stops feeding
        NOW[0] = t
        loop.cycle(t, DT)
        t += DT
    snap = loop.snapshot()
    assert snap["holding"] is True and snap["watchdog_engagements"] == 1


def test_a_unit_that_settles_early_waits_for_the_others_before_resuming():
    """goto_sleep moves antennas and head together: the antennas, at their goal well before
    the head, must not resume (and be pulled out of the sleep pose once the move ends) before
    the motors-off that follows the whole move is seen (SPEC 4A, 12A)."""
    driver = StatusServo(home=dict.fromkeys(GROUPED, 0.0))
    loop = make_with(driver, grouped_manifest())
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.speed = True, 1.0
    driver.goal = {"antenna_left": 0.4, "antenna_right": -0.4, "head.yaw": -2.5}
    t = drive(loop, driver, t, 2.5)  # the antennas arrive after 0.4 s, the head after 2.5 s
    assert loop.snapshot()["external_groups"] == ["antennas", "body_yaw+head"]
    n = len(driver.writes)
    driver.external = False  # the move is over: targets are accepted again
    t = drive(loop, driver, t, 2.0)  # the platform's wait before motors off
    driver.mode = "limp"
    drive(loop, driver, t, GRACE_S + 1.0)
    assert len(driver.writes) == n and loop.motors_off


# ---- review of slice ZE: stalls, old state on step-and-hold, reloads, bounded exemption


def _time_to_yield(loop, driver, t, yaw, every, stall, limit_s=10.0):
    t0, n = t, 0
    while t - t0 < limit_s and not loop.external:
        t = drive_irregular(loop, driver, t, DT, yaw)
        n += 1
        if every and n % every == 0:
            driver.advance(stall)
            t += stall
    return t - t0 if loop.external else None


STALLS = [None, (13, 0.21), (9, 0.21), (7, 0.21), (5, 0.05), (3, 0.05)]


@pytest.mark.parametrize("stalls", STALLS)
def test_a_hand_push_yields_under_the_overload_stall_patterns(stalls):
    """Review ZE B1: a late tick restarting the count made a push undetectable once a stall
    came every 9 ticks or more often; a late sample only does not count."""
    every, stall = stalls or (None, 0.0)
    driver = DelayedServo(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    loop = make_with(driver)
    t = drive_irregular(loop, driver, 0.0, 1.0, 0.3)
    driver.pose["head.yaw"] = 0.7  # pushed and held
    driver.target = dict(driver.pose)
    driver.external, driver.goal = True, {"head.yaw": 0.7}
    took = _time_to_yield(loop, driver, t, 0.3, every, stall)
    assert took is not None and took < 1.0


@pytest.mark.parametrize("stalls", STALLS)
def test_a_platform_move_yields_under_the_overload_stall_patterns(stalls):
    every, stall = stalls or (None, 0.0)
    driver = DelayedServo(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    loop = make_with(driver)
    t = drive_irregular(loop, driver, 0.0, 1.0, 0.3)
    driver.external, driver.speed = True, 1.0
    driver.goal = {"head.yaw": -1.0, "head.pitch": 0.5}
    took = _time_to_yield(loop, driver, t, 0.3, every, stall)
    assert took is not None and took < 1.0  # the first goto of goto_sleep lasts 1 s


def _gaze(seed: int, seconds: float):
    """Random gaze shifts of up to 1.2 rad, each held 0.3-1.5 s (step and hold)."""
    rng = random.Random(seed)
    steps, t = [], 0.0
    while t < seconds:
        t += rng.uniform(0.3, 1.5)
        steps.append((t, rng.uniform(-1.2, 1.2)))

    def at(now: float) -> float:
        i = bisect.bisect_right(steps, now, key=lambda s: s[0])
        return steps[i - 1][1] if i else 0.0

    return at


@pytest.mark.parametrize("age_s", [0.6, 0.67, 0.8])
@pytest.mark.parametrize("seed", [0, 1])
def test_old_state_on_step_and_hold_gaze_never_triggers_it(age_s, seed):
    """Review ZE S1: commands that hold still longer than a window turned the direction rule
    off, and a state 0.6 s old (the daemon queue alone can hold ~0.67 s) still showed the pose
    from before the shift. A sample still closing in on the commands is lag."""
    driver = DelayedServo(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    driver.age_s = age_s
    loop = make_with(driver)
    gaze, t = _gaze(seed, 20.0), 0.0
    for _ in range(round(20.0 / DT)):
        cmd = {"head.yaw": gaze(t), "head.pitch": 0.5 * gaze(t * 1.3)}
        loop.feed(Command("app", Priority.COMMANDED, cmd, t))
        NOW[0] = t
        loop.cycle(t, DT)
        driver.advance(DT)
        t += DT
    assert loop.external_events == 0


@pytest.mark.parametrize("age_s", [0.0, 0.2, 0.35])
def test_a_reload_in_the_middle_of_a_fast_move_keeps_the_windows(age_s):
    """Review ZE S2: a motion.external.* reload emptied the windows mid-move."""
    driver = DelayedServo(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    driver.age_s = age_s
    loop = make_with(driver)
    t = drive_irregular(loop, driver, 0.0, 1.5, -1.2)
    t = drive_irregular(loop, driver, t, 0.3, 1.2)  # moving fast toward +1.2
    ext = CFG.motion.external.model_copy(update={"tolerance_rad": 0.16, "lag_window_s": 0.5})
    loop.apply_config(
        CFG.model_copy(update={"motion": CFG.motion.model_copy(update={"external": ext})})
    )
    drive_irregular(loop, driver, t, 1.5, 1.2)
    assert loop.external_events == 0


def _held_against(commands, hold: float, seconds: float) -> float | None:
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0, yaw=commands(0.0))
    driver.pose["head.yaw"] = hold
    driver.target = dict(driver.pose)
    driver.external, driver.goal = True, {"head.yaw": hold}
    t0 = t
    while t - t0 < seconds and not loop.external:
        t = drive(loop, driver, t, DT, yaw=commands(t - t0))
    return t - t0 if loop.external else None


@pytest.mark.parametrize(
    "commands,hold",
    [
        (lambda s: math.sin(2 * math.pi * s / 6), 0.9),
        (lambda s: 0.3 * s, -0.4),  # slow tracking of a person, the head held behind
        (lambda s: 0.1 * s, -0.4),
    ],
)
def test_a_head_held_against_moving_commands_is_caught_within_the_bound(commands, hold):
    """Review ZE S3: behind the path was exempt for as long as the commands moved. A joint that
    has not moved for ``max_state_age_s`` is not lagging: the exemption ends."""
    ext = CFG.motion.external
    took = _held_against(commands, hold, 10.0)
    assert took is not None and took <= ext.max_state_age_s + ext.confirm_s + 0.3


class MinJerkPlatform(DelayedServo):
    """goto_sleep as the daemon runs it: a min-jerk goto of ``move_s`` to ``goal`` (the state
    cache ``age_s`` behind), then motors off ``off_after_s`` after it ends, in a status refreshed
    every ``status_period_s`` and delivered ``age_s`` late."""

    move_s = 2.0
    off_after_s = 2.0
    status_period_s = 1.3

    def start(self, goal: dict[str, float]) -> None:
        self.external, self.goal = True, goal
        self.start_t, self.origin = self.clock, dict(self.pose)
        self.off_at = self.clock + self.move_s + self.off_after_s

    def advance(self, dt: float) -> None:
        if self.external:
            self.clock += dt
            u = min(1.0, (self.clock - self.start_t) / self.move_s)
            s = u**3 * (10 - 15 * u + 6 * u**2)
            for k, g in self.goal.items():
                self.pose[k] = self.origin[k] + (g - self.origin[k]) * s
            self.target = dict(self.pose)
            self.history.append((self.clock, dict(self.pose)))
            if u >= 1.0:
                self.external = False  # the move is over: targets are accepted again
            return
        super().advance(dt)

    def motors_enabled(self):
        """The last status published before ``clock - age_s``; the worst phase: one is
        published just before the motors go off, the next a full period later."""
        if not hasattr(self, "off_at"):
            return True
        phase = self.off_at - 1e-3
        seen = self.clock - self.age_s
        published = phase + math.floor((seen - phase) / self.status_period_s) * self.status_period_s
        return published < self.off_at


GRACE_MARGIN_S = 0.5  # what the default must keep in hand: status build time, scheduling


@pytest.mark.parametrize("age_s", [0.0, 0.67])
@pytest.mark.parametrize(
    "goal",
    [
        {"head.pitch": 0.6, "head.yaw": -1.5, "head.z": -0.04},
        {"head.yaw": 0.3 - 0.25},  # a short travel: its min-jerk tail is still for longer
    ],
)
def test_the_resume_grace_covers_a_min_jerk_move_a_slow_status_and_old_state(age_s, goal):
    """Review ZE S4: stillness starts in the min-jerk tail, before the move ends; the daemon's
    status comes every 1 s plus the time to build it, through a queue that may be ~0.67 s old.
    Nothing may be written between the end of the move and the motors-off being seen, with
    ``GRACE_MARGIN_S`` of the default grace to spare."""
    driver = MinJerkPlatform(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    driver.age_s = age_s
    loop = make_with(driver)
    loop.platform_move_grace_s = GRACE_S - GRACE_MARGIN_S
    t = drive(loop, driver, 0.0, 2.0)
    driver.start(goal)
    while driver.external:
        t = drive(loop, driver, t, DT)
    assert loop.external
    n = len(driver.writes)
    drive(loop, driver, t, GRACE_S + 3.0)
    assert len(driver.writes) == n and loop.motors_off


def test_a_hook_feeding_a_yielding_dof_writes_nothing_to_it():
    driver = ServoDriver(home=dict.fromkeys(GROUPED, 0.0))
    loop = make_with(driver, grouped_manifest(), solo("antennas"))
    loop.tick_hooks.append(
        lambda now, core: core.feed(
            Command("reflex", Priority.REFLEX, {"antenna_left": -0.5}, now), liveness=False
        )
    )
    t = drive(loop, driver, 0.0, 1.0)
    driver.pose["antenna_left"] = 1.0
    driver.target = dict(driver.pose)
    driver.external, driver.goal = True, {"antenna_left": 1.0}
    t = drive(loop, driver, t, 0.5)
    assert loop.snapshot()["external_groups"] == ["antennas"]
    n = len(driver.writes)
    drive(loop, driver, t, 1.0)
    assert all("antenna_left" not in v for _, v in driver.writes[n:])


def test_units_that_yielded_apart_resume_apart():
    """An antenna kept moving in a hand does not keep a head pushed later from resuming."""
    driver = ServoDriver(home=dict.fromkeys(GROUPED, 0.0))
    loop = make_with(driver, grouped_manifest(), solo("antennas"))
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.speed = True, 3.0

    def hold(seconds: float, head: float | None) -> float:
        nonlocal t
        for _ in range(round(seconds / DT)):
            goal = {"antenna_left": 1.0 + 0.1 * math.sin(t * 6)}  # wiggles in the hand
            if head is not None:
                goal["head.yaw"] = head
            driver.goal = goal
            t = drive(loop, driver, t, DT)
        return t

    hold(GRACE_S + 1.0, None)
    assert loop.snapshot()["external_groups"] == ["antennas"]
    hold(1.5, -0.8)  # later, a push on the head
    assert loop.snapshot()["external_groups"] == ["antennas", "body_yaw+head"]
    hold(GRACE_S + 1.5, 0.3)  # the head let go, back where it is commanded
    assert loop.snapshot()["external_groups"] == ["antennas"]


# ---- fix round 2: the daemon's goto_sleep moves the antennas first (MuJoCo sim, SDK 1.12)


def test_a_platform_move_seen_first_on_the_antennas_stops_every_write():
    """In the sim, goto_sleep's antennas diverge ~0.6 s before the head: per-unit yielding kept
    streaming head targets meanwhile, and the daemon accepts them in the pause between its two
    gotos and after the last one, pulling the head out of the sleep pose (SPEC 4A, 12A). By
    default any divergence yields every unit: a hand cannot be told from the platform."""
    driver = ServoDriver(home=dict.fromkeys(GROUPED, 0.0))
    loop = make_with(driver, grouped_manifest())
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.speed = True, 3.0
    driver.goal = {"antenna_left": 1.5, "antenna_right": -1.5}  # the antennas go first
    while not loop.external:
        t = drive(loop, driver, t, DT)
    n = len(driver.writes)
    driver.speed, driver.goal = 0.5, {"antenna_left": 1.5, "antenna_right": -1.5, "head.yaw": -1.0}
    t = drive(loop, driver, t, 2.0)
    assert len(driver.writes) == n
    assert loop.snapshot()["external_groups"] == ["antennas", "body_yaw+head"]
    event = loop.snapshot()["external_last_event"]
    assert event["group"] == "antennas"  # the record names the unit that diverged
