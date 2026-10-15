"""SPEC 12A: link hysteresis and backoff, route reachability, robot-link recovery, responder."""

from __future__ import annotations

import math

import httpx
import pytest
from roboemot.affect import AffectEngine
from roboemot.config import ConfigStack
from roboemot.config.schema import LinkPolicy, LLMRoute, SafetyConfig
from roboemot.drivers import MockPoseDriver
from roboemot.health import Link
from roboemot.health.network import NetworkMonitor, route_endpoint
from roboemot.health.responder import OfflineResponder, perk_up, puzzled
from roboemot.llm import Done, NoRouteAvailable, Offline, Router
from roboemot.manifest import Constraint, Dof, Manifest, ManifestModel, RobotInfo
from roboemot.motion import Command, MotionLoop, Priority

CFG = ConfigStack().effective()
POLICY = LinkPolicy(
    failures_to_offline=3,
    successes_to_online=2,
    probe_initial_s=1,
    probe_max_s=8,
    probe_backoff=2,
    probe_jitter=0,
    probe_timeout_s=0.1,
)


class Events(list):
    def __call__(self, event: str, now: float) -> None:
        self.append(event)


# ---------------------------------------------------------------- Link


def test_link_hysteresis_both_ways():
    ev = Events()
    link = Link("x", POLICY, ev)
    for t in range(2):
        link.report(False, t)
    assert link.online and ev == []
    assert link.report(False, 2) and not link.online and ev == ["link.x.lost"]
    link.report(True, 3)
    assert not link.online
    link.report(False, 4)  # a failure resets the success streak
    link.report(True, 5)
    assert not link.online
    assert link.report(True, 6) and link.online and ev == ["link.x.lost", "link.x.restored"]


def test_link_backoff_grows_to_cap_and_recovery_is_confirmed_fast():
    link = Link("x", POLICY)
    delays, now = [], 0.0
    for _ in range(3 + 5):
        link.report(False, now)
        delays.append(link.next_check - now)
        now = link.next_check
    assert delays[:2] == [1, 1]  # still online: re-check a failure soon
    assert delays[2:] == [1, 2, 4, 8, 8, 8]  # offline: backoff, capped at probe_max_s
    link.report(True, now)
    assert link.next_check - now == 1  # one success while offline: confirm quickly


def test_link_checks_slowly_while_healthy_and_jitter_stays_bounded():
    link = Link("x", POLICY)
    link.report(True, 0)
    assert link.next_check == 8
    jittery = Link("y", POLICY.model_copy(update={"probe_jitter": 0.25}))
    for t in range(50):
        jittery.report(True, t)
        assert 6 <= jittery.next_check - t <= 10


def test_link_set_reports_only_changes():
    ev = Events()
    link = Link("x", POLICY, ev)
    assert not link.set(True, 0) and ev == []
    assert link.set(False, 1) and ev == ["link.x.lost"]
    assert not link.would_restore() or POLICY.successes_to_online == 1
    link.report(True, 2)
    assert link.would_restore()


# ---------------------------------------------------------------- network


def route(name: str, url: str = "https://api.example.com/v1") -> LLMRoute:
    return LLMRoute(name=name, base_url=url, model="m", first_token_timeout_s=0.3)


def test_route_endpoint_defaults_ports():
    assert route_endpoint(route("a")) == ("api.example.com", 443)
    assert route_endpoint(route("b", "http://192.168.1.5:11434/v1")) == ("192.168.1.5", 11434)
    assert route_endpoint(route("c", "http://lan.local/v1")) == ("lan.local", 80)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def monitor(reachable: dict[str, bool], ev: Events, clock: Clock) -> NetworkMonitor:
    routes = [route("cloud"), route("lan", "http://lan.local:8080/v1")]
    hosts = {"api.example.com": "cloud", "lan.local": "lan"}
    return NetworkMonitor(
        routes, POLICY, ev, probe=lambda host, port, t: reachable[hosts[host]], clock=clock
    )


def run_checks(mon: NetworkMonitor, clock: Clock, seconds: float) -> None:
    end = clock.t + seconds
    while clock.t <= end:
        mon.check_due()
        clock.t += 0.5


def test_network_lost_only_when_every_route_is_down():
    ev, clock = Events(), Clock()
    up = {"cloud": True, "lan": True}
    mon = monitor(up, ev, clock)
    up["cloud"] = False
    run_checks(mon, clock, 20)
    assert ev == ["link.route.cloud.lost"] and mon.online
    assert not mon.is_online("cloud") and mon.is_online("lan")
    up["lan"] = False
    run_checks(mon, clock, 20)
    assert ev[-2:] == ["link.route.lan.lost", "link.network.lost"] and not mon.online
    up["lan"] = True
    run_checks(mon, clock, 40)  # the backoff can reach probe_max_s before the next check
    assert ev[-2:] == ["link.route.lan.restored", "link.network.restored"] and mon.online


def test_passive_reports_count_like_checks():
    ev, clock = Events(), Clock()
    mon = monitor({"cloud": True, "lan": True}, ev, clock)
    for _ in range(3):
        mon.report("cloud", False)
        mon.report("lan", False)
    assert not mon.online and "link.network.lost" in ev
    mon.report("unknown-route", False)  # ignored


def test_hot_reload_updates_every_link():
    mon = monitor({"cloud": True, "lan": True}, Events(), Clock())
    stricter = POLICY.model_copy(update={"failures_to_offline": 1})
    mon.set_policy(stricter)
    assert all(link.policy is stricter for link in (*mon.links.values(), mon.network))


def test_monitor_without_routes_starts_and_stops():
    mon = NetworkMonitor([], POLICY)
    mon.start()
    mon.stop()
    assert mon.online


# ---------------------------------------------------------------- router


def ok_handler(request: httpx.Request) -> httpx.Response:
    chunk = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "ciao"}, "finish_reason": None}],
    }
    body = f"data: {__import__('json').dumps(chunk)}\n\ndata: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def router_with(mon: NetworkMonitor, handler) -> Router:
    return Router(
        mon.routes,
        SafetyConfig(),
        monitor=mon,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def collect(r: Router):
    return [e async for e in r.stream([{"role": "user", "content": "ciao"}])]


async def test_router_raises_offline_at_once_without_requests():
    calls = []
    mon = monitor({"cloud": True, "lan": True}, Events(), Clock())
    mon.network.set(False, 0)

    def handler(request):
        calls.append(request)
        return ok_handler(request)

    with pytest.raises(Offline):
        await collect(router_with(mon, handler))
    assert calls == [] and issubclass(Offline, NoRouteAvailable)


async def test_router_skips_unreachable_route():
    mon = monitor({"cloud": True, "lan": True}, Events(), Clock())
    for _ in range(3):
        mon.report("cloud", False)
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        return ok_handler(request)

    evs = await collect(router_with(mon, handler))
    assert hosts == ["lan.local"] and isinstance(evs[-1], Done) and evs[-1].route == "lan"


async def test_connection_errors_take_the_network_offline():
    ev = Events()
    mon = monitor({"cloud": True, "lan": True}, ev, Clock())

    def down(request):
        raise httpx.ConnectError("no route to host")

    r = router_with(mon, down)
    for _ in range(2):
        with pytest.raises(NoRouteAvailable):
            await collect(r)
    with pytest.raises(Offline):
        await collect(r)
    assert "link.network.lost" in ev


async def test_slow_but_reachable_route_never_goes_offline():
    """A timeout asks for a TCP check; the check says reachable, so the network stays up."""
    import asyncio

    ev, clock = Events(), Clock()
    mon = monitor({"cloud": True, "lan": True}, ev, clock)
    mon.routes, mon.links = mon.routes[:1], {"cloud": mon.links["cloud"]}

    async def slow(request):
        await asyncio.sleep(1.0)  # past the 0.3 s first-token timeout
        return ok_handler(request)

    r = router_with(mon, slow)
    for _ in range(5):
        with pytest.raises(NoRouteAvailable) as e:
            await collect(r)
        assert not isinstance(e.value, Offline)
        assert mon.links["cloud"].due(clock())  # a check was requested
        mon.check_due()
    assert mon.online and ev == []


async def test_http_errors_mean_reachable():
    mon = monitor({"cloud": True, "lan": True}, Events(), Clock())
    r = router_with(mon, lambda request: httpx.Response(503, json={"error": {"message": "x"}}))
    for _ in range(5):
        with pytest.raises(NoRouteAvailable) as e:
            await collect(r)
        assert not isinstance(e.value, Offline)
    assert mon.online


# ---------------------------------------------------------------- robot link in the motion loop

DOFS = {
    "head.pitch": Dof("head.pitch", "head", -0.7, 0.7, 2.0, 15.0),
    "head.yaw": Dof("head.yaw", "head", -math.pi, math.pi, 3.0, 20.0),
}


def mini_manifest() -> Manifest:
    model = ManifestModel(robot=RobotInfo(id="test"))
    return Manifest(
        model, DOFS, [Constraint(type="yaw_delta", a="head.yaw", b="head.pitch", max_deg=180)]
    )


class FlakyDriver(MockPoseDriver):
    """Fails every call while ``down``; the measured pose can move meanwhile."""

    down: bool = False

    def _check(self) -> None:
        if self.down:
            raise ConnectionError("daemon unreachable")

    def write_pose_targets(self, t, values):
        self._check()
        super().write_pose_targets(t, values)

    def read_state(self):
        self._check()
        return dict(self.home)

    def set_mode(self, mode):
        self._check()
        super().set_mode(mode)


def loop_with(driver: FlakyDriver, ev: Events) -> MotionLoop:
    # Simulated time: tests call cycle() with their own clock; the thread (if any) sees t=0.
    return MotionLoop(
        mini_manifest(), driver, CFG, rate_hz=50, on_link_change=ev, clock=lambda: 0.0
    )


def drive(loop: MotionLoop, t: float, seconds: float, yaw: float = 0.0) -> float:
    for _ in range(round(seconds * 50)):
        loop.feed(Command("app", Priority.COMMANDED, {"head.yaw": yaw}, t))
        loop.cycle(t, 0.02)
        t += 0.02
    return t


def test_robot_link_lost_then_recovered_without_jump():
    ev = Events()
    driver = FlakyDriver(home={"head.pitch": 0.0, "head.yaw": 0.0})
    loop = loop_with(driver, ev)
    loop.set_mode("active")
    t = drive(loop, 0.0, 0.5, yaw=0.3)
    driver.down = True
    t = drive(loop, t, 2.0, yaw=0.3)
    assert ev == ["link.robot.lost"] and not loop.link.online
    writes_while_down = len(driver.writes)
    # moved by hand (or the daemon restarted at home) while the link was down
    driver.home = {"head.pitch": 0.0, "head.yaw": -0.5}
    driver.down = False
    t = drive(loop, t, CFG.connectivity.robot.probe_max_s * 2, yaw=0.3)  # backoff + confirmation
    assert ev == ["link.robot.lost", "link.robot.restored"] and loop.link.online
    assert driver.modes == ["active"]  # recovery never re-enables the motors itself
    resumed = [v["head.yaw"] for _, v in driver.writes[writes_while_down:]]
    vmax = DOFS["head.yaw"].max_vel
    assert resumed[0] == pytest.approx(-0.5, abs=vmax * 0.02 + 1e-9)  # from the real pose
    assert max(abs(b - a) for a, b in zip(resumed, resumed[1:], strict=False)) <= vmax * 0.02 + 1e-9
    assert resumed[-1] == pytest.approx(0.3, abs=1e-6)


def test_robot_unreachable_at_start_waits_for_the_platform_to_enable_motors():
    ev = Events()
    driver = FlakyDriver(home={"head.pitch": 0.0, "head.yaw": 0.0})
    driver.down = True
    loop = loop_with(driver, ev)
    loop.start()  # must not raise
    loop.stop()
    assert ev == ["link.robot.lost"] and loop.mode == "active"
    driver.down = False
    t = drive(loop, 10.0, CFG.connectivity.robot.probe_max_s * 2)
    assert ev[-1] == "link.robot.restored"
    assert driver.modes == [] and driver.writes == []  # never enabled the motors itself
    driver.mode = "active"  # the platform wakes the robot
    drive(loop, t, 1.0)
    assert driver.writes and driver.modes == []


def test_motors_turned_off_by_platform_are_tracked_not_overridden():
    """SPEC 4A, 12A: a restart without wake-up (or the robot put to sleep) must not be undone."""
    ev = Events()
    driver = FlakyDriver(home={"head.pitch": 0.0, "head.yaw": 0.0})
    loop = loop_with(driver, ev)
    loop.set_mode("active")
    t = drive(loop, 0.0, 0.5, yaw=0.3)
    driver.down = True
    t = drive(loop, t, 1.0, yaw=0.3)
    driver.down, driver.mode = False, "limp"  # daemon back, motors off
    driver.home = {"head.pitch": 0.2, "head.yaw": -0.4}  # sleep pose
    n = len(driver.writes)
    t = drive(loop, t, CFG.connectivity.robot.probe_max_s * 2, yaw=0.3)
    assert ev == ["link.robot.lost", "link.robot.restored"]
    assert len(driver.writes) == n and driver.modes == ["active"]  # only the initial enable
    assert loop.motors_off
    driver.mode = "active"  # woken by the platform
    drive(loop, t, 3.0, yaw=0.3)
    resumed = [v["head.yaw"] for _, v in driver.writes[n:]]
    vmax = DOFS["head.yaw"].max_vel
    assert resumed[0] == pytest.approx(-0.4, abs=vmax * 0.02 + 1e-9)  # from the sleep pose
    assert resumed[-1] == pytest.approx(0.3, abs=1e-6) and not loop.motors_off


def test_own_faults_do_not_take_the_robot_link_down():
    ev = Events()
    driver = FlakyDriver(home={"head.pitch": 0.0, "head.yaw": 0.0})
    loop = loop_with(driver, ev)
    loop.set_mode("active")

    def broken(now, dt):
        raise ValueError("bug in the arbiter")

    loop.core.tick = broken
    drive(loop, 0.0, 2.0)
    assert ev == [] and loop.link.online and loop.errors == 100


# ---------------------------------------------------------------- responder

REACHY = {"head.roll", "head.pitch", "head.yaw", "body_yaw", "antenna_right", "antenna_left"}


class Cue:
    def __init__(self) -> None:
        self.said: list[tuple[str, str]] = []

    def say(self, text: str, language: str) -> None:
        self.said.append((text, language))


def responder(dofs=REACHY, language="it") -> tuple[OfflineResponder, Cue, AffectEngine]:
    cue, affect = Cue(), AffectEngine(CFG.affect, CFG.appraisal)
    return OfflineResponder(CFG.connectivity, set(dofs), language, affect, cue), cue, affect


def test_network_lost_is_subdued_and_cued_once_per_cooldown():
    r, cue, affect = responder()
    r.on_event("link.network.lost", 0.0)
    assert affect.state(0.0).arousal < 0 and affect.state(0.0).valence < 0
    assert cue.said == [(CFG.connectivity.offline_phrases["it"], "it")]
    r.speech_while_offline(10.0)  # within cue_cooldown_s: gesture, no second phrase
    assert len(cue.said) == 1 and r.commands(10.1)
    r.speech_while_offline(10.0 + CFG.connectivity.cue_cooldown_s)
    assert len(cue.said) == 2


def test_restored_perks_up_and_gesture_ends():
    r, _, affect = responder()
    r.on_event("link.network.restored", 0.0)
    assert affect.state(0.0).label == "joy"
    mid = r.commands(CFG.connectivity.gestures.duration_s / 2)
    assert len(mid) == 1 and mid[0].additive and mid[0].priority == Priority.EXPRESSION
    assert mid[0].targets["head.pitch"] < 0 and mid[0].targets["antenna_right"] > 0
    assert r.commands(CFG.connectivity.gestures.duration_s + 0.01) == []


def test_speech_while_offline_is_puzzled():
    r, _, affect = responder()
    r.speech_while_offline(0.0)
    assert affect.state(0.0).label == "curiosity"
    cmd = r.commands(CFG.connectivity.gestures.duration_s / 2)[0]
    assert cmd.targets["head.roll"] > 0 and cmd.targets["antenna_left"] > 0


def test_gestures_use_only_the_robots_dofs():
    g = CFG.connectivity.gestures
    assert set(puzzled(g, {"head.roll", "head.pitch"}).keyframes[1].values) == {"head.roll"}
    assert set(perk_up(g, {"neck_pan"}).keyframes[1].values) == set()
    r, _, _ = responder(dofs={"neck_pan", "neck_tilt"})
    r.on_event("link.robot.restored", 0.0)
    assert r.commands(0.5) == []


def test_no_phrase_for_unknown_language_is_silent():
    r, cue, _ = responder(language="fr")
    r.on_event("link.network.lost", 0.0)
    assert cue.said == []


# ---------------------------------------------------------------- snapshots for diagnostics


def test_link_snapshot_reports_counters_and_next_check():
    link = Link("x", POLICY)
    link.report(False, now=10.0)
    snap = link.snapshot(now=10.25)
    assert snap == {
        "online": True,
        "consecutive_failures": 1,
        "consecutive_successes": 0,
        "next_check_in_s": 0.75,
    }


def test_network_snapshot_lists_routes_by_endpoint_without_credentials():
    ev, clock = Events(), Clock()
    mon = monitor({"cloud": False, "lan": False}, ev, clock)
    run_checks(mon, clock, 3)
    snap = mon.snapshot()
    assert snap["online"] is False
    assert snap["routes"]["cloud"]["endpoint"] == "api.example.com:443"
    assert snap["routes"]["lan"]["endpoint"] == "lan.local:8080"
    assert "api_key" not in str(snap)


def test_motion_snapshot_reflects_link_loss():
    ev = Events()
    driver = FlakyDriver(home=dict.fromkeys(DOFS, 0.0))
    loop = loop_with(driver, ev)
    loop.set_mode("active")
    t = drive(loop, 0.0, 0.2)
    driver.down = True
    drive(loop, t, 0.2)
    snap = loop.snapshot()
    assert snap["mode"] == "active" and snap["errors"] >= CFG.connectivity.robot.failures_to_offline
    assert snap["robot_link"]["online"] is False
    assert "ConnectionError" in snap["last_error"]
