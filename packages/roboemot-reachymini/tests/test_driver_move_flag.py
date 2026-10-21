"""SPEC 4A, 12A: the Reachy driver reports the daemon's move flag as a cached value, and the
daemon client reads it through the one REST route that exposes ``Backend.is_move_running``."""

from __future__ import annotations

import json
import threading

import httpx
from roboemot.diagnostics import Target
from roboemot_reachymini import ReachyMiniDriver
from roboemot_reachymini.daemon import DaemonClient
from roboemot_reachymini.driver import MOVE_STALE_S
from roboemot_reachymini.platform import ReachyMiniSession
from test_driver import FakeMini


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class Probe:
    """A move flag the test flips; ``fail`` makes the next probes raise. Each probe wakes
    ``probed`` so the test waits for the poller instead of sleeping."""

    def __init__(self) -> None:
        self.running, self.fail = False, False
        self.calls = 0
        self.probed = threading.Event()

    def __call__(self) -> bool:
        self.calls += 1
        self.probed.set()
        if self.fail:
            raise ConnectionError("daemon unreachable")
        return self.running


def settle(probe: Probe) -> None:
    """Wait for one full probe after now."""
    probe.probed.clear()
    assert probe.probed.wait(2.0)
    probe.probed.clear()
    assert probe.probed.wait(2.0)


def test_move_probe_endpoint_reads_the_ignored_answer():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, json.loads(req.content)))
        if len(seen) == 1:
            return httpx.Response(200, json={"status": "ignored", "reason": "move_running"})
        return httpx.Response(200, json={"status": "ok"})

    c = DaemonClient(Target("robot"), transport=httpx.MockTransport(handler))
    assert c.move_running() is True and c.move_running() is False
    assert seen[0] == ("POST", "/api/move/set_target", {})  # an empty target sets nothing


def test_driver_without_a_probe_cannot_tell():
    drv = ReachyMiniDriver(FakeMini())
    assert drv.platform_move_running() is None
    drv.close()


def test_driver_reports_the_polled_flag_and_unknown_when_stale_or_failing():
    probe, clock = Probe(), Clock()
    drv = ReachyMiniDriver(FakeMini(), probe, clock=clock, poll_s=0.01)
    try:
        assert drv.platform_move_running() in (None, False)  # first call starts the poller
        settle(probe)
        assert drv.platform_move_running() is False
        probe.running = True
        settle(probe)
        assert drv.platform_move_running() is True
        clock.t += MOVE_STALE_S + 0.1  # the poller is not keeping up: the answer is old
        probe.probed.clear()
        assert drv.platform_move_running() is None
        clock.t = 0.0
        probe.fail = True
        settle(probe)
        assert drv.platform_move_running() is None  # a failed probe is unknown, never raises
    finally:
        drv.close()


def test_close_stops_the_poller_and_a_driver_never_called_starts_none():
    probe = Probe()
    idle = ReachyMiniDriver(FakeMini(), probe, poll_s=0.01)
    idle.close()
    assert probe.calls == 0  # nothing polls until the loop asks (FakeMini drivers leak nothing)
    drv = ReachyMiniDriver(FakeMini(), probe, poll_s=0.01)
    drv.platform_move_running()
    assert probe.probed.wait(2.0)
    drv.close()
    calls = probe.calls
    probe.probed.clear()
    assert not probe.probed.wait(0.1) and probe.calls == calls
    assert not any(t.name == "reachy-move-flag" for t in threading.enumerate())


def test_session_driver_probes_the_daemon_the_client_is_connected_to():
    mini = FakeMini()
    mini.client.host, mini.client.port = "robot.local", 8123
    drv = ReachyMiniSession(mini).driver()
    assert isinstance(drv, ReachyMiniDriver) and drv._move_probe is not None
    assert drv._move_probe.__self__.base_url == "http://robot.local:8123"  # type: ignore[attr-defined]
    drv.close()
