"""SPEC 12, 12A: the robot-asleep signal and what pauses with it."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from roboemot.bus import Bus
from roboemot.config import ConfigStack
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.runtime import Runtime, RuntimeHost
from roboemot.runtime.sleep import SleepGate
from test_robot_platform import DeskPlatform, DeskSession

CONFIGS = Path(__file__).resolve().parents[3].parent / "configs"


class Part:
    """A pausable part that logs its calls into the shared ``log``."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name, self.log, self.paused = name, log, False

    def pause(self) -> None:
        self.paused = True
        self.log.append(f"pause {self.name}")

    def resume(self) -> None:
        self.paused = False
        self.log.append(f"resume {self.name}")

    pause_listening, resume_listening = pause, resume

    def reset(self) -> None:
        self.log.append(f"reset {self.name}")

    release = reset

    def mark_interrupting(self) -> None:
        self.log.append("conversation mark")

    def interrupt(self) -> None:
        self.log.append("conversation interrupt")

    def stop(self) -> None:
        self.log.append("speech stop")

    def clear(self, source: str) -> None:
        self.log.append(f"clear {source}")


class Rig:
    def __init__(self, speaking: bool = True, listener: bool = True) -> None:
        self.log: list[str] = []
        self.enabled = True
        self.loop = Part("loop", self.log)
        self.loop.motors_off, self.loop.link = False, SimpleNamespace(online=True)
        self.bus = Bus()
        self.events: list[str] = []
        for topic in ("robot.asleep", "robot.awake"):
            self.bus.subscribe(topic, lambda m: self.events.append(m.topic))
        self.vision = Part("vision", self.log)
        self.light = Part("light", self.log)
        self.listener = Part("listener", self.log) if listener else None
        conv = Part("conv", self.log) if speaking else None
        host = SimpleNamespace(conversation=conv, speech_output=conv)
        self.gate = SleepGate(
            host,
            self.bus,
            self.loop,
            SimpleNamespace(service=self.vision),
            self.light,
            self.listener,
            Part("gaze", self.log),
            Part("scan", self.log),
            lambda: self.enabled,
        )

    def sleep(self, now: float, asleep: bool = True) -> bool:
        self.loop.motors_off = asleep
        return self.gate.tick(now)


def test_asleep_publishes_once_and_pauses_everything_then_wake_resumes():
    r = Rig()
    assert r.gate.tick(0.0) is False and r.events == [] and r.log == []
    assert r.sleep(1.0) is True
    assert r.sleep(1.1) is True  # no second event, no second pause
    assert r.events == ["robot.asleep"]
    assert r.log[:3] == [
        "conversation mark",
        "speech stop",
        "conversation interrupt",
    ]  # as a barge-in
    assert {"pause vision", "pause light", "pause listener"} <= set(r.log)
    assert r.log.count("pause vision") == 1
    snap = r.gate.snapshot(3.0)
    assert snap["asleep"] and snap["asleep_for_s"] == 2.0
    assert snap["paused"] == {
        "app_tick": True,
        "vision": True,
        "light_reader": True,
        "listener": True,
    }
    assert r.sleep(4.0, asleep=False) is False
    assert r.events == ["robot.asleep", "robot.awake"]
    assert {"resume vision", "resume light", "resume listener"} <= set(r.log)
    assert not r.vision.paused and not r.light.paused and not r.listener.paused


def test_the_opt_out_publishes_but_pauses_nothing_and_lifting_it_while_asleep_resumes():
    r = Rig()
    r.enabled = False
    assert r.sleep(1.0) is False
    assert r.events == ["robot.asleep"] and r.log == []
    r.enabled = True
    assert r.gate.tick(2.0) is True and r.vision.paused
    r.enabled = False  # a reload switches it off while asleep
    assert r.gate.tick(3.0) is False and not r.vision.paused and not r.listener.paused
    assert r.events == ["robot.asleep"]


def test_a_lost_robot_link_is_not_asleep():
    r = Rig()
    r.sleep(1.0)
    r.loop.link.online = False
    assert r.gate.tick(2.0) is False and r.events == ["robot.asleep", "robot.awake"]


def test_a_runtime_without_voice_or_vision_still_gates():
    r = Rig(speaking=False, listener=False)
    r.gate._perception = SimpleNamespace(service=None)
    assert r.sleep(1.0) is True
    assert not any(line.startswith("pause vision") for line in r.log)
    paused = r.gate.snapshot(2.0)["paused"]
    assert paused["vision"] == "absent" and paused["listener"] == "absent"
    assert r.sleep(2.0, asleep=False) is False


class SleepyDriver(MockPoseDriver):
    """Motors held off by the platform while ``enabled`` is False."""

    enabled = True

    def motors_enabled(self):
        return self.enabled


class SleepySession(DeskSession):
    built: SleepyDriver | None = None

    def driver(self):
        self.built = SleepyDriver(home={"neck_pan": 0.0, "neck_tilt": 0.0})
        return self.built


class SleepyPlatform(DeskPlatform):
    def bind(self, client):
        self.session = SleepySession(client)
        return self.session


class DaytimeHost(RuntimeHost):
    def make_circadian_clock(self):
        return SimpleNamespace(monotonic=time.monotonic, hour=lambda: 12.0)


def wait_for(predicate, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_the_runtime_idles_while_the_platform_holds_the_motors_off_and_resumes():
    stack = ConfigStack.from_files(preset="default", preset_dirs=[CONFIGS / "presets"])
    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=stack.effective().safety,
    )
    platform = SleepyPlatform()
    session = platform.bind(object())
    host = DaytimeHost()
    stop, errors = threading.Event(), []

    def run() -> None:
        try:
            Runtime(stack, manifest, platform, session, host=host).run(stop)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    events: list[str] = []
    try:
        assert wait_for(lambda: host.diagnostics is not None and host.bus is not None)
        for topic in ("robot.asleep", "robot.awake"):
            host.bus.subscribe(topic, lambda m: events.append(m.topic))
        driver = session.built

        def section():
            return host.diagnostics.report()["sections"]["roboemot.sleep"]["data"]

        assert wait_for(lambda: len(driver.writes) > 5)
        assert section()["asleep"] is False
        driver.enabled = False
        assert wait_for(lambda: section()["paused"]["app_tick"])
        assert events == ["robot.asleep"]
        assert host.diagnostics.report()["sections"]["roboemot.motion"]["data"]["asleep"] is True
        time.sleep(0.3)
        n = len(driver.writes)
        time.sleep(0.5)
        assert len(driver.writes) == n  # nothing written while asleep
        driver.enabled = True
        assert wait_for(lambda: not section()["paused"]["app_tick"])
        assert wait_for(lambda: len(driver.writes) > n)
        assert events == ["robot.asleep", "robot.awake"]
    finally:
        stop.set()
        thread.join(timeout=30)
    assert not errors, errors and errors[0]
    assert not thread.is_alive()
