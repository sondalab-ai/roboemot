"""SPEC 11 (Voice tools, Sleep) and SPEC 7A (Requested nap): the ``go_to_sleep`` skill, the
nap it arms after the calling turn's last reply audio, and the follow-up window it closes."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import test_circadian_nap as napkit
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.runtime.host import RuntimeHost
from roboemot.skills.host import tool_host
from roboemot.skills.plugins import load_plugin_skill_sets
from roboemot.skills.registry import SkillResult
from roboemot.tools import sleep
from test_conversation_tools import C, Env, Scripted
from test_conversation_worker import (
    GATE,
    rigs,  # noqa: F401  (fixture)
    wait_for,
)
from test_voice_listener import Rig as ListenerRig


class FakeRunner:
    """The circadian runner's nap calls: a fixed status and the times a nap was started."""

    def __init__(self, clock, status="ok"):
        self.clock = SimpleNamespace(monotonic=clock)
        self.status = status
        self.checked = 0
        self.started: list[float] = []

    def nap_status(self):
        self.checked += 1
        return self.status

    def request_nap(self, now):
        self.started.append(now)
        return self.status


def host_with(nap=None):
    h = tool_host(lambda: RoboEmotConfig(), Bus())
    h.nap = nap
    return h


def test_the_skill_is_go_to_sleep_visible_and_available_only_with_a_nap_part():
    h = host_with()
    (fn,) = sleep.skills(h)
    s = fn.skill
    assert s.name == "go_to_sleep" and s.llm_visible and s.params == ()
    assert "good night" in s.text() and "name wakes you" in s.text()
    # robot run 2026-11-18: "hey neve dormi" got chat ("sono sempre sveglia"), not the tool
    assert "even a single word" in s.text() and "Never answer that you do not sleep" in s.text()
    assert not s.is_available()
    h.nap = lambda: "ok"  # read live: set after the factory ran
    assert s.is_available()


def test_the_skill_returns_the_nap_status():
    h = host_with(lambda: "already_resting")
    (fn,) = sleep.skills(h)
    out = fn()
    assert isinstance(out, SkillResult) and out.content == {"status": "already_resting"}


def test_core_registers_the_sleep_plugin_and_lists_it_by_default():
    plugins = {p.name: p for p in load_plugin_skill_sets(host_with())}
    assert plugins["sleep"].distribution == "roboemot-core"
    assert [fn.skill.name for fn in plugins["sleep"].skills] == ["go_to_sleep"]
    assert "go_to_sleep" in RoboEmotConfig().conversation.llm_skills


def test_the_runtime_host_refuses_at_once_without_arming():
    runner = FakeRunner(lambda: 0.0, status="asleep")
    h = RuntimeHost()
    h.circadian_runner = runner
    assert h.request_nap() == "asleep" and runner.started == []
    assert RuntimeHost().request_nap() == "disabled"


def test_go_to_sleep_starts_after_the_calling_turn_ends(rigs):  # noqa: F811
    rig = rigs()
    Scripted(rig, [C(0, "c1", "go_to_sleep", {})], ["Buonanotte. ", GATE])
    rig.clock[0] = 0.0
    rig.speech.speaking_at = lambda now: 0.5 <= now < 2.0  # the reply's audio
    runner = FakeRunner(lambda: rig.clock[0])
    h = RuntimeHost()
    h.circadian_runner, h.conversation = runner, rig.worker
    registry = Env().registry
    for fn in sleep.skills(host_with(h.request_nap)):
        registry.add(fn)
    rig.worker.skills = registry
    n = rig.say("vai a dormire")
    assert wait_for(lambda: runner.checked == 1)  # called at t=0, nothing speaking
    rig.clock[0] = 0.5
    rig.server.gate.set()
    assert rig.done(n)
    time.sleep(0.2)
    assert runner.started == []  # the good night still plays
    rig.clock[0] = 2.0
    assert wait_for(lambda: runner.started == [2.0])
    time.sleep(0.2)
    assert runner.started == [2.0]  # once


def test_after_turn_runs_at_once_when_no_turn_is_open(rigs):  # noqa: F811
    rig = rigs()
    ran = []
    rig.worker.after_turn(lambda: ran.append(1))
    assert ran == [1]


def test_nap_start_closes_the_follow_up_window():
    r = ListenerRig(["ciao reati"], final="ciao reati vai a dormire")
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word
    h = RuntimeHost()
    h.circadian_runner = FakeRunner(lambda: r.now)
    h.voice_listener = r.listener
    assert h.request_nap() == "ok"  # no conversation worker: the nap starts at once
    r.asr.script, r.asr.final_text = ["dimmi"], "dimmi di piu"
    r.say()
    assert not r.utterances[-1].addressed  # the worker never takes it
    r.asr.script, r.asr.final_text = ["ciao reati"], "ciao reati svegliati"
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word


def test_a_refused_start_keeps_the_follow_up_window():
    r = ListenerRig(["ciao reati"], final="ciao reati vai a dormire")
    r.say()
    h = RuntimeHost()
    runner = h.circadian_runner = FakeRunner(lambda: r.now)
    runner.request_nap = lambda now: "already_resting"  # quiet hours began meanwhile
    h.voice_listener = r.listener
    assert h.request_nap() == "ok"
    r.asr.script, r.asr.final_text = ["dimmi"], "dimmi di piu"
    r.say()
    assert r.utterances[-1].addressed


# ---- fix round 1: failed turns, speech and barge-in during the good night


def sleeping(rigs, *rounds):  # noqa: F811
    """A worker whose model calls go_to_sleep, over a RuntimeHost with a FakeRunner; the reply
    plays while the clock is in [0.5, 2)."""
    rig = rigs()
    srv = Scripted(rig, [C(0, "c1", "go_to_sleep", {})], *rounds)
    rig.clock[0] = 0.0
    rig.speech.speaking_at = lambda now: 0.5 <= now < 2.0
    runner = FakeRunner(lambda: rig.clock[0])
    h = RuntimeHost()
    h.circadian_runner, h.conversation = runner, rig.worker
    registry = Env().registry
    for fn in sleep.skills(host_with(h.request_nap)):
        registry.add(fn)
    rig.worker.skills = registry
    return rig, srv, runner


def after_respond(rig, then):
    """Wrap the worker's _respond: the real answer (the tool runs), then ``then()``."""
    real = rig.worker._respond

    async def respond(turn, router, now):
        await real(turn, router, now)
        await then()

    rig.worker._respond = respond


def test_a_turn_that_raises_still_starts_the_armed_nap_once(rigs):  # noqa: F811
    rig, _, runner = sleeping(rigs, ["Buonanotte. "])

    async def boom():
        raise RuntimeError("late failure")

    after_respond(rig, boom)
    rig.clock[0] = 0.5  # the good night plays when the turn fails
    rig.say("vai a dormire")
    assert wait_for(lambda: rig.worker.snapshot().get("errors", 0) == 1)
    assert rig.worker._active_turn is None  # no stale open turn
    time.sleep(0.1)
    assert runner.started == []  # it still waits for the reply audio
    rig.clock[0] = 2.0
    assert wait_for(lambda: runner.started == [2.0])
    ran = []
    rig.worker.after_turn(lambda: ran.append(1))  # a later caller sees no open turn
    assert ran == [1]
    rig.worker._respond = type(rig.worker)._respond.__get__(rig.worker)
    rig.turn("ciao")  # the next turn runs normally
    time.sleep(0.1)
    assert runner.started == [2.0]


def test_a_cancelled_turn_still_starts_the_armed_nap_once(rigs):  # noqa: F811
    rig, _, runner = sleeping(rigs, ["Buonanotte. "])
    reached = threading.Event()

    async def hang():
        reached.set()
        await asyncio.sleep(3600)

    after_respond(rig, hang)
    rig.say("vai a dormire")
    assert reached.wait(5)
    rig.worker.close()  # cancels the turn in flight
    assert runner.started == [0.0] and rig.worker._active_turn is None


def test_speech_during_the_good_night_delays_the_nap_to_after_that_turn(rigs):  # noqa: F811
    rig, srv, runner = sleeping(rigs, ["Buonanotte. ", GATE], ["Va bene. "])
    hold = threading.Event()
    real = rig.server.handler

    async def handler(request):
        if len(srv.bodies) >= 2:  # the next turn's request waits for the test
            while not hold.is_set():
                await asyncio.sleep(0.002)
        return await real(request)

    rig.server.handler = handler
    first = rig.say("vai a dormire")
    assert wait_for(lambda: runner.checked == 1)
    rig.clock[0] = 0.5
    second = rig.say("aspetta un attimo")  # queued while the good night plays
    rig.server.gate.set()
    assert rig.done(first)
    rig.clock[0] = 2.5  # the good night ended; the next turn is being served
    time.sleep(0.2)
    assert runner.started == []
    hold.set()
    assert rig.done(second)
    assert wait_for(lambda: runner.started == [2.5])
    time.sleep(0.1)
    assert runner.started == [2.5]  # never cancelled, started once


def test_a_barge_in_during_the_good_night_still_starts_the_nap(rigs):  # noqa: F811
    rig, _, runner = sleeping(rigs, ["Buonanotte. ", GATE])
    n = rig.say("vai a dormire")
    assert wait_for(lambda: runner.checked == 1)
    rig.clock[0] = 0.5
    assert rig.worker.interrupt()
    assert rig.done(n)
    rig.clock[0] = 2.0
    assert wait_for(lambda: runner.started == [2.0])
    time.sleep(0.1)
    assert runner.started == [2.0]


# ---- RuntimeHost.request_nap over a real CircadianRunner


def runtime_over(x):
    h = RuntimeHost()
    h.circadian_runner = x.r
    return h


def test_the_runtime_host_reports_asleep_and_already_resting_from_the_runner():
    x = napkit.rig()
    x.bus.publish("robot.asleep", None, x.clock.t)
    assert runtime_over(x).request_nap() == "asleep"
    night = napkit.rig(hour=23.5)
    assert runtime_over(night).request_nap() == "already_resting"
    for y in (x, night):
        assert not y.r.nap.requested(y.clock.t)


def test_the_runtime_host_starts_the_nap_at_once_without_a_conversation():
    x = napkit.rig()
    assert runtime_over(x).request_nap() == "ok"
    assert x.r.nap.requested(x.clock.t)
    napkit.run(x, napkit.STEP)
    assert (x.r.model.state, x.r.model.reason) == ("resting", "nap")
