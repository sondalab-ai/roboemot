"""SPEC 11, Voice tools (Due items, Fallback, Repeat): the announcer of due timers and reminders,
driven on a fake clock against a real ``Scheduler`` and fake runtime parts."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from roboemot.bus import Bus
from roboemot.config import ConfigStack
from roboemot.conversation.phase import ConversationPhase
from roboemot.runtime.ticks import conversation_idle
from roboemot.tools.announce import EVENT_PREFIX, Announcer
from roboemot.tools.scheduler import DUE_TOPIC, Scheduler, SchedulerRunner
from roboemot.voice.listener import UtteranceInfo
from roboemot.voice.wake import WakeGate

START = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


class Clock:
    """Monotonic seconds and the wall clock, moved only by ``wait`` (no real sleeps)."""

    def __init__(self) -> None:
        self.t = 0.0
        self.step: float | None = None  # a wait moves at least this far (long waits)

    def __call__(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return START + timedelta(seconds=self.t)

    def wait(self, seconds: float) -> bool:
        self.t += max(seconds, self.step or 0.0)
        return False


def result(value):
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(value)
    return future


class FakeWorker:
    def __init__(self, clock: Clock, answers=(True,)) -> None:
        self.clock = clock
        self.answers = list(answers)
        self.turns: list[tuple[float, str, str]] = []
        self.lines: list[tuple[str, str]] = []
        self.voice = True
        self.after = 0

    def event_turn(self, text, language):
        self.turns.append((self.clock(), text, language))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer if isinstance(answer, concurrent.futures.Future) else result(answer)

    def local_line(self, text, language):
        self.lines.append((text, language))
        return result(self.voice)

    def after_turn(self, callback):
        self.after += 1
        callback()

    def busy(self) -> bool:
        return False


class FakeExpression:
    def __init__(self, clock: Clock, circadian=None) -> None:
        self.clock, self.circadian = clock, circadian
        self.requests: list[tuple[float, str, bool]] = []

    def request(self, label, now=None):
        ok = self.circadian is None or self.circadian.model.state != "resting"
        self.requests.append((self.clock(), label, ok))
        return ok


class FakeCircadian:
    """Resting until woken; the state leaves ``resting`` ``tick_s`` after ``wake`` (the next
    runner tick)."""

    def __init__(self, clock: Clock, *, napping=False, tick_s=2.0) -> None:
        self.clock, self.tick_s = clock, tick_s
        self.enabled = True
        self.napping = napping
        self.woken_at: float | None = None
        self.wakes = 0
        self.model = self

    @property
    def state(self) -> str:
        if self.woken_at is not None and self.clock() >= self.woken_at + self.tick_s:
            return "awake"
        return "resting"

    def nap_status(self):
        return "already_resting" if self.state == "resting" else "ok"

    def wake(self):
        self.wakes += 1
        if self.woken_at is None:
            self.woken_at = self.clock()


class FakeCues:
    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, kind, reason="", **detail):
        self.played.append(kind)
        return True


def host_for(clock: Clock, *, speaking_until=0.0, voice_active=False, circadian=None):
    speech = SimpleNamespace(speaking=lambda now: now < speaking_until)
    output = SimpleNamespace(
        speaking_at=lambda now: False,
        timeline=SimpleNamespace(sentence_at=lambda now: None),
    )
    listener = SimpleNamespace(voice_active=lambda: voice_active, opened=[])
    listener.open_follow_up = lambda: listener.opened.append(clock())
    return SimpleNamespace(
        phase=None,
        speech=speech,
        speech_output=output,
        voice_listener=listener,
        conversation=None,
        circadian_runner=circadian,
        expression=FakeExpression(clock, circadian),
        cues=FakeCues(),
        bus=Bus(),
        media=None,
        scheduler_runner=None,
    )


class Rig:
    def __init__(self, tmp_path, *, worker_answers=(True,), app=None, asleep=None, **host_kw):
        self.clock = Clock()
        self.stack = ConfigStack(app={"conversation": {"language": "en"}, **(app or {})})
        self.scheduler = Scheduler(
            tmp_path / "schedule.json",
            lambda: self.stack.effective().tools.scheduler,
            self.clock.wall,
            lambda: UTC,
        )
        self.scheduler.load(self.clock.wall())
        self.host = host_for(self.clock, **host_kw)
        self.worker = FakeWorker(self.clock, worker_answers)
        self.host.conversation = self.worker
        self.fired: list[str] = []
        fired = self.scheduler.fired

        def record(item_id):
            self.fired.append(item_id)
            fired(item_id)

        self.scheduler.fired = record  # type: ignore[method-assign]
        self.announcer = Announcer(
            self.scheduler,
            self.worker,
            self.host,
            self.stack.effective,
            clock=self.clock,
            wait=self.clock.wait,
            asleep=asleep or (lambda: False),
        )

    def timer(self, label="pasta"):
        item = self.scheduler.add_timer(60, label)
        self.clock.t += 60
        return item

    def reminder(self, text="call mum"):
        item = self.scheduler.add_reminder(
            text, (self.clock.wall() + timedelta(minutes=5)).replace(tzinfo=None), "once", ()
        )
        self.clock.t += 300
        return item


def nods(rig):
    return [r for r in rig.host.expression.requests if r[1] == "nod"]


# ------------------------------------------------------------------ the busy robot


def test_due_item_waits_for_idle_then_nods_and_announces(tmp_path):
    rig = Rig(tmp_path)
    item = rig.timer()
    t0 = rig.clock.t  # the item is due; the robot speaks until t0 + 3
    rig.host.speech.speaking = lambda now: now < t0 + 3.0
    assert rig.announcer.handle(item.id)
    assert len(rig.worker.turns) == 1
    at, text, language = rig.worker.turns[0]
    assert at >= t0 + 3.0 and language == "en"
    assert text.startswith(EVENT_PREFIX)
    assert text == EVENT_PREFIX + "A timer the person set is due: pasta. Tell them now, briefly."
    assert len(nods(rig)) == 1 and nods(rig)[0][0] >= t0 + 3.0
    assert rig.fired == [item.id] and rig.scheduler.get(item.id) is None
    assert rig.host.voice_listener.opened  # the follow-up window opened after it
    assert rig.host.cues.played == []


def test_a_reminder_event_text_carries_its_text_and_a_timer_without_label_says_so(tmp_path):
    rig = Rig(tmp_path)
    reminder, timer = rig.reminder(), rig.timer(label=None)
    rig.announcer.handle(reminder.id)
    rig.announcer.handle(timer.id)
    texts = [t for _, t, _ in rig.worker.turns]
    assert texts == [
        EVENT_PREFIX + "A reminder the person set is due: call mum. Tell them now, briefly.",
        EVENT_PREFIX + "A timer the person set is due: no label. Tell them now, briefly.",
    ]


def test_due_item_speaks_after_max_wait_even_if_busy(tmp_path):
    rig = Rig(
        tmp_path, voice_active=True, app={"tools": {"scheduler": {"announce_max_wait_s": 20}}}
    )
    item = rig.timer()
    start = rig.clock.t
    assert rig.announcer.handle(item.id)
    at = rig.worker.turns[0][0]
    assert 20.0 <= at - start < 21.0  # the person still talks: it does not wait for them
    assert rig.fired == [item.id]


def test_after_max_wait_it_still_waits_for_the_current_sentence(tmp_path):
    rig = Rig(tmp_path, voice_active=True, app={"tools": {"scheduler": {"announce_max_wait_s": 0}}})
    item = rig.timer()
    start = rig.clock.t
    rig.host.speech_output.timeline.sentence_at = lambda now: "s" if now < start + 2 else None
    rig.announcer.handle(item.id)
    assert rig.worker.turns[0][0] >= start + 2  # never cut mid-sentence


# ------------------------------------------------------------------ sleep and rest


def test_asleep_defers_and_fires_on_wake_however_late(tmp_path):
    asleep_until = [0.0]
    rig = Rig(tmp_path, asleep=lambda: rig.clock.t < asleep_until[0])
    item = rig.timer()
    asleep_until[0] = rig.clock.t + 3 * 3600
    rig.clock.step = 60.0
    assert rig.announcer.handle(item.id)
    assert rig.worker.turns[0][0] >= asleep_until[0]
    assert rig.fired == [item.id]
    assert rig.scheduler.missed() == []


def test_napping_robot_is_woken(tmp_path):
    clock_holder: list = []
    rig = Rig(tmp_path)
    circadian = FakeCircadian(rig.clock, napping=True)
    clock_holder.append(circadian)
    rig.host.circadian_runner = circadian
    rig.host.expression.circadian = circadian
    item = rig.timer()
    assert rig.announcer.handle(item.id)
    assert circadian.wakes >= 1
    assert rig.worker.turns and rig.fired == [item.id]


def test_quiet_hours_rest_is_lifted_before_the_nod(tmp_path):
    rig = Rig(tmp_path)
    circadian = FakeCircadian(rig.clock, tick_s=5.0)  # resting: the quiet hours
    rig.host.circadian_runner = circadian
    rig.host.expression.circadian = circadian
    item = rig.reminder()
    woken = rig.clock.t
    rig.announcer.handle(item.id)
    assert circadian.woken_at == woken
    [(at, _, ok)] = nods(rig)
    assert at >= woken + 5.0 and ok  # only after the state left resting, and it played
    assert rig.worker.turns[0][0] >= at


# ------------------------------------------------------------------ fallback and repeat


def test_fallback_sentence_and_cue_when_route_fails(tmp_path):
    rig = Rig(tmp_path, worker_answers=(False,))
    item = rig.timer()
    assert rig.announcer.handle(item.id)
    assert rig.worker.lines == [("The pasta timer is up.", "en")]
    assert rig.host.cues.played == ["fallback"]
    assert len(nods(rig)) == 1
    assert rig.fired == [item.id]
    assert rig.host.voice_listener.opened == []  # nobody answered: no follow-up window


def test_fallback_without_a_local_voice_is_the_cue_and_the_gesture_only(tmp_path):
    rig = Rig(tmp_path, worker_answers=(False,), app={"conversation": {"language": "it"}})
    rig.worker.voice = False
    item = rig.reminder("chiama la mamma")
    rig.announcer.handle(item.id)
    assert rig.worker.lines == [("Promemoria: chiama la mamma.", "it")]  # not said: no voice
    assert rig.host.cues.played == ["fallback"] and len(nods(rig)) == 1
    assert rig.fired == [item.id]


def test_timer_repeats_until_answered_at_most_repeat_max(tmp_path):
    app = {"tools": {"scheduler": {"repeat_after_s": 30, "repeat_max": 2}}}
    rig = Rig(tmp_path, app=app)
    item = rig.timer()
    rig.announcer.handle(item.id)
    for _ in range(5):
        rig.clock.t += 31
        rig.announcer.run_repeats()
    assert len(rig.worker.turns) == 3  # the announcement and two repeats
    assert rig.fired == [item.id]  # fired once, after the first


def test_an_addressed_utterance_stops_the_repeats(tmp_path):
    rig = Rig(tmp_path)
    rig.announcer.attach()
    item = rig.timer()
    rig.announcer.handle(item.id)
    rig.clock.t += 5
    info = UtteranceInfo("en", 2, 1, 0.0, 0.0, 0.0, True, False, -20.0, None, None)
    rig.host.bus.publish("percept.utterance", info, rig.clock.t)
    rig.clock.t += 60
    rig.announcer.run_repeats()
    assert len(rig.worker.turns) == 1


def test_a_reminder_does_not_repeat(tmp_path):
    rig = Rig(tmp_path)
    item = rig.reminder()
    rig.announcer.handle(item.id)
    rig.clock.t += 600
    rig.announcer.run_repeats()
    assert len(rig.worker.turns) == 1


def test_a_cancelled_or_not_yet_due_id_is_dropped(tmp_path):
    rig = Rig(tmp_path)
    item = rig.timer()
    rig.scheduler.cancel(item.id)
    assert not rig.announcer.handle(item.id)
    later = rig.scheduler.add_timer(600, "x")
    assert not rig.announcer.handle(later.id)
    assert rig.worker.turns == [] and rig.fired == []


def test_media_pauses_when_the_player_exists(tmp_path):
    rig = Rig(tmp_path)
    paused: list[str] = []
    rig.host.media = SimpleNamespace(pause=paused.append)
    rig.announcer.handle(rig.timer().id)
    assert paused == ["announce"]


# ------------------------------------------------------------------ the bus hand-off


def test_the_due_handler_never_waits_and_drops_ids_in_flight(tmp_path):
    rig = Rig(tmp_path)
    gate: concurrent.futures.Future = concurrent.futures.Future()
    rig.worker.answers = [gate]
    runner = SchedulerRunner(rig.scheduler, rig.host.bus, rig.scheduler.cfg)
    rig.announcer = Announcer(
        rig.scheduler,
        rig.worker,
        rig.host,
        rig.stack.effective,
        clock=rig.clock,
        asleep=lambda: False,
        runner=runner,
    )
    item = rig.timer()
    rig.announcer.start()
    try:
        assert runner.consumer_attached
        done = threading.Event()

        def publish():
            for _ in range(3):
                rig.host.bus.publish(DUE_TOPIC, [item.id], 0.0)
            done.set()

        threading.Thread(target=publish).start()
        assert done.wait(2.0)  # the publisher's thread never waits on the announcement
        deadline = 200
        while not rig.worker.turns and deadline:
            time.sleep(0.01)
            deadline -= 1
        assert len(rig.worker.turns) == 1  # the repeats of an id in flight are dropped
        gate.set_result(True)
    finally:
        rig.announcer.close()
    assert runner.consumer_attached  # the runner never marks items fired itself after close


# ------------------------------------------------------------------ idle and the window


def test_conversation_idle_needs_every_part_quiet():
    busy = [False]
    host = SimpleNamespace(
        phase=ConversationPhase(),
        speech=SimpleNamespace(speaking=lambda now: now < 1.0),
        voice_listener=SimpleNamespace(voice_active=lambda: False),
        conversation=SimpleNamespace(busy=lambda: busy[0]),
    )
    assert not conversation_idle(host, 0.5)  # the robot speaks
    assert conversation_idle(host, 2.0)
    busy[0] = True  # a turn or a tool round is open
    assert not conversation_idle(host, 2.0)
    busy[0] = False
    host.phase.begin_thinking()
    assert not conversation_idle(host, 2.0)
    host.phase.end_thinking()
    host.voice_listener = SimpleNamespace(voice_active=lambda: True)  # the person talks
    assert not conversation_idle(host, 2.0)


def test_the_wake_gate_opens_the_follow_up_window_for_anyone():
    gate = WakeGate("always", 15.0, 120.0)
    assert not gate.decide(wake_word=False, enrolled_in_view=False, start=10.0, last_speech=11.0)
    gate.open_window(20.0)
    assert gate.decide(
        wake_word=False, enrolled_in_view=False, start=30.0, last_speech=31.0, speaker="p1"
    )
    gate.open_window(100.0)
    assert not gate.decide(wake_word=False, enrolled_in_view=False, start=116.0, last_speech=117.0)
