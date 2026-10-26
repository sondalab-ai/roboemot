"""The grammar pass off the listener thread and the bounded hand-off (SPEC 8 stage 3, 13)."""

from __future__ import annotations

import threading
import time

from roboemot.voice.grammar import GrammarSpotter
from roboemot.voice.grammar_worker import GrammarWorker
from roboemot.voice.robot_name import RobotName
from test_voice_grammar import RATE, Decoder, config, vocab
from test_voice_listener import Rig, silence

WAKE = [("ciao", 1.0), ("neve", 1.0)]  # the grammar pass rescues "ciao neve" from "cadeva"


class SlowDecoder(Decoder):
    """A decode that blocks until ``release`` (or ``hold_s`` passes), on the worker thread."""

    def __init__(self, words, hold_s=5.0):
        super().__init__(words)
        self.release = threading.Event()
        self.started = threading.Event()
        self.hold_s = hold_s
        self.thread: str | None = None

    def __call__(self, language, audio, phrases):
        self.thread = threading.current_thread().name
        self.started.set()
        self.release.wait(self.hold_s)
        return super().__call__(language, audio, phrases)


def rig(decoder, background=True, **grammar):
    cfg = config(**grammar)
    return Rig(
        ["cadeva"],
        cfg=cfg,
        names=RobotName("Neve", vocab()),
        grammar=GrammarSpotter(cfg.voice.grammar, decoder, vocab(), RATE),
        grammar_background=background,
    )


def wait_for(cond, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.005)
    return cond()


def test_a_slow_pass_runs_off_thread_and_the_listener_keeps_draining_audio():
    dec = SlowDecoder(WAKE)
    r = rig(dec)
    try:
        r.say(1.0)
        assert dec.started.wait(2.0) and dec.thread == "roboemot-grammar"
        assert r.utterances == []  # waiting for the pass
        t0 = time.perf_counter()
        r.feed(silence(0.5))  # the microphone keeps coming while the pass runs
        assert time.perf_counter() - t0 < dec.hold_s / 2, "the listener waited for the pass"
        assert r.utterances == [] and r.asr.accepted > 0
        dec.release.set()
        assert wait_for(lambda: r.listener.poll_handoff() or r.utterances)
        (u,) = r.utterances
        assert u.wake_word and u.addressed  # the grammar result is applied, as inline
        snap = r.listener.snapshot()["grammar_handoff"]
        assert snap["waits"] == 1 and snap["timeouts"] == 0 and snap["superseded"] == 0
        assert r.listener.snapshot()["realtime"]["total"]["ms_per_audio_s"]["grammar"] > 0
    finally:
        dec.release.set()
        r.listener.stop()


def test_a_pass_past_its_timeout_hands_off_the_free_text_only():
    dec = SlowDecoder(WAKE)
    r = rig(dec)
    try:
        r.say(1.0)
        assert dec.started.wait(2.0)
        timeout = r.cfg.voice.grammar.handoff_timeout_s
        r.now += timeout / 2
        r.listener.poll_handoff()
        assert r.utterances == []
        r.now += timeout
        r.listener.poll_handoff()
        (u,) = r.utterances
        assert not u.wake_word and not u.addressed  # "cadeva" alone wakes nothing
        snap = r.listener.snapshot()
        assert snap["grammar_handoff"]["timeouts"] == 1
        assert not snap["grammar_handoff"]["pending"]
    finally:
        dec.release.set()
        r.listener.stop()


def test_a_new_utterance_hands_off_the_pending_one_first():
    dec = SlowDecoder(WAKE)
    r = rig(dec)
    try:
        r.say(1.0)
        assert dec.started.wait(2.0)
        r.say(1.0, tail=False)  # the next utterance starts while the pass still runs
        assert len(r.utterances) == 1
        assert r.listener.snapshot()["grammar_handoff"]["superseded"] == 1
        assert r.listener.snapshot()["state"] != "unavailable"
    finally:
        dec.release.set()
        r.listener.stop()


def test_a_gap_while_waiting_hands_the_utterance_off_never_drops_it():
    dec = SlowDecoder(WAKE)
    r = rig(dec)
    try:
        r.say(1.0)
        assert dec.started.wait(2.0)
        r.feed(silence(0.1), gap=True)
        assert len(r.utterances) == 1
    finally:
        dec.release.set()
        r.listener.stop()


def test_inline_hand_off_is_immediate_and_matches_the_background_result():
    inline = rig(Decoder(WAKE), background=False)
    inline.say(1.0)
    (a,) = inline.utterances
    assert a.wake_word and a.addressed
    assert inline.listener.snapshot()["grammar_handoff"]["waits"] == 1
    assert inline.order[-1] == "utterance"


def test_an_activity_hook_sees_listening_until_the_hand_off():
    seen: list[bool] = []
    dec = SlowDecoder(WAKE)
    cfg = config()
    r = Rig(
        ["cadeva"],
        cfg=cfg,
        names=RobotName("Neve", vocab()),
        grammar=GrammarSpotter(cfg.voice.grammar, dec, vocab(), RATE),
        grammar_background=True,
        on_activity=seen.append,
    )
    try:
        r.say(1.0)
        assert dec.started.wait(2.0)
        assert seen == [True]
        dec.release.set()
        assert wait_for(lambda: r.listener.poll_handoff() or r.utterances)
        assert seen == [True, False]
    finally:
        dec.release.set()
        r.listener.stop()


def test_worker_replaces_a_waiting_job_and_runs_one_at_a_time():
    gate = threading.Event()
    spent: list[float] = []
    worker = GrammarWorker(background=True, on_spent=spent.append)
    try:
        first = worker.submit(lambda: (gate.wait(5.0), frozenset({"stop"}))[1])
        second = worker.submit(lambda: frozenset({"yes"}))
        third = worker.submit(lambda: frozenset({"no"}))
        assert second.done and not second.ran and worker.replaced == 1
        gate.set()
        assert third._done.wait(2.0)
        assert first.result == {"stop"} and third.result == {"no"} and len(spent) == 2
    finally:
        gate.set()
        worker.close()


def test_voice_kept_flowing_through_the_endpointer_during_the_pass():
    dec = SlowDecoder(WAKE)
    r = rig(dec)
    try:
        r.say(1.0)
        assert dec.started.wait(2.0)
        before = r.asr.started[:]
        r.say(0.6, tail=False)  # speech while the pass runs: a new recognizer session starts
        assert len(r.asr.started) == len(before) + 1
    finally:
        dec.release.set()
        r.listener.stop()
