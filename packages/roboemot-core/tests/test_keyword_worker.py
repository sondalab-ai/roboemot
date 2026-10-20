"""SPEC 8 stage 8: the stop-keyword recognizer runs on its own thread, so the speech path's
cut-off never waits on it (a slow Vosk held the listener thread and the stop came late)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
from roboemot.voice.asr import ASREvent
from roboemot.voice.keyword_worker import KeywordWorker
from test_bargein import World, user_voice
from test_voice_listener import FRAME, RATE, FakeASR

SLOW_S = 0.4
WAIT_S = 5.0
SLICE_S = 0.01


class GatedASR(FakeASR):
    """``start`` and ``accept`` block until ``gate`` is set; ``active`` counts overlapping calls."""

    def __init__(self, script):
        super().__init__(script)
        self.gate = threading.Event()
        self.heard = threading.Event()
        self.in_call = 0
        self.overlap = False

    def _guard(self, fn, *a):
        if self.in_call:
            self.overlap = True
        self.in_call += 1
        try:
            self.gate.wait(WAIT_S)
            return fn(*a)
        finally:
            self.in_call -= 1

    def start(self, language):
        return self._guard(super().start, language)

    def accept(self, samples):
        event = self._guard(super().accept, samples)
        self.heard.set()
        return event

    def reset(self):
        if self.in_call:
            self.overlap = True
        super().reset()


def background_world(script, asr):
    w = World(script, hear_echo=False, keyword_background=True)
    for name in ("start", "accept", "reset"):
        setattr(w.rig.asr, name, getattr(asr, name))
    w.wall_stop = []
    inner = w.stop

    def stop():
        w.wall_stop.append(time.monotonic())
        inner()

    w.bargein.bind(stop, w.interrupt, w.rig.listener.resume)
    w.speak(4.0)
    return w


def test_the_speech_cut_off_does_not_wait_for_a_slow_recognizer():
    asr = GatedASR(["ciao"])
    w = background_world(["ciao"], asr)
    released = []

    def release():
        released.append(time.monotonic())
        asr.gate.set()

    timer = threading.Timer(SLOW_S, release)
    timer.start()
    try:
        w.feed(1.0, user_voice(0.8, 0.21), user_at=0.0)  # loud: a speech cut-off, and a candidate
    finally:
        timer.join()
        w.bargein.close()
    assert len(w.stops) == 1 and w.events == []
    assert w.wall_stop[0] < released[0], "the stop waited for the keyword recognizer"
    assert not asr.overlap, "the listener used the recognizer while the worker did"
    snap = w.bargein.snapshot()
    assert snap["triggers"] == 1 and snap["latency"]["detect_to_stop"]["last_ms"] < 100


def test_a_stop_heard_by_the_worker_cuts_the_robot_on_a_later_chunk():
    asr = GatedASR(["stop"])
    w = background_world(["stop"], asr)
    try:
        w.feed(0.3, user_voice(1.0, 0.065), user_at=0.0)  # a candidate, not a speech cut-off
        assert w.stops == []
        asr.gate.set()
        assert asr.heard.wait(WAIT_S)
        w.feed(0.2, user_voice(1.0, 0.065), user_at=0.0)
    finally:
        asr.gate.set()
        w.bargein.close()
    assert len(w.stops) == 1 and w.events == ["speech.keyword.stop"]
    snap = w.bargein.snapshot()
    assert snap["keyword_triggers"] == 1 and snap["triggers"] == 0
    assert w.rig.listener.snapshot()["barge_ins"] == 0


def test_a_hit_after_the_session_ended_is_discarded():
    seen = []
    asr = GatedASR(["stop"])
    worker = KeywordWorker(asr, lambda text, lang: True, FRAME / RATE, background=True)
    frame = np.zeros(FRAME, np.float32)
    worker.start("it", [(0.0, frame)])
    worker.end(block=False)  # the session is over before the recognizer answers
    asr.gate.set()
    worker.end(block=True)
    seen.append(worker.take_hit())
    worker.close()
    assert seen == [None] and asr.resets >= 1


def test_ending_blocks_until_the_recognizer_is_free_then_resets_it():
    asr = GatedASR(["x"])
    worker = KeywordWorker(asr, lambda text, lang: False, FRAME / RATE, background=True)
    worker.start("en", [(0.0, np.zeros(FRAME, np.float32))] * 3)
    while not asr.in_call:
        time.sleep(0.001)
    threading.Timer(0.1, asr.gate.set).start()
    began = time.monotonic()
    worker.end(block=True)
    assert time.monotonic() - began >= 0.05
    calls_at_end = asr.calls
    time.sleep(0.1)
    assert asr.calls == calls_at_end and asr.resets == 1  # nothing fed after the hand-back
    worker.close()


def test_a_recognizer_error_is_counted_and_ends_the_session():
    class Boom(FakeASR):
        def accept(self, samples):
            raise RuntimeError("x")

    asr = Boom(["x"])
    worker = KeywordWorker(asr, lambda text, lang: True, FRAME / RATE)
    worker.start("it", [(0.0, np.zeros(FRAME, np.float32))] * 2)
    assert worker.errors == 1 and asr.resets == 1 and worker.take_hit() is None


def test_close_returns_within_its_bound_when_the_worker_is_stuck_in_the_recognizer(monkeypatch):
    from roboemot.voice import keyword_worker

    monkeypatch.setattr(keyword_worker, "JOIN_S", 0.2)
    asr = GatedASR(["x"])
    worker = KeywordWorker(asr, lambda text, lang: False, FRAME / RATE, background=True)
    worker.start("it", [(0.0, np.zeros(FRAME, np.float32))])
    while not asr.in_call:
        time.sleep(0.001)
    began = time.monotonic()
    worker.close()  # the ASR call never returns until the gate opens
    try:
        assert time.monotonic() - began < 1.0
        assert asr.resets == 0, "the recognizer was reset under the stuck thread"
    finally:
        asr.gate.set()


def test_a_hit_of_a_dropped_session_is_not_taken_by_the_session_that_follows():
    class ByLanguage(GatedASR):
        def start(self, language):
            self.lang = language
            return FakeASR.start(self, language)

        def accept(self, samples):
            self._guard(lambda: None)
            self.heard.set()
            return ASREvent("stop" if self.lang == "old" else "ciao")

    asr = ByLanguage(["x"])
    worker = KeywordWorker(asr, lambda text, lang: lang == "old", FRAME / RATE, background=True)
    frame = np.zeros(FRAME, np.float32)
    worker.start("old", [(0.0, frame)])
    while not asr.in_call:
        time.sleep(0.001)
    worker.end(block=False)  # the old session is dropped while the recognizer answers it
    worker.start("new", [(1.0, frame)])  # and a new one begins at once
    asr.gate.set()
    deadline = time.monotonic() + WAIT_S
    while asr.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    worker.end(block=True)
    assert worker.take_hit() is None
    worker.close()


def test_a_config_reload_while_a_session_is_live_uses_the_new_stop_words():
    from roboemot.config.schema import RoboEmotConfig

    asr = GatedASR(["halt"])
    w = background_world(["halt"], asr)
    try:
        w.feed(0.3, user_voice(1.0, 0.065), user_at=0.0)  # a session is open, the worker is held
        cfg = RoboEmotConfig.model_validate({"voice": {"keywords": {"it": {"stop": ["halt"]}}}})
        w.bargein.apply_config(cfg)  # hot reload from the config thread
        asr.gate.set()
        assert asr.heard.wait(WAIT_S)
        w.feed(0.2, user_voice(1.0, 0.065), user_at=0.0)
    finally:
        asr.gate.set()
        w.bargein.close()
    assert len(w.stops) == 1 and w.events == ["speech.keyword.stop"]


def test_a_gap_during_a_live_session_never_uses_the_recognizer_from_two_threads():
    class Slow(GatedASR):
        def _guard(self, fn, *a):
            return super()._guard(lambda *x: (time.sleep(SLICE_S), fn(*x))[1], *a)

    asr = Slow(["ciao"])
    asr.gate.set()
    w = background_world(["ciao"], asr)
    try:
        w.feed(0.5, user_voice(1.0, 0.065), user_at=0.0)  # a session runs, the worker is busy
        gap = SimpleNamespace(
            samples=np.zeros(1024, np.float32), rate_hz=RATE, stamp=w.rig.t + 0.064, gap=True
        )
        w.rig.listener.process(gap)  # _abort -> hand back first, then the listener resets the ASR
        assert w.bargein._session is None
        assert not asr.overlap, "the listener and the worker used the recognizer together"
        time.sleep(3 * SLICE_S)
        assert asr.in_call == 0
        assert w.bargein.snapshot()["errors"] == 0
    finally:
        w.bargein.close()


def test_a_worker_error_ends_the_session_so_the_next_candidate_restarts_it():
    class FailsOnce(FakeASR):
        boom = True

        def accept(self, samples):
            if FailsOnce.boom:
                FailsOnce.boom = False
                raise RuntimeError("x")
            return super().accept(samples)

    FailsOnce.boom = True
    w = World(["ciao"], hear_echo=False, keyword_background=True)
    asr = FailsOnce(["ciao"])
    for name in ("start", "accept", "reset"):
        setattr(w.rig.asr, name, getattr(asr, name))
    w.speak(4.0)
    try:
        w.feed(0.3, user_voice(1.0, 0.065), user_at=0.0)
        deadline = time.monotonic() + WAIT_S
        while w.bargein.snapshot()["errors"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        w.feed(0.2, user_voice(1.0, 0.065), user_at=0.0)  # polls the error, drops the session
        w.feed(0.3, user_voice(1.0, 0.065), user_at=0.0)  # a candidate: a new session
        assert w.bargein.snapshot()["errors"] == 1
        assert len(asr.started) >= 2, "the session was not restarted"
    finally:
        w.bargein.close()


def test_close_between_thread_creation_and_start_does_not_join_an_unstarted_thread(monkeypatch):
    real_start = threading.Thread.start
    worker = KeywordWorker(GatedASR(["stop"]), lambda t, l: True, FRAME / RATE, background=True)

    def start_after_close(self):
        if self.name == "roboemot-keywords":
            worker.close()
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start_after_close)
    worker.start("it", [(0.0, np.zeros(FRAME, np.float32))])
    worker.close()
