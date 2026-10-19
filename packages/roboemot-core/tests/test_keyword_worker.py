"""SPEC 8 stage 8: the stop-keyword recognizer runs on its own thread, so the speech path's
cut-off never waits on it (a slow Vosk held the listener thread and the stop came late)."""

from __future__ import annotations

import threading
import time

import numpy as np
from roboemot.voice.keyword_worker import KeywordWorker
from test_bargein import World, user_voice
from test_voice_listener import FRAME, RATE, FakeASR

SLOW_S = 0.4
WAIT_S = 5.0


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
