"""SPEC 12A: the camera pump, the idle light reader and the voice listener can be paused (the
robot is asleep) without stopping their threads, and resume where they left off."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
from roboemot.circadian.sampler import IdleLightReader
from roboemot.vision.runner import pump
from test_voice_listener import RATE, Rig, silence, voice


def wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class CountingSource:
    def __init__(self) -> None:
        self.reads = 0

    def read(self):
        self.reads += 1
        return None


def test_the_camera_pump_reads_nothing_while_paused_and_goes_on_after():
    source, paused, stop = CountingSource(), threading.Event(), threading.Event()
    paused.set()
    thread = threading.Thread(
        target=pump, args=(source, None, stop, 200.0), kwargs={"paused": paused}
    )
    thread.start()
    try:
        time.sleep(0.4)
        assert source.reads == 0
        paused.clear()
        assert wait_for(lambda: source.reads > 0)
    finally:
        stop.set()
        thread.join(2)
    assert not thread.is_alive()


def test_the_idle_light_reader_reads_no_frame_while_paused():
    class Sampler:
        circadian = SimpleNamespace(idle_poll_s=0.01)
        attempts = 0

        def idle_due(self):
            return True

        def note_attempt(self):
            self.attempts += 1

        def offer(self, image, force=False):
            return True

    sampler, source = Sampler(), CountingSource()
    reader = IdleLightReader(sampler, source)
    reader.pause()
    reader.start()
    try:
        time.sleep(0.3)
        assert source.reads == 0 and reader.paused and reader.running
        reader.resume()
        assert wait_for(lambda: source.reads > 0)
    finally:
        reader.stop()


class Feed:
    """The audio hub's queue as the listener's source: ``read(timeout)`` pops a chunk or None."""

    def __init__(self) -> None:
        self.chunks: list[SimpleNamespace] = []
        self.lock = threading.Lock()
        self.t = 1000.0

    def push(self, samples: np.ndarray) -> None:
        with self.lock:
            for i in range(0, samples.size, 1024):
                part = samples[i : i + 1024]
                self.t += part.size / RATE
                self.chunks.append(
                    SimpleNamespace(samples=part, rate_hz=RATE, stamp=self.t, gap=False)
                )

    def read(self, timeout: float = 0.0):
        with self.lock:
            if self.chunks:
                return self.chunks.pop(0)
        time.sleep(min(timeout, 0.005))
        return None

    def pending(self) -> int:
        with self.lock:
            return len(self.chunks)


def test_a_paused_listener_runs_no_recognizer_drains_the_queue_and_resumes():
    rig, feed = Rig(), Feed()
    rig.listener._source = feed.read
    rig.listener.pause_listening()
    rig.listener.start()
    try:
        assert wait_for(lambda: rig.listener.snapshot()["state"] == "running")
        feed.push(voice(seconds=0.6))
        feed.push(silence(rig.cfg.voice.vad.end_silence_ms / 1000 + 0.3))
        assert wait_for(lambda: feed.pending() == 0)  # emptied, not left to go stale
        time.sleep(0.2)
        assert rig.asr.accepted == 0 and rig.utterances == []
        assert rig.listener.snapshot()["paused"] is True
        rig.listener.resume_listening()
        feed.push(voice(seconds=0.6))
        feed.push(silence(rig.cfg.voice.vad.end_silence_ms / 1000 + 0.3))
        assert wait_for(lambda: len(rig.utterances) == 1)
        assert rig.listener.snapshot()["paused"] is False
    finally:
        rig.listener.stop()


def test_pausing_mid_utterance_drops_it():
    rig, feed = Rig(), Feed()
    rig.listener._source = feed.read
    rig.listener.start()
    try:
        feed.push(voice(seconds=0.5))  # speech begins, no end yet
        assert wait_for(lambda: rig.asr.accepted > 0)
        rig.listener.pause_listening()
        assert wait_for(lambda: rig.asr.resets > 0)
        feed.push(silence(rig.cfg.voice.vad.end_silence_ms / 1000 + 0.3))
        assert wait_for(lambda: feed.pending() == 0)
        time.sleep(0.2)
        assert rig.utterances == []
    finally:
        rig.listener.stop()
