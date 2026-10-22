"""SPEC 4A, 11A: the runner's latest-frame handoff for the Live tab's preview. Nothing is kept while
nobody watches; a watch window keeps one reference (no copy) to the newest frame the pipeline
finished, with the face boxes computed on it, and lapses by itself."""

from __future__ import annotations

import threading
import time

import numpy as np
from roboemot.vision.runner import LatestFrameRunner
from roboemot.vision.types import Frame


class Module:
    name = "dummy"

    def process(self, frame):
        pass


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


def frame(stamp=0.0):
    return Frame(np.zeros((4, 4, 3), np.uint8), stamp)


def until(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.005)
    return cond()


class Gated(Module):
    """Holds each frame until released; the overlay names the frame it was computed on."""

    def __init__(self):
        self.entered, self.release = threading.Semaphore(0), threading.Semaphore(0)
        self.current = None

    def process(self, f):
        self.entered.release()
        self.release.acquire()
        self.current = f.stamp

    def overlay(self):
        return [self.current]


def started(module, clock=None, overlay=None):
    runner = LatestFrameRunner(module, clock=clock or Clock(), overlay=overlay)
    runner.start()
    return runner


def test_nothing_is_kept_without_a_watcher():
    runner = started(Module())
    try:
        runner.submit(frame())
        assert until(lambda: runner.processed == 1)
        assert runner.latest() is None
    finally:
        runner.stop()


def test_a_watched_runner_keeps_the_processed_frame_itself_not_a_copy():
    runner = started(Module())
    try:
        runner.watch(5.0)
        sent = frame(1.0)
        runner.submit(sent)
        assert until(lambda: runner.latest() is not None)
        kept, seq, overlay = runner.latest()
        assert kept is sent and seq == 1 and overlay is None
    finally:
        runner.stop()


def test_the_preview_shows_the_frame_the_boxes_were_computed_on_not_a_newer_one():
    module = Gated()
    runner = started(module, overlay=module.overlay)
    try:
        runner.watch(5.0)
        runner.submit(frame(1.0))
        assert module.entered.acquire(timeout=5)  # the pipeline is busy with frame 1
        runner.submit(frame(2.0))  # the camera is ahead (overload): frame 2 waits
        assert runner.latest() is None, "nothing processed yet"
        module.release.release()
        assert until(lambda: runner.latest() is not None)
        kept, seq, overlay = runner.latest()
        assert kept.stamp == 1.0 and seq == 1 and overlay == [1.0]
        assert module.entered.acquire(timeout=5)
        module.release.release()
        assert until(lambda: runner.latest()[0].stamp == 2.0)
        assert runner.latest()[1:] == (2, [2.0])
    finally:
        module.release.release()
        runner.stop()


def test_the_window_lapses_and_nothing_more_is_kept():
    clock = Clock()
    runner = started(Module(), clock=clock)
    try:
        runner.watch(5.0)
        runner.submit(frame())
        assert until(lambda: runner.latest() is not None)
        clock.now += 6.0
        runner.submit(frame())
        assert until(lambda: runner.processed == 2)
        assert runner.latest() is None
    finally:
        runner.stop()


def test_unwatch_drops_the_reference_at_once_and_a_frame_in_flight_is_not_kept():
    module = Gated()
    runner = started(module, overlay=module.overlay)
    try:
        runner.watch(5.0)
        runner.submit(frame(1.0))
        assert module.entered.acquire(timeout=5)
        runner.unwatch()  # viewing stopped while frame 1 is in the pipeline
        module.release.release()
        assert until(lambda: runner.processed == 1)
        assert runner.latest() is None
    finally:
        module.release.release()
        runner.stop()


def test_the_overlay_is_not_computed_while_nobody_watches():
    calls = []
    runner = started(Module(), overlay=lambda: calls.append(1))
    try:
        runner.submit(frame())
        assert until(lambda: runner.processed == 1)
        assert calls == []
    finally:
        runner.stop()


def test_the_pipeline_still_gets_every_frame_unchanged():
    seen = []

    class Recording(Module):
        def process(self, f):
            seen.append(f)

    runner = LatestFrameRunner(Recording())
    runner.start()
    try:
        runner.watch(5.0)
        sent = frame(3.0)
        runner.submit(sent)
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        runner.stop()
    assert seen == [sent] and seen[0] is sent


def test_the_cost_on_the_submit_path_with_no_watcher_is_a_comparison():
    runner = LatestFrameRunner(Module())
    f = frame()
    runs = 20000
    start = time.perf_counter()
    for _ in range(runs):
        runner.submit(f)
    per_call_us = (time.perf_counter() - start) / runs * 1e6
    assert per_call_us < 50  # generous bound; the measured figure is in the slice report
