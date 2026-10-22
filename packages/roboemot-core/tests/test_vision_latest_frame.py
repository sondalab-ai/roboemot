"""SPEC 4A, 11A: the runner's latest-frame handoff for the Live tab's preview. Nothing is kept while
nobody watches; a watch window keeps one reference (no copy) and lapses by itself."""

from __future__ import annotations

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


def test_nothing_is_kept_without_a_watcher():
    runner = LatestFrameRunner(Module(), clock=Clock())
    runner.submit(frame())
    assert runner.latest() is None


def test_a_watched_runner_keeps_the_newest_frame_itself_not_a_copy():
    clock = Clock()
    runner = LatestFrameRunner(Module(), clock=clock)
    runner.watch(5.0)
    first, second = frame(1.0), frame(2.0)
    runner.submit(first)
    runner.submit(second)
    kept, seq = runner.latest()
    assert kept is second and seq == 2


def test_the_window_lapses_and_the_next_submit_drops_the_reference():
    clock = Clock()
    runner = LatestFrameRunner(Module(), clock=clock)
    runner.watch(5.0)
    runner.submit(frame())
    assert runner.latest() is not None
    clock.now += 6.0
    runner.submit(frame())
    assert runner.latest() is None


def test_unwatch_drops_the_reference_at_once():
    runner = LatestFrameRunner(Module(), clock=Clock())
    runner.watch(5.0)
    runner.submit(frame())
    runner.unwatch()
    assert runner.latest() is None
    runner.submit(frame())
    assert runner.latest() is None


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
