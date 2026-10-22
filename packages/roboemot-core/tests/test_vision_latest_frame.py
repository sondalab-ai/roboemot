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
    """Holds each frame until released; the overlay names the frame it was computed on. Faces
    are detected on every frame unless its stamp is in ``skip``."""

    def __init__(self, skip=()):
        self.entered, self.release = threading.Semaphore(0), threading.Semaphore(0)
        self.current, self.skip = None, set(skip)

    def process(self, f):
        self.entered.release()
        self.release.acquire()
        if f.stamp not in self.skip:
            self.current = f.stamp
        return f.stamp not in self.skip

    def overlay(self, detected):
        return [self.current] if detected else None


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


def test_a_frame_without_a_detection_does_not_replace_the_kept_one():
    module = Gated(skip={2.0})
    runner = started(module, overlay=module.overlay)
    try:
        runner.watch(5.0, hold_s=5.0)
        for stamp in (1.0, 2.0):
            runner.submit(frame(stamp))
            assert module.entered.acquire(timeout=5)
            module.release.release()
            assert until(lambda n=stamp: runner.processed == n)
        kept, seq, overlay = runner.latest()
        assert kept.stamp == 1.0 and seq == 1 and overlay == [1.0]  # not frame 2 with old boxes
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
    runner = started(Module(), overlay=lambda report: calls.append(1))
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


def test_with_rare_detection_a_newer_frame_without_boxes_follows_after_the_hold():
    clock = Clock()
    runner = LatestFrameRunner(
        Module(), clock=clock, overlay=lambda report: ["box"] if detected[0] else None
    )
    detected = [True]
    runner.watch(60.0, hold_s=1.0)
    runner.submit(frame(1.0))
    runner.run_pending()
    assert runner.latest()[0].stamp == 1.0 and runner.latest().overlay == ["box"]
    detected[0] = False
    clock.now += 0.5
    runner.submit(frame(2.0))
    runner.run_pending()
    assert runner.latest()[0].stamp == 1.0, "a detection frame is held for hold_s"
    clock.now += 0.6
    runner.submit(frame(3.0))
    runner.run_pending()
    kept = runner.latest()
    assert kept.frame.stamp == 3.0 and kept.overlay == []  # live picture, no stale boxes


def test_camera_age_follows_the_camera_not_the_kept_frame():
    clock = Clock()
    runner = LatestFrameRunner(Module(), clock=clock, overlay=lambda report: ["box"])
    assert runner.camera_age_s() is None
    runner.watch(60.0, hold_s=100.0)
    runner.submit(frame(1.0))
    runner.run_pending()
    clock.now += 50.0
    runner.submit(frame(2.0))  # not processed yet (a slow detection): the camera is alive
    assert runner.latest().frame.stamp == 1.0 and runner.camera_age_s() == 0.0
    clock.now += 7.0
    assert runner.camera_age_s() == 7.0
