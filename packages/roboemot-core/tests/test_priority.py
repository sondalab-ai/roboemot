"""Lower OS priority for the listed background threads only (SPEC 13, runtime.background_*)."""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest
from roboemot import priority
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision.runner import LatestFrameRunner
from roboemot.vision.types import Frame

DEFAULTS = RoboEmotConfig().runtime


@pytest.fixture
def fake(monkeypatch):
    """Linux-like seams: every ``renice`` is recorded, thread ids come from a counter."""
    calls: list[tuple[int, int]] = []
    state = {"tasks": {1, 2}, "next": 100}
    monkeypatch.setattr(priority, "is_supported", lambda: True)
    monkeypatch.setattr(priority, "renice", lambda tid, inc: calls.append((tid, inc)))
    monkeypatch.setattr(priority, "tasks", lambda: set(state["tasks"]))
    monkeypatch.setattr(priority, "native_id", threading.get_native_id)
    priority.configure(DEFAULTS.background_nice, DEFAULTS.background_threads)
    yield calls, state
    priority.configure(0, ())


def run_in_thread(name: str) -> int:
    out: list[int] = []

    def body():
        priority.lower_current()
        out.append(threading.get_native_id())

    t = threading.Thread(target=body, name=name)
    t.start()
    t.join()
    return out[0]


def test_defaults_list_vision_and_not_the_real_time_threads():
    assert DEFAULTS.background_nice == 5
    assert {"roboemot-vision", "vision-models", "roboemot-camera"} <= set(
        DEFAULTS.background_threads
    )
    real_time = {
        "roboemot-audio",
        "roboemot-listener",
        "roboemot-grammar",
        "roboemot-keywords",
        "speech-output",
        "motion-loop",
        "MainThread",
    }
    assert real_time.isdisjoint(DEFAULTS.background_threads)


def test_only_the_listed_threads_are_lowered(fake):
    calls, _ = fake
    vision = run_in_thread("roboemot-vision")
    run_in_thread("roboemot-listener")
    run_in_thread("roboemot-audio")
    assert calls == [(vision, 5)]
    snap = priority.snapshot()
    assert snap["lowered"] == {"roboemot-vision": 1} and snap["errors"] == 0 and snap["supported"]


def test_the_vision_runner_thread_lowers_itself(fake):
    calls, _ = fake

    class Module:
        name = "m"

        def process(self, frame):
            pass

    runner = LatestFrameRunner(Module())
    runner.start()
    try:
        runner.submit(Frame(np.zeros((2, 2, 3), np.uint8), 0.0))
        deadline = time.monotonic() + 5
        while runner.processed < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        runner.stop()
    assert priority.snapshot()["lowered"] == {"roboemot-vision": 1}
    assert len(calls) == 1


def test_threads_created_while_the_models_load_are_lowered_under_their_role(fake):
    calls, state = fake
    with priority.lower_new_threads(priority.VISION_MODELS):
        state["tasks"] |= {7, 8}  # the ONNX Runtime pools of two sessions
    assert calls == [(7, 5), (8, 5)]
    with priority.lower_new_threads("not-listed"):
        state["tasks"] |= {9}
    assert calls == [(7, 5), (8, 5)]
    assert priority.snapshot()["lowered"] == {"vision-models": 2}


def test_no_op_off_linux_or_at_zero(fake, monkeypatch):
    calls, state = fake
    monkeypatch.setattr(priority, "is_supported", lambda: False)
    run_in_thread("roboemot-vision")
    with priority.lower_new_threads(priority.VISION_MODELS):
        state["tasks"] |= {7}
    monkeypatch.setattr(priority, "is_supported", lambda: True)
    priority.configure(0, DEFAULTS.background_threads)
    run_in_thread("roboemot-vision")
    assert calls == []


def test_a_refused_renice_is_counted_not_raised(fake, monkeypatch):
    def refuse(tid, inc):
        raise PermissionError("no")

    monkeypatch.setattr(priority, "renice", refuse)
    run_in_thread("roboemot-vision")
    assert priority.snapshot()["errors"] == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="per-thread nice is Linux only")
def test_linux_an_unprivileged_thread_can_raise_its_own_nice():
    """The real call (CI runs Linux, as the robot): the thread's nice goes up by the increment,
    the main thread's does not."""
    main = os.getpriority(os.PRIO_PROCESS, threading.get_native_id())
    priority.configure(3, ["probe"])
    try:
        seen: list[int] = []

        def body():
            priority.lower_current()
            seen.append(os.getpriority(os.PRIO_PROCESS, threading.get_native_id()))

        t = threading.Thread(target=body, name="probe")
        t.start()
        t.join()
        assert seen == [min(priority.NICE_MAX, main + 3)]
        assert os.getpriority(os.PRIO_PROCESS, threading.get_native_id()) == main
        assert priority.snapshot()["lowered"] == {"probe": 1}
    finally:
        priority.configure(0, ())
