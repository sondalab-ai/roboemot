"""SPEC 4A, 11: ``start_all`` registers each part's stop as it starts it, so a failure part-way
tears down exactly what already started, in reverse order."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from roboemot.runtime import RuntimeHost
from roboemot.runtime.startup import start_all


class Part:
    """A part that logs its start and stop (and the other methods the cleanup registers)."""

    def __init__(self, name: str, log: list[str], fail_start: bool = False) -> None:
        self.name, self.log, self.fail_start = name, log, fail_start
        self.running = False

    def start(self) -> None:
        self.log.append(f"start {self.name}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} failed")

    def stop(self) -> None:
        self.log.append(f"stop {self.name}")

    close = flush = stop


def test_a_failure_while_starting_stops_what_already_started_in_reverse_order():
    log: list[str] = []

    def part(name: str, fail: bool = False) -> Part:
        return Part(name, log, fail)

    p = SimpleNamespace(
        presence=part("presence"), tracker=part("tracker"), runner=part("runner"),
        circ=part("circ"), reflexes=part("reflexes"), loop=part("loop"),
        network=part("network"), worker=part("worker"),
        perception=part("perception", fail=True), audio=part("audio"),
        reflex_audio=False, listener=None, speaker=None, voice_close=None,
        light_reader=part("light_reader"), plugin=SimpleNamespace(sources=[]),
    )  # fmt: skip
    stack = SimpleNamespace(
        start_watcher=lambda: SimpleNamespace(set=lambda: log.append("stop watcher")),
        effective=lambda: SimpleNamespace(runtime=SimpleNamespace(gc_freeze=False)),
    )
    host = RuntimeHost()
    with pytest.raises(RuntimeError, match="perception failed"), ExitStack() as cleanup:
        start_all(host, stack, object(), p, cleanup)
    assert log == [
        "start loop", "start network", "start worker", "start perception",
        "stop perception",
        "stop worker", "stop network", "stop loop", "stop watcher",
        "stop reflexes", "stop circ", "stop runner", "stop tracker", "stop presence",
    ]  # fmt: skip
