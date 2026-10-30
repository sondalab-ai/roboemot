"""SPEC 4A, 11: ``start_all`` registers each part's stop as it starts it, so a failure part-way
tears down exactly what already started, in reverse order."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from roboemot.config import ConfigStack
from roboemot.runtime import RuntimeHost
from roboemot.runtime.startup import bind_barge_in, offline_cue, start_all


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
        tracker=part("tracker"), runner=part("runner"),
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
        "stop reflexes", "stop circ", "stop runner", "stop tracker",
    ]  # fmt: skip


def test_offline_cue_speaks_on_the_host_output_and_is_muted_asleep():
    said: list[dict] = []
    host, stack = RuntimeHost(), ConfigStack()
    p = SimpleNamespace(
        tracker=SimpleNamespace(active=None),
        circ=None,
        runner=SimpleNamespace(
            engine=SimpleNamespace(state=lambda now: SimpleNamespace(label="joy", intensity=0.4))
        ),
        sleep=SimpleNamespace(asleep=False, paused=False),
    )
    cue = offline_cue(host, stack, p)
    cue.say("no output yet", "it")  # the output does not exist: silent, no exception
    host.speech_output = SimpleNamespace(
        say=lambda text, **kw: said.append({"text": text, **kw}),
        backend=SimpleNamespace(name="piper"),
    )
    cue.say("ciao", "it")
    assert [s["text"] for s in said] == ["ciao"] and said[0]["mood"] == "joy"
    assert said[0]["register"] == stack.effective().register.params
    p.sleep.asleep = True
    cue.say("asleep", "it")
    p.sleep.asleep, p.sleep.paused = False, True
    cue.say("paused", "it")
    assert len(said) == 1


def test_barge_in_ducks_the_speech_output_and_its_cut_stops_it_first():
    """SPEC 8 stage 8: the speech output applies barge-in's duck; the cut is bound in order."""
    bound = []
    bargein = SimpleNamespace(duck=object(), bind=lambda *actions: bound.append(actions))
    output = SimpleNamespace(duck=None, stop=lambda: None)
    conv = SimpleNamespace(interrupt=lambda: None, mark_interrupting=lambda: None)
    listener = SimpleNamespace(resume=lambda samples, stamp: None)
    bind_barge_in(bargein, output, conv, listener)
    assert output.duck is bargein.duck
    assert bound == [(output.stop, conv.interrupt, listener.resume, conv.mark_interrupting)]
