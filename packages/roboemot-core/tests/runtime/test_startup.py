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
    bargein = SimpleNamespace(
        duck=object(),
        bind=lambda *actions, **kw: bound.append((actions, kw)),
        confirm=lambda: None,
    )
    output = SimpleNamespace(duck=None, stop=lambda: None)
    conv = SimpleNamespace(interrupt=lambda: None, mark_interrupting=lambda: None)
    route = SimpleNamespace(note_barge_in=lambda mode, seconds: None)
    audio_in = SimpleNamespace(on_confirm=None, route=route)
    listener = SimpleNamespace(
        resume=lambda samples, stamp: None,
        stream_playback=lambda samples, stamp: False,
        end_playback=lambda: None,
        server_turns=lambda: False,
        playback_job=lambda: None,
        realtime_input=audio_in,
    )
    cut_own = lambda: False  # noqa: E731
    bind_barge_in(bargein, output, conv, listener, cut_own)
    assert output.duck is bargein.duck
    actions = (output.stop, conv.interrupt, listener.resume, conv.mark_interrupting)
    server = {
        "playback": listener.stream_playback,
        "end_playback": listener.end_playback,
        "server_mode": listener.server_turns,
        "playback_job": listener.playback_job,
        "on_stop": route.note_barge_in,  # S3: each stop by turn mode (SPEC 13)
    }
    assert bound == [(actions, {"cut_own": cut_own, **server})]
    # S2: the server's speech_started of a barge-in's stream confirms it
    assert audio_in.on_confirm is bargein.confirm


def test_bind_barge_in_without_a_realtime_input_leaves_the_stops_unreported():
    """Review of #200: no realtime input (no route): the barge-in is bound and its stops go
    nowhere (the voice section keeps its own figures). Every ``RealtimeInput`` has a route; a
    ``getattr`` guard for one without is refused by the hierarchy gate (no duck typing)."""
    audio_in = None
    bound = []
    bargein = SimpleNamespace(
        duck=None, bind=lambda *a, **kw: bound.append(kw), confirm=lambda: None
    )
    listener = SimpleNamespace(
        resume=None, stream_playback=None, end_playback=None, server_turns=None,
        playback_job=None, realtime_input=audio_in,
    )  # fmt: skip
    conv = SimpleNamespace(interrupt=None, mark_interrupting=None)
    bind_barge_in(bargein, SimpleNamespace(duck=None, stop=None), conv, listener)
    assert bound[0]["on_stop"] is None


def test_the_listening_blip_waits_for_no_speech_and_no_answer_in_progress():
    """Review of #147: the cues' ``quiet`` reads the listener (speech in progress) and the
    conversation phase (a turn in flight, its gaps between sentences included, or the robot
    speaking)."""
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.conversation.phase import ConversationPhase
    from roboemot.runtime.startup import build_cues

    speaking = SimpleNamespace(on=False)
    listener = SimpleNamespace(on=False, hearing=lambda: listener.on)
    host = SimpleNamespace(phase=ConversationPhase(lambda now: speaking.on))
    p = SimpleNamespace(
        cfg=RoboEmotConfig(),
        expression=SimpleNamespace(gesture=lambda name: None),
        cue_sounds={},
        listener=listener,
        sleep=SimpleNamespace(asleep=False, paused=False),
    )
    quiet = build_cues(host, None, p, SimpleNamespace(timeline=None))._quiet
    assert quiet()
    listener.on = True  # someone's speech in progress
    assert not quiet()
    listener.on = False
    host.phase.begin_thinking()  # a turn in flight: between its sentences too
    assert not quiet()
    host.phase.end_thinking()
    speaking.on = True  # the answer's audio
    assert not quiet()
    speaking.on = False
    assert quiet()


def test_the_own_sound_cut_cuts_only_sounds_that_are_not_speech():
    """SPEC 7B, 8: someone speaking over a hum cuts it through its runner (counted) and the
    output; over the robot's speech it declines, and barge-in or the mute go on as before."""
    from roboemot.runtime.startup import own_sound_cut

    log = []
    output = SimpleNamespace(raw=True)
    output.cut_raw = lambda: log.append("cut_raw") or output.raw
    soothe = SimpleNamespace(cut_by_speech=lambda now: log.append("soothe"))
    cut = own_sound_cut(output, soothe)
    assert cut() and log == ["cut_raw", "soothe"]
    output.raw, log[:] = False, []
    assert not cut() and log == ["cut_raw"]


def test_the_soothe_tick_hands_the_listeners_speech_to_the_runner():
    from roboemot.runtime.ticks import SootheContext, soothe_tick

    seen = []
    runner = SimpleNamespace(tick=lambda now, idle, state, quiet, speech: seen.append(speech))
    for hearing in (False, True):
        ctx = SootheContext(False, False, 0, False, False, hearing=hearing)
        soothe_tick(runner, 1.0, lambda ctx=ctx: ctx, "awake", False)
    assert seen == [False, True]
