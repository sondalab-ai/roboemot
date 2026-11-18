"""The stricter conversation start (SPEC 8, Conversation start): with the skip tool, each
realtime turn's instructions say whether a conversation with the robot is in progress. With
none in progress the strict start rule applies (only words clearly meant for the robot are
answered); the robot's recent answer, a waiting confirmation, the wake word or the name open
it. Turns and skips are counted by state. The server's judgment itself is not tested here: it
needs the real server (the robot run checks it)."""

from __future__ import annotations

import dataclasses
import math
import time

import pytest
from roboemot.config import ConfigStack
from roboemot.config.schema import RegisterParams, RoboEmotConfig
from roboemot.conversation.worker import _Turn
from roboemot.llm import prompt as prompt_module
from roboemot.llm.prompt import (
    ADDRESSED,
    CONVERSATION_STATES,
    IN_PROGRESS,
    NO_CONVERSATION,
    OPEN_PROMPT,
    SKIP_PROMPT,
    START_PROMPT,
    TONE_PROMPT,
    open_prompt,
    realtime_instructions,
    skip_prompt,
    start_prompt,
    tone_prompt,
)
from roboemot.realtime.turn import SKIP_TOOL
from roboemot.voice.fastpath import FastPath
from roboemot.voice.listener import Utterance
from roboemot.voice.robot_name import RobotName
from rt_rig import build_rt, wait_for
from test_conversation_worker import utt
from test_voice_listener import RATE, voice
from turn_server import TurnServer

SKIP = {"skip_tool": True}
SPAN_S = 30.0


@pytest.fixture()
def rigs():
    made = []

    def make(server, **kw):
        rig = build_rt(server, **kw)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def instructions(server, n: int) -> str:
    return server.of("response.create")[n]["response"]["instructions"]


def submit(rig, utterance) -> None:
    n = rig.worker.snapshot().get("turns", 0)
    assert rig.worker.submit(utterance)
    assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n + 1, 10.0)


def zero(**counts: int) -> dict[str, int]:
    return {s: counts.get(s, 0) for s in CONVERSATION_STATES}


# ------------------------------------------------------------------ the turn's state


def test_with_no_conversation_the_turn_carries_the_strict_start_rule(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt={**SKIP, "conversation_span_s": SPAN_S})
        rig.turn("eh, va bene, sì")
        text = instructions(srv, 0)
        assert start_prompt() in text and open_prompt() not in text
        assert text.index(skip_prompt()) < text.index(start_prompt())
        assert rig.route.snapshot()["turns_by_state"] == zero(none=1)
        update = srv.of("session.update")[-1]["session"]["instructions"]
        assert start_prompt() not in update  # per turn only, never the session's


def test_a_recent_answer_opens_the_conversation_and_the_span_closes_it(rigs):
    with TurnServer(["reply", "reply", "reply"]) as srv:
        rig = rigs(srv, rt={**SKIP, "conversation_span_s": SPAN_S})
        rig.turn("ciao robot")  # none: nothing said yet
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        rig.turn("sì, va bene")  # the robot just answered: in progress
        second = instructions(srv, 1)
        assert open_prompt() in second and start_prompt() not in second
        rig.worker.clock = lambda: time.monotonic() + SPAN_S + 1  # past the span
        rig.turn("okay, facciamo così")
        third = instructions(srv, 2)
        assert start_prompt() in third and open_prompt() not in third
        assert rig.route.snapshot()["turns_by_state"] == zero(none=2, in_progress=1)


def test_the_wake_word_marks_the_turn_addressed(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        submit(rig, dataclasses.replace(utt("ehi robot", stamp=time.monotonic()), wake_word=True))
        text = instructions(srv, 0)
        assert open_prompt() in text and start_prompt() not in text
        assert rig.route.snapshot()["turns_by_state"] == zero(addressed=1)


def heard(text: str, *, server: bool) -> Utterance:
    """The utterance as the listener builds it from the fast path's decision on ``text``, for
    a robot named Neve (every word in vocabulary)."""
    cfg = RoboEmotConfig.model_validate({"voice": {"wake_word": "Neve"}})
    fast = FastPath(cfg.voice, lambda *a: None, names=RobotName(missing=lambda lang, w: []))
    fast.begin(0.0, 0.0, "it")
    ends = {"last_speech_stamp": 0.0, "endpoint_stamp": 0.0}
    result = fast.end(text, voice(seconds=0.5), RATE, **ends, server=server)
    return dataclasses.replace(utt(text), wake_word=result.wake_word, keywords=result.keywords)


@pytest.mark.parametrize("server", [False, True])
def test_the_name_inside_a_long_sentence_does_not_address_the_turn(rigs, server):
    """ "Neve" is also snow: only a short final with the bare name wakes (SPEC 8 stage 3)."""
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        state = lambda u: rig.worker._conversation_state(u, rig.stack.effective(), 0.0)  # noqa: E731
        long = heard("domani sulle montagne qui vicino cade tanta neve dicono tutti", server=server)
        assert len(long.text.split()) == 10 and "name" in long.keywords and not long.wake_word
        assert state(long) == NO_CONVERSATION
        assert state(heard("ciao neve", server=server)) == ADDRESSED


def test_a_waiting_confirmation_counts_as_in_progress(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        rig.worker._confirm = object()  # a confirmation question was asked
        state = rig.worker._conversation_state(utt("sì"), rig.stack.effective(), rig.worker.clock())
        assert state == IN_PROGRESS
        rig.worker._confirm = None
        state = rig.worker._conversation_state(utt("sì"), rig.stack.effective(), rig.worker.clock())
        assert state == NO_CONVERSATION


def test_a_skip_is_counted_in_its_state_and_never_opens_the_conversation(rigs):
    with TurnServer(["tool", "tool"], tool_name=SKIP_TOOL) as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn("eh, va bene, sì")
        rig.turn("okay, facciamo così")
        snap = rig.route.snapshot()
        assert snap["skipped"] == 2
        assert snap["skipped_by_state"] == zero(none=2)
        assert snap["turns_by_state"] == zero(none=2)  # the first skip left no answer
        assert start_prompt() in instructions(srv, 1)


def test_without_the_skip_tool_no_hint_and_no_count(rigs):
    with TurnServer(["reply", "reply"]) as srv:
        rig = rigs(srv)
        rig.turn("eh, va bene, sì")
        rig.turn("okay")
        hints = [start_prompt(), open_prompt()]
        for create in srv.of("response.create"):
            assert all(h not in create["response"]["instructions"] for h in hints)
        snap = rig.route.snapshot()
        assert snap["turns_by_state"] == zero() and snap["skipped_by_state"] == zero()


class Spoken:
    """The speech output as ``_finish`` reads it: which jobs played, and when each ended."""

    def __init__(self, played: dict[int, float]) -> None:
        self._played = played

    def played(self, job: int) -> bool:
        return job in self._played

    def ended_at(self, job: int) -> float | None:
        return self._played.get(job)

    def first_audio_at(self, utterance: int) -> float | None:
        return None


def spoke_at(rig, played: dict[int, float], **flags) -> float:
    """``_spoke_at`` after a turn that queued jobs 1 and 2 ends with ``flags``."""
    worker = rig.worker
    worker._spoke_at = -math.inf
    worker.speech = Spoken(played)
    turn = _Turn(utt("sì"), rig.stack.effective(), "it", RegisterParams(), 1, 3)
    turn.queued = [(1, "Uno."), (2, "Due.")]
    turn.spoken = ["Uno.", "Due."]
    for key, value in flags.items():
        setattr(turn, key, value)
    worker._finish(turn)
    return worker._spoke_at


def test_the_span_starts_when_the_played_audio_ends(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        now = time.monotonic()
        rig.worker.clock = lambda: now
        assert spoke_at(rig, {1: now + 2, 2: now + 9}) == now + 9  # a long answer still plays
        assert spoke_at(rig, {1: now - 5, 2: now - 1}) == now  # it ended before the turn did
        # a barge-in: only the played sentence counts, up to where the stop cut it
        assert spoke_at(rig, {1: now + 3}, stopped=True) == now + 3
        assert spoke_at(rig, {}, stopped=True) == -math.inf  # cut before anything played
        assert spoke_at(rig, {}, interrupted=True, stopped=True) == -math.inf
        assert spoke_at(rig, {1: now + 2, 2: now + 4}, fell_back=True) == -math.inf


def test_a_fallback_turn_leaves_no_conversation_in_progress(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        rig.worker._spoke_at = -math.inf
        turn = _Turn(utt("sì"), rig.stack.effective(), "it", RegisterParams(), 1, 3)
        rig.worker._say_fallback(turn)
        assert turn.fell_back
        rig.worker._finish(turn)
        assert rig.worker._spoke_at == -math.inf


def test_a_reload_that_lengthens_the_span_reopens_it(rigs):
    with TurnServer(["reply", "reply"]) as srv:
        rig = rigs(srv, rt={**SKIP, "conversation_span_s": 1.0})
        rig.turn("ciao robot")
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        rig.worker.clock = lambda: time.monotonic() + SPAN_S / 2  # past the old span
        rig.stack.set("voice.realtime_route.conversation_span_s", SPAN_S, source="test", admin=True)
        rig.turn("sì, va bene")
        assert open_prompt() in instructions(srv, 1)


def test_the_span_follows_a_reload(rigs):
    with TurnServer(["reply", "reply"]) as srv:
        rig = rigs(srv, rt={**SKIP, "conversation_span_s": SPAN_S})
        rig.turn("ciao robot")
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        rig.worker.clock = lambda: time.monotonic() + SPAN_S / 2  # within the old span
        rig.stack.set("voice.realtime_route.conversation_span_s", 1.0, source="test", admin=True)
        rig.turn("sì, va bene")
        assert start_prompt() in instructions(srv, 1)


# ------------------------------------------------------------------ the prompt files


@pytest.mark.parametrize("language", ["it", "en"])
def test_the_hints_are_english_for_each_language(language):
    """The hints are English whatever the turn's language (SPEC 8)."""
    cfg = ConfigStack(
        app={"voice": {"realtime_route": SKIP}, "conversation": {"language": "it"}}
    ).effective()
    strict = start_prompt()
    assert SKIP_TOOL in strict  # the strict rule points at the tool by name
    reg = RegisterParams()
    for state, text in ((NO_CONVERSATION, strict), (IN_PROGRESS, open_prompt())):
        built = realtime_instructions(cfg, reg, language=language, conversation=state)
        assert built.endswith(text)
    assert realtime_instructions(cfg, reg, language=language, conversation=ADDRESSED).endswith(
        open_prompt()
    )
    line = realtime_instructions(
        cfg, reg, language=language, skip_rule=False, conversation=NO_CONVERSATION
    )
    assert strict not in line  # a fixed line carries no hint


def test_an_italian_turn_holds_the_english_rules_and_replies_in_italian():
    """SPEC 8: an ``it`` turn's instructions carry the English tone, skip and start rules,
    and only the reply rule names Italian."""
    rt = {**SKIP, "tone_tool": True}
    cfg = ConfigStack(
        app={"voice": {"realtime_route": rt}, "conversation": {"language": "it"}}
    ).effective()
    text = realtime_instructions(cfg, RegisterParams(), language="it", conversation=NO_CONVERSATION)
    for rule in (tone_prompt(), skip_prompt(), start_prompt()):
        assert rule and rule in text
    assert "Always reply in Italian." in text
    assert "rivolte a te" not in text  # no Italian instruction left


def test_the_instruction_rules_ship_in_english_only():
    files = {p.name for p in prompt_module.PROMPTS_DIR.glob("*.txt")}
    for stem in (SKIP_PROMPT, TONE_PROMPT, START_PROMPT, OPEN_PROMPT):
        assert (prompt_module.PROMPTS_DIR / f"{stem}.en.txt").read_text(encoding="utf-8").strip()
        assert [f for f in files if f.startswith(f"{stem}.")] == [f"{stem}.en.txt"]


@pytest.mark.parametrize("content", [None, "  \n"])
def test_a_missing_or_empty_rule_file_fails_loud(monkeypatch, tmp_path, content):
    """A broken package never offers the tool without its rule: the first use raises."""
    if content is not None:
        (tmp_path / f"{SKIP_PROMPT}.en.txt").write_text(content, encoding="utf-8")
    monkeypatch.setattr(prompt_module, "PROMPTS_DIR", tmp_path)
    prompt_module._prompt_files.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=f"{SKIP_PROMPT}.en.txt"):
            skip_prompt()
        cfg = ConfigStack(app={"voice": {"realtime_route": SKIP}}).effective()
        with pytest.raises(RuntimeError, match=f"{SKIP_PROMPT}.en.txt"):
            realtime_instructions(cfg, RegisterParams())
    finally:
        prompt_module._prompt_files.cache_clear()  # the shipped files again for later tests
