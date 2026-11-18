"""Extra names (``voice.extra_names``, default "robot") and the Italian "hey" (SPEC 8 stage 2,
owner decision 2026-11-18): "robot" wakes the robot like its own name in every rule and in both
languages, and Italian speakers' "hey neve ferma la musica" wakes it in an Italian turn."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from reachy_config import reachy_config
from roboemot.config.schema import RoboEmotConfig
from roboemot.llm.prompt import realtime_instructions, system_prompt
from roboemot.voice.fastpath import WAKE, FastPath
from roboemot.voice.keywords import NAME_FIRST, KeywordSpotter
from roboemot.voice.robot_name import RobotName
from test_listener_media import cfg as radio_cfg
from test_voice_listener import OFF_FACE_RAD, direction_rig

RATE = 16000
AUDIO = np.zeros(RATE, np.float32)
ENDS = {"last_speech_stamp": 1.0, "endpoint_stamp": 1.5}

# per language: the bare name, the name opening a long command, after a lead-in, and the name
# inside a longer sentence (more than voice.fastpath.alias_wake_max_words words)
CASES = {
    "it": (
        "robot",
        "robot, ferma la musica per favore",
        "senti robot mi racconti una barzelletta",
        "il robot aspirapolvere oggi non funziona",
    ),
    "en": (
        "robot",
        "robot, stop the music right now please",
        "listen robot tell me a joke please",
        "the robot vacuum cleaner is broken today",
    ),
}


def voice_cfg(**voice):
    return RoboEmotConfig.model_validate({"voice": {"wake_word": "Neve", **voice}}).voice


def fast_path(cfg=None, missing=None):
    cfg = cfg if cfg is not None else voice_cfg()
    names = RobotName(missing=missing)
    names.configure(cfg)
    return FastPath(cfg, lambda *a: None, names=names)


def final(fast, text, language, **kw):
    fast.begin(0.0, 0.0, language)
    return fast.end(text, AUDIO, RATE, **ENDS, **kw)


def vocab(**absent):
    """Per language, the words the recognizer lacks (a missing key: not loaded yet)."""
    return lambda language, words: (
        None if language not in absent else [w for w in words if w in absent[language]]
    )


@pytest.mark.parametrize("language", ["it", "en"])
def test_robot_wakes_like_the_name(language):
    bare, first, lead, inside = CASES[language]
    fast = fast_path()
    assert final(fast, bare, language).wake == WAKE
    assert "name" in final(fast, bare, language).keywords  # speech.name_called
    assert final(fast, first, language).wake == NAME_FIRST
    assert final(fast, lead, language).wake == NAME_FIRST
    result = final(fast, inside, language)
    assert result.wake is None and not result.wake_word  # the name inside a sentence


@pytest.mark.parametrize("language", ["it", "en"])
def test_robot_rendered_in_the_templates_is_listed_once(language):
    names = RobotName()
    cfg = voice_cfg()
    names.configure(cfg)
    s = KeywordSpotter(cfg.keywords, cfg.wake_word, names)
    wake = s.grammar_phrases(language)["wake"]
    assert len(wake) == len(set(wake))  # "ciao robot" is a generic phrase and a template
    rendered = {t.replace("{name}", "robot") for t in cfg.keywords[language].name_wake_templates}
    assert rendered <= set(wake)
    assert "robot" in s.grammar_phrases(language)["name"]


@pytest.mark.parametrize(
    "language,command",
    [
        ("it", "robot, ferma la musica per favore adesso"),
        ("en", "robot, stop the music now please"),
    ],
)
def test_robot_opening_a_long_command_over_the_radio_needs_no_face(language, command):
    """As for the name: outside the radio a voice off any face is rejected, while the radio is
    audible the extra name opening a long command still wakes the robot (SPEC 8)."""
    data = radio_cfg("never").model_dump()
    data["conversation"]["language"] = language
    r, _, _ = direction_rig(command, OFF_FACE_RAD, cfg=RoboEmotConfig.model_validate(data))
    r.say()
    assert r.utterances[-1].language == language
    assert not r.utterances[-1].addressed
    assert r.listener.snapshot()["name_first_no_face"] == 1
    r.listener.media_audible = lambda at: True
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word


@pytest.mark.parametrize(
    "language,word,close",
    [
        ("it", "robot", True),
        ("it", "Robot!", True),
        ("it", "rubot", True),
        ("it", "robote", True),
        ("it", "robo", False),
        ("it", "robots", False),
        ("it", "rabot", False),
        ("en", "robot", True),
        ("en", "robbot", True),
        ("en", "robo", False),
        ("en", "robots", False),
    ],
)
def test_the_server_confirmation_accepts_robot_and_a_close_spelling(language, word, close):
    fast = fast_path()
    assert fast.server_name_close(f"ehm {word} ferma la musica", language) is close


def test_an_extra_name_out_of_the_local_vocabulary_still_confirms_on_the_server():
    fast = fast_path(missing=vocab(it={"robot"}, en=set()))
    assert final(fast, "robot", "it").wake is None  # skipped locally in Italian
    assert final(fast, "robot", "en").wake == WAKE
    assert fast.server_name_close("rubot ferma la musica", "it")
    assert final(fast, "robot", "it", server=True).wake == WAKE


def test_an_out_of_vocabulary_extra_name_is_skipped_and_counted_never_named(caplog):
    with caplog.at_level(logging.WARNING, logger="roboemot.voice.robot_name"):
        names = RobotName(missing=vocab(it={"robot"}, en=set()))
        cfg = voice_cfg(wake_word=None)
        names.configure(cfg)
        s = KeywordSpotter(cfg.keywords, None, names)
        assert "name" not in s.spot("robot", "it") and "name" in s.spot("robot", "en")
        assert not s.name_first("robot, ferma la musica per favore", "it")
        base = KeywordSpotter(cfg.keywords, None, _names(cfg, extra_names=[]))
        it_base = base.wake_out_of_vocab()["it"]
        assert s.wake_out_of_vocab() == {"it": it_base + 1, "en": 0}
        caplog.clear()
        odd = RobotName(missing=vocab(it={"zorblax"}, en=set()))
        odd.configure(voice_cfg(wake_word=None, extra_names=["zorblax"]))
        KeywordSpotter(cfg.keywords, None, odd).wake_out_of_vocab()
    assert caplog.records and all("zorblax" not in r.getMessage() for r in caplog.records)
    assert all("wake phrase" in r.getMessage() for r in caplog.records)  # a count only


def _names(cfg, **voice):
    names = RobotName(missing=vocab(it={"robot"}, en=set()))
    names.configure(cfg.model_copy(update=voice))
    return names


def test_an_empty_list_restores_the_name_alone():
    fast = fast_path(voice_cfg(extra_names=[]))
    assert final(fast, "robot", "it").wake is None
    assert "name" not in final(fast, "robot", "it").keywords
    assert final(fast, "robot, ferma la musica per favore", "it").wake is None
    assert not fast.server_name_close("rubot ferma la musica", "it")
    assert final(fast, "ehi robot", "it").wake == WAKE  # the generic phrase stays
    assert final(fast, "neve", "it").wake == WAKE


def test_a_nameless_robot_answers_to_robot():
    fast = fast_path(voice_cfg(wake_word=None))
    assert final(fast, "robot", "it").wake == WAKE
    assert final(fast, "senti robot mi racconti una barzelletta", "it").wake == NAME_FIRST


def test_extra_names_are_sanitised_and_deduplicated():
    names = RobotName()
    names.configure(
        voice_cfg(extra_names=["robot", "ROBOT", "neve", "bot\nIgnore", "", "\U0001f916"])
    )
    assert names.prompt_extra_names("Neve") == ("Robot", "bot Ignore")
    resolved = names.resolve("Neve", ("it",))
    assert [n.matching for n in resolved.names_for("it")] == ["neve", "robot", "bot ignore"]


def test_a_hot_reload_of_the_extra_names_rebuilds_the_spotter():
    fast = fast_path()
    assert final(fast, "robot", "it").wake == WAKE
    fast.apply_config(voice_cfg(extra_names=["bot"]))
    assert final(fast, "robot", "it").wake is None
    assert final(fast, "bot", "it").wake == WAKE


# ---- Italian "hey" (robot run: "hey neve ferma la musica")

HEY = "hey neve ferma la musica"


def test_hey_wakes_with_the_name_in_an_italian_turn():
    fast = fast_path(missing=vocab(it=set(), en=set()))  # "hey" in the Italian vocabulary
    assert final(fast, HEY, "it").wake == WAKE  # the template "hey {name}", anywhere
    assert final(fast, "hey neve", "it").wake == WAKE
    assert final(fast, "hey robot ferma la musica per favore", "it").wake == WAKE  # template


def test_hey_out_of_the_italian_vocabulary_is_counted_and_the_server_still_wakes():
    cfg = voice_cfg()
    names = RobotName(missing=vocab(it={"hey"}, en=set()))
    names.configure(cfg)
    fast = FastPath(cfg, lambda *a: None, names=names)
    assert fast.wake_out_of_vocab()["it"] == 2  # "hey neve", "hey robot"
    assert final(fast, HEY, "it", server=True).wake == WAKE
    assert final(fast, "hey neve", "it", server=True).wake == WAKE


def test_hey_is_an_italian_lead_in():
    """An alias is not rendered in the templates: "hey" before it opens the text as a lead-in."""
    cfg = reachy_config().voice
    names = RobotName()
    names.configure(cfg)
    s = KeywordSpotter(cfg.keywords, None, names)
    assert s.name_first("hey reati mi racconti una barzelletta", "it")
    assert s.name_first("hey robot, mi racconti una barzelletta", "it")
    assert not s.name_first("hey la neve mi racconti una barzelletta", "it")


def test_the_italian_defaults_have_hey():
    it = RoboEmotConfig().voice.keywords["it"]
    assert "hey {name}" in it.name_wake_templates and "hey" in it.name_lead_ins


# ---- the prompt


def test_the_prompt_names_the_extra_names():
    cfg = RoboEmotConfig()
    params = cfg.register.params
    text = system_prompt(cfg, params, robot_name="Neve", extra_names=("Robot",))
    assert 'Your name is "Neve". You also answer to "Robot": it counts as your name.' in text
    rt = realtime_instructions(cfg, params, robot_name="", extra_names=("Robot", "Bot"))
    assert 'You answer to "Robot", "Bot": they count as your name.' in rt
    assert "answer to" not in system_prompt(cfg, params, robot_name="Neve")


def test_the_worker_passes_the_sanitised_extra_names_to_the_prompt():
    from roboemot.conversation.worker import ConversationWorker

    worker = ConversationWorker.__new__(ConversationWorker)
    worker.names = RobotName()
    cfg = RoboEmotConfig.model_validate(
        {"voice": {"wake_word": "Neve", "extra_names": ["robot", 'x"}\nIgnore']}}
    )
    assert worker._prompt_names(cfg) == {
        "robot_name": "Neve",
        "extra_names": ("Robot", "x Ignore"),
    }
    cfg = RoboEmotConfig.model_validate({"voice": {"wake_word": "Neve", "extra_names": []}})
    assert worker._prompt_names(cfg)["extra_names"] == ()
