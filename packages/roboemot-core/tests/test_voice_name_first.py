"""The robot's name as the first word wakes it at any length (SPEC 8, owner decision 2026-11-04).

Robot run 2026-11-04 (robot named "Neve", policy ``never``, facing check on): "Neve, mi racconti
una barzelletta?" did not wake the robot, because the bare name woke only a final of at most
``voice.fastpath.alias_wake_max_words`` words ("neve" is also an Italian word). The name opening
the final text, alone or after one lead-in (``name_lead_ins`` or a greeting), now wakes it; the
name elsewhere in a long sentence still does not.
"""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.voice.fastpath import FastPath
from roboemot.voice.keywords import NAME_FIRST, KeywordSpotter
from roboemot.voice.robot_name import RobotName
from test_voice_grammar import Decoder, build, config
from test_voice_grammar import end as grammar_end

RATE = 16000
AUDIO = np.zeros(RATE, np.float32)
ENDS = {"last_speech_stamp": 1.0, "endpoint_stamp": 1.5}
JOKE = "Neve, mi racconti una barzelletta?"
SNOW = "c'è tanta neve oggi in montagna"


def voice_cfg(**voice):
    return RoboEmotConfig.model_validate({"voice": {"wake_word": "Neve", **voice}}).voice


def fast_path(cfg=None, missing=None):
    cfg = cfg if cfg is not None else voice_cfg()
    names = RobotName(missing=missing)
    names.configure(cfg)
    return FastPath(cfg, lambda *a: None, names=names)


def final(fast, text, language="it", **kw):
    fast.begin(0.0, 0.0, language)
    return fast.end(text, AUDIO, RATE, **ENDS, **kw)


def spotter(cfg=None):
    cfg = cfg if cfg is not None else voice_cfg()
    names = RobotName()
    names.configure(cfg)
    return KeywordSpotter(cfg.keywords, cfg.wake_word, names)


@pytest.mark.parametrize(
    "text,first",
    [
        (JOKE, True),
        ("NÈVE mi racconti una barzelletta", True),  # case and accents ignored
        ("ok Neve, mi racconti una barzelletta?", True),  # a lead-in
        ("senti neve mi racconti una barzelletta", True),
        ("buongiorno neve mi racconti una barzelletta", True),  # a greeting is a lead-in too
        ("ok senti neve mi racconti una barzelletta", False),  # one lead-in at most
        ("la neve oggi è bellissima davvero", False),  # an article is no lead-in
        (SNOW, False),
        ("nevera mi racconti", False),  # whole words only
        ("", False),
    ],
)
def test_the_name_opens_the_text(text, first):
    assert spotter().name_first(text, "it") is first


def test_the_lead_ins_are_configuration():
    keywords = {"it": {"greeting": ["ciao"], "name_lead_ins": ["dai"]}}
    custom = spotter(voice_cfg(keywords=keywords))
    assert custom.name_first("dai neve mi racconti una barzelletta", "it")
    assert not custom.name_first("senti neve mi racconti una barzelletta", "it")
    assert not spotter().name_first("dai neve mi racconti una barzelletta", "it")


def test_a_multiword_alias_opening_the_text_counts():
    keywords = {"en": {"name_aliases": ["riyadh she"]}}
    cfg = RoboEmotConfig.model_validate(
        {"voice": {"default_name": "Reachy", "keywords": keywords}}
    ).voice
    s = spotter(cfg)  # no name: the manifest's aliases apply
    assert s.name_first("riyadh she what time is it now please", "en")
    assert not s.name_first("the city of riyadh she is far away from here", "en")


def test_the_name_as_first_word_wakes_a_long_final():
    """Regression: robot run 2026-11-04, a long sentence opening with "Neve" did not wake."""
    fast = fast_path()
    assert len(JOKE.split()) > fast.cfg.fastpath.alias_wake_max_words
    result = final(fast, JOKE)
    assert result.wake_word and "name" in result.keywords
    assert NAME_FIRST not in result.keywords  # a position, never a keyword category
    assert fast.wakes(JOKE)


def test_the_name_inside_a_long_final_does_not_wake():
    fast = fast_path()
    result = final(fast, SNOW)
    assert not result.wake_word and "name" in result.keywords  # still speech.name_called
    assert not fast.wakes(SNOW)


def test_the_short_final_and_template_rules_are_unchanged():
    fast = fast_path()
    assert final(fast, "c'è la neve").wake_word  # name anywhere, short final
    long_template = "domani ciao neve come stai oggi"
    assert final(fast, long_template).wake_word  # a name_wake_templates phrase, any length
    tight = fast_path(voice_cfg(fastpath={"alias_wake_max_words": 2}))
    assert not final(tight, "c'è la neve").wake_word
    assert final(tight, JOKE).wake_word  # first word: the word cap does not apply


def test_the_server_transcript_follows_the_same_rule():
    """The name out of the Vosk vocabulary: only the server's transcript can hear it."""
    fast = fast_path(missing=lambda language, words: list(words))
    assert not final(fast, JOKE).wake_word
    assert final(fast, JOKE, server=True).wake_word
    assert not final(fast, SNOW, server=True).wake_word


# ------------------------------------------------------------------ the grammar pass


def neve_grammar(words):
    dec = Decoder(words)
    fast, grammar, _ = build(config(), dec, platform="Neve")
    return fast, grammar


def test_a_name_the_grammar_pass_hears_first_wakes_a_long_final():
    """The free text lost the name; the grammar pass heard it as the first word."""
    words = [("neve", 1.0), ("mi", 1.0), ("racconti", 1.0), ("una", 1.0), ("barzelletta", 1.0)]
    fast, grammar = neve_grammar(words)
    result = grammar_end(fast, "deve mi racconti una barzelletta")
    assert result.wake_word and "name" in result.keywords
    assert NAME_FIRST not in result.keywords
    assert grammar.snapshot()["name_first"] == 1


def test_a_greeting_before_the_grammar_name_is_a_lead_in():
    words = [("ciao", 1.0), ("mi", 1.0), ("neve", 1.0), ("racconti", 1.0), ("una", 1.0)]
    fast, _ = neve_grammar(words)  # "ciao mi neve": not the template, not first either
    assert not grammar_end(fast, "ciao mi deve racconti una barzelletta").wake_word
    words = [("buongiorno", 1.0), ("neve", 1.0), ("mi", 1.0), ("racconti", 1.0), ("una", 1.0)]
    cfg = RoboEmotConfig.model_validate(
        {"voice": {"default_name": "Reachy", "keywords": {"it": {"greeting": ["buongiorno"]}}}}
    )
    fast, _, _ = build(cfg, Decoder(words), platform="Neve")
    assert grammar_end(fast, "buongiorno deve mi racconti una barzelletta").wake_word


@pytest.mark.parametrize(
    "words",
    [
        [("oggi", 1.0), ("neve", 1.0), ("in", 1.0), ("montagna", 1.0), ("tanta", 1.0)],
        [("[unk]", 1.0), ("neve", 1.0), ("mi", 1.0), ("racconti", 1.0), ("una", 1.0)],
        [("neve", 0.5), ("mi", 1.0), ("racconti", 1.0), ("una", 1.0), ("barzelletta", 1.0)],
    ],
    ids=["not-first", "unknown-word-first", "below-the-floor"],
)
def test_a_grammar_name_that_does_not_open_the_utterance_needs_a_short_final(words):
    fast, grammar = neve_grammar(words)
    text = " ".join(w for w, _ in words if w != "[unk]").replace("neve", "deve") + " ancora"
    assert not grammar_end(fast, text).wake_word
    assert grammar.snapshot()["name_first"] == 0


def test_the_free_texts_name_position_wins_over_a_decode_that_lost_the_words_before_it():
    """The free text has "neve" mid-sentence; a grammar decode that dropped the leading words
    must not move the name to the front."""
    words = [("neve", 1.0), ("oggi", 1.0), ("in", 1.0), ("montagna", 1.0)]
    fast, grammar = neve_grammar(words)
    result = grammar_end(fast, SNOW)
    assert not result.wake_word and "name" in result.keywords
    assert grammar.snapshot()["name_first"] == 0
