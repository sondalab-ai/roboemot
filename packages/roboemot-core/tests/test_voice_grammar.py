"""Grammar pass of wake and keyword spotting with a fake recognizer (SPEC 8 stage 3)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.voice.asr import GrammarUnsupported, HeardWord, VoskASR
from roboemot.voice.fastpath import FastPath
from roboemot.voice.grammar import GrammarSpotter
from roboemot.voice.keywords import KeywordSpotter, fold
from roboemot.voice.robot_name import RobotName
from test_voice_listener import Rig

RATE = 16000
AUDIO = np.zeros(RATE, np.float32)
KEYWORDS = {
    "it": {
        "stop": ["fermati"],
        "yes": ["certo"],
        "no": ["no"],
        "greeting": ["ciao"],
        "wake_phrases": ["ehi robot"],
        "name_wake_templates": ["ciao {name}"],
        "name_aliases": ["reati"],
    }
}


def config(**grammar):
    return RoboEmotConfig.model_validate(
        {"voice": {"keywords": KEYWORDS, "default_name": "Reachy", "grammar": grammar}}
    )


WORD_S = 0.3  # each scripted word lasts this long


def heard(pairs, gaps=None):
    """``(word, conf)`` pairs as adjacent ``HeardWord``s; ``gaps[i]`` seconds before word i+1."""
    out, t = [], 0.0
    for i, (word, conf) in enumerate(pairs):
        out.append(HeardWord(word, conf, t, t + WORD_S))
        t += WORD_S + ((gaps or {}).get(i, 0.0))
    return out


class Decoder:
    """Scripted Vosk grammar decode: returns ``words`` and records each grammar it got."""

    def __init__(self, words=(), result=True, gaps=None):
        self.words = heard(words, gaps)
        self.result = result
        self.grammars: list[list[str]] = []

    def __call__(self, language, audio, phrases):
        self.grammars.append(list(phrases))
        if self.result is None:
            return None
        if isinstance(self.result, Exception):
            raise self.result
        return list(self.words)


def vocab(oov=()):
    """A vocabulary check that lacks the ``oov`` words."""
    return lambda language, words: [w for w in words if w in oov]


def build(cfg, decoder, platform=None, oov=()):
    names = RobotName(platform, vocab(oov))
    names.configure(cfg.voice)
    grammar = GrammarSpotter(cfg.voice.grammar, decoder, vocab(oov), RATE)
    fast = FastPath(cfg.voice, lambda *a: None, names=names, grammar=grammar)
    return fast, grammar, names


def end(fast, text, language="it", audio=AUDIO):
    fast.begin(0.0, 0.0, language)
    return fast.end(
        text, AUDIO, RATE, last_speech_stamp=1.0, endpoint_stamp=1.5, grammar_audio=audio
    )


def test_the_grammar_is_built_from_config_name_and_free_text_words():
    dec = Decoder()
    fast, _, _ = build(config(), dec, platform="Neve")
    end(fast, "che cadeva")
    (grammar,) = dec.grammars
    assert set(grammar) == {
        "fermati", "certo", "no", "ciao", "ehi robot", "ciao neve", "neve", "che", "cadeva",
    }  # fmt: skip
    assert "reati" not in grammar  # the robot is renamed: the Reachy spellings do not apply


def test_phrases_keep_their_accents_for_the_vocabulary():
    cfg = RoboEmotConfig.model_validate({"voice": {"keywords": {"it": {"yes": ["Sì", "così"]}}}})
    dec = Decoder()
    fast, _, _ = build(cfg, dec)
    end(fast, "c'è così")
    assert {"sì", "così", "c'è"} <= set(dec.grammars[0])
    assert fold("Ciao, NEVE!") == "ciao neve"


def test_out_of_vocabulary_phrases_are_skipped_and_counted_never_matched():
    dec = Decoder([("ehi", 1.0), ("robot", 1.0)])
    fast, grammar, _ = build(config(), dec, oov={"robot", "cadeva"})
    result = end(fast, "che cadeva")
    assert "ehi robot" not in dec.grammars[0] and "cadeva" not in dec.grammars[0]
    assert not result.wake_word
    assert grammar.snapshot()["out_of_vocab"] == {"it": 1}


def test_a_confident_grammar_hit_rescues_the_wake_phrase_free_text_missed():
    dec = Decoder([("ciao", 1.0), ("neve", 0.97)])
    fast, grammar, _ = build(config(), dec, platform="Neve")
    result = end(fast, "cadeva")
    assert result.wake_word and "wake" in result.keywords
    snap = grammar.snapshot()
    assert snap["rescued"]["wake"] == 1 and snap["hits"]["greeting"] == 1
    assert snap["confidence"]["accepted"]["min"] == pytest.approx(0.97)


def test_below_the_confidence_threshold_the_free_text_decides():
    dec = Decoder([("ciao", 1.0), ("neve", 0.6)])
    fast, grammar, _ = build(config(), dec, platform="Neve")
    result = end(fast, "cadeva")
    assert not result.wake_word
    snap = grammar.snapshot()
    assert snap["rejected"]["wake"] == 1 and snap["hits"]["wake"] == 0
    fast.apply_config(config(wake_min_word_conf=0.5).voice)  # hot reload of the threshold
    assert end(fast, "cadeva").wake_word


def test_keyword_threshold_is_separate_from_the_wake_threshold():
    dec = Decoder([("certo", 0.8)])
    fast, _, _ = build(config(keyword_min_word_conf=0.85), dec)
    assert end(fast, "certo").answer == "yes"  # the free text found it anyway
    assert "yes" not in end(fast, "cervo").keywords
    fast.apply_config(config(keyword_min_word_conf=0.75).voice)
    result = end(fast, "cervo")
    assert "yes" in result.keywords and result.answer == "yes"


def test_unknown_filler_and_partial_phrases_do_not_match():
    dec = Decoder([("ciao", 1.0), ("[unk]", 1.0), ("neve", 1.0)])
    fast, _, _ = build(config(), dec, platform="Neve")
    result = end(fast, "cadeva per favore dimmi qualcosa adesso")
    assert not result.wake_word  # "ciao [unk] neve" is no "ciao neve"; a long text: no bare name


def test_the_grammar_rebuilds_when_the_robot_name_changes():
    dec = Decoder()
    fast, grammar, names = build(config(), dec)
    end(fast, "che")
    assert "ciao neve" not in dec.grammars[-1]
    builds = grammar.snapshot()["builds"]
    names.set_platform("Neve")
    end(fast, "che")
    assert "ciao neve" in dec.grammars[-1]
    assert grammar.snapshot()["builds"] == builds + 1
    end(fast, "che")
    assert grammar.snapshot()["builds"] == builds + 1  # unchanged inputs: no rebuild


def test_the_grammar_rebuilds_on_config_hot_reload():
    dec = Decoder()
    fast, grammar, _ = build(config(), dec)
    end(fast, "che")
    reloaded = {"it": {**KEYWORDS["it"], "stop": ["fermati", "basta"]}}
    fast.apply_config(RoboEmotConfig.model_validate({"voice": {"keywords": reloaded}}).voice)
    end(fast, "che")
    assert "basta" in dec.grammars[-1]
    assert grammar.snapshot()["builds"] == 2


def test_without_competitor_words_there_is_no_pass():
    dec = Decoder([("fermati", 1.0)])
    fast, grammar, _ = build(config(), dec, oov={"zzz"})
    result = end(fast, "zzz")
    assert dec.grammars == [] and "stop" not in result.keywords
    assert grammar.snapshot()["skipped"]["no_competitors"] == 1


@pytest.mark.parametrize("result,reason", [(None, "no_model"), (RuntimeError(), "errors")])
def test_a_failed_decode_falls_back_to_the_free_text(result, reason):
    dec = Decoder(result=result)
    fast, grammar, _ = build(config(), dec)
    out = end(fast, "ehi robot")
    assert out.wake_word  # the free-text match
    assert grammar.snapshot()["skipped"][reason] == 1


def test_no_vocabulary_yet_means_no_pass():
    cfg = config()
    dec = Decoder()
    grammar = GrammarSpotter(cfg.voice.grammar, dec, lambda language, words: None, RATE)
    spotter = KeywordSpotter(cfg.voice.keywords, None)
    assert grammar.check(spotter, "it", AUDIO, "ciao") == frozenset()
    assert dec.grammars == [] and grammar.snapshot()["skipped"]["no_vocabulary"] == 1


def test_disabled_grammar_never_decodes():
    dec = Decoder([("fermati", 1.0)])
    fast, grammar, _ = build(config(enabled=False), dec)
    assert "stop" not in end(fast, "che").keywords
    assert dec.grammars == [] and grammar.snapshot()["passes"] == 0


def test_free_text_hits_the_grammar_did_not_confirm_count_as_missed():
    dec = Decoder([("[unk]", 1.0)])
    fast, grammar, _ = build(config(), dec)
    end(fast, "fermati")
    assert grammar.snapshot()["missed"]["stop"] == 1


def test_more_free_text_words_than_the_cap_skip_the_pass():
    dec = Decoder([("fermati", 1.0)])
    fast, grammar, _ = build(config(max_competitor_words=8), dec)
    assert "stop" not in end(fast, "uno due tre quattro cinque sei sette otto nove").keywords
    assert dec.grammars == [] and grammar.snapshot()["skipped"]["too_many_words"] == 1
    end(fast, "uno due tre quattro cinque sei sette otto")
    assert len(dec.grammars) == 1


def test_an_utterance_longer_than_max_audio_skips_the_pass():
    dec = Decoder([("fermati", 1.0)])
    fast, grammar, _ = build(config(max_audio_s=1.0), dec)
    long_audio = np.zeros(2 * RATE, np.float32)
    assert "stop" not in end(fast, "che", audio=long_audio).keywords
    assert dec.grammars == [] and grammar.snapshot()["skipped"]["too_long"] == 1


def test_a_model_without_a_runtime_graph_skips_the_pass():
    dec = Decoder(result=GrammarUnsupported("it"))
    fast, grammar, _ = build(config(), dec)
    assert end(fast, "ehi robot").wake_word  # the free-text match stands
    assert grammar.snapshot()["skipped"]["no_grammar_graph"] == 1


def test_a_wake_phrase_whose_words_the_free_text_heard_is_not_rescued():
    dec = Decoder([("ehi", 1.0), ("robot", 1.0), ("sono", 1.0), ("utili", 1.0)])
    fast, grammar, _ = build(config(), dec)
    result = end(fast, "i robot sono utili")
    assert not result.wake_word
    snap = grammar.snapshot()
    assert snap["guarded"]["wake"] == 1 and snap["hits"]["wake"] == 0


def test_rescue_wake_off_adds_no_wake_but_keeps_keywords():
    dec = Decoder([("ciao", 1.0), ("neve", 1.0), ("fermati", 1.0)])
    fast, grammar, _ = build(config(rescue_wake=False), dec, platform="Neve")
    result = end(fast, "cadeva fermarti")
    assert not result.wake_word and "stop" in result.keywords
    assert grammar.snapshot()["guarded"]["wake"] == 1


def test_phrase_words_too_far_apart_in_the_audio_do_not_match():
    dec = Decoder([("ciao", 1.0), ("neve", 1.0)], gaps={0: 0.4})
    fast, grammar, _ = build(config(max_word_gap_s=0.15), dec, platform="Neve")
    text = "cadeva per favore dimmi adesso"  # long: the bare name alone does not wake
    result = end(fast, text)
    assert not result.wake_word and "name" in result.keywords
    assert grammar.snapshot()["gapped"]["wake"] == 1
    fast.apply_config(config(max_word_gap_s=0.5).voice)
    assert end(fast, text).wake_word


def test_a_phrase_with_an_apostrophe_matches():
    cfg = RoboEmotConfig.model_validate({"voice": {"keywords": {"it": {"yes": ["d'accordo"]}}}})
    dec = Decoder([("d'accordo", 1.0)])
    fast, grammar, _ = build(cfg, dec)
    result = end(fast, "da cordo")
    assert "yes" in result.keywords and result.answer == "yes"
    assert grammar.snapshot()["rescued"]["yes"] == 1


def test_a_grammar_yes_with_a_free_text_no_gives_no_answer():
    dec = Decoder([("certo", 1.0)])
    fast, _, _ = build(config(), dec)
    result = end(fast, "no")
    assert {"yes", "no"} <= result.keywords and result.answer is None


def test_the_grammar_builds_once_the_model_vocabulary_is_available():
    cfg = config()
    dec = Decoder([("fermati", 1.0)])
    loaded = {"it": False}

    def missing(language, words):
        return [] if loaded[language] else None

    grammar = GrammarSpotter(cfg.voice.grammar, dec, missing, RATE)
    spotter = KeywordSpotter(cfg.voice.keywords, None)
    assert grammar.check(spotter, "it", AUDIO, "che") == frozenset()
    assert grammar.snapshot()["skipped"]["no_vocabulary"] == 1 and grammar.snapshot()["builds"] == 0
    loaded["it"] = True
    assert grammar.check(spotter, "it", AUDIO, "che") == {"stop"}
    snap = grammar.snapshot()
    assert snap["builds"] == 1 and snap["passes"] == 1


class FakeVosk:
    """Stands in for the vosk module: models load from any folder, decodes return one word."""

    def __init__(self):
        self.grammars = []

    def SetLogLevel(self, level):  # noqa: N802 (the vosk API)
        pass

    def Model(self, path):  # noqa: N802
        return SimpleNamespace(path=path)

    def KaldiRecognizer(self, model, rate, grammar=None):  # noqa: N802
        self.grammars.append(grammar)
        result = {"result": [{"word": "così", "conf": 1.0, "start": 0.1, "end": 0.4}]}
        return SimpleNamespace(
            SetWords=lambda on: None,
            AcceptWaveform=lambda data: True,
            FinalResult=lambda: json.dumps(result),
        )


def test_vosk_grammar_needs_a_runtime_graph(tmp_path, monkeypatch):
    fake = FakeVosk()
    monkeypatch.setitem(sys.modules, "vosk", fake)
    small, big = tmp_path / "small", tmp_path / "big"
    (small / "graph").mkdir(parents=True)
    (big / "graph").mkdir(parents=True)
    for name in ("HCLr.fst", "Gr.fst"):
        (small / "graph" / name).write_bytes(b"")
    (big / "graph" / "HCLG.fst").write_bytes(b"")
    asr = VoskASR({"it": small, "en": big}, RATE)
    assert asr.grammar_words("it", AUDIO, ["così"]) is None  # not loaded yet
    asr.load(["it", "en"])
    assert asr.grammar_words("it", AUDIO, ["così"]) == [HeardWord("così", 1.0, 0.1, 0.4)]
    assert fake.grammars[-1] == '["così", "[unk]"]'  # accents reach Vosk unescaped
    with pytest.raises(GrammarUnsupported):
        asr.grammar_words("en", AUDIO, ["stop"])


def test_listener_runs_the_pass_on_the_utterance_audio_and_reports_counts_only():
    cfg = config()
    dec = Decoder([("ciao", 1.0), ("neve", 1.0)])
    names = RobotName("Neve", vocab())
    rig = Rig(
        ["cadeva"],
        cfg=cfg,
        names=names,
        grammar=GrammarSpotter(cfg.voice.grammar, dec, vocab(), RATE),
    )
    rig.say(1.0)
    (u,) = rig.utterances
    assert u.wake_word and u.addressed
    snap = rig.listener.snapshot()["grammar"]
    assert snap["passes"] == 1 and snap["phrases"]["it"] > 0
    assert snap["cost"]["max_rtf"] >= 0
    dumped = json.dumps(snap)
    assert "cadeva" not in dumped and "neve" not in dumped and "ciao" not in dumped
