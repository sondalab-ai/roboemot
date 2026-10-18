"""The robot's name: platform or config, per-language wake phrases, vocabulary check (SPEC 8)."""

import logging

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.config.schema import KeywordSet, RoboEmotConfig
from roboemot.drivers import MockPoseDriver
from roboemot.voice.fastpath import FastPath
from roboemot.voice.keywords import KeywordSpotter
from roboemot.voice.robot_name import RobotName, display_name

SETS = RoboEmotConfig().voice.keywords
GENERIC = ("ehi robot", "hey robot")


def vocab(**absent):
    """A vocabulary checker: per language, the words it lacks (a missing key: not loaded yet)."""

    def check(language, words):
        if language not in absent:
            return None
        return [w for w in words if w in absent[language]]

    return check


def spot(names, text, language="en", configured=None):
    return KeywordSpotter(SETS, configured, names).spot(text, language)


def test_display_name_turns_separators_into_spaces():
    assert display_name("reachy_mini") == "reachy mini"
    assert display_name("  Neve-2 \n") == "Neve 2"
    assert display_name(None) == ""


def test_templates_are_rendered_with_the_normalised_name():
    names = RobotName("Néve_Due")
    assert "wake" in spot(names, "hey neve due")
    assert "wake" in spot(names, "Ciao, Neve Due!", "it")
    assert "wake" in spot(names, "ehi neve due come stai", "it")
    assert "wake" not in spot(names, "hey neve")  # the whole name is the phrase


def test_without_a_name_only_the_generic_phrases_wake():
    names = RobotName()
    assert spot(names, "hey robot", "en") == {"greeting", "wake"}
    assert "wake" not in spot(names, "hey neve")
    assert "name" not in spot(names, "neve")
    s = KeywordSpotter(SETS, None, names)
    assert s.spot("ciao robot", "it") >= {"wake"}


def test_explicit_config_beats_the_platform_name():
    names = RobotName("reachy_mini")
    assert "wake" in spot(names, "hey bruno", configured="Bruno")
    assert "wake" not in spot(names, "hey reachy mini", configured="Bruno")
    assert "wake" in spot(names, "hey reachy mini")  # unset config: the platform's name
    assert names.display("Bruno") == "Bruno" and names.display(None) == "reachy mini"
    assert names.display("  ") == "reachy mini"  # a blank config value is no value


def test_generic_phrases_stay_on_next_to_the_name():
    assert "wake" in spot(RobotName("Neve"), "hey robot")


def test_a_name_missing_from_the_vocabulary_is_not_used_and_is_logged_once(caplog):
    names = RobotName("reachy_mini", vocab(en={"reachy"}, it={"reachy"}))
    with caplog.at_level(logging.WARNING, logger="roboemot.voice.robot_name"):
        s = KeywordSpotter(SETS, None, names)
        assert "wake" not in s.spot("hey reachy mini", "en")
        assert "name" not in s.spot("reachy mini", "it")
        assert "wake" in s.spot("hey robot", "en")  # generic phrases are unaffected
        KeywordSpotter(SETS, None, names)
    warnings = [r for r in caplog.records if "vocabulary" in r.message]
    assert len(warnings) == 2 and "reachy mini" in warnings[0].message  # one per language


def test_the_vocabulary_is_judged_per_language():
    names = RobotName("Neve", vocab(en={"neve"}, it=set()))
    assert "wake" not in spot(names, "hey neve", "en")
    assert "wake" in spot(names, "ciao neve", "it")


def test_a_language_whose_model_is_not_loaded_waits_for_refresh():
    loaded: dict[str, set[str]] = {}
    names = RobotName("Neve", lambda lang, words: None if lang not in loaded else [])
    s = KeywordSpotter(SETS, None, names)
    assert "wake" not in s.spot("ciao neve", "it")
    loaded["it"] = set()
    names.refresh()
    assert "wake" in s.spot("ciao neve", "it")  # the same spotter picks the change up


def test_in_vocabulary_name_wakes_by_phrase_and_the_bare_name_only_in_a_short_final():
    names = RobotName("Neve", vocab(en=set(), it=set()))
    cfg = RoboEmotConfig().voice
    fast = FastPath(cfg, lambda *a: None, names=names)
    pcm = np.zeros(1600, np.float32)

    def end(text):
        fast.begin(0.0, 0.0, "it")
        r = fast.end(text, pcm, 16000, last_speech_stamp=0.0, endpoint_stamp=0.0)
        return r.wake_word

    assert end("ciao neve che tempo fa oggi per favore dimmi") is True  # template, any length
    assert end("neve") is True
    assert end("la neve cade piano in montagna oggi") is False  # long: the bare name is a word


def test_the_platform_name_is_polled_again_on_refresh():
    calls = iter([None, "Neve"])
    names = RobotName(source=lambda: next(calls))
    assert names.display(None) == ""
    names.refresh()
    assert names.display(None) == "Neve"


def test_a_failing_platform_source_is_no_name():
    def boom():
        raise ConnectionError

    assert RobotName(source=boom).display(None) == ""


def test_driver_default_has_no_name():
    assert MockPoseDriver(home={}).robot_name() is None


def test_template_without_the_placeholder_is_rejected():
    with pytest.raises(ValidationError):
        KeywordSet(name_wake_templates=["ciao"])
    with pytest.raises(ValidationError):
        KeywordSet(name_wake_templates=["{name} {name}"])
