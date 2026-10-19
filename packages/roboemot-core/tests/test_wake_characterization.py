"""Wake and name decisions on Reachy Mini, pinned before the robot name moved to the manifest
(SPEC 8 stage 3, 4A). The same table must hold whichever layer supplies the aliases."""

import pytest
from reachy_config import reachy_config
from roboemot.voice.keywords import KeywordSpotter
from roboemot.voice.robot_name import RobotName

VOICE = reachy_config().voice
SETS = VOICE.keywords
MAX_WORDS = VOICE.fastpath.alias_wake_max_words
VOSK_SMALL = {"it": ["reachy", "reati"], "en": ["reachy"]}  # words the small models lack


def vocab(language, words):
    return [w for w in words if w in VOSK_SMALL.get(language, [])]


def decide(text, language, platform=None, configured=None, with_vocab=True):
    """(categories, wakes) as FastPath decides it for a final text."""
    names = RobotName(platform, vocab if with_vocab else None)
    names.configure(VOICE)
    cats = KeywordSpotter(SETS, configured, names).spot(text, language)
    words = len(text.split())
    return cats, "wake" in cats or ("name" in cats and words <= MAX_WORDS)


@pytest.mark.parametrize("platform", [None, "reachy_mini", "Reachy"])
@pytest.mark.parametrize(
    "text,language,cats,wakes",
    [
        ("ehi robot che ora e", "it", {"wake"}, True),
        ("ciao robot", "it", {"greeting", "wake"}, True),
        ("hey robot", "en", {"greeting", "wake"}, True),
        ("hello robot what time is it", "en", {"greeting", "wake"}, True),
        ("ciao reati", "it", {"greeting", "name"}, True),
        ("reati", "it", {"name"}, True),
        ("hello riyadh she", "en", {"greeting", "name"}, True),
        ("il comune di reati ha approvato oggi il nuovo bilancio", "it", {"name"}, False),
        ("riyadh she is a city in the middle of the desert today", "en", {"name"}, False),
        ("che tempo fa", "it", set(), False),
        ("hello reachy", "en", {"greeting"}, False),  # out of vocabulary: no name wake
    ],
)
def test_reachy_names_and_generic_phrases(platform, text, language, cats, wakes):
    assert decide(text, language, platform) == (cats, wakes)


@pytest.mark.parametrize(
    "text,language,cats,wakes",
    [
        ("ciao neve", "it", {"greeting", "wake", "name"}, True),
        ("ehi robot", "it", {"wake"}, True),
        ("ciao reati", "it", {"greeting"}, False),  # a renamed robot ignores the Reachy spellings
        ("hello riyadh she", "en", {"greeting"}, False),
    ],
)
def test_platform_name_other_than_reachy(text, language, cats, wakes):
    assert decide(text, language, platform="Neve") == (cats, wakes)


def test_configured_wake_word_wins_over_the_platform_name():
    cats, wakes = decide("ciao bruno", "it", platform="Neve", configured="Bruno")
    assert wakes and "name" in cats
    cats, wakes = decide("ciao neve", "it", platform="Neve", configured="Bruno")
    assert not wakes
    assert decide("ciao reati", "it", configured="Bruno") == ({"greeting"}, False)


def test_without_a_vocabulary_check_a_platform_name_still_wakes():
    cats, wakes = decide("ciao neve", "it", platform="Neve", with_vocab=False)
    assert wakes


def test_without_a_platform_name_the_robot_is_nameless_for_prompt_and_lines():
    names = RobotName(None, vocab)
    names.configure(VOICE)
    assert names.display(None) == "" and names.prompt_name(None) == ""
    assert names.resolve(None, ("it", "en")).matching == ""


def test_without_a_vocabulary_or_platform_name_no_name_phrases_are_added():
    plain = RobotName()
    plain.configure(VOICE)
    reachy = KeywordSpotter(SETS, None, plain)
    assert reachy.spot("ciao reachy", "it") == {"greeting"}
    assert reachy.spot("hello reachy", "en") == {"greeting"}
    assert reachy.spot("ciao reati", "it") == {"greeting", "name"}
