"""Keyword spotting on ASR text and the wake-word policy (SPEC 8 stage 3)."""

import pytest
from roboemot.config.schema import KeywordSet, RoboEmotConfig
from roboemot.voice.keywords import KeywordSpotter, normalise
from roboemot.voice.wake import WakeGate

V = RoboEmotConfig().voice


def spotter(**kw) -> KeywordSpotter:
    return KeywordSpotter(kw.get("sets", V.keywords), kw.get("wake", "reachy"))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ciao, Reati!", "ciao reati"),
        ("  FERMATI  ", "fermati"),
        ("perché è così", "perche e cosi"),
        ("stop-stop", "stop stop"),
    ],
)
def test_normalise(text, expected):
    assert normalise(text) == expected


@pytest.mark.parametrize(
    "text,lang,found",
    [
        ("fermati subito", "it", {"stop"}),
        ("FERMATI", "it", {"stop"}),
        ("ciao come stai", "it", {"greeting"}),
        ("Buongiorno!", "it", {"greeting"}),
        ("no grazie", "it", {"no"}),
        ("stop it now", "en", {"stop"}),
        ("good morning robot", "en", {"greeting"}),
        ("hello reachy", "en", {"greeting", "name"}),
        ("hello riyadh she how are you", "en", {"greeting", "name"}),
        ("ciao reati", "it", {"greeting", "name"}),
    ],
)
def test_spot_categories(text, lang, found):
    assert spotter().spot(text, lang) == found


@pytest.mark.parametrize(
    "text,lang",
    [
        ("stopped", "en"),  # no stemming: list inflections to match them
        ("nodo", "it"),
        ("bastante", "it"),
        ("chiaro", "it"),  # contains "ciao"-like letters but not the word
        ("history", "en"),  # "hi" inside a word
        ("", "it"),
        ("ciao", "xx"),  # no list for the language
    ],
)
def test_word_boundaries_and_unknown_language(text, lang):
    assert spotter().spot(text, lang) == set()


def test_accents_match_both_ways():
    sets = {"it": KeywordSet(greeting=["perché"], stop=["fermàti"])}
    s = KeywordSpotter(sets, "reachy")
    assert s.spot("PERCHE sì", "it") == {"greeting"}
    assert s.spot("fermati", "it") == {"stop"}


def test_wake_word_itself_counts_in_any_language_and_multiword_alias():
    s = spotter()
    assert s.spot("ehi Reachy", "xx") == {"name"}
    assert s.spot("hello riyadh she", "en") >= {"name"}
    assert s.spot("riyadh", "en") == set()  # half an alias is not the alias


def test_empty_wake_word_never_matches():
    s = KeywordSpotter({"it": KeywordSet(name_aliases=["reati"])}, "")
    assert s.spot("ciao", "it") == set()
    assert s.spot("reati", "it") == {"name"}


# ---- wake word policy


def gate(policy="unless_enrolled_in_view", window=30.0) -> WakeGate:
    return WakeGate(policy, window)


def test_always_requires_the_wake_word():
    g = gate("always")
    assert not g.decide(wake_word=False, enrolled_in_view=True, start=0, last_speech=1)
    assert g.decide(wake_word=True, enrolled_in_view=True, start=2, last_speech=3)


def test_never_requires_nothing():
    g = gate("never")
    assert g.decide(wake_word=False, enrolled_in_view=False, start=0, last_speech=1)


def test_unless_enrolled_in_view_cases():
    g = gate()
    assert g.decide(wake_word=False, enrolled_in_view=True, start=0, last_speech=1)
    assert not g.decide(wake_word=False, enrolled_in_view=False, start=2, last_speech=3)
    assert g.decide(wake_word=True, enrolled_in_view=False, start=4, last_speech=5)


def test_window_opens_on_wake_word_and_covers_followups():
    g = gate(window=10.0)
    assert g.decide(wake_word=True, enrolled_in_view=False, start=0, last_speech=2)
    assert g.decide(wake_word=False, enrolled_in_view=False, start=11.9, last_speech=13)
    # the follow-up extended the window from its own last speech
    assert g.decide(wake_word=False, enrolled_in_view=False, start=22.0, last_speech=23)
    assert not g.decide(wake_word=False, enrolled_in_view=False, start=33.5, last_speech=34)


def test_window_closed_after_it_runs_out():
    g = gate(window=10.0)
    g.decide(wake_word=True, enrolled_in_view=False, start=0, last_speech=2)
    assert not g.decide(wake_word=False, enrolled_in_view=False, start=12.5, last_speech=13)


def test_unaddressed_utterance_does_not_open_window():
    g = gate(window=10.0)
    assert not g.decide(wake_word=False, enrolled_in_view=False, start=0, last_speech=1)
    assert not g.decide(wake_word=False, enrolled_in_view=False, start=2, last_speech=3)


def test_enrolled_person_utterance_does_not_open_window_for_strangers():
    g = gate(window=10.0)
    assert g.decide(wake_word=False, enrolled_in_view=True, start=0, last_speech=1)
    assert not g.decide(wake_word=False, enrolled_in_view=False, start=2, last_speech=3)


def test_apply_changes_policy_and_window():
    g = gate("always", 5.0)
    g.apply("never", 5.0)
    assert g.decide(wake_word=False, enrolled_in_view=False, start=0, last_speech=1)
