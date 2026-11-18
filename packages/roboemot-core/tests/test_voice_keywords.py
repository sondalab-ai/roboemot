"""Keyword spotting on ASR text and the wake-word policy (SPEC 8 stage 3)."""

import pytest
from reachy_config import reachy_config
from roboemot.config.schema import KeywordSet
from roboemot.voice.keywords import KeywordSpotter, normalise
from roboemot.voice.robot_name import RobotName
from roboemot.voice.wake import WakeGate

V = reachy_config().voice


def spotter(**kw) -> KeywordSpotter:
    names = RobotName()
    names.configure(V)
    return KeywordSpotter(kw.get("sets", V.keywords), kw.get("wake", "reachy"), names)


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
        ("yes please", "en", {"yes"}),
        ("certo, volentieri", "it", {"yes"}),
        ("si chiama Luca", "it", set()),  # the pronoun is not a yes (accents are folded)
        ("stop it now", "en", {"stop"}),
        ("good morning robot", "en", {"greeting", "name"}),  # an extra name
        ("hello reachy", "en", {"greeting", "name", "wake"}),  # "hello {name}"
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


# ---- wake phrases, Unicode, window cap and speaker


def test_wake_phrases_are_their_own_category():
    s = spotter()
    assert s.spot("ehi robot dimmi", "it") == {"wake", "name"}  # "robot": an extra name
    assert s.spot("Hello, ROBOT!", "en") == {"wake", "greeting", "name"}
    assert "wake" not in s.spot("ehi", "it") and "wake" not in s.spot("robot", "it")


@pytest.mark.parametrize(
    "text,expected", [("È già l'amico", "e gia l amico"), ("ÇA va", "ca va"), ("naïve", "naive")]
)
def test_normalise_is_unicode_aware(text, expected):
    assert normalise(text) == expected


def test_elided_words_match_without_the_apostrophe():
    s = KeywordSpotter({"it": KeywordSet(greeting=["amico"], no=["è"])}, "reachy")
    assert s.spot("l'amico", "it") == {"greeting"}
    assert s.spot("E' vero", "it") == {"no"}  # accents fold: è, e and E' are one word


def test_window_is_capped_from_the_wake_word():
    g = WakeGate("always", 10.0, 25.0)
    assert g.decide(wake_word=True, enrolled_in_view=False, start=0, last_speech=2)
    assert g.decide(wake_word=False, enrolled_in_view=False, start=10, last_speech=12)
    assert g.decide(wake_word=False, enrolled_in_view=False, start=20, last_speech=22)
    # the follow-ups reached the cap (2 + 25 = 27): the window stops there
    assert not g.decide(wake_word=False, enrolled_in_view=False, start=28, last_speech=29)


def test_followup_by_a_different_known_person_is_not_addressed():
    g = WakeGate("always", 10.0, 100.0)
    assert g.decide(wake_word=True, enrolled_in_view=False, start=0, last_speech=1, speaker="p-1")
    assert not g.decide(
        wake_word=False, enrolled_in_view=False, start=3, last_speech=4, speaker="p-2"
    )
    assert g.decide(wake_word=False, enrolled_in_view=False, start=5, last_speech=6, speaker="p-1")
    assert g.decide(wake_word=False, enrolled_in_view=False, start=7, last_speech=8, speaker=None)


def test_gate_snapshot_has_no_content():
    g = WakeGate("always", 10.0)
    g.decide(wake_word=True, enrolled_in_view=False, start=0, last_speech=1)
    snap = g.snapshot(5.0)
    assert {k: snap[k] for k in ("policy", "window_open", "window_left_s")} == {
        "policy": "always",
        "window_open": True,
        "window_left_s": 6.0,
    }
    assert snap["decisions"]["wake_word"] == 1
    assert g.snapshot(50.0)["window_open"] is False


def server_spotter(wake: str) -> KeywordSpotter:
    """The spotter of a server's transcript: no recognizer vocabulary limits the name."""
    names = RobotName()
    names.configure(V)
    return KeywordSpotter(V.keywords, wake, names.unchecked())  # type: ignore[arg-type]


# Owner decision 2026-11-17 (SPEC 8, Radio playback): the server wrote "Nivea" for "Neve".
# Accepted: close spellings a recognizer gives (vowels inserted, dropped, or i/e and u/o read
# alike). Rejected: frequent Italian words one consonant or one other vowel away, so a lyric
# without the name is still dropped. "nevi" (snows) is accepted: same vowels as the name.
NAME_CLOSE = [
    ("Neve", True),
    ("NEVE", True),
    ("Neve'", True),
    ("Nivea", True),
    ("Nieve", True),
    ("nevi", True),
    ("nave", False),
    ("nove", False),
    ("deve", False),
    ("vede", False),
    ("beve", False),
    ("vene", False),
    ("neva", False),
    ("nevo", False),
    ("leve", False),
    ("breve", False),
    ("nei", False),
    ("nel", False),
    ("neo", False),
    ("ne", False),
    ("eve", False),
]


@pytest.mark.parametrize("word,close", NAME_CLOSE)
def test_a_close_spelling_of_the_name_in_a_server_transcript(word, close):
    s = server_spotter("Neve")
    assert s.name_close(f"ehm {word} stoppa la musica", "it", 1, 4) is close


def test_a_close_spelling_needs_the_edit_bound_and_the_min_length():
    s = server_spotter("Neve")
    assert not s.name_close("Nivea", "it", 0, 4)  # 0 edits: the exact name only
    assert s.name_close("Neve", "it", 0, 4)
    assert not s.name_close("Nieve", "it", 1, 5)  # a name shorter than min_letters: exact
    t = server_spotter("Bo")
    assert t.name_close("bo", "it", 1, 1) and not t.name_close("boa", "it", 1, 4)


def test_a_name_of_several_words_matches_each_word_in_order():
    s = server_spotter("Neve Bianca")
    assert s.name_close("ciao nivea bianca", "it", 1, 4)
    assert not s.name_close("bianca nivea", "it", 1, 4)  # out of order
    assert not s.name_close("nivea nera", "it", 1, 4)  # every word must match


# Owner decision 2026-11-17: English transcripts spell "Neve" the English way. Known false
# accepts: "nav" and "nave" (one vowel off "neve" once a reads as e).
NAME_CLOSE_EN = [
    *[(w, True) for w in ("Navy", "Neva", "Nevay", "Neevee", "Nevvy", "Neveh", "Nev", "Neve")],
    ("nav", True),
    ("nave", True),
    *[
        (w, False)
        for w in (
            "never", "knave", "eve", "even", "nerve", "novel", "have", "give", "live",
            "love", "move", "wave", "nova", "envy",
        )
    ],
]  # fmt: skip


@pytest.mark.parametrize("word,close", NAME_CLOSE_EN)
def test_a_close_english_spelling_of_the_name(word, close):
    s = server_spotter("Neve")
    assert s.name_close(f"hey {word} stop the music", "en", 1, 4) is close


def test_the_english_rules_apply_only_to_english_transcripts():
    s = server_spotter("Neve")
    assert s.name_close("Navy", "en", 1, 4) and not s.name_close("Navy", "it", 1, 4)
    assert not s.name_close("nave", "it", 1, 4)  # Italian unchanged


def test_an_english_name_of_several_words():
    s = server_spotter("Neve Bianca")
    assert s.name_close("hi neevee biancah please", "en", 1, 4)
    assert not s.name_close("hi neevee bianka please", "en", 1, 4)  # k is not c
    assert not s.name_close("bianca navy", "en", 1, 4)
