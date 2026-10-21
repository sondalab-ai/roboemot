"""SPEC 12A (Behaviour while offline), SPEC 8: the offline phrase goes out through speech output."""

from __future__ import annotations

from roboemot.config import ConfigStack
from roboemot.health.speech_cue import SpeechCue
from test_health import responder

CFG = ConfigStack().effective()
PHRASES = CFG.connectivity.offline_phrases
COOLDOWN_S = CFG.connectivity.cue_cooldown_s


class FakeSpeech:
    def __init__(self) -> None:
        self.said: list[dict] = []

    def say(self, text, **kwargs):
        self.said.append({"text": text, **kwargs})
        return len(self.said)


def wired(language="it", *, speech=None, muted=False):
    """A real responder whose cue speaks on a fake output; ``muted`` is a one-item list."""
    r, _, _ = responder(language=language)
    out = FakeSpeech() if speech is None else speech
    mute = [muted]
    r.cue = SpeechCue(
        lambda: out if out is not False else None,
        lambda: (CFG.register.unknown, CFG.circadian.states["awake"], "joy", 0.5),
        lambda: mute[0],
    )
    return r, out, mute


def test_speech_while_offline_speaks_the_phrase_once():
    r, out, _ = wired()
    r.speech_while_offline(0.0, "it")
    assert [s["text"] for s in out.said] == [PHRASES["it"]]
    said = out.said[0]
    assert said["language"] == "it" and said["mood"] == "joy"
    assert r.commands(0.5)  # the puzzled gesture still plays


def test_cooldown_applies_to_speech_and_network_loss_together():
    r, out, _ = wired()
    r.on_event("link.network.lost", 0.0)
    r.speech_while_offline(10.0, "it")
    assert len(out.said) == 1
    r.speech_while_offline(COOLDOWN_S, "it")
    assert len(out.said) == 2


def test_network_loss_speaks_once_in_the_configured_language():
    r, out, _ = wired(language="en")
    r.on_event("link.network.lost", 0.0)
    r.on_event("link.network.restored", 1.0)
    assert [(s["text"], s["language"]) for s in out.said] == [(PHRASES["en"], "en")]


def test_the_utterance_language_wins_and_unknown_falls_back_to_the_configured_one():
    r, out, _ = wired(language="it")
    r.speech_while_offline(0.0, "en")
    r.speech_while_offline(COOLDOWN_S, "fr")
    assert [(s["text"], s["language"]) for s in out.said] == [
        (PHRASES["en"], "en"),
        (PHRASES["it"], "it"),
    ]


def test_speak_false_says_nothing_and_keeps_the_cooldown_free():
    r, out, _ = wired()
    r.speech_while_offline(0.0, "it", speak=False)
    assert out.said == [] and r.commands(0.5)
    r.speech_while_offline(1.0, "it")
    assert len(out.said) == 1


def test_muted_says_nothing_while_asleep_or_a_minor_is_present():
    r, out, mute = wired(muted=True)
    r.on_event("link.network.lost", 0.0)
    r.speech_while_offline(COOLDOWN_S, "it")
    assert out.said == []
    mute[0] = False
    r.speech_while_offline(2 * COOLDOWN_S, "it")
    assert len(out.said) == 1


def test_no_speech_output_is_silent_without_an_exception():
    r, _, _ = wired(speech=False)
    r.on_event("link.network.lost", 0.0)
    r.speech_while_offline(COOLDOWN_S, "it")


def test_a_failing_output_never_breaks_the_responder():
    class Broken:
        def say(self, text, **kwargs):
            raise RuntimeError("tts down")

    r, _, _ = wired(speech=Broken())
    r.on_event("link.network.lost", 0.0)
    r.speech_while_offline(COOLDOWN_S, "it")
    assert r.commands(COOLDOWN_S + 0.5)
