"""SPEC 8: voice delivery composition, backend mapping and mood tags."""

import pytest
from roboemot.config import ConfigStack
from roboemot.voice import compose_delivery, parse_mood_tags, resolve_tag, to_openai, to_piper

CFG = ConfigStack().effective()


def deliver(mood, intensity=1.0, register="adult", state="awake", voice=None):
    return compose_delivery(
        voice or CFG.voice,
        register=getattr(CFG.register, register),
        circadian=CFG.circadian.states[state],
        mood=mood,
        intensity=intensity,
    )


def test_neutral_adult_awake_is_base():
    d = deliver(None)
    assert (d.rate, d.pitch_st, d.volume_db) == (1.0, 0.0, 0.0)


def test_mood_scaled_by_intensity_and_expressivity():
    full = deliver("joy", 1.0)
    half = deliver("joy", 0.5)
    k = CFG.voice.expressivity
    assert full.pitch_st == pytest.approx(2 * k)
    assert half.pitch_st == pytest.approx(full.pitch_st / 2)
    flat = CFG.voice.model_copy(update={"expressivity": 0.0})
    assert deliver("joy", voice=flat).pitch_st == 0.0 and deliver("joy", voice=flat).sounds == []


def test_register_and_circadian_compose_then_clamp():
    d = deliver("sleepy", register="child", state="resting")
    lim = CFG.voice.limits
    assert lim.rate[0] <= d.rate <= lim.rate[1]
    assert d.volume_db == pytest.approx(max(lim.volume_db[0], -6.0 - 6.0 * CFG.voice.expressivity))
    assert d.pitch_st == pytest.approx(2.0)  # child register


def test_fixed_mood_source():
    v = CFG.voice.model_copy(update={"mood_source": "fixed", "fixed_mood": "sadness"})
    assert deliver(None, voice=v).mood == "sadness"


def test_piper_and_openai_mapping():
    d = deliver("sadness")
    p = to_piper(d, CFG.voice)
    assert p["length_scale"] == pytest.approx(1 / d.rate) and p["volume"] < 1.0
    assert "soft, slow" in to_openai(d)["instructions"]


def test_mood_tags_parsed_and_bounded_by_affect():
    parts = parse_mood_tags("[mood=joy] Che bello! [mood=curiosity] Cos'è?")
    assert parts == [("joy", "Che bello!"), ("curiosity", "Cos'è?")]
    # robot is sad: an LLM 'joy' tag is too far from the affect state and is overridden
    assert (
        resolve_tag(
            "joy",
            affect_label="sadness",
            valence=-0.6,
            arousal=-0.4,
            affect=CFG.affect,
            voice=CFG.voice,
        )
        == "sadness"
    )
    assert (
        resolve_tag(
            "curiosity",
            affect_label="joy",
            valence=0.5,
            arousal=0.45,
            affect=CFG.affect,
            voice=CFG.voice,
        )
        == "curiosity"
    )
