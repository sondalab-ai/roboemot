"""SPEC 8: voice delivery composition, backend mapping and mood tags."""

import pytest
from roboemot.config import ConfigStack
from roboemot.voice import compose_delivery, parse_mood_tags, resolve_tag, to_openai, to_piper

CFG = ConfigStack().effective()


def deliver(mood, intensity=1.0, register=None, state="awake", voice=None):
    return compose_delivery(
        voice or CFG.voice,
        register=register or CFG.register.params,
        circadian=CFG.circadian.states[state],
        mood=mood,
        intensity=intensity,
    )


def test_neutral_awake_is_base():
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
    raised = CFG.register.params.model_copy(update={"pitch_st": 2.0})
    d = deliver("sleepy", register=raised, state="resting")
    lim = CFG.voice.limits
    assert lim.rate[0] <= d.rate <= lim.rate[1]
    assert d.volume_db == pytest.approx(max(lim.volume_db[0], -6.0 - 6.0 * CFG.voice.expressivity))
    assert d.pitch_st == pytest.approx(2.0)  # the register's pitch


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


def _reachy_voice():
    from pathlib import Path

    from roboemot.manifest import load_manifest

    path = Path(__file__).resolve().parents[3] / "configs/manifests/reachy-mini-wireless.yaml"
    manifest = load_manifest(path, allow_unmeasured_dynamics=True)
    stack = ConfigStack()
    stack.set_manifest_behaviour(manifest.model.behaviour, manifest.channels)
    return stack.effective().voice


def test_reachy_neutral_delivery_is_five_semitones_up():
    from roboemot.persona import PersonaEffects

    voice = _reachy_voice()
    d = deliver(None, voice=voice)
    assert d.pitch_st == pytest.approx(5.0)
    # the energy curve spans -1 .. +0.5 st: still inside the limits
    for add in (-1.0, 0.5):
        e = compose_delivery(
            voice,
            register=CFG.register.params,
            circadian=CFG.circadian.states["awake"],
            mood=None,
            intensity=0.0,
            persona=PersonaEffects(voice_pitch_st_add=add),
        )
        assert e.pitch_st == pytest.approx(5.0 + add)
    assert to_piper(d, voice)["pitch_shift_st"] == pytest.approx(5.0)


def test_reachy_delivery_pitch_clamps_at_eight():
    from roboemot.persona import PersonaEffects

    d = compose_delivery(
        _reachy_voice(),
        register=CFG.register.params.model_copy(update={"pitch_st": 2.0}),
        circadian=CFG.circadian.states["awake"],
        mood=None,
        intensity=0.0,
        persona=PersonaEffects(voice_pitch_st_add=5.0),
    )
    assert d.pitch_st == 8.0
