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


def test_mood_scaled_by_intensity_curve_and_expressivity():
    full = deliver("joy", 1.0)
    half = deliver("joy", 0.5)
    k, gamma = CFG.voice.expressivity, CFG.voice.intensity_exponent
    joy = CFG.voice.moods["joy"]
    assert full.pitch_st == pytest.approx(joy.pitch_st * k)
    assert half.pitch_st == pytest.approx(full.pitch_st * 0.5**gamma)
    assert deliver("joy", 0.0).pitch_st == 0.0  # no intensity, no mood: the curve keeps 0 at 0
    linear = CFG.voice.model_copy(update={"intensity_exponent": 1.0})
    assert deliver("joy", 0.5, voice=linear).pitch_st == pytest.approx(full.pitch_st / 2)
    flat = CFG.voice.model_copy(update={"expressivity": 0.0})
    assert deliver("joy", voice=flat).pitch_st == 0.0 and deliver("joy", voice=flat).sounds == []


def test_register_and_circadian_compose_then_clamp():
    raised = CFG.register.params.model_copy(update={"pitch_st": 2.0})
    d = deliver("sleepy", register=raised, state="resting")
    lim = CFG.voice.limits
    assert lim.rate[0] <= d.rate <= lim.rate[1]
    sleepy = CFG.voice.moods["sleepy"].volume_db * CFG.voice.expressivity
    assert d.volume_db == pytest.approx(max(lim.volume_db[0], -6.0 + sleepy))
    sleepy_pitch = CFG.voice.moods["sleepy"].pitch_st * CFG.voice.expressivity
    assert d.pitch_st == pytest.approx(2.0 + sleepy_pitch)  # the register's pitch, then the mood


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


# ---------------------------------------------------------------- moods and variation (SPEC 8)


def _say(voice=None, *, mood="joy", intensity=0.5, **kw):
    return compose_delivery(
        voice or CFG.voice,
        register=CFG.register.params,
        circadian=CFG.circadian.states["awake"],
        mood=mood,
        intensity=intensity,
        **kw,
    )


def test_every_affect_label_has_a_mood_that_changes_the_delivery():
    assert set(CFG.voice.moods) == set(CFG.affect.labels)
    neutral = _say(mood=None)
    for label in CFG.affect.labels:
        d = _say(mood=label)
        assert d.mood == label
        assert any(getattr(d, f) != getattr(neutral, f) for f in ("rate", "pitch_st", "volume_db"))


def test_the_new_moods_have_the_asked_direction():
    m = CFG.voice.moods
    assert m["surprise"].rate > 0 and m["surprise"].pitch_st > 0 and m["surprise"].variability > 0
    assert m["fear"].rate > 0 and m["fear"].pitch_st > 0 and m["fear"].volume_db < 0
    assert m["annoyance"].volume_db > 0 and m["annoyance"].pause_scale < 1
    assert m["calm"].rate < 0 and m["calm"].pause_scale > 1


def test_a_mid_intensity_mood_is_audible_on_piper():
    """At intensity 0.5 every mood moves the rate by at least 2.5 % or the pitch by at least
    0.5 st (about what a listener notices), under the default expressivity."""
    neutral = _say(mood=None)
    for label in CFG.voice.moods:
        d = _say(mood=label, intensity=0.5)
        assert abs(d.rate - neutral.rate) >= 0.025 or abs(d.pitch_st - neutral.pitch_st) >= 0.5


@pytest.mark.parametrize("label", sorted(CFG.affect.labels))
def test_every_mood_at_full_intensity_stays_inside_the_limits(label):
    wide = CFG.voice.model_copy(update={"expressivity": 1.0})
    lim = wide.limits
    for seed in range(20):
        for index in range(4):
            d = _say(wide, mood=label, intensity=1.0, seed=seed, index=index, question=True)
            assert lim.rate[0] <= d.rate <= lim.rate[1]
            assert lim.pitch_st[0] <= d.pitch_st <= lim.pitch_st[1]
            assert lim.volume_db[0] <= d.volume_db <= lim.volume_db[1]
            assert lim.variability[0] <= d.variability <= lim.variability[1]


def test_sentence_variation_is_bounded_and_deterministic():
    var, k = CFG.voice.variation, CFG.voice.expressivity
    plain = _say(mood=None)
    seen = set()
    for seed in range(1, 30):
        for index in range(6):
            d = _say(mood=None, seed=seed, index=index)
            assert d == _say(mood=None, seed=seed, index=index)  # same reply, same sentence
            assert abs(d.rate - plain.rate) <= var.rate * k + 1e-9
            assert abs(d.pitch_st - plain.pitch_st) <= var.pitch_st * k + 1e-9
            seen.add((round(d.rate, 6), round(d.pitch_st, 6)))
    assert len(seen) > 100  # the sentences of a reply differ from each other
    assert _say(mood=None, seed=None, index=3) == plain  # no seed: no variation


def test_variation_follows_expressivity_and_can_be_switched_off():
    flat = CFG.voice.model_copy(update={"expressivity": 0.0})
    assert _say(flat, mood=None, seed=7, index=2, question=True) == _say(flat, mood=None)
    off = CFG.voice.model_copy(
        update={"variation": CFG.voice.variation.model_copy(update={"rate": 0, "pitch_st": 0})}
    )
    assert _say(off, mood=None, seed=7, index=2) == _say(off, mood=None)


def test_a_question_is_raised_and_still_clamped():
    k = CFG.voice.expressivity
    raised = _say(mood=None, question=True).pitch_st - _say(mood=None).pitch_st
    assert raised == pytest.approx(CFG.voice.variation.question_pitch_st * k)
    voice = _reachy_voice()
    top = _say(voice, mood="surprise", intensity=1.0, question=True, seed=1, index=0)
    assert top.pitch_st <= voice.limits.pitch_st[1]


def test_openai_rate_words_ignore_the_sentence_variation():
    """Regression: the seeded rate jitter made a neutral reply say "faster" then "slower"."""
    assert to_openai(_say(mood=None)) == {}
    for seed in range(10):
        for index in range(4):
            assert to_openai(_say(mood=None, seed=seed, index=index)) == {}
            sad = to_openai(_say(mood="sadness", seed=seed, index=index))["instructions"]
            assert "slower" in sad and "faster" not in sad
