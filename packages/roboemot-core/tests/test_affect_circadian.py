"""SPEC 7 affect engine, SPEC 7A circadian model and light sensing."""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.affect import AffectEngine
from roboemot.circadian import CircadianModel, LightEstimator
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import RoboEmotConfig

CFG = RoboEmotConfig()


def engine() -> AffectEngine:
    return AffectEngine(CFG.affect, CFG.appraisal)


# ----------------------------------------------------------------------------- affect


def test_decay_half_life_toward_baseline():
    e = engine()
    e.impulse(0.0, valence=0.8)
    e.update(CFG.affect.half_life_s)
    assert e.valence == pytest.approx(0.4)
    e.set_baseline(arousal=-0.5)
    for t in range(1, 200):
        e.update(CFG.affect.half_life_s + t)
    assert e.arousal == pytest.approx(-0.5, abs=1e-3)


def test_appraisal_rule_with_label_and_cooldown():
    e = engine()
    assert e.on_event("sound.loud", 0.0)
    assert e.state(0.1).label == "surprise"
    assert not e.on_event("sound.loud", 1.0)  # cooldown 5 s
    assert e.on_event("sound.loud", 6.0)
    after = 6.0 + CFG.affect.event_label_hold_s + 0.1
    st = e.update(after)
    assert st.label == e.nearest_label(st.valence, st.arousal)  # hold expired


def test_label_is_nearest_prototype_after_hold():
    e = engine()
    e.impulse(0.0, valence=0.7, arousal=0.5)
    assert e.state(0.0).label == "joy"
    e.impulse(0.0, valence=-1.4, arousal=-0.9)
    assert e.state(0.0).label == "sadness"


def test_llm_impulse_capped_per_turn():
    e = engine()
    dv, da = e.llm_impulse(0.0, 1.0, 0.0)
    assert dv == pytest.approx(CFG.affect.llm_impulse_max)
    assert e.llm_impulse(0.0, 1.0, 0.0) == (0.0, 0.0)  # budget used
    e.new_turn()
    assert e.llm_impulse(0.0, 0.0, 0.1)[1] == pytest.approx(0.1)


def test_unknown_label_rejected():
    with pytest.raises(KeyError):
        engine().impulse(0.0, label="ecstasy")


# ----------------------------------------------------------------------------- circadian


def replay_day(
    model: CircadianModel,
    light_for_hour,
    start_hour=7.0,
    hours=24,
    step_min=5,
    presence=lambda h: False,
):
    states = []
    steps = int(hours * 60 / step_min)
    for i in range(steps):
        hour = (start_hour + i * step_min / 60) % 24
        st = model.update(
            hour, step_min * 60, light=light_for_hour(hour), presence_active=presence(hour)
        )
        states.append(st)
    return states


def normal_light(h: float) -> str:
    return "bright" if 7 <= h < 22 else "dark"


def test_day_goes_awake_drowsy_resting_and_back():
    model = CircadianModel(CFG.circadian)
    states = replay_day(model, normal_light, hours=48)
    seen = [s.state for s in states]
    assert {"awake", "drowsy", "resting"} <= set(seen)
    night = [s.state for s in states if s.hour >= 23.5 or s.hour < 6.5]
    assert set(night) == {"resting"}  # quiet hours force rest
    noon = [s.state for s in states[len(states) // 2 :] if 11 <= s.hour < 13]
    assert set(noon) == {"awake"}


def test_late_evening_light_delays_phase_gradually():
    model = CircadianModel(CFG.circadian)
    late = lambda h: "bright" if h >= 7 or h < 2 else "dark"  # noqa: E731
    replay_day(model, late, hours=72)
    shift = model.phase_shift_h
    assert 0 < shift <= CFG.circadian.max_shift_h_per_day * 3 + 1e-9


def test_presence_keeps_awake_in_quiet_hours_and_wake_word():
    model = CircadianModel(CFG.circadian)
    st = model.update(23.0, 60, light="dark", presence_active=True)
    assert st.state != "resting"
    st = model.update(23.1, 60, light="dark", presence_active=False)
    assert st.state == "resting"
    model.wake()
    assert model.update(23.2, 60, light="dark").state == "awake"
    assert model.effects().arousal_baseline == CFG.circadian.states["awake"].arousal_baseline


def test_circadian_config_validation():
    with pytest.raises(ConfigError):
        ConfigStack(app={"circadian": {"quiet_hours": ["25:00", "07:00"]}})
    with pytest.raises(ConfigError):
        ConfigStack(app={"circadian": {"resting_threshold": 0.4}})


# ----------------------------------------------------------------------------- light


def frame(level: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((48, 64, 3), level * 255.0) + rng.normal(0, noise * 255.0, (48, 64, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_light_levels_with_hysteresis():
    est = LightEstimator(CFG.light)
    assert est.measure(frame(0.6)).level == "bright"
    assert est.measure(frame(0.34)).level == "bright"  # inside hysteresis band
    assert est.measure(frame(0.2)).level == "dim"
    assert est.measure(frame(0.05)).level == "dark"


def test_auto_exposure_pumped_dim_frame_reads_dark():
    est = LightEstimator(CFG.light)
    # Auto-exposure brought a dark room up to mid-grey, but with high sensor noise.
    assert est.measure(frame(0.25, noise=0.15)).level == "dark"
    assert LightEstimator(CFG.light).measure(frame(0.25)).level == "dim"


def test_light_with_exposure_metadata():
    cfg = CFG.light.model_copy(update={"calibration_k": 0.01})
    est = LightEstimator(cfg)
    # Same mid-grey image: long exposure at high gain means a dark room.
    assert est.measure(frame(0.5), exposure_s=0.05, gain=8.0).level == "dark"
    assert est.measure(frame(0.5), exposure_s=0.001, gain=1.0).level == "bright"


# ----------------------------------------------------------------------------- review fixes


def test_llm_nan_impulse_rejected_and_budget_intact():
    e = engine()
    with pytest.raises(ValueError):
        e.llm_impulse(0.0, float("nan"), 0.0)
    dv, _ = e.llm_impulse(0.0, 1.0, 0.0)
    assert dv == pytest.approx(CFG.affect.llm_impulse_max)
    assert e.llm_impulse(0.0, -1.0, -1.0) == (0.0, 0.0)


def test_noisy_dark_detected_after_bright_history():
    est = LightEstimator(CFG.light)
    assert est.measure(frame(0.6)).level == "bright"
    assert est.measure(frame(0.25, noise=0.15)).level == "dark"


def test_process_c_continuous_with_non_24h_period():
    cfg = CFG.circadian.model_copy(update={"period_h": 24.5})
    model = CircadianModel(cfg)
    prev = None
    for i in range(24 * 12 * 3):  # 3 days, 5-minute steps
        hour = (i * 5 / 60) % 24
        st = model.update(hour, 300, light=None)
        if prev is not None:
            assert abs(st.c - prev) < 0.02  # natural max ~0.011 per 5 min; old jump ~0.09
        prev = st.c


def test_circadian_clock_follows_wall_clock_jumps():
    """N9: a DST/NTP jump re-anchors process C instead of drifting."""
    model = CircadianModel(CFG.circadian)
    model.update(10.0, 60)
    model.update(10.0 + 1 / 60, 60)
    st = model.update(11.0 + 2 / 60, 60)  # wall clock jumped one hour forward
    assert st.c == pytest.approx(model.process_c(11.0 + 2 / 60))
    assert model.t_h % 24 == pytest.approx(11.0 + 2 / 60)


def test_affect_update_rejects_nan_time():
    with pytest.raises(ValueError):
        engine().update(float("nan"))
