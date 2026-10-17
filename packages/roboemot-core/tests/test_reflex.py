"""Reflex engine: rules, triggers, cooldown, priority, safety and latency (SPEC 10, 12, 13)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError
from roboemot.affect.appraisal import AppraisalAdapter
from roboemot.bus import Bus
from roboemot.config.schema import AppraisalConfig, ReflexConfig, ReflexRule, RoboEmotConfig
from roboemot.config.stack import violations
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionLoop, Priority
from roboemot.reflex import ReflexEngine, ReflexMap
from roboemot.vision.types import FacePercept
from roboemot.world import Observation

REACHY = Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
DT = 0.02
PITCH, YAW = "head.pitch", "head.yaw"
RMAP = ReflexMap(
    head_yaw=YAW,
    head_pitch=PITCH,
    body_yaw="body_yaw",
    antennas={"antenna_right": 1.0, "antenna_left": -1.0},
    pitch_away_sign=-1.0,
    voice_sources=("doa",),
)
QUIET, LOUD = 0.02, 0.4  # RMS: a ratio of 20 over the floor, past every default threshold


def cfg_with(*rules: dict, **top) -> ReflexConfig:
    return ReflexConfig(rules=list(rules), **top) if rules else ReflexConfig(**top)


def rule(**kw) -> dict:
    base = {
        "name": "r",
        "trigger": "loud_onset",
        "primitive": "startle",
        "threshold": 6.0,
        "amplitude_deg": 12,
        "cooldown_s": 3,
        "duration_ms": 400,
    }
    return base | kw


class Rig:
    """A motion loop on a mock driver with the engine hooked in, on a simulated clock.

    Like the app, it feeds an idle command every tick so the watchdog stays quiet.
    """

    def __init__(self, cfg: ReflexConfig, *, hold: dict[str, float] | None = None) -> None:
        manifest = load_manifest(REACHY, allow_unmeasured_dynamics=True)
        # fast dynamics: the limiter must not be what these tests measure
        manifest.dofs = {
            n: replace(d, max_vel=50.0, max_acc=5000.0) for n, d in manifest.dofs.items()
        }
        self.manifest = manifest
        self.driver = MockPoseDriver({n: d.home for n, d in manifest.dofs.items()})
        self.t = 0.0
        full = RoboEmotConfig()
        self.loop = MotionLoop(manifest, self.driver, full, rate_hz=50, clock=lambda: self.t)
        self.loop.set_mode("active")
        self.bus = Bus()
        self.engine = ReflexEngine(
            cfg,
            RMAP,
            manifest.dofs,
            manifest.constraints,
            margin_rad=math.radians(3),
            bus=self.bus,
        )
        self.engine.attach(self.bus)
        self.loop.tick_hooks.append(self.engine.on_tick)
        self.hold = hold
        self.pitch: list[float] = []

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.loop.feed(Command("idle", Priority.IDLE, {PITCH: 0.0}, self.t))
            if self.hold:
                self.loop.feed(Command("hold", Priority.COMMANDED, dict(self.hold), self.t))
            out = self.loop.step_once(self.t, DT)
            self.pitch.append(out[PITCH])
            self.t += DT

    def run(self, seconds: float) -> None:
        self.tick(round(seconds / DT))

    def settle_audio(self, seconds: float = 1.0, level: float = QUIET) -> None:
        """Quiet audio for ``seconds`` (also advances the loop), so the floor is learned."""
        for _ in range(round(seconds / DT)):
            self.engine.offer_audio_level(level, self.t)
            self.tick()

    def reflex_sources(self) -> list[str]:
        return [c.source for c in self.loop.core.arbiter.active(self.t) if c.priority == 1]

    def stats(self, name: str = "r") -> dict:
        return self.engine.snapshot()["rules"][name]


# --------------------------------------------------------------------------- config


def test_defaults_are_valid_and_inside_the_spec_ranges():
    cfg = ReflexConfig()
    assert {r.name for r in cfg.rules} == {"startle-loud", "withdraw-close", "turn-to-voice"}
    assert all(100 <= r.duration_ms <= 800 for r in cfg.rules)
    assert RoboEmotConfig().reflexes == cfg


@pytest.mark.parametrize("ms", [99.0, 801.0, 0.0, 5000.0])
def test_duration_outside_100_to_800_ms_is_rejected(ms):
    with pytest.raises(ValidationError):
        ReflexRule(**rule(duration_ms=ms))


def test_100_and_800_ms_are_accepted():
    ReflexRule(**rule(duration_ms=100))
    ReflexRule(**rule(duration_ms=800))


@pytest.mark.parametrize(
    "bad",
    [
        {"trigger": "proximity", "primitive": "startle", "threshold": 0.4},
        {"trigger": "loud_onset", "primitive": "turn_toward", "suppress_with_minor": True},
        {"trigger": "new_voice", "primitive": "withdraw", "threshold": 3},
        {"trigger": "proximity", "primitive": "withdraw", "threshold": 3.0},  # fraction > 1
        {"trigger": "loud_onset", "threshold": 1.0},  # onset ratio below 1.5
        {"name": "has space"},
        {"name": ""},
        {"unknown_key": 1},
    ],
)
def test_invalid_rules_are_rejected(bad):
    with pytest.raises(ValidationError):
        ReflexRule(**rule(**bad))


def test_a_turn_toward_rule_must_suppress_with_a_minor():
    voice = rule(trigger="new_voice", primitive="turn_toward", threshold=3)
    with pytest.raises(ValidationError, match="suppress_with_minor"):
        ReflexRule(**voice)
    ReflexRule(**voice, suppress_with_minor=True)


def test_duplicate_names_and_a_shape_without_a_hold_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        cfg_with(rule(), rule())
    with pytest.raises(ValidationError):
        ReflexConfig(attack_fraction=0.6, release_fraction=0.6)


def test_rules_load_from_yaml_shaped_data():
    cfg = RoboEmotConfig.model_validate(
        {"reflexes": {"rules": [rule(name="a"), rule(name="b", cooldown_s=9)]}}
    )
    assert [r.name for r in cfg.reflexes.rules] == ["a", "b"]


def test_runtime_overrides_may_tighten_never_loosen_a_reflex():
    approved = RoboEmotConfig.model_validate({"reflexes": {"rules": [rule()]}})

    def candidate(**kw):
        return RoboEmotConfig.model_validate({"reflexes": {"rules": [rule(**kw)]}})

    assert violations(approved, candidate(amplitude_deg=6, cooldown_s=9, duration_ms=200)) == []
    for loose in ({"amplitude_deg": 30}, {"cooldown_s": 1}, {"duration_ms": 700}):
        assert [v.path for v in violations(approved, candidate(**loose))], loose
    loosened_cap = RoboEmotConfig.model_validate({"reflexes": {"max_amplitude_deg": 90}})
    assert violations(RoboEmotConfig(), loosened_cap)
    off = RoboEmotConfig.model_validate({"reflexes": {"rules": [rule(suppress_with_minor=False)]}})
    on = RoboEmotConfig.model_validate({"reflexes": {"rules": [rule(suppress_with_minor=True)]}})
    assert violations(on, off) and not violations(off, on)


def test_removing_a_rule_does_not_read_as_loosening_the_next_one():
    two = RoboEmotConfig.model_validate(
        {"reflexes": {"rules": [rule(name="a", amplitude_deg=5), rule(name="b", amplitude_deg=40)]}}
    )
    one = RoboEmotConfig.model_validate({"reflexes": {"rules": [rule(name="b", amplitude_deg=40)]}})
    assert violations(two, one) == []


# --------------------------------------------------------------------------- loud onset


def test_a_loud_onset_plays_a_startle_at_reflex_priority_and_meets_the_audio_budget():
    rig = Rig(cfg_with(rule()))
    rig.settle_audio()
    stamp = rig.t
    rig.engine.offer_audio_level(LOUD, stamp)
    rig.t += 0.013  # the stamp is a little older than the loop tick that picks it up
    rig.tick()
    assert rig.reflex_sources() == ["reflex.r"]
    rig.run(0.2)
    lat = rig.engine.snapshot()["latency"]["loud_onset"]
    assert lat["command_ms"]["count"] == 1
    assert lat["command_ms"]["p95"] <= 100.0  # SPEC 13: audio trigger to reflex motion
    assert lat["command_ms"]["p95"] == pytest.approx(13.0, abs=1.0)
    assert lat["motion_ms"]["max"] <= 100.0
    assert min(rig.pitch) < -math.radians(6)  # the head tipped back


def test_startle_lifts_the_antennas_and_never_tips_the_head_forward():
    rig = Rig(cfg_with(rule()))
    rig.settle_audio()
    rig.engine.offer_audio_level(LOUD, rig.t)
    rig.tick(2)
    peak = {n: 0.0 for n in ("antenna_right", "antenna_left")}
    for _ in range(round(0.5 / DT)):
        out = rig.loop.step_once(rig.t, DT)
        rig.loop.feed(Command("idle", Priority.IDLE, {PITCH: 0.0}, rig.t))
        rig.t += DT
        for n in peak:
            peak[n] = max(peak[n], abs(out[n] - rig.manifest.dofs[n].home))
    assert max(rig.pitch) <= 1e-9
    assert all(v > 0 for v in peak.values())


def test_a_steady_loud_room_is_not_an_onset():
    rig = Rig(cfg_with(rule()))
    for _ in range(100):
        rig.engine.offer_audio_level(LOUD, rig.t)
        rig.tick()
    assert rig.stats()["fired"] == 0
    assert rig.reflex_sources() == []


def test_a_slow_rise_is_not_an_onset_but_a_jump_after_it_is():
    rig = Rig(cfg_with(rule(threshold=4.0)))
    level = QUIET
    for _ in range(150):
        level *= 1.02  # 2% a chunk: the floor follows
        rig.engine.offer_audio_level(level, rig.t)
        rig.tick()
    assert rig.stats()["fired"] == 0
    rig.engine.offer_audio_level(level * 10, rig.t)
    rig.tick()
    assert rig.stats()["fired"] == 1


def test_cooldown_blocks_a_second_firing_until_it_has_passed():
    rig = Rig(cfg_with(rule(cooldown_s=3)))
    rig.settle_audio()
    for gap in (1.0, 1.0, 2.5, 0.1):  # onsets at 0, 1 (blocked), 2 (blocked), 4.6 (fires)
        rig.engine.offer_audio_level(LOUD, rig.t)
        rig.tick()
        rig.settle_audio(gap)
    stats = rig.stats()
    assert stats["fired"] == 2 and stats["dropped_cooldown"] == 2


def test_a_trigger_older_than_max_age_is_dropped_not_played_late():
    rig = Rig(cfg_with(rule(), max_age_ms=150))
    rig.settle_audio()
    rig.engine.offer_audio_level(LOUD, rig.t)
    rig.t += 0.5  # the loop was stalled (motors off, link down)
    rig.tick()
    assert rig.stats()["fired"] == 0 and rig.stats()["dropped_stale"] == 1
    assert rig.reflex_sources() == []


def test_the_reflex_ends_after_its_duration_and_idle_gets_the_head_back():
    rig = Rig(cfg_with(rule(duration_ms=300)))
    rig.settle_audio()
    rig.engine.offer_audio_level(LOUD, rig.t)
    rig.tick()
    assert rig.reflex_sources() == ["reflex.r"]
    rig.run(0.28)
    assert rig.reflex_sources() == ["reflex.r"]
    rig.run(0.06)
    assert rig.reflex_sources() == [] and rig.engine.active() == []
    rig.run(0.3)  # the cross-fade back
    winner = rig.loop.core.arbiter._winner[PITCH]
    assert winner == "idle"
    assert rig.pitch[-1] == pytest.approx(0.0, abs=1e-6)


def test_a_reflex_pre_empts_commanded_idle_and_expression_motion():
    rig = Rig(cfg_with(rule()), hold={PITCH: 0.3})
    rig.settle_audio()
    pre = rig.pitch[-1]
    assert pre == pytest.approx(0.3, abs=0.01)  # commanded wins over idle
    rig.loop.feed(Command("expression", Priority.EXPRESSION, {PITCH: 0.1}, rig.t, additive=True))
    rig.engine.offer_audio_level(LOUD, rig.t)
    rig.run(0.3)
    assert rig.loop.core.arbiter._winner[PITCH] == "reflex.r"
    assert min(rig.pitch[-15:]) < pre - math.radians(10)  # from where it was, no expression added


def test_a_failing_hook_does_not_stop_the_tick():
    rig = Rig(cfg_with(rule()))

    def boom(now, core):
        raise RuntimeError("bug")

    rig.loop.tick_hooks.insert(0, boom)
    rig.tick(3)
    assert len(rig.driver.writes) == 3 and rig.loop.errors == 3


def test_disabling_at_runtime_stops_a_running_reflex():
    rig = Rig(cfg_with(rule()))
    rig.settle_audio()
    rig.engine.offer_audio_level(LOUD, rig.t)
    rig.tick(2)
    assert rig.reflex_sources()
    rig.engine.apply_config(cfg_with(rule(), enabled=False), 0.0)
    rig.tick()
    assert rig.reflex_sources() == []


def test_sound_loud_is_published_once_per_onset_and_reaches_appraisal():
    rig = Rig(cfg_with(rule()))
    seen = []
    AppraisalAdapter(rig.bus, AppraisalConfig(), lambda ev, now, payload: seen.append(ev))
    published = []
    rig.bus.subscribe("sound.loud", lambda m: published.append(m.payload))
    rig.settle_audio()
    for _ in range(5):  # one burst, several loud chunks
        rig.engine.offer_audio_level(LOUD, rig.t)
        rig.tick()
    assert len(published) == 1 and seen == ["sound.loud"]
    assert set(vars(published[0])) == {"ratio", "level"}  # scalars only


# --------------------------------------------------------------------------- proximity


def face(stamp: float, height: float, y1: float = 0.2) -> FacePercept:
    return FacePercept(
        track_id=1, stamp=stamp, box=(0.3, y1, 0.6, y1 + height), score=0.9, azimuth_rad=0.0,
        elevation_rad=0.0, owner_id=None, anonymous_id="a", age_group="adult", age_source="x",
        gender="unknown", gender_source="none", attribute_confidence=0.9, possible_minor=False,
        new_track=False,
    )  # fmt: skip


PROXIMITY = rule(name="w", trigger="proximity", primitive="withdraw", threshold=0.4, cooldown_s=0)


def test_a_close_face_withdraws_with_the_vision_budget_and_re_arms_when_it_leaves():
    rig = Rig(cfg_with(PROXIMITY))
    rig.tick(3)
    for stamp, h in ((rig.t, 0.2), (rig.t + DT, 0.5), (rig.t + 2 * DT, 0.55)):  # near, still near
        rig.bus.publish("percept.face", face(stamp, h), stamp)
        rig.tick()
    assert rig.stats("w")["fired"] == 1  # an edge, not every frame
    rig.run(0.6)
    rig.bus.publish("percept.face", face(rig.t, 0.1), rig.t)  # it left
    rig.tick()
    rig.bus.publish("percept.face", face(rig.t, 0.5), rig.t)  # and came back
    rig.tick()
    assert rig.stats("w")["fired"] == 2
    lat = rig.engine.snapshot()["latency"]["proximity"]["command_ms"]
    assert lat["p95"] <= 250.0  # cm4 vision budget; npu is 100


def test_the_tallest_face_of_a_frame_counts():
    rig = Rig(cfg_with(PROXIMITY))
    rig.tick()
    rig.bus.publish("percept.face", face(rig.t, 0.1), rig.t)
    rig.bus.publish("percept.face", face(rig.t, 0.5, y1=0.1), rig.t)
    rig.tick()
    assert rig.stats("w")["fired"] == 1


@pytest.mark.parametrize("minor", [False, True])
def test_withdraw_never_moves_the_head_forward_even_with_the_minor_band_on(minor):
    # The head is tipped back at -0.5 rad. The arbiter scales an absolute target about home by the
    # minor factor, which would pull a naive withdraw back towards home, i.e. forward.
    rig = Rig(cfg_with(PROXIMITY), hold={PITCH: -0.5})
    rig.loop.set_minor_present(minor)
    rig.run(0.5)
    start = rig.pitch[-1]
    assert start == pytest.approx(-0.5 * (0.6 if minor else 1.0), abs=0.01)
    rig.bus.publish("percept.face", face(rig.t, 0.6), rig.t)
    n = len(rig.pitch)
    rig.run(0.8)
    during = rig.pitch[n:]
    assert rig.stats("w")["fired"] == 1
    assert max(during) <= start + 1e-6  # never forward of where it was
    assert min(during) < start - math.radians(2)  # and it did tip back


def test_withdraw_at_the_back_limit_holds_instead_of_going_forward():
    rig = Rig(cfg_with(PROXIMITY), hold={PITCH: -10.0})
    rig.run(0.5)
    start = rig.pitch[-1]
    rig.bus.publish("percept.face", face(rig.t, 0.6), rig.t)
    rig.run(0.6)
    assert max(rig.pitch[-30:]) <= start + 1e-6


# --------------------------------------------------------------------------- new voice

VOICE = rule(
    name="v",
    trigger="new_voice",
    primitive="turn_toward",
    threshold=3.0,
    amplitude_deg=40,
    cooldown_s=0,
    duration_ms=800,
    suppress_with_minor=True,
)


def peak_yaw(rig: Rig) -> float:
    """The largest head yaw written so far (the reflex hands the head back to idle at its end)."""
    return max(w[YAW] for _, w in rig.driver.writes)


def doa(rig: Rig, azimuth: float, head_yaw: float = 0.0) -> None:
    obs = Observation("unknown", rig.t, "doa", azimuth, 0.0, confidence=0.5)
    rig.bus.publish("percept.entity", obs, rig.t, joint_state={YAW: head_yaw})


def test_a_new_voice_turns_the_head_toward_it_within_the_amplitude_and_audio_budget():
    rig = Rig(cfg_with(VOICE))
    rig.loop.set_minor_present(False)
    rig.tick(3)
    doa(rig, 0.4)
    rig.tick()
    rig.run(1.0)
    assert rig.stats("v")["fired"] == 1
    assert rig.engine.snapshot()["latency"]["new_voice"]["command_ms"]["max"] <= 100.0
    assert peak_yaw(rig) == pytest.approx(0.4, abs=0.01)
    assert all(abs(w["body_yaw"]) < 1e-9 for _, w in rig.driver.writes)  # yaw only


def test_the_turn_is_limited_to_the_amplitude_and_the_front_half_plane():
    rig = Rig(cfg_with(VOICE))
    rig.loop.set_minor_present(False)
    rig.tick(3)
    doa(rig, 1.4)  # 80 degrees: past the 40 degree amplitude
    rig.run(1.2)
    assert peak_yaw(rig) == pytest.approx(math.radians(40), abs=0.01)
    # a bearing from the head's own turn (head at +60 deg, voice 50 deg left of it) is 110 deg in
    # the base frame: the target stays inside the front half-plane
    rig2 = Rig(cfg_with(replace_rule(VOICE, amplitude_deg=90)))
    rig2.loop.set_minor_present(False)
    rig2.tick(3)
    doa(rig2, math.radians(50), head_yaw=math.radians(60))
    rig2.run(1.5)
    assert 0.5 < peak_yaw(rig2) <= math.pi / 2 + 1e-9


def replace_rule(base: dict, **kw) -> dict:
    return base | kw


def test_the_turn_stays_inside_the_head_body_reach():
    rig = Rig(cfg_with(replace_rule(VOICE, amplitude_deg=90)))
    rig.engine.margin_rad = math.radians(60)  # 65 - 60 = 5 degrees of reach
    rig.loop.set_minor_present(False)
    rig.tick(3)
    doa(rig, 1.2)
    rig.run(1.5)
    assert 0 < peak_yaw(rig) <= math.radians(5) + 1e-6


def test_no_reflex_turns_toward_a_voice_while_a_minor_is_present():
    rig = Rig(cfg_with(VOICE))
    rig.loop.set_minor_present(True)
    rig.tick(3)
    doa(rig, 0.4)
    rig.run(1.0)
    stats = rig.stats("v")
    assert stats["fired"] == 0 and stats["suppressed_minor"] == 1
    assert rig.reflex_sources() == []
    assert rig.driver.writes[-1][1][YAW] == pytest.approx(0.0, abs=1e-9)


def test_the_minor_flag_gates_the_turn_even_if_a_rule_says_otherwise():
    rig = Rig(cfg_with(VOICE))
    object.__setattr__(rig.engine._rules[0], "suppress_with_minor", False)  # a bypassed validator
    rig.loop.set_minor_present(True)
    rig.tick(3)
    doa(rig, 0.4)
    rig.run(0.5)
    assert rig.stats("v")["fired"] == 0


def test_a_continuing_voice_is_not_new_but_one_after_a_silence_is():
    rig = Rig(cfg_with(VOICE))
    rig.loop.set_minor_present(False)
    rig.tick(3)
    for _ in range(10):  # speech for 0.2 s, one bearing every tick
        doa(rig, 0.3)
        rig.tick()
    rig.run(4.0)
    doa(rig, -0.3)
    rig.tick()
    assert rig.stats("v")["fired"] == 2


def test_other_observation_sources_do_not_trigger_the_voice_reflex():
    rig = Rig(cfg_with(VOICE))
    rig.loop.set_minor_present(False)
    rig.tick()
    obs = Observation("person", rig.t, "camera", 0.3, 0.0)
    rig.bus.publish("percept.entity", obs, rig.t, joint_state={YAW: 0.0})
    rig.tick()
    assert rig.stats("v")["fired"] == 0


def test_startle_is_allowed_with_a_minor_but_inside_the_scaled_band():
    rig = Rig(cfg_with(rule(amplitude_deg=20)))
    rig.loop.set_minor_present(True)
    rig.settle_audio()
    rig.engine.offer_audio_level(LOUD, rig.t)
    rig.run(0.6)
    assert rig.stats()["fired"] == 1
    assert max(rig.pitch) <= 1e-9
    assert -0.6 * math.radians(20) - 1e-6 <= min(rig.pitch) < -math.radians(3)


def test_the_global_amplitude_cap_limits_every_rule():
    rig = Rig(cfg_with(rule(amplitude_deg=40), max_amplitude_deg=5))
    rig.settle_audio()
    rig.engine.offer_audio_level(LOUD, rig.t)
    rig.run(0.5)
    assert min(rig.pitch) >= -math.radians(5) - 1e-6


def test_a_runtime_added_or_renamed_rule_is_refused_whole():
    approved = RoboEmotConfig.model_validate({"reflexes": {"rules": [rule(name="a")]}})
    added = RoboEmotConfig.model_validate(
        {"reflexes": {"rules": [rule(name="a"), rule(name="b", amplitude_deg=40, cooldown_s=0)]}}
    )
    renamed = RoboEmotConfig.model_validate(
        {"reflexes": {"rules": [rule(name="c", amplitude_deg=40, cooldown_s=0, duration_ms=800)]}}
    )
    for candidate, name in ((added, "b"), (renamed, "c")):
        paths = {v.path for v in violations(approved, candidate)}
        assert f"reflexes.rules.{name}.amplitude_deg" in paths
        assert f"reflexes.rules.{name}.cooldown_s" in paths
        assert f"reflexes.rules.{name}.duration_ms" in paths
    assert violations(approved, approved) == []


def cfg_rules(*rules: dict) -> RoboEmotConfig:
    return RoboEmotConfig.model_validate({"reflexes": {"rules": list(rules)}})


def flagged(approved: RoboEmotConfig, candidate: RoboEmotConfig) -> set[str]:
    return {v.path.rsplit(".", 1)[-1] for v in violations(approved, candidate)}


def test_a_runtime_layer_cannot_change_what_a_rule_is():
    approved = cfg_rules(rule(name="a"))
    swapped = cfg_rules(rule(name="a", primitive="withdraw"))
    retriggered = cfg_rules(
        rule(name="a", trigger="proximity", primitive="withdraw", threshold=0.4)
    )
    assert "primitive" in flagged(approved, swapped)
    assert {"trigger", "primitive"} <= flagged(approved, retriggered)


def test_antenna_ratio_may_only_fall():
    approved = RoboEmotConfig.model_validate({"reflexes": {"antenna_ratio": 2.0}})
    assert not violations(
        approved, RoboEmotConfig.model_validate({"reflexes": {"antenna_ratio": 1}})
    )
    assert violations(approved, RoboEmotConfig.model_validate({"reflexes": {"antenna_ratio": 5}}))


def test_enabling_a_disabled_rule_or_disabling_a_withdraw_is_refused():
    off = cfg_rules(rule(name="a", enabled=False))
    on = cfg_rules(rule(name="a", enabled=True))
    assert "enabled" in flagged(off, on)
    assert "enabled" not in flagged(on, off)  # switching a startle off is fine
    w_on = cfg_rules(dict(PROXIMITY, enabled=True))
    w_off = cfg_rules(dict(PROXIMITY, enabled=False))
    assert "enabled" in flagged(w_on, w_off)
    assert "enabled" not in flagged(w_off, w_off)


def test_the_threshold_may_only_move_the_safe_way_per_trigger():
    loud = cfg_rules(rule(name="a", threshold=6.0))
    assert "threshold" in flagged(loud, cfg_rules(rule(name="a", threshold=3.0)))
    assert "threshold" not in flagged(loud, cfg_rules(rule(name="a", threshold=9.0)))
    voice = cfg_rules(dict(VOICE, name="a", threshold=3.0))
    assert "threshold" in flagged(voice, cfg_rules(dict(VOICE, name="a", threshold=1.0)))
    assert "threshold" not in flagged(voice, cfg_rules(dict(VOICE, name="a", threshold=8.0)))
    close = cfg_rules(dict(PROXIMITY, threshold=0.4))
    assert "threshold" in flagged(close, cfg_rules(dict(PROXIMITY, threshold=0.6)))  # fires later
    assert "threshold" not in flagged(close, cfg_rules(dict(PROXIMITY, threshold=0.3)))


def test_a_minor_appearing_mid_turn_cuts_the_turn_off_on_the_next_tick():
    rig = Rig(cfg_with(VOICE))
    rig.loop.set_minor_present(False)
    rig.tick(3)
    doa(rig, 0.6)
    rig.run(0.2)
    assert rig.reflex_sources() == ["reflex.v"]
    yaw_before = rig.driver.writes[-1][1][YAW]
    rig.loop.set_minor_present(True)
    rig.tick()
    assert rig.reflex_sources() == [] and rig.engine.active() == []
    assert rig.stats("v")["suppressed_minor"] == 1
    rig.run(0.5)
    assert max(w[YAW] for _, w in rig.driver.writes[-25:]) <= yaw_before + 1e-9  # no further turn


def test_a_minor_appearing_cuts_off_a_suppressible_startle_but_not_a_plain_one():
    rig = Rig(cfg_with(rule(name="s", suppress_with_minor=True), rule(name="p")))
    rig.settle_audio()
    rig.engine.offer_audio_level(LOUD, rig.t)
    rig.tick(2)
    assert sorted(rig.reflex_sources()) == ["reflex.p", "reflex.s"]
    rig.loop.set_minor_present(True)
    rig.tick()
    assert rig.reflex_sources() == ["reflex.p"]


def test_a_close_face_is_judged_against_the_vision_age_not_the_audio_age():
    for age, fires in ((0.2, True), (0.4, False)):
        rig = Rig(cfg_with(PROXIMITY))
        rig.t = 10.0
        rig.bus.publish("percept.face", face(rig.t - age, 0.6), rig.t - age)
        rig.tick()
        assert rig.stats("w")["fired"] == int(fires), age
        assert rig.stats("w")["dropped_stale"] == int(not fires)
