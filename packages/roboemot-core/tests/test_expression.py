"""SPEC 7, 10, 11A, 12: the expression map, the renderer and the minor cap over it."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from roboemot.affect import AffectEngine, AffectState
from roboemot.affect.clips import load_clip_library
from roboemot.affect.expression import SOURCE, ExpressionRenderer, build_primitive
from roboemot.affect.register import Person, resolve_register
from roboemot.config import ConfigStack
from roboemot.config.schema import ExpressionEntry, RegisterParams, RoboEmotConfig
from roboemot.manifest import load_manifest
from roboemot.motion import MotionCore, Priority

MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
)
CLIPS = load_clip_library(load_manifest(MANIFEST, allow_unmeasured_dynamics=True))
LABELS = ("calm", "joy", "curiosity", "surprise", "sadness", "fear", "annoyance", "sleepy")
DT = 0.02


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST, allow_unmeasured_dynamics=True)


@pytest.fixture(scope="module")
def cfg(manifest) -> RoboEmotConfig:
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest)
    )
    return stack.effective()


def state(label="joy", intensity=0.5, impulse=0) -> AffectState:
    return AffectState(0.0, 0.0, label, intensity, impulse)


DEFAULTS = RoboEmotConfig()


def active(group="adult", cfg=DEFAULTS, **params):
    people = [Person(group, "declared", params)]
    return resolve_register(people, cfg.register, cfg.persona)


def renderer(cfg, manifest, expression_map=None, **expr) -> ExpressionRenderer:
    ecfg = cfg.expression.model_copy(update=expr)
    return ExpressionRenderer(
        ecfg,
        expression_map or cfg.expression_map,
        manifest.dofs,
        0.15,
        body_yaw="body_yaw",
        clips=CLIPS,
        rng=random.Random(0),
    )


def entry(**channels) -> ExpressionEntry:
    return ExpressionEntry(channels=channels)


SURPRISE = entry(
    antenna_right=[(0, 0), (0.2, 25), (0.6, 25), (1.2, 0)],
    antenna_left=[(0, 0), (0.2, -25), (0.6, -25), (1.2, 0)],
    **{"head.pitch": [(0, 0), (0.2, -6), (0.6, -6), (1.2, 0)]},
)
SADNESS = entry(
    antenna_right=[(0, 0), (1.2, -20), (2.8, -20), (4.0, 0)],
    antenna_left=[(0, 0), (1.2, 20), (2.8, 20), (4.0, 0)],
    **{"head.pitch": [(0, 0), (1.2, 6), (2.8, 6), (4.0, 0)]},
)


# ------------------------------------------------------------------ the map


def test_the_reachy_default_map_covers_all_labels(cfg, manifest):
    assert set(cfg.expression_map) == set(LABELS) == set(cfg.affect.labels)
    angles = {n for n, d in manifest.dofs.items() if d.kind == "angle"}
    for label, e in cfg.expression_map.items():
        assert e.variants and not e.channels, label  # every label is a set of recorded clips
        for v in e.variants:
            assert v.clip in CLIPS, (label, v.clip)
            clip = CLIPS[v.clip]
            prim = build_primitive(v.clip, clip, manifest.dofs, body_yaw="body_yaw")
            assert prim is not None and prim.additive
            assert prim.keyframes[0].values == dict.fromkeys(prim.keyframes[0].values, 0.0)
            assert prim.keyframes[-1].values == dict.fromkeys(prim.keyframes[-1].values, 0.0)
            assert prim.duration == pytest.approx(max(f[-1][0] for f in clip.channels.values()))
            assert set(clip.channels) <= angles  # no LED and no translation anywhere


def test_degrees_in_the_file_are_radians_in_the_primitive(manifest):
    prim = build_primitive("surprise", SURPRISE, manifest.dofs, body_yaw="body_yaw")
    peak = max(prim.keyframes, key=lambda k: k.values["antenna_right"])
    assert peak.values["antenna_right"] == pytest.approx(math.radians(25))
    assert peak.values["antenna_left"] == pytest.approx(math.radians(-25))
    assert prim.sample(0.4)["head.pitch"] == pytest.approx(math.radians(-6))  # up


def test_antenna_signs_follow_the_connectivity_gestures(cfg, manifest):
    """Spread outward is right -, left +; lifted toward vertical is the other way (health)."""
    for gesture, sign in (("puzzled", -1), ("perk_up", 1)):
        prim = build_primitive(
            gesture, cfg.gesture_map[gesture], manifest.dofs, body_yaw="body_yaw"
        )
        mid = prim.sample(prim.duration / 2)
        assert mid["antenna_right"] * sign > 0 and mid["antenna_left"] * sign < 0


def test_repeat_unrolls_the_cycle_and_merges_the_seams():
    e = ExpressionEntry(channels={"antenna_right": [(0, 0), (0.1, 10), (0.2, 0)]}, repeat=3)
    dofs = load_manifest(MANIFEST, allow_unmeasured_dynamics=True).dofs
    prim = build_primitive("joy", e, dofs, body_yaw="body_yaw")
    assert [round(k.t, 3) for k in prim.keyframes] == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert prim.duration == pytest.approx(0.6)


def test_channels_are_merged_onto_one_time_grid(manifest):
    e = entry(
        **{"antenna_right": [(0, 0), (1, 10), (2, 0)], "head.pitch": [(0, 0), (0.5, 4), (2, 0)]}
    )
    prim = build_primitive("joy", e, manifest.dofs, body_yaw="body_yaw")
    assert [k.t for k in prim.keyframes] == [0, 0.5, 1, 2]
    assert prim.sample(1.0)["head.pitch"] == pytest.approx(math.radians(4) * (1.0 / 1.5))
    assert prim.sample(0.5)["antenna_right"] == pytest.approx(math.radians(5))


def test_channels_the_robot_lacks_are_skipped(manifest):
    e = entry(**{"antenna_right": [(0, 0), (1, 10), (2, 0)], "head.roll": [(0, 0), (1, 5), (2, 0)]})
    head_only = {n: d for n, d in manifest.dofs.items() if n.startswith("head")}
    prim = build_primitive("joy", e, head_only, body_yaw="body_yaw")
    assert set(prim.keyframes[0].values) == {"head.roll"}
    assert (
        build_primitive(
            "joy", entry(antenna_right=[(0, 0), (1, 0)]), head_only, body_yaw="body_yaw"
        )
        is None
    )
    assert build_primitive("joy", e, {}, body_yaw="body_yaw") is None


def test_schema_rejects_bad_maps(cfg):
    ok = [(0, 0), (1, 10), (2, 0)]
    bad = (
        {"channels": {"antenna_right": ok}, "speed": 2},  # unknown key
        {"channels": {"antenna_right": [(0, 0), (1, 10), (2, 5)]}},  # does not return to zero
        {"channels": {"antenna_right": [(0.1, 0), (1, 0)]}},  # does not start at t=0
        {"channels": {"antenna_right": [(0, 0), (1, 0), (0.5, 0)]}},  # time goes back
        {"channels": {"antenna_right": ok, "head.roll": [(0, 0), (3, 0)]}},  # ends differ
        {"channels": {"antenna_right": ok}, "additive": False},
        {"channels": {"antenna_right": ok}, "curve": "wobbly"},
        {"channels": {}},
    )
    for raw in bad:
        with pytest.raises(ValidationError):
            ExpressionEntry.model_validate(raw)
    with pytest.raises(ValidationError, match="not in affect.labels"):
        RoboEmotConfig.model_validate(
            {"expression_map": {"rage": {"channels": {"antenna_right": ok}}}}
        )
    with pytest.raises(ValidationError):
        RoboEmotConfig.model_validate({"expression": {"intensity_factor": {"extreme": 1}}})


def test_the_manifest_ships_the_map_and_yaml_loads_it(cfg):
    raw = yaml.safe_load(MANIFEST.read_text())["behaviour"]["expression_map"]
    assert set(raw) == set(LABELS)
    assert all(set(e) == {"variants"} for e in raw.values())
    assert {v.clip for v in cfg.expression_map["joy"].variants} >= {"cheerful1", "laughing1"}


# ------------------------------------------------------------------ triggers


def play_for(r, st, reg, t0, seconds, step=0.05):
    starts, t = 0, t0
    while t < t0 + seconds:
        starts += r.update(st, reg, t)
        r.commands(t)
        t += step
    return starts


def test_a_label_change_starts_an_expression_and_a_held_label_does_not_spam(cfg, manifest):
    r = renderer(cfg, manifest)
    reg = active("adult")
    assert not r.update(state("calm", 0.0), reg, 0.0)  # the first state is only noted
    assert r.update(state("joy", 0.4), reg, 1.0)
    assert r.last_label == "joy" and r.playing is False  # commands() not yet called
    assert r.commands(1.0)
    # the same label held for 25 s: no replay before retrigger_s
    assert play_for(r, state("joy", 0.4), reg, 1.05, 25) == 0
    # past retrigger_s it plays again, once
    assert play_for(r, state("joy", 0.4), reg, 26.0, 8) == 1


def test_a_faint_held_label_is_not_replayed(cfg, manifest):
    r = renderer(cfg, manifest)
    reg = active("adult")
    r.update(state("calm", 0.0), reg, 0.0)
    assert play_for(r, state("calm", 0.01), reg, 1.0, 100) == 0


def test_a_repeated_labelled_impulse_starts_again_but_not_within_the_min_interval(cfg, manifest):
    r = renderer(cfg, manifest)
    reg = active("adult")
    r.update(state("joy", 0.4, impulse=1), reg, 0.0)
    assert r.update(state("joy", 0.4, impulse=2), reg, 5.0)  # a new impulse, same label
    assert not r.update(state("joy", 0.4, impulse=3), reg, 5.2)  # too soon: held back
    assert r.update(state("joy", 0.4, impulse=3), reg, 5.2 + cfg.expression.min_interval_s)


@pytest.mark.parametrize("label", ["calm", "joy", "curiosity", "sleepy"])
def test_a_self_soothe_event_starts_the_clip_of_its_label_through_engine_and_renderer(
    cfg, manifest, label
):
    engine = AffectEngine(cfg.affect, cfg.appraisal)
    r = renderer(cfg, manifest)
    reg = active("adult")
    r.update(engine.state(0.0), reg, 0.0)  # the first state is only noted
    assert engine.on_event("self.soothe", 10.0, {"label": label, "sound": "hum"})
    assert r.update(engine.state(10.0), reg, 10.0)
    assert r.last_label == label
    assert r.commands(10.0)  # the clip is playing


def test_a_label_without_a_map_entry_or_a_usable_channel_plays_nothing(cfg, manifest):
    sparse = {"joy": entry(antenna_right=[(0, 0), (0.5, 10), (1, 0)])}
    r = renderer(cfg, manifest, sparse)
    reg = active("adult")
    r.update(state("calm", 0.4), reg, 0.0)
    assert not r.update(state("sadness", 0.4), reg, 2.0)
    head_only = {n: d for n, d in manifest.dofs.items() if n.startswith("head")}
    r2 = ExpressionRenderer(
        cfg.expression, sparse, head_only, 0.15, body_yaw="body_yaw", clips=CLIPS
    )
    r2.update(state("calm", 0.4), reg, 0.0)
    assert not r2.update(state("joy", 0.4), reg, 2.0)
    assert r2.commands(2.0) == []


# ------------------------------------------------------------------ amplitude


def test_amplitude_scales_by_register_child_adult_unknown(cfg, manifest):
    r = renderer(cfg, manifest)
    st = state("joy", 0.5)
    child, adult = r.amplitude(active("child"), st), r.amplitude(active("adult"), st)
    unknown = r.amplitude(resolve_unknown(cfg), st)
    assert child > adult > unknown > 0
    f = cfg.expression.intensity_factor
    floor = cfg.expression.intensity_floor
    affect = floor + (1 - floor) * 0.5
    assert child == pytest.approx(1.2 * f.high * affect)
    assert adult == pytest.approx(1.0 * f.medium * affect)
    assert unknown == pytest.approx(0.7 * f.low * affect)


def resolve_unknown(cfg):
    return resolve_register([Person("unknown", "none", {})], cfg.register, cfg.persona)


def test_amplitude_follows_affect_intensity_circadian_and_the_ceiling(cfg, manifest):
    r = renderer(cfg, manifest)
    reg = active("adult")
    floor = cfg.expression.intensity_floor
    assert r.amplitude(reg, state(intensity=0.0)) == pytest.approx(floor)
    assert r.amplitude(reg, state(intensity=1.0)) == pytest.approx(1.0)
    r.circadian_factor = lambda: 0.5
    assert r.amplitude(reg, state(intensity=1.0)) == pytest.approx(0.5)
    r.circadian_factor = lambda: 100.0
    assert r.amplitude(reg, state(intensity=1.0)) == cfg.expression.max_amplitude


def test_persona_is_in_the_register_and_not_applied_again(cfg, manifest):
    lively = RoboEmotConfig.model_validate({"persona": {"traits": {"energy": 1.0}}})
    shaped = resolve_register([Person("adult", "declared", {})], lively.register, lively.persona)
    r = renderer(cfg, manifest)
    st = state("joy", 1.0)
    # 1.2 (persona energy) x 1.0 (adult) x 1.0 (medium) x 1.0 (full intensity), once
    assert r.amplitude(shaped, st) == pytest.approx(1.2)


def test_the_started_player_carries_the_amplitude_at_expression_priority(cfg, manifest):
    r = renderer(cfg, manifest)
    r.update(state("calm", 0.0), active("child"), 0.0)
    assert r.update(state("joy", 0.5), active("child"), 1.0)
    cmd = r.commands(1.2)[0]
    assert cmd.source == SOURCE and cmd.priority == Priority.EXPRESSION and cmd.additive
    assert cmd.expires == pytest.approx(1.2 + cfg.expression.command_ttl_s)
    assert r.last_amplitude == pytest.approx(r.amplitude(active("child"), state("joy", 0.5)))


# ------------------------------------------------------------------ playing


def drain(r, t0, seconds):
    out, t = [], t0
    while t < t0 + seconds:
        out.append((t, r.commands(t)))
        t += DT
    return out


def test_an_expression_ends_at_zero_and_stops_feeding(cfg, manifest):
    r = renderer(cfg, manifest, {"surprise": SURPRISE})
    reg = active("adult")
    r.update(state("calm", 0.0), reg, 0.0)
    r.update(state("surprise", 0.5), reg, 1.0)
    frames = drain(r, 1.0, 2.0)
    fed = [(t, c[0].targets) for t, c in frames if c]
    assert fed and max(abs(v) for _, tg in fed for v in tg.values()) > 0.1
    assert all(not c for t, c in frames if t > 1.0 + 1.2 + DT)  # nothing after the primitive
    assert not r.playing


def test_a_new_expression_blends_in_from_the_current_offset(cfg, manifest):
    r = renderer(cfg, manifest, {"sadness": SADNESS, "surprise": SURPRISE})
    reg = active("adult")
    r.update(state("calm", 0.0), reg, 0.0)
    r.update(state("sadness", 0.5), reg, 1.0)
    drain(r, 1.0, 1.6)  # mid-droop
    before = r.commands(2.6)[0].targets["antenna_right"]
    assert before < -0.05
    assert r.update(state("surprise", 0.5), reg, 2.6)
    first = r.commands(2.6)[0].targets["antenna_right"]
    assert first == pytest.approx(before, abs=1e-6)  # no jump at the switch
    later = r.commands(2.6 + 0.15 + DT)[0].targets["antenna_right"]
    assert later > 0  # the surprise lift has taken over


def test_offsets_are_clamped_to_the_manifest_range_about_home(cfg, manifest):
    huge = entry(
        **{"antenna_right": [(0, 0), (0.5, 170), (1, 0)], "head.pitch": [(0, 0), (0.5, 90), (1, 0)]}
    )
    r = renderer(cfg, manifest, {"joy": huge}, max_amplitude=3)
    reg = active("child")
    r.update(state("calm", 1.0), reg, 0.0)
    r.update(state("joy", 1.0), reg, 1.0)
    for _, cmds in drain(r, 1.0, 1.0):
        for c in cmds:
            for name, v in c.targets.items():
                d = manifest.dofs[name]
                assert d.min - d.home - 1e-9 <= v <= d.max - d.home + 1e-9


# ------------------------------------------------------------------ safety


BODY = entry(
    **{"body_yaw": [(0, 0), (0.5, 20), (1, 0)], "antenna_right": [(0, 0), (0.5, 10), (1, 0)]}
)


def test_no_body_yaw_for_a_register_that_must_not_approach(cfg, manifest):
    """SPEC 7, 12: a register with approach not allowed never gets a body turn."""
    for reg in (active("child"), resolve_unknown(cfg), active("teen", approach_allowed=False)):
        assert not reg.params.approach_allowed
        r = renderer(cfg, manifest, {"joy": BODY})
        r.update(state("calm", 0.5), reg, 0.0)
        assert r.update(state("joy", 0.5), reg, 1.0)
        seen = {n for _, cmds in drain(r, 1.0, 1.2) for c in cmds for n in c.targets}
        assert "antenna_right" in seen and "body_yaw" not in seen
    adult = active("adult")
    r = renderer(cfg, manifest, {"joy": BODY})
    r.update(state("calm", 0.5), adult, 0.0)
    r.update(state("joy", 0.5), adult, 1.0)
    assert any("body_yaw" in c.targets for _, cmds in drain(r, 1.0, 1.2) for c in cmds)


def test_expressions_never_translate_the_robot(cfg, manifest):
    angle = {n for n, d in manifest.dofs.items() if d.kind == "angle"}
    assert angle < set(manifest.dofs)  # the manifest does have translations
    for label in LABELS:
        r = renderer(cfg, manifest)
        reg = active("adult")
        r.update(state("calm", 0.0), reg, 0.0)
        r.update(state(label if label != "calm" else "joy", 0.5), reg, 1.0)
        for _, cmds in drain(r, 1.0, 6.0):
            assert all(set(c.targets) <= angle for c in cmds)


def run_core(manifest, cfg, reg, st, minor, entry_map=None):
    """Play ``st`` through the arbiter and limiter; return the largest deviation per dof."""
    core = MotionCore(
        manifest.dofs,
        manifest.constraints,
        crossfade_s=cfg.motion.arbiter.crossfade_ms / 1000,
        watchdog_s=10.0,
        minor_factor=cfg.safety.minor_motion_factor,
        watchdog_return_factor=cfg.safety.watchdog_return_factor,
    )
    core.minor_present = minor
    r = renderer(cfg, manifest, entry_map, max_amplitude=3)
    r.update(state("calm", 0.0), reg, 0.0)
    r.update(st, reg, 1.0)
    up = dict.fromkeys(manifest.dofs, 0.0)
    down = dict.fromkeys(manifest.dofs, 0.0)
    t = 1.0
    while t < 9.0:
        for cmd in r.commands(t):
            core.feed(cmd)
        pos = core.tick(t, DT)
        for n, d in manifest.dofs.items():
            up[n] = max(up[n], pos[n] - d.home)
            down[n] = max(down[n], d.home - pos[n])
        t += DT
    return up, down


def within_band(manifest, factor, up, down, names=None):
    """Each side of home stays inside ``factor`` x that side's range (signed, not the wider)."""
    for n in names or manifest.dofs:
        d = manifest.dofs[n]
        assert up[n] <= factor * (d.max - d.home) + 1e-9, (n, "up")
        assert down[n] <= factor * (d.home - d.min) + 1e-9, (n, "down")


def test_the_minor_cap_wins_over_the_child_register(cfg, manifest):
    """SPEC 7, 12: the register scales first; the minor cap, applied after, always wins."""
    factor = cfg.safety.minor_motion_factor
    assert factor < 1
    reg = active("child")
    assert reg.params.motion_amplitude > 1  # the child's 1.2x
    for label in LABELS:
        st = state(label, 1.0, impulse=1)
        up, down = run_core(manifest, cfg, reg, st, minor=True)
        free_up, free_down = run_core(manifest, cfg, reg, st, minor=False)
        within_band(manifest, factor, up, down)
        moved = {n for n in up if max(up[n], down[n]) > 1e-4}
        assert moved, label
        assert all(up[n] <= free_up[n] + 1e-9 and down[n] <= free_down[n] + 1e-9 for n in moved)


@pytest.mark.parametrize("clip", sorted(CLIPS))
def test_every_recorded_clip_stays_inside_the_minor_share_of_the_range(cfg, manifest, clip):
    """SPEC 7, 12: the minor cap holds for each vendored clip at the child's amplitude."""
    factor = cfg.safety.minor_motion_factor
    only = ExpressionEntry(variants=[{"clip": clip}])
    reg = active("child")
    up, down = run_core(manifest, cfg, reg, state("joy", 1.0), True, {"joy": only})
    within_band(manifest, factor, up, down)


def test_the_minor_cap_holds_at_the_worst_case_the_schema_allows(cfg, manifest):
    factor = cfg.safety.minor_motion_factor
    worst = RoboEmotConfig.model_validate(
        {"expression": {"intensity_factor": {"low": 2, "medium": 2, "high": 2}, "max_amplitude": 3}}
    )
    reg = resolve_register(
        [Person("child", "declared", {"motion_amplitude": 2, "reaction_intensity": "high"})],
        worst.register,
        worst.persona,
    )
    assert reg.params.motion_amplitude == 2
    huge = entry(**{
        "antenna_right": [(0, 0), (0.5, 170), (1, 0)],
        "antenna_left": [(0, 0), (0.5, -170), (1, 0)],
        "head.pitch": [(0, 0), (0.5, 90), (1, 0)],
        "head.roll": [(0, 0), (0.5, -90), (1, 0)],
    })  # fmt: skip
    up, down = run_core(manifest, worst, reg, state("joy", 1.0), True, {"joy": huge})
    names = ("antenna_right", "antenna_left", "head.pitch", "head.roll")
    within_band(manifest, factor, up, down, names)
    assert all(max(up[n], down[n]) > 0 for n in names)
    assert RegisterParams(motion_amplitude=2).motion_amplitude == 2


def test_a_reload_swaps_the_map_and_parameters(cfg, manifest):
    r = renderer(cfg, manifest)
    reg = active("adult")
    r.update(state("calm", 0.0), reg, 0.0)
    r.update(state("joy", 0.5), reg, 1.0)
    first = r.last_amplitude
    quiet = cfg.expression.model_copy(
        update={
            "intensity_factor": cfg.expression.intensity_factor.model_copy(update={"medium": 0.5})
        }
    )
    r.apply_config(quiet, {}, 0.15)
    assert not r.update(state("sadness", 0.5), reg, 9.0)  # the new map has no entries
    r.apply_config(quiet, cfg.expression_map, 0.15)
    assert r.update(state("joy", 0.5, impulse=1), reg, 20.0)
    assert r.last_amplitude == pytest.approx(first / 2)


def test_the_manifest_behaviour_becomes_part_of_the_installed_config(manifest):
    stack = ConfigStack()
    assert stack.effective().expression_map == {}
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest)
    )
    assert set(stack.effective().expression_map) == set(LABELS)
    # an app layer merges channel by channel into the manifest's entry
    stack.app = {"expression_map": {"joy": {"channels": {"head.roll": [[0, 0], [0.45, 0]]}}}}
    joy = stack.effective().expression_map["joy"]
    assert set(joy.channels) == {"head.roll"} and not joy.variants  # channels drop the clips


def test_set_register_acts_within_one_tick_even_when_the_affect_updates_are_rare(cfg, manifest):
    """Regression: update_hz = 1 means no update() for a second; the register still lands."""
    slow = cfg.model_copy(update={"affect": cfg.affect.model_copy(update={"update_hz": 1.0})})
    r = renderer(slow, manifest, {"joy": BODY})
    adult = active("adult")
    r.update(state("calm", 0.5), adult, 0.0)
    assert r.update(state("joy", 0.5), adult, 1.0)
    assert "body_yaw" in r.commands(1.25)[0].targets
    r.set_register(resolve_unknown(cfg))  # no update() call: the next state is a second away
    assert "body_yaw" not in r.commands(1.27)[0].targets


def test_a_lowered_ceiling_applies_to_the_expression_already_playing(cfg, manifest):
    r = renderer(cfg, manifest)
    r.update(state("calm", 0.5), active("child"), 0.0)
    r.update(state("surprise", 1.0), active("child"), 1.0)
    full = r.commands(1.3)[0].targets["antenna_right"]
    assert r.last_amplitude > 1.0
    r.apply_config(
        cfg.expression.model_copy(update={"max_amplitude": 0.5}), cfg.expression_map, 0.15
    )
    capped = r.commands(1.3)[0].targets["antenna_right"]
    assert capped == pytest.approx(full * 0.5 / r.last_amplitude)


def test_the_renderer_keeps_its_offsets_inside_the_minor_share_of_the_range(cfg, manifest):
    huge = entry(**{"antenna_right": [(0, 0), (0.5, 170), (1, 0)]})
    r = renderer(cfg, manifest, {"joy": huge}, max_amplitude=3)
    reg = active("child")
    r.update(state("calm", 1.0), reg, 0.0)
    r.update(state("joy", 1.0), reg, 1.0)
    r.set_register(reg, 0.6)
    d = manifest.dofs["antenna_right"]
    assert r.commands(1.5)[0].targets["antenna_right"] <= 0.6 * (d.max - d.home) + 1e-9
    r.set_register(reg, 1.0)
    assert r.commands(1.5)[0].targets["antenna_right"] > 0.6 * (d.max - d.home)


def test_keyframe_times_are_bounded():
    with pytest.raises(ValidationError, match="longer than"):
        ExpressionEntry(channels={"antenna_right": [(0, 0), (500, 0)]})
    with pytest.raises(ValidationError):
        ExpressionEntry(channels={"antenna_right": [(0, 0), (1, 0)]}, repeat=21)


def test_manifest_behaviour_may_only_set_known_keys_and_is_audited(manifest):
    from roboemot.config import ConfigError

    stack = ConfigStack()
    with pytest.raises(ConfigError, match="may set"):
        stack.set_manifest_behaviour({"safety": {"minor_motion_factor": 1.0}}, manifest.channels)
    assert stack.manifest == {} and stack.effective().safety.minor_motion_factor == 0.6
    with pytest.raises(ConfigError):
        stack.set_manifest_behaviour(
            {"expression_map": {"rage": {"channels": {}}}}, manifest.channels
        )
    assert stack.manifest == {}
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest)
    )
    assert stack.audit[-1].source == "manifest" and stack.audit[-1].new == [
        "circadian_motion",
        "expression_map",
        "gesture_map",
        "voice",
    ]


def test_a_stricter_register_mid_expression_drops_the_body_and_the_amplitude_at_once(cfg, manifest):
    """Regression: the approach and amplitude decisions are re-checked every tick."""
    r = renderer(cfg, manifest, {"joy": BODY})
    adult, stranger = active("adult"), resolve_unknown(cfg)
    r.update(state("calm", 0.5), adult, 0.0)
    assert r.update(state("joy", 0.5), adult, 1.0)
    assert "body_yaw" in r.commands(1.25)[0].targets
    r.update(state("joy", 0.5), stranger, 1.26)  # someone arrives: tighter on both counts
    after = r.commands(1.27)[0].targets
    assert "body_yaw" not in after
    f = cfg.expression.intensity_factor
    ratio = 0.7 * f.low / (1.0 * f.medium)
    assert after["antenna_right"] == pytest.approx(
        r._player.command(1.27).targets["antenna_right"] * ratio
    )
    # the cross-fade's previous pose is the filtered one: a new expression starting now keeps
    # the body out while blending
    assert r.update(state("joy", 0.5, impulse=9), stranger, 3.0)
    for t in (3.0, 3.05, 3.1, 3.2):
        assert all("body_yaw" not in c.targets for c in r.commands(t))


def test_started_counts_every_expression_the_renderer_starts(cfg, manifest):
    """The counter tests wait on: it rises once per started expression, not per tick."""
    r = renderer(cfg, manifest)
    r.update(state("calm", 0.5), active(), 0.0)  # the first state only sets the label
    assert r.started == 0
    assert r.update(state("joy", 0.5, impulse=1), active(), 2.0)
    assert r.started == 1
    r.update(state("joy", 0.5, impulse=1), active(), 2.1)  # nothing new: no start
    assert r.started == 1
    assert r.update(state("curiosity", 0.5, impulse=2), active(), 8.0)
    assert r.started == 2


def test_a_requested_gesture_plays_through_the_same_path_as_an_affect_one(cfg, manifest):
    """SPEC 11: the play skill picks the label; the interval, register and minor cap still rule."""
    r = renderer(cfg, manifest)
    reg = active("adult")
    assert r.request("nonsense") is False and r.request("joy") is True
    assert r.update(state("calm", 0.5), reg, 0.0) is True
    assert r.last_label == "joy" and r.started == 1
    r.request("curiosity")
    assert r.update(state("calm", 0.5), reg, 0.2) is False  # min_interval_s not over
    assert r.update(state("calm", 0.5), reg, 1.5) is True and r.last_label == "curiosity"
    assert r.update(state("calm", 0.5), reg, 3.0) is False  # one request, one play


def test_a_requested_gesture_respects_the_minor_cap_and_the_register(cfg, manifest):
    factor = cfg.safety.minor_motion_factor
    huge = entry(**{"antenna_right": [(0, 0), (0.5, 170), (1, 0)]})
    r = renderer(cfg, manifest, {"joy": huge}, max_amplitude=3)
    reg = active("child")
    r.set_register(reg, factor)
    r.request("joy")
    assert r.update(state("calm", 1.0), reg, 0.0)
    d = manifest.dofs["antenna_right"]
    assert r.commands(0.5)[0].targets["antenna_right"] <= factor * (d.max - d.home) + 1e-9


# ------------------------------------------------------------------ named gestures (SPEC 7, 11)


def test_the_manifest_ships_nod_and_shake_apart_from_the_affect_labels(cfg, manifest):
    assert {"nod", "shake"} <= set(cfg.gesture_map)
    assert not set(cfg.gesture_map) & set(cfg.affect.labels)
    nod, shake = cfg.gesture_map["nod"], cfg.gesture_map["shake"]
    assert set(nod.channels) == {"head.pitch"} and nod.repeat >= 2
    assert set(shake.channels) == {"head.yaw"} and shake.repeat >= 2
    for e in (nod, shake):  # data within the manifest range, in degrees
        (frames,) = e.channels.values()
        d = manifest.dofs["head.pitch" if e is nod else "head.yaw"]
        peak = math.radians(max(abs(v) for _, v in frames))
        assert 0 < peak <= min(d.max - d.home, d.home - d.min)


def test_the_schema_rejects_bad_gestures():
    ok = [(0, 0), (1, 10), (2, 0)]
    with pytest.raises(ValidationError, match="affect labels"):
        RoboEmotConfig.model_validate({"gesture_map": {"joy": {"channels": {"head.yaw": ok}}}})
    with pytest.raises(ValidationError):
        RoboEmotConfig.model_validate(
            {"gesture_map": {"nod": {"channels": {"head.yaw": [(0, 0), (1, 10)]}}}}
        )


def test_the_manifest_may_set_gesture_map_and_it_is_audited(manifest):
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest)
    )
    assert "gesture_map" in stack.audit[-1].new
    assert set(stack.effective().gesture_map) >= {"nod", "shake"}


def gesture_renderer(cfg, manifest, **expr):
    ecfg = cfg.expression.model_copy(update=expr)
    return ExpressionRenderer(
        ecfg,
        cfg.expression_map,
        manifest.dofs,
        0.15,
        cfg.gesture_map,
        body_yaw="body_yaw",
        clips=CLIPS,
    )


def test_play_offers_affect_labels_and_gestures_and_refuses_unknown_names(cfg, manifest):
    r = gesture_renderer(cfg, manifest)
    assert r.playable() == [*cfg.expression_map, "nod", "shake", "puzzled", "perk_up"]
    assert r.request("nod") and not r.request("wave")
    assert (
        ExpressionRenderer(
            cfg.expression,
            cfg.expression_map,
            manifest.dofs,
            0.15,
            body_yaw="body_yaw",
            clips=CLIPS,
        ).request("nod")
        is False
    )  # a robot without a gesture_map offers none


def test_a_nod_oscillates_head_pitch_within_limits_and_returns_to_zero(cfg, manifest):
    r = gesture_renderer(cfg, manifest)
    reg = active("adult")
    r.request("nod")
    assert r.update(state("calm", 0.5), reg, 0.0) and r.last_label == "nod"
    d = manifest.dofs["head.pitch"]
    values = []
    for _, cmds in drain(r, 0.0, 3.0):
        for c in cmds:
            assert set(c.targets) == {"head.pitch"} and c.priority == Priority.EXPRESSION
            values.append(c.targets["head.pitch"])
            assert d.min - d.home <= values[-1] <= d.max - d.home
    signs = [v > 1e-4 for v in values if abs(v) > 1e-4]
    flips = sum(a != b for a, b in zip(signs, signs[1:], strict=False))
    assert max(values) > 0.05 and flips >= 3  # dips and lifts, twice
    assert not r.playing


def test_a_shake_oscillates_head_yaw_and_a_stricter_register_scales_it(cfg, manifest):
    peaks = {}
    for group in ("adult", "child"):
        r = gesture_renderer(cfg, manifest, max_amplitude=3)
        r.request("shake")
        reg = active(group)
        assert r.update(state("calm", 1.0), reg, 0.0)
        ys = [c.targets["head.yaw"] for _, cmds in drain(r, 0.0, 3.0) for c in cmds]
        assert min(ys) < -0.05 and max(ys) > 0.05  # both sides
        peaks[group] = max(ys)
    assert peaks["child"] > peaks["adult"]  # the register's 1.2x applies, as for expressions


def test_gestures_respect_the_interval_and_the_cap_on_amplitude(cfg, manifest):
    r = gesture_renderer(cfg, manifest)
    reg = active("adult")
    assert r.update(state("joy", 0.5), reg, 0.0) is False  # first state sets the label
    r.request("nod")
    assert r.update(state("joy", 0.5), reg, 0.1) is True
    r.request("shake")
    assert r.update(state("joy", 0.5), reg, 0.2) is False  # min_interval_s not over
    assert r.update(state("joy", 0.5), reg, 2.0) is True and r.last_label == "shake"
    assert r.last_amplitude <= cfg.expression.max_amplitude


def test_an_oversized_gesture_is_clamped_to_the_range_and_the_minor_cap(cfg, manifest):
    factor = cfg.safety.minor_motion_factor
    huge = entry(**{"head.pitch": [(0, 0), (0.5, 90), (1, 0)]})
    gestures = {"nod": huge}
    d = manifest.dofs["head.pitch"]
    # renderer alone: clamped to the range about home
    r = ExpressionRenderer(
        cfg.expression.model_copy(update={"max_amplitude": 3}),
        cfg.expression_map,
        manifest.dofs,
        0.15,
        gestures,
        body_yaw="body_yaw",
        clips=CLIPS,
    )
    r.request("nod")
    assert r.update(state("calm", 1.0), active("child"), 0.0)
    assert r.commands(0.5)[0].targets["head.pitch"] <= d.max - d.home + 1e-9
    # through the arbiter and limiter, with a minor present
    core = MotionCore(
        manifest.dofs,
        manifest.constraints,
        crossfade_s=cfg.motion.arbiter.crossfade_ms / 1000,
        watchdog_s=10.0,
        minor_factor=factor,
        watchdog_return_factor=cfg.safety.watchdog_return_factor,
    )
    core.minor_present = True
    r = ExpressionRenderer(
        cfg.expression.model_copy(update={"max_amplitude": 3}),
        cfg.expression_map,
        manifest.dofs,
        0.15,
        gestures,
        body_yaw="body_yaw",
        clips=CLIPS,
    )
    r.set_register(active("child"), factor)
    r.request("nod")
    r.update(state("calm", 1.0), active("child"), 1.0)
    up, t = 0.0, 1.0
    while t < 3.0:
        for cmd in r.commands(t):
            core.feed(cmd)
        up = max(up, core.tick(t, DT)["head.pitch"] - d.home)
        t += DT
    assert 0 < up <= factor * (d.max - d.home) + 1e-9


async def test_play_nod_through_the_skill_executor_reaches_the_renderer(cfg, manifest):
    from roboemot.skills import SkillContext, execute
    from roboemot.skills.defaults import SkillEnv, default_skills

    r = gesture_renderer(cfg, manifest)
    reg = default_skills(
        SkillEnv(
            gestures=r.playable,
            play=r.request,
            look_names=list,
            look_at=lambda t: True,
            affect=AffectEngine(cfg.affect, cfg.appraisal),
        )
    )
    enum = reg.tools(reg.offered(RoboEmotConfig().conversation.llm_skills, minor=True))[0]
    assert {"nod", "shake", "joy"} <= set(
        enum["function"]["parameters"]["properties"]["gesture"]["enum"]
    )
    assert "nod" in enum["function"]["description"]
    ctx = SkillContext(speaker="p1", language="en", minor=False, now=0.0)
    exec_kw = {"timeout_s": 1.0, "max_chars": 200, "enum_max": 100}
    out = await execute(reg, reg.names(), "play", {"gesture": "nod"}, ctx, **exec_kw)
    assert out.ok
    assert r.update(state("calm", 0.5), active(), 0.0) and r.last_label == "nod"
    bad = await execute(reg, reg.names(), "play", {"gesture": "wave"}, ctx, **exec_kw)
    assert not bad.ok


def through_core(cfg, manifest, name, dof, reflex=None):
    """Play gesture ``name`` through the arbiter and the limiter with the dev fallback caps
    (0.5 rad/s, 2 rad/s^2); return the positions of ``dof`` about home, 6 s at 50 Hz."""
    core = MotionCore(
        manifest.dofs,
        manifest.constraints,
        crossfade_s=cfg.motion.arbiter.crossfade_ms / 1000,
        watchdog_s=10.0,
        minor_factor=cfg.safety.minor_motion_factor,
        watchdog_return_factor=cfg.safety.watchdog_return_factor,
    )
    r = gesture_renderer(cfg, manifest)
    r.request(name)
    assert r.update(state("calm", 0.5), active("adult"), 0.0)
    home, out, t = manifest.dofs[dof].home, [], 0.0
    while t < 6.0:
        for cmd in r.commands(t):
            core.feed(cmd)
        if reflex is not None:
            core.feed(reflex(t))
        out.append(core.tick(t, DT)[dof] - home)
        t += DT
    return out, r.last_amplitude


def commanded_peak(cfg, name, dof, amplitude):
    return amplitude * math.radians(max(abs(v) for _, v in cfg.gesture_map[name].channels[dof]))


def test_a_nod_still_reads_as_a_nod_under_the_dev_fallback_caps(cfg, manifest):
    assert manifest.dofs["head.pitch"].max_vel == pytest.approx(0.5)
    assert manifest.dofs["head.pitch"].max_acc == pytest.approx(2.0)
    out, amp = through_core(cfg, manifest, "nod", "head.pitch")
    want = commanded_peak(cfg, "nod", "head.pitch", amp)
    assert max(out) >= 0.5 * want and max(out) > 0.05  # the dip is not smeared away
    assert out[-1] == pytest.approx(0.0, abs=1e-6)  # and the head comes home


def test_a_shake_swings_both_ways_under_the_dev_fallback_caps(cfg, manifest):
    out, amp = through_core(cfg, manifest, "shake", "head.yaw")
    want = commanded_peak(cfg, "shake", "head.yaw", amp)
    assert max(out) >= 0.5 * want and -min(out) >= 0.5 * want
    assert out[-1] == pytest.approx(0.0, abs=1e-6)


def test_a_gesture_is_refused_while_the_circadian_factor_is_zero(cfg, manifest):
    r = gesture_renderer(cfg, manifest)
    r.circadian_factor = lambda: 0.0
    assert r.request("nod") is False and r.request("joy") is False
    assert r.update(state("calm", 0.5), active(), 0.0) is False and r.started == 0
    r.circadian_factor = lambda: 1.0
    assert r.request("nod") is True


async def test_play_tells_the_model_when_nothing_was_played(cfg, manifest):
    from roboemot.skills import SkillContext, execute
    from roboemot.skills.defaults import SkillEnv, default_skills

    r = gesture_renderer(cfg, manifest)
    r.circadian_factor = lambda: 0.0
    reg = default_skills(
        SkillEnv(
            gestures=r.playable,
            play=r.request,
            look_names=list,
            look_at=lambda t: True,
            affect=AffectEngine(cfg.affect, cfg.appraisal),
        )
    )
    ctx = SkillContext(speaker="p1", language="en", minor=False, now=0.0)
    kw = {"timeout_s": 1.0, "max_chars": 200, "enum_max": 100}
    out = await execute(reg, reg.names(), "play", {"gesture": "nod"}, ctx, **kw)
    body = json.loads(out.content)
    assert body["error"] == "not_now" and "not played" in body["detail"]


def test_a_reload_resets_the_primitive_cache_and_drops_a_removed_pending_gesture(cfg, manifest):
    r = gesture_renderer(cfg, manifest)
    first = r._primitive("nod", True)
    assert first is not None and r._primitive("nod", True) is first
    wide = entry(**{"head.pitch": [(0, 0), (0.5, 20), (1, 0)]})
    r.apply_config(r.cfg, cfg.expression_map, 0.15, {"nod": wide})
    assert r._primitive("nod", True) is not first
    assert r.playable() == [*cfg.expression_map, "nod"]
    assert r.request("nod")
    r.apply_config(r.cfg, cfg.expression_map, 0.15, {})  # reload removes it while pending
    assert not r.request("nod") and "nod" not in r.playable()
    assert r.update(state("calm", 0.5), active(), 0.0) is False and r.started == 0
    assert r.request("shake") is False


def test_a_reflex_keeps_the_head_while_a_gesture_plays(cfg, manifest):
    from roboemot.motion import Command

    target = -0.2

    def reflex(t):
        return Command("reflex", Priority.REFLEX, {"head.pitch": target}, t, expires=t + 0.1)

    out, _ = through_core(cfg, manifest, "nod", "head.pitch", reflex)
    assert out[len(out) // 2] == pytest.approx(target, abs=1e-6)
    assert all(v == pytest.approx(target, abs=0.02) for v in out[len(out) // 3 :])


def test_the_connectivity_gestures_keep_the_old_literals_and_the_minjerk_easing(cfg, manifest):
    """puzzled and perk_up were GestureConfig literals (1.2 s, pose at 30% and left at 70%,
    minjerk) before they moved into the manifest; the move must not change the motion."""
    r = gesture_renderer(cfg, manifest)
    expected = {
        "puzzled": {"head.roll": 0.15, "antenna_right": -0.4, "antenna_left": 0.4},
        "perk_up": {"head.pitch": -0.08, "antenna_right": 0.4, "antenna_left": -0.4},
    }
    for name, pose in expected.items():
        prim = r.gesture(name)
        assert prim.method == "minjerk" and prim.additive
        assert [round(k.t, 2) for k in prim.keyframes] == [0.0, 0.36, 0.84, 1.2]
        for key in (prim.keyframes[1], prim.keyframes[2]):
            assert key.values == pytest.approx(pose, abs=1e-6)
        for key in (prim.keyframes[0], prim.keyframes[3]):
            assert set(key.values) == set(pose) and not any(key.values.values())


def test_a_gesture_for_a_robot_that_may_not_approach_keeps_the_body_still(cfg, manifest):
    gestures = {
        "turn": entry(
            **{"body_yaw": [(0, 0), (0.5, 20), (1, 0)], "head.yaw": [(0, 0), (0.5, 5), (1, 0)]}
        )
    }
    r = ExpressionRenderer(
        cfg.expression,
        cfg.expression_map,
        manifest.dofs,
        0.15,
        gestures,
        body_yaw="body_yaw",
        clips=CLIPS,
    )
    r.set_register(active("adult", approach_allowed=False), 1.0)
    assert "body_yaw" not in r.gesture("turn").keyframes[1].values
    r.set_register(active("adult", approach_allowed=True), 1.0)
    assert "body_yaw" in r.gesture("turn").keyframes[1].values
