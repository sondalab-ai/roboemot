"""SPEC 4A, 7, 7A: recorded clips as variants of an expression, the clip library and its loader."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import pytest
from pydantic import ValidationError
from roboemot.affect import AffectState
from roboemot.affect import clips as clips_module
from roboemot.affect.clips import ClipLibraryError, load_clip_library, validate_clip_library
from roboemot.affect.expression import ExpressionRenderer, _sample, pick_variant
from roboemot.affect.register import Person, resolve_register
from roboemot.circadian.motion import CircadianMotion
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import (
    EXPRESSION_MAX_CYCLE_S,
    ClipVariant,
    ExpressionEntry,
    RoboEmotConfig,
)
from roboemot.manifest import load_manifest
from roboemot.motion.curves import CURVES
from roboemot.motion.primitives import Keyframe, Primitive
from roboemot.plugins import GROUPS, contract_violation

MANIFESTS = Path(__file__).resolve().parents[3] / "configs" / "manifests"
SWAY = [(0, 0), (0.5, 10), (1, 0)]
DEFAULTS = RoboEmotConfig()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFESTS / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)


def clip(**channels) -> ExpressionEntry:
    return ExpressionEntry(channels=channels)


def library() -> dict[str, ExpressionEntry]:
    return {
        "soft": clip(**{"head.pitch": [(0, 0), (0.5, 4), (1, 0)]}),
        "wide": clip(**{"head.pitch": [(0, 0), (0.5, 8), (1, 0)]}),
        "turn": clip(**{"body_yaw": [(0, 0), (0.5, 20), (1, 0)], "head.pitch": SWAY}),
        "big": clip(**{"head.yaw": [(0, 0), (0.5, 500), (1, 0)]}),
    }


def variants(*items) -> ExpressionEntry:
    return ExpressionEntry(variants=[ClipVariant(**i) for i in items])


def state(label="joy", intensity=0.5, impulse=0) -> AffectState:
    return AffectState(0.0, 0.0, label, intensity, impulse)


def adult():
    return resolve_register([Person("adult", "declared", {})], DEFAULTS.register, DEFAULTS.persona)


def child():
    return resolve_register([Person("child", "declared", {})], DEFAULTS.register, DEFAULTS.persona)


def renderer(manifest, entry, *, seed=1, lib=None, **expr) -> ExpressionRenderer:
    cfg = DEFAULTS.expression.model_copy(update=expr)
    return ExpressionRenderer(
        cfg,
        {"joy": entry},
        manifest.dofs,
        0.15,
        body_yaw="body_yaw",
        clips=library() if lib is None else lib,
        rng=random.Random(seed),
    )


def play(r, intensity, now, reg=None) -> str:
    """Start the joy entry at ``now`` and return the clip it chose (via the last amplitude)."""
    reg = reg or adult()
    r.update(state("calm", intensity), reg, now - 0.5)
    assert r.update(state("joy", intensity, impulse=int(now)), reg, now)
    return r.last_label


def chosen_clips(r, n, intensity=0.5) -> list[str]:
    out = []
    for i in range(n):
        r.update(state("calm", intensity), adult(), i * 100.0)
        r.update(state("joy", intensity, impulse=i + 1), adult(), i * 100.0 + 50)
        out.append(next(iter(r._last_clip.values())))
    return out


# ----------------------------------------------------------------------------- schema


def test_an_entry_is_channels_or_variants_never_both_or_neither():
    assert ExpressionEntry(channels={"a": SWAY}).variants == []
    assert ExpressionEntry(variants=[ClipVariant(clip="soft")]).channels == {}
    with pytest.raises(ValidationError, match="exactly one"):
        ExpressionEntry(channels={"a": SWAY}, variants=[ClipVariant(clip="soft")])
    with pytest.raises(ValidationError, match="exactly one"):
        ExpressionEntry()


def test_variant_defaults_and_ranges():
    v = ClipVariant(clip="soft")
    assert (v.weight, v.min_intensity, v.max_intensity, v.gain) == (1, 0, 1, 1.0)
    for bad in ({"gain": 2.1}, {"gain": -0.1}, {"weight": 0}, {"min_intensity": 1.1}):
        with pytest.raises(ValidationError):
            ClipVariant(clip="soft", **bad)
    with pytest.raises(ValidationError, match="min_intensity must not exceed"):
        ClipVariant(clip="soft", min_intensity=0.6, max_intensity=0.4)


def test_bands_must_cover_zero_to_one():
    low = {"clip": "soft", "max_intensity": 0.5}
    high = {"clip": "wide", "min_intensity": 0.5}
    assert ExpressionEntry(variants=[ClipVariant(**low), ClipVariant(**high)])
    overlap = {"clip": "wide", "min_intensity": 0.3}
    assert ExpressionEntry(variants=[ClipVariant(**low), ClipVariant(**overlap)])
    gap = {"clip": "wide", "min_intensity": 0.6}
    with pytest.raises(ValidationError, match="cover"):
        ExpressionEntry(variants=[ClipVariant(**low), ClipVariant(**gap)])
    with pytest.raises(ValidationError, match="cover"):
        ExpressionEntry(variants=[ClipVariant(**high)])
    with pytest.raises(ValidationError, match="cover"):
        ExpressionEntry(variants=[ClipVariant(clip="soft", max_intensity=0.9)])


def test_gesture_map_is_channels_only_and_circadian_may_use_variants():
    entry = {"variants": [{"clip": "soft"}]}
    with pytest.raises(ValidationError, match="channels only"):
        RoboEmotConfig.model_validate({"gesture_map": {"nod": entry}})
    cfg = RoboEmotConfig.model_validate(
        {"expression_map": {"joy": entry}, "circadian_motion": {"yawn": entry, "stretch": entry}}
    )
    assert cfg.circadian_motion.yawn.variants[0].clip == "soft"


# ----------------------------------------------------------------------------- merge and names


def stack_with(manifest, behaviour, clips=("soft", "wide")) -> ConfigStack:
    stack = ConfigStack()
    stack.set_manifest_behaviour(behaviour, manifest.channels, clips)
    return stack


CHANNELS_JOY = {"expression_map": {"joy": {"channels": {"head.pitch": SWAY}}}}
VARIANTS_JOY = {"expression_map": {"joy": {"variants": [{"clip": "soft"}]}}}


def test_a_layer_with_variants_drops_the_channels_below_and_the_reverse(manifest):
    stack = stack_with(manifest, CHANNELS_JOY)
    stack.app = VARIANTS_JOY
    joy = stack.effective().expression_map["joy"]
    assert joy.channels == {} and joy.variants[0].clip == "soft"
    back = stack_with(manifest, VARIANTS_JOY)
    back.app = CHANNELS_JOY
    joy = back.effective().expression_map["joy"]
    assert joy.variants == [] and set(joy.channels) == {"head.pitch"}


def test_channels_still_merge_channel_by_channel_within_the_same_form(manifest):
    stack = stack_with(manifest, CHANNELS_JOY)
    stack.app = {"expression_map": {"joy": {"channels": {"head.roll": SWAY}}}}
    assert set(stack.effective().expression_map["joy"].channels) == {"head.pitch", "head.roll"}


def test_the_circadian_entries_switch_form_too(manifest):
    base = {"circadian_motion": {"yawn": {"channels": {"head.pitch": SWAY}}}}
    stack = stack_with(manifest, base)
    stack.app = {"circadian_motion": {"yawn": {"variants": [{"clip": "wide"}]}}}
    yawn = stack.effective().circadian_motion.yawn
    assert yawn.channels == {} and yawn.variants[0].clip == "wide"


def test_a_runtime_override_switches_the_form(manifest):
    stack = stack_with(manifest, CHANNELS_JOY)
    stack.set("expression_map.joy", {"variants": [{"clip": "wide"}]})
    joy = stack.effective().expression_map["joy"]
    assert joy.channels == {} and joy.variants[0].clip == "wide"


def test_an_unknown_clip_is_refused_in_any_layer(manifest):
    with pytest.raises(ConfigError, match="unknown clip 'ghost'"):
        stack_with(manifest, {"expression_map": {"joy": {"variants": [{"clip": "ghost"}]}}})
    stack = stack_with(manifest, VARIANTS_JOY)
    stack.app = {"expression_map": {"joy": {"variants": [{"clip": "ghost"}]}}}
    with pytest.raises(ConfigError, match="unknown clip 'ghost'"):
        stack.effective()
    yawn = {"circadian_motion": {"yawn": {"variants": [{"clip": "ghost"}]}}}
    with pytest.raises(ConfigError, match="unknown clip 'ghost'"):
        stack_with(manifest, yawn)


def test_a_robot_without_a_library_refuses_any_variant(manifest):
    with pytest.raises(ConfigError, match="unknown clip 'soft'"):
        stack_with(manifest, VARIANTS_JOY, clips=())
    assert stack_with(manifest, CHANNELS_JOY, clips=()).effective().expression_map["joy"].channels


def test_a_refused_manifest_leaves_the_previous_clips_in_place(manifest):
    stack = stack_with(manifest, VARIANTS_JOY)
    with pytest.raises(ConfigError):
        stack.set_manifest_behaviour(VARIANTS_JOY, manifest.channels, ())
    assert stack.clips == {"soft", "wide"}


# ----------------------------------------------------------------------------- library loader


def test_a_valid_library_passes_and_dicts_are_converted(manifest):
    lib = validate_clip_library({"a": {"channels": {"head.pitch": SWAY}}}, manifest.channels)
    assert isinstance(lib["a"], ExpressionEntry)
    assert validate_clip_library(library(), manifest.channels).keys() == library().keys()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"head.pitch": [(0, 0), (1, 5)]}, "first and last offsets"),
        ({"head.pitch": [(0, 1), (1, 0)]}, "first and last offsets"),
        ({"head.pitch": [(0, 0), (EXPRESSION_MAX_CYCLE_S + 1, 0)]}, "longer than"),
        ({"head.pitch": SWAY, "head.roll": [(0, 0), (2, 0)]}, "same time"),
        ({"head.pitch": [(0, 0)]}, "two keyframes"),
        ({"nope": SWAY}, "unknown channel 'nope'"),
        ({"head.x": SWAY}, "unknown channel 'head.x'"),
    ],
)
def test_a_clip_that_breaks_the_entry_rules_refuses_the_library(manifest, raw, message):
    with pytest.raises(ClipLibraryError, match=f"(?s)clip 'bad'.*{message}"):
        validate_clip_library({"bad": {"channels": raw}}, manifest.channels)


def test_a_translation_dof_is_not_a_clip_channel(manifest):
    translations = [n for n, d in manifest.dofs.items() if d.kind == "translation"]
    if translations:
        with pytest.raises(ClipLibraryError, match="unknown channel"):
            validate_clip_library({"bad": {"channels": {translations[0]: SWAY}}}, manifest.channels)


def test_a_clip_cannot_hold_variants_or_nothing(manifest):
    with pytest.raises(ClipLibraryError, match="clip 'v'"):
        validate_clip_library({"v": {"variants": [{"clip": "a"}]}}, manifest.channels)
    with pytest.raises(ClipLibraryError, match="clip 'e'"):
        validate_clip_library({"e": {}}, manifest.channels)


class FakeEntryPoint:
    def __init__(self, name, factory):
        self.name, self.group, self._factory = name, clips_module.ENTRY_POINT_GROUP, factory

    def load(self):
        return self._factory


def patch_entry_points(monkeypatch, *eps):
    monkeypatch.setattr(clips_module, "entry_points", lambda group: list(eps))


def test_the_loader_resolves_the_plugin_by_robot_id_and_validates(manifest, monkeypatch):
    seen = []

    def with_manifest(m):
        seen.append(m)
        return {"a": {"channels": {"head.pitch": SWAY}}}

    patch_entry_points(
        monkeypatch,
        FakeEntryPoint("other-robot", lambda: {"zzz": {}}),
        FakeEntryPoint(manifest.robot.id, with_manifest),
    )
    assert set(load_clip_library(manifest)) == {"a"}
    assert seen == [manifest]
    patch_entry_points(monkeypatch, FakeEntryPoint(manifest.robot.id, lambda: library()))
    assert set(load_clip_library(manifest)) == set(library())


def test_the_loader_refuses_a_broken_plugin_and_a_bad_clip(manifest, monkeypatch):
    def boom():
        raise RuntimeError("no file")

    patch_entry_points(monkeypatch, FakeEntryPoint(manifest.robot.id, boom))
    with pytest.raises(ClipLibraryError, match="RuntimeError"):
        load_clip_library(manifest)
    bad = {"c": {"channels": {"head.pitch": [(0, 0), (1, 3)]}}}
    patch_entry_points(monkeypatch, FakeEntryPoint(manifest.robot.id, lambda: bad))
    with pytest.raises(ClipLibraryError, match="clip 'c'"):
        load_clip_library(manifest)


def test_no_registered_library_is_an_empty_one(manifest, monkeypatch):
    patch_entry_points(monkeypatch, FakeEntryPoint("other-robot", lambda: library()))
    assert load_clip_library(manifest) == {}


def test_the_group_is_registered_and_accepts_both_factory_shapes():
    assert clips_module.ENTRY_POINT_GROUP in GROUPS
    group = clips_module.ENTRY_POINT_GROUP
    assert contract_violation(group, lambda: {}) is None
    assert contract_violation(group, lambda manifest: {}) is None
    assert contract_violation(group, lambda a, b: {}) is not None
    assert contract_violation(group, {}) is not None


# ----------------------------------------------------------------------------- selection


def test_the_pick_follows_the_intensity_band():
    low = ClipVariant(clip="soft", max_intensity=0.4)
    high = ClipVariant(clip="wide", min_intensity=0.4)
    rng = random.Random(0)
    assert {pick_variant([low, high], 0.1, None, rng).clip for _ in range(20)} == {"soft"}
    assert {pick_variant([low, high], 0.9, None, rng).clip for _ in range(20)} == {"wide"}
    assert pick_variant([low], 0.9, None, rng) is None


def test_the_pick_is_weighted():
    heavy = ClipVariant(clip="soft", weight=9)
    light = ClipVariant(clip="wide", weight=1)
    rng = random.Random(7)
    picks = [pick_variant([heavy, light], 0.5, None, rng).clip for _ in range(2000)]
    assert 0.85 < picks.count("soft") / len(picks) < 0.95


def test_the_same_clip_never_plays_twice_in_a_row_unless_it_is_the_only_one(manifest):
    two = variants({"clip": "soft"}, {"clip": "wide"})
    seq = chosen_clips(renderer(manifest, two, min_interval_s=0), 40)
    assert all(a != b for a, b in zip(seq, seq[1:], strict=False)) and set(seq) == {"soft", "wide"}
    only = variants({"clip": "soft"}, {"clip": "wide", "min_intensity": 0.9})
    seq = chosen_clips(renderer(manifest, only, min_interval_s=0), 5, intensity=0.2)
    assert seq == ["soft"] * 5


def test_each_entry_remembers_its_own_last_clip(manifest):
    entry = variants({"clip": "soft"}, {"clip": "wide"})
    r = ExpressionRenderer(
        DEFAULTS.expression.model_copy(update={"min_interval_s": 0}),
        {"joy": entry, "sadness": entry},
        manifest.dofs,
        0.15,
        body_yaw="body_yaw",
        clips=library(),
        rng=random.Random(2),
    )
    r.update(state("calm"), adult(), 0.0)
    r.update(state("joy", impulse=1), adult(), 10.0)
    r.update(state("sadness", impulse=2), adult(), 20.0)
    assert set(r._last_clip) == {"joy", "sadness"}


def test_selection_is_deterministic_with_a_seeded_source(manifest):
    entry = variants({"clip": "soft"}, {"clip": "wide"}, {"clip": "turn"})
    a = chosen_clips(renderer(manifest, entry, seed=5, min_interval_s=0), 12)
    b = chosen_clips(renderer(manifest, entry, seed=5, min_interval_s=0), 12)
    assert a == b


def test_the_primitive_cache_keys_on_the_chosen_clip(manifest):
    r = renderer(manifest, variants({"clip": "soft"}, {"clip": "wide"}))
    soft, wide = r._primitive("joy", True, "soft"), r._primitive("joy", True, "wide")
    assert soft is not wide and soft is r._primitive("joy", True, "soft")
    assert soft.keyframes[1].values["head.pitch"] == pytest.approx(math.radians(4))
    assert wide.keyframes[1].values["head.pitch"] == pytest.approx(math.radians(8))


# ----------------------------------------------------------------------------- amplitude and safety


def peak(r, name="head.pitch", t0=1.0) -> float:
    return max(abs(c.targets.get(name, 0.0)) for k in range(50) for c in r.commands(t0 + k * 0.02))


def test_gain_multiplies_the_amplitude_before_the_cap(manifest):
    plain = renderer(manifest, variants({"clip": "soft"}))
    gained = renderer(manifest, variants({"clip": "soft", "gain": 1.5}))
    play(plain, 0.5, 1.0)
    play(gained, 0.5, 1.0)
    assert gained.last_amplitude == pytest.approx(plain.last_amplitude * 1.5)
    ceiling = plain.last_amplitude * 1.2
    capped = renderer(manifest, variants({"clip": "soft", "gain": 2.0}), max_amplitude=ceiling)
    play(capped, 0.5, 1.0)
    assert capped.last_amplitude == pytest.approx(ceiling)
    assert capped.cfg.max_amplitude == pytest.approx(ceiling)


def test_a_clip_plays_through_the_same_pipeline_as_an_entry(manifest):
    by_clip = renderer(manifest, variants({"clip": "soft"}))
    by_hand = renderer(manifest, clip(**{"head.pitch": [(0, 0), (0.5, 4), (1, 0)]}))
    play(by_clip, 0.5, 1.0)
    play(by_hand, 0.5, 1.0)
    assert by_clip.last_amplitude == by_hand.last_amplitude
    for k in range(60):
        t = 1.0 + k * 0.02
        assert [c.targets for c in by_clip.commands(t)] == [c.targets for c in by_hand.commands(t)]


def test_the_body_yaw_gate_applies_to_a_clip(manifest):
    entry = variants({"clip": "turn"})
    free = renderer(manifest, entry)
    play(free, 0.5, 1.0)
    assert peak(free, "body_yaw") > 0
    shut = renderer(manifest, entry)
    play(shut, 0.5, 1.0, reg=child())
    assert not child().params.approach_allowed
    assert peak(shut, "body_yaw") == 0 and peak(shut, "head.pitch") > 0


def test_the_register_tightening_applies_to_a_playing_clip(manifest):
    r = renderer(manifest, variants({"clip": "turn"}))
    play(r, 0.5, 1.0)
    r.set_register(child())
    assert all("body_yaw" not in c.targets for k in range(50) for c in r.commands(1.1 + k * 0.01))


def test_a_clip_is_clamped_to_the_range_and_to_the_minor_share(manifest):
    r = renderer(manifest, variants({"clip": "big"}), max_amplitude=3)
    play(r, 1.0, 1.0)
    dof = manifest.dofs["head.yaw"]
    top = peak(r, "head.yaw")
    assert top == pytest.approx(dof.max - dof.home)
    r = renderer(manifest, variants({"clip": "big"}), max_amplitude=3)
    r.set_register(adult(), 0.5)
    play(r, 1.0, 1.0)
    assert peak(r, "head.yaw") == pytest.approx((dof.max - dof.home) * 0.5)


def test_a_variant_naming_a_missing_clip_or_no_band_starts_nothing(manifest):
    ghost = renderer(manifest, variants({"clip": "ghost"}))
    ghost.update(state("calm"), adult(), 4.0)
    assert not ghost.update(state("joy", 0.5, impulse=1), adult(), 5.0)
    r = renderer(manifest, variants({"clip": "soft", "max_intensity": 0.5}, {"clip": "wide"}))
    assert play(r, 0.5, 5.0) == "joy"


# ----------------------------------------------------------------------------- circadian


def motion(manifest, entry_yawn=None, entry_stretch=None, seed=0, **kw) -> CircadianMotion:
    cm = DEFAULTS.circadian_motion.model_copy(update={"yawn": entry_yawn, "stretch": entry_stretch})
    return CircadianMotion(
        DEFAULTS.circadian,
        cm,
        manifest.dofs,
        body_yaw="body_yaw",
        rng=random.Random(seed),
        clips=library(),
        **kw,
        affect_intensity=lambda: 0.0,
    )


def test_a_stretch_of_variants_picks_a_clip_and_never_repeats_it(manifest):
    m = motion(manifest, entry_stretch=variants({"clip": "soft"}, {"clip": "wide"}))
    m.set_register(1.0, 2.0)
    seen = []
    for i in range(20):
        assert m.start_waking(i * 10.0)
        seen.append(m._last_clip["stretch"])
        m.reset()
    assert all(a != b for a, b in zip(seen, seen[1:], strict=False)) and len(set(seen)) == 2


def test_a_yawn_of_variants_plays_with_gain_and_cap(manifest):
    m = motion(manifest, entry_yawn=variants({"clip": "soft", "gain": 2.0}))
    m.set_register(1.0, 1.5)
    t, played = 0.0, False
    while t < 400 and not played:
        played = bool(m.commands(t, "drowsy"))
        t += 0.5
    assert played and m._amp == pytest.approx(1.5)
    assert m.playing == "yawn"


def test_the_circadian_clip_follows_the_affect_intensity(manifest):
    entry = variants({"clip": "soft", "max_intensity": 0.5}, {"clip": "wide", "min_intensity": 0.5})
    m = motion(manifest, entry_stretch=entry)
    m.set_register(1.0, 2.0)
    m.affect_intensity = lambda: 0.9
    assert m.start_waking(0.0) and m._last_clip["stretch"] == "wide"


def test_a_stretch_or_yawn_without_any_clip_is_not_played(manifest):
    cm = DEFAULTS.circadian_motion.model_copy(
        update={"stretch": variants({"clip": "ghost"}), "yawn": variants({"clip": "ghost"})}
    )
    m = CircadianMotion(
        DEFAULTS.circadian, cm, manifest.dofs, body_yaw="body_yaw", affect_intensity=lambda: 0.0
    )
    assert not m.start_waking(0.0)
    assert all(not m.commands(t, "drowsy") for t in range(0, 600, 5))


# ----------------------------------------------------------------------------- bisect sampling


def linear_sample(frames, t):
    for (t0, v0), (t1, v1) in zip(frames, frames[1:], strict=False):
        if t <= t1:
            return v0 + (v1 - v0) * ((t - t0) / (t1 - t0))
    return frames[-1][1]


def random_frames(rng, n):
    times = sorted(rng.sample(range(1, 100000), n - 1))
    return [(0.0, 0.0)] + [(t / 1000, rng.uniform(-30, 30)) for t in times]


def test_bisect_sampling_equals_the_linear_scan_on_random_keyframes():
    rng = random.Random(11)
    for n in (2, 3, 10, 400):
        frames = random_frames(rng, n)
        end = frames[-1][0]
        probes = [t for t, _ in frames] + [rng.uniform(-1, end + 1) for _ in range(300)]
        for t in probes:
            assert _sample(frames, t) == linear_sample(frames, t)


def linear_primitive_sample(prim: Primitive, t: float):
    kfs = prim.keyframes
    if t <= kfs[0].t:
        return dict(kfs[0].values)
    for a, b in zip(kfs, kfs[1:], strict=False):
        if t <= b.t:
            s = CURVES[prim.method]((t - a.t) / (b.t - a.t))
            return {k: a.values[k] + (b.values[k] - a.values[k]) * s for k in a.values}
    return dict(kfs[-1].values)


@pytest.mark.parametrize("method", ["linear", "minjerk"])
def test_primitive_sampling_equals_the_linear_scan(method):
    rng = random.Random(3)
    times = [0.0, *sorted(rng.uniform(0.01, 20) for _ in range(300))]
    prim = Primitive(
        "p", tuple(Keyframe(t, {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1)}) for t in times),
        method=method,
    )  # fmt: skip
    probes = [*times, *(rng.uniform(-1, 21) for _ in range(500))]
    for t in probes:
        assert prim.sample(t) == linear_primitive_sample(prim, t)


# ----------------------------------------------------------------------------- fix round


@pytest.mark.parametrize("field", ["weight", "gain"])
@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, 0, -1])
def test_weight_and_gain_must_be_finite_and_positive(field, bad):
    with pytest.raises(ValidationError):
        ClipVariant(clip="soft", **{field: bad})


def test_duplicate_clips_and_ignored_keys_are_refused():
    with pytest.raises(ValidationError, match="more than once"):
        variants({"clip": "soft", "max_intensity": 0.5}, {"clip": "soft", "min_intensity": 0.5})
    one = [ClipVariant(clip="soft")]
    with pytest.raises(ValidationError, match="repeat and curve"):
        ExpressionEntry(variants=one, repeat=2)
    with pytest.raises(ValidationError, match="repeat and curve"):
        ExpressionEntry(variants=one, curve="minjerk")


def test_pick_clamps_the_intensity_and_band_edges_are_inclusive():
    low = ClipVariant(clip="soft", max_intensity=0.4)
    high = ClipVariant(clip="wide", min_intensity=0.4)
    rng = random.Random(0)
    for intensity in (1.0, 1.0 + 1e-12, 1.5):
        assert pick_variant([low, high], intensity, None, rng).clip == "wide"
    assert pick_variant([low, high], -1e-12, None, rng).clip == "soft"
    assert pick_variant([low], 0.4, None, rng).clip == "soft"
    assert pick_variant([high], 0.4, None, rng).clip == "wide"


def test_the_loader_arity_matches_the_group_contract(manifest, monkeypatch):
    class Callable0:
        def __call__(self):
            return {}

    patch_entry_points(monkeypatch, FakeEntryPoint(manifest.robot.id, Callable0()))
    assert load_clip_library(manifest) == {}
    patch_entry_points(monkeypatch, FakeEntryPoint(manifest.robot.id, lambda m=None: {}))
    assert load_clip_library(manifest) == {}


def test_a_reload_with_an_unknown_clip_keeps_the_last_valid_config(manifest, tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text("expression_map:\n  joy:\n    variants:\n      - clip: wide\n")
    stack = ConfigStack.from_files(app=app)
    stack.set_manifest_behaviour(VARIANTS_JOY, manifest.channels, ("soft", "wide"))
    assert stack.effective().expression_map["joy"].variants[0].clip == "wide"
    app.write_text("expression_map:\n  joy:\n    variants:\n      - clip: ghost\n")
    os.utime(app, (1e9, 2e9))
    assert stack.poll() is None
    assert "unknown clip 'ghost'" in stack.last_error
    assert stack.effective().expression_map["joy"].variants[0].clip == "wide"


def test_runtime_leaf_writes_switch_the_form(manifest):
    stack = stack_with(manifest, CHANNELS_JOY)
    stack.set("expression_map.joy.variants", [{"clip": "soft"}])
    joy = stack.effective().expression_map["joy"]
    assert joy.channels == {} and joy.variants[0].clip == "soft"
    stack = stack_with(manifest, VARIANTS_JOY)
    stack.set("expression_map.joy.channels", {"head.pitch": SWAY})
    joy = stack.effective().expression_map["joy"]
    assert joy.variants == [] and set(joy.channels) == {"head.pitch"}


def test_apply_config_drops_the_cached_primitives(manifest):
    entry = variants({"clip": "soft"})
    r = renderer(manifest, entry)
    play(r, 0.5, 1.0)
    assert r._cache
    r.apply_config(r.cfg, {"joy": entry}, 0.15)
    assert not r._cache


def test_request_on_a_variants_entry_picks_a_variant(manifest):
    r = renderer(manifest, variants({"clip": "wide"}))
    r.update(state("calm", 0.7), adult(), 0.0)
    assert r.request("joy")
    assert r.update(state("calm", 0.7), adult(), 5.0)
    assert r.last_label == "joy" and r._last_clip == {"joy": "wide"}
    assert max(abs(c.targets["head.pitch"]) for k in range(25) for c in r.commands(5 + k * 0.02))


def test_circadian_apply_config_resets_the_cached_primitives(manifest):
    entry = variants({"clip": "soft"})
    m = motion(manifest, entry_stretch=entry)
    m.set_register(1.0, 2.0)
    assert m.start_waking(0.0) and m._prims
    cm = DEFAULTS.circadian_motion.model_copy(update={"stretch": entry})
    m.apply_config(DEFAULTS.circadian, cm)
    assert not m._prims


def test_a_circadian_clip_never_emits_body_yaw(manifest):
    m = motion(manifest, entry_stretch=variants({"clip": "turn"}))
    m.set_register(1.0, 2.0)
    assert m.start_waking(0.0)
    cmds = [c for k in range(60) for c in m.commands(k * 0.02, "awake")]
    assert cmds and all("body_yaw" not in c.targets for c in cmds)
    assert any(c.targets.get("head.pitch") for c in cmds)


def test_a_boosted_circadian_clip_tightens_mid_play(manifest):
    def run(change):
        m = motion(manifest, entry_stretch=variants({"clip": "soft", "gain": 2.0}))
        m.set_register(1.0, 3.0)
        assert m.start_waking(0.0)
        m.commands(0.0, "awake")
        change(m)
        return max(abs(c.targets["head.pitch"]) for c in m.commands(0.5, "awake"))

    free = run(lambda m: None)
    assert run(lambda m: m.set_register(0.5, 3.0)) == pytest.approx(free * 0.5)
    assert run(lambda m: m.set_register(1.0, 1.0)) == pytest.approx(free * 0.5)
