"""The Reachy Mini clip library plugin and the shipped mapping end to end (SPEC 4A, 7, 7A)."""

from __future__ import annotations

import math

import pytest
from roboemot.affect.clips import ENTRY_POINT_GROUP, load_clip_library
from roboemot.config import ConfigStack
from roboemot.manifest import load_manifest
from roboemot_reachymini import app as app_module
from roboemot_reachymini import clips as clips_module
from test_app_perception import data, new_app, start, wait_for  # noqa: F401  (data is a fixture)

MANIFEST = load_manifest(
    app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
)


def variant_names(behaviour: dict) -> dict[str, list[str]]:
    """Clip names by entry: every expression label and the circadian yawn."""
    entries = dict(behaviour["expression_map"])
    entries["yawn"] = behaviour["circadian_motion"]["yawn"]
    return {name: [v["clip"] for v in e["variants"]] for name, e in entries.items()}


def test_the_plugin_is_registered_under_the_manifest_robot_id():
    from importlib.metadata import entry_points

    found = {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}
    assert found[MANIFEST.robot.id] == "roboemot_reachymini.clips:library"


def test_the_library_is_read_once_and_read_only():
    lib = clips_module.library()
    assert lib is clips_module.library()
    with pytest.raises(TypeError):
        lib["x"] = {}  # type: ignore[index]


def test_core_loads_and_validates_the_whole_library():
    library = load_clip_library(MANIFEST)
    assert set(library) == set(clips_module.library())
    assert library and all(e.channels and not e.variants for e in library.values())


def test_every_variant_of_the_shipped_mapping_is_a_clip_and_the_stack_accepts_it():
    library = load_clip_library(MANIFEST)
    named = variant_names(MANIFEST.model.behaviour)
    assert all(clips for clips in named.values())  # no hand-written label is left
    for entry, names in named.items():
        assert len(set(names)) == len(names), entry  # no clip twice in one entry
        assert set(names) <= set(library), entry
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        MANIFEST.model.behaviour, MANIFEST.channels, library, body_yaw=None
    )
    cfg = stack.effective()
    assert set(cfg.expression_map) == set(cfg.affect.labels)
    assert cfg.circadian_motion.yawn.variants and cfg.circadian_motion.stretch.channels


def test_a_clip_missing_from_the_library_is_refused():
    from roboemot.config import ConfigError

    library = load_clip_library(MANIFEST)
    del library["cheerful1"]
    with pytest.raises(ConfigError, match="unknown clip 'cheerful1'"):
        ConfigStack().set_manifest_behaviour(
            MANIFEST.model.behaviour, MANIFEST.channels, library, body_yaw=None
        )


def test_no_mapped_clip_overshoots_the_manifest_range_about_home():
    """Raw peak x gain of every mapped clip (labels and yawn) fits each channel's range."""
    from roboemot.config.schema import ExpressionEntry

    library = load_clip_library(MANIFEST)
    behaviour = MANIFEST.model.behaviour
    entries = dict(behaviour["expression_map"])
    entries["yawn"] = behaviour["circadian_motion"]["yawn"]
    over = []
    for name, raw in entries.items():
        for v in ExpressionEntry.model_validate(raw).variants:
            for channel, frames in library[v.clip].channels.items():
                dof = MANIFEST.dofs[channel]
                lo, hi = math.degrees(dof.min - dof.home), math.degrees(dof.max - dof.home)
                peak = [f[1] * v.gain for f in frames]
                if max(peak) > hi or min(peak) < lo:
                    over.append((name, v.clip, channel, min(peak), max(peak)))
    assert over == []


def test_every_label_plays_each_of_its_clips_through_the_renderer():
    """Public path: update starts the entry, commands feeds it; one variant at a time."""
    from roboemot.affect import AffectState
    from roboemot.affect.expression import ExpressionRenderer
    from roboemot.affect.register import Person, resolve_register
    from roboemot.config.schema import ExpressionEntry

    library = load_clip_library(MANIFEST)
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        MANIFEST.model.behaviour, MANIFEST.channels, library, body_yaw=None
    )
    cfg = stack.effective()
    register = resolve_register([Person({})], cfg.register, cfg.persona)
    for label, entry in cfg.expression_map.items():
        for variant in entry.variants:
            only = {
                label: ExpressionEntry(
                    variants=[variant.model_copy(update={"min_intensity": 0, "max_intensity": 1})]
                )
            }
            r = ExpressionRenderer(
                cfg.expression, only, MANIFEST.dofs, 0.15, body_yaw="body_yaw", clips=library
            )
            r.update(AffectState(0.0, 0.0, "calm", 0.0, 0), register, 0.0)
            assert r.update(AffectState(0.0, 0.0, label, 0.6, 1), register, 5.0), variant.clip
            assert any(c.targets for c in r.commands(5.5)), (label, variant.clip)


def test_the_app_starts_with_the_library_and_offers_every_label(data):  # noqa: F811
    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.expression is not None and app.diagnostics is not None)
        assert set(MANIFEST.model.behaviour["expression_map"]) <= set(app.expression.playable())
        assert app.circadian_runner is not None
    finally:
        stop.set()
        thread.join(15)
    assert not errors


WORST = {
    "expression": {
        "intensity_factor": {"low": 2, "medium": 2, "high": 2},
        "max_amplitude": 3,
    }
}


@pytest.mark.parametrize("clip", sorted(load_clip_library(MANIFEST)))
def test_every_vendored_clip_stays_inside_the_range_at_the_schema_worst_case(clip):
    """SPEC 7, 12: at the register's top amplitude and the schema's upper bounds for the factors
    and ``max_amplitude``, no rendered target leaves a dof's ``min..max`` about home."""
    from roboemot.affect import AffectState
    from roboemot.affect.expression import ExpressionRenderer
    from roboemot.affect.register import Person, resolve_register
    from roboemot.config.schema import ExpressionEntry, RoboEmotConfig

    library = load_clip_library(MANIFEST)
    worst = RoboEmotConfig.model_validate(WORST)
    register = resolve_register(
        [Person({"motion_amplitude": 2, "reaction_intensity": "high"})],
        worst.register,
        worst.persona,
    )
    assert register.params.motion_amplitude == 2
    only = {"joy": ExpressionEntry(variants=[{"clip": clip}])}
    r = ExpressionRenderer(
        worst.expression, only, MANIFEST.dofs, 0.15, body_yaw="body_yaw", clips=library
    )
    r.update(AffectState(0.0, 0.0, "calm", 0.0, 0), register, 0.0)
    assert r.update(AffectState(0.0, 0.0, "joy", 1.0, 1), register, 1.0)
    moved = False
    for k in range(int(32 / 0.05)):  # a clip is at most 30 s
        for cmd in r.commands(1.0 + k * 0.05):
            for name, value in cmd.targets.items():
                dof = MANIFEST.dofs[name]
                assert dof.min - dof.home - 1e-9 <= value <= dof.max - dof.home + 1e-9, (
                    clip,
                    name,
                    value,
                )
                moved = moved or abs(value) > 1e-6
    assert moved
