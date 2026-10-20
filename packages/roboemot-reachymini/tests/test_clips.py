"""The Reachy Mini clip library plugin and the shipped mapping end to end (SPEC 4A, 7, 7A)."""

from __future__ import annotations

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
    stack.set_manifest_behaviour(MANIFEST.model.behaviour, MANIFEST.channels, library)
    cfg = stack.effective()
    assert set(cfg.expression_map) == set(cfg.affect.labels)
    assert cfg.circadian_motion.yawn.variants and cfg.circadian_motion.stretch.channels


def test_a_clip_missing_from_the_library_is_refused():
    from roboemot.config import ConfigError

    library = load_clip_library(MANIFEST)
    del library["cheerful1"]
    with pytest.raises(ConfigError, match="unknown clip 'cheerful1'"):
        ConfigStack().set_manifest_behaviour(MANIFEST.model.behaviour, MANIFEST.channels, library)


def test_the_app_finds_the_library_and_plays_a_clip_for_every_label(data):  # noqa: F811
    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.expression is not None and app.diagnostics is not None)
        renderer = app.expression
        named = variant_names(MANIFEST.model.behaviour)
        assert set(renderer.clips) == set(clips_module.library())
        for label in app.expression.expression_map:
            for clip in named[label]:
                assert renderer._primitive(label, True, clip) is not None, (label, clip)
        assert app.circadian_runner is not None
    finally:
        stop.set()
        thread.join(15)
    assert not errors
