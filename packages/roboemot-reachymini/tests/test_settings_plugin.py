"""Reachy Mini settings plugin: entry point, claimed paths, degree conversion (SPEC 4A, 11A, 12)."""

from __future__ import annotations

import math
from importlib.metadata import entry_points

import pytest
from roboemot.config.catalogue import DESCRIPTOR_KEYS
from roboemot.config.stack import ConfigStack
from roboemot.manifest import load_manifest
from roboemot.webui.robot_section import claimed
from roboemot.webui.settings import SettingsEditor
from roboemot_reachymini import app as app_module
from roboemot_reachymini.settings import CLAIMS, ReachySettings, plugin

MANIFEST = load_manifest(
    app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
)
BASE = "robot.reachy-mini-wireless"


@pytest.fixture
def stack():
    stack = ConfigStack(app={"webui": {"bind": "127.0.0.1", "port": 0}})
    stack.set_manifest_behaviour(MANIFEST.model.behaviour)
    return stack


@pytest.fixture
def split(stack, tmp_path):
    editor = SettingsEditor(stack, store=tmp_path / "s.json", manifest=MANIFEST)
    return editor, *editor.robot_split(editor.fields())


def test_entry_point_is_registered_under_the_robot_id():
    (found,) = [e for e in entry_points(group="roboemot.settings") if e.name == MANIFEST.robot.id]
    assert found.load() is plugin


def test_editor_loads_the_plugin_through_the_entry_point(split):
    _, core, robot = split
    assert robot["id"] == "reachy-mini-wireless" and robot["title"] == "Reachy Mini"


def test_hardware_rows_are_read_only_manifest_rows_with_every_key(split):
    _, _, robot = split
    own = [f for f in robot["fields"] if f["path"].startswith(BASE + ".")]
    assert own
    for row in own:
        assert set(row) == DESCRIPTOR_KEYS
        assert row["writable"] is False and row["source"] == "manifest"


def test_angles_are_shown_in_degrees_translations_in_mm(split):
    _, _, robot = split
    by = {f["path"]: f for f in robot["fields"]}
    pitch = by[f"{BASE}.limits.head.pitch.max_deg"]
    assert pitch["value"] == pytest.approx(40) and pitch["unit"] == "deg"
    assert pitch["value"] == pytest.approx(math.degrees(MANIFEST.dofs["head.pitch"].max))
    assert by[f"{BASE}.limits.antenna_left.min_deg"]["value"] == pytest.approx(-175)
    z = by[f"{BASE}.limits.head.z.min_mm"]
    assert z["value"] == pytest.approx(-55) and z["unit"] == "mm"
    assert by[f"{BASE}.constraints.head_yaw_body_yaw.max_deg"]["value"] == 65
    assert by[f"{BASE}.safety.watchdog_ms"]["value"] == 200
    assert by[f"{BASE}.safety.watchdog_ms"]["unit"] == "ms"
    assert by[f"{BASE}.safety.estop"]["value"] == "none"


def test_unmeasured_dynamics_are_null_not_zero(split):
    _, _, robot = split
    by = {f["path"]: f for f in robot["fields"]}
    row = by[f"{BASE}.dynamics.head.max_vel_dps"]
    assert row["value"] is None and row["nullable"] is True and row["unit"] == "deg/s"
    assert f"{BASE}.dynamics.body_yaw.max_lin_vel_mps" not in by


def test_claims_resolve_in_the_catalogue_and_leave_the_core_list(split):
    editor, core, robot = split
    every = {f["path"] for f in editor.fields()}
    for claim in CLAIMS:
        assert any(claimed(p, (claim,)) for p in every), claim
    assert not [f for f in core if claimed(f["path"], CLAIMS)]
    moved = {f["path"] for f in robot["fields"] if claimed(f["path"], CLAIMS)}
    assert {"expression_map.joy.repeat", "gesture_map.nod.curve"} <= moved
    assert "circadian_motion.rest_pose_deg" in moved
    assert {f["path"] for f in core} | moved | {
        f["path"] for f in robot["fields"] if f["path"].startswith(BASE)
    } >= every


def test_behaviour_paths_stay_writable_by_path(split, stack):
    editor, _, robot = split
    row = next(f for f in robot["fields"] if f["path"] == "expression_map.joy.repeat")
    assert row["writable"] is True and row["source"] == "manifest"
    editor.change("expression_map.joy.repeat", 2)
    assert stack.effective().expression_map["joy"].repeat == 2


def test_hardware_rows_cannot_be_written(split):
    from roboemot.webui.settings import SettingNotAllowed

    editor, _, _ = split
    with pytest.raises(SettingNotAllowed):
        editor.change(f"{BASE}.limits.head.pitch.max_deg", 90)


def test_section_needs_no_stack():
    section = ReachySettings().section(None, MANIFEST)
    assert section.title == "Reachy Mini" and section.claims == CLAIMS
