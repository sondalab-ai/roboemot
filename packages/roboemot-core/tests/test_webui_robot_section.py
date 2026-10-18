"""Robot settings section: plugin loading, claimed paths, read-only rows (SPEC 11A, 12)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

from roboemot.config.catalogue import DESCRIPTOR_KEYS, descriptor
from roboemot.config.stack import ConfigStack
from roboemot.manifest import load_manifest
from roboemot.webui.robot_section import RobotSection
from roboemot.webui.settings import SettingsEditor
from test_webui_settings import PIN, put, serve

ROBOT_ID = "fake-bot"
BASE = f"robot.{ROBOT_ID}"
MANIFEST_RAW = {
    "robot": {"id": ROBOT_ID},
    "pose_groups": {"arm": {"type": "joint", "limits": {"min_deg": -90, "max_deg": 90}}},
    "dynamics": {"arm": {"max_vel_dps": 50, "max_acc_dps2": 100}},
}
SECRET = "sk-PLUGIN-CANARY-31"


def manifest():
    return load_manifest(MANIFEST_RAW)


class FakePlugin:
    """Claims ``voice.wake_word`` (hooked) and ``circadian.location`` (private), adds one row."""

    def __init__(self, rows=None, claims=("voice.wake_word", "circadian.location")):
        self.rows = rows
        self.claims = claims

    def section(self, stack, manifest):
        rows = self.rows or [
            descriptor(
                f"{BASE}.arm.max_deg", "number", 90.0, source="manifest", unit="deg", writable=True
            )
        ]
        return RobotSection("Fake Bot", rows, self.claims)


def ep(obj=None, name=ROBOT_ID, error=None):
    def load():
        if error:
            raise error
        return obj

    return SimpleNamespace(name=name, load=load)


def editor(tmp_path, eps, stack=None):
    stack = stack or ConfigStack(app={"webui": {"bind": "127.0.0.1", "port": 0}})
    with patch("roboemot.webui.robot_section.entry_points", return_value=eps):
        return SettingsEditor(stack, store=tmp_path / "s.json", manifest=manifest()), stack


def test_plugin_is_found_by_robot_id_and_claimed_paths_move(tmp_path):
    ed, _ = editor(tmp_path, [ep(FakePlugin(), name="other"), ep(FakePlugin())])
    core, robot = ed.robot_split(ed.fields())
    assert robot["id"] == ROBOT_ID and robot["title"] == "Fake Bot"
    paths = [f["path"] for f in robot["fields"]]
    assert paths == [f"{BASE}.arm.max_deg", "circadian.location", "voice.wake_word"]
    assert all(set(f) == DESCRIPTOR_KEYS for f in robot["fields"])
    core_paths = {f["path"] for f in core}
    assert not core_paths & {"voice.wake_word", "circadian.location"}
    assert "affect.half_life_s" in core_paths


def test_claim_covers_the_path_and_everything_below_it(tmp_path):
    ed, _ = editor(tmp_path, [ep(FakePlugin(claims=("llm_routes",)))])
    core, robot = ed.robot_split(ed.fields())
    assert not [f for f in core if f["path"].startswith("llm_routes")]
    assert "llm_routes" in [f["path"] for f in robot["fields"]]
    assert "llm_routes_other" not in {f["path"] for f in robot["fields"]}


def test_no_plugin_means_no_robot_section(tmp_path):
    ed, _ = editor(tmp_path, [ep(FakePlugin(), name="other")])
    fields = ed.fields()
    assert ed.robot_split(fields) == (fields, None)


def test_no_manifest_means_no_robot_section(tmp_path):
    ed = SettingsEditor(ConfigStack(), store=tmp_path / "s.json")
    fields = ed.fields()
    assert ed.robot_split(fields) == (fields, None)


def test_import_error_is_fail_soft_and_logs_only_the_type(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    ed, _ = editor(tmp_path, [ep(error=ImportError(f"no module {SECRET}"))])
    fields = ed.fields()
    assert ed.robot_split(fields) == (fields, None)
    assert "ImportError" in caplog.text and SECRET not in caplog.text


def test_exception_in_section_is_fail_soft_and_logs_only_the_type(tmp_path, caplog):
    class Broken:
        def section(self, stack, manifest):
            raise RuntimeError(SECRET)

    ed, _ = editor(tmp_path, [ep(Broken())])
    fields = ed.fields()
    with caplog.at_level(logging.INFO):
        assert ed.robot_split(fields) == (fields, None)
    assert "RuntimeError" in caplog.text and SECRET not in caplog.text


def test_rows_outside_the_robot_prefix_are_dropped_and_rows_are_read_only(tmp_path):
    rows = [
        descriptor("voice.language", "text", "xx", writable=True),
        descriptor(f"{BASE}.ok", "text", "v", writable=True),
    ]
    ed, _ = editor(tmp_path, [ep(FakePlugin(rows=rows, claims=()))])
    _, robot = ed.robot_split(ed.fields())
    assert [(f["path"], f["writable"]) for f in robot["fields"]] == [(f"{BASE}.ok", False)]


def test_api_lists_robot_block_and_refuses_writes_to_robot_rows(tmp_path):
    ed, stack = editor(tmp_path, [ep(FakePlugin())])
    srv, client = serve(tmp_path, ed, stack)
    try:
        data = client.request("GET", "/api/settings").json
        assert data["robot"]["id"] == ROBOT_ID
        top = {f["path"] for f in data["fields"]}
        assert "voice.wake_word" not in top
        assert "voice.wake_word" in [f["path"] for f in data["robot"]["fields"]]
        refused = put(client, f"{BASE}.arm.max_deg", 10)
        assert refused.status == 403 and refused.json["error"] == "setting_not_allowed"
        reset = client.request(
            "POST", "/api/settings/reset", {"path": f"{BASE}.arm.max_deg", "pin": PIN}
        )
        assert reset.status == 403
        assert put(client, "voice.wake_word", "Robbie").status == 200
        data = client.request("GET", "/api/settings").json
        row = next(f for f in data["robot"]["fields"] if f["path"] == "voice.wake_word")
        assert row["value"] == "Robbie" and row["writable"] is True
    finally:
        srv.stop()


def test_api_without_plugin_has_robot_null(tmp_path):
    ed, stack = editor(tmp_path, [])
    srv, client = serve(tmp_path, ed, stack)
    try:
        data = client.request("GET", "/api/settings").json
        assert data["robot"] is None
        assert "voice.wake_word" in {f["path"] for f in data["fields"]}
    finally:
        srv.stop()


def test_reveal_still_returns_claimed_private_fields(tmp_path):
    ed, stack = editor(tmp_path, [ep(FakePlugin())])
    stack.set("circadian.location", {"lat": 45.4642, "lon": 9.19}, source="t")
    srv, client = serve(tmp_path, ed, stack)
    try:
        data = client.request("GET", "/api/settings").json
        row = next(f for f in data["robot"]["fields"] if f["path"] == "circadian.location")
        assert row["private"] and row["value"] == "***"
        assert b"45.4642" not in client.request("GET", "/api/settings").body
        shown = client.request("POST", "/api/settings/reveal", {"pin": PIN}).json["fields"]
        real = next(f for f in shown if f["path"] == "circadian.location")
        assert real["value"]["lat"] == 45.4642
    finally:
        srv.stop()


def test_plugin_rows_with_secret_shaped_values_are_redacted(tmp_path):
    rows = [
        descriptor(f"{BASE}.token", "text", SECRET, default=SECRET),
        descriptor(f"{BASE}.note", "text", "fine"),
        descriptor(f"{BASE}.env", "string_dict", {"api_key": "x"}),
    ]
    ed, stack = editor(tmp_path, [ep(FakePlugin(rows=rows, claims=()))])
    srv, client = serve(tmp_path, ed, stack)
    try:
        resp = client.request("GET", "/api/settings")
        assert SECRET.encode() not in resp.body and b'"api_key": "x"' not in resp.body
        by_path = {f["path"]: f for f in resp.json["robot"]["fields"]}
        assert by_path[f"{BASE}.token"]["secret"] and by_path[f"{BASE}.token"]["value"] == "***"
        assert by_path[f"{BASE}.note"]["value"] == "fine"
        assert by_path[f"{BASE}.env"]["secret"] and by_path[f"{BASE}.env"]["value"] == "***"
    finally:
        srv.stop()
