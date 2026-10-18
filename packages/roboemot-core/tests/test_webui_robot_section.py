"""Robot settings section: plugin loading, claimed paths, read-only rows (SPEC 11A, 12)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
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
SECRET = "sk-LEAK-CANARY-7f3a9c41"
HALF_LIFE = "affect.half_life_s"
HOLD = "affect.event_label_hold_s"


def manifest():
    return load_manifest(MANIFEST_RAW)


class FakePlugin:
    """Claims two affect paths and adds one read-only row."""

    def __init__(self, rows=None, claims=(HALF_LIFE, HOLD)):
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
    assert paths == [f"{BASE}.arm.max_deg", HALF_LIFE, HOLD]
    assert all(set(f) == DESCRIPTOR_KEYS for f in robot["fields"])
    core_paths = {f["path"] for f in core}
    assert not core_paths & {HALF_LIFE, HOLD}
    assert "affect.labels" in core_paths


def test_claim_covers_the_path_and_everything_below_it(tmp_path):
    ed, _ = editor(tmp_path, [ep(FakePlugin(claims=("affect.labels",)))])
    core, robot = ed.robot_split(ed.fields())
    assert not [f for f in core if f["path"].startswith("affect.labels")]
    assert "affect.labels" in [f["path"] for f in robot["fields"]]
    assert HALF_LIFE in {f["path"] for f in core}


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
        assert HALF_LIFE not in top
        assert HALF_LIFE in [f["path"] for f in data["robot"]["fields"]]
        assert data["robot"]["actions"] == []
        refused = put(client, f"{BASE}.arm.max_deg", 10)
        assert refused.status == 403 and refused.json["error"] == "setting_not_allowed"
        reset = client.request(
            "POST", "/api/settings/reset", {"path": f"{BASE}.arm.max_deg", "pin": PIN}
        )
        assert reset.status == 403
        assert put(client, HALF_LIFE, 77).status == 200
        data = client.request("GET", "/api/settings").json
        row = next(f for f in data["robot"]["fields"] if f["path"] == HALF_LIFE)
        assert row["value"] == 77 and row["writable"] is True
    finally:
        srv.stop()


def test_api_without_plugin_has_robot_null(tmp_path):
    ed, stack = editor(tmp_path, [])
    srv, client = serve(tmp_path, ed, stack)
    try:
        data = client.request("GET", "/api/settings").json
        assert data["robot"] is None
        assert HALF_LIFE in {f["path"] for f in data["fields"]}
    finally:
        srv.stop()


def test_reveal_still_works_with_a_robot_section(tmp_path):
    ed, stack = editor(tmp_path, [ep(FakePlugin())])
    stack.set("circadian.location", {"lat": 45.4642, "lon": 9.19}, source="t")
    srv, client = serve(tmp_path, ed, stack)
    try:
        listing = client.request("GET", "/api/settings")
        assert b"45.4642" not in listing.body
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


def test_rebootable_plugin_gets_the_reboot_action(tmp_path):
    class Rebooting(FakePlugin):
        def reboot(self) -> None:
            pass

    ed, _ = editor(tmp_path, [ep(Rebooting())])
    assert ed.robot_split(ed.fields())[1]["actions"] == ["reboot"]


def test_actions_a_plugin_sets_itself_are_ignored(tmp_path):
    class Pushy(FakePlugin):
        def section(self, stack, manifest):
            base = super().section(stack, manifest)
            return RobotSection(base.title, base.fields, base.claims, ("reboot",))

    ed, _ = editor(tmp_path, [ep(Pushy())])
    assert ed.robot_split(ed.fields())[1]["actions"] == []


def raw_row(path, **over):
    row = dict.fromkeys(DESCRIPTOR_KEYS)
    row.update(path=path, kind="text", nullable=False, restart=False, admin_only=False)
    row.update(opt_in=False, description="", secret=False, private=False, writable=True)
    row.update(source="manifest", **over)
    return row


def test_core_reseals_a_hand_built_row_and_never_mutates_the_plugin_dict(tmp_path):
    leaky = "https://u:pw-CANARY-8@host/v1"
    raw = raw_row(f"{BASE}.leaky", value=leaky, default=leaky)
    ed, _ = editor(tmp_path, [ep(FakePlugin(rows=[raw], claims=()))])
    _, robot = ed.robot_split(ed.fields())
    (row,) = robot["fields"]
    assert row["secret"] is True and row["value"] == "***" and row["default"] == "***"
    assert row["writable"] is False
    assert raw["value"] == leaky and raw["writable"] is True and raw["secret"] is False


def test_rows_that_are_not_full_descriptors_are_dropped_with_the_path_only(tmp_path, caplog):
    short = {"path": f"{BASE}.short", "kind": "text", "value": SECRET}
    ed, _ = editor(tmp_path, [ep(FakePlugin(rows=[short, "nope"], claims=()))])
    with caplog.at_level(logging.WARNING):
        _, robot = ed.robot_split(ed.fields())
    assert robot["fields"] == []
    assert f"{BASE}.short" in caplog.text and SECRET not in caplog.text


@pytest.mark.parametrize(
    "claim",
    ["safety", "safety.watchdog_ms", "webui", "privacy", "llm_routes", "", "voice.wake_word"],
)
def test_protected_and_admin_claims_are_refused(tmp_path, caplog, claim):
    ed, _ = editor(tmp_path, [ep(FakePlugin(claims=(claim,)))])
    fields = ed.fields()
    with caplog.at_level(logging.WARNING):
        core, robot = ed.robot_split(fields)
    assert core == fields
    assert [f["path"] for f in robot["fields"]] == [f"{BASE}.arm.max_deg"]
    assert "claim refused" in caplog.text


def test_a_parent_of_private_rows_is_refused(tmp_path):
    ed, _ = editor(tmp_path, [ep(FakePlugin(claims=("circadian", "register")))])
    fields = ed.fields()
    assert ed.robot_split(fields)[0] == fields


def test_a_claim_that_matches_nothing_is_warned(tmp_path, caplog):
    ed, _ = editor(tmp_path, [ep(FakePlugin(claims=("no.such.path",)))])
    with caplog.at_level(logging.WARNING):
        ed.robot_split(ed.fields())
    assert "matches nothing: no.such.path" in caplog.text


def test_a_plugin_row_that_repeats_a_claimed_core_path_is_dropped(tmp_path, caplog):
    dup = descriptor(HALF_LIFE, "number", 1.0)
    ed, _ = editor(tmp_path, [ep(FakePlugin(rows=[dup], claims=(HALF_LIFE,)))])
    with caplog.at_level(logging.WARNING):
        _, robot = ed.robot_split(ed.fields())
    assert [f["path"] for f in robot["fields"]] == [HALF_LIFE]
    assert robot["fields"][0]["value"] != 1.0
    assert f"outside {BASE}." in caplog.text


def test_a_plugin_without_section_is_fail_soft(tmp_path):
    ed, _ = editor(tmp_path, [ep(object())])
    fields = ed.fields()
    assert ed.robot_split(fields) == (fields, None)
