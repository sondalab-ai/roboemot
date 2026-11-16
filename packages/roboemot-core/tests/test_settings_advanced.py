"""The Advanced fold of the settings page: which fields are tuning (SPEC 11A, 12).

``param(advanced=True)`` marks a field the page folds under "Advanced"; on a container it covers
every field inside (``field_meta``). These tests pin the rules of the split over every schema
path, dict and list entries included, and the flag on the API rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args, get_origin

import pytest
from roboemot.config.catalogue import build_catalogue
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import ConfigStack, field_meta, strip_optional
from roboemot.webui.settings import SettingsEditor
from roboemot.webui.strings import STRINGS
from test_webui_settings import serve

SETTINGS_JS = (
    Path(__file__).resolve().parents[1] / "src" / "roboemot" / "webui" / "static" / "settings.js"
)
ENTRY = "_"  # one path element for a dict key or a list index (field_meta skips it)
NEVER_ADVANCED = ("safety", "privacy")
# Sections with no basic field on purpose: tuning only, or shipped by the robot manifest.
WHOLLY_ADVANCED = {
    "appraisal",
    "catalogue",
    "circadian_motion",
    "connectivity",
    "dance_motion_map",
    "data_dir",
    "enrollment",
    "expression_map",
    "gesture_map",
    "identity",
    "light",
    "llm_check",
    "motion",
    "speech_motion_map",
}


def _model(annotation):
    """The model a field holds (itself, or as dict values or list items), else ``None``."""
    annotation = strip_optional(annotation)
    if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
        return annotation
    if get_origin(annotation) in (dict, list):
        args = get_args(annotation)
        if args and isinstance(args[-1], type) and hasattr(args[-1], "model_fields"):
            return args[-1]
    return None


def schema_leaves(model=RoboEmotConfig, prefix: str = "") -> list[str]:
    """Every leaf path of the schema, with ``_`` for each dict key or list index."""
    out: list[str] = []
    for name, info in model.model_fields.items():
        path = prefix + name
        sub = _model(info.annotation)
        if sub is None:
            out.append(path)
            continue
        keyed = get_origin(strip_optional(info.annotation)) in (dict, list)
        out += schema_leaves(sub, f"{path}.{ENTRY}." if keyed else f"{path}.")
    return out


LEAVES = {path: field_meta(path) for path in schema_leaves()}


def test_every_schema_leaf_is_reached():
    assert len(LEAVES) > 800 and all(LEAVES.values())


def test_no_safety_privacy_or_opt_in_field_is_advanced():
    wrong = [
        p
        for p, meta in LEAVES.items()
        if meta.get("advanced") and (p.split(".")[0] in NEVER_ADVANCED or meta.get("opt_in"))
    ]
    assert wrong == []


def test_every_section_keeps_a_basic_field_or_is_wholly_advanced_on_purpose():
    basic: dict[str, int] = {}
    for path, meta in LEAVES.items():
        section = path.split(".")[0]
        basic[section] = basic.get(section, 0) + (not meta.get("advanced"))
    assert {s for s, n in basic.items() if n == 0} == WHOLLY_ADVANCED


@pytest.mark.parametrize(
    ("path", "advanced"),
    [
        ("voice.vad.threshold", True),  # inherited from the container field voice.vad
        ("voice.realtime_route.connect_timeout_s", True),
        ("voice.realtime_route.enabled", False),
        ("conversation.language", False),
        ("circadian.quiet_hours", False),
        ("webui.live.video", False),  # an opt-in
        ("llm_routes._.name", False),
        ("llm_routes._.first_token_timeout_s", True),
        ("appraisal.rules._.valence", True),
    ],
)
def test_field_meta_reads_the_flag(path, advanced):
    assert bool(field_meta(path).get("advanced")) is advanced


def test_the_flag_reaches_the_catalogue_rows_and_the_entry_forms():
    rows = {r["path"]: r for r in build_catalogue(ConfigStack())}
    assert rows["voice.vad.threshold"]["advanced"] is True
    assert rows["voice.enabled"]["advanced"] is False
    assert rows["appraisal.rules"]["advanced"] is True
    entry = {e["path"]: e for e in rows["appraisal.rules"]["entry"]}
    assert entry["valence"]["advanced"] is True


def test_the_flag_is_carried_to_the_api_row(tmp_path):
    stack = ConfigStack(app={"webui": {"bind": "127.0.0.1", "port": 0}})
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / "s.json"), stack)
    try:
        rows = {r["path"]: r for r in client.request("GET", "/api/settings").json["fields"]}
    finally:
        srv.stop()
    assert rows["voice.vad.threshold"]["advanced"] is True
    assert rows["voice.wake_word"]["advanced"] is False


def page_purposes() -> list[tuple[str, list[str]]]:
    """``SETTING_PURPOSES`` of ``settings.js``: (purpose id, its section names) in page order."""
    source = SETTINGS_JS.read_text()
    start = source.index("const SETTING_PURPOSES = [")
    block = source[start : source.index("];", start)]
    return [
        (pid, re.findall(r'"([a-z_]+)"', names))
        for pid, names in re.findall(r'\{ id: "([a-z]+)", sections: \[([^\]]*)\] \}', block)
    ]


def test_every_schema_section_is_in_one_purpose_group_and_each_has_its_heading():
    purposes = page_purposes()
    listed = [s for _, names in purposes for s in names]
    assert [p for p, _ in purposes] == [
        "voice", "behaviour", "perception", "motion", "models", "system"
    ]  # fmt: skip
    assert sorted(listed) == sorted(RoboEmotConfig.model_fields), "each section once, none unknown"
    for lang in STRINGS:
        for pid in [*(p for p, _ in purposes), "other"]:
            assert STRINGS[lang].get(f"settings_purpose_{pid}"), (lang, pid)
