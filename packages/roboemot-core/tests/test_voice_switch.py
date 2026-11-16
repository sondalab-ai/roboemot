"""The voice-provider switch of the Settings page (SPEC 8, 11A, 12): one choice per way the robot
speaks, each a full key set written as one batch, and the current choice computed from the keys
a robot already has, so a robot installed before the switch shows its choice with no change."""

from __future__ import annotations

import json

import pytest
import yaml
from roboemot import install, paths
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import ConfigStack, get_path
from roboemot.manifest import load_manifest
from roboemot.secret_store import SecretStore
from roboemot.webui import voice_switch as vs
from roboemot.webui.robot_section import (
    VoiceProfile,
    VoiceProfiled,
    load_settings_plugin,
    platform_voice,
)
from roboemot.webui.settings import SettingsEditor, piper_installed
from roboemot.webui.strings import DEFAULT_LANGUAGE, STRINGS
from test_compat_fixtures import COMPAT, MANIFEST, app_stack, platform_route, robot_data, start
from test_webui import PIN
from test_webui_settings import serve

GEMINI = "route:gemini"
LOOPBACK = {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3}
SWITCH_KEYS = vs.profiles().keys


def platform_choice() -> str:
    return vs.ROUTE_CHOICE + platform_route()


def started(name: str):
    """Fixture ``name`` through the real start path: its editor and effective config."""
    robot_data(name)
    admin, cfg, _ = start(app_stack())
    assert admin is not None
    return admin.settings, cfg


def switch_to(editor: SettingsEditor, choice_id: str) -> dict:
    """Pick ``choice_id`` the way the page does: its change list as one batch save."""
    choice = next(c for c in editor.voice_switch()["choices"] if c["id"] == choice_id)
    return editor.change_many(choice["changes"])


def voice_keys(cfg: RoboEmotConfig) -> dict:
    return {key: get_path(cfg, key) for key in SWITCH_KEYS}


# ------------------------------------------------------------------ matching on real robots


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [("cloud_voice", None), ("gemini_switch", GEMINI), ("local_voice", vs.LOCAL)],
)
def test_a_robot_shows_the_choice_its_keys_make(fixture, expected):
    editor, _ = started(fixture)
    assert editor.voice_switch()["value"] == (expected or platform_choice())


def test_the_platform_route_is_offered_once_with_the_plugin_keys():
    editor, _ = started("gemini_switch")
    by_id = {c["id"]: c for c in editor.voice_switch()["choices"]}
    assert list(by_id) == [vs.LOCAL, platform_choice(), GEMINI]
    mine = by_id[platform_choice()]
    assert mine["kind"] == vs.KIND_PLATFORM and mine["label"]
    assert by_id[GEMINI]["kind"] == vs.KIND_DIRECT and by_id[GEMINI]["label"] is None


def test_the_platform_keys_are_the_installer_cloud_voice_keys():
    frozen = yaml.safe_load((COMPAT / "cloud_voice" / "app.yaml").read_text())
    editor, _ = started("cloud_voice")
    mine = next(c for c in editor.voice_switch()["choices"] if c["id"] == platform_choice())
    written = {c["path"]: c["value"] for c in mine["changes"]}
    installed = {".".join(path): v for path, v in install.leaves(frozen).items()}
    defaults = vs.schema_defaults()
    assert written == {key: installed.get(key, defaults[key]) for key in SWITCH_KEYS}


def test_a_hand_mix_is_custom():
    editor, _ = started("cloud_voice")
    editor.change("voice.realtime_route.mode", "direct")  # the platform route, connected directly
    assert editor.voice_switch()["value"] == vs.CUSTOM
    editor, _ = started("local_voice")
    editor.change("voice.backend", "none")
    assert editor.voice_switch()["value"] == vs.CUSTOM


def test_a_tuned_key_keeps_the_choice():
    editor, _ = started("cloud_voice")
    editor.change("voice.realtime_route.transcript_timeout_s", 7.0)
    editor.change("voice.wake_word_policy", "always")
    assert editor.voice_switch()["value"] == platform_choice()


def test_local_ignores_a_route_left_named_while_the_route_is_off():
    editor, _ = started("cloud_voice")
    switch_to(editor, vs.LOCAL)
    editor.change("voice.realtime_route.route", platform_route())
    assert editor.voice_switch()["value"] == vs.LOCAL


# ------------------------------------------------------------------ the switch itself


def test_every_choice_writes_the_same_keys():
    editor, _ = started("gemini_switch")
    lists = [[c["path"] for c in ch["changes"]] for ch in editor.voice_switch()["choices"]]
    assert all(keys == list(SWITCH_KEYS) for keys in lists)


def test_every_switch_key_is_an_existing_setting():
    editor, _ = started("local_voice")
    rows = {row["path"]: row for row in editor.fields()}
    assert all(rows[key]["writable"] for key in SWITCH_KEYS)


def test_gemini_runs_direct_with_robot_turns_and_another_route_direct():
    cfg = RoboEmotConfig.model_validate(
        {
            "llm_routes": [
                {
                    "name": "g",
                    "kind": "realtime",
                    "protocol": "gemini_live",
                    "base_url": "wss://g",
                    "model": "m",
                },
                {"name": "o", "kind": "realtime", "base_url": "https://o", "model": "m"},
                {"name": "chat", "kind": "chat", "base_url": "https://c", "model": "m"},
            ]
        }
    )
    by_id = {c.id: c.values for c in vs.choices(cfg, None)}
    assert list(by_id) == [vs.LOCAL, "route:g", "route:o"]
    g, o = by_id["route:g"], by_id["route:o"]
    assert (g["voice.realtime_route.mode"], g["voice.realtime_route.turns"]) == ("direct", "robot")
    assert (g["voice.realtime_route.route"], o["voice.realtime_route.route"]) == ("g", "o")
    assert o["voice.realtime_route.mode"] == "direct"
    assert g["privacy.realtime_audio"] and not g["privacy.realtime_audio_unaddressed"]
    assert o["privacy.realtime_audio"] and o["privacy.realtime_audio_unaddressed"]


def test_local_takes_the_schema_defaults():
    local = vs.choices(RoboEmotConfig(), None)[0].values
    defaults = vs.schema_defaults()
    assert local["voice.enabled"] and not local["voice.realtime_route.enabled"]
    assert (local["voice.backend"], local["voice.fallback_backend"]) == ("piper", "piper")
    for key in ("voice.wake_word_policy", "voice.direction.enabled", "voice.facing.enabled"):
        assert local[key] == defaults[key]
    assert not local["privacy.realtime_audio"] and not local["privacy.realtime_audio_unaddressed"]


def test_a_round_trip_ends_where_the_cloud_install_starts():
    robot_data("cloud_voice")
    _, installed, _ = start(app_stack())
    editor, _ = started("gemini_switch")
    for choice in (platform_choice(), GEMINI, vs.LOCAL, platform_choice()):
        switch_to(editor, choice)
        assert editor.voice_switch()["value"] == choice
    assert voice_keys(editor.stack.effective()) == voice_keys(installed)
    _, replayed, _ = start(app_stack())  # a restart replays the overrides file
    assert voice_keys(replayed) == voice_keys(installed)


def test_one_switch_is_one_write_of_the_overrides_file(monkeypatch):
    editor, _ = started("gemini_switch")
    writes = []
    original = editor._persist
    monkeypatch.setattr(editor, "_persist", lambda: writes.append(1) or original())
    result = switch_to(editor, platform_choice())
    assert writes == [1] and result["restart_required"]


def test_the_switch_adds_no_key_to_app_yaml():
    data = robot_data("cloud_voice")
    before = (data / "app.yaml").read_bytes()
    editor, _ = started("cloud_voice")
    switch_to(editor, vs.LOCAL)
    assert (data / "app.yaml").read_bytes() == before
    stored = json.loads(paths.settings_overrides_path(str(data)).read_text())["overrides"]
    assert set(stored) <= set(SWITCH_KEYS)


# ------------------------------------------------------------------ availability


def test_a_route_whose_secret_is_missing_cannot_be_picked():
    robot_data("gemini_switch")
    SecretStore.for_data_dir().remove("gemini_key")
    admin, _, _ = start(app_stack())
    by_id = {c["id"]: c for c in admin.settings.voice_switch()["choices"]}
    assert (by_id[GEMINI]["available"], by_id[GEMINI]["reason"]) == (False, vs.SECRET_MISSING)
    assert by_id[GEMINI]["secret"] == "gemini_key"
    assert by_id[platform_choice()]["available"]


def local_choice(stack, models: bool, piper: bool) -> dict:
    """The Local choice of an editor whose voice models and Piper interpreter are as given."""
    lang = stack.effective().conversation.language
    editor = SettingsEditor(
        stack, installed=lambda cfg: [lang] if models else [], piper_ready=lambda cfg: piper
    )
    return editor.voice_switch()["choices"][0]


def test_local_needs_the_voice_models_and_the_piper_interpreter(tmp_path):
    stack = ConfigStack(app={"voice": {"enabled": True}}, audit_path=tmp_path / "audit.jsonl")
    missing = local_choice(stack, models=False, piper=True)
    assert (missing["available"], missing["reason"]) == (False, vs.MODELS_MISSING)
    no_piper = local_choice(stack, models=True, piper=False)
    assert (no_piper["available"], no_piper["reason"]) == (False, vs.PIPER_MISSING)
    assert local_choice(stack, models=True, piper=True)["available"]


def test_the_piper_check_is_the_interpreter_local_voice_runs(tmp_path):
    python = tmp_path / "piper" / "bin" / "python"
    cfg = RoboEmotConfig.model_validate({"voice": {"piper": {"python": str(python)}}})
    assert not piper_installed(cfg)
    python.parent.mkdir(parents=True)
    python.write_text("")
    assert piper_installed(cfg)


# ------------------------------------------------------------------ plugin contract


def test_the_robot_plugin_gives_the_platform_profile():
    robot_id = load_manifest(MANIFEST, allow_unmeasured_dynamics=True).robot.id
    plugin = load_settings_plugin(robot_id)
    assert isinstance(plugin, VoiceProfiled)
    profile = platform_voice(plugin, RoboEmotConfig(), "en")
    assert profile is not None and profile.route == platform_route()
    assert set(profile.keys) <= set(SWITCH_KEYS)


class Raising:
    def section(self, stack, manifest):
        raise AssertionError

    def voice_profile(self, cfg, language):
        raise RuntimeError("boom")


class Wrong(Raising):
    def voice_profile(self, cfg, language):
        return {"route": "x"}


@pytest.mark.parametrize("plugin", [Raising(), Wrong(), None, object()])
def test_a_broken_or_missing_plugin_profile_gives_none(plugin):
    assert platform_voice(plugin, RoboEmotConfig(), "en") is None


def test_a_plugin_key_outside_the_list_is_not_written():
    cfg = RoboEmotConfig.model_validate(
        {"llm_routes": [{"name": "p", "kind": "realtime", "base_url": "https://p", "model": "m"}]}
    )
    keys = {"voice.realtime_route.mode": "platform", "safety.x": 1}
    profile = VoiceProfile("p", "P", keys, "https://p/session")
    values = vs.choices(cfg, profile)[1].values
    assert list(values) == list(SWITCH_KEYS) and values["voice.realtime_route.mode"] == "platform"


def test_a_route_that_borrows_the_platform_name_runs_direct():
    cfg = RoboEmotConfig.model_validate(
        {"llm_routes": [{"name": "p", "kind": "realtime", "base_url": "https://x", "model": "m"}]}
    )
    profile = VoiceProfile("p", "Platform", {"voice.realtime_route.mode": "platform"}, "https://p")
    choice = vs.choices(cfg, profile)[1]
    assert (choice.kind, choice.label, choice.audio_to) == (vs.KIND_DIRECT, None, "x")
    assert choice.values["voice.realtime_route.mode"] == "direct"


# ------------------------------------------------------------------ API


@pytest.fixture
def served_cloud(tmp_path):
    robot_data("gemini_switch")
    app = paths.app_config_path()  # the test's copy, never the frozen fixture
    config = yaml.safe_load(app.read_text())
    app.write_text(yaml.safe_dump({**config, "webui": LOOPBACK}))
    admin, _, _ = start(app_stack())
    srv, client = serve(tmp_path, admin.settings, admin.settings.stack)
    yield client, admin.settings
    srv.stop()


def test_the_api_lists_the_switch_and_the_preview_shows_its_diff(served_cloud):
    client, editor = served_cloud
    data = client.request("GET", "/api/settings/voice").json
    assert set(data) == {"value", "revealed", "choices"} and data["value"] == GEMINI
    keys = {"id", "kind", "route", "protocol", "label", "audio_to", "available", "reason"}
    keys |= {"secret", "private_match", "changes"}
    assert all(set(c) == keys for c in data["choices"])
    local = next(c for c in data["choices"] if c["id"] == vs.LOCAL)
    body = {"changes": local["changes"], "pin": PIN}
    diff = client.request("POST", "/api/settings/batch/preview", body).json["changes"]
    by_path = {d["path"]: d for d in diff}
    assert by_path["voice.realtime_route.enabled"]["new"] is False
    assert by_path["privacy.realtime_audio"]["opt_in"]
    assert by_path["privacy.realtime_audio"]["new"] == "***"  # private, masked as in the list
    assert editor.voice_switch()["value"] == GEMINI  # a preview writes nothing
    saved = client.request("POST", "/api/settings/batch", body).json
    assert saved["restart_required"] and editor.voice_switch()["value"] == vs.LOCAL


def test_every_reason_kind_and_custom_is_worded():
    words = STRINGS[DEFAULT_LANGUAGE]
    assert {f"voice_reason_{r}" for r in (vs.SECRET_MISSING, vs.MODELS_MISSING)} <= set(words)
    kinds = (vs.KIND_LOCAL, vs.KIND_PLATFORM, vs.KIND_DIRECT, vs.CUSTOM)
    assert {f"voice_choice_{k}" for k in kinds} <= set(words)


# ------------------------------------------------------------------ private opt-ins (SPEC 12)

OPT_INS = ("privacy.realtime_audio", "privacy.realtime_audio_unaddressed")


def without_pin(name: str, opt_ins: bool) -> dict:
    """Fixture ``name`` with both privacy opt-ins set to ``opt_ins`` (as runtime edits), read
    the way the page reads it without the PIN."""
    editor, _ = started(name)
    editor.change_many([{"path": p, "value": opt_ins} for p in OPT_INS])
    return editor.voice_switch()


@pytest.mark.parametrize("name", ["cloud_voice", "local_voice"])
def test_without_the_pin_the_answer_does_not_depend_on_the_opt_ins(name):
    assert without_pin(name, True) == without_pin(name, False)


def test_the_changes_are_the_choice_values_not_the_robot_state():
    cloud, local = without_pin("cloud_voice", True), without_pin("local_voice", False)
    assert cloud["choices"][0]["changes"] == local["choices"][0]["changes"]  # Local on both
    on, off = without_pin("gemini_switch", True), without_pin("gemini_switch", False)
    assert [c["changes"] for c in on["choices"]] == [c["changes"] for c in off["choices"]]


def test_the_pin_path_matches_on_the_opt_ins_too():
    editor, _ = started("local_voice")
    editor.change("privacy.realtime_audio", True)
    assert editor.voice_switch()["value"] == vs.LOCAL
    assert editor.voice_switch(revealed=True) == {
        **editor.voice_switch(),
        "value": vs.CUSTOM,
        "revealed": True,
    }


def test_the_revealed_switch_needs_the_pin(served_cloud):
    client, _ = served_cloud
    assert client.request("GET", "/api/settings/voice").json["revealed"] is False
    shown = client.request("POST", "/api/settings/voice", {"pin": PIN}).json
    assert shown["revealed"] is True and shown["value"] == GEMINI
    refused = client.request("POST", "/api/settings/voice", {"pin": "000000"})
    assert refused.status == 401 and refused.json == {"error": "bad_pin"}


# ------------------------------------------------------------------ review round


def test_a_text_input_on_the_route_is_custom():
    editor, _ = started("gemini_switch")
    editor.change("voice.realtime_route.input", "text")
    assert editor.voice_switch(revealed=True)["value"] == vs.CUSTOM
    assert editor.voice_switch()["value"] == vs.CUSTOM


def test_gemini_writes_the_unaddressed_opt_in_off_and_an_old_switch_still_reads_gemini():
    editor, cfg = started("gemini_switch")
    assert cfg.privacy.realtime_audio_unaddressed  # the two-step switch kept it on
    assert editor.voice_switch(revealed=True)["value"] == GEMINI
    gemini = next(c for c in editor.voice_switch()["choices"] if c["id"] == GEMINI)
    diff = {d["path"]: d for d in editor.preview_many(gemini["changes"])}
    assert diff["privacy.realtime_audio_unaddressed"]["opt_in"]
    assert gemini["private_match"] == ["privacy.realtime_audio"]
    switch_to(editor, GEMINI)
    assert not editor.stack.effective().privacy.realtime_audio_unaddressed
    assert editor.voice_switch(revealed=True)["value"] == GEMINI


def test_a_switch_between_routes_with_the_opt_in_on_names_the_receiver():
    editor, _ = started("gemini_switch")
    switch_to(editor, platform_choice())
    gemini = next(c for c in editor.voice_switch()["choices"] if c["id"] == GEMINI)
    diff = {d["path"] for d in editor.preview_many(gemini["changes"])}
    assert "voice.realtime_route.route" in diff
    assert "privacy.realtime_audio" not in diff  # on before and after: no opt-in flips
    assert gemini["audio_to"] == "generativelanguage.googleapis.com"
    mine = next(c for c in editor.voice_switch()["choices"] if c["id"] == platform_choice())
    assert mine["audio_to"] == mine["label"]


def test_an_openai_realtime_route_switch_passes_the_validators(tmp_path):
    robot_data("cloud_voice")
    app = paths.app_config_path()
    config = yaml.safe_load(app.read_text())
    other = {
        "name": "other",
        "kind": "realtime",
        "base_url": "https://realtime.example.com/v1",
        "model": "m",
        "api_key": "${secret:hf_token}",
    }
    app.write_text(yaml.safe_dump({**config, "llm_routes": [*config["llm_routes"], other]}))
    admin, _, _ = start(app_stack())
    editor = admin.settings
    switch_to(editor, "route:other")
    rt = editor.stack.effective().voice.realtime_route
    assert (rt.route, rt.mode, rt.enabled) == ("other", "direct", True)
    assert editor.voice_switch(revealed=True)["value"] == "route:other"


@pytest.mark.parametrize("lang", sorted(set(STRINGS) - {DEFAULT_LANGUAGE}))
def test_every_voice_string_has_its_translation(lang):
    mine = {k for k in STRINGS[DEFAULT_LANGUAGE] if k.startswith(("voice_switch", "voice_choice"))}
    mine |= {k for k in STRINGS[DEFAULT_LANGUAGE] if k.startswith("voice_reason_")}
    assert mine <= set(STRINGS[lang])
    assert f"voice_reason_{vs.PIPER_MISSING}" in mine
