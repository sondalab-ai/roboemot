"""All-or-nothing batch settings save and its preview (SPEC 11A, 12): every entry is checked before
anything is written, the stack validates the final state once and the overrides file is written
once, so a provider switch that spans fields never lands half done."""

from __future__ import annotations

import copy
import json

import pytest
from roboemot.config.stack import ConfigError, ConfigStack, get_path
from roboemot.webui.api import WebApi
from roboemot.webui.settings import MAX_BATCH_CHANGES, SettingNotAllowed, SettingsEditor
from roboemot.webui.validation import ValidationError
from test_webui import PIN
from test_webui_settings import FILE, serve

MODE = "voice.realtime_route.mode"
ROUTE = "voice.realtime_route.route"
GEMINI_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
PLATFORM_URL = "https://platform.example/session"
# what the cloud-voice installer writes today (platform route, platform mode), plus a gemini route
CLOUD_VOICE = {
    "webui": {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3},
    "llm_routes": [
        {
            "name": "platform",
            "kind": "realtime",
            "base_url": PLATFORM_URL,
            "model": "s2s",
            "api_key": "${secret:platform_token}",
        },
        {
            "name": "gemini",
            "kind": "realtime",
            "protocol": "gemini_live",
            "base_url": GEMINI_URL,
            "model": "gemini-live",
            "api_key": "${secret:gemini_key}",
            "connect_timeout_s": 15,
        },
    ],
    "voice": {
        "backend": "none",
        "fallback_backend": "none",
        "wake_word_policy": "never",
        "direction": {"enabled": True},
        "facing": {"enabled": True},
        "realtime_route": {
            "enabled": True,
            "route": "platform",
            "mode": "platform",
            "input": "audio",
            "skip_tool": True,
            "tone_tool": True,
            "transcript_timeout_s": 10.0,
        },
    },
    "privacy": {"realtime_audio": True, "realtime_audio_unaddressed": True},
}
TO_GEMINI = [{"path": MODE, "value": "direct"}, {"path": ROUTE, "value": "gemini"}]


def make_stack(tmp_path):
    stack = ConfigStack(app=copy.deepcopy(CLOUD_VOICE), audit_path=tmp_path / "audit.jsonl")
    stack.changes = []
    stack.subscribe(stack.changes.append)
    return stack


@pytest.fixture
def stack(tmp_path):
    return make_stack(tmp_path)


@pytest.fixture
def editor(tmp_path, stack):
    return SettingsEditor(stack, store=tmp_path / FILE)


def snapshot(stack, tmp_path):
    """Everything a refused batch must leave as it was: values, runtime layer, audit, file."""
    file = tmp_path / FILE
    rt = stack.effective().voice.realtime_route
    return (
        (rt.mode, rt.route, stack.effective().affect.update_hz),
        dict(stack._runtime),
        len(stack.audit),
        file.read_text() if file.exists() else None,
    )


def saved(tmp_path):
    return json.loads((tmp_path / FILE).read_text())["overrides"]


# ------------------------------------------------------------------ atomicity


def test_a_refused_second_key_leaves_the_first_unwritten(editor, stack, tmp_path):
    editor.change("affect.half_life_s", 9.0)  # a file already on disk
    before = snapshot(stack, tmp_path)
    batch = [{"path": "affect.update_hz", "value": 25.0}, {"path": "affect.update_hz2", "value": 1}]
    with pytest.raises(SettingNotAllowed):
        editor.change_many(batch)
    bad_range = [{"path": "affect.update_hz", "value": 25.0}, {"path": MODE, "value": "nope"}]
    with pytest.raises(ValidationError) as caught:
        editor.change_many(bad_range)
    assert caught.value.code == "value_choice" and caught.value.extra == {"path": MODE}
    assert snapshot(stack, tmp_path) == before


def test_a_batch_the_stack_refuses_writes_nothing_in_memory_or_on_disk(editor, stack, tmp_path):
    editor.change("affect.half_life_s", 9.0)
    before = snapshot(stack, tmp_path)
    batch = [{"path": "affect.update_hz", "value": 25.0}, {"path": ROUTE, "value": "gemini"}]
    with pytest.raises(ConfigError):
        editor.change_many(batch)  # gemini needs mode: direct, which the batch does not set
    assert snapshot(stack, tmp_path) == before
    assert saved(tmp_path) == {"affect.half_life_s": 9.0}


def test_a_safety_loosening_refuses_the_whole_batch(editor, stack, tmp_path):
    before = snapshot(stack, tmp_path)
    app = copy.deepcopy(stack.app)
    batch = [
        {"path": "affect.update_hz", "value": 25.0},
        {"path": "webui.name_max_chars", "value": 100},
    ]
    with pytest.raises(ValidationError) as caught:
        editor.change_many(batch)
    assert caught.value.code == "batch_loosens_safety"
    assert snapshot(stack, tmp_path) == before
    assert stack.app == app, "a batch never falls back to the app layer"


def test_the_platform_to_gemini_switch_lands_in_one_save(editor, stack, tmp_path):
    out = editor.change_many(TO_GEMINI)
    rt = stack.effective().voice.realtime_route
    assert (rt.mode, rt.route, rt.turns) == ("direct", "gemini", "robot")
    assert out["changed"] and out["layer"] == "runtime"
    assert out["persisted"] and out["applied"] and out["restart_required"]
    assert [r["path"] for r in out["results"]] == [MODE, ROUTE]
    assert all(r["layer"] == "runtime" and r["persisted"] for r in out["results"])
    assert saved(tmp_path) == {MODE: "direct", ROUTE: "gemini"}
    assert len(stack.changes) == 1, "one reload for the whole batch"
    replayed = make_stack(tmp_path)
    SettingsEditor(replayed, store=tmp_path / FILE).restore()
    assert replayed.effective().voice.realtime_route.route == "gemini"


def test_the_gemini_route_alone_is_refused(tmp_path):
    for one in TO_GEMINI[1:]:
        stack = make_stack(tmp_path)
        with pytest.raises(ConfigError, match="mode must be direct"):
            SettingsEditor(stack, store=tmp_path / FILE).change(one["path"], one["value"])


def test_restart_required_aggregates_over_the_paths(editor):
    quiet = editor.change_many([{"path": "affect.update_hz", "value": 25.0}])
    assert quiet["restart_required"] is False
    mixed = editor.change_many(
        [{"path": "affect.update_hz", "value": 26.0}, {"path": MODE, "value": "direct"}]
    )
    assert mixed["restart_required"] is True
    assert [r["restart_required"] for r in mixed["results"]] == [False, True]


def test_an_unchanged_batch_writes_nothing(editor, stack, tmp_path):
    out = editor.change_many([{"path": MODE, "value": "platform"}])
    assert not out["changed"] and not out["restart_required"] and out["persisted"]
    assert out["results"][0]["changed"] is False
    assert stack.audit == [] and stack.changes == [] and not (tmp_path / FILE).exists()


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("at", [0, 1])
@pytest.mark.parametrize("value", ["${secret:k}", "https://u:p@host/v1", ["x", "${secret:k}"]])
def test_a_secret_shaped_value_in_any_change_is_refused(editor, stack, tmp_path, at, value):
    batch = [{"path": "affect.update_hz", "value": 25.0}]
    batch.insert(at, {"path": "personality", "value": value})
    with pytest.raises(ValidationError) as caught:
        editor.change_many(batch)
    assert caught.value.code == "value_type" and caught.value.extra == {"path": "personality"}
    assert stack.audit == [] and not (tmp_path / FILE).exists()


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ([], "batch_empty"),
        ([{"path": MODE, "value": "direct"}] * 2, "batch_duplicate"),
        ([{"path": "affect.update_hz", "value": 25.0}] * (MAX_BATCH_CHANGES + 1), "batch_too_long"),
        ({"path": MODE, "value": "direct"}, "value_type"),
        ([{"path": MODE}], "value_type"),
        ([{"path": MODE, "value": "direct", "pin": PIN}], "value_type"),
    ],
)
def test_a_malformed_batch_is_refused_whole(editor, stack, tmp_path, changes, code):
    with pytest.raises(ValidationError) as caught:
        editor.change_many(changes)
    assert caught.value.code == code
    with pytest.raises(ValidationError):
        editor.preview_many(changes)
    assert stack.audit == [] and not (tmp_path / FILE).exists()


MOVE = "llm_routes.platform.base_url"


@pytest.mark.parametrize(
    "batch",
    [
        [{"path": MOVE, "value": "https://moved.example/session"}],
        [
            {"path": MOVE, "value": "https://moved.example/session"},
            {"path": "llm_routes.platform.kind", "value": "chat"},
            {"path": MODE, "value": "direct"},
        ],
        [
            {"path": MOVE, "value": "wss://moved.example/ws"},
            {"path": "llm_routes.platform.protocol", "value": "gemini_live"},
            {"path": MODE, "value": "direct"},
        ],
    ],
    ids=["alone", "with_kind", "with_protocol"],
)
def test_a_batch_never_moves_a_keyed_route_endpoint(editor, stack, tmp_path, batch):
    """Regression: an acknowledged move exempted the route's key from the secret check for the
    whole final state, so a batch could also change its kind or protocol past that check."""
    editor.change("affect.half_life_s", 9.0)
    before = snapshot(stack, tmp_path)
    for call in (editor.change_many, editor.preview_many):
        with pytest.raises(ValidationError) as caught:
            call(batch)
        assert caught.value.code == "batch_endpoint_move"
        assert caught.value.extra == {"path": MOVE}
    assert snapshot(stack, tmp_path) == before
    assert stack.effective().llm_routes[0].base_url == PLATFORM_URL


def test_a_file_that_cannot_be_written_keeps_the_edit_live(editor, stack, monkeypatch):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(editor._file, "save", fail)
    out = editor.change_many(TO_GEMINI)
    assert out["persisted"] is False and all(r["persisted"] is False for r in out["results"])
    assert stack.effective().voice.realtime_route.route == "gemini"
    assert editor._overrides == {MODE: "direct", ROUTE: "gemini"}


# ------------------------------------------------------------------ preview


def test_preview_lists_the_diff_and_never_writes(editor, stack, tmp_path):
    keep = [{"path": "affect.update_hz", "value": 20.0}]  # the default: unchanged, not listed
    diff = editor.preview_many([*TO_GEMINI, *keep])
    assert diff == [
        {"path": MODE, "old": "platform", "new": "direct", "restart": True, "opt_in": False},
        {"path": ROUTE, "old": "platform", "new": "gemini", "restart": True, "opt_in": False},
    ]
    assert stack.effective().voice.realtime_route.mode == "platform"
    assert stack._runtime == {} and stack.audit == [] and stack.changes == []
    assert not (tmp_path / FILE).exists()


def test_preview_masks_private_values_and_names_opt_ins(editor):
    diff = editor.preview_many([{"path": "privacy.realtime_audio", "value": False}])
    assert diff == [
        {
            "path": "privacy.realtime_audio",
            "old": "***",
            "new": "***",
            "restart": diff[0]["restart"],
            "opt_in": True,
        }
    ]


def test_preview_masks_a_credential_holding_route_url(tmp_path):
    app = copy.deepcopy(CLOUD_VOICE)
    app["llm_routes"].append(
        {"name": "lan", "base_url": "https://user:pass@lan.example/v1", "model": "m"}
    )
    stack = ConfigStack(app=app, audit_path=tmp_path / "audit.jsonl")
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    diff = editor.preview_many(
        [{"path": "llm_routes.lan.base_url", "value": "https://lan.example/v2"}]
    )
    assert (diff[0]["old"], diff[0]["new"]) == ("***", "***")
    assert stack.effective().llm_routes[2].base_url == "https://user:pass@lan.example/v1"


def test_preview_of_an_invalid_final_state_is_refused(editor):
    with pytest.raises(ConfigError):
        editor.preview_many(TO_GEMINI[1:])


# ------------------------------------------------------------------ API


@pytest.fixture
def served(tmp_path, editor, stack):
    srv, client = serve(tmp_path, editor, stack)
    yield client
    srv.stop()


def test_the_api_checks_the_pin_once_for_the_batch(served, stack, monkeypatch):
    checks = []
    real = WebApi._check_pin
    monkeypatch.setattr(
        WebApi, "_check_pin", lambda self, req, pin: (checks.append(pin), real(self, req, pin))
    )
    resp = served.request("POST", "/api/settings/batch", {"changes": TO_GEMINI, "pin": PIN})
    assert resp.status == 200 and checks == [PIN]
    assert resp.json["restart_required"] and len(resp.json["results"]) == 2
    assert stack.effective().voice.realtime_route.route == "gemini"


def test_a_wrong_pin_writes_nothing(served, stack, tmp_path):
    resp = served.request("POST", "/api/settings/batch", {"changes": TO_GEMINI, "pin": "000000"})
    assert resp.status == 401 and resp.json["error"] == "bad_pin"
    assert stack.effective().voice.realtime_route.route == "platform"
    assert not (tmp_path / FILE).exists()


def test_a_malformed_body_costs_no_pin_attempt(served, monkeypatch):
    checks = []
    monkeypatch.setattr(WebApi, "_check_pin", lambda self, req, pin: checks.append(pin))
    for changes in ([], [{"path": MODE}], "x"):
        resp = served.request("POST", "/api/settings/batch", {"changes": changes, "pin": PIN})
        assert resp.status == 422
    assert served.request("POST", "/api/settings/batch", {"pin": PIN}).status == 400
    assert checks == []


def test_the_api_maps_batch_refusals_to_codes(served):
    def post(url, changes):
        return served.request("POST", url, {"changes": changes, "pin": PIN})

    loosen = [{"path": "webui.name_max_chars", "value": 100}]
    resp = post("/api/settings/batch", loosen)
    assert resp.status == 422 and resp.json["error"] == "batch_loosens_safety"
    resp = post("/api/settings/batch", [{"path": MODE, "value": "nope"}])
    assert resp.status == 422 and resp.json == {
        "error": "value_choice",
        "field": "value",
        "path": MODE,
    }
    resp = post("/api/settings/batch", TO_GEMINI[1:])
    assert resp.status == 422 and resp.json["error"] == "setting_invalid"
    resp = post("/api/settings/batch", [{"path": MOVE, "value": "https://moved.example/s"}])
    assert resp.status == 422 and resp.json == {
        "error": "batch_endpoint_move",
        "field": "changes",
        "path": MOVE,
    }
    assert served.request(
        "POST", "/api/settings/batch", {"changes": TO_GEMINI, "pin": PIN, "ack_endpoint": "x"}
    ).json == {"error": "unknown_field", "field": "ack_endpoint"}


def test_an_entry_that_is_not_allowed_maps_to_403_and_writes_nothing(served, stack, tmp_path):
    changes = [{"path": "affect.update_hz", "value": 25.0}, {"path": "nope.x", "value": 1}]
    resp = served.request("POST", "/api/settings/batch", {"changes": changes, "pin": PIN})
    assert resp.status == 403 and resp.json == {
        "error": "setting_not_allowed",
        "field": "path",
        "path": "nope.x",
    }
    assert stack.effective().affect.update_hz == 20.0 and not (tmp_path / FILE).exists()


def test_a_guardrail_refusal_names_the_entry_and_costs_no_pin_attempt(served, stack):
    path = "reflexes.rules.withdraw-close.amplitude_deg"
    wider = get_path(stack.effective(), path) + 5
    changes = [{"path": "affect.update_hz", "value": 25.0}, {"path": path, "value": wider}]
    for _ in range(4):  # more than webui.pin_max_attempts
        resp = served.request("POST", "/api/settings/batch", {"changes": changes, "pin": PIN})
        assert resp.status == 403 and resp.json == {
            "error": "setting_guardrail",
            "field": "value",
            "path": path,
        }
    assert stack.effective().affect.update_hz == 20.0
    resp = served.request("POST", "/api/settings/batch", {"changes": TO_GEMINI, "pin": PIN})
    assert resp.status == 200


def test_the_preview_api_returns_the_diff(served, stack):
    resp = served.request("POST", "/api/settings/batch/preview", {"changes": TO_GEMINI, "pin": PIN})
    assert resp.status == 200
    assert [(c["path"], c["new"]) for c in resp.json["changes"]] == [
        (MODE, "direct"),
        (ROUTE, "gemini"),
    ]
    assert stack.effective().voice.realtime_route.route == "platform"


def test_the_image_switch_moves_between_two_routes_in_one_save(tmp_path):
    routes = [
        {"name": n, "base_url": "https://api.example.com/v1", "model": "m", "images": n == "a"}
        for n in ("a", "b")
    ]
    stack = ConfigStack(app={"llm_routes": routes}, audit_path=tmp_path / "audit.jsonl")
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    editor.change_many(
        [
            {"path": "llm_routes.a.images", "value": False},
            {"path": "llm_routes.b.images", "value": True},
        ]
    )
    assert [r.images for r in stack.effective().llm_routes] == [False, True]
    replayed = ConfigStack(app={"llm_routes": routes}, audit_path=tmp_path / "audit2.jsonl")
    SettingsEditor(replayed, store=tmp_path / FILE).restore()
    assert [r.images for r in replayed.effective().llm_routes] == [False, True]
