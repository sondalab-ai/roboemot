"""conversation.language is a choice in the web UI (SPEC 8 Voice pipeline, 11A): only languages
whose voice models are installed are offered, and the server refuses any other value."""

from __future__ import annotations

import json

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import ConfigStack, get_path
from roboemot.webui import languages
from roboemot.webui.languages import installed_languages
from roboemot.webui.settings import LANGUAGE_DICTS, SettingsEditor
from roboemot.webui.validation import ValidationError
from test_webui_settings import FILE, put, serve

PATH = "conversation.language"


@pytest.fixture
def stack(tmp_path):
    return ConfigStack(
        app={"webui": {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3}},
        audit_path=tmp_path / "audit.jsonl",
    )


def editor_for(tmp_path, stack, languages=("it", "en")):
    return SettingsEditor(stack, store=tmp_path / FILE, installed=lambda cfg: set(languages))


def row(editor):
    return next(f for f in editor.fields() if f["path"] == PATH)


def test_the_row_is_a_choice_of_the_installed_languages(tmp_path, stack):
    r = row(editor_for(tmp_path, stack))
    assert r["kind"] == "enum" and r["choices"] == ["en", "it"] and r["value"] == "it"
    assert r["writable"] and not r["nullable"]


def test_the_current_value_is_listed_even_without_models(tmp_path, stack):
    assert row(editor_for(tmp_path, stack, ()))["choices"] == ["it"]


def test_an_installed_language_is_accepted(tmp_path, stack):
    result = editor_for(tmp_path, stack).change(PATH, "en")
    assert result["changed"] and stack.effective().conversation.language == "en"


@pytest.mark.parametrize("value", ["xx", "EN", "en-US", "", " en", "it\n", None, 5, True, ["en"]])
def test_any_other_value_is_refused_by_the_server(tmp_path, stack, value):
    with pytest.raises(ValidationError) as caught:
        editor_for(tmp_path, stack).change(PATH, value)
    assert caught.value.code in {"value_choice", "value_type"}
    assert stack.effective().conversation.language == "it"
    assert stack.audit == []


def test_a_language_without_installed_models_is_refused(tmp_path, stack):
    editor = editor_for(tmp_path, stack, ("it",))
    with pytest.raises(ValidationError) as caught:
        editor.change(PATH, "en")
    assert caught.value.code == "value_choice"


def test_the_api_refuses_it_too(tmp_path, stack):
    srv, client = serve(tmp_path, editor_for(tmp_path, stack), stack)
    try:
        resp = put(client, PATH, "klingon")
        assert resp.status == 422 and resp.json == {
            "error": "value_choice",
            "field": "value",
        }
        assert b"klingon" not in resp.body
        assert put(client, PATH, "en").status == 200
        rows = {f["path"]: f for f in client.request("GET", "/api/settings").json["fields"]}
        assert rows[PATH]["kind"] == "enum" and "en" in rows[PATH]["choices"]
    finally:
        srv.stop()


def test_a_stored_language_that_is_no_longer_installed_is_dropped_at_start(tmp_path, stack):
    def store():
        (tmp_path / FILE).write_text(json.dumps({"v": 1, "overrides": {PATH: "en"}}))
        (tmp_path / FILE).chmod(0o600)

    store()
    editor_for(tmp_path, stack, ("it",)).restore()
    assert stack.effective().conversation.language == "it"
    store()
    editor_for(tmp_path, stack, ("it", "en")).restore()
    assert stack.effective().conversation.language == "en"


# ------------------------------------------------------------------ which languages are installed


@pytest.fixture
def status(monkeypatch):
    """Model status by manifest model name; unlisted names are missing."""
    table: dict[str, str] = {}
    monkeypatch.setattr(
        languages, "model_status", lambda spec, root: table.get(spec.name, "missing")
    )
    return table


ALL_OK = {
    "vosk-small-it": "ok",
    "piper-it-paola": "ok",
    "vosk-small-en": "ok",
    "piper-en-amy": "ok",
}


def test_a_language_needs_its_asr_and_tts_models(status):
    cfg = RoboEmotConfig()
    status.update(ALL_OK)
    assert installed_languages(cfg) == {"it", "en"}
    status["piper-en-amy"] = "missing"
    assert installed_languages(cfg) == {"it"}
    status.update({"piper-en-amy": "ok", "vosk-small-en": "corrupt"})
    assert installed_languages(cfg) == {"it"}


def test_a_stale_ort_model_still_counts(status):
    status.update({**ALL_OK, "vosk-small-it": "stale-ort"})
    assert installed_languages(RoboEmotConfig()) == {"it", "en"}


def test_the_configuration_must_list_the_language_for_listening_and_speaking(status):
    status.update(ALL_OK)
    only_it = RoboEmotConfig(voice={"asr": {"models": {"it": "vosk-small-it"}}})
    assert installed_languages(only_it) == {"it"}
    no_voice = RoboEmotConfig(voice={"voices": {"it": "it_IT-paola-medium"}})
    assert installed_languages(no_voice) == {"it"}


def test_a_language_outside_the_manifest_is_never_offered(status):
    status.update(ALL_OK)
    cfg = RoboEmotConfig(
        voice={
            "asr": {"models": {"it": "vosk-small-it", "fr": "x"}},
            "voices": {"it": "a", "fr": "b"},
        }
    )
    assert installed_languages(cfg) == {"it"}


def test_nothing_is_installed_on_a_machine_without_models(tmp_path):
    cfg = RoboEmotConfig(data_dir=str(tmp_path / "empty"))
    assert installed_languages(cfg) == frozenset()


# ------------------------------------------------------------------ other language-keyed maps


@pytest.mark.parametrize("path", sorted(LANGUAGE_DICTS))
def test_a_new_language_key_must_be_one_the_voice_stack_supports(tmp_path, stack, path):
    editor = editor_for(tmp_path, stack)
    current = dict(get_path(stack.effective(), path))
    with pytest.raises(ValidationError) as caught:
        editor.change(path, {**current, "xx": "text"})
    assert caught.value.code == "language_unsupported"
    assert "xx" not in get_path(stack.effective(), path)
    assert not editor.change(path, current)["changed"]  # the keys it already has stay


def test_the_language_dicts_are_all_string_dicts_keyed_by_language(tmp_path, stack):
    rows = {f["path"]: f for f in editor_for(tmp_path, stack).fields()}
    assert {p: rows[p]["kind"] for p in LANGUAGE_DICTS} == dict.fromkeys(
        LANGUAGE_DICTS, "string_dict"
    )
