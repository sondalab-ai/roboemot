"""GET /api/strings is serialised once per language and consent mode (SPEC 11A, 12)."""

from __future__ import annotations

import json

import pytest
from roboemot.config.stack import ConfigStack
from roboemot.webui.api import Request, WebApi
from roboemot.webui.auth import PinStore
from roboemot.webui.strings import EN, IT
from test_webui import FakeService

HOST = {"host": "127.0.0.1:8042"}


@pytest.fixture
def stack(tmp_path):
    return ConfigStack(app={"conversation": {"language": "en"}}, audit_path=tmp_path / "a.jsonl")


@pytest.fixture
def api(tmp_path, stack):
    return WebApi(FakeService(), None, stack.effective, PinStore(tmp_path))


def strings(api):
    resp = api.handle(Request("GET", "/api/strings", HOST, b"", "127.0.0.1"))
    assert resp.status == 200
    assert resp.body == json.dumps(resp.data, allow_nan=False).encode()
    return resp


def test_a_second_request_reuses_the_body(api):
    assert strings(api).body is strings(api).body


def test_a_language_change_gives_the_new_strings(api, stack):
    first = strings(api).data
    assert first["language"] == "en" and first["strings"]["consent_cloud"] in EN.values()
    stack.set("conversation.language", "it", source="t")
    later = strings(api).data
    assert later["language"] == "it"
    assert later["strings"]["consent_cloud"] == IT["consent_cloud_names"]
    assert later["strings"]["consent_cloud"] != first["strings"]["consent_cloud"]


def test_the_consent_mode_picks_the_consent_text(api, stack):
    assert strings(api).data["strings"]["consent_cloud"] == EN["consent_cloud_names"]
    stack.set("privacy.pseudonyms_to_cloud", True, source="t", admin=True)
    assert strings(api).data["strings"]["consent_cloud"] == EN["consent_cloud_pseudonyms"]
    stack.clear("privacy.pseudonyms_to_cloud", admin=True)
    assert strings(api).data["strings"]["consent_cloud"] == EN["consent_cloud_names"]


def test_the_shared_table_is_never_changed(api):
    before = dict(EN)
    strings(api)
    assert before == EN and "consent_cloud" not in EN
