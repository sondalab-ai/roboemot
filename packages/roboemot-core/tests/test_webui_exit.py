"""Exit check of the settings mirror (SPEC 11A, 12): an edit through the API is heard on the next
utterance, a restart field says so, and no response holds a secret.

The volume check is unit level (no simulator harness covers speech): a real ``SpeechOutput`` with
a fake backend. The subscription copies the app's own line (the `reload` closure in
`RoboEmotApp.run` sets `speech_output.config = change.config`); that closure is inline in a long
method and cannot be called alone, so the line is mirrored here. The only link between the page
and the voice is the configuration.
"""

from __future__ import annotations

import pytest
from roboemot.config.stack import ConfigStack
from roboemot.voice.speech import SpeechOutput
from roboemot.webui.settings import SettingsEditor
from test_speech import FakeBackend, FakeSink, say, wait_for
from test_webui_settings import FILE, put, serve

VOLUME = "voice.base.volume_db"
RESTART_FIELD = "vision.threads"


@pytest.fixture
def stack(tmp_path):
    return ConfigStack(
        app={"webui": {"bind": "127.0.0.1", "port": 0}}, audit_path=tmp_path / "audit.jsonl"
    )


@pytest.fixture
def web(tmp_path, stack):
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    yield client
    srv.stop()


def test_a_volume_edited_through_the_api_takes_effect_on_the_next_utterance(web, stack):
    backend = FakeBackend()
    out = SpeechOutput(stack.effective(), FakeSink(), backend)
    stack.subscribe(lambda change: setattr(out, "config", change.config))
    out.start()
    try:
        say(out)
        assert wait_for(lambda: out.sentences_spoken == 1)
        assert backend.deliveries[-1].volume_db == 0.0
        assert put(web, VOLUME, -6.0).status == 200
        say(out)
        assert wait_for(lambda: out.sentences_spoken == 2)
        assert backend.deliveries[-1].volume_db == -6.0
    finally:
        out.close()


def test_a_restart_field_is_flagged_in_the_list_and_in_the_write_response(web):
    rows = {r["path"]: r for r in web.request("GET", "/api/settings").json["fields"]}
    assert rows[RESTART_FIELD]["restart"] is True and rows[RESTART_FIELD]["writable"] is True
    assert rows[VOLUME]["restart"] is False
    resp = put(web, RESTART_FIELD, rows[RESTART_FIELD]["value"] + 1)
    assert resp.status == 200 and resp.json["restart_required"] is True
    assert put(web, VOLUME, -3.0).json["restart_required"] is False
