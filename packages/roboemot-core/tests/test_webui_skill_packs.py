"""The web UI's Skills section API (SPEC 11A, 12; design 2026-11-24-skill-packs 5.2, 5.6)."""

from __future__ import annotations

import base64
import io
import zipfile
from types import SimpleNamespace

import pytest
from roboemot.config.stack import ConfigStack
from roboemot.diagnostics import Diagnostics
from roboemot.skill_packs.admin import SkillPacksAdmin
from roboemot.skill_packs.library import SkillPackLibrary
from roboemot.skill_packs.session import SkillPackSession
from roboemot.webui import make_server
from roboemot.webui.settings import SettingsEditor
from test_webui import PIN, Client, FakeService

BASE = "/api/tools/skill_packs"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def skill(name: str = "quiz", body: str = "QUIZ-BODY") -> bytes:
    return f"---\nname: {name}\ndescription: A quiz.\n---\n{body}\n".encode()


@pytest.fixture
def served(tmp_path):
    stack = ConfigStack(
        app={"data_dir": str(tmp_path / "data"), "webui": {"bind": "127.0.0.1", "port": 0}},
        audit_path=tmp_path / "audit.jsonl",
    )
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    library = SkillPackLibrary(tmp_path / "packs", stack.effective)
    session = SkillPackSession(library, stack.effective, lambda: 0.0)
    stack.subscribe(lambda change: session.on_config(change.config))
    admin = SkillPacksAdmin(
        lambda: session,
        lambda names: editor.change_owned("skill_packs.enabled", names, "webui"),
        stack.effective,
    )
    srv = make_server(
        FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor, skill_packs=admin
    )
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    yield SimpleNamespace(client=client, stack=stack, session=session, root=tmp_path / "packs")
    srv.stop()


def upload(c, data: bytes, filename: str = "quiz.md", **kw):
    body = {"filename": filename, "data_b64": b64(data), "replace": False, "pin": PIN}
    body.update(confirm=True, **kw)
    return c.request("POST", f"{BASE}/upload", body)


def test_list_needs_a_session_and_never_carries_the_body(served):
    c = served.client
    assert c.request("GET", BASE).status == 401
    c.login()
    assert upload(c, skill()).json == {"name": "quiz"}
    listing = c.request("GET", BASE)
    assert listing.status == 200
    assert listing.json["packs"][0]["state"] == "off"
    assert "QUIZ-BODY" not in listing.body.decode()
    detail = c.request("GET", f"{BASE}/quiz")
    assert detail.status == 200 and detail.json["body"] == "QUIZ-BODY"
    assert c.request("GET", f"{BASE}/nope").json["error"] == "unknown_pack"


def test_writes_need_pin_and_confirm(served):
    c = served.client
    c.login()
    upload(c, skill())
    no_confirm = c.request("POST", f"{BASE}/quiz/enable", {"on": True, "pin": PIN})
    assert no_confirm.status == 409 and no_confirm.json["error"] == "confirm_required"
    no_csrf = c.request(
        "POST", f"{BASE}/quiz/enable", {"on": True, "pin": PIN, "confirm": True}, csrf=False
    )
    assert no_csrf.status == 403
    assert served.stack.effective().skill_packs.enabled == []
    ok = c.request("POST", f"{BASE}/quiz/enable", {"on": True, "pin": PIN, "confirm": True})
    assert ok.status == 200
    assert served.stack.effective().skill_packs.enabled == ["quiz"]
    bad_pin = c.request(
        "POST", f"{BASE}/quiz/enable", {"on": False, "pin": "0000000", "confirm": True}
    )
    assert bad_pin.status == 401
    assert served.stack.effective().skill_packs.enabled == ["quiz"], "a wrong PIN changes nothing"


def test_enable_refusal_names_the_field(served):
    c = served.client
    c.login()
    upload(c, skill())
    served.stack.set("conversation.llm_skills", ["play"], admin=True)
    resp = c.request("POST", f"{BASE}/quiz/enable", {"on": True, "pin": PIN, "confirm": True})
    assert resp.status == 409 and resp.json["error"] == "llm_skills_missing"


def test_upload_error_codes_map_to_422_and_409(served):
    c = served.client
    c.login()
    bad = upload(c, b"no front matter")
    assert bad.status == 422 and bad.json["error"] == "invalid"
    assert bad.json["reason"] == "no_front_matter"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", skill())
        zf.writestr("../evil", "x")
    assert upload(c, buf.getvalue(), "p.zip").json["error"] == "bad_path"
    assert upload(c, skill()).status == 200
    again = upload(c, skill())
    assert again.status == 409 and again.json["error"] == "exists"


def test_upload_body_limit_only_with_session(served):
    c = served.client
    big = {"filename": "q.md", "data_b64": "A" * 200_000, "replace": False, "pin": PIN}
    assert c.request("POST", f"{BASE}/upload", big).status == 413
    c.login()
    resp = c.request("POST", f"{BASE}/upload", {**big, "confirm": True})
    assert resp.status == 422  # read: the body fits the upload limit, the data is refused


def test_remove_rescan_and_end(served):
    c = served.client
    c.login()
    upload(c, skill())
    c.request("POST", f"{BASE}/quiz/enable", {"on": True, "pin": PIN, "confirm": True})
    served.session.start("quiz")
    assert c.request("GET", BASE).json["active"] == "quiz"
    ended = c.request("POST", f"{BASE}/end", {"pin": PIN})
    assert ended.status == 200 and ended.json == {"ended": True}
    removed = c.request("POST", f"{BASE}/quiz/remove", {"pin": PIN, "confirm": True})
    assert removed.status == 200
    assert served.stack.effective().skill_packs.enabled == []
    (served.root / "copied").mkdir()
    (served.root / "copied" / "SKILL.md").write_bytes(skill("copied"))
    assert c.request("POST", f"{BASE}/rescan", {"pin": PIN}).status == 200
    assert [p["name"] for p in c.request("GET", BASE).json["packs"]] == ["copied"]


def test_unavailable_without_a_port(tmp_path):
    stack = ConfigStack(
        app={"data_dir": str(tmp_path / "d"), "webui": {"bind": "127.0.0.1", "port": 0}}
    )
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        c = Client(srv.port)
        c.login()
        assert c.request("GET", BASE).status == 503
    finally:
        srv.stop()


def test_the_settings_tab_cannot_write_the_enabled_list(tmp_path):
    """Review finding 2: only the Skills section turns a skill on (after showing its text)."""
    from roboemot.webui.settings import SettingNotAllowed

    stack = ConfigStack(app={"data_dir": str(tmp_path / "d")})
    editor = SettingsEditor(stack, store=tmp_path / "s.json")
    row = next(f for f in editor.fields() if f["path"] == "skill_packs.enabled")
    assert row["writable"] is False
    with pytest.raises(SettingNotAllowed):
        editor.change("skill_packs.enabled", ["quiz"], "webui")
    editor.change_owned("skill_packs.enabled", ["quiz"], "webui")
    assert stack.effective().skill_packs.enabled == ["quiz"]


def test_a_skill_turned_on_stays_on_after_a_restart(tmp_path):
    store = tmp_path / "s.json"
    first = SettingsEditor(ConfigStack(app={"data_dir": str(tmp_path / "d")}), store=store)
    first.change_owned("skill_packs.enabled", ["quiz"], "webui")
    stack = ConfigStack(app={"data_dir": str(tmp_path / "d")})
    SettingsEditor(stack, store=store).restore()
    assert stack.effective().skill_packs.enabled == ["quiz"]
