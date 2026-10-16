"""Web UI API against a fake enrollment service on an ephemeral localhost port (SPEC 6, 11A, 12)."""

from __future__ import annotations

import http.client
import json
import stat
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics, FnSource
from roboemot.registry.records import Consent, ConsentError, RecordSummary
from roboemot.webui import make_server
from roboemot.webui.api import holds_vector
from roboemot.webui.auth import PinError
from roboemot.webui.strings import STRINGS, language_for

PIN = "246810"
OLD = datetime(2026, 1, 1, tzinfo=UTC)


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.rows = [
            RecordSummary("p1", "Anna", "active", "person", OLD, "adult"),
            RecordSummary("a1", "Micio", "active", "animal", None),
            RecordSummary("a2", "Micio 2", "active", "animal", None),
            RecordSummary("o1", "Tazza", "active", "object", None),
        ]
        self.keys = ["badkey"]
        self.state = "collecting"
        self.can_complete = False
        self.fail_forget: Exception | None = None
        self.known_ids = {"p1", "a1", "a2", "o1"}

    def start(self, session_id):
        self.calls.append(("start", len(session_id)))

    def status(self):
        # A hostile or sloppy service: extra fields must not reach the browser.
        return {
            "session_id": "secret-session",
            "state": self.state,
            "reason": "/home/pollen/.local/share/roboemot",
            "collected": 3,
            "target": 8,
            "minimum": 5,
            "can_complete": self.can_complete,
            "pose_varied": False,
            "hint": "turn the head slightly left and right",
            "face_visible": False,
            "rejected": {"score": 1},
            "crop": [0.1] * 128,
            "image": b"\xff\xd8",
        }

    def complete(self, name, consent, declared_age_group, declared_gender):
        if declared_age_group == "child" and consent.guardian == "Refused":
            raise ConsentError("registry says no")
        self.calls.append(("complete", name, consent, declared_age_group, declared_gender))

    def cancel(self):
        self.calls.append(("cancel",))

    def list_records(self):
        return list(self.rows)

    def rename(self, record_id, display_name):
        if display_name == "Boom":
            raise ValueError("internal secret detail")
        self.calls.append(("rename", record_id, display_name))

    def merge(self, keep_id, drop_id):
        self.calls.append(("merge", keep_id, drop_id))

    def delete(self, record_id):
        if self.fail_forget:
            raise self.fail_forget
        if record_id not in self.known_ids and record_id not in self.keys:
            raise KeyError(record_id)
        self.calls.append(("delete", record_id))
        if record_id in self.keys:
            self.keys.remove(record_id)

    def quarantined(self):
        return list(self.keys)


class Client:
    def __init__(self, port: int) -> None:
        self.port = port
        self.cookie: str | None = None
        self.csrf: str | None = None

    def request(self, method, path, body=None, *, csrf=True, raw=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdrs = dict(headers or {})
        if self.cookie:
            hdrs["Cookie"] = self.cookie
        if csrf and self.csrf and method != "GET":
            hdrs["X-CSRF-Token"] = self.csrf
        payload = raw
        if body is not None:
            payload = json.dumps(body).encode()
        if payload is not None:
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        out = SimpleNamespace(status=resp.status, headers=resp.headers, body=data)
        out.json = json.loads(data) if "json" in resp.getheader("Content-Type", "") else None
        conn.close()
        return out

    def login(self, pin=PIN):
        resp = self.request("POST", "/api/login", {"pin": pin})
        if resp.status == 200:
            self.cookie = resp.headers["Set-Cookie"].split(";")[0]
            self.csrf = resp.json["csrf"]
        return resp


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def service():
    return FakeService()


@pytest.fixture
def server(tmp_path, service, clock):
    cfg = RoboEmotConfig(
        webui={"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3, "pin_lockout_s": 60}
    )
    diagnostics = Diagnostics().add(FnSource("demo", lambda: {"fps": 12.5, "box": [1.0, 2.0]}))
    srv = make_server(service, diagnostics, cfg, tmp_path, clock=clock)
    srv.set_pin(PIN)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def client(server):
    return Client(server.port)


@pytest.fixture
def authed(client):
    assert client.login().status == 200
    return client


FORM = {
    "display_name": "Anna",
    "consent_who": "Anna",
    "scope": ["face", "refresh"],
    "age_group": "adult",
    "gender": "female",
    "agreed": True,
}


# ------------------------------------------------------------------ auth


PROTECTED = [
    ("GET", "/api/enrollment/status"),
    ("POST", "/api/enrollment/start"),
    ("POST", "/api/enrollment/complete"),
    ("POST", "/api/enrollment/cancel"),
    ("GET", "/api/registry"),
    ("GET", "/api/registry/quarantined"),
    ("POST", "/api/registry/p1/rename"),
    ("POST", "/api/registry/p1/forget"),
    ("POST", "/api/registry/a1/delete"),
    ("POST", "/api/registry/merge"),
    ("POST", "/api/registry/quarantined/purge"),
    ("GET", "/api/debug"),
    ("GET", "/api/catalogue"),
    ("POST", "/api/logout"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_protected_routes_need_login(client, method, path):
    resp = client.request(method, path, {} if method == "POST" else None)
    assert resp.status == 401
    assert resp.json == {"error": "unauthorized"}


def test_open_routes_work_without_login(client):
    assert client.request("GET", "/").status == 200
    assert client.request("GET", "/static/app.js").status == 200
    assert client.request("GET", "/static/style.css").status == 200
    assert client.request("GET", "/api/strings").status == 200
    session = client.request("GET", "/api/session").json
    assert session["authenticated"] is False
    assert "csrf" not in session


def test_login_sets_strict_httponly_cookie(client):
    resp = client.login()
    cookie = resp.headers["Set-Cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert PIN not in cookie
    assert client.request("GET", "/api/session").json["authenticated"] is True


def test_wrong_logins_from_the_host_itself_are_slowed_but_not_lost(client, clock):
    for _ in range(5):
        assert client.login("000000").status == 401
        clock.now += 100
    assert client.login(PIN).status == 200


def test_success_resets_failure_count(client, clock):
    assert client.login("000000").status == 401
    clock.now += 100
    assert client.login("000000").status == 401
    clock.now += 100
    assert client.login(PIN).status == 200
    clock.now += 100
    assert client.login("000000").status == 401


def test_session_expires_when_idle(authed, clock):
    clock.now += 901
    assert authed.request("GET", "/api/registry").status == 401


def test_session_has_an_absolute_lifetime(authed, clock):
    for _ in range(5):  # active the whole time, so never idle
        clock.now += 800
        if authed.request("GET", "/api/registry").status == 401:
            break
    assert authed.request("GET", "/api/registry").status == 401


def test_session_cookie_has_no_max_age(client):
    cookie = client.login().headers["Set-Cookie"]
    assert "Max-Age" not in cookie and "Expires" not in cookie
    out = client.request("POST", "/api/logout", {})
    assert "Max-Age=0" in out.headers["Set-Cookie"]


def test_new_pin_ends_existing_sessions(authed, server):
    assert authed.request("GET", "/api/registry").status == 200
    server.set_pin("135790")
    assert authed.request("GET", "/api/registry").status == 401


def test_logout_ends_session(authed):
    assert authed.request("POST", "/api/logout", {}).status == 200
    assert authed.request("GET", "/api/registry").status == 401


def test_pin_hashed_with_salt_and_private(server, tmp_path):
    path = tmp_path / "webui_pin.json"
    text = path.read_text()
    assert PIN not in text
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    before = json.loads(text)["salt"]
    server.set_pin(PIN)
    assert json.loads(path.read_text())["salt"] != before


def test_pin_format_rules(server):
    with pytest.raises(PinError):
        server.set_pin("12")
    with pytest.raises(PinError):
        server.set_pin("abcdefgh")


def test_there_is_no_web_setup_route(tmp_path, service):
    cfg = RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0})
    srv = make_server(service, Diagnostics(), cfg, tmp_path).start()
    try:
        c = Client(srv.port)
        assert c.request("GET", "/api/session").json["pin_set"] is False
        assert c.login().json == {"error": "pin_not_set"}
        assert c.request("POST", "/api/setup", {"pin": PIN}).status in (401, 404)
        assert not (tmp_path / "webui_pin.json").exists()
        assert "no_pin" in c.request("GET", "/api/strings").json["strings"]
    finally:
        srv.stop()


# ------------------------------------------------------------------ CSRF, origin, body


def test_csrf_required_on_state_change(authed):
    resp = authed.request("POST", "/api/enrollment/cancel", {}, csrf=False)
    assert resp.status == 403
    assert resp.json == {"error": "csrf"}
    bad = authed.request(
        "POST", "/api/enrollment/cancel", {}, csrf=False, headers={"X-CSRF-Token": "nope"}
    )
    assert bad.status == 403


def test_csrf_ok_with_token(authed, service):
    assert authed.request("POST", "/api/enrollment/cancel", {}).status == 200
    assert ("cancel",) in service.calls


def test_cross_origin_post_refused(authed):
    resp = authed.request("POST", "/api/enrollment/cancel", {}, headers={"Origin": "http://evil.x"})
    assert resp.status == 403
    assert resp.json == {"error": "origin"}
    own = f"http://127.0.0.1:{authed.port}"
    assert (
        authed.request("POST", "/api/enrollment/cancel", {}, headers={"Origin": own}).status == 200
    )


def test_body_size_limit(authed):
    resp = authed.request("POST", "/api/enrollment/complete", raw=b"{" + b" " * 20000 + b"}")
    assert resp.status == 413


def test_json_only(authed):
    resp = authed.request(
        "POST",
        "/api/registry/merge",
        raw=b"keep_id=a1",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status == 415
    assert authed.request("POST", "/api/registry/merge", raw=b"[1]").status == 400
    assert authed.request("POST", "/api/registry/merge", raw=b"{nope").status == 400
    assert authed.request("POST", "/api/registry/merge", raw=b'{"keep_id": NaN}').status == 400


def test_chunked_body_refused(authed):
    resp = authed.request(
        "POST", "/api/enrollment/cancel", raw=b"0\r\n\r\n", headers={"Transfer-Encoding": "chunked"}
    )
    assert resp.status == 411


def test_methods_other_than_get_post_refused(authed):
    assert authed.request("PUT", "/api/registry", {}).status == 405
    assert authed.request("DELETE", "/api/registry/a1").status == 405


# ------------------------------------------------------------------ static and headers


@pytest.mark.parametrize(
    "path",
    [
        "/static/../server.py",
        "/static/%2e%2e/server.py",
        "/../../etc/passwd",
        "/static/..%2fserver.py",
        "/static/app.js/../../auth.py",
        "/webui_pin.json",
        "/static/",
        "/%2e%2e/%2e%2e/etc/passwd",
    ],
)
def test_path_traversal_blocked(client, path):
    resp = client.request("GET", path)
    assert resp.status == 404
    assert b"scrypt" not in resp.body
    assert b"root:" not in resp.body


def test_security_headers_everywhere(client, authed):
    for resp in [
        client.request("GET", "/"),
        client.request("GET", "/static/app.js"),
        client.request("GET", "/api/strings"),
        client.request("GET", "/api/registry"),
        client.request("GET", "/nope"),
        client.request("POST", "/api/login", {"pin": "1"}),
    ]:
        csp = resp.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "unsafe-inline" not in csp
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["Referrer-Policy"] == "no-referrer"


def test_page_has_no_inline_script_and_no_cdn(client):
    html = client.request("GET", "/").body.decode()
    assert '<script src="/static/app.js">' in html
    assert "onclick=" not in html
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "https://" not in html
    js = client.request("GET", "/static/app.js").body.decode()
    assert "innerHTML" not in js
    assert "eval(" not in js


# ------------------------------------------------------------------ enrollment


def test_enrollment_status_is_whitelisted(authed):
    assert authed.request("POST", "/api/enrollment/start", {}).status == 200
    resp = authed.request("GET", "/api/enrollment/status")
    assert resp.json == {
        "state": "collecting",
        "frames_collected": 3,
        "frames_needed": 8,
        "can_complete": False,
        "hints": ["no_face", "turn_head"],
    }
    for leaked in (b"image", b"crop", b"secret-session", b".local/share", b"rejected"):
        assert leaked not in resp.body


def test_complete_builds_server_side_consent(authed, service):
    before = datetime.now(UTC)
    assert authed.request("POST", "/api/enrollment/complete", FORM).status == 200
    _, name, consent, age, gender = service.calls[-1]
    assert (name, age, gender) == ("Anna", "adult", "female")
    assert isinstance(consent, Consent)
    assert consent.who == "Anna"
    assert consent.scope == ("face", "refresh")
    assert consent.guardian is None
    assert before <= consent.when <= datetime.now(UTC)


def test_client_cannot_set_consent_time(authed):
    resp = authed.request("POST", "/api/enrollment/complete", FORM | {"when": "2000-01-01"})
    assert resp.status == 422
    assert resp.json["error"] == "unknown_field"


@pytest.mark.parametrize("age", ["child", "teen", "unknown", None])
def test_guardian_required_unless_adult_or_senior(authed, service, age):
    form = FORM | {"age_group": age}
    resp = authed.request("POST", "/api/enrollment/complete", form)
    assert resp.status == 422
    assert resp.json == {"error": "guardian_required", "field": "guardian"}
    assert not service.calls


def test_guardian_accepted_for_minor(authed, service):
    form = {k: v for k, v in FORM.items() if k != "consent_who"}
    form |= {"age_group": "child", "guardian": "Maria Rossi"}
    assert authed.request("POST", "/api/enrollment/complete", form).status == 200
    consent = service.calls[-1][2]
    assert (consent.guardian, consent.who) == ("Maria Rossi", "Maria Rossi")


def test_minor_consent_must_come_from_the_guardian(authed, service):
    base = FORM | {"age_group": "teen", "guardian": "Maria Rossi"}
    mismatch = authed.request("POST", "/api/enrollment/complete", base | {"consent_who": "Anna"})
    assert mismatch.status == 422
    assert mismatch.json["error"] == "consent_who_guardian"
    same = authed.request("POST", "/api/enrollment/complete", base | {"consent_who": "maria rossi"})
    assert same.status == 200
    assert service.calls[-1][2].who == "Maria Rossi" or service.calls[-1][2].who == "maria rossi"


def test_guardian_cannot_be_the_person_enrolled(authed, service):
    form = FORM | {"age_group": "child", "guardian": "anna", "consent_who": "anna"}
    resp = authed.request("POST", "/api/enrollment/complete", form)
    assert resp.status == 422
    assert resp.json["error"] == "guardian_is_subject"
    assert not service.calls


def test_senior_needs_no_guardian(authed):
    assert (
        authed.request("POST", "/api/enrollment/complete", FORM | {"age_group": "senior"}).status
        == 200
    )


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"agreed": False}, "consent_not_agreed"),
        ({"agreed": "yes"}, "consent_not_agreed"),
        ({"scope": []}, "scope_invalid"),
        ({"scope": ["face", "dna"]}, "scope_invalid"),
        ({"scope": "face"}, "scope_invalid"),
        ({"display_name": ""}, "name_required"),
        ({"display_name": "   "}, "name_required"),
        ({"display_name": "x" * 65}, "name_too_long"),
        ({"display_name": "<b>Anna</b>"}, "name_invalid"),
        ({"display_name": "Anna\nRoss"}, "name_invalid"),
        ({"display_name": 7}, "name_invalid"),
        ({"age_group": "ancient"}, "choice_invalid"),
        ({"gender": "robot"}, "choice_invalid"),
    ],
)
def test_form_validation(authed, service, patch, code):
    resp = authed.request("POST", "/api/enrollment/complete", FORM | patch)
    assert resp.status == 422
    assert resp.json["error"] == code
    assert not service.calls


def test_unicode_names_accepted(authed):
    form = FORM | {"display_name": "Zoë d'Angelo-Rossi", "consent_who": "Zoë d'Angelo-Rossi"}
    assert authed.request("POST", "/api/enrollment/complete", form).status == 200


def test_registry_remains_the_authority_on_consent(authed):
    form = {k: v for k, v in FORM.items() if k != "consent_who"}
    form |= {"age_group": "child", "guardian": "Refused"}
    resp = authed.request("POST", "/api/enrollment/complete", form)
    assert resp.status == 422
    assert resp.json["error"] == "consent_refused"


# ------------------------------------------------------------------ registry


def test_registry_list_has_summaries_only(authed):
    rows = authed.request("GET", "/api/registry").json["records"]
    assert [r["id"] for r in rows] == ["p1", "a1", "a2", "o1"]
    assert set(rows[0]) == {"id", "display_name", "status", "kind", "last_seen", "age_group"}
    assert rows[0]["last_seen"].startswith("2026-01-01")


def test_rename(authed, service):
    resp = authed.request("POST", "/api/registry/a1/rename", {"display_name": "Gatto"})
    assert resp.status == 200
    assert service.calls[-1] == ("rename", "a1", "Gatto")
    assert authed.request("POST", "/api/registry/a1/rename", {"display_name": "<x>"}).status == 422
    assert (
        authed.request("POST", "/api/registry/zzz/rename", {"display_name": "Gatto"}).status == 404
    )
    assert authed.request("POST", "/api/registry/a%2f1/rename", {"display_name": "G"}).status in (
        404,
        422,
    )


def test_merge_entities_only(authed, service):
    ok = authed.request("POST", "/api/registry/merge", {"keep_id": "a1", "drop_id": "a2"})
    assert ok.status == 200
    assert service.calls[-1] == ("merge", "a1", "a2")
    person = authed.request("POST", "/api/registry/merge", {"keep_id": "a1", "drop_id": "p1"})
    assert person.status == 422
    assert person.json["error"] == "wrong_kind"
    same = authed.request("POST", "/api/registry/merge", {"keep_id": "a1", "drop_id": "a1"})
    assert same.status == 422


def test_forget_needs_confirm_and_is_synchronous(authed, service):
    first = authed.request("POST", "/api/registry/p1/forget", {})
    assert first.status == 409
    assert first.json == {"error": "confirm_required"}
    assert not any(c[0] == "delete" for c in service.calls)
    assert authed.request("POST", "/api/registry/p1/forget", {"confirm": "true"}).status == 409
    done = authed.request("POST", "/api/registry/p1/forget", {"confirm": True, "pin": PIN})
    assert done.status == 200
    assert service.calls[-1] == ("delete", "p1")


def test_forget_and_purge_ask_for_the_pin_again(authed, service, clock):
    forget = "/api/registry/p1/forget"
    assert authed.request("POST", forget, {"confirm": True}).status == 400
    assert authed.request("POST", forget, {"confirm": True, "pin": "000000"}).status == 401
    purge = "/api/registry/quarantined/purge"
    assert authed.request("POST", purge, {"key": "badkey", "confirm": True}).status == 400
    clock.now += 100
    wrong = authed.request("POST", purge, {"key": "badkey", "confirm": True, "pin": "000000"})
    assert wrong.status == 401
    assert not any(c[0] == "delete" for c in service.calls)


def test_forget_of_an_unknown_id_is_not_found(authed, service):
    service.known_ids.discard("p1")
    resp = authed.request("POST", "/api/registry/p1/forget", {"confirm": True, "pin": PIN})
    assert resp.status == 404


def test_forget_failures_of_any_kind_are_incomplete(authed, service):
    service.fail_forget = RuntimeError("secret detail")
    resp = authed.request("POST", "/api/registry/p1/forget", {"confirm": True, "pin": PIN})
    assert resp.json == {"error": "forget_incomplete"}
    assert b"secret" not in resp.body


def test_forget_only_for_people_and_delete_only_for_entities(authed):
    assert authed.request("POST", "/api/registry/a1/forget", {"confirm": True}).status == 422
    assert authed.request("POST", "/api/registry/p1/delete", {"confirm": True}).status == 422


def test_delete_entity_needs_confirm(authed, service):
    assert authed.request("POST", "/api/registry/a1/delete", {}).status == 409
    assert authed.request("POST", "/api/registry/a1/delete", {"confirm": True}).status == 200
    assert service.calls[-1] == ("delete", "a1")


def test_incomplete_forget_is_reported(authed, service):
    service.fail_forget = ExceptionGroup("forget incomplete", [OSError("disk")])
    resp = authed.request("POST", "/api/registry/p1/forget", {"confirm": True, "pin": PIN})
    assert resp.status == 500
    assert resp.json == {"error": "forget_incomplete"}


def test_quarantined_list_and_purge(authed, service):
    assert authed.request("GET", "/api/registry/quarantined").json == {"keys": ["badkey"]}
    assert (
        authed.request("POST", "/api/registry/quarantined/purge", {"key": "badkey"}).status == 409
    )
    other = authed.request(
        "POST", "/api/registry/quarantined/purge", {"key": "p1", "confirm": True, "pin": PIN}
    )
    assert other.status == 404
    done = authed.request(
        "POST", "/api/registry/quarantined/purge", {"key": "badkey", "confirm": True, "pin": PIN}
    )
    assert done.status == 200
    assert service.keys == []


def test_catalogue_is_a_placeholder(authed):
    resp = authed.request("GET", "/api/catalogue")
    assert resp.status == 501
    assert resp.json == {"error": "not_implemented"}


# ------------------------------------------------------------------ debug and privacy


def test_debug_report_is_served(authed):
    resp = authed.request("GET", "/api/debug")
    assert resp.status == 200
    assert resp.json["sections"]["demo"]["data"]["fps"] == 12.5


def test_a_section_with_a_vector_is_withheld_not_the_whole_report(tmp_path, service):
    leaky = Diagnostics().add(
        FnSource("emb", lambda: {"v": [0.01 * i for i in range(128)]}),
        FnSource("ok", lambda: {"fps": 3}),
    )
    cfg = RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0})
    srv = make_server(service, leaky, cfg, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        c = Client(srv.port)
        c.login()
        resp = c.request("GET", "/api/debug")
        assert resp.status == 200
        assert resp.json["sections"]["emb"] == {
            "scope": "runtime",
            "ok": False,
            "error": "withheld",
        }
        assert resp.json["sections"]["ok"]["data"] == {"fps": 3}
        assert b"0.01" not in resp.body
    finally:
        srv.stop()


def test_service_error_text_is_never_echoed(authed):
    resp = authed.request("POST", "/api/registry/a1/rename", {"display_name": "Boom"})
    assert resp.status == 422
    assert resp.json == {"error": "bad_request"}
    assert b"secret" not in resp.body


def test_every_normal_response_is_free_of_float_arrays(authed):
    authed.request("POST", "/api/enrollment/start", {})
    for path in [
        "/api/enrollment/status",
        "/api/registry",
        "/api/registry/quarantined",
        "/api/debug",
        "/api/session",
        "/api/strings",
    ]:
        body = authed.request("GET", path).json
        assert not holds_vector(body), path


def test_holds_vector_detects_nested_lists():
    assert holds_vector({"a": [{"b": [0.5] * 40}]})
    assert holds_vector([[1] * 40])
    assert not holds_vector({"box": [0.1, 0.2, 0.3, 0.4], "names": ["x"] * 100})


# ------------------------------------------------------------------ i18n


def test_italian_and_english_cover_the_same_keys():
    assert set(STRINGS["it"]) == set(STRINGS["en"])
    assert all(STRINGS["it"][k].strip() for k in STRINGS["it"])
    assert STRINGS["it"]["login"] != STRINGS["en"]["login"]


def test_error_codes_have_strings():
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "roboemot" / "webui"
    codes = set()
    for name in ("api.py", "validation.py", "auth.py"):
        codes |= set(
            re.findall(
                r'(?:ApiError\(\s*HTTPStatus\.\w+,\s*|ValidationError\()"(\w+)"',
                (src / name).read_text(),
            )
        )
    codes |= {"pin_digits", "pin_too_short"}
    missing = [c for c in sorted(codes) if f"err_{c}" not in STRINGS["en"]]
    assert not missing


@pytest.mark.parametrize(
    ("configured", "lang"), [("it", "it"), ("en", "en"), ("it-IT", "it"), ("fr", "en")]
)
def test_language_follows_config(configured, lang):
    assert language_for(configured) == lang


def test_strings_endpoint_follows_conversation_language(tmp_path, service):
    for configured, expected in (("it", "Accedi"), ("en", "Log in")):
        cfg = RoboEmotConfig(
            webui={"bind": "127.0.0.1", "port": 0}, conversation={"language": configured}
        )
        srv = make_server(service, Diagnostics(), cfg, tmp_path).start()
        try:
            data = Client(srv.port).request("GET", "/api/strings").json
            assert data["language"] == configured
            assert data["strings"]["login"] == expected
        finally:
            srv.stop()


def test_consent_text_names_cloud_handling(tmp_path, service):
    texts = {}
    for pseudo in (False, True):
        cfg = RoboEmotConfig(
            webui={"bind": "127.0.0.1", "port": 0},
            privacy={"pseudonyms_to_cloud": pseudo},
            conversation={"language": "en"},
        )
        srv = make_server(service, Diagnostics(), cfg, tmp_path).start()
        try:
            s = Client(srv.port).request("GET", "/api/strings").json["strings"]
            texts[pseudo] = s["consent_cloud"]
            assert "GDPR" in STRINGS["en"]["consent_biometric"]
        finally:
            srv.stop()
    assert "pseudonyms instead" in texts[True]
    assert "real" in texts[False]
