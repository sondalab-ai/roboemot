"""Compression and validators of the web UI server (SPEC 12).

Vendored text files and large JSON bodies are sent gzipped only to a request that accepts gzip,
with ``Vary: Accept-Encoding`` on both forms; first-party files stay as they are (``no-store``,
no ETag). Every security header is present on gzipped and 304 responses alike.
"""

from __future__ import annotations

import gzip
import http.client
import json
from types import SimpleNamespace

import pytest
from roboemot.config.stack import ConfigStack
from roboemot.webui import server as server_module
from roboemot.webui.api import Request, WebApi
from roboemot.webui.auth import PinStore
from roboemot.webui.server import SECURITY_HEADERS, accepts_gzip
from roboemot.webui.settings import SettingsEditor
from test_webui import PIN, FakeService
from test_webui_robot_section import FakePlugin, manifest
from test_webui_settings import serve

ROUTE = server_module.VENDOR_ROUTE
VENDOR = server_module.VENDOR_DIR
MANIFEST = json.loads((VENDOR / "VENDORED.json").read_text())
CSS = "tokens.css"
FONT = "assets/fonts/geist-latin.woff2"
GZIP = {"Accept-Encoding": "gzip, deflate, br"}
SECURITY = {k: v for k, v in SECURITY_HEADERS.items() if k != "Cache-Control"}


@pytest.fixture
def stack(tmp_path):
    return ConfigStack(
        app={"webui": {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3}},
        audit_path=tmp_path / "audit.jsonl",
    )


@pytest.fixture
def served(tmp_path, stack):
    srv, client = serve(tmp_path, SettingsEditor(stack), stack)
    yield client
    srv.stop()


def bare(client, method, path, headers=None):
    """A request with no ``Accept-Encoding`` header at all (http.client adds identity)."""
    conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=5)
    conn.putrequest(method, path, skip_accept_encoding=True)
    if client.cookie:
        conn.putheader("Cookie", client.cookie)
    for name, value in (headers or {}).items():
        conn.putheader(name, value)
    conn.endheaders()
    resp = conn.getresponse()
    out = SimpleNamespace(status=resp.status, headers=resp.headers, body=resp.read())
    conn.close()
    return out


def secure(resp) -> None:
    for name, value in SECURITY.items():
        assert resp.headers[name] == value, name
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


# ------------------------------------------------------------------ vendored files


def test_a_vendored_text_file_is_gzipped_for_a_request_that_accepts_it(served):
    resp = served.request("GET", ROUTE + CSS, headers=GZIP)
    assert resp.status == 200 and resp.headers["Content-Encoding"] == "gzip"
    assert gzip.decompress(resp.body) == (VENDOR / CSS).read_bytes()
    assert int(resp.headers["Content-Length"]) == len(resp.body)
    assert resp.headers["Vary"] == "Accept-Encoding"
    assert resp.headers["ETag"] == f'"{MANIFEST["files"][CSS]}-gz"'
    assert resp.headers["Cache-Control"] == "no-cache"
    secure(resp)


@pytest.mark.parametrize("header", [None, "identity", "gzip;q=0", "br", "*;q=0"])
def test_without_gzip_the_file_is_sent_as_it_is_with_vary(served, header):
    headers = {} if header is None else {"Accept-Encoding": header}
    resp = bare(served, "GET", ROUTE + CSS, headers)
    assert resp.status == 200 and "Content-Encoding" not in resp.headers
    assert resp.body == (VENDOR / CSS).read_bytes()
    assert resp.headers["ETag"] == f'"{MANIFEST["files"][CSS]}"'
    assert resp.headers["Vary"] == "Accept-Encoding"
    secure(resp)


@pytest.mark.parametrize("gz", [True, False])
def test_a_matching_etag_of_either_form_gets_304_with_every_header(served, gz):
    headers = GZIP if gz else {"Accept-Encoding": "identity"}
    etag = served.request("GET", ROUTE + CSS, headers=headers).headers["ETag"]
    resp = served.request("GET", ROUTE + CSS, headers={**headers, "If-None-Match": etag})
    assert resp.status == 304 and resp.body == b""
    assert resp.headers["ETag"] == etag and resp.headers["Vary"] == "Accept-Encoding"
    assert resp.headers["Cache-Control"] == "no-cache"
    assert "Content-Encoding" not in resp.headers
    secure(resp)


def test_the_etag_of_the_other_form_gets_the_file(served):
    plain = served.request("GET", ROUTE + CSS).headers["ETag"]
    resp = served.request("GET", ROUTE + CSS, headers={**GZIP, "If-None-Match": plain})
    assert resp.status == 200 and resp.headers["Content-Encoding"] == "gzip"


def test_a_font_is_never_gzipped(served):
    resp = served.request("GET", ROUTE + FONT, headers=GZIP)
    assert resp.status == 200 and "Content-Encoding" not in resp.headers
    assert "Vary" not in resp.headers
    assert resp.body == (VENDOR / FONT).read_bytes()


def test_head_of_a_gzipped_file_has_its_length_and_no_body(served):
    full = served.request("GET", ROUTE + CSS, headers=GZIP)
    resp = served.request("HEAD", ROUTE + CSS, headers=GZIP)
    assert resp.status == 200 and resp.body == b""
    assert resp.headers["Content-Encoding"] == "gzip"
    assert int(resp.headers["Content-Length"]) == len(full.body)


def test_a_range_request_gets_the_whole_file(served):
    resp = served.request("GET", ROUTE + CSS, headers={"Range": "bytes=0-9"})
    assert resp.status == 200 and resp.body == (VENDOR / CSS).read_bytes()


# ------------------------------------------------------------------ first-party files


@pytest.mark.parametrize("path", ["/", "/static/app.js", "/static/style.css"])
def test_first_party_files_stay_no_store_and_uncompressed(served, path):
    resp = served.request("GET", path, headers={**GZIP, "If-None-Match": "*"})
    assert resp.status == 200 and "Content-Encoding" not in resp.headers
    assert resp.headers["Cache-Control"] == "no-store" and "ETag" not in resp.headers
    assert "Vary" not in resp.headers


# ------------------------------------------------------------------ JSON


def test_a_large_json_body_is_gzipped_when_accepted(served):
    plain = served.request("GET", "/api/settings")
    resp = bare(served, "GET", "/api/settings", GZIP)
    assert resp.status == 200 and resp.headers["Content-Encoding"] == "gzip"
    assert gzip.decompress(resp.body) == plain.body
    assert len(resp.body) < len(plain.body) // 4
    for each in (plain, resp):
        assert each.headers["Vary"] == "Accept-Encoding"
        assert each.headers["Cache-Control"] == "no-store" and "ETag" not in each.headers
        secure(each)
    assert "Content-Encoding" not in plain.headers


def test_a_json_body_without_an_accept_encoding_header_is_plain(served):
    resp = bare(served, "GET", "/api/settings")
    assert resp.status == 200 and "Content-Encoding" not in resp.headers
    assert json.loads(resp.body)["pin_required"] is True


def test_a_small_json_body_is_never_gzipped(served):
    resp = served.request("GET", "/api/session", headers=GZIP)
    assert resp.status == 200 and "Content-Encoding" not in resp.headers
    assert "Vary" not in resp.headers


def test_a_repeated_body_reuses_its_gzip_form(served, monkeypatch):
    calls = []
    real = server_module.gzipped
    monkeypatch.setattr(
        server_module, "gzipped", lambda body, level: calls.append(level) or real(body, level)
    )
    first = bare(served, "GET", "/api/settings", GZIP).body
    again = bare(served, "GET", "/api/settings", GZIP).body
    assert first == again and calls == [server_module.JSON_GZIP_LEVEL]


@pytest.mark.parametrize("plugin", [None, FakePlugin()])
def test_the_prebuilt_settings_body_is_the_json_of_its_data(tmp_path, stack, plugin):
    pins = PinStore(tmp_path)
    pins.set(PIN)
    editor = SettingsEditor(stack, manifest=manifest() if plugin else None, plugin=plugin)
    api = WebApi(FakeService(), None, stack.effective, pins, settings=editor)
    host = {"host": "127.0.0.1:8042"}
    origin = {**host, "origin": "http://127.0.0.1:8042", "content-type": "application/json"}
    login = api.handle(Request("POST", "/api/login", origin, json.dumps({"pin": PIN}).encode(), ""))
    cookie = {**host, "cookie": login.cookies[0].split(";")[0]}
    for _ in range(2):
        resp = api.handle(Request("GET", "/api/settings", cookie, b"", "127.0.0.1"))
        assert resp.status == 200 and resp.body is not None
        assert resp.body == json.dumps(resp.data, allow_nan=False).encode()
        assert (resp.data["robot"] is None) is (plugin is None)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, False),
        ("", False),
        ("gzip", True),
        ("GZIP", True),
        ("deflate, gzip;q=0.5", True),
        ("gzip;q=0", False),
        ("gzip; q=0.0", False),
        ("gzip;q=bad", False),
        ("*", True),
        ("*;q=0", False),
        ("gzip;q=0, *", False),
        ("identity", False),
        ("br, deflate", False),
    ],
)
def test_accepts_gzip(header, expected):
    assert accepts_gzip(header) is expected
