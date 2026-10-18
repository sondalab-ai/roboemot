"""The vendored @sondalab/ui-kit copy: pinned, hash-checked, served offline with ETags (SPEC 12)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics
from roboemot.webui import make_server
from roboemot.webui import server as server_module
from test_webui import PIN, Client, FakeService

ROOT = Path(__file__).resolve().parents[3]
VENDOR = server_module.VENDOR_DIR
ROUTE = server_module.VENDOR_ROUTE


def _sync_script():
    spec = importlib.util.spec_from_file_location("sync_ui_kit", ROOT / "scripts/sync_ui_kit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYNC = _sync_script()
MANIFEST = json.loads((VENDOR / "VENDORED.json").read_text())


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cfg = RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0})
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path_factory.mktemp("vendor"))
    srv.set_pin(PIN)
    srv.start()
    yield Client(srv.port)
    srv.stop()


# ------------------------------------------------------------------ pin and manifest


def test_pin_is_the_released_kit_version():
    assert SYNC.UI_KIT_VERSION == "0.37.1"
    assert MANIFEST["version"] == SYNC.UI_KIT_VERSION
    assert MANIFEST["integrity"] == SYNC.UI_KIT_INTEGRITY
    assert json.loads((VENDOR / "package.json").read_text())["version"] == SYNC.UI_KIT_VERSION


def test_disk_matches_the_manifest_and_the_script_file_list(capsys):
    assert SYNC.check() == 0, capsys.readouterr().out
    assert set(MANIFEST["files"]) == set(SYNC.FILES)


def test_every_file_hash_is_recomputed_here():
    for name, digest in MANIFEST["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name


def test_a_changed_vendored_file_stops_start_up(tmp_path, monkeypatch):
    copy = tmp_path / "kit"
    shutil.copytree(VENDOR, copy)
    with (copy / "tokens.css").open("a") as handle:
        handle.write("/* edited */")
    monkeypatch.setattr(server_module, "VENDOR_DIR", copy)
    with pytest.raises(RuntimeError, match="tokens.css"):
        server_module.load_vendor()


def test_vendored_css_pulls_nothing_from_the_network():
    for name in MANIFEST["files"]:
        if name.endswith(".css"):
            text = (VENDOR / name).read_text()
            assert not re.search(r"@import|url\(\s*[\"']?(https?:)?//", text), name


def test_vendored_javascript_has_no_remote_imports():
    for name in MANIFEST["files"]:
        if name.endswith(".js"):
            text = (VENDOR / name).read_text()
            assert not re.search(r"""(from|import\()\s*["']https?:""", text), name


# ------------------------------------------------------------------ serving and caching


def test_each_manifest_file_is_served_with_a_strong_etag(client):
    for name, digest in MANIFEST["files"].items():
        resp = client.request("GET", ROUTE + name)
        assert resp.status == 200, name
        assert hashlib.sha256(resp.body).hexdigest() == digest
        assert resp.headers["ETag"] == f'"{digest}"'
        assert resp.headers["Cache-Control"] == "no-cache"
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]
        assert resp.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize(
    ("name", "ctype"),
    [
        ("components.css", "text/css"),
        ("effects.js", "text/javascript"),
        ("assets/fonts/geist-latin.woff2", "font/woff2"),
        ("VENDORED.json", None),
    ],
)
def test_content_types(client, name, ctype):
    resp = client.request("GET", ROUTE + name)
    if ctype is None:
        assert resp.status == 404  # the manifest itself is not served
    else:
        assert resp.headers["Content-Type"].startswith(ctype)


@pytest.mark.parametrize("sent", ["{etag}", "W/{etag}", "*", '"other", {etag}'])
def test_if_none_match_gets_304_without_a_body(client, sent):
    path = ROUTE + "tokens.css"
    etag = client.request("GET", path).headers["ETag"]
    resp = client.request("GET", path, headers={"If-None-Match": sent.format(etag=etag)})
    assert resp.status == 304
    assert resp.body == b""
    assert resp.headers["ETag"] == etag
    assert resp.headers["Cache-Control"] == "no-cache"
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


def test_a_stale_etag_gets_the_file(client):
    resp = client.request("GET", ROUTE + "tokens.css", headers={"If-None-Match": '"stale"'})
    assert resp.status == 200
    assert resp.body


def test_everything_else_stays_no_store_without_an_etag(client):
    for path in ["/", "/static/app.js", "/static/style.css", "/static/fx.js", "/api/strings"]:
        resp = client.request("GET", path, headers={"If-None-Match": "*"})
        assert resp.status == 200, path
        assert resp.headers["Cache-Control"] == "no-store", path
        assert "ETag" not in resp.headers, path


@pytest.mark.parametrize(
    "path",
    [
        ROUTE + "VENDORED.json",
        ROUTE + "workbench.css",
        ROUTE + "glass.css",
        ROUTE + "../style.css",
        ROUTE + "%2e%2e/style.css",
        ROUTE + "themes/../tokens.css",
        "/static/vendor/",
        "/static/vendor/sondalab-ui-kit",
    ],
)
def test_only_manifest_files_are_served(client, path):
    assert client.request("GET", path).status == 404
