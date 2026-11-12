"""Fixtures of the realtime tests: the scripted Gemini server over TLS (``gemini_server``)."""

from __future__ import annotations

import shutil

import pytest
from gemini_server import GeminiServer, tls


@pytest.fixture(scope="module")
def certs(tmp_path_factory):
    """One self-signed certificate per test module (``openssl`` makes it: skipped without)."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed: no certificate for the TLS fake server")
    return tls(tmp_path_factory.mktemp("gemini_tls"))


@pytest.fixture()
def gemini(certs, monkeypatch):
    """``gemini(answers, **server)`` -> a started server whose certificate the client trusts."""
    monkeypatch.setenv("SSL_CERT_FILE", str(certs[0]))
    made: list[GeminiServer] = []

    def make(answers=None, **kw):
        srv = GeminiServer(certs, answers, **kw).__enter__()
        made.append(srv)
        return srv

    yield make
    for srv in made:
        srv.__exit__(None, None, None)
