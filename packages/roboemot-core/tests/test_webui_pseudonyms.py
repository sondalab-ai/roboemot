"""Admin-only list of registry names refused for aliasing (SPEC 8 Pseudonyms, 12 Data, Web UI)."""

from __future__ import annotations

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics
from roboemot.llm.pseudonyms import NameSource, split_names
from roboemot.webui import make_server
from roboemot.webui.strings import EN, IT
from test_webui import PIN, Client, FakeService

TEMPLATES = {"en": "Person-{label}", "it": "Persona-{label}"}
NAMES = ["Marta", "user", "!!!", "Person", "!!!", "Luca Bianchi", "a\\b"]
PATH = "/api/pseudonyms/refused"


def test_split_names_separates_accepted_and_refused_without_duplicates():
    accepted, refused = split_names(NAMES, TEMPLATES)
    assert sorted(accepted.values()) == ["Luca Bianchi", "Marta"]
    assert sorted(refused.values()) == ["!!!", "Person", "a\\b", "user"]


def make_cfg(on: bool = True) -> RoboEmotConfig:
    return RoboEmotConfig(
        webui={"bind": "127.0.0.1", "port": 0}, privacy={"pseudonyms_to_cloud": on}
    )


@pytest.fixture
def served(tmp_path):
    srv = make_server(
        FakeService(),
        Diagnostics(),
        make_cfg(),
        tmp_path,
        pseudonyms=NameSource(lambda: NAMES),
    )
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    assert client.login().status == 200
    yield client
    srv.stop()


def test_the_right_pin_lists_the_refused_names(served):
    resp = served.request("POST", PATH, {"pin": PIN})
    assert resp.status == 200
    assert resp.json == {"enabled": True, "names": ["!!!", "Person", "a\\b", "user"]}


def test_no_session_is_refused_and_sees_no_name(served):
    resp = Client(served.port).request("POST", PATH, {"pin": PIN})
    assert resp.status == 401 and "user" not in str(resp.json)
    assert Client(served.port).request("GET", PATH).status == 401


@pytest.mark.parametrize(
    "body", [{}, {"pin": ""}, {"pin": "000000"}, {"pin": 5}, {"pin": PIN, "x": 1}]
)
def test_a_session_without_the_right_pin_is_refused(served, body):
    resp = served.request("POST", PATH, body)
    assert resp.status in (400, 401, 422) and "names" not in (resp.json or {})


def test_the_csrf_token_is_required(served):
    resp = served.request("POST", PATH, {"pin": PIN}, csrf=False)
    assert resp.status == 403


def test_the_list_is_empty_when_pseudonyms_are_off(tmp_path):
    srv = make_server(
        FakeService(),
        Diagnostics(),
        make_cfg(on=False),
        tmp_path,
        pseudonyms=NameSource(lambda: NAMES),
    )
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        resp = client.request("POST", PATH, {"pin": PIN})
        assert resp.status == 200 and resp.json == {"enabled": False, "names": []}
    finally:
        srv.stop()


def test_unavailable_without_a_source_or_when_the_names_cannot_be_read(tmp_path):
    def broken():
        raise OSError("registry down")

    for source in (None, NameSource(broken)):
        srv = make_server(FakeService(), Diagnostics(), make_cfg(), tmp_path, pseudonyms=source)
        srv.set_pin(PIN)
        srv.start()
        try:
            client = Client(srv.port)
            client.login()
            resp = client.request("POST", PATH, {"pin": PIN})
            assert resp.status == 503 and resp.json == {"error": "pseudonyms_unavailable"}
        finally:
            srv.stop()


def test_strings_exist_in_both_languages():
    for key in (
        "err_pseudonyms_unavailable",
        "refused_title",
        "refused_hint",
        "refused_none",
        "refused_off",
        "refused_show",
    ):
        assert EN[key] and IT[key], key
