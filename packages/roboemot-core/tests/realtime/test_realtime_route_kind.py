"""``kind: realtime`` routes outside the realtime client: the router, the conformance check (CLI
and web UI), the web UI route form and the secret guard (SPEC 8, Realtime speech route; 11A, 12).
Fixtures and helpers come from the existing route tests."""

from __future__ import annotations

import datetime as dt

import pytest
import test_cli_llm_check as cli_check
import test_llm_conformance as conformance
import test_llm_router as lr
import test_secret_store as guard
import test_webui_route_api as route_api
from roboemot import cli
from roboemot.config import GuardrailError
from roboemot.config.schema import LLMRoute, SafetyConfig
from roboemot.llm import NoRouteAvailable, Router
from roboemot.llm.conformance import RouteKindUnsupported
from roboemot.webui.settings import SettingsEditor
from roboemot.webui.validation import ValidationError
from test_cli_llm_check import app_yaml  # noqa: F401  (fixture)
from test_webui_route_api import stack, store, web  # noqa: F401  (fixtures)

# ------------------------------------------------------------------- router


async def test_a_realtime_route_never_serves_chat():
    srv = lr.Server()
    rt = lr.route("rt", "lan.local", kind="realtime")
    r = lr.router(srv, [rt, lr.route("lan", "lan.local")])
    assert [x.name for x in r.eligible()] == ["lan"]
    assert (await lr.collect(r))[-1].route == "lan"
    only = lr.router(srv, [rt])
    with pytest.raises(NoRouteAvailable):
        await lr.collect(only)
    with pytest.raises(ValueError):
        async for _ in only.stream_route(rt, [{"role": "user", "content": "x"}]):
            pass


def test_realtime_route_lookup():
    day = [dt.date(2026, 10, 26)]
    rt = lr.route("rt", "lan.local", kind="realtime", daily_token_budget=10)
    r = lr.router(lr.Server(), [rt, lr.route("lan", "lan.local")], today=lambda: day[0])
    assert r.realtime_route("rt") == rt
    assert r.realtime_route("lan") is None  # a chat route
    assert r.realtime_route("nope") is None
    r.charge("rt", 10)
    assert r.realtime_route("rt") is None  # budget spent
    keyed = LLMRoute(**{**rt.model_dump(), "api_key": "${secret:missing}"})
    assert Router([keyed], SafetyConfig()).realtime_route("rt") is None  # key unavailable


# ------------------------------------------------------------------- conformance check


async def test_the_chat_check_refuses_a_realtime_route_before_any_request():
    fake = conformance.Fake()
    with pytest.raises(RouteKindUnsupported):
        await conformance.run(fake, conformance.route(kind="realtime"))
    assert fake.bodies == []


def test_the_cli_check_refuses_a_realtime_route(app_yaml, monkeypatch, capsys):  # noqa: F811
    app_yaml.write_text(app_yaml.read_text().replace("model: m", "model: m\n  kind: realtime"))
    fake = conformance.Fake()
    cli_check.serve(monkeypatch, fake)
    assert cli.main(cli_check.argv(app_yaml)) == 1
    assert "realtime route" in capsys.readouterr().err and fake.bodies == []


# ------------------------------------------------------------------- web UI and the secret guard


def test_a_realtime_route_is_added_from_the_web_and_its_kind_stays_put(web, stack):  # noqa: F811
    """``kind`` is chosen at add time; a later change of a keyed route's kind is the stack's
    secret rule (the admin), which the web refuses."""
    post, cloud = route_api.post, route_api.CLOUD
    body = {**cloud, "kind": "realtime", "secret": "cloud_key", "ack_endpoint": route_api.ORIGIN}
    assert post(web, "/api/routes/add", body).status == 200
    listing = web.request("GET", "/api/routes").json["routes"]
    assert next(r for r in listing if r["name"] == "cloud")["kind"] == "realtime"
    resp = post(web, "/api/settings", {"path": "llm_routes.cloud.kind", "value": "chat"})
    assert resp.status == 403, resp.json
    assert next(r for r in stack.effective().llm_routes if r.name == "cloud").kind == "realtime"
    assert post(web, "/api/routes/add", {**cloud, "name": "x", "kind": "voice"}).status == 422


def test_a_kind_change_of_a_keyed_route_needs_the_admin():
    guarded = guard.secret_stack()
    routes = [{**guard.SECRET_ROUTE, "kind": "realtime"}]
    with pytest.raises(GuardrailError):
        guarded.set("llm_routes", routes)
    assert guarded.effective().llm_routes[0].kind == "chat"
    guarded.set("llm_routes", routes, admin=True)
    assert guarded.effective().llm_routes[0].kind == "realtime"


def test_a_keyed_chat_route_switched_to_realtime_from_the_web_is_refused(web, stack):  # noqa: F811
    """Fix round 1: the other direction of the kind guard (chat -> realtime)."""
    post, cloud = route_api.post, route_api.CLOUD
    body = {**cloud, "secret": "cloud_key", "ack_endpoint": route_api.ORIGIN}
    assert post(web, "/api/routes/add", body).status == 200
    resp = post(web, "/api/settings", {"path": "llm_routes.cloud.kind", "value": "realtime"})
    assert resp.status == 403, resp.json
    assert next(r for r in stack.effective().llm_routes if r.name == "cloud").kind == "chat"
    assert web.request("GET", "/api/routes").json["realtime_key_hosts"] == []


@pytest.mark.parametrize(("kind", "named"), [("realtime", True), ("chat", False)])
def test_the_key_confirmation_names_the_provider_key_hosts(stack, kind, named):  # noqa: F811
    """Fix round 1, item 6: a realtime route's key may also go to the provider's key_hosts."""
    editor = SettingsEditor(
        stack, known_secrets=lambda: ["cloud_key"], realtime_key_hosts=lambda: [".rt.example.net"]
    )
    with pytest.raises(ValidationError) as exc:
        editor.add_route({**route_api.CLOUD, "kind": kind}, "cloud_key", None)
    assert exc.value.code == "endpoint_ack_required"
    assert exc.value.extra["endpoint"] == route_api.ORIGIN
    assert ("key_hosts" in exc.value.extra) is named
    if named:
        assert exc.value.extra["key_hosts"] == [".rt.example.net"]
    assert editor.realtime_key_hosts() == [".rt.example.net"]
