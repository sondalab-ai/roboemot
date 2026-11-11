"""The ``gemini_live`` route fields (SPEC 8, Gemini Live wire; SPEC 11A): ``protocol``, the
route's own ``voice`` and ``connect_timeout_s``, their validation, their precedence over the
shared ``voice.realtime_route`` values, and the web UI route form's round trip. A route that sets
none of them behaves and serializes as before."""

from __future__ import annotations

import pytest
import test_secret_store as guard
import test_webui_route_api as route_api
from realtime_helpers import route as rt_route
from realtime_helpers import settings as rt_settings
from roboemot.config import ConfigStack
from roboemot.config.schema import (
    LLMRoute,
    RealtimeRouteConfig,
    RoboEmotConfig,
    active_realtime_route,
    realtime_connect_timeout_s,
    realtime_voice,
)
from roboemot.config.stack import secret_violations
from roboemot.health.network import NetworkMonitor
from roboemot.realtime import route as rt_mod
from roboemot.realtime.input import RealtimeInput, reopen_limit_s
from roboemot.realtime.route import RealtimeRoute
from roboemot.realtime.session import DirectSession
from roboemot.webui import descriptions
from roboemot.webui.settings import SettingsEditor
from roboemot.webui.validation import ValidationError
from test_webui_route_api import stack, store, web  # noqa: F401  (fixtures)

GEMINI_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
GEMINI = {
    "name": "gemini",
    "kind": "realtime",
    "protocol": "gemini_live",
    "base_url": GEMINI_URL,
    "model": "gemini-3.8-live",
}
POLLEN = {"name": "pollen", "kind": "realtime", "base_url": "https://rt.example/v1", "model": "s2s"}
NEW_FIELDS = ("protocol", "voice", "connect_timeout_s")


def config(route: dict, **realtime: object) -> RoboEmotConfig:
    return RoboEmotConfig.model_validate(
        {"llm_routes": [route], "voice": {"realtime_route": {"route": route["name"], **realtime}}}
    )


# ------------------------------------------------------------------- defaults


def test_an_existing_route_keeps_its_values_and_serialization():
    route = LLMRoute(**POLLEN)
    assert (route.protocol, route.voice, route.connect_timeout_s) == ("openai_realtime", "", None)
    assert not set(NEW_FIELDS) & route.model_dump(exclude_defaults=True).keys()
    assert LLMRoute.model_validate(route.model_dump()) == route


def test_a_gemini_route_loads_in_direct_mode_with_robot_turns():
    cfg = config({**GEMINI, "voice": "Kore", "connect_timeout_s": 15})
    route = cfg.llm_routes[0]
    assert (route.protocol, route.voice, route.connect_timeout_s) == ("gemini_live", "Kore", 15.0)


# ------------------------------------------------------------------- validation


def test_gemini_live_needs_a_realtime_route():
    with pytest.raises(ValueError, match="protocol gemini_live needs kind: realtime"):
        LLMRoute(**{**GEMINI, "kind": "chat"})


@pytest.mark.parametrize(
    "url",
    ["https://generativelanguage.googleapis.com/ws", "ws://g.example/ws", "wss:///ws", "wss://g x"],
)
def test_gemini_live_needs_a_wss_url_with_a_host(url):
    with pytest.raises(ValueError, match="gemini_live base_url must be a wss:// URL"):
        LLMRoute(**{**GEMINI, "base_url": url})


@pytest.mark.parametrize("kind", ["chat", "realtime"])
def test_any_other_route_still_needs_http_or_https(kind):
    with pytest.raises(ValueError, match="base_url must be an http"):
        LLMRoute(**{**POLLEN, "kind": kind, "base_url": GEMINI_URL})


def test_an_unknown_protocol_is_refused():
    with pytest.raises(ValueError):
        LLMRoute(**{**POLLEN, "protocol": "gemini"})


@pytest.mark.parametrize("value", [0.5, 31])
def test_the_route_connect_timeout_keeps_the_shared_range(value):
    with pytest.raises(ValueError):
        LLMRoute(**{**GEMINI, "connect_timeout_s": value})


def test_the_active_gemini_route_needs_direct_mode():
    with pytest.raises(ValueError, match="gemini_live route: voice.realtime_route.mode must be"):
        config(GEMINI, mode="platform")


def test_the_active_gemini_route_needs_robot_turns():
    with pytest.raises(ValueError, match="gemini_live route: voice.realtime_route.turns must be"):
        config(GEMINI, turns="server")


def test_an_inactive_gemini_route_leaves_the_active_route_alone():
    cfg = RoboEmotConfig.model_validate(
        {
            "llm_routes": [GEMINI, POLLEN],
            "voice": {"realtime_route": {"route": "pollen", "mode": "platform"}},
        }
    )
    assert cfg.voice.realtime_route.mode == "platform"


def test_a_gemini_route_in_platform_mode_fails_at_load():
    with pytest.raises(ValueError, match="mode must be direct"):
        ConfigStack(
            app={
                "llm_routes": [GEMINI],
                "voice": {"realtime_route": {"route": "gemini", "mode": "platform"}},
            }
        )


# ------------------------------------------------------------------- precedence


def test_the_route_voice_wins_over_the_shared_voice():
    shared = RealtimeRouteConfig(voice="Serena")
    assert realtime_voice(LLMRoute(**POLLEN), shared) == "Serena"
    assert realtime_voice(LLMRoute(**{**GEMINI, "voice": "Kore"}), shared) == "Kore"
    assert realtime_voice(None, shared) == "Serena"


def test_the_route_connect_timeout_wins_over_the_shared_one():
    shared = RealtimeRouteConfig(connect_timeout_s=5)
    assert realtime_connect_timeout_s(LLMRoute(**POLLEN), shared) == 5
    assert realtime_connect_timeout_s(LLMRoute(**{**GEMINI, "connect_timeout_s": 15}), shared) == 15
    assert realtime_connect_timeout_s(None, shared) == 5


def voice_of(update: dict) -> str | None:
    return update["session"]["audio"]["output"].get("voice")


def test_session_update_asks_for_the_route_voice():
    cfg = config({**POLLEN, "voice": "Kore"}, voice="Serena")
    own = cfg.llm_routes[0]
    assert voice_of(rt_mod.session_update(cfg, "x", route=own)) == "Kore"
    assert voice_of(rt_mod.session_update(cfg, "x", route=LLMRoute(**POLLEN))) == "Serena"
    assert voice_of(rt_mod.session_update(cfg, "x")) == "Serena"  # no route: the shared one


def test_the_route_session_update_reads_the_active_route_voice():
    cfg_stack = ConfigStack(
        app={
            "llm_routes": [{**POLLEN, "voice": "Kore"}],
            "voice": {"realtime_route": {"enabled": True, "route": "pollen", "voice": "Serena"}},
        }
    )
    route = RealtimeRoute(
        cfg_stack,
        lambda: None,
        NetworkMonitor([], cfg_stack.effective().connectivity.network),
        DirectSession(),
    )
    assert voice_of(route.session_update(cfg_stack.effective(), "x")) == "Kore"


def test_the_client_uses_the_route_connect_timeout():
    pytest.importorskip("websockets")
    from roboemot.realtime.client import RealtimeClient

    def client(**kw: object) -> RealtimeClient:
        return RealtimeClient(
            rt_route("http://s2s.local/v1", **kw), rt_settings(), DirectSession(), auth_headers=dict
        )

    assert client()._connect_timeout_s() == 2.0
    assert client(connect_timeout_s=15)._connect_timeout_s() == 15


# ------------------------------------------------------------------- web UI


def test_protocol_and_voice_round_trip_through_the_web(web, stack):  # noqa: F811
    post = route_api.post
    body = {
        **GEMINI,
        "voice": "Kore",
        "locality": "cloud",
        "secret": "cloud_key",
        "ack_endpoint": "wss://generativelanguage.googleapis.com",
    }
    done = post(web, "/api/routes/add", body)
    assert done.status == 200, done.json
    row = next(r for r in web.request("GET", "/api/routes").json["routes"] if r["name"] == "gemini")
    assert (row["kind"], row["protocol"], row["voice"]) == ("realtime", "gemini_live", "Kore")
    moved = post(web, "/api/settings", {"path": "llm_routes.gemini.voice", "value": "Aoede"})
    assert moved.status == 200, moved.json
    assert next(r for r in stack.effective().llm_routes if r.name == "gemini").voice == "Aoede"
    lan = next(r for r in web.request("GET", "/api/routes").json["routes"] if r["name"] == "lan")
    assert "protocol" not in lan and "voice" not in lan  # a row without them stays as it was


@pytest.mark.parametrize(
    ("query", "code"), [("?alt=x", "url_query"), ("?key=x", "url_credentials")]
)
def test_the_web_still_refuses_a_query_on_a_gemini_url(web, stack, query, code):  # noqa: F811
    """The key goes in the header only: a query, a key-shaped one first, is refused."""
    added = route_api.post(web, "/api/routes/add", {**GEMINI, "base_url": GEMINI_URL + query})
    assert added.status == 422 and added.json["error"] == code
    assert "gemini" not in route_api.names(web)


def test_a_bad_protocol_from_the_web_is_refused(web):  # noqa: F811
    added = route_api.post(web, "/api/routes/add", {**GEMINI, "kind": "chat"})
    assert added.status == 422
    assert "gemini" not in route_api.names(web)


def test_a_keyed_route_cannot_switch_protocol_from_the_web(web, stack):  # noqa: F811
    """gemini_live needs a wss:// URL, so a protocol switch moves the key's endpoint too: the
    schema and the stack's secret rule (the admin) refuse it, one field at a time."""
    post = route_api.post
    body = {
        **POLLEN,
        "locality": "cloud",
        "secret": "cloud_key",
        "ack_endpoint": "https://rt.example",
    }
    assert post(web, "/api/routes/add", body).status == 200
    path = "llm_routes.pollen."
    resp = post(web, "/api/settings", {"path": path + "protocol", "value": "gemini_live"})
    assert resp.status in (403, 422), resp.json
    url = post(web, "/api/settings", {"path": path + "base_url", "value": GEMINI_URL})
    assert url.status in (403, 422), url.json
    pollen = next(r for r in stack.effective().llm_routes if r.name == "pollen")
    assert (pollen.protocol, pollen.base_url) == ("openai_realtime", POLLEN["base_url"])


def test_the_new_descriptions_have_an_italian_text():
    table = descriptions.table("it")
    for name in NEW_FIELDS:
        assert LLMRoute.model_fields[name].description in table


# ------------------------------------------------------------------- fix round 1


@pytest.mark.parametrize("tail", ["?alt=sse", "?", "#frag"])
def test_a_gemini_url_takes_no_query_or_fragment_at_load(tail):
    with pytest.raises(ValueError, match="takes no query, fragment or user info"):
        LLMRoute(**{**GEMINI, "base_url": GEMINI_URL + tail})


def test_a_gemini_url_takes_no_user_info_at_load():
    url = GEMINI_URL.replace("wss://", "wss://user:pw@")
    with pytest.raises(ValueError, match="takes no query, fragment or user info"):
        LLMRoute(**{**GEMINI, "base_url": url})


def test_the_shared_and_route_connect_timeout_ranges_match():
    shared = RealtimeRouteConfig.model_json_schema()["properties"]["connect_timeout_s"]
    own = LLMRoute.model_json_schema()["properties"]["connect_timeout_s"]["anyOf"][0]
    assert (own["minimum"], own["maximum"]) == (shared["minimum"], shared["maximum"])


def keyed(**kw: object) -> RoboEmotConfig:
    return RoboEmotConfig.model_validate(
        {"llm_routes": [{**guard.SECRET_ROUTE, "kind": "realtime", **kw}]}
    )


def test_a_protocol_change_alone_on_a_keyed_route_needs_the_admin():
    """The schema ties gemini_live to a wss URL, so this builds the candidate past it: the
    secret rule must still catch a protocol that changes alone."""
    approved = keyed()
    route = approved.llm_routes[0].model_copy(update={"protocol": "gemini_live"})
    candidate = approved.model_copy(update={"llm_routes": [route]})
    (violation,) = secret_violations(approved, candidate)
    assert violation.check == "secret" and violation.path == "llm_routes.cloud.api_key"
    assert secret_violations(approved, approved) == []


def test_an_existing_approval_reads_as_before():
    """The trusted string of an openai_realtime route carries no protocol: approvals made
    before the field existed still match."""
    approved = keyed()
    route = approved.llm_routes[0].model_copy(update={"base_url": "https://evil.example/v1"})
    (violation,) = secret_violations(approved, approved.model_copy(update={"llm_routes": [route]}))
    assert violation.approved == f"{guard.SECRET_ROUTE['base_url']} ${{secret:cloud_key}} realtime"


def test_the_key_confirmation_of_a_gemini_route_names_no_provider_hosts(stack):  # noqa: F811
    editor = SettingsEditor(
        stack, known_secrets=lambda: ["cloud_key"], realtime_key_hosts=lambda: [".rt.example.net"]
    )
    with pytest.raises(ValidationError) as exc:
        editor.add_route({**GEMINI, "locality": "cloud"}, "cloud_key", None)
    assert exc.value.code == "endpoint_ack_required"
    assert exc.value.extra == {"endpoint": "wss://generativelanguage.googleapis.com"}


def test_session_update_without_a_route_voice_is_unchanged():
    """Golden: the update an openai_realtime route without a voice of its own sends, as it was
    before the route fields existed."""
    cfg = config(POLLEN, voice="Serena")
    golden = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": "x",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": None},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 600,
                        "create_response": False,
                        "interrupt_response": False,
                    },
                },
                "output": {"format": {"type": "audio/pcm", "rate": None}, "voice": "Serena"},
            },
        },
    }
    for update in (
        rt_mod.session_update(cfg, "x", route=cfg.llm_routes[0]),
        rt_mod.session_update(cfg, "x"),
    ):
        update.pop("event_id", None)
        assert update == golden


def test_the_reopen_and_stream_session_waits_follow_the_active_route():
    cfg = config({**POLLEN, "connect_timeout_s": 15}, connect_timeout_s=5)
    assert active_realtime_route(cfg) is cfg.llm_routes[0]
    assert reopen_limit_s(cfg) == 15
    assert reopen_limit_s(config(POLLEN, connect_timeout_s=5)) == 5


def test_an_opening_waits_for_the_reconnect_past_the_shared_timeout():
    """A route with connect_timeout_s 15 and the shared 5: the opening after this input's reset
    keeps waiting until the reconnect lands at 10 s (a fake clock; nothing sleeps)."""
    cfg = config({**POLLEN, "connect_timeout_s": 15}, connect_timeout_s=5)
    now = [0.0]
    counts: list[str] = []

    class Route:  # the reconnect lands at 10 s
        connected = property(lambda _self: now[0] >= 10.0)
        opening = staticmethod(lambda: True)
        count = staticmethod(counts.append)

    route = Route()

    class Job:
        def wait(self, timeout_s: float) -> bool:
            now[0] += timeout_s
            return False

    inp = RealtimeInput(route)
    inp._clock = lambda: now[0]
    inp._stream_reopening = True
    inp._await_reopen(Job(), reopen_limit_s(cfg))
    assert counts == ["audio_reopen_wait"]
    assert 10.0 <= now[0] < 15.0  # past the shared 5 s, ended by the reconnect
