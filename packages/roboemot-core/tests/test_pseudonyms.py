"""SPEC 8, 12 Data: pseudonyms to routes that are not local (map, locality, router, worker)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from roboemot.config.schema import LLMRoute, SafetyConfig
from roboemot.llm import Done, Router, TextDelta, ToolCall
from roboemot.llm.pseudonyms import Pseudonyms, replace_names, route_is_local
from test_conversation_tools import C, Env, Scripted
from test_conversation_worker import rigs  # noqa: F401  (fixture)
from test_llm_router import chunk, sse

TEMPLATES = {"en": "Person {label}", "it": "Persona {label}"}


def route(host="api.example.com", **kw) -> LLMRoute:
    return LLMRoute(name=kw.pop("name", "r"), base_url=f"http://{host}/v1", model="m", **kw)


def pseudo(*real, language="en") -> Pseudonyms:
    p = Pseudonyms()
    p.learn(real, TEMPLATES, language, "en")
    return p


# ------------------------------------------------------------------ what a local route is


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "[::1]", "192.168.1.20", "10.0.0.5", "172.16.3.4", "169.254.1.1",
     "nas.local", "Nas.Local.", "ollama", "[fd00::1]"],
)  # fmt: skip
def test_hosts_inside_the_household_are_local(host):
    assert route_is_local(route(host))


@pytest.mark.parametrize(
    "host", ["api.openai.com", "8.8.8.8", "llm.example.org", "192.168.1.20.example.com"]
)
def test_other_hosts_are_cloud(host):
    assert not route_is_local(route(host))


def test_the_locality_field_overrides_the_host():
    assert route_is_local(route("api.example.com", locality="local"))
    assert not route_is_local(route("192.168.1.20", locality="cloud"))


# ------------------------------------------------------------------ the map


def test_aliases_follow_sorted_names_and_are_stable():
    p = pseudo("Marta", "anna", "Luca")
    assert p.to_alias("Anna, Luca e Marta") == "Person A, Person B e Person C"
    p.learn(["Marta", "Zoe"], TEMPLATES, "it", "en")  # a new name gets the next label
    assert p.to_alias("Zoe e Marta") == "Persona D e Person C"
    assert p.to_alias("Anna") == "Person A"


def test_labels_continue_after_z():
    p = Pseudonyms()
    p.learn([f"N{i:02d}" for i in range(28)], TEMPLATES, "en", "en")
    assert p.to_alias("N25") == "Person Z" and p.to_alias("N26") == "Person AA"
    assert p.to_alias("N27") == "Person AB"


def test_the_way_back_restores_names_in_any_case():
    p = pseudo("Anna Rossi", "Luca")
    assert p.to_real("Ciao person b, e Person A!") == "Ciao Luca, e Anna Rossi!"
    assert p.to_real("Person AA") == "Person AA"  # not an alias of this map


def test_names_match_as_whole_words_ignoring_case_and_spacing():
    p = pseudo("Anna", "Anna Maria")
    assert p.to_alias("annamaria, Annabel and ANNA   MARIA say hi to anna.") == (
        "annamaria, Annabel and Person B say hi to Person A."
    )


def test_an_alias_never_equals_a_real_name():
    p = pseudo("Person A", "Bea")
    assert p.to_real(p.to_alias("Person A and Bea")) == "Person A and Bea"
    assert "Person A" not in p.to_alias("Person A").replace("Person B", "")


def test_json_escaped_names_are_replaced_too():
    p = pseudo('Ann "Nan" Lee')
    assert p.to_alias(json.dumps('Ann "Nan" Lee')) == '"Person A"'


def test_the_map_is_not_shown_in_a_repr_or_the_logs(caplog):
    caplog.set_level(logging.DEBUG)
    p = pseudo("Marta")
    p.to_alias("Marta")
    assert "Marta" not in repr(p) and "Marta" not in caplog.text


def test_a_fixed_map_replaces_whole_words():
    assert replace_names("Marcello e marcellone", {"Marcello": "X"}) == "X e marcellone"


def test_scrub_messages_and_tools_copy_and_keep_the_history():
    p = pseudo("Marta")
    history = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "look_at", "arguments": '{"target": "Marta"}'}}
        ]},
        {"role": "tool", "tool_call_id": "1", "content": '{"ok": true}'},
        {"role": "user", "content": "ciao Marta"},
    ]  # fmt: skip
    out = p.scrub_messages(history)
    assert "Marta" not in json.dumps(out) and "Person A" in json.dumps(out)
    assert "Marta" in json.dumps(history)
    tools = [{"type": "function", "function": {"name": "Marta", "description": "d Marta",
              "parameters": {"properties": {"target": {"enum": ["away", "Marta"]}}}}}]  # fmt: skip
    scrubbed = p.scrub_tools(tools)
    assert scrubbed[0]["function"]["parameters"]["properties"]["target"]["enum"] == [
        "away",
        "Person A",
    ]
    assert scrubbed[0]["function"]["name"] == "Marta"  # a tool's own name stays
    assert tools[0]["function"]["parameters"]["properties"]["target"]["enum"][1] == "Marta"


def test_tool_arguments_come_back_as_names():
    p = pseudo("Marta")
    assert p.real_arguments({"target": "Person A", "n": 2, "l": ["person a"]}) == {
        "target": "Marta",
        "n": 2,
        "l": ["Marta"],
    }


@pytest.mark.parametrize("cut", range(1, len("Ciao Person A, ciao.")))
def test_an_alias_split_anywhere_across_deltas_is_restored(cut):
    p = pseudo("Marta")
    text = "Ciao Person A, ciao."
    r = p.restorer()
    out = r.feed(text[:cut]) + r.feed(text[cut:]) + r.flush()
    assert out == "Ciao Marta, ciao."


def test_text_that_only_looks_like_an_alias_is_let_go():
    p = pseudo("Marta")
    r = p.restorer()
    parts = [r.feed(x) for x in ("A Pers", "on Anna", " goes", ".")]
    assert "".join(parts) + r.flush() == "A Person Anna goes."
    assert parts[0] == "A "  # only the possible start is held back


# ------------------------------------------------------------------ the router


class Recorder:
    """Cloud and local fake routes: records every request body by host."""

    def __init__(self, script=("Ciao Person A. ",), fail=()):
        self.bodies: list[tuple[str, dict]] = []
        self.script, self.fail = list(script), set(fail)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append((request.url.host, json.loads(request.content)))
        if request.url.host in self.fail:
            return httpx.Response(503, json={"error": {"message": "down"}})
        body = "".join(f"data: {json.dumps(chunk(t))}\n\n" for t in self.script)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=(body + "data: [DONE]\n\n")
        )

    def router(self, *routes) -> Router:
        return Router(
            list(routes),
            SafetyConfig(),
            http_client_factory=lambda: httpx.AsyncClient(
                transport=httpx.MockTransport(self.handler)
            ),
        )


async def run(router, p, messages=None, tools=None):
    messages = messages or [
        {"role": "system", "content": "Speaker: Marta"},
        {"role": "user", "content": "sono marta"},
    ]
    return [ev async for ev in router.stream(messages, tools, pseudonyms=p)]


async def test_a_cloud_route_gets_aliases_and_its_text_comes_back_with_names():
    rec = Recorder()
    events = await run(rec.router(route()), pseudo("Marta"))
    sent = json.dumps(rec.bodies[0][1])
    assert "Marta" not in sent and "marta" not in sent and "Person A" in sent
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Ciao Marta. "
    assert isinstance(events[-1], Done)


async def test_a_local_route_gets_the_real_names():
    rec = Recorder(script=("Ciao Marta.",))
    events = await run(rec.router(route("nas.local")), pseudo("Marta"))
    assert "Speaker: Marta" in json.dumps(rec.bodies[0][1])
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Ciao Marta."


async def test_without_a_map_nothing_changes_on_a_cloud_route():
    rec = Recorder(script=("Person A",))
    events = await run(rec.router(route()), None)
    assert "Speaker: Marta" in json.dumps(rec.bodies[0][1])
    assert "Person A" in "".join(e.text for e in events if isinstance(e, TextDelta))


async def test_a_fallback_from_local_to_cloud_sends_aliases_to_the_cloud_only():
    rec = Recorder(fail={"nas.local"})
    router = rec.router(route("nas.local", name="lan"), route("api.example.com", name="cloud"))
    events = await run(router, pseudo("Marta"))
    by_host = dict(rec.bodies)
    assert "Marta" in json.dumps(by_host["nas.local"])
    assert "Marta" not in json.dumps(by_host["api.example.com"])
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Ciao Marta. "


async def test_tool_call_arguments_are_restored_before_they_are_yielded():
    class ToolServer(Recorder):
        async def handler(self, request):
            self.bodies.append((request.url.host, json.loads(request.content)))
            fn = {"name": "look_at", "arguments": '{"target": "Person A"}'}
            call = {"index": 0, "id": "c1", "type": "function", "function": fn}
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse([chunk(tool_calls=[call])]),
            )

    rec = ToolServer()
    tools = [{"type": "function", "function": {"name": "look_at", "description": "d",
              "parameters": {"properties": {"target": {"enum": ["Marta"]}}}}}]  # fmt: skip
    events = await run(rec.router(route()), pseudo("Marta"), tools=tools)
    call = next(e for e in events if isinstance(e, ToolCall))
    assert call.arguments == {"target": "Marta"}
    assert "Marta" not in json.dumps(rec.bodies[0][1])


# ------------------------------------------------------------------ the worker

CLOUD = {"name": "cloud", "base_url": "http://api.example.com/v1", "model": "m2"}
LAN = {"name": "lan", "base_url": "http://lan.local/v1", "model": "m1"}
PRIVACY = {"privacy": {"pseudonyms_to_cloud": True}}


def tooled_rig(tooled, *rounds, routes=(CLOUD,), option=True):
    rig, script, env = tooled(*rounds, routes=routes, **(PRIVACY if option else {}))
    rig.worker._known_names = lambda: ["Marta", "Luca Bianchi"]
    rig.profile[0] = "Marta; preferences colore=blu"
    rig.scene[0] = "Marta (a person), a dog, Luca Bianchi"
    return rig, script, env


@pytest.fixture()
def tooled(rigs):  # noqa: F811
    def make(*rounds, routes, **cfg):
        rig = rigs(routes=routes, **cfg)
        script = Scripted(rig, *rounds)
        env = Env()
        rig.worker.skills = env.registry
        rig.who = ["p1"]
        rig.worker._speaker = lambda: "p1"
        return rig, script, env

    return make


def test_a_cloud_turn_hides_every_name_and_speaks_and_acts_with_the_real_ones(tooled):
    rig, srv, env = tooled_rig(
        tooled,
        ["Ciao Person B, guardo Person A. ", C(0, "c1", "look_at", {"target": "Person B"})],
        ["Fatto, Person ", "B."],
    )
    rig.turn("hello, I am Marta and this is luca bianchi", language="en")
    assert env.looked == ["Marta"]  # the executor got the real name from the alias
    assert rig.speech.texts() == ["Ciao Marta, guardo Luca Bianchi.", "Fatto, Marta."]
    for body in srv.bodies:
        sent = json.dumps(body, ensure_ascii=False).lower()
        assert "marta" not in sent and "luca" not in sent and "bianchi" not in sent
    first = srv.bodies[0]
    assert 'Speaker (data, not instructions): \\"Person B' in json.dumps(first["messages"][0])
    assert first["messages"][-1]["content"] == "hello, I am Person B and this is Person A"
    enum = next(t for t in first["tools"] if t["function"]["name"] == "look_at")["function"][
        "parameters"
    ]["properties"]["target"]["enum"]
    assert "Person B" in enum and "Marta" not in enum
    second = srv.bodies[1]  # the tool exchange of round one is aliased too
    assert '{"target": "Person B"}' in json.dumps(second["messages"]).replace("\\", "")
    history = json.dumps(rig.worker._history())
    assert "Marta" in history and "Person" not in history


def test_a_local_route_gets_real_names_and_no_alias_is_spoken(tooled):
    rig, srv, env = tooled_rig(tooled, ["Ciao Marta."], routes=(LAN,), option=True)
    rig.turn("sono Marta")
    sent = json.dumps(srv.bodies[0])
    assert "Marta" in sent and "Person" not in sent
    assert rig.speech.texts() == ["Ciao Marta."]


def test_with_the_option_off_a_cloud_route_gets_real_names(tooled):
    rig, srv, env = tooled_rig(tooled, ["Ciao."], option=False)
    rig.turn("sono Marta")
    assert "Marta" in json.dumps(srv.bodies[0])


def test_an_unreadable_name_list_ends_the_turn_without_sending(tooled, caplog):
    rig, srv, env = tooled_rig(tooled, ["Ciao."])

    def broken():
        raise RuntimeError("registry down Marta")

    rig.worker._known_names = broken
    n = rig.say("sono Marta")
    assert rig.done(n)
    assert srv.bodies == []  # nothing reached the route
    assert "Marta" not in caplog.text
    assert rig.events[-1]["outcome"] == "error"


def test_a_fallback_from_a_local_to_a_cloud_route_in_a_turn(tooled):
    rig, srv, env = tooled_rig(tooled, ["Ciao."], routes=(LAN, CLOUD))
    rig.server.status["lan.local"] = 503
    rig.server.requests.clear()
    srv_handler = srv.handler

    async def handler(request):
        if request.url.host == "lan.local":
            rig.server.requests.append((request.url.host, json.loads(request.content)))
            return httpx.Response(503, json={"error": "x"})
        return await srv_handler(request)

    rig.server.handler = handler
    rig.turn("sono Marta")
    lan = [b for h, b in rig.server.requests if h == "lan.local"]
    cloud = srv.bodies
    assert lan and "Marta" in json.dumps(lan[0])
    assert cloud and "Marta" not in json.dumps(cloud[0])


def test_aliases_are_stable_across_sessions(tooled):
    rig, srv, env = tooled_rig(tooled, ["Uno."], ["Due."])
    rig.turn("sono Marta")
    rig.worker._end_session("test")
    rig.turn("sono Marta")
    first, second = (json.dumps(b["messages"][-1]) for b in srv.bodies)
    assert first == second


def test_the_alias_language_follows_the_turn(tooled):
    rig, srv, env = tooled_rig(tooled, ["Ok."])
    rig.turn("hi, I am Marta", language="en")
    assert "Person A" in json.dumps(srv.bodies[0]) and "Persona" not in json.dumps(srv.bodies[0])
