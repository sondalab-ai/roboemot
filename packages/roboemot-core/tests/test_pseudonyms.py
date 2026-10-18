"""SPEC 8, 12 Data: pseudonyms to routes that are not local (map, locality, router, worker,
speech, guards)."""

from __future__ import annotations

import json
import logging
import unicodedata

import httpx
import pytest
from roboemot.config import ConfigStack, GuardrailError
from roboemot.config.schema import LLMRoute, PrivacyConfig, RoboEmotConfig, SafetyConfig
from roboemot.config.stack import violations
from roboemot.llm import Done, Router, TextDelta, ToolCall, build_messages
from roboemot.llm.pseudonyms import (
    Pseudonyms,
    replace_names,
    route_is_local,
)
from test_conversation_tools import C, Env, Scripted
from test_conversation_worker import rigs  # noqa: F401  (fixture)
from test_llm_router import chunk, sse

TEMPLATES = {"en": "Person-{label}", "it": "Persona-{label}"}


def route(host="api.example.com", **kw) -> LLMRoute:
    return LLMRoute(name=kw.pop("name", "r"), base_url=f"http://{host}/v1", model="m", **kw)


def pseudo(*real, language="en") -> Pseudonyms:
    p = Pseudonyms()
    p.sync(real, TEMPLATES, language, "en")
    return p


def alias_of(p: Pseudonyms, name: str) -> str:
    out = p.to_alias(name)
    assert out != name
    return out


def sent_to_cloud(p: Pseudonyms, messages, tools=None):
    return p.prepare(messages, tools)


# ------------------------------------------------------------------ what a local route is


@pytest.mark.parametrize(
    "host",
    [
        "localhost", "127.0.0.1", "127.5.5.5", "[::1]", "LOCALHOST.",
        "192.168.1.20", "10.0.0.5", "172.16.3.4", "172.31.255.1", "169.254.1.1",
        "[fe80::1]", "[fd00::1]", "[fc00::1]", "nas.local", "Nas.Local.", "ollama",
    ],
)  # fmt: skip
def test_this_machine_and_the_home_network_are_local_by_default(host):
    assert route_is_local(route(host))


@pytest.mark.parametrize(
    "host",
    [
        "api.openai.com", "8.8.8.8", "0.0.0.0", "100.64.0.1", "198.18.0.1", "198.19.255.1",
        "240.0.0.1", "192.0.0.8", "172.32.0.1", "172.15.0.1", "11.0.0.1",
        "134744072", "2130706433", "0x7f.1", "0x7f000001", "3232235777",
        "[2002:808:808::1]", "[::ffff:8.8.8.8]", "[::ffff:192.168.1.1]", "[64:ff9b::808:808]",
        "evil.localhost.example.com", "localhost.example.com", "nas.local.example.com",
        "api.openai.com\\@127.0.0.1", "127.0.0.1\\@api.openai.com",
        "user@127.0.0.1", "user@127.0.0.1@api.openai.com",
    ],
)  # fmt: skip
def test_everything_else_is_cloud_by_default(host):
    assert not route_is_local(route(host))


def test_the_admin_can_mark_a_lan_gateway_as_cloud_or_a_host_as_local():
    assert not route_is_local(route("192.168.1.20", locality="cloud"))
    assert not route_is_local(route("127.0.0.1", locality="cloud"))
    assert route_is_local(route("api.example.com", locality="local"))
    assert route_is_local(route("198.18.0.1", locality="local"))


@pytest.mark.parametrize("host", ["134744072", "0x7f.1", "2130706433", "user@127.0.0.1"])
def test_a_numeric_or_ambiguous_host_is_cloud_even_when_set_to_local(host):
    assert not route_is_local(route(host, locality="local"))


# ------------------------------------------------------------------ the map


def test_aliases_are_neutral_and_not_in_alphabetical_order():
    seen = set()
    for _ in range(40):
        p = pseudo("Anna", "Bea", "Carlo", "Dino", "Elena", "Fabio")
        seen.add(tuple(alias_of(p, n) for n in ("Anna", "Bea", "Carlo", "Dino", "Elena", "Fabio")))
    assert len(seen) > 1  # the label does not follow the name's rank
    assert all(a.startswith("Person-") for t in seen for a in t)


def test_a_known_name_keeps_its_alias_and_a_new_one_gets_a_fresh_label():
    p = pseudo("Marta", "Luca")
    marta = alias_of(p, "Marta")
    p.sync(["Marta", "Luca", "Zoe"], TEMPLATES, "it", "en")
    assert alias_of(p, "Marta") == marta
    assert alias_of(p, "Zoe").startswith("Persona-")
    assert len({alias_of(p, n) for n in ("Marta", "Luca", "Zoe")}) == 3


def test_labels_continue_after_z():
    p = pseudo(*[f"N{i:02d}" for i in range(28)])
    labels = {alias_of(p, f"N{i:02d}").removeprefix("Person-") for i in range(28)}
    assert "Z" in labels and "AA" in labels and "AB" in labels and len(labels) == 28


def test_a_name_that_equals_an_existing_alias_moves_that_alias():
    p = pseudo("Marta")
    old = alias_of(p, "Marta")
    p.sync(["Marta", old], TEMPLATES, "en", "en")  # someone is now registered under the alias
    assert alias_of(p, "Marta") != old
    assert p.to_alias(old) != old and p.to_alias(old) != p.to_alias("Marta")


def test_a_forgotten_name_is_dropped_and_no_longer_restorable():
    p = pseudo("Marta", "Luca")
    marta = alias_of(p, "Marta")
    p.sync(["Luca"], TEMPLATES, "en", "en")
    assert p.to_alias("Marta") == "Marta"
    out = p.prepare([{"role": "user", "content": f"ciao {marta}"}], None)
    assert out.restoration.text(f"{marta}") == marta


def test_the_map_is_not_shown_in_a_repr_or_the_logs(caplog):
    caplog.set_level(logging.DEBUG)
    p = pseudo("Marta")
    p.to_alias("Marta")
    p.prepare([{"role": "user", "content": "Marta"}], None)
    assert "Marta" not in repr(p) and "Marta" not in caplog.text


# ------------------------------------------------------------------ matching


@pytest.mark.parametrize(
    "name, text",
    [
        ("Strauß", "Strauß"),
        ("Strauß", "STRAUSS"),
        ("Strauß", "strauss"),
        ("İlker", "İlker"),
        ("İlker", "ilker"),
        ("Zoë", "Zoë"),
        ("Zoë", unicodedata.normalize("NFD", "Zoë")),
        ("Zoë", "ZOE"),
        ("Ömer", "ömer"),
        ("Niccolò", "NICCOLÒ"),
        ("José", "Jose"),
        ("Ann  Lee", "ann lee"),
        ("A.B", "say A.B now"),
        ("(Bob)", "hi (Bob)!"),
        ("Bob [x]", "Bob [x]"),
    ],
)
def test_names_are_matched_whatever_the_case_normalisation_or_punctuation(name, text):
    p = pseudo(name)
    out = p.to_alias(text)
    assert out != unicodedata.normalize("NFC", text)
    assert "Person-" in out
    assert not p.mentions(out)


def test_names_match_as_whole_words_and_the_longest_wins():
    p = pseudo("Anna", "Anna Maria")
    out = p.to_alias("annamaria, Annabel and ANNA   MARIA say hi to anna.")
    assert out.startswith("annamaria, Annabel and Person-")
    first, second = alias_of(p, "Anna"), alias_of(p, "Anna Maria")
    assert out.endswith(f"{second} say hi to {first}.")


def test_regex_metacharacters_in_a_name_are_literal():
    p = pseudo("A.B", "a+b")
    assert (
        p.to_alias("AxB and a+b and A.B")
        == f"AxB and {alias_of(p, 'a+b')} and {alias_of(p, 'A.B')}"
    )


def test_json_spellings_of_a_name_are_replaced():
    p = pseudo('Ann "Nan" Lee', "Ömer")
    assert "Ann" not in p.to_alias(json.dumps('Ann "Nan" Lee'))
    assert "mer" not in p.to_alias(json.dumps("Ömer"))  # Ömer
    assert "mer" not in p.to_alias(json.dumps("Ömer", ensure_ascii=False))


def test_a_fixed_map_replaces_whole_words():
    assert replace_names("Marcello e marcellone", {"Marcello": "X"}) == "X e marcellone"


# ------------------------------------------------------------------ requests


NAMES = ["Ömer", "Niccolò", "José", "Zoë", "Luca Bianchi"]


def wire(obj) -> list[str]:
    return [json.dumps(obj, ensure_ascii=a) for a in (False, True)]


def test_names_with_accents_do_not_leak_in_tool_calls_and_results_in_either_json_spelling():
    p = pseudo(*NAMES)
    exchange = ConversationWorkerExchange.build({"target": "Ömer"}, '{"seen": ["Zoë", "José"]}')
    history = [
        {"role": "user", "content": "guarda Niccolò"},
        *exchange,
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"n": "Niccolò"})},
        {"role": "tool", "tool_call_id": "c3", "content": "Zoë"},
    ]
    out = p.prepare(history, None)
    for body in wire(out.messages):
        for name in ("Ömer", "mer", "Niccol", "Jos", "Zo\\u00eb", "Zoë", "Zoe"):
            assert name not in body, name
    assert not p.leaks(out.messages)
    assert p.leaks(history)
    assert "Ömer" in json.dumps(history, ensure_ascii=False)  # the RAM history is untouched


class ConversationWorkerExchange:
    @staticmethod
    def build(args, result):
        from roboemot.conversation import ConversationWorker
        from roboemot.skills import ToolOutcome

        return ConversationWorker._exchange(
            [ToolCall("c1", "look_at", args)], [ToolOutcome(result, True)], 1, "ok"
        )


def test_the_exchange_is_written_without_unicode_escapes():
    msgs = ConversationWorkerExchange.build({"target": "Ömer"}, '{"ok": true}')
    assert "Ömer" in msgs[0]["tool_calls"][0]["function"]["arguments"]


def test_scrubbing_copies_and_tools_keep_their_own_names():
    p = pseudo("Marta")
    tools = [{"type": "function", "function": {"name": "Marta", "description": "d Marta",
              "parameters": {"properties": {"target": {"enum": ["away", "Marta"]}}}}}]  # fmt: skip
    out = p.prepare([{"role": "user", "content": "Marta"}], tools)
    enum = out.tools[0]["function"]["parameters"]["properties"]["target"]["enum"]
    assert enum == ["away", alias_of(p, "Marta")]
    assert out.tools[0]["function"]["name"] == "Marta"  # a tool's own name stays
    assert tools[0]["function"]["parameters"]["properties"]["target"]["enum"][1] == "Marta"


@pytest.mark.parametrize("which", ["profile", "scene"])
def test_a_summary_cap_never_cuts_inside_a_name(which):
    p = pseudo("Luca Bianchi")
    text = "Marta (a person), a dog, Luca Bianchi"
    cut_inside = text.index("Bianchi") + 3
    kw = {"profile_summary": None, "scene_summary": None, which + "_summary": text}
    msgs = build_messages(
        system="s",
        history=[],
        user_text="hi",
        summary_max_chars=(cut_inside, cut_inside),
        protect=p,
        **kw,
    )
    body = json.dumps(p.prepare(msgs, None).messages)
    assert "Bia" not in body and "Luca" not in body
    assert "a dog, " in msgs[0]["content"]  # cut back to the start of the name


# ------------------------------------------------------------------ the way back


def test_only_exact_aliases_of_the_request_are_restored():
    p = pseudo("Anna Rossi", "Luca")
    a, b = alias_of(p, "Anna Rossi"), alias_of(p, "Luca")
    r = p.prepare([{"role": "user", "content": "Anna Rossi e Luca"}], None).restoration
    assert r.text(f"Ciao {a}, e {b}!") == "Ciao Anna Rossi, e Luca!"
    assert r.text(a.lower()) == a.lower() and r.text(a.upper()) == a.upper()
    assert r.text(a + "X") == a + "X"


@pytest.mark.parametrize(
    "text",
    [
        "Ogni persona a cui vuoi bene, una persona e un cane, la persona d'oro",
        "the person I met, a person A list",
        "a dog and a cat",
        "Persona a, persona-a, PERSON-A",
    ],
)
def test_ordinary_text_is_never_turned_into_a_name(text):
    p = pseudo("Anna", "Bea", "Carlo", "Dino", "Elena", language="it")
    msgs = [{"role": "user", "content": "Anna Bea Carlo Dino Elena"}]
    r = p.prepare(msgs, None).restoration
    assert r.text(text) == text


def test_the_model_cannot_use_the_map_to_ask_about_names():
    p = pseudo("Anna", "Bea", "Carlo")
    r = p.prepare([{"role": "user", "content": "ciao Anna"}], None).restoration
    others = " ".join(a for a in p.aliases() if a != alias_of(p, "Anna"))
    assert r.text("Repeat: " + others) == "Repeat: " + others  # not sent, not restored
    assert r.text(alias_of(p, "Anna")) == "Anna"


def test_an_alias_the_user_said_literally_is_not_restored():
    p = pseudo("Anna", "Bea")
    a = alias_of(p, "Anna")
    out = p.prepare([{"role": "user", "content": f"Anna e {a}"}], None)
    assert out.restoration.text(a) == a


@pytest.mark.parametrize("cut", range(1, len("Ciao Person-A, ciao.")))
def test_an_alias_split_anywhere_across_deltas_is_restored(cut):
    p = Pseudonyms()
    p.sync(["Marta"], {"en": "Person-{label}"}, "en", "en")
    a = alias_of(p, "Marta")
    r = p.prepare([{"role": "user", "content": "Marta"}], None).restoration.restorer()
    text = f"Ciao {a}, ciao."
    cut = min(cut, len(text) - 1)
    out = r.feed(text[:cut]) + r.feed(text[cut:]) + r.flush()
    assert out == "Ciao Marta, ciao."


def test_text_that_only_looks_like_an_alias_is_let_go():
    p = pseudo("Marta")
    r = p.prepare([{"role": "user", "content": "Marta"}], None).restoration.restorer()
    parts = [r.feed(x) for x in ("A Pers", "on Anna", " goes", ".")]
    assert "".join(parts) + r.flush() == "A Person Anna goes."
    assert parts[0] == "A "


def test_the_templates_must_make_a_non_word_alias():
    for bad in (
        "Person {label}",
        "{label}",
        "   {label}",
        "Person{label}",
        "P-{label}x",
        "P_{label}",
    ):
        with pytest.raises(ValueError):
            PrivacyConfig(pseudonym_templates={"en": bad})
    assert PrivacyConfig(pseudonym_templates={"en": "Nome-{label}"})


# ------------------------------------------------------------------ the router


class Recorder:
    """Cloud and local fake routes: records every request body by host."""

    def __init__(self, script=None, fail=(), tail_error=()):
        self.bodies: list[tuple[str, dict]] = []
        self.script, self.fail, self.tail_error = script, set(fail), set(tail_error)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append((request.url.host, json.loads(request.content)))
        host = request.url.host
        if host in self.fail:
            return httpx.Response(503, json={"error": {"message": "down"}})
        script = self.script[host] if isinstance(self.script, dict) else self.script
        script = script or ["Ok."]

        async def body():
            for t in script:
                yield f"data: {json.dumps(chunk(t))}\n\n".encode()
            if host in self.tail_error:
                raise httpx.ReadError("broken")
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    def router(self, *routes) -> Router:
        return Router(
            list(routes),
            SafetyConfig(),
            http_client_factory=lambda: httpx.AsyncClient(
                transport=httpx.MockTransport(self.handler)
            ),
        )


MESSAGES = [
    {"role": "system", "content": "Speaker: Marta"},
    {"role": "user", "content": "sono marta"},
]


async def run(router, p, messages=None, tools=None):
    return [ev async for ev in router.stream(messages or MESSAGES, tools, pseudonyms=p)]


def text_of(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


async def test_a_cloud_route_gets_aliases_and_its_text_comes_back_with_names():
    p = pseudo("Marta")
    rec = Recorder([f"Ciao {alias_of(p, 'Marta')}. "])
    events = await run(rec.router(route()), p)
    sent = json.dumps(rec.bodies[0][1])
    assert "arta" not in sent and alias_of(p, "Marta") in sent
    assert text_of(events) == "Ciao Marta. " and isinstance(events[-1], Done)


async def test_a_local_route_gets_the_real_names():
    rec = Recorder(["Ciao Marta."])
    events = await run(rec.router(route("127.0.0.1")), pseudo("Marta"))
    assert "Speaker: Marta" in json.dumps(rec.bodies[0][1])
    assert text_of(events) == "Ciao Marta."


async def test_a_lan_host_gets_real_names_unless_the_admin_marks_it_cloud():
    rec = Recorder()
    await run(rec.router(route("192.168.1.20")), pseudo("Marta"))
    assert "Speaker: Marta" in json.dumps(rec.bodies[0][1])
    rec = Recorder()  # a LiteLLM gateway on the LAN that forwards to a cloud provider
    await run(rec.router(route("192.168.1.20", locality="cloud")), pseudo("Marta"))
    assert "Marta" not in json.dumps(rec.bodies[0][1])


async def test_without_a_map_nothing_changes_on_a_cloud_route():
    rec = Recorder(["Person-A"])
    events = await run(rec.router(route()), None)
    assert "Speaker: Marta" in json.dumps(rec.bodies[0][1])
    assert text_of(events) == "Person-A"


async def test_a_fallback_from_local_to_cloud_sends_aliases_to_the_cloud_only():
    p = pseudo("Marta")
    rec = Recorder([f"Ciao {alias_of(p, 'Marta')}. "], fail={"127.0.0.1"})
    router = rec.router(route("127.0.0.1", name="lan"), route("api.example.com", name="cloud"))
    events = await run(router, p)
    by_host = dict(rec.bodies)
    assert "Marta" in json.dumps(by_host["127.0.0.1"])
    assert "Marta" not in json.dumps(by_host["api.example.com"])
    assert text_of(events) == "Ciao Marta. "


async def test_a_request_that_still_holds_a_name_is_not_sent(monkeypatch):
    p = pseudo("Marta")
    real_prepare = p.prepare

    def unscrubbed(messages, tools):
        out = real_prepare(messages, tools)
        out.messages, out.tools = messages, tools  # a scrub that failed
        return out

    monkeypatch.setattr(p, "prepare", unscrubbed)
    rec = Recorder()
    router = rec.router(route())
    with pytest.raises(Exception, match="all routes failed"):
        await run(router, p)
    assert rec.bodies == [] and router.failures[-1][1] == "PseudonymLeak"


async def test_tool_call_arguments_are_restored_before_they_are_yielded():
    p = pseudo("Marta")
    a = alias_of(p, "Marta")

    class ToolServer(Recorder):
        async def handler(self, request):
            self.bodies.append((request.url.host, json.loads(request.content)))
            fn = {"name": "look_at", "arguments": json.dumps({"target": a})}
            call = {"index": 0, "id": "c1", "type": "function", "function": fn}
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse([chunk(tool_calls=[call])]),
            )

    rec = ToolServer()
    tools = [{"type": "function", "function": {"name": "look_at", "description": "d",
              "parameters": {"properties": {"target": {"enum": ["Marta"]}}}}}]  # fmt: skip
    events = await run(rec.router(route()), p, tools=tools)
    call = next(e for e in events if isinstance(e, ToolCall))
    assert call.arguments == {"target": "Marta"}
    assert "Marta" not in json.dumps(rec.bodies[0][1])


async def test_a_barge_in_in_the_middle_of_an_alias_drops_the_held_text():
    p = pseudo("Marta")
    a = alias_of(p, "Marta")
    rec = Recorder(["Ciao ", a[:-1], a[-1:], " e basta."])
    stream = rec.router(route()).stream(MESSAGES, None, pseudonyms=p)
    first = await stream.__anext__()
    assert isinstance(first, TextDelta) and first.text == "Ciao "
    await stream.aclose()  # the person talks over the robot
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


async def test_a_stream_that_breaks_after_only_held_text_falls_back_silently():
    p = pseudo("Marta")
    a = alias_of(p, "Marta")
    rec = Recorder({"api.example.com": [a[:-1]], "backup.example.com": ["Ok."]},
                   tail_error={"api.example.com"})  # fmt: skip
    router = rec.router(route(name="one"), route("backup.example.com", name="two"))
    events = await run(router, p)
    assert text_of(events) == "Ok."  # nothing had been said, so the next route answers


# ------------------------------------------------------------------ the worker

CLOUD = {"name": "cloud", "base_url": "http://api.example.com/v1", "model": "m2"}
LAN = {"name": "lan", "base_url": "http://127.0.0.1/v1", "model": "m1"}
PRIVACY = {"privacy": {"pseudonyms_to_cloud": True}}
KNOWN = ["Marta", "Luca Bianchi", "Ömer"]


@pytest.fixture()
def tooled(rigs):  # noqa: F811
    def make(rounds, routes, **cfg):
        rig = rigs(routes=routes, **cfg)
        script = Scripted(rig)
        env = Env()
        rig.worker.skills = env.registry
        rig.who = ["p1"]
        rig.worker._speaker = lambda: "p1"
        rig.worker._known_names = lambda: KNOWN
        pseudonyms = rig.worker._pseudonyms
        pseudonyms.sync(KNOWN, TEMPLATES, "en", "en")  # so the test knows the aliases
        script.rounds = rounds(lambda name: alias_of(pseudonyms, name))
        rig.profile[0] = "Marta; preferences colore=blu"
        rig.scene[0] = "Marta (a person), a dog, Luca Bianchi, Ömer"
        return rig, script, env

    return make


def test_a_cloud_turn_hides_every_name_and_speaks_and_acts_with_the_real_ones(tooled):
    rig, srv, env = tooled(
        lambda al: [
            [
                f"Ciao {al('Marta')}, guardo {al('Luca Bianchi')}. ",
                C(0, "c1", "look_at", {"target": al("Marta")}),
            ],
            [f"Fatto, {al('Marta')[:-1]}", al("Marta")[-1:] + "."],
        ],
        routes=(CLOUD,),
        **PRIVACY,
    )
    rig.turn("hello, I am Marta and this is luca bianchi, with Ömer", language="en")
    assert env.looked == ["Marta"]  # the executor got the real name from the alias
    assert rig.speech.texts() == ["Ciao Marta, guardo Luca Bianchi.", "Fatto, Marta."]
    for body in srv.bodies:
        for text in wire(body):
            assert not rig.worker._pseudonyms.leaks(body), text
            for word in ("Marta", "marta", "Luca", "Bianchi", "Ömer", "u00d6"):
                assert word not in text, word
    first = srv.bodies[0]
    alias = rig.worker._pseudonyms.to_alias("Marta")
    assert alias in json.dumps(first["messages"][0])
    assert alias in first["messages"][-1]["content"]
    enum = next(t for t in first["tools"] if t["function"]["name"] == "look_at")["function"][
        "parameters"
    ]["properties"]["target"]["enum"]
    assert alias in enum and "Marta" not in enum
    assert f'"target": "{alias}"' in json.dumps(srv.bodies[1]["messages"]).replace("\\", "")
    history = json.dumps(rig.worker._history(), ensure_ascii=False)
    assert "Marta" in history and "Person-" not in history


def test_a_local_route_gets_real_names_and_no_alias_is_spoken(tooled):
    rig, srv, env = tooled(lambda al: [["Ciao Marta."]], routes=(LAN,), **PRIVACY)
    rig.turn("sono Marta")
    sent = json.dumps(srv.bodies[0], ensure_ascii=False)
    assert "Marta" in sent and "Persona-" not in sent and "Person-" not in sent
    assert rig.speech.texts() == ["Ciao Marta."]


def test_with_the_option_off_a_cloud_route_gets_real_names(tooled):
    rig, srv, env = tooled(lambda al: [["Ciao."]], routes=(CLOUD,))
    rig.turn("sono Marta")
    assert "Marta" in json.dumps(srv.bodies[0])


def test_an_unreadable_name_list_ends_the_turn_without_sending(tooled, caplog):
    rig, srv, env = tooled(lambda al: [["Ciao."]], routes=(CLOUD,), **PRIVACY)

    def broken():
        raise RuntimeError("registry down Marta")

    rig.worker._known_names = broken
    assert rig.done(rig.say("sono Marta"))
    assert srv.bodies == [] and "Marta" not in caplog.text
    assert rig.events[-1]["outcome"] == "error"


def test_a_forgotten_name_is_gone_from_the_next_request(tooled):
    rig, srv, env = tooled(lambda al: [["Uno."], ["Due."]], routes=(CLOUD,), **PRIVACY)
    rig.turn("sono Marta")
    rig.worker._known_names = lambda: ["Luca Bianchi"]
    rig.turn("sono Marta")
    assert "Marta" in json.dumps(srv.bodies[1])  # no longer registered: sent as typed
    assert rig.worker._pseudonyms.to_alias("Marta") == "Marta"


def test_aliases_are_stable_across_sessions(tooled):
    rig, srv, env = tooled(lambda al: [["Uno."], ["Due."]], routes=(CLOUD,), **PRIVACY)
    rig.turn("sono Marta")
    rig.worker._end_session("test")
    rig.turn("sono Marta")
    first, second = (json.dumps(b["messages"][-1]) for b in srv.bodies)
    assert first == second


def test_the_alias_language_follows_the_turn_for_a_new_name(tooled):
    rig, srv, env = tooled(lambda al: [["Ok."]], routes=(CLOUD,), **PRIVACY)
    rig.worker._known_names = lambda: [*KNOWN, "Zoe"]
    rig.turn("hi, I am Zoe", language="en")
    assert "Person-" in json.dumps(srv.bodies[0]["messages"][-1])


# ------------------------------------------------------------------ speech


async def test_the_speech_route_still_hears_the_names_it_speaks():
    from roboemot.config.schema import OpenAISpeechConfig, VoiceConfig
    from roboemot.voice.openai_speech import OpenAISpeechBackend
    from roboemot.voice.prosody import Delivery

    posted: list[bytes] = []

    async def handler(request):
        posted.append(request.content)
        return httpx.Response(200, content=b"\x00\x00" * 8)

    router = Router(
        [route(name="r")],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    backend = OpenAISpeechBackend(OpenAISpeechConfig(route="r"), lambda: router)
    delivery = Delivery(1.0, 0.0, 0.0, 0.5, 1.0)
    chunks = [
        c
        async for c in backend.synthesize("Ciao Zoë", delivery, language="it", voice=VoiceConfig())
    ]
    assert chunks and b"Zo" in b"".join(posted)  # owner decision: no rerouting to Piper


# ------------------------------------------------------------------ guards (SPEC 11A)


def test_the_option_cannot_be_turned_off_at_runtime_but_can_be_turned_on():
    stack = ConfigStack(app=PRIVACY)
    with pytest.raises(GuardrailError):
        stack.set("privacy.pseudonyms_to_cloud", False)
    off = ConfigStack()
    off.set("privacy.pseudonyms_to_cloud", True)
    assert off.get("privacy.pseudonyms_to_cloud") is True


def test_a_preset_switch_cannot_turn_the_option_off(tmp_path):
    (tmp_path / "off.yaml").write_text("privacy:\n  pseudonyms_to_cloud: false\n")
    stack = ConfigStack(preset=PRIVACY, preset_dirs=[tmp_path])
    with pytest.raises(GuardrailError):
        stack.apply_preset("off")
    assert stack.effective().privacy.pseudonyms_to_cloud is True


def test_templates_and_locality_need_admin():
    stack = ConfigStack(app={"llm_routes": [CLOUD]})
    with pytest.raises(GuardrailError):
        stack.set("privacy.pseudonym_templates.en", "Nome-{label}")
    stack.set("privacy.pseudonym_templates.en", "Nome-{label}", admin=True)
    with pytest.raises(GuardrailError):
        stack.set("llm_routes", [{**CLOUD, "locality": "local"}])
    with pytest.raises(GuardrailError):
        stack.set("llm_routes", [CLOUD, {**LAN, "name": "extra", "locality": "local"}])
    stack.set("llm_routes", [CLOUD, {**LAN, "name": "plain"}])  # a new route stays "auto"


def test_violations_flag_a_new_local_route_and_a_changed_one():
    approved = RoboEmotConfig(llm_routes=[CLOUD])
    new_local = RoboEmotConfig(llm_routes=[CLOUD, {**LAN, "name": "x", "locality": "local"}])
    changed = RoboEmotConfig(llm_routes=[{**CLOUD, "locality": "local"}])
    assert any(v.kind == "admin" and "locality" in v.path for v in violations(approved, new_local))
    assert any(v.kind == "admin" and "locality" in v.path for v in violations(approved, changed))
    assert not violations(approved, RoboEmotConfig(llm_routes=[CLOUD, {**LAN, "name": "y"}]))
