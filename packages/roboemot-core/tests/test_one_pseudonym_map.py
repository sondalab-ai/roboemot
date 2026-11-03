"""One pseudonym map per process (SPEC 8 Pseudonyms, 12 Data): the traits rating, the route
checks and the conversation give a registry name the same alias."""

from __future__ import annotations

import asyncio
import functools
import json
import random
import threading
import time

import httpx
import test_conversation_worker
from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig, SafetyConfig
from roboemot.conversation import ConversationWorker
from roboemot.llm import Router, TextDelta
from roboemot.llm.pseudonyms import NameSource
from roboemot.persona.llm_traits import TraitCache, TraitsWorker
from test_conversation_worker import rigs  # noqa: F401  (fixture)
from test_llm_conformance import Fake
from test_llm_router import chunk, route
from test_traits_worker import Server as TraitsServer
from test_traits_worker import energy
from test_webui_routes import ROUTE, checker_for, finish, serve

NAME = "Italian"  # the route check's own prompt holds this word
NAMES = [NAME, "Marta", "Luca Bianchi", "Ömer"]
PRIVACY = {"pseudonyms_to_cloud": True}
CLOUD = {"name": "cloud", "base_url": "http://api.example.com/v1", "model": "m1"}


def cfg_on(**more) -> RoboEmotConfig:
    return RoboEmotConfig(privacy=PRIVACY, conversation={"language": "en"}, **more)


def mapping(source: NameSource) -> dict[str, str]:
    snapshot = source.for_config(cfg_on())
    assert snapshot is not None
    return {n: snapshot.to_alias(n) for n in NAMES}


def test_traits_route_check_and_conversation_send_the_same_alias(tmp_path, rigs, monkeypatch):  # noqa: F811
    source = NameSource(lambda: NAMES)

    # The persona traits rating (its own thread and loop).
    traits_server = TraitsServer()
    stack = ConfigStack(
        app={
            "persona": {
                "traits_source": "llm",
                "prompt": f"Be as warm as an {NAME}.",
                "traits_retry_s": 0,
            },
            "llm_routes": [CLOUD],
            "privacy": PRIVACY,
            "conversation": {"language": "en"},
        }
    )
    traits = TraitsWorker(
        stack,
        traits_server.router(stack.effective()),
        TraitCache(tmp_path / "t.json"),
        traits_server.router,
        names=source,
    )
    traits.start()
    try:
        assert test_conversation_worker.wait_for(lambda: energy(stack) == 0.8)
    finally:
        traits.close()
    alias = source.peek().to_alias(NAME)
    assert alias != NAME
    rated = json.dumps(traits_server.requests, ensure_ascii=False)
    assert alias in rated and NAME not in rated

    # The web UI route check (a thread of the web server).
    cloud = {**ROUTE, "base_url": "https://api.example.com/v1"}
    cfg = RoboEmotConfig(
        webui={"bind": "127.0.0.1", "port": 0},
        llm_routes=[cloud],
        privacy=PRIVACY,
        conversation={"language": "en"},
    )
    fake = Fake()
    srv, client = serve(tmp_path, cfg, checker_for(cfg, fake, names=source))
    try:
        assert client.request("POST", "/api/routes/lan/check", {}).status == 202
        assert finish(client)["last_check"] is not None
    finally:
        srv.stop()
    checked = json.dumps(fake.bodies, ensure_ascii=False)
    assert alias in checked and NAME not in checked

    # The conversation (its own thread and loop), built with the same source.
    monkeypatch.setattr(
        test_conversation_worker,
        "ConversationWorker",
        functools.partial(ConversationWorker, name_source=source),
    )
    rig = rigs(routes=(CLOUD,), privacy=PRIVACY)
    assert rig.worker._name_source is source
    rig.turn(f"I am {NAME}", language="en")
    sent = json.dumps([body for _, body in rig.server.requests], ensure_ascii=False)
    assert alias in sent and NAME not in sent
    assert source.peek().to_alias(NAME) == alias  # still the one alias


def test_concurrent_callers_get_one_alias_per_name():
    source = NameSource(lambda: NAMES)
    seen: list[dict[str, str]] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def caller(language: str, keys: tuple[str, ...]) -> None:
        start.wait()
        for _ in range(50):
            snapshot = source.for_config(cfg_on(), language, keys)
            assert snapshot is not None
            got = {n: snapshot.to_alias(n) for n in NAMES}
            with lock:
                seen.append(got)

    threads = [
        threading.Thread(target=caller, args=(("en", "it")[i % 2], ("target",) if i % 3 else ()))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(seen) == 8 * 50
    assert all(s == seen[0] for s in seen)
    assert len(set(seen[0].values())) == len(NAMES)


def test_a_snapshot_keeps_its_own_structural_keys_and_ignores_later_syncs():
    registry = [*NAMES, "Target"]
    source = NameSource(lambda: registry)
    conversation = source.for_config(cfg_on(), "en", ("target",))
    strict = source.for_config(cfg_on())
    assert conversation is not None and strict is not None
    tool = [{"role": "tool", "tool_call_id": "c1", "content": '{"target": "ok"}'}]
    assert not conversation.leaks(tool)
    assert strict.leaks(tool)  # traits and route checks stay strict (fail closed)
    alias = conversation.to_alias("Marta")
    registry.remove("Marta")
    later = source.for_config(cfg_on())
    assert later is not None and later.to_alias("Marta") == "Marta"
    assert conversation.to_alias("Marta") == alias  # a request in flight is not changed


def test_a_deleted_or_renamed_name_is_dropped_and_a_new_one_gets_a_new_alias():
    registry = list(NAMES)
    source = NameSource(lambda: registry)
    before = mapping(source)
    registry[registry.index("Marta")] = "Martina"  # renamed in the registry
    after = source.for_config(cfg_on())
    assert after is not None
    assert after.to_alias("Marta") == "Marta"  # no longer registered: sent as typed
    assert after.to_alias("Martina") not in before.values()
    assert after.to_alias("Luca Bianchi") == before["Luca Bianchi"]  # the others keep theirs
    registry.append("Marta")  # registered again
    again = source.for_config(cfg_on())
    assert again is not None
    assert again.to_alias("Marta") not in (before["Marta"], "Marta")


def test_a_restart_hands_out_new_aliases():
    names = [f"Name{c}" for c in "abcdefghijkl"]
    first, second = NameSource(lambda: names), NameSource(lambda: names)
    one, two = first.for_config(cfg_on()), second.for_config(cfg_on())
    assert one is not None and two is not None
    a = {n: one.to_alias(n) for n in names}
    b = {n: two.to_alias(n) for n in names}
    assert set(a.values()) == set(b.values())  # the same labels ...
    assert a != b  # ... given out in another order (12! orders)


def test_the_option_off_gives_no_map_and_reads_no_names():
    def broken():
        raise AssertionError("names read with the option off")

    assert NameSource(broken).for_config(RoboEmotConfig()) is None


def test_repr_shows_no_name():
    source = NameSource(lambda: NAMES)
    source.for_config(cfg_on())
    assert "Marta" not in repr(source) and "Person-" not in repr(source)


def test_a_name_added_while_callers_race_keeps_one_alias_and_is_never_dropped():
    """Why ``NameSource`` reads and syncs under one lock: a caller holding a list read before a
    name was added must not sync after a caller that saw it (the name would be dropped, then
    come back with a new label)."""
    registry = list(NAMES)
    added = threading.Event()

    jitter = random.Random(7)

    def slow_names() -> list[str]:
        out = list(registry)
        time.sleep(jitter.uniform(0, 0.004))  # reads take varying time: they finish out of order
        return out

    source = NameSource(slow_names)
    aliases: dict[str, set[str]] = {}
    dropped: list[str] = []
    lock = threading.Lock()

    def caller() -> None:
        seen: set[str] = set()
        for _ in range(40):
            snapshot = source.for_config(cfg_on())
            assert snapshot is not None
            for name in [*NAMES, *(f"New{i}" for i in range(10))]:
                alias = snapshot.to_alias(name)
                if alias == name:
                    if name in seen:
                        dropped.append(name)  # once seen by this caller, it must stay
                    continue
                seen.add(name)
                with lock:
                    aliases.setdefault(name, set()).add(alias)

    def writer() -> None:
        for i in range(10):
            registry.append(f"New{i}")
            time.sleep(0.004)
        added.set()

    threads = [threading.Thread(target=caller) for _ in range(6)]
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert added.is_set()
    assert dropped == []
    assert {n: len(a) for n, a in aliases.items() if len(a) != 1} == {}
    assert all(f"New{i}" in aliases for i in range(10))


async def test_a_request_in_flight_keeps_its_snapshot_while_another_thread_renames():
    registry = ["Marta", "Luca"]
    source = NameSource(lambda: registry)
    snapshot = source.for_config(cfg_on(), "en", ("target",))
    assert snapshot is not None
    alias = snapshot.to_alias("Marta")
    bodies: list[dict] = []

    def rename() -> None:  # the traits worker or a route check, on its own thread
        registry[0] = "Martina"
        later = source.for_config(cfg_on())
        assert later is not None and later.to_alias("Marta") == "Marta"

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))

        async def body():
            yield f"data: {json.dumps(chunk(f'Ciao {alias[:-1]}'))}\n\n".encode()
            await asyncio.to_thread(rename)  # mid-stream
            yield f"data: {json.dumps(chunk(alias[-1:] + '.'))}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    r = Router(
        [route("cloud", "api.example.com", timeout=2.0)],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    messages = [{"role": "user", "content": "sono Marta"}]
    events = [ev async for ev in r.stream(messages, None, pseudonyms=snapshot)]
    text = "".join(ev.text for ev in events if isinstance(ev, TextDelta))
    assert text == "Ciao Marta."  # restored by the request's own snapshot
    assert "Marta" not in json.dumps(bodies, ensure_ascii=False)
    assert source.peek().to_alias("Marta") == "Marta"  # the live map moved on ...
    assert snapshot.to_alias("Marta") == alias  # ... the snapshot did not
    assert snapshot.leaks([{"role": "user", "content": "Marta"}])
