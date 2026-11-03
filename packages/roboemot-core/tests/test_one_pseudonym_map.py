"""One pseudonym map per process (SPEC 8 Pseudonyms, 12 Data): the traits rating, the route
checks and the conversation give a registry name the same alias."""

from __future__ import annotations

import functools
import json
import threading

import test_conversation_worker
from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation import ConversationWorker
from roboemot.llm.pseudonyms import NameSource
from roboemot.persona.llm_traits import TraitCache, TraitsWorker
from test_conversation_worker import rigs  # noqa: F401  (fixture)
from test_llm_conformance import Fake
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
    alias = source.pseudonyms.to_alias(NAME)
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
    assert rig.worker._pseudonyms is source.pseudonyms
    rig.turn(f"I am {NAME}", language="en")
    sent = json.dumps([body for _, body in rig.server.requests], ensure_ascii=False)
    assert alias in sent and NAME not in sent
    assert source.pseudonyms.to_alias(NAME) == alias  # still the one alias


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
