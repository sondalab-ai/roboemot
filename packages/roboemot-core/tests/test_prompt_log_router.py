"""SPEC 8, 11A, 12: the router's tap into the Live tab's LLM prompts ring. A cloud chat request is
recorded as it leaves (trimmed, aliased); a local route is not; a router the hub rebuilds on a
config reload keeps the ring."""

from __future__ import annotations

import json

from roboemot.config.schema import LiveConfig, RoboEmotConfig
from roboemot.llm.hub import RouterHub
from roboemot.llm.prompt_log import PromptRing
from roboemot.llm.pseudonyms import NameSource
from roboemot.runtime.host import RuntimeHost
from test_llm_router import Server, route, router

KEY = "sk-prompts-SECRET"


def ring() -> PromptRing:
    return PromptRing(LiveConfig(enabled=True, prompts=True), wall=lambda: 1.0)


async def ask(r, text="ciao", **kw):
    return [ev async for ev in r.stream([{"role": "user", "content": text}], **kw)]


async def test_a_cloud_request_is_recorded_with_its_model_and_route():
    prompts = ring()
    r = router(Server(), [route("cloud", "api.example.com", api_key=KEY)], prompts=prompts)
    tools = [{"type": "function", "function": {"name": "nod", "parameters": {}}}]
    await ask(r, tools=tools)
    (entry,) = prompts.snapshot(0)["entries"]
    assert (entry["route"], entry["via"], entry["label"]) == ("cloud", "m", "chat.completions")
    assert entry["text"] == "[user] ciao\n[tools] nod"
    dumped = json.dumps(prompts.snapshot(0))
    assert KEY not in dumped and "api.example.com" not in dumped


async def test_a_local_route_is_not_recorded():
    prompts = ring()
    r = router(Server(), [route("lan", "lan.local")], prompts=prompts)
    await ask(r)
    assert prompts.size == 0


async def test_the_entry_holds_the_aliased_messages():
    prompts = ring()
    cfg = RoboEmotConfig(privacy={"pseudonyms_to_cloud": True}, conversation={"language": "en"})
    snapshot = NameSource(lambda: ["Marta"]).for_config(cfg)
    assert snapshot is not None
    alias = snapshot.to_alias("Marta")
    assert alias != "Marta"
    srv = Server()
    r = router(srv, [route("cloud", "api.example.com")], prompts=prompts)
    await ask(r, "sono Marta", pseudonyms=snapshot)
    text = prompts.snapshot(0)["entries"][0]["text"]
    assert text == f"[user] sono {alias}" and "Marta" not in text


async def test_the_entry_holds_the_trimmed_history():
    prompts = ring()
    r = router(
        Server(), [route("cloud", "api.example.com", context_budget_tokens=256)], prompts=prompts
    )
    history = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 else "assistant", "content": f"turn{i} " + "x" * 400}
        for i in range(10)
    ]
    _ = [ev async for ev in r.stream(history)]
    text = prompts.snapshot(0)["entries"][0]["text"]
    assert text.startswith("[system] sys") and "turn0 " not in text and "turn9 " in text


async def test_a_router_the_hub_rebuilds_still_records(tmp_path):
    host = RuntimeHost()
    host.prompts = ring()
    cloud = {"name": "cloud", "base_url": "https://api.example.com/v1", "model": "m"}
    cfg = RoboEmotConfig(data_dir=str(tmp_path), llm_routes=[cloud])
    monitor = None
    hub = RouterHub(host.make_router(cfg, monitor), lambda c: host.make_router(c, monitor))
    first = hub.router
    rebuilt = hub.sync(
        RoboEmotConfig(data_dir=str(tmp_path), llm_routes=[{**cloud, "model": "m2"}])
    )
    assert rebuilt is not first and rebuilt.prompts is host.prompts
    srv = Server()
    rebuilt.http_client_factory = router(srv, []).http_client_factory
    await ask(rebuilt)
    (entry,) = host.prompts.snapshot(0)["entries"]
    assert entry["via"] == "m2"
