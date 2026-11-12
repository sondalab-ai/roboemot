"""The realtime check on a ``gemini_live`` route against the TLS fake Gemini server (SPEC 8,
Realtime check and Gemini Live wire; SPEC 12, 12A).

Direct mode only (never a provider allocation), the setup with the turn's tools, a tool round
answered and followed, the item delete reported as emulated, the late usage correction, a 1008
refusal by its close code only, a slow setup within the route's own connect budget, and the
session closed by the check itself. Never the key, the server's reason text or the reply.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import sys
import threading
import time

import pytest
from gemini_server import REFUSAL, REPLY
from realtime_helpers import KEY
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.conformance import RouteBudgetExhausted
from roboemot.llm.router import Router
from roboemot.realtime import check, protocol
from roboemot.realtime.session import Allocation, DirectSession
from roboemot.realtime.turn import SKIP_TOOL, TONE_TOOL
from roboemot.skills.defaults import LOOK_CATEGORIES

CLOSE_NORMAL = 1000
GATE_OPEN_S = 0.5  # when the held turn total is let out
WAIT_SLACK_S = 0.8  # what a bounded wait may overrun on a loaded CI runner
TOOL = {"type": "function", "name": "only_this", "description": "", "parameters": {}}


def gemini_route(base_url: str, **kw: object) -> LLMRoute:
    return LLMRoute(
        name="gemini",
        base_url=base_url,
        model="gemini-3.8-live",
        kind="realtime",
        protocol="gemini_live",
        api_key="${secret:gemini_key}",
        locality="local",
        **kw,
    )


def make_cfg(route: LLMRoute, *, mode: str = "direct", **rt: object) -> RoboEmotConfig:
    return RoboEmotConfig(llm_routes=[route], voice={"realtime_route": {"mode": mode, **rt}})


def make_router(cfg: RoboEmotConfig) -> Router:
    return Router(cfg.llm_routes, cfg.safety, secrets=lambda _name: KEY)


class Allocator:
    """A platform provider that must never be asked for a Gemini route."""

    key_hosts: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.asked = 0

    async def allocate(self, request):
        self.asked += 1
        return Allocation("wss://alloc.example.net/session")


def run(cfg: RoboEmotConfig, router: Router | None = None, provider=None, **kw):
    return check.check_realtime_route(
        cfg.llm_routes[0],
        cfg,
        router=router or make_router(cfg),
        provider=provider if provider is not None else DirectSession(),
        **kw,
    )


def leaks(report: check.RealtimeReport, caplog) -> list[str]:
    text = json.dumps(report.to_dict()) + caplog.text
    return [s for s in (KEY, REFUSAL, REPLY) if s in text]


def test_a_gemini_check_passes_with_the_delete_emulated_and_closes_its_session(gemini, caplog):
    caplog.set_level(logging.DEBUG)
    srv = gemini([["audio"]], usage=1785)
    cfg = make_cfg(gemini_route(srv.base_url()))
    router = make_router(cfg)
    report = run(cfg, router)
    assert report.ok, report.to_dict()
    assert report.protocol == "gemini_live" and report.mode == "direct"
    assert report.allocator_host is None and report.connect_host == "127.0.0.1"
    assert report.header_to_connect_host == "sent"
    headers = {k.lower(): v for k, v in srv.requests[0]["headers"].items()}
    assert headers["x-goog-api-key"] == KEY and "authorization" not in headers
    assert report.update_seen is True and report.setup_ms is not None
    assert report.first_audio_ms is not None and report.close_code is None
    assert report.item_delete == "emulated"
    assert {c.name: c.status for c in report.checks} == {
        check.CONNECT: "pass",
        check.FIRST_AUDIO: "pass",
        check.ITEM_DELETE: "skip",  # the codec's ack: not a server confirmation, not a failure
    }
    assert report.checks[-1].code == "item_delete_emulated"
    # the late total replaces the estimate: charged once, in full
    assert report.usage_reported == 1785 == report.charged == router.used("gemini")
    assert srv.client_closes == [CLOSE_NORMAL] and srv.idle_closes == 0
    assert leaks(report, caplog) == []


def test_the_setup_carries_the_tools_a_turn_offers(gemini):
    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()), skip_tool=True, tone_tool=True)
    report = run(cfg)
    names = [d["name"] for d in srv.setups[0]["tools"][0]["functionDeclarations"]]
    assert SKIP_TOOL in names and TONE_TOOL in names
    assert {"look_at", "express_emotion", "stop_talking"} <= set(names)  # offline default skills
    assert report.tools_sent == len(names)


def test_the_given_tools_go_in_the_setup(gemini):
    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()))
    report = run(cfg, tools=lambda: [TOOL])
    assert report.ok, report.to_dict()
    names = [d["name"] for d in srv.setups[0]["tools"][0]["functionDeclarations"]]
    assert names == ["only_this"] and report.tools_sent == 1
    assert check.TOOLS not in [c.name for c in report.checks]


def test_tools_that_cannot_be_built_send_no_setup(gemini, caplog):
    def broken():
        raise RuntimeError(KEY)  # an exception's text is never logged

    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()))
    for tools in (lambda: None, broken):
        report = run(cfg, tools=tools)
        assert not report.ok and report.failure == check.SETUP_UNREADY
        assert report.checks[0].params == {"failure": check.SETUP_UNREADY}
    assert srv.requests == [] and KEY not in caplog.text


def test_a_setup_without_tools_is_an_advisory_warning(gemini):
    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()))
    report = run(cfg, tools=list)
    assert report.ok and report.tools_sent == 0
    tools = [c for c in report.checks if c.name == check.TOOLS]
    assert [(c.status, c.code) for c in tools] == [("skip", "no_tools")]


@pytest.mark.parametrize("rounds", [0, 1])
def test_calls_still_open_past_the_round_cap_are_reported(gemini, rounds):
    calls = ["call:note_tone", "call:note_tone"]  # a call again after each answer
    srv = gemini([calls], call_args={"note_tone": {"tone": "neutral"}})
    cfg = RoboEmotConfig(
        llm_routes=[gemini_route(srv.base_url())],
        conversation={"max_tool_rounds": rounds},
        voice={"realtime_route": {"mode": "direct", "tone_tool": True}},
    )
    started = time.monotonic()
    report = run(cfg)
    tools = [c for c in report.checks if c.name == check.TOOLS]
    assert [(c.status, c.code, c.params) for c in tools] == [
        ("skip", "tool_rounds_exceeded", {"rounds": rounds})
    ]
    assert len(srv.tool_responses) == rounds
    assert report.usage_reported is None  # no usage wait: Gemini still waits on a call
    assert time.monotonic() - started < cfg.voice.realtime_route.check_usage_wait_s


def test_a_tool_call_is_answered_and_the_continuation_followed(gemini):
    srv = gemini([["call:note_tone", "audio"]], call_args={"note_tone": {"tone": "neutral"}})
    cfg = make_cfg(gemini_route(srv.base_url()))
    router = make_router(cfg)
    report = run(cfg, router)
    assert report.ok, report.to_dict()
    assert report.first_audio_ms is not None
    assert srv.tool_responses == [
        [{"id": "fc_0_note_tone", "name": "note_tone", "response": {"ok": True}}]
    ]
    assert report.usage_reported == 1785 == router.used("gemini")
    assert report.charged == 1785  # one total for both responses: their estimates came off


def test_a_1008_refusal_is_shown_by_its_close_code_only(gemini, caplog):
    caplog.set_level(logging.DEBUG)
    srv = gemini(refuse=1)
    cfg = make_cfg(gemini_route(srv.base_url()))
    report = run(cfg)
    assert not report.ok
    assert report.failure == "connect_failed" and report.close_code == 1008
    assert report.checks[0].params == {"failure": "connect_failed"}
    assert report.item_delete == "not_run" and report.setup_ms is None
    assert report.charged == 0
    assert leaks(report, caplog) == []


def test_a_slow_setup_passes_within_the_routes_own_connect_budget(gemini):
    srv = gemini([["audio"]], setup_delay_s=1.2)
    cfg = make_cfg(gemini_route(srv.base_url(), connect_timeout_s=5.0), connect_timeout_s=1.0)
    report = run(cfg)
    assert report.ok, report.to_dict()
    assert report.setup_ms is not None and report.setup_ms >= 1200


def test_a_gemini_route_never_asks_a_session_provider(gemini):
    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()), mode="platform")
    allocator = Allocator()
    report = run(cfg, provider=allocator)
    assert report.ok, report.to_dict()
    assert allocator.asked == 0
    assert report.mode == "direct" and report.allocator_host is None
    assert report.header_to_connect_host == "sent"


def test_a_missing_late_total_keeps_the_estimate_and_the_wait_ends_at_its_bound(gemini):
    srv = gemini([["audio"]], turn_complete_s=30.0)
    cfg = make_cfg(gemini_route(srv.base_url()), check_usage_wait_s=1.0)
    router = make_router(cfg)
    run_ = check._Run.__dict__["_await_usage"]
    spent: list[float] = []

    def timed(self):
        started = time.monotonic()
        run_(self)
        spent.append(time.monotonic() - started)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(check._Run, "_await_usage", timed)
        report = run(cfg, router)
    assert report.ok, report.to_dict()
    assert report.usage_reported is None
    assert report.charged > 0 and report.charged == router.used("gemini")
    assert len(spent) == 1 and 1.0 <= spent[0] < 1.0 + WAIT_SLACK_S


def test_a_total_held_past_the_answer_ends_the_wait_when_it_comes(gemini):
    gate = threading.Event()
    srv = gemini([["audio"]], usage_gate=gate, turn_complete_s=0.0)
    cfg = make_cfg(gemini_route(srv.base_url()), check_usage_wait_s=10.0)
    router = make_router(cfg)
    threading.Timer(GATE_OPEN_S, gate.set).start()
    started = time.monotonic()
    report = run(cfg, router)
    assert report.usage_reported == 1785 == report.charged == router.used("gemini")
    assert time.monotonic() - started < GATE_OPEN_S + 2 * WAIT_SLACK_S


def test_a_close_during_the_turn_shows_its_code(gemini):
    srv = gemini([["audio"]], delay_s=2.0, idle_close_s=0.3)
    cfg = make_cfg(gemini_route(srv.base_url()))
    report = run(cfg)
    assert not report.ok and report.update_seen is True
    assert report.failure == "closed_by_server" and report.close_code == 1008


def test_a_spent_budget_refuses_the_check_before_anything_is_sent(gemini):
    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()))
    router = make_router(cfg)
    router.charge("gemini", cfg.llm_routes[0].daily_token_budget)
    with pytest.raises(RouteBudgetExhausted):
        run(cfg, router)
    assert srv.requests == []


def test_a_pollen_report_has_the_new_fields_neutral():
    from fake_server import FakeServer

    with FakeServer(delete_reply="deleted") as server:
        route = LLMRoute(
            name="rt", base_url=server.http_base(), model="s2s", kind="realtime",
            api_key="none", locality="local",
        )  # fmt: skip
        cfg = make_cfg(route)
        data = run(cfg).to_dict()
    assert data["protocol"] == "openai_realtime" and data["ok"] is True
    assert data["tools_sent"] == 0 and data["usage_reported"] is None
    assert data["close_code"] is None and data["setup_ms"] >= 0


# ------------------------------------------------------------------ usage, by scripted events


class ScriptClient:
    """A client double: ``response.create`` makes it deliver the next scripted events."""

    def __init__(self, route, settings, provider, *, on_event, on_close, **_kw):
        self.on_event, self.reason, self.scripts = on_event, None, list(SCRIPTS)

    def connect(self, _update, **_kw):
        return True

    def send(self, event):
        if event.get("type") == protocol.ITEM_DELETE:
            self.on_event(protocol.ItemDeleted(protocol.ITEM_DELETED, event["item_id"], True))
        if event.get("type") == protocol.RESPONSE_CREATE and self.scripts:
            for ev in self.scripts.pop(0):
                self.on_event(ev)
        done = concurrent.futures.Future()
        done.set_result(True)
        return done

    def snapshot(self):
        return {"endpoint": "wss://127.0.0.1:1", "key_sent": True}

    def close(self):
        pass


SCRIPTS: list[list[protocol.ServerEvent]] = []


def scripted(monkeypatch, *responses):
    monkeypatch.setattr(sys.modules[__name__], "SCRIPTS", list(responses))
    cfg = make_cfg(gemini_route("wss://127.0.0.1:1/ws"), check_usage_wait_s=0.5)
    router = make_router(cfg)
    report = run(cfg, router, tools=lambda: [TOOL], client_factory=ScriptClient)
    return report, router


def answer(rid: str, *, call: bool = False) -> list[protocol.ServerEvent]:
    out: list[protocol.ServerEvent] = [protocol.ResponseCreated(protocol.RESPONSE_CREATED, rid)]
    if call:
        out.append(protocol.FunctionCall(protocol.FUNCTION_CALL_DONE, rid, "c1", "note_tone", "{}"))
    else:
        out.append(protocol.AudioDelta(protocol.AUDIO_DELTA, rid, "o1", b"\0\0"))
    return [*out, protocol.ResponseDone(protocol.RESPONSE_DONE, rid, "completed")]


def usage(total: int, *ids: str) -> protocol.UsageReport:
    return protocol.UsageReport(protocol.USAGE_REPORT, ids, total)


def test_a_report_naming_no_response_never_charges_on_top_of_the_estimates(monkeypatch):
    report, router = scripted(monkeypatch, [*answer("r1"), usage(500)])
    assert report.usage_reported == 500 == report.charged == router.used("gemini")


def test_a_repeated_report_charges_nothing_more(monkeypatch):
    report, router = scripted(monkeypatch, [*answer("r1"), usage(500, "r1"), usage(500, "r1")])
    assert report.charged == 500 == router.used("gemini")


def test_a_report_before_the_response_done_is_not_charged_twice(monkeypatch):
    events = answer("r1")
    report, router = scripted(monkeypatch, [*events[:-1], usage(500, "r1"), events[-1]])
    assert report.charged == 500 == router.used("gemini")


def test_one_total_over_a_tool_round_charges_the_sum_once(monkeypatch):
    report, router = scripted(
        monkeypatch, answer("r1", call=True), [*answer("r2"), usage(900, "r1", "r2")]
    )
    assert report.ok, report.to_dict()
    assert report.charged == 900 == router.used("gemini")


def test_the_wait_ends_at_the_first_report_after_the_last_response(monkeypatch):
    started = time.monotonic()
    report, _ = scripted(monkeypatch, [*answer("r1"), usage(500, "r1")])
    assert report.usage_reported == 500
    assert time.monotonic() - started < WAIT_SLACK_S


# ------------------------------------------------------------------ offline tools


def names(tools) -> list[str]:
    return [t["name"] for t in tools]


def test_offline_tools_follow_the_round_cap_the_tool_switches_and_the_skill_list():
    cfg = RoboEmotConfig(voice={"realtime_route": {"skip_tool": True, "tone_tool": True}})
    assert {SKIP_TOOL, TONE_TOOL, "look_at"} <= set(names(check.offline_tools(cfg)))
    capped = cfg.model_copy(update={"conversation": cfg.conversation.model_copy(
        update={"max_tool_rounds": 0})})  # fmt: skip
    assert names(check.offline_tools(capped)) == [SKIP_TOOL, TONE_TOOL]
    plain = RoboEmotConfig(conversation={"llm_skills": ["look_at"]})
    assert names(check.offline_tools(plain)) == ["look_at"]


def test_offline_tools_act_on_nothing_and_offer_no_name_in_view(monkeypatch):
    acted: list[str] = []

    class Spy(check.SkillEnv):
        def __post_init__(self):
            play, look = self.play, self.look_at
            self.play = lambda n: acted.append("play") or play(n)
            self.look_at = lambda t: acted.append("look_at") or look(t)
            self.look_names = lambda: acted.append("names") or []

    monkeypatch.setattr(check, "SkillEnv", Spy)
    tools = check.offline_tools(RoboEmotConfig(), gestures=lambda: ["nod"])
    look = next(t for t in tools if t["name"] == "look_at")
    assert look["parameters"]["properties"]["target"]["enum"] == list(LOOK_CATEGORIES)
    play = next(t for t in tools if t["name"] == "play")
    assert play["parameters"]["properties"]["gesture"]["enum"] == ["nod"]
    assert "play" not in acted and "look_at" not in acted


def test_the_cli_checks_a_gemini_route_without_a_session_provider(
    gemini, tmp_path, capsys, monkeypatch
):
    import yaml
    from roboemot import cli
    from roboemot.realtime import route as realtime_route

    def no_provider(*_args):
        raise AssertionError("a gemini_live route never builds a session provider")

    monkeypatch.setattr(realtime_route, "select_provider", no_provider)
    srv = gemini([["audio"]])
    entry = {"name": "gemini", "base_url": srv.base_url(), "model": "gemini-3.8-live"}
    entry |= {"kind": "realtime", "protocol": "gemini_live", "locality": "local", "api_key": "none"}
    path = tmp_path / "app.yaml"
    body = {"llm_routes": [entry], "voice": {"realtime_route": {"mode": "platform"}}}
    path.write_text(yaml.safe_dump(body))
    args = ["realtime", "check", "gemini", "--app", str(path), "--robot", "no-such-robot"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "protocol                gemini_live" in out
    assert "item_delete             emulated" in out
    assert "usage_reported          1785" in out
    assert "route gemini (gemini-3.8-live): ok" in out


def test_the_routes_tab_passes_the_workers_tools_and_shows_the_running_session(gemini):
    from roboemot.webui.routes import RouteChecker

    srv = gemini([["audio"]])
    route = gemini_route(srv.base_url())
    cfg = RoboEmotConfig(
        llm_routes=[route],
        voice={"realtime_route": {"mode": "direct", "enabled": True, "route": "gemini"}},
    )
    snapshot = {
        "state": "closed", "reason": "server_refused", "idle_closes": 2, "go_aways": 1,
        "usage_reports": 4, "items_delete_emulated": 3, "audio_sent": 9,
        "wire": {"session": {"held_discarded": 0, "calls_skipped": 1},
                 "total": {"held_discarded": 2, "calls_skipped": 1}},
    }  # fmt: skip
    checker = RouteChecker(
        lambda: cfg,
        None,
        router=lambda: make_router(cfg),
        realtime_tools=lambda: [TOOL],
        realtime_status=lambda: snapshot,
    )
    checker.start("gemini")
    checker.thread.join(15)
    entry = checker.list_routes()[0]
    names = [d["name"] for d in srv.setups[0]["tools"][0]["functionDeclarations"]]
    assert names == ["only_this"]
    assert entry["last_check"]["item_delete"] == "emulated"
    assert entry["session"] == {
        "state": "closed", "reason": "server_refused", "idle_closes": 2, "go_aways": 1,
        "usage_reports": 4, "items_delete_emulated": 3,
        "wire_session": {"calls_skipped": 1},
        "wire_total": {"held_discarded": 2, "calls_skipped": 1},
    }  # fmt: skip
    assert KEY not in json.dumps(entry)


def test_the_routes_tab_shows_no_session_while_realtime_is_off():
    from roboemot.webui.routes import RouteChecker

    cfg = make_cfg(gemini_route("wss://127.0.0.1:1/ws"))
    checker = RouteChecker(lambda: cfg, None, realtime_status=lambda: {"state": "connected"})
    assert checker.list_routes()[0]["session"] is None


def test_the_routes_tab_session_is_none_when_the_status_fails_or_is_missing(caplog):
    from roboemot.webui.routes import RouteChecker

    def broken():
        raise RuntimeError(KEY)

    cfg = RoboEmotConfig(
        llm_routes=[gemini_route("wss://127.0.0.1:1/ws")],
        voice={"realtime_route": {"mode": "direct", "enabled": True, "route": "gemini"}},
    )
    for status in (broken, lambda: None):
        checker = RouteChecker(lambda: cfg, None, realtime_status=status)
        assert checker.list_routes()[0]["session"] is None
    assert KEY not in caplog.text


def test_the_runtime_wiring_builds_offline_tools_and_reads_the_route_snapshot():
    from types import SimpleNamespace

    from roboemot.config import ConfigStack
    from roboemot.runtime import webui as runtime_webui

    stack = ConfigStack()
    p = SimpleNamespace(expression=SimpleNamespace(playable=lambda: ["nod", "shake"]))
    tools = runtime_webui.check_tools(stack, p)()
    play = next(t for t in tools if t["name"] == "play")
    assert play["parameters"]["properties"]["gesture"]["enum"] == ["nod", "shake"]
    host = SimpleNamespace(realtime_route=None)
    status = runtime_webui.route_status(host)
    assert status() is None
    host.realtime_route = SimpleNamespace(snapshot=lambda: {"state": "connected"})
    assert status() == {"state": "connected"}
