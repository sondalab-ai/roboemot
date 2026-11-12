"""The realtime check on a ``gemini_live`` route against the TLS fake Gemini server (SPEC 8,
Realtime check and Gemini Live wire; SPEC 12, 12A).

Direct mode only (never a provider allocation), the setup with the turn's tools, a tool round
answered and followed, the item delete reported as emulated, the late usage correction, a 1008
refusal by its close code only, a slow setup within the route's own connect budget, and the
session closed by the check itself. Never the key, the server's reason text or the reply.
"""

from __future__ import annotations

import json
import logging

from gemini_server import REFUSAL, REPLY
from realtime_helpers import KEY
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.router import Router
from roboemot.realtime import check
from roboemot.realtime.session import Allocation, DirectSession
from roboemot.realtime.turn import SKIP_TOOL, TONE_TOOL

CLOSE_NORMAL = 1000


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


def test_the_running_workers_tools_go_in_the_setup(gemini):
    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()))
    asked: list[float] = []
    tool = {"type": "function", "name": "only_this", "description": "", "parameters": {}}

    def tools(wait_s: float):
        asked.append(wait_s)
        return [tool]

    report = run(cfg, tools=tools)
    assert report.ok, report.to_dict()
    names = [d["name"] for d in srv.setups[0]["tools"][0]["functionDeclarations"]]
    assert names == ["only_this"] and report.tools_sent == 1
    assert asked == [cfg.voice.realtime_route.connect_timeout_s]  # the connect budget


def test_no_tools_from_the_worker_sends_no_setup(gemini):
    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()))
    report = run(cfg, tools=lambda _wait_s: None)
    assert not report.ok and report.failure == check.SETUP_UNREADY
    assert srv.requests == []


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


def test_a_missing_late_total_keeps_the_estimate_and_is_no_failure(gemini):
    srv = gemini([["audio"]], turn_complete_s=30.0)
    cfg = make_cfg(gemini_route(srv.base_url()))
    router = make_router(cfg)
    report = run(cfg, router, deadline_s=2.0)
    assert report.ok, report.to_dict()
    assert report.usage_reported is None
    assert report.charged > 0 and report.charged == router.used("gemini")


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
    tool = {"type": "function", "name": "only_this", "description": "", "parameters": {}}
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
        realtime_tools=lambda _wait_s: [tool],
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
