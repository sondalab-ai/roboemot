"""``POST /api/routes/voice_preview`` (SPEC 8, Voice preview; SPEC 12): session and CSRF as for
the route check, the body checked, one preview at a time (never beside a check), refusals and
failures by code only, a WAV body, and the known voices in ``GET /api/routes``."""

from __future__ import annotations

import io
import json
import logging
import threading
import time
import wave

import pytest
from fake_server import REPLY_TEXT, FakeServer
from realtime_helpers import KEY
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics
from roboemot.llm.conformance import (
    RouteBudgetExhausted,
    RouteKeyUnavailable,
    RouteKindUnsupported,
)
from roboemot.llm.router import Router
from roboemot.realtime.preview import GEMINI_VOICES, QWEN3_VOICES, PreviewFailed
from roboemot.webui import make_server
from roboemot.webui.routes import PreviewRate, RouteBusy, RouteChecker
from test_webui import PIN, Client, FakeService

PATH = "/api/routes/voice_preview"
BODY = {"route": "rt", "voice": "Serena"}


def make_cfg(base_url: str = "http://127.0.0.1:1/v1", route=None, **rt) -> RoboEmotConfig:
    """One local Pollen-style route ``rt`` (or ``route``); no rate limit unless ``rt`` sets
    one, so a test may preview twice in a row."""
    route = route or {
        "name": "rt",
        "base_url": base_url,
        "model": "s2s",
        "kind": "realtime",
        "api_key": "none",
        "locality": "local",
    }
    return RoboEmotConfig(
        webui={"bind": "127.0.0.1", "port": 0},
        llm_routes=[route],
        voice={"realtime_route": {"mode": "direct", "preview_min_interval_s": 0, **rt}},
    )


def serve(tmp_path, cfg, routes):
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path, routes=routes)
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    assert client.login().status == 200
    return srv, client


class Port:
    """A routes port whose preview returns ``wav`` or raises ``exc``; records its calls."""

    def __init__(self, wav: bytes = b"RIFF", exc: Exception | None = None) -> None:
        self.wav, self.exc, self.calls = wav, exc, []

    def list_routes(self):
        return []

    def start(self, name):
        raise AssertionError("not a check")

    def preview(self, name, voice, wait_s):
        self.calls.append((name, voice, wait_s))
        if self.exc is not None:
            raise self.exc
        return self.wav


@pytest.fixture
def port_served(tmp_path):
    made = []

    def make(port):
        srv, client = serve(tmp_path, make_cfg(), port)
        made.append(srv)
        return client

    yield make
    for srv in made:
        srv.stop()


def test_it_needs_a_session_and_the_csrf_token(port_served, tmp_path):
    port = Port()
    client = port_served(port)
    assert client.request("POST", PATH, BODY, csrf=False).status == 403
    anonymous = Client(client.port)
    assert anonymous.request("POST", PATH, BODY).status == 401
    assert port.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"route": "rt", "voice": "Kore", "extra": 1},
        {"voice": "Kore"},
        {"route": "../x", "voice": "Kore"},
        {"route": "rt", "voice": 3},
        {"route": "rt", "voice": "Kore\n"},
        {"route": "rt", "voice": "x" * 65},
        {"route": "rt", "voice": "   "},
        {"route": "rt", "voice": ".Kore"},
        {"route": "rt", "voice": "-Kore"},
        {"route": "rt", "voice": "https://x.example/"},
    ],
)
def test_a_bad_body_is_refused_before_the_port(port_served, body):
    port = Port()
    assert port_served(port).request("POST", PATH, body).status == 400
    assert port.calls == []


@pytest.mark.parametrize(
    ("voice", "sent"), [("x" * 64, "x" * 64), ("", ""), ("Uncle_Fu ", "Uncle_Fu")]
)
def test_a_voice_of_64_characters_or_none_passes(port_served, voice, sent):
    port = Port()
    assert port_served(port).request("POST", PATH, {"route": "rt", "voice": voice}).status == 200
    assert port.calls[0][1] == sent


def test_a_preview_sooner_than_the_interval_is_429_with_its_wait(port_served):
    resp = port_served(Port(exc=PreviewRate(2.2))).request("POST", PATH, BODY)
    assert (resp.status, resp.json) == (429, {"error": "preview_rate"})
    assert resp.headers["Retry-After"] == "3"


def test_the_wait_is_the_request_work_share_of_the_deadline(port_served):
    port = Port()
    port_served(port).request("POST", PATH, BODY)
    web = make_cfg().webui
    assert port.calls[0][2] == pytest.approx(web.request_deadline_s * web.request_work_share)


def test_the_wav_comes_back_as_audio_with_the_voice_asked(port_served):
    port = Port(b"RIFF....WAVE")
    resp = port_served(port).request("POST", PATH, BODY)
    assert resp.status == 200 and resp.headers["Content-Type"] == "audio/wav"
    assert resp.body == b"RIFF....WAVE"
    ((name, voice, wait_s),) = port.calls
    assert (name, voice) == ("rt", "Serena") and 0 < wait_s < make_cfg().webui.request_deadline_s


@pytest.mark.parametrize(
    ("exc", "status", "body"),
    [
        (KeyError("rt"), 404, {"error": "route_unknown"}),
        (RouteBusy("rt"), 503, {"error": "route_busy"}),
        (RouteBudgetExhausted("rt"), 409, {"error": "route_budget"}),
        (RouteKeyUnavailable(f"rt: {KEY}"), 409, {"error": "route_secret"}),
        (
            PreviewFailed("no_audio"),
            502,
            {"error": "preview_failed", "code": "no_audio", "close_code": None},
        ),
        (
            PreviewFailed("closed", 1008),
            502,
            {"error": "preview_failed", "code": "closed", "close_code": 1008},
        ),
        (
            PreviewFailed("audio_cap"),
            502,
            {"error": "preview_failed", "code": "audio_cap", "close_code": None},
        ),
    ],
)
def test_failures_are_codes_only(port_served, exc, status, body):
    resp = port_served(Port(exc=exc)).request("POST", PATH, BODY)
    assert (resp.status, resp.json) == (status, body)
    assert KEY not in json.dumps(resp.json)


def test_a_real_preview_through_the_checker_returns_a_wav_and_logs_no_words(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer() as server:
        cfg = make_cfg(server.http_base())
        srv, client = serve(tmp_path, cfg, RouteChecker(lambda: cfg))
        try:
            resp = client.request("POST", PATH, BODY)
            routes = client.request("GET", "/api/routes").json
        finally:
            srv.stop()
    assert resp.status == 200, resp.json
    with wave.open(io.BytesIO(resp.body)) as w:
        assert w.getframerate() == 16000 and w.getnframes() > 0
    assert routes["voices"] == {
        "gemini_live": list(GEMINI_VOICES),
        "openai_realtime": list(QWEN3_VOICES),
    }
    assert routes["routes"][0]["last_check"] is None  # a preview is no check: nothing is kept
    assert REPLY_TEXT not in caplog.text


def idle(checker: RouteChecker, wait_s: float = 5.0) -> bool:
    """Whether the checker's preview flag clears within ``wait_s``."""
    end = time.monotonic() + wait_s
    while checker._previewing and time.monotonic() < end:
        time.sleep(0.01)
    return not checker._previewing


def test_one_preview_at_a_time_and_never_beside_a_check(monkeypatch):
    cfg = make_cfg()
    checker = RouteChecker(lambda: cfg)
    started, release = threading.Event(), threading.Event()

    def slow(*_a, ready, **_kw):
        started.set()
        release.wait(5)
        ready(b"RIFF")
        return b"RIFF"

    monkeypatch.setattr("roboemot.webui.routes.preview_voice", slow)
    out: list[bytes] = []
    first = threading.Thread(target=lambda: out.append(checker.preview("rt", "Kore", 5)))
    first.start()
    assert started.wait(5)
    with pytest.raises(RouteBusy):
        checker.preview("rt", "Kore", 5)
    with pytest.raises(RouteBusy):
        checker.start("rt")
    release.set()
    first.join(5)
    assert out == [b"RIFF"]
    assert idle(checker)  # the flag clears once the session is closed
    assert checker.preview("rt", "Kore", 5) == b"RIFF"


def test_a_preview_past_its_wait_is_a_timeout_and_the_flag_clears_after(monkeypatch):
    cfg = make_cfg()
    checker = RouteChecker(lambda: cfg)
    release = threading.Event()

    def stuck(*_a, ready, **_kw):
        release.wait(5)
        raise PreviewFailed("timeout")

    monkeypatch.setattr("roboemot.webui.routes.preview_voice", stuck)
    with pytest.raises(PreviewFailed) as exc:
        checker.preview("rt", "Kore", 0.1)
    assert exc.value.code == "timeout" and checker._previewing
    release.set()
    assert idle(checker)


def test_a_chat_route_and_an_unknown_one_are_refused(tmp_path):
    cfg = RoboEmotConfig(
        llm_routes=[
            {"name": "c", "base_url": "http://127.0.0.1:1/v1", "model": "m", "api_key": "none"}
        ]
    )
    checker = RouteChecker(lambda: cfg)
    with pytest.raises(RouteKindUnsupported):
        checker.preview("c", "Kore", 1)
    with pytest.raises(KeyError):
        checker.preview("nope", "Kore", 1)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_previews_of_one_route_are_spaced_by_the_interval_others_are_not(monkeypatch):
    two = {"name": "two", "base_url": "http://127.0.0.1:1/v1", "model": "s2s", "kind": "realtime"}
    cfg = make_cfg(preview_min_interval_s=3)
    cfg = cfg.model_copy(
        update={"llm_routes": [*cfg.llm_routes, make_cfg(route=two).llm_routes[0]]}
    )
    clock = Clock()
    checker = RouteChecker(lambda: cfg, clock=clock)
    monkeypatch.setattr(
        "roboemot.webui.routes.preview_voice", lambda *_a, ready, **_k: ready(b"RIFF")
    )
    assert checker.preview("rt", "Kore", 5) == b"RIFF"
    assert idle(checker)
    clock.now += 1.0
    with pytest.raises(PreviewRate) as exc:
        checker.preview("rt", "Kore", 5)
    assert exc.value.retry_after_s == pytest.approx(2.0)
    assert checker.preview("two", "Kore", 5) == b"RIFF"  # per route
    assert idle(checker)
    clock.now += 2.0
    assert checker.preview("rt", "Kore", 5) == b"RIFF"


def test_the_preview_gives_up_a_margin_before_the_callers_wait(monkeypatch):
    cfg = make_cfg(preview_reply_margin_s=0.4)
    seen: list[float] = []

    def failing(*_a, ready, deadline_s, **_k):
        seen.append(deadline_s)
        exc = PreviewFailed("connect_timeout")
        ready(exc)
        raise exc

    monkeypatch.setattr("roboemot.webui.routes.preview_voice", failing)
    with pytest.raises(PreviewFailed) as exc:
        RouteChecker(lambda: cfg).preview("rt", "Kore", 2.0)
    assert exc.value.code == "connect_timeout"  # its own code, not the caller's timeout
    assert seen == [pytest.approx(1.6)]


def test_a_running_check_makes_a_preview_busy(monkeypatch):
    cfg = make_cfg()
    checker = RouteChecker(lambda: cfg)
    started, release = threading.Event(), threading.Event()

    def slow_check(*_a, **_k):
        started.set()
        release.wait(5)
        raise RuntimeError("done")

    monkeypatch.setattr("roboemot.webui.routes.check_realtime_route", slow_check)
    checker.start("rt")
    assert started.wait(5)
    with pytest.raises(RouteBusy):
        checker.preview("rt", "Kore", 1)
    release.set()
    checker.thread.join(5)
    assert not checker._previewing


@pytest.mark.parametrize("broken", ["session_update", "client_factory"])
def test_the_busy_flag_clears_when_the_session_cannot_be_built(monkeypatch, broken, caplog):
    import roboemot.realtime.client as client_module
    import roboemot.realtime.preview as preview_module

    def boom(*_a, **_k):
        raise RuntimeError(KEY)

    if broken == "client_factory":
        monkeypatch.setattr(client_module, "RealtimeClient", boom)
    else:
        monkeypatch.setattr(preview_module, "session_update", boom)
    cfg = make_cfg()
    checker = RouteChecker(lambda: cfg)
    with pytest.raises(PreviewFailed) as exc:
        checker.preview("rt", "Kore", 5)
    assert exc.value.code == "failed" and idle(checker)
    assert KEY not in caplog.text


def test_a_cap_reached_mid_stream_in_the_second_tool_round_fails_and_clears_busy(gemini):
    srv = gemini([["call:note_tone", "audio"]], chunks=40, chunk_ms=100)
    route = {
        "name": "gemini",
        "base_url": srv.base_url(),
        "model": "gemini-3.8-live",
        "kind": "realtime",
        "protocol": "gemini_live",
        "api_key": "${secret:gemini_key}",
        "locality": "local",
    }
    cfg = make_cfg(route=route, tone_tool=True, preview_max_bytes=65_536, send_timeout_s=1)
    router = Router(cfg.llm_routes, cfg.safety, secrets=lambda _name: KEY)
    checker = RouteChecker(lambda: cfg, router=lambda: router)
    with pytest.raises(PreviewFailed) as exc:
        checker.preview("gemini", "Kore", 8)
    assert exc.value.code == "audio_cap"
    assert srv.tool_responses, "the cap came in the round after the tool call"
    assert idle(checker)
