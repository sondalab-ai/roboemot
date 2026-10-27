"""RealtimeClient against the fake server (SPEC 8, Realtime speech route; SPEC 12, 12A)."""

from __future__ import annotations

import concurrent.futures
import logging

import pytest
from fake_server import ERROR_MESSAGE, PCM, REPLY_TEXT, FakeServer
from realtime_helpers import UPDATE, route, settings, wait_until
from roboemot.realtime import protocol
from roboemot.realtime.client import RealtimeClient
from roboemot.realtime.session import DirectSession


def client(server: FakeServer, events: list, **kw) -> RealtimeClient:
    rkw = {k: kw.pop(k) for k in ("first_token_timeout_s", "stall_timeout_s") if k in kw}
    return RealtimeClient(
        route(server.http_base(), **rkw),
        settings(**kw),
        DirectSession(),
        auth_headers=dict,
        on_event=events.append,
    )


@pytest.fixture
def events() -> list:
    return []


def test_connect_session_update_and_a_response(events, caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer() as server:
        c = client(server, events)
        try:
            assert c.connect(UPDATE)
            assert c.state == "connected"
            assert server.received[0]["type"] == "session.update"
            assert server.requests[0]["path"] == "/v1/realtime"
            assert [e.type for e in events[:2]] == ["session.created", "session.updated"]
            assert c.send_audio(PCM).result(2)
            assert c.send(protocol.response_create()).result(2)
            assert wait_until(lambda: any(isinstance(e, protocol.ResponseDone) for e in events))
            audio = [e for e in events if isinstance(e, protocol.AudioDelta)]
            assert audio and audio[0].pcm == PCM
            done = next(e for e in events if isinstance(e, protocol.ResponseDone))
            assert done.total_tokens == 42 and done.latency_s == {"stt": 0.2, "e2e": 1.1}
            assert server.received[1]["type"] == "input_audio_buffer.append"
            snap = c.snapshot()
            assert snap["state"] == "connected"
            assert snap["endpoint"] == f"ws://127.0.0.1:{server.port}"
            assert snap["audio_bytes_sent"] == len(PCM)
            assert snap["audio_bytes_received"] == len(PCM)
            assert snap["key_sent"] is False and snap["local"] is True
        finally:
            c.close()
    assert REPLY_TEXT not in caplog.text
    assert c._thread is None


def test_a_send_without_a_socket_is_dropped(events):
    with FakeServer() as server:
        c = client(server, events)
        c.start()
        try:
            assert c.send(protocol.response_cancel()).result(2) is False
            assert c.snapshot()["dropped"] == 1
        finally:
            c.close()


def test_a_server_error_before_session_updated_refuses_the_session(events, caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer("error_first") as server:
        c = client(server, events)
        try:
            assert c.connect(UPDATE) is False
            assert c.reason == "server_error"
            err = next(e for e in events if isinstance(e, protocol.ServerError))
            assert err.code == "bad_session"
        finally:
            c.close()
    assert ERROR_MESSAGE not in caplog.text
    assert "secret request" not in repr(c.snapshot())


def test_no_session_updated_within_the_connect_timeout(events):
    with FakeServer("no_session") as server:
        c = client(server, events, connect_timeout_s=0.4)
        try:
            assert c.connect(UPDATE) is False
            assert c.reason == "connect_timeout"
            assert c.snapshot()["failures"] == {"connect_timeout": 1}
        finally:
            c.close()


def test_nothing_listening_fails_to_connect(events):
    with FakeServer() as server:
        base = server.http_base()
    c = RealtimeClient(
        route(base), settings(), DirectSession(), auth_headers=dict, on_event=events.append
    )
    try:
        assert c.connect(UPDATE) is False
        assert c.reason == "connect_failed"
    finally:
        c.close()


def test_the_server_closing_mid_stream(events):
    with FakeServer("close_mid") as server:
        c = client(server, events)
        try:
            assert c.connect(UPDATE)
            c.send(protocol.response_create()).result(2)
            assert wait_until(lambda: c.state == "closed")
            assert c.reason == "closed_by_server"
            assert any(isinstance(e, protocol.AudioDelta) for e in events)
            assert c.send(protocol.response_cancel()).result(2) is False
        finally:
            c.close()


@pytest.mark.parametrize("mode", ["stall", "silent"])
def test_a_stalled_response_closes_the_socket(events, mode):
    with FakeServer(mode) as server:
        c = client(server, events, first_token_timeout_s=0.3, stall_timeout_s=0.3)
        try:
            assert c.connect(UPDATE)
            c.send(protocol.response_create()).result(2)
            assert wait_until(lambda: c.state == "closed")
            assert c.reason == "stall"
            assert c.snapshot()["stalls"] == 1
        finally:
            c.close()


def test_no_stall_when_no_response_is_due(events):
    with FakeServer("stall") as server:
        c = client(server, events, first_token_timeout_s=0.2, stall_timeout_s=0.2)
        try:
            assert c.connect(UPDATE)
            assert not wait_until(lambda: c.state == "closed", 0.6)
        finally:
            c.close()


def test_a_send_stuck_on_a_server_that_stopped_reading_aborts(events):
    with FakeServer("stop_reading") as server:
        c = client(server, events, send_timeout_s=0.5)
        try:
            assert c.connect(UPDATE)
            chunk = b"\x00\x01" * 32_768
            stuck = False
            for _ in range(2000):
                if not c.send_audio(chunk).result(5):
                    stuck = True
                    break
            assert stuck
            assert c.state == "closed" and c.reason == "send_stuck"
            assert c.snapshot()["send_stuck"] == 1
        finally:
            c.close()


def test_a_socket_lost_while_a_send_is_blocked_is_a_stuck_send(events):
    """Fix round 1, item 4: whatever ends the socket while a write is blocked (here an abort,
    as the keepalive does when the pong never comes) is ``send_stuck``, not a race between
    timers. The send timeout is long, so it cannot be what fires."""
    with FakeServer("stop_reading") as server:
        c = client(server, events, send_timeout_s=30.0)
        try:
            assert c.connect(UPDATE)
            chunk = b"\x00\x01" * 32_768
            pending = None
            for _ in range(2000):
                future = c.send_audio(chunk)
                try:
                    future.result(0.3)
                except concurrent.futures.TimeoutError:
                    pending = future  # the write is blocked: the server reads nothing
                    break
            assert pending is not None
            ws = c._ws
            c._loop.call_soon_threadsafe(ws.transport.abort)
            assert pending.result(5) is False
            assert wait_until(lambda: c.state == "closed")
            assert c.reason == "send_stuck"
            assert c.snapshot()["send_stuck"] == 1
        finally:
            c.close()


def test_an_oversize_message_closes_the_socket(events):
    with FakeServer("oversize", oversize_bytes=70_000) as server:
        c = client(server, events, max_message_bytes=65_536)
        try:
            assert c.connect(UPDATE)
            assert wait_until(lambda: c.state == "closed")
            assert c.reason == "oversize"
            assert c.snapshot()["oversize"] == 1
        finally:
            c.close()


def test_a_failing_event_handler_is_counted(events):
    def boom(_ev):
        raise RuntimeError("handler text")

    with FakeServer() as server:
        c = RealtimeClient(
            route(server.http_base()), settings(), DirectSession(), auth_headers=dict, on_event=boom
        )
        try:
            assert c.connect(UPDATE)
            assert c.snapshot()["callback_errors"] >= 2
        finally:
            c.close()


def test_repr_names_the_route_and_state_only(events):
    with FakeServer() as server:
        c = client(server, events)
        assert repr(c) == "RealtimeClient(route='rt', state='idle')"


class Counting(DirectSession):
    def __init__(self) -> None:
        self.calls = 0

    async def allocate(self, request):
        self.calls += 1
        return await super().allocate(request)


def no_key() -> dict[str, str]:
    raise RuntimeError("secret store unreadable")


@pytest.mark.parametrize("headers", [dict, lambda: {"Authorization": ""}, no_key])
def test_a_keyed_route_without_a_resolved_key_does_not_connect(events, headers):
    """Fix round 1, item 2: the key decision follows the resolved header, not the config."""
    with FakeServer() as server:
        provider = Counting()
        c = RealtimeClient(
            route(server.http_base(), keyed=True),
            settings(),
            provider,
            auth_headers=headers,
            on_event=events.append,
        )
        try:
            assert c.connect(UPDATE) is False
            assert c.reason == "key_unavailable"
            assert provider.calls == 0 and server.requests == []
            assert c.snapshot()["key_sent"] is False
        finally:
            c.close()


def test_auth_headers_is_required():
    with pytest.raises(TypeError):
        RealtimeClient(route("http://127.0.0.1:1/v1"), settings(), DirectSession())  # type: ignore[call-arg]


def test_send_after_close_resolves_false(events):
    with FakeServer() as server:
        c = client(server, events)
        assert c.connect(UPDATE)
        c.close()
        assert c.send(protocol.response_cancel()).result(1) is False
        assert c.send_audio(PCM).result(1) is False


def test_unknown_events_neither_reach_the_handler_nor_keep_a_response_alive(events):
    """Fix round 1, item 7: only events of the subset count as progress."""
    with FakeServer("unknown_only") as server:
        c = client(server, events, first_token_timeout_s=0.3, stall_timeout_s=0.3)
        try:
            assert c.connect(UPDATE)
            c.send(protocol.response_create()).result(2)
            assert wait_until(lambda: c.state == "closed")
            assert c.reason == "stall"
            assert c.snapshot()["unknown"] >= 2
            assert not any(isinstance(e, protocol.Unknown) for e in events)
        finally:
            c.close()


def test_closing_a_connecting_client_never_hangs_the_connect():
    """The loop stops under a connect in flight (a close from another thread): ``connect``
    returns False within its bound instead of waiting forever."""
    import asyncio
    import threading
    import time

    client = RealtimeClient(
        route("http://127.0.0.1:9/v1"),
        settings(connect_timeout_s=0.3, send_timeout_s=0.1),
        DirectSession(),
        auth_headers=dict,
    )

    async def hang(update):
        await asyncio.Event().wait()

    client._connect = hang  # type: ignore[method-assign]
    result = []
    thread = threading.Thread(target=lambda: result.append(client.connect(UPDATE)), daemon=True)
    started = time.monotonic()
    thread.start()
    time.sleep(0.05)
    client.close()  # stops the loop: the connect's future never resolves
    thread.join(5)
    assert result == [False] and time.monotonic() - started < 2
    assert client.state == "closed" and client.reason == "closed"  # its close, reported once
