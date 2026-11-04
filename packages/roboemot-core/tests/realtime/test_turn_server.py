"""The fake server's own hold (``TurnServer(hold=...)``): a hold nobody releases fails loudly
when the server closes, instead of letting a test pass on a slower path."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from realtime_helpers import wait_until
from turn_server import TurnServer
from websockets.sync.client import connect

COMMIT = json.dumps({"type": "input_audio_buffer.commit"})
DONE = "conversation.item.input_audio_transcription.completed"
CLOSE_S = 0.2


def commit(server: TurnServer) -> Any:
    """Open a session and commit one burst; the socket stays open (the caller closes it after
    the server, so its close never waits on a server that is still holding)."""
    ws = connect(server.url, close_timeout=CLOSE_S)
    ws.recv()  # session.created
    ws.send(COMMIT)
    assert wait_until(lambda: server.hold_timeouts or server.holding or server.sent_of(DONE))
    return ws


def test_a_hold_nobody_releases_times_out_and_fails_the_close():
    server = TurnServer(hold=threading.Event(), hold_max_s=0.05)
    with pytest.raises(AssertionError, match="never released: 1 timed out"), server:
        ws = commit(server)
        assert wait_until(lambda: server.hold_timeouts == 1)
    ws.close()
    assert server.hold_timeouts == 1


def test_a_hold_still_waiting_at_the_close_fails_it():
    server = TurnServer(hold=threading.Event())  # the default bound: still waiting at the close
    with pytest.raises(AssertionError, match="1 still waiting"), server:
        ws = commit(server)
        assert server.holding == 1
    ws.close()


def test_a_released_hold_closes_cleanly():
    hold = threading.Event()
    with TurnServer(hold=hold) as server, connect(server.url, close_timeout=CLOSE_S) as ws:
        ws.recv()
        ws.send(COMMIT)
        assert wait_until(lambda: server.holding == 1)
        hold.set()
        assert wait_until(lambda: server.sent_of(DONE))
    assert server.hold_timeouts == 0
