"""The fake server's own hold (``TurnServer(hold=...)``): a hold nobody releases fails loudly
when the server closes, instead of letting a test pass on a slower path."""

from __future__ import annotations

import json
import threading

import pytest
from realtime_helpers import wait_until
from turn_server import TurnServer
from websockets.sync.client import connect

COMMIT = json.dumps({"type": "input_audio_buffer.commit"})
DONE = "conversation.item.input_audio_transcription.completed"


def commit(server: TurnServer) -> None:
    with connect(server.url) as ws:
        ws.recv()  # session.created
        ws.send(COMMIT)
        assert wait_until(lambda: server.hold_timeouts or server.holding or server.sent_of(DONE))


def test_a_hold_nobody_releases_times_out_and_fails_the_close():
    server = TurnServer(hold=threading.Event(), hold_max_s=0.05)
    with pytest.raises(AssertionError, match="never released: 1 timed out"), server:
        commit(server)
        assert wait_until(lambda: server.hold_timeouts == 1)
    assert server.hold_timeouts == 1


def test_a_hold_still_waiting_at_the_close_fails_it():
    server = TurnServer(hold=threading.Event())
    with pytest.raises(AssertionError, match="1 still waiting"), server:
        commit(server)
        assert server.holding == 1


def test_a_released_hold_closes_cleanly():
    hold = threading.Event()
    with TurnServer(hold=hold) as server, connect(server.url) as ws:
        ws.recv()
        ws.send(COMMIT)
        assert wait_until(lambda: server.holding == 1)
        hold.set()
        assert wait_until(lambda: server.sent_of(DONE))
    assert server.hold_timeouts == 0
