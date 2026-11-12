"""An ``openai_realtime`` route (Pollen) is byte-identical on the wire with the codec in place
(SPEC 8, Realtime speech route): the frames and the handshake's key header below were recorded
on the client before the codec existed."""

# ruff: noqa: E501 (the recorded frames stay whole, byte for byte)

from __future__ import annotations

from fake_server import PCM, FakeServer
from realtime_helpers import KEY, UPDATE, bearer, route, settings, wait_until
from roboemot.realtime import protocol
from roboemot.realtime.client import RealtimeClient
from roboemot.realtime.session import DirectSession

EVENTS = [
    protocol.audio_append(PCM),
    protocol.audio_commit(),
    protocol.text_item("ciao à tutti", "re_1_1"),
    protocol.function_output("call_1", '{"ok": true}', "re_1_2"),
    protocol.response_create("be brief", "auto", event_id="rq_1"),
    protocol.response_cancel("resp_1", "rc_2"),
    protocol.item_delete("item_9"),
    protocol.item_truncate("item_1", 120),
    protocol.response_create(),
]

GOLDEN = [
    '{"event_id":"ev_1","type":"session.update","session":{"type":"realtime","instructions":"be kind","audio":{"input":{"format":{"type":"audio/pcm","rate":16000},"turn_detection":{"type":"server_vad","threshold":0.5,"silence_duration_ms":500,"create_response":false,"interrupt_response":false}},"output":{"format":{"type":"audio/pcm","rate":null},"voice":"v"}}}}',
    '{"type":"input_audio_buffer.append","audio":"AQACAAMABAA=","event_id":"ev_2"}',
    '{"type":"input_audio_buffer.commit","event_id":"ev_3"}',
    '{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"ciao à tutti"}],"id":"re_1_1"},"event_id":"ev_4"}',
    '{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_1","output":"{\\"ok\\": true}","id":"re_1_2"},"event_id":"ev_5"}',
    '{"type":"response.create","event_id":"rq_1","response":{"instructions":"be brief","tool_choice":"auto"}}',
    '{"type":"response.cancel","event_id":"rc_2","response_id":"resp_1"}',
    '{"type":"conversation.item.delete","item_id":"item_9","event_id":"ev_6"}',
    '{"type":"conversation.item.truncate","item_id":"item_1","content_index":0,"audio_end_ms":120,"event_id":"ev_7"}',
    '{"type":"response.create","event_id":"ev_8"}',
]


def test_pollen_frames_are_unchanged():
    events: list = []
    with FakeServer() as server:
        c = RealtimeClient(
            route(server.http_base(), keyed=True),
            settings(),
            DirectSession(),
            auth_headers=bearer,
            on_event=events.append,
        )
        try:
            assert c.connect(UPDATE)
            for event in EVENTS:
                assert c.send(event).result(2)
            assert wait_until(lambda: len(server.frames) == 1 + len(EVENTS))
            headers = server.requests[0]["headers"]
        finally:
            c.close()
    assert server.requests[0]["path"] == "/v1/realtime"
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert "x-goog-api-key" not in {k.lower() for k in headers}
    assert server.frames == GOLDEN
