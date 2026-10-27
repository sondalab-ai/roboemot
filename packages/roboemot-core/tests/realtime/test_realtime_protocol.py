"""Realtime protocol subset: builders and parser (SPEC 8, Realtime speech route; SPEC 12)."""

import base64
import json

import pytest
from roboemot.realtime import protocol as p

MAX = 65_536


def frame(event: dict) -> str:
    return json.dumps(event)


def test_session_update_carries_the_format_vad_and_tools():
    tools = p.function_tools(
        [
            {
                "type": "function",
                "function": {"name": "get_time", "description": "d", "parameters": {}},
            }
        ]
    )
    ev = p.session_update(
        instructions="be kind",
        input_rate_hz=16000,
        output_rate_hz=None,
        voice="v",
        vad_threshold=0.5,
        silence_ms=500,
        create_response=False,
        interrupt_response=False,
        tools=tools,
    )
    s = ev["session"]
    assert ev["type"] == "session.update"
    assert s["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 16000}
    assert s["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": None}
    td = s["audio"]["input"]["turn_detection"]
    assert td["create_response"] is False and td["interrupt_response"] is False
    assert s["tools"] == [
        {"type": "function", "name": "get_time", "description": "d", "parameters": {}}
    ]
    assert s["tool_choice"] == "auto"
    assert json.loads(p.encode(ev)) == ev


def test_client_builders():
    assert p.audio_append(b"\x01\x00") == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(b"\x01\x00").decode(),
    }
    with pytest.raises(ValueError):
        p.audio_append(b"\x01")
    assert p.audio_commit() == {"type": "input_audio_buffer.commit"}
    assert p.text_item("hi")["item"]["content"] == [{"type": "input_text", "text": "hi"}]
    assert p.function_output("c1", "{}")["item"]["type"] == "function_call_output"
    assert p.item_delete("i1") == {"type": "conversation.item.delete", "item_id": "i1"}
    assert p.item_truncate("i1", -5)["audio_end_ms"] == 0
    assert p.response_create() == {"type": "response.create"}
    assert p.response_create("x", "none")["response"] == {
        "instructions": "x",
        "tool_choice": "none",
    }
    with pytest.raises(ValueError):
        p.response_create(tool_choice="required")  # type: ignore[arg-type]
    assert p.response_cancel() == {"type": "response.cancel"}


def test_parse_the_server_subset():
    pcm = b"\x01\x00\x02\x00"
    audio = p.parse(
        frame(
            {
                "type": "response.output_audio.delta",
                "response_id": "r",
                "item_id": "i",
                "delta": base64.b64encode(pcm).decode(),
            }
        ),
        MAX,
    )
    assert isinstance(audio, p.AudioDelta) and audio.pcm == pcm
    t = p.parse(
        frame(
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "ciao"}
        ),
        MAX,
    )
    assert isinstance(t, p.InputTranscript) and t.final and t.text == "ciao"
    s = p.parse(
        frame(
            {
                "type": "session.updated",
                "session": {"audio": {"output": {"format": {"rate": 24000}}}},
            }
        ),
        MAX,
    )
    assert isinstance(s, p.SessionEvent) and s.output_rate_hz == 24000
    call = p.parse(
        frame(
            {
                "type": "response.function_call_arguments.done",
                "call_id": "c",
                "name": "get_time",
                "arguments": "{}",
            }
        ),
        MAX,
    )
    assert isinstance(call, p.FunctionCall) and call.name == "get_time"
    done = p.parse(
        frame(
            {
                "type": "response.done",
                "response": {
                    "status": "cancelled",
                    "usage": {"total_tokens": 7},
                    "metadata": {
                        "speech_to_speech.turn_latency": {"stt": 0.25, "llm": "x", "e2e": -1}
                    },
                },
            }
        ),
        MAX,
    )
    assert isinstance(done, p.ResponseDone)
    assert (done.status, done.total_tokens, done.latency_s) == ("cancelled", 7, {"stt": 0.25})
    odd = p.parse(frame({"type": "response.done", "response": {"status": "weird"}}), MAX)
    assert odd.status == "unknown" and odd.total_tokens is None


def test_error_keeps_the_code_only():
    ev = p.parse(
        frame({"type": "error", "error": {"code": "rate_limited", "message": "you said: secret"}}),
        MAX,
    )
    assert isinstance(ev, p.ServerError) and ev.code == "rate_limited"
    assert "secret" not in repr(ev)
    bad = p.parse(frame({"type": "error", "error": {"code": "has spaces: secret"}}), MAX)
    assert bad.code is None


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"binary", "binary"),
        ("{not json", "json"),
        ("[1, 2]", "shape"),
        ('{"type": 3}', "type"),
        ('{"type": "Has Caps"}', "type"),
        (json.dumps({"type": "response.output_audio.delta", "delta": 5}), "field"),
        (json.dumps({"type": "response.output_audio.delta", "delta": "@@@"}), "base64"),
        (
            json.dumps(
                {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x01").decode()}
            ),
            "odd_pcm",
        ),
        (json.dumps({"type": "response.created", "response": "x"}), "shape"),
        (json.dumps({"type": "response.created", "response": {"id": "x" * 1000}}), "field"),
        (json.dumps({"type": "response.function_call_arguments.done", "name": "f"}), "field"),
        (
            json.dumps({"type": "input_audio_buffer.speech_started", "audio_start_ms": True}),
            "field",
        ),
    ],
)
def test_malformed_frames_never_raise(raw, reason):
    ev = p.parse(raw, MAX)
    assert isinstance(ev, p.Malformed) and ev.reason == reason


def test_oversize_frame():
    ev = p.parse(json.dumps({"type": "x", "pad": "é" * 40}), 64)
    assert isinstance(ev, p.Malformed) and ev.reason == "oversize"


def test_unknown_events_are_counted_not_fatal():
    ev = p.parse(frame({"type": "rate_limits.updated", "x": 1}), MAX)
    assert isinstance(ev, p.Unknown) and ev.type == "rate_limits.updated"


def test_content_never_reaches_repr():
    ev = p.parse(
        frame(
            {
                "type": "response.output_audio_transcript.done",
                "transcript": "SECRET",
                "item_id": "ID9",
            }
        ),
        MAX,
    )
    assert "SECRET" not in repr(ev) and "ID9" not in repr(ev)
    call = p.FunctionCall("t", "r", "CALLID", "f", "ARGS")
    assert "ARGS" not in repr(call) and "CALLID" not in repr(call)
    assert "\\x01" not in repr(p.AudioDelta("t", pcm=b"\x01\x00"))
