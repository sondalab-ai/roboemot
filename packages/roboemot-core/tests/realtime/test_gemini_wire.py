"""Gemini Live wire codec (SPEC 8, Realtime speech route): pure unit tests, no sockets.

Server messages are the shapes the S0 probe saw on ``gemini-3.8-live`` (2026-11-11): JSON in
binary frames, ``voiceActivity`` with a cumulative ``audioOffset``, the input transcript in one
message before ``ACTIVITY_END``, a ``toolCall`` before any audio, ``generationComplete`` then
``turnComplete`` with ``usageMetadata`` seconds later.
"""

import base64
import json

import pytest
from roboemot.realtime import protocol as p
from roboemot.realtime.gemini_wire import OUTPUT_RATE_HZ, GeminiWire
from roboemot.realtime.wire import OpenAIWire

MAX = 1 << 20
MODEL = "gemini-3.8-live"
PCM_IN = b"\x01\x00" * 160
PCM_OUT = b"\x02\x00" * 240


def update(instructions="Sei Neve.", tools=None, voice="Kore", rate=None):
    return p.session_update(
        instructions=instructions,
        input_rate_hz=rate,
        output_rate_hz=None,
        voice=voice,
        vad_threshold=0.5,
        silence_ms=500,
        create_response=False,
        interrupt_response=False,
        tools=tools,
    )


TOOLS = p.function_tools(
    [
        {
            "type": "function",
            "function": {
                "name": "note_tone",
                "description": "Record the user's tone of voice.",
                "parameters": {"type": "object", "properties": {"tone": {"type": "string"}}},
            },
        },
        {"type": "function", "function": {"name": "get_time", "description": "Local time."}},
    ]
)


def frame(obj, binary=True):
    text = json.dumps(obj)
    return text.encode() if binary else text


def b64(pcm):
    return base64.b64encode(pcm).decode()


def audio(pcm=PCM_OUT, said=None):
    content = {
        "modelTurn": {
            "parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": b64(pcm)}}]
        }
    }
    if said is not None:
        content["outputTranscription"] = {"text": said}
    return {"serverContent": content}


def activity(kind, offset):
    return {"voiceActivity": {"type": kind, "audioOffset": offset}}


def heard(text):
    return {"serverContent": {"inputTranscription": {"text": text}}}


def tool_call(*calls):
    return {"toolCall": {"functionCalls": [{"name": n, "args": a, "id": i} for n, a, i in calls]}}


GENERATED = {"serverContent": {"generationComplete": True}}
INTERRUPTED = {"serverContent": {"interrupted": True}}


def turn_complete(total=1785):
    return {"serverContent": {"turnComplete": True}, "usageMetadata": {"totalTokenCount": total}}


def wire(**kw):
    w = GeminiWire(MODEL, MAX, **kw)
    w.encode(update(tools=TOOLS))
    assert w.decode(frame({"setupComplete": {}})) == [p.SessionEvent(p.SESSION_UPDATED, 24000)]
    return w


def feed(w, *messages):
    out = []
    for m in messages:
        out += w.decode(frame(m))
    return out


def send(w, event):
    frames, local = w.encode({"event_id": "ev_9", **event})
    return frames, local


def burst(w, pieces=1):
    frames = []
    for _ in range(pieces):
        frames += send(w, p.audio_append(PCM_IN))[0]
    frames += send(w, p.audio_commit())[0]
    return frames


def heard_turn(w, text="Ciao Neve, come stai oggi?", start="0s", end="1.689375s"):
    """One robot turn's input side: the burst, then the server's start, transcript, end."""
    burst(w)
    return feed(w, activity("ACTIVITY_START", start), heard(text), activity("ACTIVITY_END", end))


def types(events):
    return [e.type for e in events]


RESPONSE_TYPES = (
    p.ResponseCreated,
    p.AudioDelta,
    p.AudioDone,
    p.ReplyTranscript,
    p.FunctionCall,
    p.ResponseDone,
)


# --------------------------------------------------------------------------- session


def test_first_session_update_becomes_setup():
    w = GeminiWire(MODEL, MAX)
    frames, local = w.encode({"event_id": "ev_1", **update(tools=TOOLS)})
    assert local == []
    assert frames == [
        {
            "setup": {
                "model": "models/gemini-3.8-live",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
                },
                "systemInstruction": {"parts": [{"text": "Sei Neve."}]},
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "note_tone",
                                "description": "Record the user's tone of voice.",
                                "parametersJsonSchema": {
                                    "type": "object",
                                    "properties": {"tone": {"type": "string"}},
                                },
                                "behavior": "BLOCKING",
                            },
                            {
                                "name": "get_time",
                                "description": "Local time.",
                                "parametersJsonSchema": {"type": "object", "properties": {}},
                                "behavior": "BLOCKING",
                            },
                        ]
                    }
                ],
                "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}},
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "contextWindowCompression": {"slidingWindow": {}},
            }
        }
    ]


def test_setup_takes_the_codec_voice_and_a_full_model_name():
    w = GeminiWire("models/gemini-3.8-live", MAX, voice="Aoede")
    setup = w.encode(update(voice="Serena"))[0][0]["setup"]
    assert setup["model"] == "models/gemini-3.8-live"
    voice = setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
    assert voice == {"voiceName": "Aoede"}
    bare = GeminiWire(MODEL, MAX).encode(update(voice=None, tools=[]))[0][0]["setup"]
    assert "speechConfig" not in bare["generationConfig"] and "tools" not in bare


def test_setup_complete_is_session_updated_at_24_khz():
    w = GeminiWire(MODEL, MAX)
    w.encode(update())
    [event] = w.decode(frame({"setupComplete": {}}))
    assert event == p.SessionEvent(p.SESSION_UPDATED, OUTPUT_RATE_HZ)
    assert OUTPUT_RATE_HZ == 24000


def test_later_session_update_is_context_before_the_next_activity():
    w = wire()
    assert send(w, update(instructions="Sei Neve. Parli con Marcello.", tools=[])) == ([], [])
    assert w.counts["tools_update_ignored"] == 1
    frames = burst(w)
    assert frames[0] == {
        "clientContent": {
            "turns": [
                {
                    "role": "user",
                    "parts": [{"text": "[Updated instructions]\nSei Neve. Parli con Marcello."}],
                }
            ],
            "turnComplete": False,
        }
    }
    assert frames[1] == {"realtimeInput": {"activityStart": {}}}
    # the same instructions again: nothing more goes
    send(w, update(instructions="Sei Neve. Parli con Marcello."))
    assert "clientContent" not in burst(w)[0]


def test_unchanged_instructions_send_no_context():
    w = wire()
    send(w, update(instructions="Sei Neve."))
    assert burst(w)[0] == {"realtimeInput": {"activityStart": {}}}


# --------------------------------------------------------------------------- audio in


def test_first_append_of_a_burst_starts_the_activity():
    w = wire()
    frames, local = send(w, p.audio_append(PCM_IN))
    assert local == []
    assert frames == [
        {"realtimeInput": {"activityStart": {}}},
        {"realtimeInput": {"audio": {"data": b64(PCM_IN), "mimeType": "audio/pcm;rate=16000"}}},
    ]
    frames, _ = send(w, p.audio_append(PCM_IN))
    assert frames == [
        {"realtimeInput": {"audio": {"data": b64(PCM_IN), "mimeType": "audio/pcm;rate=16000"}}}
    ]


def test_append_takes_the_rate_the_update_names():
    w = GeminiWire(MODEL, MAX)
    w.encode(update(rate=24000))
    frames, _ = w.encode(p.audio_append(PCM_IN))
    assert frames[1]["realtimeInput"]["audio"]["mimeType"] == "audio/pcm;rate=24000"


def test_commit_ends_the_activity():
    w = wire()
    send(w, p.audio_append(PCM_IN))
    assert send(w, p.audio_commit()) == ([{"realtimeInput": {"activityEnd": {}}}], [])
    assert send(w, p.audio_commit()) == ([], [])  # no activity: nothing to end
    assert w.counts["commits_without_audio"] == 1


# --------------------------------------------------------------------------- input events


def test_voice_activity_is_speech_boundaries_of_the_burst_item():
    w = wire()
    burst(w)
    started, stopped = feed(
        w, activity("ACTIVITY_START", "0s"), activity("ACTIVITY_END", "1.689375s")
    )
    assert isinstance(started, p.SpeechBoundary) and started.type == p.SPEECH_STARTED
    assert isinstance(stopped, p.SpeechBoundary) and stopped.type == p.SPEECH_STOPPED
    assert started.item_id == stopped.item_id and started.item_id.startswith("gitem_")
    assert (started.audio_ms, stopped.audio_ms) == (0, 1689)


def test_speech_boundaries_follow_the_cumulative_offsets():
    """S0's four turns: each start is the previous end (``holds`` uses >=), in ms."""
    offsets = [
        ("0s", "1.689375s"),
        ("1.689375s", "3.126562500s"),
        ("3.126562500s", "5.446312500s"),
        ("5.446312500s", "7.127250s"),
    ]
    w = wire()
    seen, items = [], []
    for start, end in offsets:
        events = heard_turn(w, start=start, end=end)
        bounds = [e for e in events if isinstance(e, p.SpeechBoundary)]
        seen.append(tuple(b.audio_ms for b in bounds))
        items.append(bounds[0].item_id)
        send(w, p.item_delete(bounds[0].item_id))  # the route declines: next turn
    assert seen == [(0, 1689), (1689, 3126), (3126, 5446), (5446, 7127)]
    assert len(set(items)) == len(items)
    for (_, end), (start, _) in zip(seen, seen[1:], strict=False):
        assert start >= end


def test_offsets_never_lose_a_millisecond_to_float_error():
    w = wire()
    burst(w)
    [ev] = feed(w, activity("ACTIVITY_START", "4.35s"))
    assert ev.audio_ms == 4350


def test_input_transcript_is_delta_then_completed_before_the_stop():
    w = wire()
    events = heard_turn(w)
    assert types(events) == [
        p.SPEECH_STARTED,
        p.TRANSCRIPTION_DELTA,
        p.TRANSCRIPTION_COMPLETED,
        p.SPEECH_STOPPED,
    ]
    item = events[0].item_id
    assert all(e.item_id == item for e in events)
    assert events[1].text == events[2].text == "Ciao Neve, come stai oggi?"
    assert events[2].final


def test_transcript_in_pieces_completes_with_all_of_them():
    w = wire()
    burst(w)
    events = feed(
        w,
        activity("ACTIVITY_START", "0s"),
        heard("Ciao Neve,"),
        heard(" come stai?"),
        activity("ACTIVITY_END", "1.5s"),
    )
    [done] = [e for e in events if isinstance(e, p.InputTranscript) and e.final]
    assert done.text == "Ciao Neve, come stai?"


def test_transcript_after_the_end_completes_at_once():
    w = wire()
    burst(w)
    feed(w, activity("ACTIVITY_START", "0s"), activity("ACTIVITY_END", "1.5s"))
    events = feed(w, heard("Ciao"))
    assert types(events) == [p.TRANSCRIPTION_DELTA, p.TRANSCRIPTION_COMPLETED]
    assert events[1].text == "Ciao"
    assert feed(w, heard(" ancora")) == []  # completed: a late piece is only counted
    assert w.counts["late_input_transcripts"] == 1


def test_no_transcript_completes_empty_at_the_first_answer():
    w = wire()
    burst(w)
    feed(w, activity("ACTIVITY_START", "0s"), activity("ACTIVITY_END", "1.5s"))
    events = feed(w, tool_call(("note_tone", {"tone": "calm"}, "fc_1")))
    assert types(events) == [p.TRANSCRIPTION_COMPLETED]  # the answer itself is held
    assert events[0].text == "" and events[0].final


# --------------------------------------------------------------------------- hold and release


def test_answer_is_held_until_response_create_then_released_in_order():
    w = wire()
    item = heard_turn(w)[0].item_id
    held = feed(
        w,
        {},
        {"serverContent": {}},
        audio(PCM_OUT, said="Ciao! Sto benissimo, "),
        audio(PCM_OUT),
        audio(PCM_OUT, said="tu come stai?"),
        GENERATED,
    )
    assert held == []
    frames, local = send(w, p.response_create("mood: happy", "auto"))
    assert frames == []
    assert types(local) == [
        p.RESPONSE_CREATED,
        p.AUDIO_DELTA,
        p.REPLY_TRANSCRIPT_DELTA,
        p.AUDIO_DELTA,
        p.AUDIO_DELTA,
        p.REPLY_TRANSCRIPT_DELTA,
        p.AUDIO_DONE,
        p.REPLY_TRANSCRIPT_DONE,
        p.RESPONSE_DONE,
    ]
    rid = local[0].response_id
    assert rid == "gresp_1"
    assert all(e.response_id == rid for e in local)
    out_items = {e.item_id for e in local if not isinstance(e, p.ResponseCreated | p.ResponseDone)}
    assert len(out_items) == 1 and item not in out_items
    assert [e.pcm for e in local if isinstance(e, p.AudioDelta)] == [PCM_OUT] * 3
    final = local[-2]
    assert final.final and final.text == "Ciao! Sto benissimo, tu come stai?"
    assert local[-1].status == "completed" and local[-1].total_tokens is None
    # what comes after the release is the released response's, live
    assert feed(w, turn_complete(1785)) == [p.UsageReport(p.USAGE_REPORT, ("gresp_1",), 1785)]


def test_input_events_are_never_held():
    w = wire()
    burst(w)
    feed(w, activity("ACTIVITY_START", "0s"))
    assert types(feed(w, heard("Ciao"))) == [p.TRANSCRIPTION_DELTA]
    assert types(feed(w, activity("ACTIVITY_END", "1s"))) == [
        p.TRANSCRIPTION_COMPLETED,
        p.SPEECH_STOPPED,
    ]
    assert feed(w, audio()) == []  # the answer is


def test_release_mid_answer_passes_the_rest_on_live():
    w = wire()
    heard_turn(w)
    feed(w, audio(said="Ciao"))
    _, local = send(w, p.response_create())
    assert types(local) == [p.RESPONSE_CREATED, p.AUDIO_DELTA, p.REPLY_TRANSCRIPT_DELTA]
    live = feed(w, audio(), GENERATED)
    assert types(live) == [p.AUDIO_DELTA, p.AUDIO_DONE, p.REPLY_TRANSCRIPT_DONE, p.RESPONSE_DONE]
    assert {e.response_id for e in live} == {local[0].response_id}


def test_response_create_before_any_answer_announces_it_at_once():
    """S0: the transcript lands before any model content, so the route may ask first."""
    w = wire()
    heard_turn(w)
    frames, local = send(w, p.response_create())
    assert frames == []
    assert types(local) == [p.RESPONSE_CREATED]
    rid = local[0].response_id
    live = feed(w, {"serverContent": {}}, audio(said="Ciao"), GENERATED)
    assert types(live) == [
        p.AUDIO_DELTA,
        p.REPLY_TRANSCRIPT_DELTA,
        p.AUDIO_DONE,
        p.REPLY_TRANSCRIPT_DONE,
        p.RESPONSE_DONE,
    ]
    assert {e.response_id for e in live} == {rid}


def test_an_empty_answer_still_ends_its_response():
    """Gate 3: generationComplete with no audio still gives the route its response.done."""
    w = wire()
    heard_turn(w)
    feed(w, GENERATED)
    _, local = send(w, p.response_create())
    assert types(local) == [p.RESPONSE_CREATED, p.RESPONSE_DONE]
    assert local[1].status == "completed"


def test_deleting_the_turn_item_discards_the_held_answer():
    w = wire()
    item = heard_turn(w)[0].item_id
    feed(w, audio(said="Ciao"))
    frames, local = send(w, p.item_delete(item))
    assert frames == []
    assert local == [p.ItemDeleted(p.ITEM_DELETED, item, emulated=True)]
    assert w.counts["held_discarded"] == 1
    # the rest of that answer never reaches the route, its usage still does
    assert feed(w, audio(), GENERATED) == []
    assert feed(w, turn_complete(900)) == [p.UsageReport(p.USAGE_REPORT, (), 900)]
    # the next turn is held and released as usual
    heard_turn(w, start="1.689375s", end="3s")
    feed(w, audio(said="Sì"), GENERATED)
    _, local = send(w, p.response_create())
    assert types(local)[0] == p.RESPONSE_CREATED and local[-1].status == "completed"
    assert len([e for e in local if isinstance(e, p.AudioDelta)]) == 1


def test_a_new_burst_discards_an_answer_never_released():
    w = wire()
    heard_turn(w)
    feed(w, audio(said="Ciao"))
    burst(w)
    assert w.counts["held_discarded"] == 1
    feed(w, activity("ACTIVITY_START", "1.689375s"))
    feed(w, heard("Neve?"), activity("ACTIVITY_END", "2.5s"))
    feed(w, audio(said="Dimmi"), GENERATED)
    _, local = send(w, p.response_create())
    texts = [e.text for e in local if isinstance(e, p.ReplyTranscript) and e.final]
    assert texts == ["Dimmi"]


def test_usage_while_holding_is_held_after_the_done():
    w = wire()
    heard_turn(w)
    assert feed(w, audio(), GENERATED, turn_complete(1249)) == []
    _, local = send(w, p.response_create())
    assert types(local)[-2:] == [p.RESPONSE_DONE, p.USAGE_REPORT]
    assert local[-1] == p.UsageReport(p.USAGE_REPORT, (local[0].response_id,), 1249)


def test_usage_of_a_discarded_hold_is_reported_at_the_discard():
    w = wire()
    item = heard_turn(w)[0].item_id
    feed(w, audio(), GENERATED, turn_complete(1249))
    _, local = send(w, p.item_delete(item))
    assert local == [
        p.ItemDeleted(p.ITEM_DELETED, item, emulated=True),
        p.UsageReport(p.USAGE_REPORT, (), 1249),
    ]


def test_a_late_turn_complete_never_ends_the_next_answer():
    """turnComplete comes 3-5 s after generationComplete: the next turn may be under way."""
    w = wire()
    heard_turn(w)
    feed(w, audio(), GENERATED)
    send(w, p.response_create())
    heard_turn(w, start="1.689375s", end="3s")
    _, local = send(w, p.response_create())
    rid = local[0].response_id
    assert feed(w, turn_complete(1785)) == [p.UsageReport(p.USAGE_REPORT, ("gresp_1",), 1785)]
    live = feed(w, audio(said="Sì"), GENERATED)
    assert types(live)[-1] == p.RESPONSE_DONE and live[-1].response_id == rid


# --------------------------------------------------------------------------- tools


def test_tool_round():
    """toolCall (held) -> FunctionCall + done; output + response.create -> toolResponse;
    Gemini's continuation is a new response; the turn's usage covers both."""
    w = wire()
    heard_turn(w)
    assert feed(w, tool_call(("note_tone", {"tone": "happy"}, "fc_3768515884713047136"))) == []
    _, local = send(w, p.response_create())
    assert types(local) == [p.RESPONSE_CREATED, p.FUNCTION_CALL_DONE, p.RESPONSE_DONE]
    call = local[1]
    assert call.name == "note_tone" and call.call_id == "fc_3768515884713047136"
    assert json.loads(call.arguments) == {"tone": "happy"}
    assert call.item_id.startswith("gcall_") and call.response_id == "gresp_1"
    assert local[2].status == "completed"
    assert send(w, p.function_output(call.call_id, '{"ok": true}', "out_1")) == ([], [])
    frames, local = send(w, p.response_create(None, "none"))
    assert frames == [
        {
            "toolResponse": {
                "functionResponses": [
                    {"id": call.call_id, "name": "note_tone", "response": {"ok": True}}
                ]
            }
        }
    ]
    assert w.counts["tool_choice_dropped"] == 1
    assert types(local) == [p.RESPONSE_CREATED]
    second = local[0].response_id
    assert second == "gresp_2"
    live = feed(w, audio(said="Ciao!"), GENERATED)
    assert {e.response_id for e in live} == {second}
    assert types(live)[-1] == p.RESPONSE_DONE
    assert feed(w, turn_complete(1785)) == [
        p.UsageReport(p.USAGE_REPORT, ("gresp_1", "gresp_2"), 1785)
    ]


def test_several_calls_in_one_tool_call():
    w = wire()
    heard_turn(w, text="Neve, che ore sono adesso?")
    feed(
        w,
        tool_call(
            ("note_tone", {"tone": "calm"}, "fc_14580876985390141047"),
            ("get_time", {}, "fc_14580876985390138608"),
        ),
    )
    _, local = send(w, p.response_create())
    calls = [e for e in local if isinstance(e, p.FunctionCall)]
    assert [c.name for c in calls] == ["note_tone", "get_time"]
    assert len({c.item_id for c in calls}) == 2
    assert types(local)[-1] == p.RESPONSE_DONE
    assert len([e for e in local if isinstance(e, p.ResponseDone)]) == 1
    send(w, p.function_output("fc_14580876985390138608", '{"time": "22:30"}'))
    send(w, p.function_output("fc_14580876985390141047", "noted"))
    frames, _ = send(w, p.response_create())
    assert frames == [
        {
            "toolResponse": {
                "functionResponses": [
                    {
                        "id": "fc_14580876985390141047",
                        "name": "note_tone",
                        "response": {"result": "noted"},
                    },
                    {
                        "id": "fc_14580876985390138608",
                        "name": "get_time",
                        "response": {"time": "22:30"},
                    },
                ]
            }
        }
    ]


def test_a_call_the_route_skips_is_answered_before_the_next_input():
    """The skip path sends no output and deletes the call item: Gemini's BLOCKING call must
    not stay open, so the codec answers it itself and drops what it says after."""
    w = wire()
    heard_turn(w, text="Marco, mi passi il sale, per favore?")
    feed(w, tool_call(("ignore_utterance", {}, "fc_18349770890208281133")))
    _, local = send(w, p.response_create())
    call = local[1]
    frames, acks = send(w, p.item_delete(call.item_id))
    assert acks == [p.ItemDeleted(p.ITEM_DELETED, call.item_id, emulated=True)]
    assert frames == [
        {
            "toolResponse": {
                "functionResponses": [
                    {
                        "id": "fc_18349770890208281133",
                        "name": "ignore_utterance",
                        "response": {"result": "skipped"},
                    }
                ]
            }
        }
    ]
    assert feed(w, GENERATED) == []  # its continuation is nobody's
    assert feed(w, turn_complete(1249)) == [p.UsageReport(p.USAGE_REPORT, ("gresp_1",), 1249)]


def test_open_calls_are_answered_before_a_new_activity():
    w = wire()
    heard_turn(w)
    feed(w, tool_call(("get_time", {}, "fc_1")))
    send(w, p.response_create())
    frames = burst(w)  # the route moved on without an output
    assert frames[0] == {
        "toolResponse": {
            "functionResponses": [
                {"id": "fc_1", "name": "get_time", "response": {"result": "skipped"}}
            ]
        }
    }
    assert frames[1] == {"realtimeInput": {"activityStart": {}}}
    assert feed(w, audio()) == []  # the continuation before the new activity is dropped
    feed(w, activity("ACTIVITY_START", "1.689375s"), activity("ACTIVITY_END", "2s"))
    feed(w, audio(said="Ok"), GENERATED)
    _, local = send(w, p.response_create())
    assert [e.text for e in local if isinstance(e, p.ReplyTranscript) and e.final] == ["Ok"]


def test_a_text_turn_after_a_declined_turn_still_sends_its_text():
    """A declined turn's held toolCall leaves a call open the route never saw: the next text
    turn answers it ``skipped`` and still goes; Gemini's one merged answer is the text
    turn's, never dropped."""
    w = wire()
    item = heard_turn(w)[0].item_id
    feed(w, tool_call(("note_tone", {"tone": "calm"}, "fc_1")))
    send(w, p.item_delete(item))
    send(w, p.text_item("Che ore sono?"))
    frames, local = send(w, p.response_create())
    assert frames == [
        {
            "toolResponse": {
                "functionResponses": [
                    {"id": "fc_1", "name": "note_tone", "response": {"result": "skipped"}}
                ]
            }
        },
        {
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": "Che ore sono?"}]}],
                "turnComplete": True,
            }
        },
    ]
    assert types(local) == [p.RESPONSE_CREATED]
    live = feed(w, audio(said="Ok. "), audio(said="Sono le dieci."), GENERATED)
    assert {e.response_id for e in live} == {local[0].response_id}
    assert [e.text for e in live if isinstance(e, p.ReplyTranscript) and e.final] == [
        "Ok. Sono le dieci."
    ]
    assert types(live)[-1] == p.RESPONSE_DONE


def test_function_output_that_is_not_an_object_is_wrapped():
    w = wire()
    heard_turn(w)
    feed(w, tool_call(("get_time", {}, "fc_1")))
    send(w, p.response_create())
    send(w, p.function_output("fc_1", "[1, 2]"))
    frames, _ = send(w, p.response_create())
    assert frames[0]["toolResponse"]["functionResponses"][0]["response"] == {"result": [1, 2]}


# --------------------------------------------------------------------------- text turns


def test_text_item_and_response_create_send_one_complete_turn():
    w = GeminiWire(MODEL, MAX)
    w.encode(update(instructions="Sei Neve."))
    w.decode(frame({"setupComplete": {}}))
    assert send(w, p.text_item("Ciao Neve", "check_item")) == ([], [])
    frames, local = send(w, p.response_create(event_id="check_resp"))
    assert frames == [
        {
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": "Ciao Neve"}]}],
                "turnComplete": True,
            }
        }
    ]
    assert types(local) == [p.RESPONSE_CREATED]
    live = feed(w, audio(said="Ciao!"), GENERATED)
    assert {e.response_id for e in live} == {local[0].response_id}
    _, acks = send(w, p.item_delete("check_item"))
    assert acks == [p.ItemDeleted(p.ITEM_DELETED, "check_item", emulated=True)]


def test_text_turn_carries_the_changed_instructions_first():
    w = wire()
    send(w, p.text_item("Che ore sono?"))
    frames, _ = send(w, p.response_create("Sei Neve. Umore: allegro."))
    assert frames[0]["clientContent"]["turns"] == [
        {"role": "user", "parts": [{"text": "[Updated instructions]\nSei Neve. Umore: allegro."}]},
        {"role": "user", "parts": [{"text": "Che ore sono?"}]},
    ]


def test_a_line_with_only_instructions_is_sent_as_the_turn():
    """``_rt_line``: response.create with the line in its instructions and no item."""
    w = wire()
    frames, local = send(w, p.response_create("Say exactly: Scusa, non ho capito.", "none"))
    assert frames == [
        {
            "clientContent": {
                "turns": [
                    {"role": "user", "parts": [{"text": "Say exactly: Scusa, non ho capito."}]}
                ],
                "turnComplete": True,
            }
        }
    ]
    assert types(local) == [p.RESPONSE_CREATED]


# --------------------------------------------------------------------------- cancel, interrupt


def test_cancel_ends_the_response_and_drops_the_rest_of_its_turn():
    w = wire()
    heard_turn(w)
    _, local = send(w, p.response_create())
    rid = local[0].response_id
    feed(w, audio(said="Ciao"))
    frames, local = send(w, p.response_cancel(event_id="cancel_1"))
    assert frames == []
    assert local == [p.ResponseDone(p.RESPONSE_DONE, rid, "cancelled")]
    assert feed(w, audio(), GENERATED) == []
    assert w.counts["dropped_after_cancel"] == 2
    assert feed(w, turn_complete(800)) == [p.UsageReport(p.USAGE_REPORT, (rid,), 800)]
    # nothing open: a cancel is a no-op (the route's error path never sees it)
    assert send(w, p.response_cancel()) == ([], [])
    assert w.counts["cancels_idle"] == 1


def test_cancel_names_another_response_is_a_no_op():
    w = wire()
    heard_turn(w)
    send(w, p.response_create())
    assert send(w, p.response_cancel("gresp_99")) == ([], [])


def test_interrupted_is_a_cancelled_done():
    w = wire()
    heard_turn(w)
    _, local = send(w, p.response_create())
    feed(w, audio())
    events = feed(w, INTERRUPTED)
    assert types(events) == [p.AUDIO_DONE, p.REPLY_TRANSCRIPT_DONE, p.RESPONSE_DONE]
    assert events[-1].status == "cancelled" and events[-1].response_id == local[0].response_id


def test_interrupted_after_a_cancel_ends_no_second_done():
    w = wire()
    heard_turn(w)
    send(w, p.response_create())
    feed(w, audio())
    send(w, p.response_cancel())
    assert feed(w, INTERRUPTED, GENERATED) == []


def test_a_barge_in_burst_cancels_the_live_answer_not_the_new_one():
    """A new burst while the reply still streams: Gemini sends the old turn's last audio and
    its ``interrupted`` before it starts the new activity; neither touches the new hold."""
    w = wire()
    heard_turn(w)
    _, local = send(w, p.response_create())
    old = local[0].response_id
    feed(w, audio())
    burst(w)
    late = feed(w, audio(), INTERRUPTED)
    assert types(late) == [p.AUDIO_DELTA, p.AUDIO_DONE, p.REPLY_TRANSCRIPT_DONE, p.RESPONSE_DONE]
    assert {e.response_id for e in late} == {old} and late[-1].status == "cancelled"
    feed(w, activity("ACTIVITY_START", "1.689375s"), heard("Basta"))
    feed(w, activity("ACTIVITY_END", "2.5s"), audio(said="Ok"), GENERATED)
    _, local = send(w, p.response_create())
    assert local[0].response_id != old
    assert [e.text for e in local if isinstance(e, p.ReplyTranscript) and e.final] == ["Ok"]


def test_truncate_is_an_emulated_ack():
    w = wire()
    assert send(w, p.item_truncate("gout_1", 1200)) == (
        [],
        [p.ItemTruncated(p.ITEM_TRUNCATED, "gout_1", emulated=True)],
    )


# --------------------------------------------------------------------------- other messages


def test_go_away_says_the_time_left():
    w = wire()
    assert feed(w, {"goAway": {"timeLeft": "50s"}}) == [p.GoAway(p.GO_AWAY, 50_000)]
    assert feed(w, {"goAway": {}}) == [p.GoAway(p.GO_AWAY, None)]


@pytest.mark.parametrize(
    ("message", "counter"),
    [
        (
            {"sessionResumptionUpdate": {"newHandle": "1101efa5", "resumable": True}},
            "resumption_updates",
        ),
        ({}, "empty_frames"),
        ({"serverContent": {}}, "empty_content"),
        ({"toolCallCancellation": {"ids": ["fc_1"]}}, "tool_call_cancellations"),
    ],
)
def test_ignored_messages_are_counted(message, counter):
    w = wire()
    assert feed(w, message) == []
    assert w.counts[counter] == 1


def test_unknown_message_is_unknown():
    w = wire()
    [event] = feed(w, {"somethingNew": {}})
    assert isinstance(event, p.Unknown)


def test_thought_parts_are_dropped():
    w = wire()
    heard_turn(w)
    send(w, p.response_create())
    thought = {"serverContent": {"modelTurn": {"parts": [{"thought": True, "text": "plan"}]}}}
    assert feed(w, thought) == []
    assert w.counts["thought_parts"] == 1
    live = feed(w, audio(said="Ciao"), GENERATED)
    [final] = [e for e in live if isinstance(e, p.ReplyTranscript) and e.final]
    assert final.text == "Ciao"


def test_binary_and_text_frames_decode_alike():
    a, b = wire(), wire()
    for w, binary in ((a, True), (b, False)):
        burst(w)
        w.decode(frame(activity("ACTIVITY_START", "0s"), binary))
    assert a.decode(frame(heard("Ciao"))) == b.decode(frame(heard("Ciao"), binary=False))


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"not json", "json"),
        (b"\xff\xfe", "json"),
        (b"[1, 2]", "shape"),
        (json.dumps(audio(b"\x01\x02\x03")), "odd_pcm"),
        (
            json.dumps(
                {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": "@@"}}]}}}
            ),
            "base64",
        ),
        (json.dumps({"toolCall": {"functionCalls": [{"name": "x"}]}}), "field"),
        (json.dumps({"serverContent": {"inputTranscription": {"text": 5}}}), "field"),
    ],
)
def test_malformed_frames(raw, reason):
    w = wire()
    burst(w)
    assert w.decode(raw) == [p.Malformed("malformed", reason)]


def test_oversize_frame_is_malformed():
    w = GeminiWire(MODEL, 64)
    assert w.decode(b"{" + b" " * 100 + b"}") == [p.Malformed("malformed", "oversize")]


def test_a_malformed_frame_changes_no_state():
    w = wire()
    heard_turn(w)
    bad = {
        "serverContent": {
            "modelTurn": {
                "parts": [{"inlineData": {"data": b64(PCM_OUT)}}, {"inlineData": {"data": "@@"}}]
            }
        }
    }
    w.decode(frame(bad))
    feed(w, GENERATED)
    _, local = send(w, p.response_create())
    assert types(local) == [p.RESPONSE_CREATED, p.RESPONSE_DONE]


def test_events_keep_content_out_of_repr():
    w = wire()
    events = heard_turn(w, text="segreto")
    feed(w, audio(said="riservato"), GENERATED)
    _, local = send(w, p.response_create())
    shown = repr(events + local) + repr(w)
    assert "segreto" not in shown and "riservato" not in shown


# --------------------------------------------------------------------------- review round


BAD_DURATIONS = [
    "1e999999s",
    "1E5s",
    "-1s",
    "1_0s",
    "NaNs",
    "Infinitys",
    "s",
    "1.s",
    ".5s",
    "1.2345678901s",
    "9" * 40 + "s",
    "1.5",
    " 1s",
    "1s ",
    "١s",
    3,
    1.5,
    True,
    [],
    {},
]


@pytest.mark.parametrize("value", BAD_DURATIONS, ids=repr)
def test_bad_durations_never_raise_and_are_ignored(value):
    w = wire()
    burst(w)
    [start] = w.decode(frame(activity("ACTIVITY_START", value)))
    assert start == p.SpeechBoundary(p.SPEECH_STARTED, start.item_id, None)
    assert w.decode(frame({"goAway": {"timeLeft": value}})) == [p.GoAway(p.GO_AWAY, None)]
    assert w.counts["bad_durations"] == 2


def test_decode_never_raises_on_random_messages():
    """Seeded fuzz: random nested values under every key the codec reads."""
    import random

    rng = random.Random(237)
    leaves = [None, True, 0, -1, 2**70, 1.5, "", "x", "1e999999s", "1_0s", b64(PCM_OUT), "@@"]

    def value(depth):
        if depth > 3 or rng.random() < 0.4:
            return rng.choice(leaves)
        if rng.random() < 0.5:
            return [value(depth + 1) for _ in range(rng.randint(0, 3))]
        keys = [
            "type",
            "audioOffset",
            "timeLeft",
            "parts",
            "inlineData",
            "data",
            "mimeType",
            "text",
            "thought",
            "functionCalls",
            "id",
            "name",
            "args",
            "modelTurn",
            "inputTranscription",
            "outputTranscription",
            "generationComplete",
            "turnComplete",
            "interrupted",
            "totalTokenCount",
        ]
        return {rng.choice(keys): value(depth + 1) for _ in range(rng.randint(0, 4))}

    tops = [
        "setupComplete",
        "serverContent",
        "toolCall",
        "toolCallCancellation",
        "voiceActivity",
        "usageMetadata",
        "goAway",
        "sessionResumptionUpdate",
    ]
    w = wire()
    for _ in range(3000):
        if rng.random() < 0.1:
            burst(w)
        if rng.random() < 0.05:
            send(w, p.response_create())
        message = {rng.choice(tops): value(0) for _ in range(rng.randint(1, 3))}
        events = w.decode(frame(message, binary=rng.random() < 0.5))
        assert all(isinstance(e, p.ServerEvent) for e in events)


def test_a_late_turn_complete_reports_usage_and_never_ends_the_next_turn():
    """N generationComplete, then N+1's commit and audio, then N's turnComplete, then the
    rest of N+1: N+1 is released whole, and the usage names N's response only."""
    w = wire()
    heard_turn(w)
    send(w, p.response_create())
    first = feed(w, audio(said="Ciao"), GENERATED)
    n = first[-1].response_id
    heard_turn(w, start="1.689375s", end="3s")
    assert feed(w, audio(PCM_OUT, said="Sto "), turn_complete(1785)) == []  # all held
    assert feed(w, audio(PCM_OUT, said="bene."), GENERATED) == []
    _, local = send(w, p.response_create())
    n1 = local[0].response_id
    assert n1 != n
    assert types(local) == [
        p.RESPONSE_CREATED,
        p.AUDIO_DELTA,
        p.REPLY_TRANSCRIPT_DELTA,
        p.USAGE_REPORT,
        p.AUDIO_DELTA,
        p.REPLY_TRANSCRIPT_DELTA,
        p.AUDIO_DONE,
        p.REPLY_TRANSCRIPT_DONE,
        p.RESPONSE_DONE,
    ]
    assert local[3] == p.UsageReport(p.USAGE_REPORT, (n,), 1785)
    assert [e.text for e in local if isinstance(e, p.ReplyTranscript) and e.final] == ["Sto bene."]
    assert len([e for e in local if isinstance(e, p.ResponseDone)]) == 1
    assert feed(w, turn_complete(2239)) == [p.UsageReport(p.USAGE_REPORT, (n1,), 2239)]


def test_a_late_turn_complete_never_ends_a_live_answer():
    w = wire()
    heard_turn(w)
    send(w, p.response_create())
    feed(w, audio(), GENERATED)
    heard_turn(w, start="1.689375s", end="3s")
    _, local = send(w, p.response_create())
    feed(w, audio(said="Sto "))
    assert feed(w, turn_complete(1785)) == [p.UsageReport(p.USAGE_REPORT, ("gresp_1",), 1785)]
    live = feed(w, audio(said="bene."), GENERATED)
    assert types(live)[-1] == p.RESPONSE_DONE and live[-1].response_id == local[0].response_id
    assert [e.text for e in live if isinstance(e, p.ReplyTranscript) and e.final] == ["Sto bene."]


def test_usage_reports_stay_in_order_across_a_release_and_the_next_turn():
    w = wire()
    heard_turn(w)
    feed(w, audio(), GENERATED)
    _, first = send(w, p.response_create())
    assert types(first)[-1] == p.RESPONSE_DONE
    assert feed(w, turn_complete(100)) == [p.UsageReport(p.USAGE_REPORT, ("gresp_1",), 100)]
    heard_turn(w, start="1.689375s", end="3s")
    feed(w, audio(), GENERATED, turn_complete(200))
    _, second = send(w, p.response_create())
    assert types(second)[-2:] == [p.RESPONSE_DONE, p.USAGE_REPORT]
    assert second[-1] == p.UsageReport(p.USAGE_REPORT, (second[0].response_id,), 200)


def test_tool_schemas_go_as_json_schema():
    from roboemot.skills.schema import Param, Range, json_schema

    params = json_schema(
        [
            Param("minutes", int, range=Range(1, 120), description="How long."),
            Param("label", str, required=False, max_len=40),
        ]
    )
    tool = {
        "type": "function",
        "name": "set_timer",
        "description": "A timer.",
        "parameters": params,
    }
    w = GeminiWire(MODEL, MAX)
    setup = w.encode(update(tools=[tool]))[0][0]["setup"]
    [declaration] = setup["tools"][0]["functionDeclarations"]
    assert declaration["parametersJsonSchema"] == params
    assert "parameters" not in declaration
    assert params["additionalProperties"] is False
    assert params["properties"]["minutes"]["maximum"] == 120
    assert params["properties"]["label"]["maxLength"] == 40


def test_a_user_text_supersedes_a_held_answer():
    w = wire()
    heard_turn(w)
    feed(w, audio(said="Ciao"))
    assert send(w, p.text_item("Scusa, volevo dire altro.")) == ([], [])
    assert w.counts["held_discarded"] == 1
    frames, local = send(w, p.response_create())
    assert frames[-1]["clientContent"]["turns"] == [
        {"role": "user", "parts": [{"text": "Scusa, volevo dire altro."}]}
    ]
    assert types(local) == [p.RESPONSE_CREATED]


def test_a_text_item_deleted_before_response_create_is_never_sent():
    w = wire()
    send(w, p.text_item("prima", "t1"))
    send(w, p.text_item("seconda", "t2"))
    _, acks = send(w, p.item_delete("t1"))
    assert acks == [p.ItemDeleted(p.ITEM_DELETED, "t1", emulated=True)]
    frames, _ = send(w, p.response_create())
    assert frames[0]["clientContent"]["turns"] == [{"role": "user", "parts": [{"text": "seconda"}]}]


def test_skipped_calls_drop_only_until_the_first_turn_end():
    w = wire()
    heard_turn(w)
    feed(w, tool_call(("get_time", {}, "fc_1")))
    _, local = send(w, p.response_create())
    send(w, p.item_delete(local[1].item_id))  # skipped: Gemini continues
    assert feed(w, audio(), GENERATED) == []  # the continuation is dropped, up to its end
    _, local = send(w, p.response_create("Say exactly: Ciao."))
    live = feed(w, audio(said="Ciao."), GENERATED)
    assert {e.response_id for e in live} == {local[0].response_id}
    assert types(live)[-1] == p.RESPONSE_DONE


def test_a_burst_without_activity_start_neither_stalls_nor_misaligns():
    w = wire()
    burst(w)
    events = feed(w, heard("Ciao"), activity("ACTIVITY_END", "1.5s"))
    assert types(events) == [
        p.SPEECH_STARTED,
        p.TRANSCRIPTION_DELTA,
        p.TRANSCRIPTION_COMPLETED,
        p.SPEECH_STOPPED,
    ]
    first = events[0].item_id
    assert events[0].audio_ms is None and {e.item_id for e in events} == {first}
    assert w.counts["activity_starts_missing"] == 1
    feed(w, audio(said="Ciao!"), GENERATED)
    _, local = send(w, p.response_create())
    assert types(local)[-1] == p.RESPONSE_DONE and len(local) > 2
    second = heard_turn(w, start="1.5s", end="3s")
    assert second[0].item_id not in (None, first)
    assert {e.item_id for e in second} == {second[0].item_id}


def test_content_without_any_activity_event_adopts_the_held_activity():
    w = wire()
    burst(w)
    events = feed(w, audio(said="Sì"))
    assert types(events) == [p.SPEECH_STARTED]  # the answer itself is held
    feed(w, GENERATED)
    _, local = send(w, p.response_create())
    assert types(local)[-1] == p.RESPONSE_DONE


def test_a_reply_transcript_after_its_done_is_dropped_not_misplaced():
    """The route follows no response after its done: a late tail is counted, and never
    lands on the next response (documented limit)."""
    w = wire()
    heard_turn(w)
    send(w, p.response_create())
    feed(w, audio(said="Ciao"), GENERATED)
    _, local = send(w, p.response_create("Say exactly: Altro."))
    assert feed(w, {"serverContent": {"outputTranscription": {"text": " amico"}}}) == []
    assert w.counts["late_reply_transcripts"] == 1
    live = feed(w, turn_complete(10), audio(said="Altro."), GENERATED)
    finals = [e for e in live if isinstance(e, p.ReplyTranscript) and e.final]
    assert [e.text for e in finals] == ["Altro."]
    assert finals[0].response_id == local[0].response_id


def test_context_sends_only_the_lines_the_session_lacks():
    w = GeminiWire(MODEL, MAX)
    w.encode(update(instructions="Sei Neve.\nRispondi in italiano."))
    send(w, update(instructions="Sei Neve.\nRispondi in italiano.\nParli con Marcello."))
    frames = burst(w)
    assert frames[0]["clientContent"]["turns"] == [
        {"role": "user", "parts": [{"text": "[Updated instructions]\nParli con Marcello."}]}
    ]
    send(w, update(instructions="Sei Neve.\nRispondi in italiano."))  # a line removed only
    assert burst(w)[0] == {"realtimeInput": {"activityStart": {}}}
    assert w.counts["context_removals_unsent"] == 1


def test_audio_at_another_rate_is_dropped():
    w = wire()
    heard_turn(w)
    send(w, p.response_create())
    other = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"inlineData": {"mimeType": "audio/pcm;rate=16000", "data": b64(PCM_OUT)}}
                ]
            }
        }
    }
    assert feed(w, other) == []
    assert w.counts["audio_rate_mismatch"] == 1
    assert types(feed(w, audio())) == [p.AUDIO_DELTA]


def test_a_text_answer_part_is_counted_and_dropped():
    w = wire()
    heard_turn(w)
    send(w, p.response_create())
    assert feed(w, {"serverContent": {"modelTurn": {"parts": [{"text": "Ciao"}]}}}) == []
    assert w.counts["text_answer_parts"] == 1


@pytest.mark.parametrize("before_start", [True, False])
def test_interrupted_around_the_new_activity_after_a_cancel(before_start):
    """Barge-in: the route cancels, a new burst goes, and Gemini's ``interrupted`` for the
    old turn comes before or after the new ``ACTIVITY_START``: one done for the old answer,
    the new one is held and released whole."""
    w = wire()
    heard_turn(w)
    _, local = send(w, p.response_create())
    old = local[0].response_id
    feed(w, audio())
    assert send(w, p.response_cancel())[1] == [p.ResponseDone(p.RESPONSE_DONE, old, "cancelled")]
    burst(w)
    order = [INTERRUPTED, activity("ACTIVITY_START", "1.689375s")]
    if not before_start:
        order.reverse()
    assert [e for e in feed(w, *order) if not isinstance(e, p.SpeechBoundary)] == []
    feed(w, heard("Basta"), activity("ACTIVITY_END", "2.5s"))
    feed(w, audio(said="Va bene."), GENERATED)
    _, local = send(w, p.response_create())
    assert local[0].response_id != old
    assert types(local)[-1] == p.RESPONSE_DONE and local[-1].status == "completed"
    assert [e.text for e in local if isinstance(e, p.ReplyTranscript) and e.final] == ["Va bene."]


def test_two_calls_one_output_one_deleted():
    w = wire()
    heard_turn(w)
    feed(w, tool_call(("note_tone", {"tone": "calm"}, "fc_a"), ("get_time", {}, "fc_b")))
    _, local = send(w, p.response_create())
    calls = [e for e in local if isinstance(e, p.FunctionCall)]
    send(w, p.function_output("fc_a", '{"ok": true}'))
    frames, _ = send(w, p.item_delete(calls[1].item_id))
    assert frames == [
        {
            "toolResponse": {
                "functionResponses": [
                    {"id": "fc_a", "name": "note_tone", "response": {"ok": True}},
                    {"id": "fc_b", "name": "get_time", "response": {"result": "skipped"}},
                ]
            }
        }
    ]


def test_a_call_item_deleted_after_its_output_keeps_the_output():
    w = wire()
    heard_turn(w)
    feed(w, tool_call(("get_time", {}, "fc_1")))
    _, local = send(w, p.response_create())
    send(w, p.function_output("fc_1", '{"time": "22:30"}'))
    frames, _ = send(w, p.item_delete(local[1].item_id))
    assert frames == []
    frames, _ = send(w, p.response_create())
    assert frames == [
        {
            "toolResponse": {
                "functionResponses": [
                    {"id": "fc_1", "name": "get_time", "response": {"time": "22:30"}}
                ]
            }
        }
    ]


# --------------------------------------------------------------------------- OpenAI identity


POLLEN_FRAMES = [
    {"type": "session.updated", "session": {"audio": {"output": {"format": {"rate": 16000}}}}},
    {"type": "input_audio_buffer.speech_started", "item_id": "item_1", "audio_start_ms": 120},
    {"type": "input_audio_buffer.speech_stopped", "item_id": "item_1", "audio_end_ms": 980},
    {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item_1",
        "transcript": "ciao",
    },
    {"type": "response.created", "response": {"id": "resp_1"}},
    {
        "type": "response.output_audio.delta",
        "response_id": "resp_1",
        "item_id": "it_2",
        "delta": b64(PCM_OUT),
    },
    {
        "type": "response.output_audio_transcript.delta",
        "response_id": "resp_1",
        "item_id": "it_2",
        "delta": "ci",
    },
    {
        "type": "response.function_call_arguments.done",
        "response_id": "resp_1",
        "call_id": "call_1",
        "name": "get_time",
        "arguments": "{}",
    },
    {
        "type": "response.done",
        "response": {"id": "resp_1", "status": "completed", "usage": {"total_tokens": 42}},
    },
    {"type": "conversation.item.deleted", "item_id": "item_1"},
    {"type": "error", "error": {"code": "response_cancel_not_active", "event_id": "ev_3"}},
    {"type": "rate_limits.updated"},
]


@pytest.mark.parametrize("message", POLLEN_FRAMES, ids=lambda m: m["type"])
def test_openai_wire_decodes_exactly_as_parse(message):
    text = json.dumps(message)
    assert OpenAIWire(MAX).decode(text) == [p.parse(text, MAX)]


def test_openai_wire_keeps_parse_refusals():
    w = OpenAIWire(16)
    assert w.decode(b'{"type": "x"}') == [p.Malformed("malformed", "binary")]
    assert w.decode('{"type": "' + "x" * 32 + '"}') == [p.Malformed("malformed", "oversize")]


POLLEN_EVENTS = [
    update(tools=TOOLS),
    p.audio_append(PCM_IN),
    p.audio_commit(),
    p.text_item("ciao", "it_1"),
    p.function_output("call_1", "{}", "it_2"),
    p.item_delete("it_1"),
    p.item_truncate("it_3", 500),
    p.response_create("be kind", "none", event_id="ev_1"),
    p.response_cancel("resp_1", "ev_2"),
]


@pytest.mark.parametrize("event", POLLEN_EVENTS, ids=lambda e: e["type"])
def test_openai_wire_sends_events_unchanged(event):
    before = json.loads(json.dumps(event))
    frames, local = OpenAIWire(MAX).encode(event)
    assert frames == [before] and local == []
    assert p.encode(frames[0]) == p.encode(before)
