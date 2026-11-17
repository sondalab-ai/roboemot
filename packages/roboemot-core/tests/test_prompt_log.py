"""SPEC 11A, 12: the LLM prompts ring of the Live tab. In memory only, opt-in, bounded by count
and text length; nothing is kept while the flag is off; audio is summarized, never kept."""

from __future__ import annotations

import base64
import json
import threading

from roboemot.config.schema import LiveConfig
from roboemot.llm.prompt_log import PromptRing, render_chat

AUDIO = base64.b64encode(b"\x01\x02" * 1600).decode()  # 3200 bytes of PCM


def ring(**kw):
    cfg = LiveConfig(enabled=kw.pop("enabled", True), prompts=kw.pop("prompts", True), **kw)
    return PromptRing(cfg, wall=lambda: 5000.0)


def chat(r, text="hello", route="cloud", model="gpt-x"):
    r.chat(
        route,
        model,
        [{"role": "system", "content": "be kind"}, {"role": "user", "content": text}],
        None,
    )


def test_off_keeps_nothing_at_all():
    for kw in ({"prompts": False}, {"enabled": False}):
        r = ring(**kw)
        chat(r)
        r.frame("rt", "gemini_live", {"setup": {"model": "m"}})
        snap = r.snapshot(0)
        assert snap["entries"] == [] and snap["rev"] == 0 and not snap["enabled"]
        assert r.size == 0


def test_a_chat_entry_has_role_lines_and_tool_names():
    r = ring()
    tools = [{"type": "function", "function": {"name": "set_timer", "parameters": {}}}]
    msgs = [
        {"role": "system", "content": "You are Robo."},
        {"role": "user", "content": "line one\n  indented"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "set_timer", "arguments": '{"s":5}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    r.chat("cloud_chat", "gpt-x", msgs, tools)
    (entry,) = r.snapshot(0)["entries"]
    assert entry == {
        "seq": 1,
        "t": 5000.0,
        "route": "cloud_chat",
        "via": "gpt-x",
        "label": "chat.completions",
        "text": (
            "[system] You are Robo.\n[user] line one\n  indented\n"
            '[assistant] -> set_timer({"s":5})\n[tool] ok\n[tools] set_timer'
        ),
    }


def test_cursor_returns_only_new_entries_and_epoch_restarts():
    r = ring()
    chat(r, "a")
    first = r.snapshot(0)
    chat(r, "b")
    later = r.snapshot(first["rev"], first["epoch"])
    assert [e["seq"] for e in later["entries"]] == [2]
    assert r.snapshot(later["rev"], later["epoch"])["entries"] == []
    # a cursor from another life (or ahead of the ring) gets everything
    assert len(r.snapshot(later["rev"], "deadbeef.0")["entries"]) == 2
    assert len(r.snapshot(999)["entries"]) == 2


def test_clear_changes_the_epoch_and_keeps_seq_rising():
    r = ring()
    chat(r)
    before = r.snapshot(0)
    r.clear()
    after = r.snapshot(before["rev"], before["epoch"])
    assert after["epoch"] != before["epoch"] and after["entries"] == []
    chat(r)
    assert r.snapshot(0)["entries"][0]["seq"] == 2


def test_the_count_bound_drops_the_oldest():
    r = ring(prompts_max_entries=3)
    for i in range(5):
        chat(r, f"t{i}")
    snap = r.snapshot(0)
    assert [e["seq"] for e in snap["entries"]] == [3, 4, 5]
    assert snap["min_seq"] == 3 and snap["max_entries"] == 3


def test_a_long_entry_is_cut_with_a_visible_marker():
    r = ring(prompts_max_chars=1000)
    chat(r, "x" * 5000)
    full = len("[system] be kind\n[user] ") + 5000
    text = r.snapshot(0)["entries"][0]["text"]
    assert text.startswith("[system] be kind\n[user] xxx")
    assert text.endswith(f"\n... [truncated {full - 1000} chars]")
    assert len(text) < 1100


def test_one_poll_carries_the_newest_entries_within_its_budget():
    r = ring(prompts_max_chars=10000, prompts_response_max_chars=10000)
    for i in range(4):
        chat(r, str(i) * 4000)
    assert [e["seq"] for e in r.snapshot(0)["entries"]] == [3, 4]


def test_turning_the_flag_off_clears_and_stops():
    r = ring()
    chat(r)
    r.apply_config(LiveConfig(enabled=True, prompts=False))
    assert r.size == 0
    chat(r)
    assert r.size == 0
    r.apply_config(LiveConfig(enabled=True, prompts=True))
    chat(r)
    assert r.size == 1


def test_openai_audio_appends_collapse_into_one_growing_summary():
    r = ring()
    r.frame(
        "rt", "openai_realtime", {"type": "session.update", "session": {"audio": {"input": {}}}}
    )
    first = r.snapshot(0)
    for _ in range(3):
        r.frame("rt", "openai_realtime", {"type": "input_audio_buffer.append", "audio": AUDIO})
    snap = r.snapshot(first["rev"], first["epoch"])
    (entry,) = snap["entries"]
    assert entry["label"] == "audio" and entry["text"] == "[audio 9600 bytes in 3 frames]"
    r.frame("rt", "openai_realtime", {"type": "input_audio_buffer.commit"})
    r.frame("rt", "openai_realtime", {"type": "input_audio_buffer.append", "audio": AUDIO})
    labels = [e["label"] for e in r.snapshot(0)["entries"]]
    assert labels == ["session.update", "audio", "input_audio_buffer.commit", "audio"]
    # the session's audio settings object is not mistaken for a payload
    assert '"input": {}' in r.snapshot(0)["entries"][0]["text"]
    assert AUDIO not in json.dumps(r.snapshot(0))


def test_gemini_audio_is_summarized_and_setup_is_pretty_json():
    r = ring()
    setup = {"setup": {"model": "models/x", "systemInstruction": {"parts": [{"text": "Hi"}]}}}
    r.frame("gem", "gemini_live", setup)
    r.frame("gem", "gemini_live", {"realtimeInput": {"activityStart": {}}})
    for _ in range(2):
        r.frame(
            "gem",
            "gemini_live",
            {"realtimeInput": {"audio": {"data": AUDIO, "mimeType": "audio/pcm"}}},
        )
    entries = r.snapshot(0)["entries"]
    assert [e["label"] for e in entries] == ["setup", "realtimeInput", "audio"]
    assert entries[0]["via"] == "gemini_live"
    assert json.loads(entries[0]["text"]) == setup and "\n  " in entries[0]["text"]
    assert entries[2]["text"] == "[audio 6400 bytes in 2 frames]"
    assert AUDIO not in json.dumps(entries)


def test_binary_parts_are_redacted_wherever_they_sit():
    r = ring()
    part = {"inlineData": {"data": AUDIO, "mimeType": "image/jpeg"}}
    r.frame("gem", "gemini_live", {"clientContent": {"turns": [{"parts": [part]}]}})
    text = r.snapshot(0)["entries"][0]["text"]
    assert AUDIO not in text and '"inlineData": "[binary]"' in text


def test_content_parts_keep_text_and_redact_the_rest():
    text = render_chat(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}, {"type": "x", "audio": AUDIO}],
            }
        ],
        None,
    )
    assert text.startswith("[user] hi\n") and AUDIO not in text


def test_writers_on_threads_and_a_reader_agree():
    r = ring(prompts_max_entries=500)

    def write(n):
        for i in range(50):
            r.frame(f"rt{n}", "openai_realtime", {"type": "conversation.item.create", "i": i})

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = r.snapshot(0)
    assert len(snap["entries"]) == 200 and snap["rev"] == 200


def test_repr_shows_no_entry():
    r = ring()
    chat(r, "secret words")
    assert "secret" not in repr(r)
