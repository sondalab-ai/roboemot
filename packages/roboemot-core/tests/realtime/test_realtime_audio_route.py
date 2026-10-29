"""The route's audio send gate (SPEC 8, Realtime speech route; SPEC 12): one burst per
utterance, sent atomically with the committed opt-in, the gate and the upload budget."""

from __future__ import annotations

import base64
import datetime as dt

import pytest
from roboemot.config import ConfigStack
from roboemot.health.network import NetworkMonitor
from roboemot.realtime import protocol
from roboemot.realtime.route import RealtimeRoute
from test_realtime_route import CHAT, DIRECT, POLICY, RT, FakeRouter

RATE = 16000
SECOND = b"\x01\x00" * RATE  # one second of PCM16 at RATE


class LocalStub:
    """A connected client on a local host that records what it is asked to send."""

    def __init__(self, route, settings, provider, **kw):
        self.route, self.state, self.reason = route, "idle", None
        self.on_event = kw.get("on_event", lambda event: None)
        self.on_close = kw.get("on_close", lambda reason: None)
        self.sent: list[dict] = []

    def connect(self, update):
        self.state = "connected"
        return True

    def snapshot(self):
        return {"local": True, "endpoint": "ws://s2s.local:8765"}

    def send(self, event):
        self.sent.append(event)

    def close(self):
        self.state = "closed"


def audio_stack(rt=None, **app):
    cfg = {
        "llm_routes": [CHAT, RT],
        "voice": {
            "sample_rate_hz": RATE,
            "realtime_route": {"enabled": True, "route": "rt", "input": "audio", **(rt or {})},
        },
        "privacy": {"realtime_audio": True, **app.pop("privacy", {})},
        **app,
    }
    return ConfigStack(app=cfg)


class Day:
    def __init__(self):
        self.day = dt.date(2026, 10, 27)

    def __call__(self):
        return self.day


def audio_route(stack, today=None, upload_path=None):
    made: list[LocalStub] = []

    def factory(*a, **kw):
        made.append(LocalStub(*a, **kw))
        return made[-1]

    r = RealtimeRoute(
        stack,
        lambda: FakeRouter(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        DIRECT,
        extra=lambda: True,
        client_factory=factory,
        today=today or Day(),
        upload_path=upload_path,
    )
    return r, made


def kinds(client):
    return [e["type"] for e in client.sent]


def test_audio_input_needs_the_opt_in_and_then_the_route_may_answer():
    off = audio_stack(privacy={"realtime_audio": False})
    assert audio_route(off)[0].blocked(off.effective()) == "no_opt_in"
    on = audio_stack()
    r, _ = audio_route(on)
    assert r.blocked(on.effective()) is None
    assert r.ready()
    assert r.audio_egress
    snap = r.snapshot()
    assert snap["audio_egress"] is True and snap["gate"] == "addressed"


def test_text_input_never_lets_audio_leave():
    stack = audio_stack({"input": "text"})
    r, made = audio_route(stack)
    assert r.ready()
    assert not r.audio_egress and r.snapshot()["gate"] is None
    assert r.send_audio(SECOND, gate="addressed", instructions="hi") == "input"
    assert made[0].sent == []


def test_a_burst_is_the_instructions_then_the_audio_then_the_commit():
    stack = audio_stack()
    r, made = audio_route(stack)
    assert r.ready()
    assert r.send_audio(SECOND, gate="addressed", instructions="be kind") is None
    assert kinds(made[0]) == [
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
    ]
    session = made[0].sent[0]["session"]
    assert session["instructions"] == "be kind"
    vad = session["audio"]["input"]["turn_detection"]
    assert vad["silence_duration_ms"] == 600
    assert vad["create_response"] is False and vad["interrupt_response"] is False
    assert "tools" not in session  # the turn's own update sets the tools
    assert base64.b64decode(made[0].sent[1]["audio"]) == SECOND
    assert r.snapshot()["upload_s_today"] == 1.0


def test_a_burst_is_cut_into_appends_under_the_message_cap():
    stack = audio_stack({"max_message_bytes": 65_536})
    r, made = audio_route(stack)
    assert r.ready()
    pcm = SECOND * 3  # 96000 bytes
    assert r.send_audio(pcm, gate="addressed", instructions="") is None
    appends = [e for e in made[0].sent if e["type"] == "input_audio_buffer.append"]
    assert len(appends) > 1
    assert all(len(protocol.encode(e)) < 65_536 for e in appends)
    assert b"".join(base64.b64decode(e["audio"]) for e in appends) == pcm


def test_the_opt_in_turned_off_refuses_the_burst_before_the_reload_reaches_the_route():
    stack = audio_stack()
    r, made = audio_route(stack)
    assert r.ready()
    stack.set("privacy.realtime_audio", False, source="test", admin=True)  # committed only
    assert r.send_audio(SECOND, gate="addressed", instructions="") is not None
    assert made[0].sent == []
    assert not r.connected  # a session the committed config would not open is closed


def test_only_the_configured_gate_lets_audio_out():
    stack = audio_stack()
    r, made = audio_route(stack)
    assert r.ready()
    assert r.send_audio(SECOND, gate="live", instructions="") == "gate"
    assert made[0].sent == []


def test_the_upload_budget_refuses_a_burst_that_does_not_fit_and_resets_each_day():
    stack = audio_stack({"daily_upload_budget_s": 2.5})
    today = Day()
    r, made = audio_route(stack, today)
    assert r.ready()
    assert r.send_audio(SECOND * 2, gate="addressed", instructions="") is None
    assert r.send_audio(SECOND, gate="addressed", instructions="") == "upload_budget"
    assert kinds(made[0]).count("input_audio_buffer.commit") == 1
    snap = r.snapshot()
    assert snap["upload_s_today"] == 2.0 and snap["audio_blocked"] == "upload_budget"
    assert r.connected  # text turns still go: only audio is refused
    today.day += dt.timedelta(days=1)
    assert r.send_audio(SECOND, gate="addressed", instructions="") is None
    assert r.snapshot()["upload_s_today"] == 1.0


def test_no_session_sends_nothing():
    stack = audio_stack()
    r, _ = audio_route(stack)
    assert r.send_audio(SECOND, gate="addressed", instructions="") == "no_session"


def test_send_control_carries_no_data():
    stack = audio_stack()
    r, made = audio_route(stack)
    assert r.ready()
    assert r.send_control(protocol.response_cancel())
    assert r.send_control(protocol.item_delete("i"))
    assert r.send_control(protocol.item_truncate("i", 10))
    assert not r.send_control(protocol.audio_append(SECOND))
    assert not r.send_control(protocol.audio_commit())
    assert not r.send_control(protocol.text_item("hello"))
    assert kinds(made[0]) == [
        "response.cancel",
        "conversation.item.delete",
        "conversation.item.truncate",
    ]


def test_input_events_of_the_current_session_reach_the_audio_input_only():
    stack = audio_stack()
    r, made = audio_route(stack)
    seen = []
    r.input = type("Sink", (), {"on_event": lambda self, ev, gen: seen.append((ev, gen))})()
    assert r.ready()
    started = protocol.SpeechBoundary(protocol.SPEECH_STARTED, "a", 0)
    done = protocol.InputTranscript(protocol.TRANSCRIPTION_COMPLETED, "a", "ciao", True)
    made[0].on_event(started)
    made[0].on_event(done)
    gen = r.generation  # each event names the session it is of
    assert seen == [(started, gen), (done, gen)]
    old = made[0]
    r.close_soon("conversation_end")
    old.on_event(done)  # a late event of a detached session
    assert seen == [(started, gen), (done, gen)]


def test_the_session_update_uses_the_server_silence():
    stack = audio_stack({"server_silence_ms": 800, "close_silence_ms": 1200})
    r, _ = audio_route(stack)
    update = r.session_update(stack.effective(), "x")
    vad = update["session"]["audio"]["input"]["turn_detection"]
    assert vad["silence_duration_ms"] == 800


def test_audio_egress_never_reads_the_configuration_stack():
    """The Live tab polls it: it is kept current by the route's own config steps (SPEC 11A)."""
    stack = audio_stack()
    r, _ = audio_route(stack)
    assert r.ready() and r.audio_egress
    reads = []
    effective = stack.effective
    stack.effective = lambda *a, **kw: (reads.append(1), effective(*a, **kw))[1]
    for _ in range(5):
        assert r.audio_egress
    assert reads == []
    stack.effective = effective
    stack.set("privacy.realtime_audio", False, source="test", admin=True)
    r.apply_config(stack.effective())
    assert not r.audio_egress


# ------------------------------------------------------------------ fix round 1


def test_a_hold_evaluated_with_the_send_refuses_the_burst():
    """B-1, M-1, I-1: asleep, speaking or a cancelled job are decided inside the gate."""
    stack = audio_stack()
    r, made = audio_route(stack)
    assert r.ready()
    for why in ("asleep", "speaking", "cancelled"):
        assert r.send_audio(SECOND, gate="addressed", instructions="", hold=lambda w=why: w) == why
    assert made[0].sent == []
    assert r.snapshot()["upload_s_today"] == 0.0  # nothing reserved
    assert r.snapshot()["audio_blocked"] == "cancelled"


def test_a_realtime_turn_holds_audio_until_it_ends():
    """M-7: no burst between a turn's response.create calls (its tool rounds)."""
    stack = audio_stack()
    r, made = audio_route(stack)
    assert r.ready()
    r.hold_input()
    assert r.send_audio(SECOND, gate="addressed", instructions="") == "turn"
    r.release_input()
    assert r.send_audio(SECOND, gate="addressed", instructions="") is None


def test_the_burst_reports_the_session_it_went_to():
    """M-3, I-7: under the gate, with the send, the session the burst goes to (the input reads
    where that session's buffer ends in the same step)."""
    stack = audio_stack()
    r, _ = audio_route(stack)
    assert r.ready()
    spans = []
    assert (
        r.send_audio(SECOND, gate="addressed", instructions="", track=lambda *a: spans.append(a))
        is None
    )
    gen = r.generation
    r.close_soon("conversation_end")
    assert r.ready()
    assert (
        r.send_audio(SECOND, gate="addressed", instructions="", track=lambda *a: spans.append(a))
        is None
    )
    assert spans == [(gen,), (r.generation,)] and r.generation == gen + 1


def test_the_upload_budget_resets_only_when_the_date_moves_forward():
    """I-4: a clock that jumps back (no RTC, an NTP fix, a time zone) is not a new day."""
    stack = audio_stack({"daily_upload_budget_s": 2.5})
    today = Day()
    r, _ = audio_route(stack, today)
    assert r.ready()
    assert r.send_audio(SECOND * 2, gate="addressed", instructions="") is None
    today.day -= dt.timedelta(days=1)
    assert r.send_audio(SECOND, gate="addressed", instructions="") == "upload_budget"
    today.day += dt.timedelta(days=1)
    assert r.send_audio(SECOND, gate="addressed", instructions="") == "upload_budget"
    today.day += dt.timedelta(days=1)
    assert r.send_audio(SECOND, gate="addressed", instructions="") is None


def test_the_upload_budget_survives_a_restart(tmp_path):
    """I-4: the day's seconds are kept in the data folder (0600, written atomically)."""
    import json
    import stat

    path = tmp_path / "realtime_upload.json"
    stack = audio_stack({"daily_upload_budget_s": 2.5})
    today = Day()
    r, _ = audio_route(stack, today, upload_path=path)
    assert r.ready()
    assert r.send_audio(SECOND * 2, gate="addressed", instructions="") is None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"day": "2026-10-27", "seconds": 2.0}
    again, _ = audio_route(stack, today, upload_path=path)
    assert again.ready()
    assert again.send_audio(SECOND, gate="addressed", instructions="") == "upload_budget"
    path.write_text("not json")
    third, _ = audio_route(stack, today, upload_path=path)  # unreadable: counted as spent
    assert third.ready()
    assert third.send_audio(SECOND, gate="addressed", instructions="") == "upload_budget"
    assert third.snapshot()["upload_s_today"] == 2.5  # JSON: the budget, never infinity


def test_a_ready_with_an_older_turn_config_does_not_hide_the_egress():
    """M-8: the flag follows the committed configuration, not a turn's copy."""
    stack = audio_stack(privacy={"realtime_audio": False})
    old = stack.effective()
    r, _ = audio_route(stack)
    stack.set("privacy.realtime_audio", True, source="test", admin=True)
    r.apply_config(stack.effective())
    assert r.audio_egress
    r.ready(old)
    assert r.audio_egress


def test_a_request_for_another_session_is_refused():
    """M-6: the response.create of a turn whose words are on an older session is not sent."""
    stack = audio_stack()
    r, made = audio_route(stack)
    assert r.ready()
    gen = r.generation
    assert r.request(None, "x", generation=gen + 1) is None
    assert made[0].sent == []
    assert r.request(None, "x", generation=gen) is not None


def test_a_reconnect_keeps_the_day_upload_count():
    """M-13: the count lives on the route, not on a session."""
    stack = audio_stack({"daily_upload_budget_s": 2.5})
    r, _ = audio_route(stack)
    assert r.ready()
    assert r.send_audio(SECOND * 2, gate="addressed", instructions="") is None
    r.close_soon("conversation_end")
    assert r.ready()
    assert r.send_audio(SECOND, gate="addressed", instructions="") == "upload_budget"


# ------------------------------------------------------------------ fix round 2


@pytest.mark.parametrize(
    "seconds", ["-1e9", "NaN", "Infinity", "-Infinity", '"two"', "[1]", "true", "null"]
)
def test_a_count_that_is_not_a_finite_non_negative_number_counts_the_day_as_spent(
    tmp_path, seconds
):
    """I-N3: a stored value that would open the cap fails closed."""
    import json

    path = tmp_path / "realtime_upload.json"
    path.write_text('{"day": "2026-10-27", "seconds": ' + seconds + "}")
    stack = audio_stack({"daily_upload_budget_s": 2.5})
    r, _ = audio_route(stack, Day(), upload_path=path)
    assert r.ready()
    assert r.send_audio(SECOND, gate="addressed", instructions="") == "upload_budget"
    snap = r.snapshot()
    assert snap["upload_s_today"] == 2.5
    json.dumps(snap, allow_nan=False)  # strict JSON


def test_the_count_is_written_as_strict_json(tmp_path):
    import json

    path = tmp_path / "realtime_upload.json"
    stack = audio_stack()
    r, _ = audio_route(stack, Day(), upload_path=path)
    assert r.ready()
    assert r.send_audio(SECOND, gate="addressed", instructions="") is None
    json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-strict {c}"))
    r._save_upload(dt.date(2026, 10, 27), float("nan"))  # never written
    assert json.loads(path.read_text())["seconds"] == 1.0


def test_a_future_dated_count_is_today_with_its_seconds(tmp_path):
    """A stored day after today (a clock that jumped forward, then back) is taken as today:
    its seconds still count, but only until the date really moves forward."""
    path = tmp_path / "realtime_upload.json"
    path.write_text('{"day": "2099-01-01", "seconds": 2.0}')
    stack = audio_stack({"daily_upload_budget_s": 2.5})
    today = Day()
    r, _ = audio_route(stack, today, upload_path=path)
    assert r.ready()
    assert r.send_audio(SECOND, gate="addressed", instructions="") == "upload_budget"
    today.day += dt.timedelta(days=1)
    assert r.send_audio(SECOND, gate="addressed", instructions="") is None


def test_confirmed_deletes_and_truncates_are_counted():
    """SR-5 fix round 1, I2: sent truncates, and the server's confirmations of deletes and
    truncates, are counted (the HIL asserts confirmed >= sent)."""
    stack = audio_stack()
    r, _ = audio_route(stack)
    assert r.ready()
    assert r.send_control(protocol.item_truncate("i", 10))
    r._on_event(protocol.ItemDeleted(protocol.ITEM_DELETED, "i"))
    r._on_event(protocol.ItemTruncated(protocol.ITEM_TRUNCATED, "i"))
    snap = r.snapshot()
    assert snap["items_truncated"] == 1
    assert snap["items_delete_confirmed"] == 1 and snap["items_truncate_confirmed"] == 1


@pytest.mark.parametrize(("capture_hz", "sent"), [(16000, None), (24000, 24000)])
def test_the_input_rate_is_the_servers_native_one_at_16_khz(capture_hz, sent):
    """Robot run 2026-10-28: Pollen's server refused session.update with input rate 16000
    (unknown_or_invalid_event) and accepted rate null, its native 16 kHz (as Pollen's app sends
    it); another capture rate stays explicit."""
    stack = audio_stack()
    r, _ = audio_route(stack)
    cfg = stack.effective()
    cfg = cfg.model_copy(
        update={"voice": cfg.voice.model_copy(update={"sample_rate_hz": capture_hz})}
    )
    update = r.session_update(cfg, "x")
    assert update["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": sent}


# ------------------------------------------------------------------ input resets


def test_reset_session_resets_only_the_current_session():
    """An audio burst the server may still report on resets its session: never a stale
    generation, never with no session; counted, and the new session has a new generation."""
    from rt_rig import wait_for

    r, made = audio_route(audio_stack())
    assert r.reset_session(r.generation) is False  # no session yet
    assert r.ready()
    gen = r.generation
    assert r.reset_session(gen - 1) is False  # an older session: nothing done
    assert r.snapshot()["input_resets"] == 0 and len(made) == 1
    assert r.reset_session(gen) is True
    assert wait_for(lambda: r.connected and not r.opening())
    assert r.generation == gen + 1 and len(made) == 2
    assert r.reset_session(gen) is False  # already reset: that session is gone
    snap = r.snapshot()
    assert snap["input_resets"] == 1 and snap["resets"] == 0


def test_a_reset_never_reopens_a_session_closed_for_sleep():
    """SPEC 12A: the robot falls asleep between the reset's detach and its reconnect; the
    reconnect is refused (the close came after the reset), the session stays closed."""
    from rt_rig import wait_for

    r, made = audio_route(audio_stack())
    assert r.ready()
    gen = r.generation
    ready = r.ready

    def asleep_first(*a, **kw):
        r.close_soon("asleep")
        return ready(*a, **kw)

    r.ready = asleep_first
    assert r.reset_session(gen)
    assert wait_for(lambda: not r.opening())
    assert len(made) == 1 and r.generation == gen and not r.connected
