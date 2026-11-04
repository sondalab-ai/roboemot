"""Decision events for the Live tab's "What's happening" panel (SPEC 11A, 12, 12A; CV-10): each
decision point publishes one ``diag.event`` with its reason code, the ring keeps the last ones,
and nothing in an event is text, audio, an embedding or a key."""

from __future__ import annotations

import math

import numpy as np
import pytest
import test_cues
from roboemot.bus import TOPICS, Bus
from roboemot.config.schema import CueConfig, LiveConfig
from roboemot.diag import (
    FIELDS,
    REASONS,
    ROUTE_REASONS,
    SERVER_CODES,
    TOPIC,
    BusEvents,
    EventRing,
    clean,
    event_kind,
    exception_code,
)
from roboemot.motion.primitives import Primitive
from roboemot.voice.cues import Cues, CueSound
from test_voice_listener import BusyCpu, facing_rig

SECRET = "sk-test-0123456789abcdef"
TEXT = "ciao Neve come stai oggi"


class Recorder:
    """An event sink that keeps what it was given."""

    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def __call__(self, kind, reason, **fields):
        self.events.append((kind, reason, fields))

    def of(self, kind, reason=None):
        return [f for k, r, f in self.events if k == kind and reason in (None, r)]


def live(**kw) -> LiveConfig:
    return LiveConfig(enabled=True, **kw)


def published(bus: Bus) -> list[dict]:
    out: list[dict] = []
    bus.subscribe(TOPIC, lambda m: out.append(m.payload))
    return out


# ------------------------------------------------------------------ the sink and the ring


def test_the_topic_is_a_bus_topic():
    assert TOPIC in TOPICS


def test_an_event_keeps_only_declared_fields_with_allowed_values():
    bus = Bus()
    got = published(bus)
    BusEvents(bus, wall=lambda: 1000.0)(
        "cue",
        "no_voice",
        cue="fallback",
        failure="refused",
        detail="function_call_output_pending",
        yaw_deg=12.3456,
        robot_playing=True,
        rounds=3,
        reset=None,
        text=TEXT,
        key=SECRET,
        name="mia",
        audio=np.zeros(4, np.float32),
        embedding=[0.1, 0.2],
        limit_deg=math.nan,
    )
    assert got == [
        {
            "kind": "cue", "reason": "no_voice", "t": 1000.0, "cue": "fallback",
            "failure": "refused", "detail": "function_call_output_pending", "yaw_deg": 12.35,
            "robot_playing": True, "rounds": 3, "reset": None,
        }
    ]  # fmt: skip


LOWERCASE_LEAKS = ("sk0123456789abcdef", "ciao", "mia", "telefono")  # a key, a word, a name


@pytest.mark.parametrize("leak", LOWERCASE_LEAKS)
def test_a_lowercase_word_name_or_key_never_passes_a_code_field(leak):
    bus = Bus()
    got = published(bus)
    fields = {name: leak for name, spec in FIELDS.items() if isinstance(spec, tuple)}
    BusEvents(bus)("session", "lost", **fields, cue_text=leak)
    BusEvents(bus)("turn", leak, detail=leak)
    BusEvents(bus)(leak, "reset")
    for ev in got:
        assert leak not in ev.values()
    assert {v for k, v in got[0].items() if k in fields} == {"other"}
    assert (got[1]["reason"], got[1]["detail"]) == ("other", "other")
    assert (got[2]["kind"], got[2]["reason"]) == ("other", "other")


def test_a_number_or_flag_field_takes_nothing_else():
    assert clean({"yaw_deg": "40", "moving": 1, "rounds": True, "level_db": math.inf}) == {}


def test_every_kind_and_reason_of_the_catalogue_passes_and_nothing_else():
    for kind, reasons in REASONS.items():
        for reason in reasons:
            assert event_kind(kind, reason) == (kind, reason)
        assert event_kind(kind, "not_a_reason") == (kind, "other")


def test_the_route_reasons_cover_the_routes_own_sets():
    from roboemot.realtime import route

    assert set(route.STATES) | set(route.BLOCKS) | set(route.OWN_CLOSES) <= set(ROUTE_REASONS)
    assert route.PENDING_CALLS in SERVER_CODES


def test_an_exception_maps_to_a_fixed_code():
    assert exception_code(TimeoutError()) == "timeout"
    assert exception_code(type("SecretNameError", (Exception,), {})()) == "other"


def test_a_failing_subscriber_never_reaches_the_producer(caplog):
    bus = Bus()
    bus.subscribe(TOPIC, lambda m: 1 / 0)
    BusEvents(bus)("cue", "offline")  # no exception
    assert "ZeroDivisionError" in caplog.text


TURNS = REASONS["turn"]


def ring_with(cfg: LiveConfig, n: int, *, reasons=TURNS) -> EventRing:
    bus = Bus()
    ring = EventRing(cfg)
    ring.attach(bus)
    events = BusEvents(bus)
    for i in range(n):
        events("turn", reasons[i])
    return ring


def test_the_ring_keeps_the_last_n_newest_first():
    ring = ring_with(live(activity_events=5), 8)
    rows = ring.events()
    assert [r["reason"] for r in rows] == list(reversed(TURNS[3:8]))
    assert [r["seq"] for r in rows] == sorted((r["seq"] for r in rows), reverse=True)
    assert all("_at" not in r for r in rows)


def test_a_repeated_decision_is_one_row_with_a_count():
    ring = ring_with(live(activity_events=5), 4, reasons=["ok", "error", "error", "error"])
    rows = ring.events()
    assert [(r["reason"], r["count"]) for r in rows] == [("error", 3), ("ok", 1)]


def test_a_repeat_with_other_labels_is_its_own_row_and_numbers_may_differ():
    bus = Bus()
    ring = EventRing(live())
    ring.attach(bus)
    events = BusEvents(bus)
    events("session", "reset", why="expired", in_row=1)
    events("session", "reset", why="input_reset", in_row=2)
    events("unanswered", "not_facing", facing="not_facing", yaw_deg=40.0)
    events("unanswered", "not_facing", facing="not_facing", yaw_deg=55.0)
    events("unanswered", "not_facing", facing="cpu_high", yaw_deg=None)
    rows = [(r["reason"], r.get("why") or r.get("facing"), r["count"]) for r in ring.events()]
    assert rows == [
        ("not_facing", "cpu_high", 1),
        ("not_facing", "not_facing", 2),
        ("reset", "input_reset", 1),
        ("reset", "expired", 1),
    ]
    assert ring.events()[1]["yaw_deg"] == 55.0  # the newest numbers


def test_the_ring_collects_only_while_the_live_tab_and_the_panel_are_on_and_resizes():
    ring = ring_with(LiveConfig(), 3)  # webui.live.enabled off by default
    assert ring.events() == []
    ring = ring_with(live(activity=False), 3)
    assert ring.events() == []
    ring = ring_with(live(activity_events=10), 8)
    ring.apply_config(live(activity_events=5))
    assert len(ring.events()) == 5
    ring.apply_config(live(activity=False))
    assert ring.events() == []  # off forgets them


def test_the_ring_takes_publishers_on_many_threads_while_it_resizes_and_clears():
    import threading

    bus = Bus()
    ring = EventRing(live(activity_events=50))
    ring.attach(bus)
    events = BusEvents(bus)
    stop = threading.Event()

    def publish(reason):
        while not stop.is_set():
            events("turn", reason)

    def reload():
        for k in range(200):
            ring.apply_config(live(activity_events=5 + k % 40, activity=k % 7 != 0))

    threads = [threading.Thread(target=publish, args=(r,)) for r in TURNS[:6]]
    for t in threads:
        t.start()
    reload()
    stop.set()
    for t in threads:
        t.join()
    rows = ring.events()
    assert len(rows) <= ring._events.maxlen
    seqs = [r["seq"] for r in rows]
    assert seqs == sorted(seqs, reverse=True) and len(set(seqs)) == len(seqs)
    assert all(r["count"] >= 1 and r["reason"] in TURNS for r in rows)
    ring.apply_config(live(activity=False))
    events("turn", "ok")
    assert ring.events() == []  # off after the last reload: nothing is kept


def test_age_of_finds_the_newest_event_of_a_kind():
    bus = Bus()
    clock = [100.0]
    ring = EventRing(live(), clock=lambda: clock[0])
    ring.attach(bus)
    BusEvents(bus, clock=lambda: 99.0)("bargein", "cut")
    assert ring.age_of("bargein", "cut") == 1.0
    assert ring.age_of("bargein", "released") is None


def test_an_own_sound_cut_is_not_the_live_cut_status():
    """SPEC 7B, 8: speech over a hum cuts the hum, not the robot: the Live status ``cut``
    reads ``bargein``/``cut`` only."""
    bus = Bus()
    ring = EventRing(live(), clock=lambda: 100.0)
    ring.attach(bus)
    BusEvents(bus, clock=lambda: 99.0)("bargein", "cut_own")
    assert ring.age_of("bargein", "cut") is None
    assert ring.events()[0]["reason"] == "cut_own"  # a known reason, not "other"


# ------------------------------------------------------------------ cues


def prim() -> Primitive:
    return test_cues.prim("cue")


class Out:
    def __init__(self, job=1, aside=True):
        self.job, self.aside = job, aside

    def play(self, samples, rate_hz, volume_db=0.0):
        return self.job

    def play_aside(self, samples, rate_hz, volume_db=0.0):
        return self.aside


SOUND = CueSound(np.zeros(160, np.float32), 16000)


def cues(output=None, muted=False, quiet=True, **cfg) -> tuple[Cues, Recorder]:
    c = Cues(
        CueConfig(**cfg),
        gesture=lambda name: prim(),
        sounds={"fallback": SOUND, "listening": SOUND},
        output=lambda: output,
        muted=lambda: muted,
        quiet=lambda: quiet,
    )
    rec = Recorder()
    c.events = rec
    return c, rec


def test_a_played_cue_publishes_its_reason_and_detail():
    c, rec = cues(Out())
    assert c.play("fallback", "no_voice", failure="refused", detail="function_call_output_pending")
    assert rec.events == [
        ("cue", "no_voice", {
            "cue": "fallback", "result": "played", "failure": "refused",
            "detail": "function_call_output_pending",
        })
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("kw", "result"),
    [({"muted": True}, "muted"), ({"enabled": False}, "disabled")],
)
def test_a_cue_that_does_not_play_says_why(kw, result):
    c, rec = cues(Out(), **kw)
    assert not c.play("fallback", "error")
    assert rec.of("cue", "error")[0]["result"] == result


def test_a_cue_whose_sound_the_speaker_dropped_says_gesture_only():
    c, rec = cues(Out(job=None))
    c.play("fallback", "no_route")
    assert rec.of("cue")[0]["result"] == "sound_dropped"


def test_the_listening_blip_publishes_played_skipped_or_dropped():
    c, rec = cues(Out())
    c.listening_begin(1.0)
    c, rec2 = cues(Out(), quiet=False)
    c.listening_begin(1.0)
    c, rec3 = cues(Out(aside=False))
    c.listening_begin(1.0)
    results = [r.of("cue", "audio_sent")[0]["result"] for r in (rec, rec2, rec3)]
    assert results == ["played", "skipped", "dropped"]
    assert rec.of("cue")[0]["cue"] == "listening"


def test_the_runtime_cue_callers_pass_their_reason():
    from roboemot.runtime.startup import audio_cues
    from roboemot.voice.listener import AUDIO_NOT_HEARD

    c, rec = cues(Out())
    c._sounds["not_heard"] = SOUND
    audio_cues(c)(AUDIO_NOT_HEARD, 1.0)
    assert rec.of("cue", "no_transcript")[0]["cue"] == "not_heard"


# ------------------------------------------------------------------ the wake decision


def test_not_facing_publishes_the_rule_with_the_yaw_and_the_limit():
    r, _ = facing_rig("che bel video", math.radians(40.0))
    rec = Recorder()
    r.listener.events = rec
    r.say()
    assert not r.utterances[0].addressed
    (ev,) = rec.of("unanswered", "not_facing")
    assert ev["yaw_deg"] == pytest.approx(40.0)
    assert ev["limit_deg"] == r.cfg.voice.facing.max_yaw_deg
    assert ev["facing"] == "not_facing" and ev["policy"] == "never"


def test_an_addressed_utterance_publishes_no_rejection():
    r, _ = facing_rig("come stai", math.radians(5.0))
    rec = Recorder()
    r.listener.events = rec
    r.say()
    assert r.utterances[0].addressed and rec.events == []


def test_a_face_with_no_yaw_publishes_no_facing_data():
    r, _ = facing_rig("che bel video", None)
    rec = Recorder()
    r.listener.events = rec
    r.say()
    (ev,) = rec.of("unanswered", "no_facing_data")
    assert ev["yaw_deg"] is None


def test_a_voice_off_every_face_publishes_no_face_at_voice():
    r, _ = facing_rig("la la la", math.radians(5.0), bearing=FACE_OFF)
    rec = Recorder()
    r.listener.events = rec
    r.say()
    assert len(rec.of("unanswered", "no_face_at_voice")) == 1


FACE_OFF = 0.2 + math.radians(60.0)


def test_no_wake_word_publishes_no_window():
    r, _ = facing_rig("che bel video", math.radians(5.0), policy="always")
    rec = Recorder()
    r.listener.events = rec
    r.say()
    (ev,) = rec.of("unanswered", "no_window")
    assert ev["policy"] == "always"


def test_the_cpu_fallback_of_the_facing_check_is_published():
    r, _ = facing_rig("aspetta", math.radians(40.0), cpu=BusyCpu())
    rec = Recorder()
    r.listener.events = rec
    r.say()
    assert r.utterances[0].addressed  # heard at the face: the voice-at-face check decides
    (ev,) = rec.of("facing", "cpu_high")
    assert ev["path"] == "heard" and ev["cpu_percent"] > ev["cpu_max_percent"]


# ------------------------------------------------------------------ barge-in


def test_a_barge_in_cut_and_a_released_duck_are_published(monkeypatch):
    import roboemot.voice.bargein as bargein
    from test_bargein import FRAME, RATE, hiss, onset_run, user_voice

    rec = Recorder()
    monkeypatch.setattr(bargein, "no_events", rec)
    stops, _ = onset_run([0.5, 0.6, 0.55, 0.3], hiss(4), keyword_enabled=False)
    assert len(stops) == 1
    (cut,) = rec.of("bargein", "cut")
    assert cut["confirmed"] == "onset" and cut["fired"] == "onset"
    assert set(cut) == {*bargein.EVENT_FIELDS, "ducking"} and cut["ducking"] is True
    probs = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]  # a cough: ducked, then released
    onset_run(probs, user_voice(8 * FRAME / RATE, 0.1), keyword_enabled=False)
    (released,) = rec.of("bargein", "released")
    assert released["released_by"] == "quiet" and released["confirmed"] is None


# ------------------------------------------------------------------ catalogue and privacy


def test_every_published_reason_and_status_has_a_sentence_in_both_languages():
    from roboemot.runtime.live import STATUSES
    from roboemot.webui.strings import EN, IT

    needed = [f"live_status_{s}" for s in STATUSES]
    needed += [f"live_note_{c}" for c in ("offline", "cloud_down")]
    for kind, reasons in REASONS.items():
        for reason in reasons:
            needed.append(f"live_why_{reason}" if kind == "cue" else f"live_ev_{kind}_{reason}")
    needed += [f"live_ev_cue_{c}" for c in ("fallback", "not_heard", "offline", "listening")]
    needed += [
        f"live_fail_{f}" for f in ("refused", "closed", "no_session", "not_ready", "exception")
    ]
    needed += [
        f"live_res_{r}"
        for r in ("disabled", "muted", "unsupported", "sound_dropped", "skipped", "dropped")
    ]
    needed += [f"live_conv_{c}" for c in ("none", "in_progress", "addressed")]
    needed += [f"live_voice_{v}" for v in ("realtime", "local")]
    for lang in (EN, IT):
        assert [k for k in needed if not lang.get(k)] == []


def test_every_event_of_these_scenarios_is_codes_and_numbers_only():
    """Scans what the decision points above publish through the real sink: no transcript word,
    no key, no array, no free text."""
    bus = Bus()
    got = published(bus)
    events = BusEvents(bus)
    r, _ = facing_rig(TEXT, math.radians(40.0))
    r.listener.events = events
    r.say()
    c, _ = cues(Out())
    c.events = events
    c.play("fallback", "no_voice", failure="refused", detail=SECRET)
    assert got
    words = set(TEXT.lower().split())
    for ev in got:
        for key, value in ev.items():
            assert isinstance(value, bool | int | float | str | type(None)), key
            if isinstance(value, str):
                assert value not in words and " " not in value and SECRET not in value
    assert clean({"detail": SECRET}) == {"detail": "other"}
