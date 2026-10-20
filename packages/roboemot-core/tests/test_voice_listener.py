"""Voice listener and fast path with a fake ASR and the energy VAD (SPEC 8 stages 2-4)."""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from reachy_config import reachy_config
from roboemot.affect.runner import AffectRunner
from roboemot.bus import Bus
from roboemot.config.schema import APPRAISAL_EVENTS, RoboEmotConfig
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.asr import ASREvent, resolve_language
from roboemot.voice.fastpath import EVENT_FOR, FastPath
from roboemot.voice.listener import UtteranceInfo, VoiceListener
from roboemot.voice.vad import Endpointer, EnergyVAD

RATE = 16000
FRAME = 512
CHUNK = 1024  # two VAD frames
T0 = 1000.0


class FakeASR:
    """Scripted: the i-th ``accept`` of an utterance reports ``script[i]`` (last repeats)."""

    rate_hz = RATE

    def __init__(self, script, final=None):
        self.script, self.final_text = list(script), final
        self.loaded: list[str] = []
        self.languages_loaded: list[str] | None = None  # None: whatever load() was asked for
        self.started: list[str] = []
        self.load_calls: list[list[str]] = []
        self.accepted = 0  # samples
        self.calls = 0
        self.resets = 0
        self.last = ""

    @property
    def languages(self):
        return self.loaded if self.languages_loaded is None else self.languages_loaded

    def load(self, languages):
        self.load_calls.append(list(languages))
        self.loaded += [x for x in languages if x not in self.loaded]

    def start(self, language):
        self.started.append(language)
        self.calls, self.last = 0, ""

    def accept(self, samples):
        self.accepted += samples.size
        text = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if text == self.last:
            return None
        self.last = text
        return ASREvent(text)

    def finish(self):
        text = self.script[-1] if self.final_text is None else self.final_text
        self.last = ""
        return text

    def reset(self):
        self.resets += 1


class Selector:
    def __init__(self):
        self.hints = []

    def hint_speaker(self, hint):
        self.hints.append(hint)


def voice(amp=0.3, seconds=1.0):
    t = np.arange(round(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(round(seconds * RATE), np.float32)


class Rig:
    def __init__(self, script=("ciao",), final=None, cfg=None, **kw):
        self.cfg = cfg or reachy_config()
        self.bus = Bus()
        self.events: list[tuple[str, dict]] = []
        self.order: list[str] = []
        self.utterances = []
        self.infos = []
        self.bus.subscribe("percept.utterance", lambda m: self.infos.append(m.payload))
        self.now = T0
        self.asr = FakeASR(script, final)
        self.asr.loaded = list(self.cfg.voice.asr.models)  # as after start()
        self.speech = SpeechActivity(self.cfg.voice.echo_tail_ms)
        self.selector = Selector()

        def emit(name, now, payload):
            self.events.append((name, payload))
            self.order.append(name)

        self.fast = FastPath(
            self.cfg.voice,
            emit,
            self.selector,
            bearing=kw.pop("bearing", lambda now: None),
            target_entry=kw.pop("target_entry", lambda: None),
            clock=lambda: self.now,
        )
        vad = EnergyVAD(RATE, FRAME)
        self.listener = VoiceListener(
            self.cfg,
            lambda timeout: None,
            Endpointer(vad, self.cfg.voice.vad),
            self.asr,
            self.speech,
            self.fast,
            lambda u: (self.utterances.append(u), self.order.append("utterance")),
            self.bus,
            clock=lambda: self.now,
            **kw,
        )
        self.t = T0  # audio time: start of the next chunk

    def feed(self, samples, gap=False):
        for i in range(0, samples.size, CHUNK):
            part = samples[i : i + CHUNK]
            end = self.t + part.size / RATE
            self.now = end
            self.listener.process(
                SimpleNamespace(samples=part, rate_hz=RATE, stamp=end, gap=gap and i == 0)
            )
            self.t = end

    def say(self, seconds=1.0, tail=True):
        self.feed(voice(seconds=seconds))
        if tail:
            self.feed(silence(self.cfg.voice.vad.end_silence_ms / 1000 + 0.2))


def names(rig):
    return [n for n, _ in rig.events]


def test_utterance_stamps_and_text_reach_the_consumer():
    r = Rig(["ciao", "ciao reati come stai"])
    r.feed(silence(0.5))
    start = r.t
    r.say(1.0)
    (u,) = r.utterances
    assert u.text == "ciao reati come stai" and u.final
    assert u.language == "it"
    assert u.start_stamp == pytest.approx(start, abs=0.05)
    assert u.last_speech_stamp == pytest.approx(start + 1.0, abs=0.07)
    assert u.endpoint_stamp > u.last_speech_stamp + 0.45
    assert u.prosody.rms_db == pytest.approx(-13.5, abs=1.0)
    assert u.prosody.pitch_hz == pytest.approx(180, rel=0.03)


def test_asr_only_hears_speech_with_its_pre_roll():
    r = Rig(["ciao"])
    r.feed(silence(3.0))
    assert r.asr.accepted == 0 and r.asr.started == []
    r.say(1.0)
    spoken = 1.0 + r.cfg.voice.vad.pre_roll_ms / 1000 + r.cfg.voice.vad.end_silence_ms / 1000
    assert r.asr.accepted <= (spoken + 0.1) * RATE
    assert r.asr.accepted >= 1.0 * RATE


def test_nothing_recognised_means_no_utterance():
    r = Rig(["ciao"], final="")
    r.say(1.0)
    assert r.utterances == [] and r.infos == []
    assert r.listener.snapshot()["empty"] == 1


def test_robot_speaking_drops_the_microphone():
    r = Rig(["ciao"])
    r.speech.extend(r.t + 5.0)
    r.say(1.0)
    assert r.asr.started == [] and r.utterances == []
    assert r.listener.snapshot()["dropped_speaking"] > 0
    r.feed(silence(5.0 + r.cfg.voice.echo_tail_ms / 1000))
    r.say(1.0)
    assert len(r.utterances) == 1


def test_speaking_starting_mid_utterance_discards_it():
    r = Rig(["ciao"])
    r.feed(voice(seconds=0.6))
    assert r.asr.started
    r.speech.extend(r.t + 1.0)
    r.feed(silence(1.0))
    r.feed(silence(1.0))
    assert r.utterances == [] and r.asr.resets >= 1
    r.say(1.0)
    assert len(r.utterances) == 1


def test_a_gap_resets_the_pipeline():
    r = Rig(["ciao"])
    r.feed(voice(seconds=0.6))
    resets = r.asr.resets
    r.feed(voice(seconds=0.3), gap=True)
    assert r.asr.resets == resets + 1
    assert r.listener.snapshot()["gaps"] == 1
    r.feed(silence(1.0))
    assert len(r.utterances) <= 1  # the post-gap fragment is its own, short utterance


def test_language_from_speaker_profile_then_default():
    r = Rig(["hello"], speaker_language=lambda: "en")
    r.say()
    r2 = Rig(["ciao"], speaker_language=lambda: None)
    r2.say()
    r3 = Rig(["ciao"], speaker_language=lambda: "fr")
    r3.say()
    assert (r.asr.started, r2.asr.started, r3.asr.started) == (["en"], ["it"], ["it"])
    assert r.utterances[0].language == "en"


def test_language_failure_falls_back_to_default():
    def boom():
        raise KeyError("gone")

    r = Rig(["ciao"], speaker_language=boom)
    r.say()
    assert r.asr.started == ["it"]


def test_resolve_language_rule():
    assert resolve_language(["it", "en"], "en", "it") == "en"
    assert resolve_language(["it", "en"], None, "en") == "en"
    assert resolve_language(["it", "en"], "fr", "de") == "it"
    with pytest.raises(ValueError):
        resolve_language([], "en", "it")


def test_start_loads_the_primary_language_first_then_the_others():
    r = Rig(["x"], cfg=RoboEmotConfig.model_validate({"conversation": {"language": "en"}}))
    r.asr.loaded = []
    r.listener.start()
    deadline = time.monotonic() + 5
    while r.listener.snapshot()["state"] != "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    r.listener.stop()
    assert r.asr.load_calls == [["en"], ["it"]]


def test_a_model_that_fails_to_load_leaves_the_listener_unavailable(caplog):
    r = Rig(["x"])
    r.asr.load = lambda languages: (_ for _ in ()).throw(OSError("no model"))
    with caplog.at_level(logging.WARNING):
        r.listener.start()
        deadline = time.monotonic() + 5
        while r.listener.snapshot()["state"] == "off" and time.monotonic() < deadline:
            time.sleep(0.01)
        r.listener.stop()
    assert r.listener.snapshot()["state"] == "unavailable" and "OSError" in caplog.text


# ---- fast path


def test_keyword_event_fires_from_a_partial_before_the_utterance_ends():
    r = Rig(["ciao", "ciao reati", "ciao reati come stai"])
    r.say(1.0)
    assert r.order.index("speech.name_called") < r.order.index("utterance")
    assert r.order.index("speech.keyword.greeting") < r.order.index("utterance")
    assert names(r).count("speech.keyword.greeting") == 1  # once per utterance
    keyword_lag = r.fast.stats.snapshot()["keyword_vs_last_speech"]["last_ms"]
    assert keyword_lag < 0  # before the person stopped talking


def test_partials_and_final_do_not_repeat_events():
    r = Rig(["fermati", "fermati basta"], final="fermati basta cosi")
    r.say()
    assert names(r).count("speech.keyword.stop") == 1


def test_keyword_only_in_the_final_text_still_fires():
    r = Rig(["ehm"], final="ehm basta")
    r.say()
    assert "speech.keyword.stop" in names(r)


def test_speech_heard_and_every_event_is_a_known_appraisal_event():
    r = Rig(["ciao reati", "ciao reati come stai oggi"])
    r.say(0.8)
    assert names(r)[0] == "speech.heard"
    assert set(EVENT_FOR.values()) <= APPRAISAL_EVENTS
    assert set(names(r)) <= APPRAISAL_EVENTS


def test_loud_and_fast_prosody_impulses():
    loud = Rig(["uno due tre quattro cinque sei sette otto"], enrolled_in_view=lambda now: True)
    loud.feed(voice(amp=0.6, seconds=1.5))
    loud.feed(silence(1.0))
    assert "speech.loud" in names(loud) and "speech.fast" in names(loud)
    quiet = Rig(["uno due tre quattro cinque sei sette otto"], enrolled_in_view=lambda now: True)
    quiet.feed(voice(amp=0.12, seconds=4.0))
    quiet.feed(silence(1.0))
    assert "speech.loud" not in names(quiet) and "speech.fast" not in names(quiet)


def test_hint_uses_the_bearing_then_the_attended_person_then_nothing():
    r = Rig(["x"], bearing=lambda now: 0.4)
    r.say()
    h = r.selector.hints[0]
    assert h.azimuth_rad == 0.4 and h.entry_id is None
    r = Rig(["x"], target_entry=lambda: "e-7")
    r.say()
    assert r.selector.hints[0].entry_id == "e-7" and r.selector.hints[0].azimuth_rad is None
    r = Rig(["x"])
    r.say()
    assert r.selector.hints == []


def test_hint_comes_at_speech_start_not_at_the_end():
    r = Rig(["x"], bearing=lambda now: 0.1)
    r.feed(voice(seconds=0.6))
    assert len(r.selector.hints) == 1 and r.utterances == []
    snap = r.fast.stats.snapshot()
    assert snap["hint_lag"]["last_ms"] < 100


def test_fast_path_drives_the_affect_engine():
    cfg = RoboEmotConfig()
    bus = Bus()
    runner = AffectRunner(cfg.affect, cfg.appraisal, bus, 0.0)
    rig = Rig(["ciao"])
    rig.fast._emit = runner.on_event
    before = runner.engine.update(1.0)
    rig.say()
    after = runner.engine.update(rig.now)
    assert after.valence > before.valence  # a greeting is welcome


# ---- wake word policy through the listener


def test_stranger_must_say_the_wake_word_and_it_opens_a_window():
    r = Rig(["ciao"], final="ciao come stai")
    r.say()
    assert not r.utterances[0].addressed
    r.asr.script, r.asr.final_text = ["ciao reati"], "ciao reati cosa fai"
    r.say()
    assert r.utterances[1].addressed and r.utterances[1].wake_word
    r.asr.script, r.asr.final_text = ["dimmi"], "dimmi di piu"
    r.say()
    assert r.utterances[2].addressed and not r.utterances[2].wake_word
    r.feed(silence(r.cfg.voice.fastpath.addressed_window_s + 1))
    r.say()
    assert not r.utterances[3].addressed


def test_enrolled_person_in_view_needs_no_wake_word():
    r = Rig(["come stai"], enrolled_in_view=lambda now: True)
    r.say()
    assert r.utterances[0].addressed and not r.utterances[0].wake_word


def test_policy_always_and_never():
    cfg = RoboEmotConfig.model_validate({"voice": {"wake_word_policy": "always"}})
    r = Rig(["ciao"], cfg=cfg, enrolled_in_view=lambda now: True)
    r.say()
    assert not r.utterances[0].addressed
    cfg = RoboEmotConfig.model_validate({"voice": {"wake_word_policy": "never"}})
    r = Rig(["ciao"], cfg=cfg)
    r.say()
    assert r.utterances[0].addressed


def test_failing_presence_check_counts_as_nobody_enrolled():
    def boom(now):
        raise RuntimeError

    r = Rig(["ciao"], enrolled_in_view=boom)
    r.say()
    assert not r.utterances[0].addressed


# ---- privacy and robustness


SECRET = "segretissimo"


def test_bus_payload_and_logs_never_carry_the_text(caplog):
    caplog.set_level(logging.DEBUG)
    r = Rig([SECRET], final=f"{SECRET} reati")
    r.say()
    (info,) = r.infos
    assert isinstance(info, UtteranceInfo)
    assert info.chars == len(f"{SECRET} reati") and info.words == 2
    assert SECRET not in repr(info) and SECRET not in str(asdict(info))
    assert info.language == "it" and info.addressed and info.wake_word
    assert SECRET not in repr(r.utterances[0]) and SECRET not in str(r.listener.snapshot())
    assert SECRET not in caplog.text and SECRET not in str(r.events)


def test_failing_consumer_or_subscriber_does_not_stop_the_listener(caplog):
    r = Rig(["ciao"])
    r.listener._on_utterance = lambda u: 1 / 0
    r.bus.subscribe("percept.utterance", lambda m: 1 / 0)
    with caplog.at_level(logging.WARNING):
        r.say()
        r.say()
    assert r.listener.snapshot()["utterances"] == 2
    assert "ZeroDivisionError" in caplog.text


def test_thread_consumes_the_source_and_survives_a_bad_chunk():
    r = Rig(["ciao"])
    q: queue.Queue = queue.Queue()
    r.listener._source = lambda timeout: q.get(timeout=timeout) if not q.empty() else None
    r.listener.start()
    q.put(SimpleNamespace(samples=np.zeros(3, np.float64), rate_hz=RATE, stamp=T0, gap=False))
    t = T0
    for seg in (voice(seconds=0.8), silence(1.0)):
        for i in range(0, seg.size, CHUNK):
            part = seg[i : i + CHUNK]
            t += part.size / RATE
            q.put(SimpleNamespace(samples=part, rate_hz=RATE, stamp=t, gap=False))
    deadline = time.monotonic() + 5
    while not r.utterances and time.monotonic() < deadline:
        time.sleep(0.01)
    r.listener.stop()
    assert len(r.utterances) == 1
    assert r.listener.snapshot()["errors"] == 1


def test_latency_snapshot_has_the_slice_8_fields():
    r = Rig(["ciao", "ciao reati"])
    r.say()
    snap = r.listener.snapshot()
    assert {"keyword_vs_last_speech", "final_vs_last_speech", "reaction_vs_last_speech"} <= set(
        snap["latency"]
    )
    # final: the endpoint plus the frame of the last speech, as the fake clock sees it
    assert snap["latency"]["final_vs_last_speech"]["last_ms"] > 0


# ---- review round: final text decides, wake phrases, windows, hooks


def test_a_partial_hit_the_final_text_revises_away_does_not_address_or_wake():
    r = Rig(["ciao reati", "ciao reati"], final="ciao rete")
    r.say()
    (u,) = r.utterances
    assert "speech.name_called" in names(r)  # the early reaction happened ...
    assert not u.wake_word and not u.addressed and "name" not in u.keywords  # ... the verdict not
    assert r.infos[0].wake_word is False


def test_wake_phrase_wakes_anywhere_in_a_long_final_text():
    long_text = "oggi vorrei sapere ehi robot che tempo fa domani mattina in citta"
    r = Rig(["oggi"], final=long_text)
    r.say()
    assert r.utterances[0].wake_word and r.utterances[0].addressed


def test_alias_wakes_only_in_a_short_final_text():
    news = "il comune di reati ha approvato oggi il nuovo bilancio"
    r = Rig(["il"], final=news)
    r.say()
    assert "name" in r.utterances[0].keywords  # speech.name_called may fire ...
    assert not r.utterances[0].wake_word and not r.utterances[0].addressed  # ... no wake
    short = Rig(["reati"], final="ciao reati come stai")
    short.say()
    assert short.utterances[0].wake_word


def test_unaddressed_speech_gives_no_prosody_impulse():
    r = Rig(["uno due tre quattro cinque sei sette otto"])
    r.feed(voice(amp=0.6, seconds=1.5))
    r.feed(silence(1.0))
    assert not r.utterances[0].addressed
    assert "speech.loud" not in names(r) and "speech.fast" not in names(r)


def test_loud_fast_addressed_speech_stays_below_saturation_for_a_minute():
    cfg = RoboEmotConfig()
    runner = AffectRunner(cfg.affect, cfg.appraisal, Bus(), 0.0)
    t, nxt, peak = 0.0, 0.0, 0.0
    while t < 60.0:
        if t >= nxt:  # an utterance every 3 s, each loud and fast
            for event in ("speech.heard", "speech.loud", "speech.fast"):
                runner.on_event(event, t, {"language": "it"})
            nxt += 3.0
        peak = max(peak, runner.engine.update(t).arousal)
        t += 0.1
    # Measured 0.57 with the defaults (arousal 1.0 is the ceiling). 0.6 leaves some margin, and
    # a rule change that doubles the impulses fails this: speaking at the robot must not pin it.
    assert peak < 0.6


def test_hot_reloaded_language_without_a_model_falls_back():
    r = Rig(["x"], speaker_language=lambda: "en")
    r.asr.languages_loaded = ["it"]  # en is configured but its model is not loaded
    r.say()
    assert r.asr.started == ["it"]


def test_speaking_chunks_reach_the_barge_in_hook():
    seen = []
    r = Rig(["x"], on_speaking_chunk=seen.append)
    r.speech.extend(r.t + 5.0)
    r.say(1.0)
    assert seen and r.asr.started == []
    r.listener._on_speaking_chunk = lambda c: 1 / 0
    r.say(0.2, tail=False)  # a raising hook does not stop the listener


def test_followup_window_is_capped_and_follows_the_same_person():
    who = ["p-1"]
    r = Rig(["ciao reati"], final="ciao reati", speaker_id=lambda: who[0])
    r.say()
    assert r.utterances[-1].addressed
    r.asr.script, r.asr.final_text = ["dimmi"], "dimmi"
    who[0] = "p-2"
    r.say()
    assert not r.utterances[-1].addressed
    who[0] = "p-1"
    r.say()
    assert r.utterances[-1].addressed
    snap = r.listener.snapshot()["wake"]
    assert snap["policy"] == "unless_enrolled_in_view" and "window_open" in snap


def test_hint_records_both_lags():
    r = Rig(["x"], bearing=lambda now: 0.1)
    r.feed(voice(seconds=0.6))
    snap = r.fast.stats.snapshot()
    assert snap["hint_vs_speech_start"]["last_ms"] >= snap["hint_lag"]["last_ms"] >= 0


def test_unavailable_models_drain_the_source(caplog):
    r = Rig(["x"])
    r.asr.load = lambda languages: (_ for _ in ()).throw(OSError("no model"))
    reads = []
    r.listener._source = lambda timeout: reads.append(timeout)
    with caplog.at_level(logging.WARNING):
        r.listener.start()
        deadline = time.monotonic() + 5
        while len(reads) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        r.listener.stop()
    assert len(reads) >= 3 and r.listener.snapshot()["state"] == "unavailable"


def test_the_asr_is_handed_back_before_every_abort_and_a_failing_hook_does_not_stop_it():
    calls = []

    def hook():
        calls.append(len(rig.asr.started))
        raise RuntimeError("x")

    rig = Rig(before_asr_reset=hook)
    rig.listener._abort()
    rig.listener.process(
        SimpleNamespace(samples=np.zeros(CHUNK, np.float32), rate_hz=RATE, stamp=T0 + 1.0, gap=True)
    )
    assert len(calls) == 2 and rig.asr.resets == 2


def test_stop_between_thread_creation_and_start_does_not_join_an_unstarted_thread(monkeypatch):
    import threading

    rig = Rig()
    real_start = threading.Thread.start

    def start_after_stop(self):
        if self.name == "roboemot-listener":
            rig.listener.stop()
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start_after_stop)
    rig.listener.start()
    rig.listener.stop()
    assert rig.listener._thread is None
