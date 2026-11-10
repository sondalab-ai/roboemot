"""SPEC 8, 7: the classifier decides whether music plays; the beat detector gives the tempo."""

from __future__ import annotations

import pytest
from audio_signals import drum_loop, silence
from pydantic import ValidationError
from roboemot.config.schema import MusicClassifierConfig, MusicConfig
from roboemot.motion.beat import room_beat
from roboemot.voice.music import MusicDetector, MusicState
from test_voice_listener import CHUNK as RIG_CHUNK
from test_voice_listener import RATE as RIG_RATE
from test_voice_listener import Rig

RATE = 16000
CHUNK = 320
MUSIC, NOT_MUSIC = (0.9, 0.0), (0.0, 0.0)
FREE = (60.0, 0.6)
CHUNKS_PER_RUN = round(2.0 * RIG_RATE / RIG_CHUNK)


def primary(**overrides):
    det = MusicDetector(MusicConfig(**overrides), RATE)
    det.classifier_primary = True
    return det


def score(det, runs, t0=0.0, step=2.0):
    """Give ``det`` one classifier run per item of ``runs``; returns its events, by run."""
    return [det.classified(*r, t0 + i * step) for i, r in enumerate(runs)]


def feed(det, signal, t0=0.0):
    return [
        event
        for i in range(0, signal.size, CHUNK)
        if (event := det.feed(signal[i : i + CHUNK], t0 + i / RATE)) is not None
    ]


def test_three_runs_of_five_start_music_with_no_tempo():
    det = primary()
    assert score(det, [MUSIC, MUSIC, MUSIC]) == [None, None, "started"]
    feed(det, silence(0.3, RATE), 4.0)
    state = det.state(4.3)
    assert state.playing and state.tempo_bpm is None
    assert det.snapshot()["classifier_starts"] == 1
    assert det.snapshot()["beat_locked"] is False


def test_the_interval_is_back_dated_to_the_first_run():
    det = primary()
    score(det, [NOT_MUSIC, MUSIC, MUSIC, MUSIC])
    feed(det, silence(0.3, RATE), 6.0)
    assert det.during(2.0, 2.0) and not det.during(0.0, 1.0)


def test_runs_that_are_too_few_do_not_start_music():
    det = primary()
    assert score(det, [MUSIC, NOT_MUSIC, NOT_MUSIC, MUSIC, NOT_MUSIC, NOT_MUSIC]) == [None] * 6


def test_old_runs_do_not_count():
    det = primary()
    score(det, [MUSIC, MUSIC])
    assert det.classified(*MUSIC, 100.0) is None


def test_speech_above_music_does_not_start_music():
    det = primary()
    assert score(det, [(0.6, 0.9)] * 5) == [None] * 5


def test_runs_at_keep_score_keep_music_playing():
    det = primary(release_s=8.0)
    score(det, [MUSIC, MUSIC, MUSIC])
    score(det, [(0.3, 0.0)] * 6, t0=6.0)
    assert not feed(det, silence(1.0, RATE), 16.0)
    assert det.state(17.0).playing


def test_music_ends_after_the_release_with_no_run_at_keep_score():
    det = primary(release_s=8.0)
    score(det, [MUSIC, MUSIC, MUSIC])
    assert "stopped" in feed(det, silence(9.0, RATE), 4.0)


def test_a_periodic_sound_cannot_start_music_while_the_classifier_decides():
    det = primary()
    assert feed(det, drum_loop(120, 30.0, RATE)) == []
    assert not det.state(30.0).playing


def test_the_beat_gives_the_tempo_of_music_the_classifier_started():
    det = primary()
    signal = drum_loop(120, 30.0, RATE)
    for second in range(0, 30, 2):
        feed(det, signal[second * RATE : (second + 2) * RATE], float(second))
        det.classified(*MUSIC, second + 2.0)
    assert det.state(30.0).tempo_bpm is not None
    assert det.snapshot()["beat_locked"] is True


def test_a_stale_tempo_is_not_reused():
    det = primary()
    feed(det, drum_loop(120, 20.0, RATE))
    feed(det, silence(4.0, RATE), 20.0)
    score(det, [MUSIC, MUSIC, MUSIC], t0=24.0)
    assert det.state(29.0).tempo_bpm is None


def test_a_beat_keeps_no_music_playing_while_the_classifier_decides():
    det = primary(release_s=4.0)
    score(det, [MUSIC, MUSIC, MUSIC])
    assert "stopped" in feed(det, drum_loop(120, 12.0, RATE), 4.0)


def test_without_the_classifier_the_beat_decides():
    det = MusicDetector(MusicConfig(), RATE)
    assert feed(det, drum_loop(120, 30.0, RATE))[0] == "started"
    assert det.classified(*MUSIC, 30.0) is None
    assert det.snapshot()["classifier_starts"] == 0


def test_the_rule_needs_consistent_settings():
    with pytest.raises(ValidationError):
        MusicClassifierConfig(start_runs=6, window_runs=5)
    with pytest.raises(ValidationError):
        MusicClassifierConfig(keep_score=0.6, start_score=0.5)


def test_music_with_no_tempo_gets_a_free_beat():
    state = MusicState(True, None, None, 0.1, 5.0)
    beat = room_beat(state, FREE)
    assert (beat.period_s, beat.strength, beat.source) == (1.0, 0.6, "free")
    assert room_beat(state) is None
    assert room_beat(MusicState(False, None, None, 0.1, 5.0), FREE) is None


def test_a_locked_tempo_is_not_free():
    assert room_beat(MusicState(True, 120.0, 0.1, 0.5, 5.0), FREE).source == "room"


class Scripted:
    """A classifier for the listener: one (music, speech) per run, every ``every`` chunks."""

    running = True

    def __init__(self, scores, every=CHUNKS_PER_RUN):
        self.scores, self.every, self.chunks = list(scores), every, 0

    def feed(self, samples, start):
        self.chunks += 1
        if self.chunks % self.every or not self.scores:
            return None
        return self.scores.pop(0)

    def snapshot(self):
        return {"state": "running"}


def rig(scores):
    r = Rig(["ciao"])
    r.seen = []
    for topic in ("music.started", "music.stopped"):
        r.bus.subscribe(topic, lambda m: r.seen.append(m.topic))
    r.listener.music_classifier = Scripted(scores)
    return r


def test_the_classifier_alone_drives_the_bus_the_wake_rule_and_the_dance():
    r = rig([MUSIC] * 3)
    start = r.t
    r.feed(silence(7.0, RIG_RATE))
    assert r.seen == ["music.started"]
    state = r.listener.music_state(r.t)
    assert state.playing and state.tempo_bpm is None
    assert r.listener._music_during(start, r.t)
    assert room_beat(state, FREE).source == "free"


def test_a_failing_classifier_hands_the_decision_back():
    r = rig([])
    r.listener.music_classifier.running = False
    r.feed(silence(0.1, RIG_RATE))
    assert r.listener.music.classifier_primary is False


def test_the_setting_hands_the_decision_back():
    r = rig([MUSIC] * 3)
    classifier = r.cfg.music.classifier.model_copy(update={"decides": False})
    music = r.cfg.music.model_copy(update={"classifier": classifier})
    r.listener.apply_config(r.cfg.model_copy(update={"music": music}))
    r.feed(silence(7.0, RIG_RATE))
    assert r.seen == [] and r.listener.music.classifier_primary is False
