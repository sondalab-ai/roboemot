"""Report-only music classifier (SPEC 8, 12A): when it runs, what it reports, how it fails."""

import json

import numpy as np
from roboemot.config.schema import MusicClassifierConfig, RoboEmotConfig
from roboemot.voice.music_classifier import WINDOW_SAMPLES, MusicClassifier
from roboemot.voice.pipeline import build_music_classifier

RATE = 16000
CHUNK = 320


class Scripted:
    """A backend that returns fixed scores and keeps the size of every window."""

    def __init__(self, music=0.9, speech=0.1, error=None):
        self.result, self.error, self.sizes = (music, speech), error, []

    def scores(self, window):
        if self.error is not None:
            raise self.error
        self.sizes.append(window.size)
        return self.result


def classifier(backend, interval_s=2.0):
    return MusicClassifier(
        MusicClassifierConfig(interval_s=interval_s), RATE, backend, timer=clock()
    )


def clock():
    ticks = iter(range(10**6))
    return lambda: next(ticks) * 0.01


def feed(c, seconds, start=0.0):
    n = round(seconds * RATE / CHUNK)
    for i in range(n):
        c.feed(np.zeros(CHUNK, np.float32), start + i * CHUNK / RATE)
    return start + n * CHUNK / RATE


def test_scores_full_windows_at_the_interval():
    backend = Scripted()
    c = classifier(backend)
    feed(c, 7.0)
    assert len(backend.sizes) == 4  # the window is full after about 1 s, then every 2 s
    assert set(backend.sizes) == {WINDOW_SAMPLES}


def test_a_hole_restarts_the_window():
    backend = Scripted()
    c = classifier(backend, interval_s=1.0)
    end = feed(c, 0.5)
    feed(c, 0.5, end + 1.0)  # neither side of the hole fills a window
    assert backend.sizes == []
    assert c.snapshot()["holes"] == 1


def test_the_report_has_numbers_only():
    c = classifier(Scripted(music=0.4, speech=0.2))
    feed(c, 7.0)
    snap = c.snapshot()
    assert snap["state"] == "running" and snap["reason"] is None and snap["runs"] > 0
    assert snap["music"]["median"] == 0.4 and snap["speech"]["median"] == 0.2
    assert snap["music_share_at"] == {"0.3": 1.0, "0.5": 0.0}
    assert snap["cost_ms"]["samples"] == snap["runs"]
    json.dumps(snap)


def test_a_model_error_is_counted_and_not_raised():
    c = classifier(Scripted(error=ValueError("boom")))
    feed(c, 4.0)
    snap = c.snapshot()
    assert snap["errors"] > 0 and snap["runs"] == 0
    assert snap["music_share_at"] == {"0.3": None, "0.5": None}


def test_without_a_backend_it_is_unavailable_and_idle():
    c = MusicClassifier(MusicClassifierConfig(), RATE, None, "model")
    feed(c, 2.0)
    snap = c.snapshot()
    assert (snap["state"], snap["reason"], snap["runs"]) == ("unavailable", "model", 0)


def test_a_missing_model_leaves_it_unavailable(tmp_path):
    snap = build_music_classifier(RoboEmotConfig(), tmp_path).snapshot()
    assert (snap["state"], snap["reason"]) == ("unavailable", "model")
