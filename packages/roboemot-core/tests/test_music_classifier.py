"""Report-only music classifier (SPEC 8, 12A): when it runs, what it reports, how it fails."""

import json

import numpy as np
import pytest
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


class Recorder:
    """A backend that keeps a copy of every window it scores."""

    def __init__(self):
        self.windows = []

    def scores(self, window):
        self.windows.append(np.array(window, copy=True))
        return 0.5, 0.5


class ConcatWindow:
    """The window logic before the ring buffer (a reference copy): concatenate every chunk and
    keep the last WINDOW_SAMPLES, so the ring buffer can be checked against it."""

    def __init__(self, interval_s, backend):
        self.interval_s, self.backend = interval_s, backend
        self.window = np.zeros(0, dtype=np.float32)
        self.expected, self.last_run_s, self.holes, self.runs = float("nan"), -float("inf"), 0, []

    def feed(self, samples, start):
        audio = np.asarray(samples, dtype=np.float32)
        chunk_s = audio.size / RATE
        if abs(start - self.expected) > chunk_s:
            self.window = np.zeros(0, dtype=np.float32)
            self.holes += 1
        self.expected = start + chunk_s
        self.window = np.concatenate([self.window, audio])[-WINDOW_SAMPLES:]
        if self.window.size == WINDOW_SAMPLES and start - self.last_run_s >= self.interval_s:
            self.last_run_s = start
            self.runs.append(start)
            return self.backend.scores(self.window)
        return None


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("hole_share", [0.0, 0.01])
@pytest.mark.parametrize(
    "sizes",
    [
        (320,),
        (WINDOW_SAMPLES,),
        (WINDOW_SAMPLES + 1, 20000),
        (1, 7, 333, 1000, 4096, WINDOW_SAMPLES - 1, WINDOW_SAMPLES, 31201),
    ],
)
def test_the_ring_buffer_scores_the_same_windows_as_concatenating(seed, hole_share, sizes):
    rng = np.random.default_rng(seed)
    new_rec, old_rec = Recorder(), Recorder()
    new = MusicClassifier(MusicClassifierConfig(interval_s=1.0), RATE, new_rec, timer=clock())
    old = ConcatWindow(1.0, old_rec)
    start, new_runs = 0.0, []
    for i in range(400):
        chunk = rng.normal(0, 0.1, int(rng.choice(sizes))).astype(np.float32)
        if hole_share and (i == 200 or rng.random() < hole_share):
            start += rng.uniform(0.1, 3.0)
        if new.feed(chunk, start) is not None:
            new_runs.append(start)
        old.feed(chunk, start)
        start += chunk.size / RATE
    assert new_runs == old.runs and len(new_runs) > 0
    assert len(new_rec.windows) == len(old_rec.windows)
    for a, b in zip(new_rec.windows, old_rec.windows, strict=True):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    snap = new.snapshot()
    assert (snap["runs"], snap["holes"]) == (len(old.runs), old.holes)
    assert (old.holes > 0) == (hole_share > 0)
