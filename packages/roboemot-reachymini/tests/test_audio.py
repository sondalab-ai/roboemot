"""Microphone level for the loud-onset reflex (SPEC 4A, 10, 12), against a fake SDK media."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot_reachymini.audio import ReachyAudio, chunk_rms


def media(chunks=None, audio="present", record_error=None):
    """A fake ``mini``: ``get_audio_sample`` pops the scripted chunks, then returns None."""
    queue = list(chunks or [])
    calls = SimpleNamespace(started=0, stopped=0, got=0)

    def start():
        calls.started += 1
        if record_error:
            raise record_error

    def get():
        calls.got += 1
        return queue.pop(0) if queue else None

    m = SimpleNamespace(
        audio=None if audio is None else object(),
        start_recording=start,
        stop_recording=lambda: setattr(calls, "stopped", calls.stopped + 1),
        get_audio_sample=get,
    )
    return SimpleNamespace(media=m), calls


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_rms_of_mono_and_multichannel_chunks():
    assert chunk_rms(np.full(100, 0.5, np.float32)) == pytest.approx(0.5)
    assert chunk_rms(np.full((100, 2), -0.25, np.float32)) == pytest.approx(0.25)
    assert chunk_rms(np.zeros(10, np.float32)) == 0.0


@pytest.mark.parametrize("junk", [None, [0.1, 0.2], "x", np.array([], np.float32), object()])
def test_anything_but_a_non_empty_array_has_no_level(junk):
    assert chunk_rms(junk) is None


def test_the_levels_go_to_the_sink_as_scalars_with_the_read_time():
    chunks = [np.full(64, v, np.float32) for v in (0.1, 0.2, 0.4)]
    mini, calls = media(chunks)
    got = []
    audio = ReachyAudio(mini, lambda rms, stamp: got.append((rms, stamp)), 500, lambda: 42.0)
    audio.start()
    assert wait_for(lambda: len(got) == 3)
    audio.stop()
    assert [round(r, 3) for r, _ in got] == [0.1, 0.2, 0.4]
    assert all(type(r) is float and s == 42.0 for r, s in got)
    assert calls.started == 1 and calls.stopped == 1
    assert audio.snapshot()["state"] == "off"


def test_no_audio_system_degrades_to_unavailable_without_a_thread():
    mini, calls = media(audio=None)
    audio = ReachyAudio(mini, lambda *_: None, 50)
    before = threading.active_count()
    audio.start()
    snap = audio.snapshot()
    assert snap["state"] == "unavailable" and "no audio" in snap["reason"]
    assert threading.active_count() == before and calls.started == 0 and calls.got == 0
    audio.stop()  # safe without a thread


def test_a_capture_that_will_not_start_is_unavailable_and_names_only_the_error_type():
    mini, _ = media(record_error=RuntimeError("secret device path"))
    audio = ReachyAudio(mini, lambda *_: None, 50)
    audio.start()
    snap = audio.snapshot()
    assert snap["state"] == "unavailable"
    assert snap["reason"] == "start_recording failed: RuntimeError"


def test_a_stream_without_usable_chunks_reports_no_data_not_running():
    mini, _ = media()
    audio = ReachyAudio(mini, lambda *_: None, 500)
    audio.start()
    time.sleep(0.05)
    assert audio.snapshot()["state"] == "no_data"
    audio.stop()


def test_a_failing_sink_does_not_end_the_thread_and_logs_once(caplog):
    chunks = [np.full(8, 0.1, np.float32)] * 5
    mini, _ = media(chunks)
    seen = []

    def sink(rms, stamp):
        seen.append(rms)
        raise ValueError("boom with 0.1")

    audio = ReachyAudio(mini, sink, 500)
    with caplog.at_level(logging.DEBUG):
        audio.start()
        assert wait_for(lambda: len(seen) == 5)
        audio.stop()
    warnings = [r for r in caplog.records if r.name.endswith("audio")]
    assert len(warnings) == 1 and warnings[0].getMessage() == "audio: ValueError"


def test_no_buffer_is_kept_or_logged(caplog):
    marker = 0.123456
    chunks = [np.full(256, marker, np.float32) for _ in range(4)]
    mini, _ = media(chunks)
    got = []
    audio = ReachyAudio(mini, lambda r, s: got.append(r), 500)
    with caplog.at_level(logging.DEBUG):
        audio.start()
        assert wait_for(lambda: len(got) == 4)
        audio.stop()

    def arrays(obj, depth=0):
        if isinstance(obj, np.ndarray):
            yield obj
        elif depth < 3 and isinstance(obj, dict):
            for v in obj.values():
                yield from arrays(v, depth + 1)
        elif depth < 3 and isinstance(obj, list | tuple | set):
            for v in obj:
                yield from arrays(v, depth + 1)
        elif depth < 3 and hasattr(obj, "__dict__"):
            yield from arrays(vars(obj), depth + 1)

    assert list(arrays(audio)) == []
    assert all(str(marker) not in r.getMessage() for r in caplog.records)
    assert set(audio.snapshot()) == {"state", "reason", "chunks", "dropped_stale"}


def test_a_drain_keeps_only_the_newest_chunk_and_counts_the_rest():
    chunks = [np.full(8, v, np.float32) for v in (0.1, 0.2, 0.3, 0.4)]
    mini, _ = media(chunks)
    got = []
    audio = ReachyAudio(mini, lambda rms, stamp: got.append(rms), 50)
    assert audio.poll_once(drain=True)
    assert [round(r, 3) for r in got] == [0.4]
    assert audio.snapshot()["dropped_stale"] == 3


def test_after_a_stall_of_the_reading_thread_the_backlog_is_dropped_not_played_late():
    chunks = [np.full(8, v, np.float32) for v in (0.1, 0.2, 0.3)]
    mini, _ = media(chunks)
    got = []
    ticks = iter(range(1000))
    # every clock read is 1 s after the last: each loop pass looks like a stall (gap > max_gap_s)
    audio = ReachyAudio(
        mini, lambda rms, stamp: got.append(rms), 500, lambda: float(next(ticks)), max_gap_s=0.15
    )
    audio.start()
    assert wait_for(lambda: len(got) >= 1)
    audio.stop()
    assert [round(r, 3) for r in got] == [0.3]


def test_without_a_stall_every_chunk_is_used():
    chunks = [np.full(8, v, np.float32) for v in (0.1, 0.2, 0.3)]
    mini, _ = media(chunks)
    got = []
    audio = ReachyAudio(mini, lambda rms, stamp: got.append(rms), 500, max_gap_s=60.0)
    audio.start()
    assert wait_for(lambda: len(got) == 3)
    audio.stop()
    assert audio.snapshot()["dropped_stale"] == 0
