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
    assert set(audio.snapshot()) == {"state", "reason", "chunks", "dropped_stale", "voice"}
    assert audio.snapshot()["voice"] is None  # no voice queue, so no audio is kept


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


# --------------------------------------------------------------------------- voice side (SPEC 8)


def voice_media(chunks, rate=16000, channels=2):
    mini, calls = media(chunks)
    mini.media.get_input_audio_samplerate = lambda: rate
    mini.media.get_input_channels = lambda: channels
    return mini, calls


def manual(audio, mini):
    """Open the voice side without the reading thread, for tests that call ``poll_once``."""
    audio._open_voice(mini.media)
    return audio


def test_the_voice_queue_keeps_every_chunk_while_the_reflex_drops_the_stale_ones():
    chunks = [np.full(160, v, np.float32) for v in (0.1, 0.2, 0.3, 0.4)]
    mini, _ = voice_media(chunks, channels=1)
    got = []
    audio = ReachyAudio(mini, lambda rms, stamp: got.append(rms), 50, voice=True)
    manual(audio, mini)
    assert audio.poll_once(drain=True)
    assert [round(r, 3) for r in got] == [0.4]
    assert audio.snapshot()["dropped_stale"] == 3
    heard = []
    while (c := audio.read()) is not None:
        heard.append(round(float(c.samples[0]), 3))
    assert heard == [0.1, 0.2, 0.3, 0.4]
    audio.stop()


def test_stereo_is_averaged_to_mono_and_the_rate_travels_with_the_frames():
    stereo = np.stack([np.full(160, 0.2, np.float32), np.full(160, 0.6, np.float32)], axis=1)
    mini, _ = voice_media([stereo])
    audio = ReachyAudio(mini, None, 50, clock=lambda: 7.0, voice=True)
    manual(audio, mini)
    assert audio.poll_once()
    chunk = audio.read()
    assert chunk.samples.shape == (160,) and chunk.samples.dtype == np.float32
    assert chunk.samples == pytest.approx(0.4) and chunk.rate_hz == 16000 and chunk.stamp == 7.0
    voice = audio.snapshot()["voice"]
    assert voice["input_rate_hz"] == 16000 and voice["input_channels"] == 2
    audio.stop()


@pytest.mark.parametrize("rate", [48000, 44100, 8000])
def test_the_input_rate_is_read_from_the_sdk_and_converted_to_the_voice_rate(rate):
    t = np.arange(rate) / rate
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chunks = [tone[i : i + 480] for i in range(0, rate, 480)]
    mini, _ = voice_media(chunks, rate=rate, channels=1)
    audio = ReachyAudio(mini, None, 50, voice=True, voice_queue_s=60)
    manual(audio, mini)
    while audio.poll_once():
        pass
    frames = []
    while (c := audio.read()) is not None:
        assert c.rate_hz == 16000
        frames.append(c.samples)
    out = np.concatenate(frames)
    assert out.size == pytest.approx(16000, abs=700)  # the filter lookahead holds a few back
    assert float(np.sqrt(np.mean(out[1000:-1000] ** 2))) == pytest.approx(0.5 / 2**0.5, rel=0.02)
    audio.stop()


def test_the_queue_is_bounded_in_seconds_and_counts_what_it_drops():
    chunks = [np.full(1600, v, np.float32) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    mini, _ = voice_media(chunks, channels=1)
    audio = ReachyAudio(mini, None, 50, voice=True, voice_queue_s=0.25)  # 4000 samples
    manual(audio, mini)
    while audio.poll_once():
        pass
    snap = audio.snapshot()["voice"]
    assert snap["dropped"] == 3 and snap["frames"] == 8000
    assert snap["queued_s"] == pytest.approx(0.2)
    assert [round(float(audio.read().samples[0]), 1) for _ in range(2)] == [0.4, 0.5]
    assert audio.read() is None
    audio.stop()


def test_read_waits_for_a_chunk_and_returns_none_on_timeout():
    mini, _ = voice_media([], channels=1)
    audio = ReachyAudio(mini, None, 50, voice=True)
    audio.start()
    t0 = time.monotonic()
    assert audio.read(timeout=0.05) is None and time.monotonic() - t0 >= 0.04
    audio.stop()


def test_the_hub_runs_for_voice_alone_and_feeds_no_reflex():
    mini, calls = voice_media([np.full(160, 0.3, np.float32)] * 3, channels=1)
    audio = ReachyAudio(mini, None, 500, voice=True)
    audio.start()
    assert wait_for(lambda: audio.snapshot()["voice"]["frames"] == 480)
    assert audio.snapshot()["state"] == "running" and calls.started == 1
    audio.stop()
    assert calls.stopped == 1


def test_an_unusable_input_rate_leaves_the_voice_side_silent(caplog):
    mini, _ = voice_media([np.full(160, 0.3, np.float32)], rate=-1, channels=1)
    audio = ReachyAudio(mini, None, 50, voice=True)
    manual(audio, mini)
    assert not audio.poll_once() and audio.read() is None
    audio.stop()


def test_voice_frames_are_never_logged(caplog):
    marker = 0.123456
    mini, _ = voice_media([np.full(256, marker, np.float32)] * 3, channels=1)
    audio = ReachyAudio(mini, None, 500, voice=True)
    with caplog.at_level(logging.DEBUG):
        audio.start()
        assert wait_for(lambda: audio.snapshot()["voice"]["frames"] == 768)
        audio.stop()
    assert all(str(marker) not in r.getMessage() for r in caplog.records)
    assert all(not isinstance(v, np.ndarray) for v in audio.snapshot()["voice"].values())


def test_stamps_of_a_drained_batch_are_monotone_and_end_at_the_read_time():
    chunks = [np.full(1600, v, np.float32) for v in (0.1, 0.2, 0.3)]  # 0.1 s each
    mini, _ = voice_media(chunks, channels=1)
    audio = manual(ReachyAudio(mini, lambda *_: None, 50, clock=lambda: 50.0, voice=True), mini)
    assert audio.poll_once(drain=True)
    stamps = [audio.read().stamp for _ in range(3)]
    assert stamps == pytest.approx([49.8, 49.9, 50.0])


def test_the_resampler_lookahead_is_subtracted_from_the_stamp():
    mini, _ = voice_media([np.zeros(4800, np.float32)] * 3, rate=48000, channels=1)
    audio = manual(ReachyAudio(mini, None, 50, clock=lambda: 10.0, voice=True), mini)
    while audio.poll_once():
        pass
    chunk = audio.read()
    assert chunk.stamp == pytest.approx(10.0 - audio._resampler.lookahead_s)


def test_overflow_marks_a_gap_and_counts_samples_and_chunks():
    chunks = [np.full(1600, v, np.float32) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    mini, _ = voice_media(chunks, channels=1)
    audio = manual(ReachyAudio(mini, None, 50, voice=True, voice_queue_s=0.25), mini)
    while audio.poll_once():
        pass
    snap = audio.snapshot()["voice"]
    assert snap["dropped"] == 3 and snap["dropped_samples"] == 4800
    first, second = audio.read(), audio.read()
    assert first.gap and not second.gap


def test_non_finite_samples_become_silence_marked_as_a_gap():
    bad = np.full(160, 0.3, np.float32)
    bad[5] = np.nan
    inf = np.full(160, 0.3, np.float32)
    inf[0] = np.inf
    good = np.full(160, 0.3, np.float32)
    mini, _ = voice_media([bad, good, inf], channels=1)
    audio = manual(ReachyAudio(mini, None, 50, voice=True), mini)
    while audio.poll_once():
        pass
    a, b, c = audio.read(), audio.read(), audio.read()
    assert a.gap and not b.gap and c.gap
    assert not a.samples.any() and b.samples[0] == pytest.approx(0.3) and not c.samples.any()
    assert audio.snapshot()["voice"]["sanitised"] == 2


def test_start_clears_the_old_queue_and_stop_wakes_a_waiting_reader():
    mini, _ = voice_media([np.full(160, 0.3, np.float32)], channels=1)
    audio = ReachyAudio(mini, None, 50, voice=True)
    manual(audio, mini)
    assert audio.poll_once()
    audio.start()  # restart: the stale chunk must not survive
    assert audio.snapshot()["voice"]["queued_s"] == 0.0
    out = []
    t = threading.Thread(target=lambda: out.append(audio.read(timeout=5.0)))
    t0 = time.monotonic()
    t.start()
    time.sleep(0.05)
    audio.stop()
    t.join(2)
    assert out == [None] and time.monotonic() - t0 < 2.0
    audio.start()  # start/stop/start works and reads again
    assert audio.snapshot()["state"] in ("no_data", "running")
    audio.stop()


def test_stop_closes_the_capture_even_when_the_thread_did_not_end_in_time(monkeypatch):
    from roboemot import shutdown
    from roboemot_reachymini import audio as audio_module

    monkeypatch.setattr(audio_module, "JOIN_TIMEOUT_S", 0.05)
    mini, calls = media([])
    release = threading.Event()
    audio = ReachyAudio(mini, None, 50)
    audio._thread = threading.Thread(target=release.wait, args=(5.0,), daemon=True)
    audio._thread.start()
    try:
        with shutdown.budget(0.0):  # the budget is spent: a short wait, then give up
            audio.stop()
        assert calls.stopped == 1
        assert audio._thread is not None  # a stuck thread keeps its handle
    finally:
        release.set()
