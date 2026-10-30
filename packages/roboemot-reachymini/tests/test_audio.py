"""Microphone level for the loud-onset reflex (SPEC 4A, 10, 12), against a fake SDK media."""

from __future__ import annotations

import json
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
    warnings = [
        r for r in caplog.records if r.name.endswith("audio") and r.levelno >= logging.WARNING
    ]
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
    assert set(audio.snapshot()) == {
        "state",
        "reason",
        "chunks",
        "dropped_stale",
        "voice",
        "levels",
    }
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
    ticks = iter([0.0])
    # the clock jumps 1 s after the thread's first read of it: the first pass follows a stall
    audio = ReachyAudio(
        mini, lambda rms, stamp: got.append(rms), 500, lambda: next(ticks, 1.0), max_gap_s=0.15
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


def test_drained_chunks_end_one_chunk_after_another_from_the_last_chunk_before_the_stall():
    chunks = [np.full(1600, v, np.float32) for v in (0.1, 0.2, 0.3, 0.4)]  # 0.1 s each
    mini, _ = voice_media(chunks, channels=1)
    now = [49.7]
    audio = manual(ReachyAudio(mini, lambda *_: None, 50, clock=lambda: now[0], voice=True), mini)
    assert audio.poll_once()
    now[0] = 50.05  # the queued chunks were captured during the stall, not at the read
    assert audio.poll_once(drain=True, stall_s=0.3)
    stamps = [audio.read().stamp for _ in range(4)]
    assert stamps == pytest.approx([49.7, 49.8, 49.9, 50.0])


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


# ------------------------------------------------------------------ channel levels (SPEC 12A)

BLOCK = 1600  # 0.1 s at 16 kHz: one chunk closes a block of level_block_s=0.1


def levels_of(columns, blocks=3, channels=None, **kw):
    """The ``levels`` report after ``blocks`` chunks of the given channel columns."""
    rng = np.random.default_rng(1)
    base = (0.2 * rng.standard_normal(BLOCK)).astype(np.float32)
    chunk = np.stack([f(base) for f in columns], axis=1) if columns else base
    mini, _ = voice_media([chunk] * blocks, channels=channels or max(1, len(columns)))
    audio = ReachyAudio(mini, None, 50, voice=True, level_block_s=0.1, **kw)
    manual(audio, mini)
    for _ in range(blocks):
        audio.poll_once()
    snap = audio.snapshot()["levels"]
    audio.stop()
    return snap


def test_identical_channels_correlate_and_have_equal_levels():
    lv = levels_of([lambda x: x, lambda x: x])
    assert lv["block_s"] == 0.1 and lv["blocks"] == 3
    assert lv["corr"]["min"] == pytest.approx(1.0) and lv["corr"]["max"] == pytest.approx(1.0)
    assert lv["ch0_dbfs"] == pytest.approx(lv["ch1_dbfs"])
    assert lv["ch0_dbfs"] == pytest.approx(lv["mono_dbfs"])
    assert -20 < lv["ch0_dbfs"]["p50"] < -10  # rms 0.2 is -14 dBFS


def test_anti_phase_channels_cancel_in_the_mono_mix():
    lv = levels_of([lambda x: x, lambda x: -x])
    assert lv["corr"]["max"] == pytest.approx(-1.0)
    assert lv["mono_dbfs"]["max"] == -120.0
    assert lv["ch0_dbfs"]["p50"] == pytest.approx(lv["ch1_dbfs"]["p50"])


def test_a_silent_channel_has_zero_correlation_and_the_floor_level():
    lv = levels_of([lambda x: x, lambda x: x * 0])
    assert lv["corr"]["min"] == 0.0 and lv["corr"]["max"] == 0.0
    assert lv["ch1_dbfs"]["max"] == -120.0 and lv["ch0_dbfs"]["p50"] > -20
    assert lv["mono_dbfs"]["p50"] == pytest.approx(lv["ch0_dbfs"]["p50"] - 6.02, abs=0.01)


def test_mono_input_has_no_levels():
    assert levels_of([], channels=1) is None


def test_a_block_closes_only_after_level_block_s_of_input_and_the_window_is_bounded():
    half = np.full((BLOCK // 2, 2), 0.1, np.float32)
    mini, _ = voice_media([half] * 30)
    audio = ReachyAudio(mini, None, 50, voice=True, level_block_s=0.1, level_window=10)
    manual(audio, mini)
    audio.poll_once()
    assert audio.snapshot()["levels"] is None
    for _ in range(29):
        audio.poll_once()
    assert audio.snapshot()["levels"]["blocks"] == 10
    audio.stop()


def test_levels_are_plain_json_and_the_chunk_is_not_kept():
    lv = levels_of([lambda x: x, lambda x: 0.5 * x])
    assert json.loads(json.dumps(lv)) == lv
    assert all(type(v) is float for k in ("ch0_dbfs", "corr") for v in lv[k].values())


def test_a_reload_takes_the_new_stall_limit():
    """Sweep (SPEC 11A): the catch-up limit follows ``reflexes.max_age_ms`` without a restart."""
    from roboemot.config.schema import RoboEmotConfig

    mini, _ = media([])
    audio = ReachyAudio(mini, None, 50, max_gap_s=0.15)
    audio.apply_config(RoboEmotConfig(reflexes={"max_age_ms": 400}))
    assert audio.max_gap_s == 0.4


def test_the_levels_are_kept_without_the_voice_pipeline():
    """R4 S8/T6: with no listener (voice off) the reflex path still reads every chunk, and the
    channel levels are reported from it: that setup is where the question comes up."""
    rng = np.random.default_rng(2)
    base = (0.2 * rng.standard_normal(BLOCK)).astype(np.float32)
    mini, _ = voice_media([np.stack([base, base], axis=1)] * 3)
    audio = ReachyAudio(mini, lambda rms, stamp: None, 200, voice=False, level_block_s=0.1)
    audio.start()
    try:
        assert wait_for(lambda: (audio.snapshot()["levels"] or {}).get("blocks") == 3)
    finally:
        audio.stop()
    snap = audio.snapshot()
    assert snap["voice"] is None and snap["levels"]["corr"]["min"] == pytest.approx(1.0)


def test_drained_chunks_count_for_the_levels():
    chunk = np.full((BLOCK, 2), 0.1, np.float32)
    mini, _ = voice_media([chunk] * 4)
    audio = ReachyAudio(mini, lambda rms, stamp: None, 50, voice=False, level_block_s=0.1)
    audio._read_format(mini.media)
    assert audio.poll_once(drain=True)
    assert audio.snapshot()["dropped_stale"] == 3 and audio.snapshot()["levels"]["blocks"] == 4


# ------------------------------------------------- stalls of the reading thread (SPEC 4A, 8, 10)

CHUNK = 256  # 16 ms at 16 kHz, the chunk the robot's microphone gives
CHUNK_S = CHUNK / 16000
MAX_GAP_S = 0.15


class FakeMic:
    """A microphone on a fake clock: one chunk ends every ``CHUNK_S``, and each chunk's samples
    hold its capture end time (the ground truth for order and stamps).

    ``get_audio_sample`` returns the oldest chunk already captured at once. With none ready,
    ``mode="wait"`` (SDK 1.12) moves the clock to the next chunk's end and returns it, and
    ``mode="none"`` (SDK 1.13, not verified) returns ``None`` at once. The hub's stop event is
    replaced so that each wait of the reading thread advances the clock, ``stall_s`` is added to
    the wait of pass ``stall_at``, and the thread ends after ``passes`` passes. ``on_read`` runs
    before every read (a consumer that keeps up)."""

    def __init__(
        self, mode="wait", passes=200, stall_at=None, stall_s=0.0, read_cost_s=0.0, lost_s=0.0
    ):
        self.t, self.mode, self.read_cost_s, self.lost_s = 1.0, mode, read_cost_s, lost_s
        self.next_end = self.t + CHUNK_S  # end time of the next chunk to hand out
        self.passes, self.stall_at, self.stall_s = passes, stall_at, stall_s
        self.pass_n = 0
        self.on_read = lambda: None
        self.handed = []  # capture end of every chunk handed out, in order

    def clock(self):
        return self.t

    def get(self):
        self.on_read()
        if self.next_end > self.t + 1e-9:
            if self.mode == "none":
                return None
            self.t = self.next_end
        self.t += self.read_cost_s
        end, self.next_end = self.next_end, self.next_end + CHUNK_S
        self.handed.append(end)
        return np.full(CHUNK, end, np.float32)

    def is_set(self):
        return self.pass_n >= self.passes

    def wait(self, timeout):
        self.pass_n += 1
        self.t += timeout
        if self.pass_n == self.stall_at:
            self.t += self.stall_s
            self.next_end += round(self.lost_s / CHUNK_S) * CHUNK_S  # the SDK lost these
        return self.is_set()

    def hub(self, sink, voice=True):
        mini = SimpleNamespace(
            media=SimpleNamespace(
                audio=object(),
                get_audio_sample=self.get,
                get_input_audio_samplerate=lambda: 16000,
                get_input_channels=lambda: 1,
            )
        )
        audio = ReachyAudio(
            mini, sink, 50, clock=self.clock, max_gap_s=MAX_GAP_S, voice=voice, voice_queue_s=10.0
        )
        audio._open_voice(mini.media)
        audio._stop = self
        return audio


def run_stalled(mode="wait", stall_s=0.5, passes=200, read_cost_s=0.0, lost_s=0.0):
    """Run the hub's reading loop on a ``FakeMic`` with one stall; a consumer that keeps up
    reads the voice queue before every SDK read. Returns the mic, the hub, the reflex stamps
    and the voice chunks as (chunk, time received). A ``FakeMic`` chunk holds its capture end,
    so the RMS the reflex sink gets is the capture end of its chunk."""
    mic = FakeMic(
        mode, passes=passes, stall_at=20, stall_s=stall_s, read_cost_s=read_cost_s, lost_s=lost_s
    )
    reflex = []  # (capture end of the chunk the sink got, its stamp)
    audio = mic.hub(lambda rms, stamp: reflex.append((rms, stamp)))
    heard = []

    def consume():
        while (c := audio.read()) is not None:
            heard.append((c, mic.t))

    mic.on_read = consume
    audio._run()
    consume()
    return mic, audio, reflex, heard


@pytest.mark.parametrize("mode", ["wait", "none"])
def test_one_stall_drains_once_and_the_next_pass_does_not(mode):
    mic, audio, reflex, _ = run_stalled(mode)
    backlog = round(0.5 / CHUNK_S)
    assert audio.snapshot()["dropped_stale"] >= backlog - 1  # the stall's backlog, not the reflex's
    if mode == "wait":  # in step, a pass reads one chunk: only the stall's backlog is dropped
        assert audio.snapshot()["dropped_stale"] <= backlog + 1
    # not a chain of drains: the reflex only ever gets a fresh chunk, and the thread kept up
    assert max(stamp - end for end, stamp in reflex) < CHUNK_S + 1 / 50
    assert mic.t - mic.handed[-1] < MAX_GAP_S


@pytest.mark.parametrize("mode", ["wait", "none"])
def test_the_reflex_fires_at_least_every_max_gap_after_a_stall(mode):
    mic, _, reflex, _ = run_stalled(mode)
    gaps = np.diff([stamp for _, stamp in reflex])
    stall = int(np.argmax(gaps))
    assert gaps[stall] <= 0.5 + MAX_GAP_S  # the stall itself, then at once a fresh chunk
    assert max(np.delete(gaps, stall)) <= MAX_GAP_S


@pytest.mark.parametrize("mode", ["wait", "none"])
def test_the_voice_consumer_gets_every_chunk_in_order_as_soon_as_it_is_read(mode):
    mic, audio, _, heard = run_stalled(mode)
    ends = [float(c.samples[0]) for c, _ in heard]
    assert ends == pytest.approx(mic.handed)  # every chunk read, in order, none lost
    snap = audio.snapshot()["voice"]
    assert snap["dropped"] == 0 and snap["dropped_samples"] == 0
    assert not any(c.gap for c, _ in heard)
    # nothing waits for a drain to end: the longest delay is the stall, not a batch
    delays = [got - end for (c, got), end in zip(heard, ends, strict=True)]
    assert max(delays) <= 0.5 + MAX_GAP_S


@pytest.mark.parametrize("mode", ["wait", "none"])
def test_a_backlog_is_stamped_with_its_capture_times(mode):
    _, _, _, heard = run_stalled(mode)
    stamps = [c.stamp for c, _ in heard]
    ends = [float(c.samples[0]) for c, _ in heard]
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))  # strictly monotone
    backlog = [(c.stamp, end) for (c, got), end in zip(heard, ends, strict=True) if got - end > 0.1]
    assert len(backlog) >= round(0.4 / CHUNK_S)
    assert [s for s, _ in backlog] == pytest.approx([e for _, e in backlog], abs=1e-6)
    # a read in a normal pass is stamped at the read time: it trails the capture by at most
    # one poll period
    assert max(abs(s - e) for s, e in zip(stamps, ends, strict=True)) <= 1 / 50 + 1e-6


@pytest.mark.parametrize("mode", ["wait", "none"])
@pytest.mark.parametrize("read_cost_s", [0.004, 0.009])
def test_slow_reads_still_catch_up_and_never_feed_the_reflex_old_sound(mode, read_cost_s):
    """A loaded CPU (vision on the CM4) makes every read slow, queued ones too. A pass still
    reads until it is in step with the microphone, so the backlog is not left for later passes
    to feed to the reflex as if it were new, and the voice lag stays bounded. At 9 ms a pass of
    one chunk plus the idle wait takes longer than a chunk, so this also needs a catch-up outside
    a stall. The reflex is held off while the backlog is read (about 0.64 s at 9 ms)."""
    mic, audio, reflex, heard = run_stalled(mode, read_cost_s=read_cost_s)
    assert audio.snapshot()["dropped_stale"] >= round(0.5 / CHUNK_S) - 1  # the drain ran
    assert max(stamp - end for end, stamp in reflex) <= CHUNK_S + 1 / 50 + read_cost_s
    assert [float(c.samples[0]) for c, _ in heard] == pytest.approx(mic.handed)
    assert mic.t - mic.handed[-1] < MAX_GAP_S
    assert max(got - float(c.samples[0]) for c, got in heard) < 0.5 + 0.7


def test_without_an_input_rate_a_drain_lasts_no_longer_than_the_stall():
    """No rate, no chain: on SDK 1.12 every read waits for live audio, so only the wall clock
    can end the drain (it once ran 1000 reads, 16 s)."""
    mic = FakeMic("wait")
    mini = SimpleNamespace(media=SimpleNamespace(audio=object(), get_audio_sample=mic.get))
    got = []
    audio = ReachyAudio(mini, lambda r, s: got.append(r), 50, clock=mic.clock, max_gap_s=0.15)
    assert audio._in_rate is None
    t0 = mic.t
    assert audio.poll_once(drain=True, stall_s=0.1)
    assert mic.t - t0 <= 0.1 + CHUNK_S and len(mic.handed) <= round(0.1 / CHUNK_S) + 1
    assert len(got) == 1


def test_a_chained_stamp_never_ends_after_its_read_time():
    """Chunks handed over faster than they last (a burst) end at the read time, not later, so
    stamps stay monotone; a batch back-dated from the read time ran backwards."""
    mini, _ = voice_media([np.full(1600, 0.1, np.float32)] * 3, channels=1)  # 0.1 s each
    now = [49.7]
    audio = manual(ReachyAudio(mini, lambda *_: None, 50, clock=lambda: now[0], voice=True), mini)
    assert audio.poll_once()
    now[0] = 49.75
    while audio.poll_once(drain=True):
        pass
    stamps = [audio.read().stamp for _ in range(3)]
    assert stamps == pytest.approx([49.7, 49.75, 49.75])


def test_start_restarts_the_chain():
    mini, _ = voice_media([np.full(CHUNK, 0.1, np.float32)], channels=1)
    audio = manual(ReachyAudio(mini, None, 50, clock=lambda: 3.0, voice=True), mini)
    assert audio.poll_once() and audio._last_end == 3.0
    audio.start()
    try:
        assert audio._last_end is None
    finally:
        audio.stop()


@pytest.mark.parametrize("mode", ["wait", "none"])
def test_audio_lost_in_a_stall_restarts_the_stamps_instead_of_reading_live_audio_forever(mode):
    """The SDK dropped 0.3 s during the stall, so the chained stamps trail the capture by 0.3 s
    and can never catch up: the reader restarts them within a window instead of reading live
    audio until ``DRAIN_MAX`` in every pass."""
    mic, _, reflex, heard = run_stalled(mode, lost_s=0.3)
    stamps = [stamp for _, stamp in reflex]
    # one window that gains (the backlog), one that does not (live reads), then the restart
    assert max(np.diff(stamps)) <= 0.5 + 2 * MAX_GAP_S + 3 * CHUNK_S
    assert max(stamp - end for end, stamp in reflex) <= MAX_GAP_S
    late = [(c.stamp, float(c.samples[0])) for c, got in heard[-20:]]
    assert [st for st, _ in late] == pytest.approx([e for _, e in late], abs=1 / 50 + 1e-6)
