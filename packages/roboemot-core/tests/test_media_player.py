"""SPEC 8 (Radio playback), 11 (Voice tools, Media player), 12: the media player on a fake clock
with a fake ``PcmStream`` and a fake speech output. No network and no subprocess, except the one
decoder test, which feeds a local WAV file through gst-launch-1.0 and is skipped without it."""

from __future__ import annotations

import logging
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.bus import Bus
from roboemot.circadian.model import CircadianState
from roboemot.config.schema import RoboEmotConfig, Station
from roboemot.conversation.phase import IDLE, LISTENING, SPEAKING, THINKING
from roboemot.media import decode
from roboemot.media.decode import StreamError
from roboemot.media.player import MediaPlayer

URL = "http://radio.example.test/secret-path/stream.mp3"
JAZZ = Station(name="Jazz FM", url=URL, genres=["jazz"], language="en")
MADE: list[MediaPlayer] = []  # every player a test builds: closed at its teardown


@pytest.fixture(autouse=True)
def _close_players():
    """No feeder thread or bus subscription outlives its test."""
    yield
    while MADE:
        player = MADE.pop()
        player.close()
        assert player._thread is None or not player._thread.is_alive()


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeStream:
    """An endless live stream: every read gives ``frames`` samples at once (or None after
    ``ends_after`` reads)."""

    def __init__(self, ends_after: int | None = None) -> None:
        self.reads = 0
        self.killed = False
        self.closed = False
        self.close_thread: threading.Thread | None = None
        self.ends_after = ends_after

    def read(self, frames: int) -> np.ndarray | None:
        if self.killed or self.closed:
            return None
        if self.ends_after is not None and self.reads >= self.ends_after:
            return None
        self.reads += 1
        return np.full(frames, 0.25, np.float32)

    def kill(self) -> None:
        self.killed = True

    def close(self) -> None:
        self.killed = self.closed = True
        self.close_thread = threading.current_thread()


class FakeOutput:
    """The speech output as the player sees it: ``blocked`` stands for robot speech, a cue or
    an announcement queued or playing."""

    def __init__(self, clock=None) -> None:
        self.blocked = False
        self.pushed: list[tuple[int, int, float]] = []
        self.cuts = 0
        self.clock = clock or (lambda: 0.0)
        self.until = 0.0  # the speaker plays what it holds back to back

    def play_media(self, samples, rate_hz, volume_db=0.0) -> float:
        """The stamp until which the piece will have played, 0.0 when refused."""
        if self.blocked:
            return 0.0
        self.pushed.append((samples.size, rate_hz, volume_db))
        self.until = max(self.until, self.clock()) + samples.size / rate_hz
        return self.until

    def media_blocked(self) -> bool:
        return self.blocked

    def media_ahead_s(self) -> float:
        return 0.0

    def cut_media(self) -> bool:
        self.cuts += 1
        return True


class Phase:
    def __init__(self, name: str = IDLE, listening: bool = False) -> None:
        self.name, self.listening_flag = name, listening

    def current(self):
        return self.name, self.listening_flag

    @property
    def listening(self):  # the player must never pause on the listening phase
        raise AssertionError("the player read the listening flag")


class Conversation:
    def __init__(self) -> None:
        self.open = False

    def busy(self) -> bool:
        return self.open


class Rig:
    def __init__(self, cfg: dict | None = None, opener=None) -> None:
        self.cfg = RoboEmotConfig.model_validate(
            {"tools": {"radio": {"enabled": True, "stations": [JAZZ.model_dump()]}}, **(cfg or {})}
        )
        self.clock = Clock()
        self.bus = Bus()
        self.output = FakeOutput(self.clock)
        self.conversation = Conversation()
        self.phase = Phase()
        self.host = SimpleNamespace(
            bus=self.bus, conversation=self.conversation, phase=self.phase, voice_listener=None
        )
        self.streams: list[FakeStream] = []
        self.opened: list[tuple[str, int, float]] = []

        def open_fake(url: str, rate_hz: int, timeout_s: float) -> FakeStream:
            self.opened.append((url, rate_hz, timeout_s))
            stream = FakeStream()
            self.streams.append(stream)
            return stream

        self.player = MediaPlayer(
            self.output,
            self.host,
            lambda: self.cfg,
            opener=opener or open_fake,
            clock=self.clock,
            wait=self.wait,
        )
        MADE.append(self.player)
        self.events: list[tuple[str, dict]] = []
        for topic in ("media.playing", "media.stopped"):
            self.bus.subscribe(topic, lambda m: self.events.append((m.topic, m.payload)))

    def wait(self, seconds: float) -> bool:
        self.clock.t += seconds
        return False

    @property
    def chunk_s(self) -> float:
        return self.cfg.tools.media.chunk_ms / 1000.0

    def step(self, n: int = 1) -> int:
        """Run ``n`` feeder steps, each a chunk later; how many chunks were pushed."""
        before = len(self.output.pushed)
        for _ in range(n):
            self.player.step()
            self.clock.t += self.chunk_s
        return len(self.output.pushed) - before


def test_play_pushes_chunks_at_the_hub_rate_and_volume():
    r = Rig()
    assert r.player.play(JAZZ) == "ok"
    assert r.player.now_playing() == JAZZ
    assert r.step(5) == 5
    rate = r.cfg.voice.sample_rate_hz
    frames = round(r.cfg.tools.media.chunk_ms * rate / 1000)
    assert r.output.pushed[0] == (frames, rate, r.cfg.tools.media.volume_db)
    assert r.events == [("media.playing", {"station": "Jazz FM"})]
    assert r.opened[0][0] == URL and r.opened[0][1] == rate


def test_player_pauses_while_robot_speaks_and_resumes_after_idle():
    r = Rig()
    r.player.play(JAZZ)
    assert r.step(3) == 3
    r.output.blocked = True  # the robot speaks (or a cue, an announcement)
    reads = r.streams[0].reads
    assert r.step(10) == 0  # no chunk while it speaks
    assert r.streams[0].reads > reads  # but the live stream is read and dropped: it stays live
    r.output.blocked = False
    quiet = r.cfg.tools.media.resume_after_s
    held = int(quiet / r.chunk_s) - 1
    assert r.step(held) == 0  # still inside resume_after_s
    assert r.step(3) > 0  # rejoins the live stream after it
    assert r.player.now_playing() == JAZZ


def test_an_open_addressed_turn_pauses_the_player():
    r = Rig()
    r.player.play(JAZZ)
    r.conversation.open = True  # the wake word or an accepted utterance: a turn is open
    assert r.step(5) == 0
    r.conversation.open = False
    for name in (THINKING, SPEAKING):  # its reply is still on its way or playing
        r.phase.name = name
        assert r.step(3) == 0
    r.phase.name = IDLE
    r.step(round(r.cfg.tools.media.resume_after_s / r.chunk_s))
    assert r.step(2) == 2


def test_vad_on_radio_sound_does_not_pause_the_player():
    """Review Focus 3: the microphone hears the radio itself, so the VAD, ``voice_active()`` and
    the ``listening`` phase never pause it."""

    class Ears:
        def voice_active(self):
            raise AssertionError("the player read the VAD")

        def hearing(self):
            raise AssertionError("the player read the VAD")

        def awaiting_transcript(self):
            return False  # no addressed utterance waits for the server

        def reply_window_open(self):
            return False  # no announcement's reply window is open

    r = Rig()
    r.host.voice_listener = Ears()
    r.phase.name, r.phase.listening_flag = LISTENING, True
    r.player.play(JAZZ)
    assert r.step(10) == 10


def test_the_announcements_reply_window_keeps_the_player_paused():
    """Review I2 (owner decision 2026-11-17): while an announcement's reply window is open the
    radio stays paused, so the reply is heard without the wake word; it rejoins after the
    window closes and ``resume_after_s`` passed."""

    class Ears:
        reply = False

        def awaiting_transcript(self):
            return False

        def reply_window_open(self):
            return self.reply

        def voice_active(self):
            raise AssertionError("the player read the VAD")

    r = Rig()
    r.host.voice_listener = ears = Ears()
    r.player.play(JAZZ)
    assert r.step(2) == 2
    ears.reply = True
    assert r.step(10) == 0
    ears.reply = False
    r.step(round(r.cfg.tools.media.resume_after_s / r.chunk_s))
    assert r.step(2) == 2


def test_pause_from_the_announcer_pauses_now_and_resumes_after_idle():
    r = Rig()
    r.player.play(JAZZ)
    r.step(2)
    r.player.pause("announce")
    assert r.output.cuts >= 1  # what is still in the sink stops now
    assert r.step(round(r.cfg.tools.media.resume_after_s / r.chunk_s) - 1) == 0
    assert r.step(3) > 0
    assert r.player.now_playing() == JAZZ


def circadian(state: str) -> CircadianState:
    return CircadianState(12.0, 0.0, 0.0, 0.0, 0.0, state, None)


def test_player_stops_on_nap_and_asleep():
    for topic, payload, reason in (
        ("circadian.state", circadian("resting"), "nap"),
        ("robot.asleep", None, "asleep"),
    ):
        r = Rig()
        r.player.start(feeder=False)
        r.player.play(JAZZ)
        r.bus.publish("circadian.state", circadian("awake"), 0.0)
        assert r.player.now_playing() == JAZZ  # awake keeps it playing
        r.bus.publish(topic, payload, 0.0)
        assert r.player.now_playing() is None
        assert r.streams[0].killed
        assert r.events[-1] == ("media.stopped", {"station": "Jazz FM", "reason": reason})
        assert r.step(3) == 0


def test_player_refuses_to_play_while_asleep():
    r = Rig()
    r.player.start(feeder=False)
    r.bus.publish("robot.asleep", None, 0.0)
    assert r.player.play(JAZZ) == "disabled" and r.opened == []
    r.bus.publish("robot.awake", None, 0.0)
    assert r.player.play(JAZZ) == "ok"


def test_player_stops_after_max_play_s():
    r = Rig()
    r.player.play(JAZZ)
    r.step(2)
    r.clock.t = r.cfg.tools.media.max_play_s + 1
    r.step()
    assert r.player.now_playing() is None and r.streams[0].killed
    assert r.events[-1] == ("media.stopped", {"station": "Jazz FM", "reason": "max_play"})


def test_a_stream_that_ends_or_drops_stops_the_player(caplog):
    r = Rig()
    r.player.play(JAZZ)
    r.streams[0].ends_after = r.streams[0].reads + 2
    with caplog.at_level(logging.INFO, logger="roboemot.media.player"):
        r.step(4)
    assert r.player.now_playing() is None
    assert r.events[-1] == ("media.stopped", {"station": "Jazz FM", "reason": "dropped"})
    assert "Jazz FM" in caplog.text and URL not in caplog.text


def test_unreachable_stream_reports_error_and_logs_no_url(caplog):
    def refuse(url, rate_hz, timeout_s):
        raise StreamError("unreachable")

    def boom(url, rate_hz, timeout_s):
        raise OSError(f"cannot open {url}")  # a decoder error may quote the URL

    for opener in (refuse, boom):
        caplog.clear()
        r = Rig(opener=opener)
        with caplog.at_level(logging.DEBUG):
            assert r.player.play(JAZZ) == "unreachable"
        assert r.player.now_playing() is None and r.events == []
        assert "Jazz FM" in caplog.text
        assert URL not in caplog.text and "secret-path" not in caplog.text


def test_play_waits_at_most_the_given_connect_timeout():
    r = Rig()
    r.player.play(JAZZ, connect_timeout_s=2.5)
    r.player.play(JAZZ)
    assert [t for _, _, t in r.opened] == [2.5, r.cfg.tools.media.connect_timeout_s]


def test_a_new_station_replaces_the_one_playing():
    r = Rig()
    other = Station(name="Rock", url="https://rock.example.test/live", genres=["rock"])
    r.player.play(JAZZ)
    r.player.play(other)
    assert r.player.now_playing() == other and r.streams[0].killed
    assert r.events == [
        ("media.playing", {"station": "Jazz FM"}),
        ("media.stopped", {"station": "Jazz FM", "reason": "replaced"}),
        ("media.playing", {"station": "Rock"}),
    ]


def test_stop_ends_the_stream_and_cuts_the_music():
    r = Rig()
    r.player.play(JAZZ)
    r.step(2)
    r.player.stop("tool")
    assert r.player.now_playing() is None and r.streams[0].killed and r.output.cuts >= 1
    assert r.events[-1] == ("media.stopped", {"station": "Jazz FM", "reason": "tool"})
    r.player.stop("tool")  # nothing plays: no second event
    assert [e for e, _ in r.events].count("media.stopped") == 1


def test_volume_steps_within_the_configured_range():
    r = Rig({"tools": {"media": {"volume_db": -6}}})
    assert r.player.volume(3) == pytest.approx(-3)
    assert r.player.volume(30) == pytest.approx(0)  # the ceiling of tools.media.volume_db
    assert r.player.volume(-100) == pytest.approx(-40)  # and its floor
    r.player.volume(10)
    r.player.play(JAZZ)
    r.step()
    assert r.output.pushed[-1][2] == pytest.approx(-30)


def test_the_feeder_thread_plays_and_stops_on_close():
    r = Rig()
    r.player = MediaPlayer(r.output, r.host, lambda: r.cfg, opener=lambda *a: FakeStream())
    MADE.append(r.player)
    r.player.start()
    r.player.play(JAZZ)
    done = threading.Event()
    for _ in range(200):
        if r.output.pushed:
            done.set()
            break
        threading.Event().wait(0.01)
    assert done.is_set()
    r.player.close()
    assert not r.player._thread.is_alive()


@pytest.mark.skipif(not decode.available(), reason="gst-launch-1.0 is not installed")
def test_the_decoder_turns_a_local_wav_into_pcm_at_the_asked_rate(tmp_path):
    src_rate, rate = 22050, 16000
    t = np.arange(src_rate) / src_rate
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(src_rate)
        w.writeframes(tone.tobytes())
    stream = decode.open_pipeline(["filesrc", f'location="{path}"'], rate, 5.0)
    try:
        got = [stream.read(1600) for _ in range(5)]
        assert all(g is not None and g.size == 1600 for g in got)
        assert 0.3 < float(np.max(np.abs(np.concatenate(got)))) < 0.6
        while stream.read(1600) is not None:
            pass  # the file ends: the stream ends
    finally:
        stream.close()


def test_the_decoder_argv_quotes_the_url_and_takes_no_proxy(monkeypatch):
    argv = decode.pipeline(decode.http_source(URL), 16000)
    assert argv[0] == decode.GST_LAUNCH and "-q" in argv
    assert f'location="{URL}"' in argv and "automatic-redirect=false" in argv
    assert "iradio-mode=false" in argv  # no stream metadata asked for
    monkeypatch.setenv("http_proxy", "http://proxy.test:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
    env = decode.child_env()
    assert "http_proxy" not in env and "HTTPS_PROXY" not in env


def test_no_decoder_is_a_stream_error(monkeypatch):
    monkeypatch.setattr(decode, "available", lambda: False)
    with pytest.raises(StreamError) as err:
        decode.open_stream(URL, 16000, 1.0)
    assert URL not in str(err.value)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://radio.example.test/a",
        "http://user:pw@radio.example.test/a",
        'http://radio.example.test/a"!fakesink',
        "http://radio.example.test/a b",
        "http://radio.example.test/a'x",
        "http://radio.example.test/a!x",
        "radio.example.test/a",
    ],
)
def test_station_urls_are_checked_at_load(url):
    with pytest.raises(ValueError):
        Station(name="x", url=url)


def test_plain_http_and_shoutcast_urls_are_accepted():
    for url in ("http://radio.example.test:8000/;", "https://radio.example.test/live?type=.mp3"):
        assert Station(name="x", url=url).url == url


# ------------------------------------------------------------------ fix round 1 (review T6)


def test_the_decoder_env_disables_adaptive_demuxers_and_debug_output(monkeypatch):
    """Review I1, m7: a station must never make the robot fetch other URLs (an HLS, DASH or
    Smooth Streaming manifest would), and no GStreamer debug file may record the URL."""
    monkeypatch.setenv("GST_DEBUG", "5")
    monkeypatch.setenv("GST_DEBUG_FILE", "/tmp/gst.log")
    monkeypatch.setenv("GST_TRACERS", "latency")
    monkeypatch.setenv("GST_PLUGIN_FEATURE_RANK", "hlsdemux:MAX")
    env = decode.child_env()
    ranks = dict(item.split(":") for item in env["GST_PLUGIN_FEATURE_RANK"].split(","))
    for demuxer in ("hlsdemux", "hlsdemux2", "dashdemux", "dashdemux2", "mssdemux", "mssdemux2"):
        assert ranks[demuxer] == "NONE"
    assert not any(k.startswith(("GST_DEBUG", "GST_TRACERS")) for k in env)


class PipeProc:
    """A stand-in for the decoder process over a real OS pipe (no subprocess): ``kill`` only
    marks it; the test writes the PCM itself."""

    def __init__(self) -> None:
        import os

        self.read_fd, self.write_fd = os.pipe()
        self.stdout = os.fdopen(self.read_fd, "rb", buffering=0)
        self.killed = False

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        return 0


def test_a_kill_during_a_blocked_read_never_delivers_the_old_streams_bytes():
    """Review I2: another thread only signals (``kill``); bytes that arrive after it are never
    delivered, and the reader alone closes the pipe."""
    import os

    proc = PipeProc()
    stream = decode.GstStream(proc, 5.0)
    got: list[object] = []
    reading = threading.Event()

    def reader() -> None:
        reading.set()
        got.append(stream.read(4))

    t = threading.Thread(target=reader)
    t.start()
    assert reading.wait(2.0)
    threading.Event().wait(0.05)  # the reader is blocked in select
    stream.kill()
    assert not proc.stdout.closed  # never closed by the killing thread
    os.write(proc.write_fd, b"\x00\x10" * 4)  # the old decoder's last bytes
    t.join(2.0)
    os.close(proc.write_fd)
    assert not t.is_alive() and got == [None]
    assert proc.killed and proc.stdout.closed  # closed by the reader once it saw the kill


def test_play_kills_the_old_stream_before_it_opens_the_new_one():
    r = Rig()
    seen_old_killed: list[bool] = []
    real = r.player._opener

    def opener(url, rate_hz, timeout_s):
        if r.streams:
            seen_old_killed.append(r.streams[-1].killed)
        return real(url, rate_hz, timeout_s)

    r.player._opener = opener
    r.player.play(JAZZ)
    r.player.play(JAZZ)
    assert seen_old_killed == [True]


def test_retired_streams_are_closed_on_the_feeder_thread():
    r = Rig()
    r.player = MediaPlayer(r.output, r.host, lambda: r.cfg, opener=lambda *a: FakeStream())
    MADE.append(r.player)
    made: list[FakeStream] = []

    def opener(*args):
        made.append(FakeStream())
        return made[-1]

    r.player._opener = opener
    r.player.start()
    r.player.play(JAZZ)
    r.player.play(JAZZ)
    for _ in range(200):
        if made[0].closed:
            break
        threading.Event().wait(0.01)
    assert made[0].closed and made[0].close_thread is r.player._thread
    r.player.close()


def test_turning_the_radio_off_or_removing_the_station_stops_playback():
    """Review I3: withdrawing the opt-in ends the outbound stream at the next config change."""
    r = Rig()
    r.player.play(JAZZ)
    r.player.on_config(SimpleNamespace(config=r.cfg))  # unrelated change: keeps playing
    assert r.player.now_playing() == JAZZ
    off = RoboEmotConfig.model_validate(
        {"tools": {"radio": {"enabled": False, "stations": [JAZZ.model_dump()]}}}
    )
    r.player.on_config(SimpleNamespace(config=off))
    assert r.player.now_playing() is None
    assert r.events[-1] == ("media.stopped", {"station": "Jazz FM", "reason": "disabled"})
    r.player.play(JAZZ)
    other = Station(name="Rock", url="https://rock.example.test/live")
    gone = RoboEmotConfig.model_validate(
        {"tools": {"radio": {"enabled": True, "stations": [other.model_dump()]}}}
    )
    r.player.on_config(SimpleNamespace(config=gone))
    assert r.player.now_playing() is None
    assert r.events[-1] == ("media.stopped", {"station": "Jazz FM", "reason": "removed"})


def test_a_transcript_in_flight_pauses_the_player():
    """Review I4: on the realtime audio path the wake decision is taken before the server's
    transcript comes; the radio pauses from then on."""

    class Ears:
        waiting = False

        def awaiting_transcript(self):
            return self.waiting

        def reply_window_open(self):
            return False

        def voice_active(self):
            raise AssertionError("the player read the VAD")

    r = Rig()
    r.host.voice_listener = ears = Ears()
    r.player.play(JAZZ)
    assert r.step(2) == 2
    ears.waiting = True
    assert r.step(5) == 0
    ears.waiting = False
    r.step(round(r.cfg.tools.media.resume_after_s / r.chunk_s))
    assert r.step(2) == 2


def test_audible_follows_the_holds_by_stamp():
    """SPEC 8, Radio playback: the radio rule applies only while the music can be heard. A
    hold covers its start to its last moment plus ``resume_after_s``; the listener asks by an
    utterance's start stamp, and ``hold_from`` back-dates a turn's hold."""
    r = Rig()
    resume = r.cfg.tools.media.resume_after_s
    assert not r.player.audible()  # no station
    r.clock.t = 10.0
    r.player.play(JAZZ)
    assert r.player.audible() and not r.player.audible_at(9.0)  # not before it was set
    r.step(2)
    r.output.blocked = True  # the robot speaks
    held_from = r.clock.t
    r.step(5)
    last_hold = r.clock.t - r.chunk_s
    r.output.blocked = False
    assert not r.player.audible_at(held_from) and not r.player.audible()
    assert r.player.audible_at(held_from - r.chunk_s)  # before the hold: audible
    r.step(round(resume / r.chunk_s) + 1)
    assert r.player.audible()
    assert not r.player.audible_at(last_hold + resume - r.chunk_s)  # by stamp, after the fact
    r.step(10)
    r.player.hold_from(r.clock.t - 0.5)  # a turn the wake word opened half a second ago
    assert not r.player.audible_at(r.clock.t - 0.4)
    assert r.player.audible_at(r.clock.t - 0.6)
    r.player.stop("user")
    assert not r.player.audible_at(r.clock.t - 0.6)  # no station: nothing is audible


def test_each_piece_that_played_goes_on_with_its_play_stamp():
    """SPEC 8, Radio playback (dance): behind a hardware echo canceller the microphone does not
    hear the radio, so the listener gets the decoded pieces, before the volume, stamped with the
    moment each starts to play; a refused piece goes nowhere, and a consumer error never stops
    the music."""
    r = Rig()
    got: list[tuple[float, int, float]] = []
    r.player.on_piece = lambda pcm, rate, start: got.append((float(pcm[0]), rate, start))
    r.clock.t = 5.0
    r.player.play(JAZZ)
    assert r.step(2) == 2
    rate = r.cfg.voice.sample_rate_hz
    assert [g[:2] for g in got] == [(0.25, rate)] * 2  # the stream's own level
    assert got[0][2] == pytest.approx(5.0)  # the speaker was idle: it starts now
    assert got[1][2] == pytest.approx(5.0 + r.chunk_s)  # back to back after the first
    r.output.blocked = True
    r.step(2)
    assert len(got) == 2  # held: nothing played, nothing goes on
    r.output.blocked = False
    r.player.on_piece = lambda pcm, rate, start: 1 / 0
    r.step(round(r.cfg.tools.media.resume_after_s / r.chunk_s) + 2)
    assert r.player.now_playing() == JAZZ and len(r.output.pushed) > 2
