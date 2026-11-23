"""SPEC 8 (Realtime speech route, Lost reply retry), 12, 12A: the robot records its status lines
in its own cloud voice while the conversation is idle, and plays them."""

from __future__ import annotations

import concurrent.futures
import time
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import RegisterParams, StateEffects
from roboemot.runtime.clips import CaptureSpeech, VoiceClips
from roboemot.voice.clip_cache import Clip, ClipCache, same_words
from roboemot.voice.speech import StreamFeed
from rt_rig import build_rt, wait_for
from turn_server import TurnServer

RETRYING = "Un attimo, riprovo."


class FakeMedia:
    def __init__(self, audible: bool = False) -> None:
        self.on = audible

    def playing(self) -> bool:
        return self.on

    def audible(self) -> bool:
        return self.on


@pytest.fixture()
def rigs():
    made = []

    def make(server, **kw):
        rig = build_rt(server, **kw)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def clips_for(rig, tmp_path, *, media=None, hearing=lambda: False, volume_db=lambda: 0.0):
    host = SimpleNamespace(
        conversation=rig.worker,
        phase=None,
        speech=None,
        voice_listener=None,
        speech_output=rig.speech,
        media=media,
        realtime_route=rig.route,
    )
    return VoiceClips(
        rig.worker,
        host,
        rig.stack.effective,
        ClipCache(tmp_path),
        time.monotonic,
        hearing=hearing,
        volume_db=volume_db,
    )


def cached(clips, kind, samples=None):
    """Put a clip of ``kind`` for the current key in its file and load it (as the recorder
    thread does)."""
    key = clips.key(kind)
    assert key is not None
    data = np.zeros(1600, np.float32) if samples is None else samples
    clips.cache.put(kind, key, Clip(data, 16000))
    clips.preload()
    return key


def connect(rig) -> None:
    assert rig.route.ready(rig.stack.effective())
    assert wait_for(rig.route.up, 5.0)


def test_a_missing_clip_is_recorded_while_idle_and_then_plays(rigs, tmp_path):
    with TurnServer(transcript=RETRYING) as server:
        rig = rigs(server)
        connect(rig)
        clips = clips_for(rig, tmp_path)
        assert clips.play("retrying") == "missing"  # nothing recorded yet
        clips.start()
        try:
            assert wait_for(lambda: clips.snapshot()["retrying"]["recorded"] == 1, 10.0)
        finally:
            clips.close()
        snap = clips.snapshot()
        assert snap["retrying"]["cached"] is True and snap["retrying"]["missing"] is False
        pushed = len(rig.sink.pushed)
        assert clips.play("retrying") == "played"
        assert wait_for(lambda: len(rig.sink.pushed) > pushed, 5.0)
        assert clips.snapshot()["retrying"]["played"] == 1


def test_a_recording_plays_nothing_and_leaves_no_history(rigs, tmp_path):
    with TurnServer(transcript=RETRYING) as server:
        rig = rigs(server)
        connect(rig)
        clips = clips_for(rig, tmp_path)
        history, pushed = rig.history(), len(rig.sink.pushed)
        assert clips.record_next() == "retrying"
        assert clips.snapshot()["retrying"]["recorded"] == 1
        assert len(rig.sink.pushed) == pushed  # captured, never played
        assert rig.history() == history
        create = server.of("response.create")[-1]["response"]
        assert create.get("tool_choice") == "none"
        assert RETRYING in create["instructions"]
        spoken = {e["item_id"] for e in server.sent_of("response.output_audio.delta")}
        assert spoken
        assert wait_for(  # the reply's items leave the server's conversation too
            lambda: spoken <= {e["item_id"] for e in server.of("conversation.item.delete")}, 5.0
        )


def test_a_recording_whose_words_differ_is_rejected(rigs, tmp_path):
    with TurnServer(transcript="Ciao, come stai?") as server:
        rig = rigs(server, rt={"retry": {"clip_record_attempts": 1}})
        connect(rig)
        clips = clips_for(rig, tmp_path)
        for _ in range(3):
            clips.record_next()
        assert clips.record_next() is None  # one attempt per kind today, all spent
        snap = clips.snapshot()
        assert snap["retrying"]["rejected_mismatch"] == 1
        assert snap["retrying"]["recorded"] == 0 and snap["retrying"]["missing"] is True
        assert not any(tmp_path.iterdir())


def test_no_clip_plays_over_audible_media(rigs, tmp_path):
    with TurnServer() as server:
        rig = rigs(server)
        media = FakeMedia(audible=True)
        clips = clips_for(rig, tmp_path, media=media)
        cached(clips, "sorry")  # the key is from the configuration: also while no session is up
        assert clips.play("sorry") == "refused"
        media.on = False
        assert clips.play("sorry") == "played"


def test_no_clip_plays_while_the_robot_speaks(rigs, tmp_path):
    with TurnServer() as server:
        rig = rigs(server)
        clips = clips_for(rig, tmp_path)
        cached(clips, "sorry", np.zeros(16000, np.float32))
        rig.speech.say("Ciao a tutti", **_say_args(rig))
        assert wait_for(lambda: rig.speech.speaking_at(time.monotonic()), 5.0)
        assert clips.play("sorry") == "refused"


def test_nothing_is_recorded_while_media_is_audible_or_the_session_is_down(rigs, tmp_path):
    with TurnServer(transcript=RETRYING) as server:
        rig = rigs(server)
        clips = clips_for(rig, tmp_path, media=FakeMedia(audible=True))
        assert clips.record_next() is None  # no session up: never connects just to record
        connect(rig)
        assert clips.record_next() is None  # the radio plays
        assert server.of("response.create") == []


def test_a_language_without_a_line_has_no_clip(rigs, tmp_path):
    with TurnServer() as server:
        rig = rigs(server, app={"conversation": {"language": "de"}})
        clips = clips_for(rig, tmp_path)
        assert clips.key("retrying") is None
        assert clips.play("retrying") == "refused"  # no line, so no clip is missing


def test_the_key_follows_the_route_and_its_voice(rigs, tmp_path):
    with TurnServer() as server:
        rig = rigs(server, rt={"voice": "alloy"})
        key = clips_for(rig, tmp_path).key("thinking")
        assert key is not None
        assert key.provider == "127.0.0.1" and key.model == "s2s" and key.voice == "alloy"
        assert key.language == "it" and key.line == "Fammi pensare…"


def test_same_words_ignores_case_punctuation_and_spaces():
    assert same_words("fammi pensare...", "Fammi pensare…")
    assert same_words("Un attimo riprovo", "Un attimo, riprovo.")
    assert not same_words("Un attimo", "Un attimo, riprovo.")
    assert not same_words("", "Un attimo, riprovo.")


def test_a_capture_keeps_the_audio_and_plays_nothing():
    capture = CaptureSpeech()
    feed = StreamFeed()
    job = capture.stream(feed, utterance=capture.new_utterance(), mood=None, volume_db=0.0)
    assert job is not None
    feed.put(np.full(4, 0.5, np.float32), 24000)
    feed.put(b"\x00\x40\x00\x40", 24000)
    feed.end()
    clip = capture.clip()
    assert clip is not None and clip.rate_hz == 24000
    np.testing.assert_allclose(clip.samples, [0.5] * 6)


def _say_args(rig):
    return {
        "language": "it",
        "register": RegisterParams(),
        "circadian": StateEffects(),
        "mood": None,
        "intensity": 0.0,
        "utterance": rig.speech.new_utterance(),
    }


class RaisingWorker:
    """A worker whose recording request goes out and then fails."""

    def __init__(self) -> None:
        self.calls = 0

    def record_line(self, text, language, capture):
        self.calls += 1
        future = concurrent.futures.Future()
        future.set_exception(RuntimeError("boom"))
        return future

    def busy(self) -> bool:
        return False


def test_a_recording_that_raises_counts_as_an_attempt(rigs, tmp_path):
    with TurnServer() as server:
        rig = rigs(server, rt={"retry": {"clip_record_attempts": 1}})
        connect(rig)
        clips = clips_for(rig, tmp_path)
        worker = clips.worker = RaisingWorker()
        clips.host.conversation = worker
        assert [clips.record_next() for _ in range(4)] == ["retrying", "sorry", "thinking", None]
        assert worker.calls == 3  # one per kind, then the daily cap holds
        assert all(clips.snapshot()[k]["failed"] == 1 for k in ("retrying", "sorry", "thinking"))


def test_nothing_is_recorded_with_the_retry_off(rigs, tmp_path):
    with TurnServer(transcript=RETRYING) as server:
        rig = rigs(server, rt={"retry": {"enabled": False}})
        connect(rig)
        clips = clips_for(rig, tmp_path)
        assert clips.record_next() is None
        assert server.of("response.create") == []


def test_no_clip_starts_while_someone_speaks(rigs, tmp_path):
    """A clip opens the speaking window and the listener drops what it hears then: it must not
    cut off the first words of the person's next utterance (as a one-shot cue's sound)."""
    with TurnServer() as server:
        rig = rigs(server)
        voice = {"active": True}
        clips = clips_for(rig, tmp_path, hearing=lambda: voice["active"])
        cached(clips, "thinking")
        pushed = len(rig.sink.pushed)
        assert clips.play("thinking") == "refused"
        assert clips.snapshot()["thinking"]["skipped_voice_active"] == 1
        assert len(rig.sink.pushed) == pushed
        voice["active"] = False
        assert clips.play("thinking") == "played"


class VolumeOutput:
    """A speech output that records the volume each raw job is queued at."""

    def __init__(self) -> None:
        self.volumes: list[float] = []

    def play(self, samples, rate_hz, volume_db=0.0, *, yields=False):
        self.volumes.append(volume_db)
        return len(self.volumes)

    def speaking_at(self, now) -> bool:
        return False


def test_a_clip_plays_at_the_circadian_volume(rigs, tmp_path):
    with TurnServer() as server:
        rig = rigs(server)
        clips = clips_for(rig, tmp_path, volume_db=lambda: -12.0)
        output = clips.host.speech_output = VolumeOutput()
        cached(clips, "sorry", np.full(1600, 0.5, np.float32))
        assert clips.play("sorry") == "played"
        assert output.volumes == [-12.0]


class CountingCache(ClipCache):
    def __init__(self, folder) -> None:
        super().__init__(folder)
        self.reads = 0

    def get(self, kind, key):
        self.reads += 1
        return super().get(kind, key)


def test_play_reads_no_file_and_a_new_key_waits_for_its_preload(rigs, tmp_path):
    """``play`` runs on the listener's and the worker's threads: it plays only a clip already in
    memory (the recorder thread loads them), and one of an older key never."""
    with TurnServer() as server:
        rig = rigs(server)
        clips = clips_for(rig, tmp_path)
        clips.cache = CountingCache(tmp_path)
        key = clips.key("sorry")
        clips.cache.put("sorry", key, Clip(np.zeros(1600, np.float32), 16000))
        reads = clips.cache.reads
        assert clips.play("sorry") == "missing"  # on disk, not yet in memory
        assert clips.cache.reads == reads
        clips.preload()
        reads = clips.cache.reads
        assert clips.play("sorry") == "played"
        assert clips.cache.reads == reads
        rig.stack.set_many({"conversation.language": "en"})
        assert clips.key("sorry") != key
        assert clips.play("sorry") == "missing"  # the Italian clip never plays for English
        assert clips.cache.reads == reads


def test_a_recording_holds_the_audio_input(rigs, tmp_path):
    """A person's burst must not land on the session while the recording's response runs: the
    model would answer it, and the person's turn would wait for the recording."""
    with TurnServer(transcript=RETRYING) as server:
        rig = rigs(server)
        connect(rig)
        clips = clips_for(rig, tmp_path)
        held = []
        request = rig.route.request

        def spy(*args, **kw):
            held.append(rig.route.input_held())
            return request(*args, **kw)

        rig.route.request = spy
        assert clips.record_next() == "retrying"
        assert held == [True]
        assert not rig.route.input_held()


def test_a_recording_that_raises_releases_the_audio_input(rigs, tmp_path):
    with TurnServer(transcript=RETRYING) as server:
        rig = rigs(server)
        connect(rig)
        clips = clips_for(rig, tmp_path)

        def boom(*args, **kw):
            raise RuntimeError("boom")

        rig.route.request = boom
        assert clips.record_next() == "retrying"
        assert clips.snapshot()["retrying"]["failed"] == 1
        assert not rig.route.input_held()


def test_no_recording_while_a_person_s_burst_waits_for_its_turn(rigs, tmp_path):
    with TurnServer(transcript=RETRYING) as server:
        rig = rigs(server)
        connect(rig)
        clips = clips_for(rig, tmp_path)
        rig.route._burst_at = time.monotonic()  # a burst just went: its turn is coming
        assert clips.record_next() is None  # skipped, not an attempt
        assert server.of("response.create") == []
        assert not rig.route.input_held()
        assert clips.snapshot()["retrying"]["failed"] == 0
