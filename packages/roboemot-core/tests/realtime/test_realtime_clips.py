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


def clips_for(rig, tmp_path, *, media=None):
    host = SimpleNamespace(
        conversation=rig.worker,
        phase=None,
        speech=None,
        voice_listener=None,
        speech_output=rig.speech,
        media=media,
        realtime_route=rig.route,
    )
    return VoiceClips(rig.worker, host, rig.stack.effective, ClipCache(tmp_path), time.monotonic)


def connect(rig) -> None:
    assert rig.route.ready(rig.stack.effective())
    assert wait_for(rig.route.up, 5.0)


def test_a_missing_clip_is_recorded_while_idle_and_then_plays(rigs, tmp_path):
    with TurnServer(transcript=RETRYING) as server:
        rig = rigs(server)
        connect(rig)
        clips = clips_for(rig, tmp_path)
        assert clips.play("retrying") is False  # nothing recorded yet
        clips.start()
        try:
            assert wait_for(lambda: clips.snapshot()["retrying"]["recorded"] == 1, 10.0)
        finally:
            clips.close()
        snap = clips.snapshot()
        assert snap["retrying"]["cached"] is True and snap["retrying"]["missing"] is False
        pushed = len(rig.sink.pushed)
        assert clips.play("retrying") is True
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
        key = clips.key("sorry")
        assert key is not None  # from the configuration: also while no session is up
        clips.cache.put("sorry", key, Clip(np.zeros(1600, np.float32), 16000))
        assert clips.play("sorry") is False
        media.on = False
        assert clips.play("sorry") is True


def test_no_clip_plays_while_the_robot_speaks(rigs, tmp_path):
    with TurnServer() as server:
        rig = rigs(server)
        clips = clips_for(rig, tmp_path)
        key = clips.key("sorry")
        clips.cache.put("sorry", key, Clip(np.zeros(16000, np.float32), 16000))
        rig.speech.say("Ciao a tutti", **_say_args(rig))
        assert wait_for(lambda: rig.speech.speaking_at(time.monotonic()), 5.0)
        assert clips.play("sorry") is False


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
        assert clips.play("retrying") is False


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
