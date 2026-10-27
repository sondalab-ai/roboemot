"""Addressed audio leaves the robot, after the local decision (SPEC 8 stage 2 and Realtime
speech route; SPEC 12): the listener, the realtime input, the route and the worker against the
fake server. Unaddressed speech, a voice-at-face reject, the robot speaking or asleep and the
opt-in off send no audio frame; the wake word on the Vosk path sends the buffer; a transcript
that does not come goes to a local decode and the chat routes."""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest
from realtime_helpers import wait_until
from roboemot.realtime.input import RealtimeInput
from roboemot.voice.direction import DirectionSample, VoiceDirection
from rt_rig import build_rt
from test_voice_listener import CHUNK, RATE, FakeASR, Rig, silence, voice
from turn_server import TurnServer

FACE_RAD = 0.2
OFF_FACE_RAD = 1.3


class DecodeASR(FakeASR):
    """The fake recognizer, plus the whole-utterance decode of the realtime fallback."""

    def __init__(self, script, final=None, decoded=None, decode_s=0.0):
        super().__init__(script, final)
        self.decoded = decoded
        self.decodes = 0
        self.decode_s = decode_s  # a slow decode (a long utterance on the CM4)
        self.decode_threads: list[str] = []

    def decode(self, language, samples):
        import threading

        self.decodes += 1
        self.decode_threads.append(threading.current_thread().name)
        if self.decode_s:
            time.sleep(self.decode_s)
        return self.decoded if self.decoded is not None else self.finish()


class Ear:
    """One listener (fake ASR, energy VAD, fake clock) bound to a real realtime input, route,
    client and worker against the fake server."""

    def __init__(
        self,
        rt,
        *,
        script,
        final=None,
        decoded=None,
        bearing=None,
        to_worker=False,
        decode_s=0.0,
        background=False,
        asleep=lambda: False,
    ):
        self.rt = rt
        cfg = rt.stack.effective()
        direction = None
        if bearing is not None:
            self.bearing = bearing

            def lookup(now):
                return DirectionSample(self.bearing, (FACE_RAD,), True)

            direction = VoiceDirection(cfg.voice.direction, lookup, doa_available=True)
        self.r = Rig(script, final, cfg=cfg, direction=direction, grammar_background=background)
        self.r.asr = DecodeASR(script, final, decoded, decode_s)
        self.r.asr.loaded = list(cfg.voice.asr.models)
        self.r.listener._asr = self.r.asr
        self.input = RealtimeInput(rt.route, asleep=lambda: asleep())
        self.r.listener.bind_realtime(self.input)
        if to_worker:
            sink = self.r.listener._on_utterance

            def both(u):
                sink(u)
                rt.worker.submit(u)

            self.r.listener._on_utterance = both
        assert rt.route.ready()

    @property
    def utterances(self):
        return self.r.utterances

    def say(self, seconds=1.0):
        self.r.say(seconds)

    def settle(self, cond, timeout_s=5.0):
        """Poll the hand-off (as the listener thread does between reads) until ``cond``."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            self.r.listener.poll_handoff()
            if cond():
                return True
            time.sleep(0.01)
        self.r.listener.poll_handoff()
        return cond()

    def close(self):
        self.r.listener.stop()
        self.input.close()


@pytest.fixture
def ears():
    made = []

    def make(server, *, policy="never", rt=None, app=None, **kw):
        voice_cfg = {"wake_word_policy": policy}
        cfg = {"privacy": {"realtime_audio": True}, "voice": voice_cfg, **(app or {})}
        rig = build_rt(server, rt={"input": "audio", **(rt or {})}, app=cfg)
        ear = Ear(rig, **kw)
        made.append(ear)
        return ear

    yield make
    for ear in made:
        ear.close()
        ear.rt.close()


def appended(server):
    return server.of("input_audio_buffer.append")


# ------------------------------------------------------------------ nothing leaves


def test_an_unaddressed_utterance_sends_no_audio(ears):
    """No face, no wake word: the wake word is required and absent."""
    with TurnServer() as server:
        ear = ears(server, policy="always", script=["come stai"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert not ear.utterances[0].addressed
        assert ear.r.asr.started  # the Vosk path
        assert appended(server) == [] and server.of("input_audio_buffer.commit") == []


def test_a_voice_at_face_reject_is_never_sent(ears):
    with TurnServer() as server:
        ear = ears(server, script=["la la la amore mio"], bearing=OFF_FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        assert not u.addressed and u.text == "la la la amore mio"
        assert ear.r.asr.started == []  # addressable at its start: Vosk skipped, then decoded
        assert ear.r.asr.decodes == 1
        snap = ear.r.listener.snapshot()
        assert snap["wake"]["decisions"]["rejected_no_face_at_voice"] == 1
        assert sum(snap["wake"]["decisions"].values()) == 1  # decided once
        assert snap["direction"]["checks"]["rejected"] == 1  # checked once
        assert appended(server) == []


def test_a_rejected_voice_with_the_wake_word_in_its_buffer_is_sent(ears):
    with TurnServer(heard=[["ciao robot accendi la luce"]]) as server:
        ear = ears(server, script=["ciao robot accendi"], bearing=OFF_FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        assert u.addressed and u.text == "ciao robot accendi la luce"
        assert u.realtime is not None
        assert len(server.of("input_audio_buffer.commit")) == 1


def test_nothing_is_sent_while_the_robot_speaks(ears):
    with TurnServer() as server:
        ear = ears(server, script=["ciao robot come stai"])
        ear.r.speech.extend(ear.r.t + 10.0)
        ear.say()
        assert ear.settle(lambda: ear.r.listener.snapshot()["dropped_speaking"] > 0)
        assert ear.utterances == []
        assert appended(server) == []


def test_nothing_is_sent_while_asleep(ears):
    """Sleep closes the server session (SleepGate): an addressed utterance goes the Vosk way."""
    with TurnServer() as server:
        ear = ears(server, script=["ciao robot come stai"])
        ear.rt.route.close_soon("asleep")
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].addressed and ear.utterances[0].realtime is None
        assert ear.r.asr.started  # not addressable for the audio path: Vosk ran
        assert appended(server) == []


def test_nothing_is_sent_once_the_opt_in_is_off_even_before_the_listener_reloads(ears):
    with TurnServer() as server:
        ear = ears(server, script=["come stai"], decoded="come stai")
        ear.rt.stack.set("privacy.realtime_audio", False, source="test", admin=True)
        ear.say()  # the listener's config still has the opt-in: the route's gate refuses
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        assert u.addressed and u.text == "come stai" and u.realtime is None
        assert not u.local_only  # nothing left the robot: the text path as usual
        assert appended(server) == []
        assert ear.r.listener.snapshot()["realtime_audio"]["vosk_skipped"] == 1  # it tried
        snap = ear.rt.route.snapshot()
        assert snap["audio_refused"] == 1 and snap["sends_refused"] == 1  # the gate said no
        ear.r.listener.apply_config(ear.rt.stack.effective())
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 2)
        assert ear.r.asr.started  # the opt-in off: Vosk runs again
        assert appended(server) == []


# ------------------------------------------------------------------ what leaves


def test_an_addressable_utterance_skips_vosk_and_its_final_is_the_server_text(ears):
    with TurnServer(heard=[["ciao Neve come stai"]]) as server:
        ear = ears(server, script=["vosk text"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        assert u.addressed and u.text == "ciao Neve come stai"
        assert u.realtime == (ear.rt.route.generation, ("in_1_0",))
        assert ear.r.asr.started == [] and ear.r.asr.decodes == 0
        assert len(server.of("input_audio_buffer.commit")) == 1
        snap = ear.r.listener.snapshot()
        assert snap["realtime_audio"]["vosk_skipped"] == 1
        assert snap["realtime_audio"]["transcribed"] == 1


def test_the_wake_word_on_the_vosk_path_then_sends_the_buffer(ears):
    with TurnServer(heard=[["ciao robot che ore sono"]]) as server:
        ear = ears(server, policy="always", script=["ciao robot che ore"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        assert ear.r.asr.started  # the Vosk path: not addressable at its start
        assert u.addressed and u.wake_word
        assert u.text == "ciao robot che ore sono"  # the server's final
        assert len(server.of("input_audio_buffer.commit")) == 1
        pcm = server.audio()
        silence_ms = ear.rt.stack.effective().voice.realtime_route.close_silence_ms
        assert len(pcm) > 2 * RATE * (1.0 + silence_ms / 1000) * 0.9


def test_keywords_and_speech_fast_come_from_the_server_text(ears):
    fast = "stop " + " ".join(["parola"] * 12)  # many words in one second of speech
    with TurnServer(heard=[[fast]]) as server:
        ear = ears(server, script=["vosk"], bearing=FACE_RAD)
        ear.say(1.0)
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        assert "stop" in u.keywords
        names = [n for n, _ in ear.r.events]
        assert "speech.keyword.stop" in names and "speech.fast" in names
        assert u.prosody.words_per_s is not None and u.prosody.words_per_s > 6


def test_a_pause_inside_one_utterance_and_a_server_that_splits_it_give_one_answer(ears):
    end_silence = 0.5
    with TurnServer(heard=[["ciao Neve", "come stai"]], transcript_delay_s=0.1) as server:
        ear = ears(server, script=["x"], bearing=FACE_RAD, to_worker=True)
        ear.r.feed(voice(seconds=0.6))
        ear.r.feed(silence(end_silence * 0.5))  # a pause the endpointer keeps inside
        ear.say(0.6)
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].text == "ciao Neve come stai"
        assert len(server.of("input_audio_buffer.commit")) == 1
        assert ear.settle(lambda: len(ear.rt.events) == 1, 10.0)
        assert len(server.of("response.create")) == 1
        assert server.of("conversation.item.create") == []  # the server holds the words
        history = ear.rt.history()
        assert history[0][0]["rt"][1][:2] == ["in_1_0", "in_1_1"]  # then the reply item


def test_a_transcript_timeout_goes_to_the_local_decode_and_the_chat_path(ears):
    with TurnServer(heard=[[]]) as server:
        ear = ears(server, script=["x"], decoded="che tempo fa", bearing=FACE_RAD, to_worker=True)
        ear.say()
        assert ear.settle(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        assert ear.utterances == []  # waiting for the server
        timeout = ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s
        ear.r.now += timeout  # the listener's clock passes the deadline
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        assert u.text == "che tempo fa" and u.local_only and u.realtime is None
        assert ear.r.listener.snapshot()["realtime_audio"]["transcript_timeouts"] == 1
        assert ear.rt.route.snapshot()["transcript_timeouts"] == 1
        assert ear.settle(lambda: len(ear.rt.events) == 1, 10.0)
        assert ear.rt.chat_hosts()  # the chat routes answered
        assert server.of("response.create") == []


def test_the_upload_budget_sends_the_next_utterance_as_text(ears):
    with TurnServer(heard=[["primo"]]) as server:
        ear = ears(
            server,
            script=["x"],
            decoded="secondo",
            bearing=FACE_RAD,
            rt={"daily_upload_budget_s": 2.0},
        )
        ear.say(1.0)  # 1.0 s and more of speech plus 0.9 s of silence: the budget is spent
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].realtime is not None
        ear.say(1.0)
        assert ear.settle(lambda: len(ear.utterances) == 2)
        second = ear.utterances[1]
        assert second.text == "secondo" and second.realtime is None and not second.local_only
        assert len(server.of("input_audio_buffer.commit")) == 1


def test_the_listener_never_waits_on_the_network(ears):
    """The listener's own thread reads paced audio while the server's transcript is late: it
    never lags behind by the transcript's delay, its share of the audio time stays under
    budget, and no hand-off wait is spent on it."""
    delay = 1.5
    with TurnServer(heard=[["ciao Neve"]], transcript_delay_s=delay) as server:
        ear = ears(server, script=["x"], bearing=FACE_RAD)
        r, listener = ear.r, ear.r.listener
        chunks: queue.Queue = queue.Queue()

        def source(timeout):
            try:
                return chunks.get(timeout=timeout)
            except queue.Empty:
                return None

        listener._source = source
        listener.start()
        assert wait_until(lambda: listener.snapshot()["state"] == "running")
        audio = np.concatenate([voice(seconds=1.0), silence(0.6)])
        t0 = time.monotonic()
        i = 0
        while not ear.utterances and time.monotonic() - t0 < delay + 4.0:
            part = audio[i : i + CHUNK] if i < audio.size else silence(CHUNK / RATE)
            i += CHUNK
            r.t += part.size / RATE
            r.now = r.t
            chunks.put(SimpleNamespace(samples=part, rate_hz=RATE, stamp=r.t, gap=False))
            time.sleep(part.size / RATE)  # paced like the microphone
        assert wait_until(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].text == "ciao Neve"
        listener.stop()
        total = listener.snapshot()["realtime"]["total"]
        assert total["audio_s"] > 1.6 + delay * 0.5  # it kept reading while it waited
        assert total["lag_max_s"] < delay / 2
        budget = ear.rt.stack.effective().voice.listener.budget_share
        assert total["share"] < budget
        assert total["share"] * total["audio_s"] < delay / 4  # busy time, not the delay
        assert total["ms_per_audio_s"]["grammar_wait"] == 0.0
        assert total["ms_per_audio_s"]["asr_accept"] == 0.0  # Vosk skipped


def test_the_server_text_hears_the_name_the_vosk_vocabulary_lacks():
    """The name's words missing from the Vosk vocabulary: the local text cannot wake with it,
    the server's transcript can (its recognizer has no such limit)."""
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.voice.fastpath import FastPath
    from roboemot.voice.robot_name import RobotName

    cfg = RoboEmotConfig.model_validate({"voice": {"wake_word": "Neve"}})
    names = RobotName(missing=lambda language, words: list(words))  # nothing is in vocabulary
    fast = FastPath(cfg.voice, lambda *a: None, names=names)
    audio = voice(seconds=0.5)
    ends = {"last_speech_stamp": 0.0, "endpoint_stamp": 0.0}
    fast.begin(0.0, 0.0, "it")
    assert not fast.end("ciao neve", audio, RATE, **ends).wake_word
    fast.begin(0.0, 0.0, "it")
    assert fast.end("ciao neve", audio, RATE, **ends, server=True).wake_word


def paced(ear, until, max_s):
    """Run the listener's own thread on microphone-paced audio (one utterance, then silence)
    until ``until()`` or ``max_s``; returns the listener's realtime account."""
    r, listener = ear.r, ear.r.listener
    chunks: queue.Queue = queue.Queue()

    def source(timeout):
        try:
            return chunks.get(timeout=timeout)
        except queue.Empty:
            return None

    listener._source = source
    listener.start()
    assert wait_until(lambda: listener.snapshot()["state"] == "running")
    audio = np.concatenate([voice(seconds=1.0), silence(0.6)])
    t0 = time.monotonic()
    i = 0
    while not until() and time.monotonic() - t0 < max_s:
        part = audio[i : i + CHUNK] if i < audio.size else silence(CHUNK / RATE)
        i += CHUNK
        r.t += part.size / RATE
        r.now = r.t
        chunks.put(SimpleNamespace(samples=part, rate_hz=RATE, stamp=r.t, gap=False))
        time.sleep(part.size / RATE)
    listener.stop()
    return listener.snapshot()["realtime"]["total"]


def test_a_voice_at_face_reject_is_decoded_off_the_listener_thread(ears):
    """I-5: the whole-utterance decode of a reject never stalls the microphone."""
    slow = 1.5
    with TurnServer() as server:
        ear = ears(
            server,
            script=["x"],
            decoded="la la la",
            bearing=OFF_FACE_RAD,
            decode_s=slow,
            background=True,
        )
        total = paced(ear, lambda: bool(ear.utterances), slow + 4.0)
        assert len(ear.utterances) == 1 and not ear.utterances[0].addressed
        assert ear.r.asr.decode_threads == ["roboemot-decode"]
        assert total["lag_max_s"] < slow / 2
        assert total["share"] * total["audio_s"] < slow / 4  # listener busy time
        assert total["ms_per_audio_s"]["decode"] > 0  # spent, on its own thread
        assert appended(server) == []


def test_a_transcript_timeout_is_decoded_off_the_listener_thread(ears):
    slow = 1.5
    with TurnServer(heard=[[]]) as server:
        ear = ears(
            server,
            script=["x"],
            decoded="che ore sono",
            bearing=FACE_RAD,
            decode_s=slow,
            background=True,
            rt={"transcript_timeout_s": 0.5},
        )
        total = paced(ear, lambda: bool(ear.utterances), slow + 5.0)
        assert len(ear.utterances) == 1
        u = ear.utterances[0]
        assert u.text == "che ore sono" and u.local_only
        assert ear.r.asr.decode_threads == ["roboemot-decode"]
        assert total["lag_max_s"] < slow / 2
        assert total["share"] * total["audio_s"] < slow / 4


def sleep_gate(ear, pause):
    """The runtime's own SleepGate over the rig's listener and route (the other parts empty)."""
    from roboemot.runtime.sleep import SleepGate

    loop = SimpleNamespace(
        motors_off=False, link=SimpleNamespace(online=True), clear=lambda source: None
    )
    gate = SleepGate(
        SimpleNamespace(conversation=ear.rt.worker, speech_output=ear.rt.speech),
        SimpleNamespace(publish=lambda *a, **k: None),
        loop,
        SimpleNamespace(service=None),
        SimpleNamespace(pause=lambda: None, resume=lambda: None, paused=False),
        ear.r.listener,
        SimpleNamespace(release=lambda: None),
        None,
        lambda: pause,
        lambda: ear.rt.route,
    )
    return gate, loop


@pytest.mark.parametrize("pause", [False, True])
def test_a_sleeping_robot_sends_no_audio_whatever_the_pause(ears, pause):
    """B-1: with runtime.pause_when_asleep off nothing is paused or closed, and asleep is
    still a send condition; with it on, an utterance queued behind a burst in flight is not
    sent after sleep began."""
    box = {}
    with TurnServer(heard=[["primo"]], speech_delay_s=[2.0]) as server:
        ear = ears(
            server,
            script=["x"],
            decoded="secondo",
            bearing=FACE_RAD,
            asleep=lambda: box["gate"].asleep,
        )
        gate, loop = sleep_gate(ear, pause)
        box["gate"] = gate
        ear.say()  # the first burst goes; its transcript is late
        assert ear.settle(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        ear.say()  # the second waits behind it in the input
        assert ear.r.listener.snapshot()["realtime_audio"]["sent"] == 2
        loop.motors_off = True
        gate.tick(time.monotonic())
        assert gate.asleep and gate.paused == pause
        if pause:  # the pause closed the session: a queued worker turn reopens it (path 2)
            assert ear.rt.route.ready()
        # the queued burst reaches the gate with a session open: asleep is what refuses it
        assert ear.settle(lambda: ear.rt.route.snapshot()["audio_blocked"] is not None)
        assert ear.rt.route.snapshot()["audio_blocked"] == "asleep"
        assert len(server.of("input_audio_buffer.commit")) == 1
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s
        assert ear.settle(lambda: len(ear.utterances) == 2)
        ear.say()  # an utterance while asleep: the Vosk way, never sent
        assert ear.settle(lambda: len(ear.utterances) == 3)
        assert len(server.of("input_audio_buffer.commit")) == 1
        assert ear.utterances[2].realtime is None


def test_the_end_of_speech_to_transcript_latency_is_measured(ears):
    """SR-5 (SPEC 13): the HIL session reads end of speech to the server's transcript, p50/p95."""
    with TurnServer(heard=[["ciao Neve come stai"]]) as server:
        ear = ears(server, script=["vosk text"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        latency = ear.rt.route.snapshot()["latency_ms"]["transcript"]
        assert latency["count"] == 1 and latency["p50"] >= 0


def test_a_wake_word_in_the_servers_transcript_is_counted(ears):
    """SR-5 fix round 1, I3: the HIL reads whether the server heard the name ("Neve")."""
    with TurnServer(heard=[["ciao Neve come stai"], ["come stai"]]) as server:
        ear = ears(
            server,
            script=["vosk text"],
            bearing=FACE_RAD,
            app={"voice": {"wake_word_policy": "never", "wake_word": "Neve"}},
        )
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].wake_word
        assert ear.r.listener.snapshot()["realtime_audio"]["wake_word_heard"] == 1
