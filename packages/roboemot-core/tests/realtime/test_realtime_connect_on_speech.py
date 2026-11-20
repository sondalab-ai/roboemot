"""Connect on speech (SPEC 8, Realtime speech route and Gemini Live wire; owner decision
2026-11-20, "A + B + D"). The robot (2026-11-20): Gemini closed the idle session, the next
utterance started with no session, so Vosk ran, «metti un po' di musica» went to Gemini as the
Vosk text «metri semplicità», and over the radio «ferma la musica» was rejected locally
(«fermandosi») because the face rule needs a session.

A session the route reopens on demand (Gemini's idle close, a conversation's end on any
provider) is reopened when speech starts, so the utterance's audio goes once it is up and the
turn uses the server's transcript (A); a burst waits for a connect in flight within its
deadline (B); a connect that fails leaves the local path as before, nothing lost. No connect
starts from a refusal, a backoff, sleep, the opt-in off or text input, and one at a time. The
Live transcript says whose recognizer gave the words (D). Both fakes: the OpenAI realtime turn
server and the Gemini Live one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from gemini_server import HEARD
from roboemot.conversation import TranscriptRing
from rt_rig import build_rt, wait_for
from test_gemini_route import FACE_RADIO, gemini_route, radio_face_ear, text_turns
from test_realtime_face_media import LOOKING, LYRIC, app, on_radio
from test_realtime_listener import FACE_RAD, Ear
from test_voice_listener import silence, voice
from turn_server import TurnServer

OPENAI, GEMINI = "openai_realtime", "gemini_live"
PROVIDERS = (OPENAI, GEMINI)
OPT_IN = {"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}}
SAID = "ciao Neve come stai"  # what the OpenAI fake hears by default
IDLE_S = 1.2  # the Gemini fake closes a session idle that long (code 1008)
IDLE_AGE = {"idle_close_min_age_s": 1.0}  # ... and the route takes that as an idle close
SLOW_S = 1.0  # a slow connect: longer than an utterance's end silence takes to feed
WAIT_S = 10.0


@dataclass
class Closed:
    """One provider's fake server, rig and ear, with the session closed the way the route
    reopens only on demand: Gemini's idle close, or the OpenAI one's conversation end."""

    provider: str
    server: Any
    rig: Any
    ear: Any
    heard: str
    stops: list[str]

    @property
    def route(self):
        return self.rig.route

    def connections(self) -> int:
        srv = self.server
        return srv.connections if self.provider == GEMINI else len(srv.connections)

    def slow(self, seconds: float) -> None:
        if self.provider == GEMINI:
            self.server.setup_delay_s = seconds
        else:
            self.server.connect_delay_s = seconds

    def refuse(self, n: int) -> None:
        if self.provider == GEMINI:
            self.server.refuse = n
        else:
            self.server.refuse_connects = n

    def speak(self, seconds: float = 1.0) -> None:
        """Speech that starts, with no end yet."""
        self.ear.r.feed(silence(0.3))
        self.ear.r.feed(voice(seconds=seconds))

    def end(self) -> None:
        self.ear.r.feed(silence(self.ear.r.cfg.voice.vad.end_silence_ms / 1000 + 0.2))

    def counts(self) -> dict[str, int]:
        snap = self.route.snapshot()
        keys = ("connects_on_speech", "audio_connect_wait", "audio_connect_fallbacks")
        return {k: snap[k] for k in keys}


@pytest.fixture
def closed(gemini):
    made: list[Closed] = []

    def make(provider, *, heard=None, radio=False, rt=None, app_cfg=None, **ear_kw):
        rt = {"input": "audio", **IDLE_AGE, **(rt or {})}
        stops: list[str] = []
        if provider == GEMINI:
            said = heard or HEARD
            srv = gemini([["audio"]], heard=[said], idle_close_s=IDLE_S)
            cfg = app_cfg or (FACE_RADIO if radio else OPT_IN)
            rig = build_rt(srv, route=gemini_route(srv), rt=rt, app=cfg)
            if radio:
                ear, stops = radio_face_ear(rig)
            else:
                ear = Ear(rig, script=["vosk text"], bearing=FACE_RAD, to_worker=True, **ear_kw)
            assert wait_for(lambda: rig.route.snapshot()["idle_closes"] == 1, WAIT_S)
            srv.idle_close_s = None  # the next session stays up
        else:
            said = heard or SAID
            srv = TurnServer(heard=[[said]]).__enter__()
            cfg = {**OPT_IN, **(app_cfg or {})}
            if radio:
                cfg = {"privacy": {"realtime_audio": True}, **app()}
            rig = build_rt(srv, rt=rt, app=cfg)
            if radio:
                ear = Ear(rig, script=[LYRIC], bearing=FACE_RAD, yaw=LOOKING, to_worker=True)
                stops = on_radio(ear)["stop"]
            else:
                ear = Ear(rig, script=["vosk text"], bearing=FACE_RAD, to_worker=True, **ear_kw)
            rig.route.close_soon("conversation_end")  # the conversation ended: closed
        assert wait_for(lambda: not rig.route.connected and rig.route.state == "closed", WAIT_S)
        c = Closed(provider, srv, rig, ear, said, stops)
        made.append(c)
        return c

    yield make
    for c in made:
        c.ear.close()
        c.rig.close()
        if c.provider == OPENAI:
            c.server.__exit__(None, None, None)


def live(rig, **webui):
    """The Live transcript on, so the worker records the user's words and their source."""
    rig.stack.set("webui.live.enabled", True, source="test", admin=True)
    rig.stack.set("webui.live.transcript", True, source="test", admin=True)
    rig.worker.transcript = TranscriptRing(rig.stack.effective().webui.live)


def user_turns(rig):
    return [
        (t["text"], t["source"]) for t in rig.worker.transcript.snapshot()["turns"]
        if t["who"] == "user"
    ]  # fmt: skip


def no_text_item(c: Closed) -> bool:
    """The turn sent the utterance's audio, never its words as a text item."""
    if c.provider == GEMINI:
        words = [p.get("text", "") for t in text_turns(c.server) for x in t["turns"]
                 for p in x["parts"]]  # fmt: skip
        return not any(c.heard in w or "vosk text" in w for w in words)
    items = [e["item"] for e in c.server.of("conversation.item.create")]
    return not [i for i in items if i.get("type") == "message"]


# ------------------------------------------------------------------ A: connect at speech start


@pytest.mark.parametrize("provider", PROVIDERS)
def test_speech_after_an_on_demand_close_connects_at_its_start_and_its_audio_goes(closed, provider):
    c = closed(provider)
    live(c.rig)
    before = c.connections()
    c.speak()
    assert c.route.snapshot()["connects_on_speech"] == 1  # at the speech start, not its end
    assert wait_for(lambda: c.route.connected, WAIT_S)  # up before the utterance ends
    c.end()
    assert c.ear.settle(lambda: len(c.ear.utterances) == 1, WAIT_S)
    u = c.ear.utterances[0]
    assert u.addressed and u.text == c.heard  # the server's transcript, not Vosk's
    assert u.realtime is not None and u.realtime[0] == c.route.generation
    assert c.ear.r.asr.started == [] and c.ear.r.asr.decodes == 0  # Vosk skipped
    assert c.connections() == before + 1
    assert c.ear.settle(lambda: len(c.rig.events) == 1, WAIT_S)
    assert c.rig.events[-1]["outcome"] == "ok" and no_text_item(c)
    assert c.counts() == {
        "connects_on_speech": 1, "audio_connect_wait": 0, "audio_connect_fallbacks": 0
    }  # fmt: skip
    assert user_turns(c.rig) == [(c.heard, "cloud")]  # D: the cloud's words


# ------------------------------------------------------------------ B: wait for the connect


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_connect_slower_than_the_utterance_is_waited_for_and_the_audio_goes(closed, provider):
    c = closed(provider)
    c.slow(SLOW_S)
    c.speak()
    c.end()  # the utterance is over before the session is up
    assert not c.route.connected and c.route.opening()
    assert c.ear.settle(lambda: len(c.ear.utterances) == 1, WAIT_S)
    u = c.ear.utterances[0]
    assert u.text == c.heard and u.realtime is not None and not u.local_only
    assert c.ear.r.asr.started == [] and c.ear.r.asr.decodes == 0
    assert c.counts() == {
        "connects_on_speech": 1, "audio_connect_wait": 1, "audio_connect_fallbacks": 0
    }  # fmt: skip


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_connect_that_fails_leaves_the_local_path_and_loses_nothing(closed, provider):
    c = closed(provider, decoded="ciao Neve accendi la luce")
    live(c.rig)
    c.refuse(100)  # every connect fails from now on (the backoff then holds the retries)
    if provider == OPENAI:
        c.slow(SLOW_S)  # it fails after a while: the burst waits for it first
    c.speak()
    c.end()
    assert c.ear.settle(lambda: len(c.ear.utterances) == 1, WAIT_S)
    u = c.ear.utterances[0]
    assert u.addressed and u.text == "ciao Neve accendi la luce"  # decoded from its buffer
    assert u.realtime is None and not u.local_only  # nothing left: the chat routes answer
    snap = c.route.snapshot()
    assert snap["connects_on_speech"] == 1 and snap["connect_failures"] >= 1
    assert snap["audio_connect_fallbacks"] == snap["audio_connect_wait"]
    if provider == OPENAI:  # it waited for the slow failing connect
        assert snap["audio_connect_wait"] == 1
    assert c.ear.r.listener.snapshot()["realtime_audio"]["local_fallbacks"] == 1
    assert c.ear.settle(lambda: len(c.rig.events) == 1, WAIT_S)
    assert user_turns(c.rig) == [(u.text, "robot")]  # D: the robot's own words


def test_a_connect_that_comes_after_the_deadline_decodes_locally(closed):
    c = closed(OPENAI, decoded="ciao Neve accendi la luce")
    c.slow(WAIT_S)  # never up in time
    c.speak()
    c.end()
    assert wait_for(lambda: c.route.snapshot()["audio_connect_wait"] == 1, WAIT_S)
    c.ear.r.now += c.rig.stack.effective().voice.realtime_route.transcript_timeout_s
    assert c.ear.settle(lambda: len(c.ear.utterances) == 1, WAIT_S)
    u = c.ear.utterances[0]
    assert u.text == "ciao Neve accendi la luce" and u.realtime is None and not u.local_only
    assert wait_for(lambda: c.route.snapshot()["audio_connect_fallbacks"] == 1, WAIT_S)


# ------------------------------------------------------------------ no connect from these states


def _asleep(c, flag):
    flag[0] = True
    c.route.close_soon("asleep")


def _opt_in_off(c, flag):
    c.rig.stack.set("privacy.realtime_audio", False, source="test", admin=True)


def _text_input(c, flag):
    c.rig.stack.set("voice.realtime_route.input", "text", source="test", admin=True)


def _refused(c, flag):
    c.refuse(100)
    assert not c.route.ready()  # the connect failed: a refusal, retried with the backoff
    assert c.route.state == "fallback_local"


def _backoff(c, flag):
    c.route._reopen_at = c.route.clock() + 60.0  # a reset streak's backoff holds the reopen


@pytest.mark.parametrize(
    "state",
    [_asleep, _opt_in_off, _text_input, _refused, _backoff],
    ids=["asleep", "opt_in_off", "text_input", "refused", "backoff"],
)
def test_no_connect_starts_from_a_state_the_route_must_not_reopen_from(closed, state):
    flag = [False]
    c = closed(OPENAI, asleep=lambda: flag[0])
    state(c, flag)
    assert not c.route.connect_on_speech()
    c.speak()
    c.end()
    assert c.ear.settle(lambda: len(c.ear.utterances) == 1, WAIT_S)
    assert c.route.snapshot()["connects_on_speech"] == 0
    assert c.ear.utterances[0].realtime is None and c.server.of("input_audio_buffer.append") == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_repeated_speech_starts_trigger_one_connect(closed, provider):
    c = closed(provider)
    c.slow(SLOW_S)
    before = c.connections()
    c.speak(0.5)
    c.end()
    c.speak(0.5)  # a second utterance starts while the connect is in flight
    assert not c.ear.input.speech_started(c.rig.stack.effective())  # none stacks on it
    c.end()
    assert c.ear.settle(lambda: len(c.ear.utterances) == 2, WAIT_S)
    assert c.route.snapshot()["connects_on_speech"] == 1
    assert c.connections() == before + 1
    assert c.ear.utterances[0].realtime is not None  # the first waited for the connect


# ------------------------------------------------------------------ the face rule over the radio


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_face_rule_over_the_radio_applies_after_an_on_demand_close(closed, provider):
    """The robot's case: «ferma la musica» from a face over the radio after Gemini's idle
    close. The connect starts at the speech start, so the face rule skips Vosk and the server's
    words stop the radio."""
    c = closed(provider, heard="Ferma la musica.", radio=True)
    c.ear.say()
    assert c.ear.settle(lambda: len(c.ear.utterances) == 1, WAIT_S)
    assert not c.ear.utterances[0].addressed and c.stops == ["voice_stop"]
    audio = c.ear.r.listener.snapshot()["realtime_audio"]
    assert audio["face_wake_media"] == 1 and audio["vosk_skipped"] == 1
    assert c.ear.r.asr.started == [] and c.ear.r.asr.decodes == 0
    assert c.route.snapshot()["connects_on_speech"] == 1


def test_a_face_wake_whose_connect_fails_is_decoded_for_the_wake_word(closed):
    c = closed(OPENAI, heard="Ferma la musica.", radio=True)
    c.ear.r.asr.decoded = "Neve ferma la musica"
    c.refuse(100)
    c.slow(SLOW_S)
    c.ear.say()
    assert c.ear.settle(lambda: len(c.ear.utterances) == 1, WAIT_S)
    u = c.ear.utterances[0]
    assert u.addressed and u.text == "Neve ferma la musica" and u.realtime is None
    audio = c.ear.r.listener.snapshot()["realtime_audio"]
    assert audio["face_wake_media"] == 1 and audio["face_wake_unsent"] == 1
    assert c.route.snapshot()["audio_connect_fallbacks"] == 1
