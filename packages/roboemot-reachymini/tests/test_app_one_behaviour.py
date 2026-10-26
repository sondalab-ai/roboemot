"""Owner decision 2026-10-23 (SPEC 5, 7, 8, 10, 12), through the real app: a face is in view (no
age is estimated at all since the age model went), and the robot behaves as with anybody. Each
case failed while the adult/minor split existed: the possible-minor latch capped the motion, held
the turn toward a voice, swapped the fallback line for a child line and kept every sentence off
the remote voice.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from roboemot.config.schema import RegisterParams, StateEffects
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionLoop, Priority
from roboemot.voice.speech import SpeechOutput
from roboemot.voice.tts import AudioChunk, FallbackTTS
from roboemot.world import Observation
from roboemot_reachymini import app as app_module
from roboemot_reachymini.driver import ReachyMiniDriver
from test_app_conversation import utterance
from test_app_perception import (  # noqa: F401  (data is a fixture)
    AppWithFakes,
    FakeDetector,
    data,
    new_app,
    start,
    wait_for,
)
from test_app_reflex import ScriptedMini, enable, run_app
from test_app_voice import VOICE_YAML, fakes  # noqa: F401  (fakes is a fixture)

HOLD = 0.5  # rad away from home commanded on an antenna
FALLBACK_IT = "Scusa, adesso non riesco a pensare."


class VoiceApp(AppWithFakes):
    """The fake camera sees one face of unknown age; voice is on."""


@pytest.fixture()
def loops(monkeypatch):
    """Every motion loop the app builds."""
    made: list[MotionLoop] = []
    real = MotionLoop.__init__

    def init(self, *a, **k):
        real(self, *a, **k)
        made.append(self)

    monkeypatch.setattr(MotionLoop, "__init__", init)
    return made


@pytest.fixture()
def commanded(monkeypatch):
    """The latest pose the motion loop wrote to the driver (after the arbiter and limiter)."""
    last: dict[str, float] = {}
    real = ReachyMiniDriver.write_pose_targets

    def write(self, t, values):
        last.update(values)
        real(self, t, values)

    monkeypatch.setattr(ReachyMiniDriver, "write_pose_targets", write)
    return last


def face_in_view(app) -> bool:
    tracker = app.register_tracker
    return tracker is not None and tracker.active is not None and tracker.active.people > 0


def test_the_motion_reaches_its_full_amplitude_with_an_unknown_age_face_in_view(
    data,  # noqa: F811
    loops,
    commanded,
):
    home = (
        load_manifest(app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)
        .dofs["antenna_right"]
        .home
    )
    FakeDetector.empty = False
    app = new_app(AppWithFakes)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: loops and face_in_view(app), 30), "no face in view"
        loop = loops[0]

        def hold() -> bool:
            loop.feed(
                Command(
                    "test", Priority.COMMANDED, {"antenna_right": home + HOLD}, time.monotonic()
                )
            )
            return commanded.get("antenna_right", home) - home > 0.95 * HOLD

        assert wait_for(hold, 15), "the antenna never reached the commanded offset"
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_the_robot_turns_toward_a_voice_with_an_unknown_age_face_in_view(data):  # noqa: F811
    FakeDetector.empty = False
    app = new_app(AppWithFakes)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.reflexes is not None and face_in_view(app), 30)

        def voice() -> bool:
            now = time.monotonic()
            obs = Observation("unknown", now, "doa", 0.4, 0.0, confidence=0.5)
            app.bus.publish("percept.entity", obs, now, joint_state={"head.yaw": 0.0})
            return app.reflexes.snapshot()["rules"]["turn-to-voice"]["fired"] > 0

        assert wait_for(voice, 15), "the turn toward the voice never fired"
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_a_turn_with_no_route_gets_the_normal_line_with_an_unknown_age_face_in_view(
    data,  # noqa: F811
    fakes,  # noqa: F811
    monkeypatch,
):
    said: list[str] = []
    monkeypatch.setattr(SpeechOutput, "say", lambda self, text, **kw: said.append(text) or 1)
    enable(data, VOICE_YAML)  # no llm_routes: no route can answer
    FakeDetector.empty = False
    app = new_app(VoiceApp)
    stop, thread, errors = run_app(app, ScriptedMini().mini)
    try:
        assert wait_for(lambda: app.conversation is not None and face_in_view(app), 30)
        app.voice_listener._on_utterance(utterance())
        assert wait_for(lambda: said, 15), "nothing was said"
        assert said == [FALLBACK_IT]
    finally:
        stop.set()
        thread.join(15)
    assert not errors


class Recorder:
    """A TTS backend stand-in that records what it was asked to say."""

    controls = frozenset()

    def __init__(self, name: str) -> None:
        self.name, self.texts = name, []

    def available(self, **_kw) -> bool:
        return True

    async def synthesize(self, text, delivery, *, language, voice, **_kw):
        self.texts.append(text)
        yield AudioChunk(np.full(1600, 0.1, np.float32), 16000)

    def cancel(self):
        pass

    async def warm(self, language, voice):
        pass

    def apply_config(self, voice):
        pass

    async def aclose(self):
        pass


def test_the_remote_voice_speaks_with_an_unknown_age_face_in_view(data, fakes):  # noqa: F811
    enable(data, VOICE_YAML)
    FakeDetector.empty = False
    app = new_app(VoiceApp)
    remote, local = Recorder("remote"), Recorder("piper")
    stop, thread, errors = run_app(app, ScriptedMini().mini)
    try:
        assert wait_for(lambda: app.speech_output is not None and face_in_view(app), 30)
        app.speech_output.backend = FallbackTTS(remote, local)
        app.speech_output.say(
            "Ciao.",
            language="it",
            register=RegisterParams(),
            circadian=StateEffects(),
            mood=None,
            intensity=0.0,
        )
        assert wait_for(lambda: remote.texts, 10), "the remote voice was never asked"
        assert remote.texts == ["Ciao."] and local.texts == []
    finally:
        stop.set()
        thread.join(15)
    assert not errors
