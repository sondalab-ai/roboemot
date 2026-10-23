"""Owner decision 2026-10-23 (SPEC 5, 7, 8, 10, 12): one behaviour for everyone.

A face whose age is unknown (side-on, or a face-like object at a fixed spot) once kept the robot
in minor mode all day: damped motion, no turn toward a voice, child lines instead of answers.
Now such a face, or a child's, changes nothing: the motion amplitude, the reflexes, the spoken
lines and the TTS backend are the ones the robot uses with anybody.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from roboemot.affect.register import RegisterTracker
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from test_conversation_worker import LAN, rigs  # noqa: F401  (rigs is a fixture)
from test_reflex import PITCH, VOICE, Rig, cfg_with, doa, peak_yaw
from test_reflex import face as reflex_face
from test_tts import SpeechServer, collect, openai_backend

CFG = RoboEmotConfig()
AGES = [("unknown", "none"), ("child", "declared"), ("teen", "estimated")]


def face(stamp: float, group: str, source: str, height: float = 0.1):
    """A small face (no close-face withdraw) of the given age group."""
    return dataclasses.replace(reflex_face(stamp, height), age_group=group, age_source=source)


@pytest.mark.parametrize(("group", "source"), AGES)
def test_the_motion_is_the_same_with_any_face_in_view(group, source):
    def run(with_face: bool) -> list[float]:
        rig = Rig(cfg_with(), hold={PITCH: 0.3})
        for _ in range(50):
            if with_face:
                rig.bus.publish("percept.face", face(rig.t, group, source), rig.t)
            rig.tick()
        return rig.pitch

    alone, seen = run(False), run(True)
    assert seen == alone and max(seen) == pytest.approx(0.3, abs=1e-6)  # full amplitude


@pytest.mark.parametrize(("group", "source"), AGES)
def test_the_robot_turns_toward_a_voice_with_any_face_in_view(group, source):
    rig = Rig(cfg_with(VOICE))
    rig.tick(3)
    rig.bus.publish("percept.face", face(rig.t, group, source), rig.t)
    doa(rig, 0.4)
    rig.tick()
    rig.run(1.0)
    assert rig.stats("v")["fired"] == 1
    assert peak_yaw(rig) == pytest.approx(0.4, abs=0.01)


@pytest.mark.parametrize(("group", "source"), AGES)
def test_the_register_is_the_one_register_with_any_face_in_view(group, source):
    bus = Bus()
    tracker = RegisterTracker(bus, CFG.expression)
    bus.publish("percept.face", face(0.0, group, source), 0.0)
    got = tracker.tick(0.1, CFG.register, CFG.persona)
    assert got.people == 1 and got.params == CFG.register.params
    assert got.params.approach_allowed and got.params.motion_amplitude == 1.0


def test_the_lines_are_the_normal_ones(rigs):  # noqa: F811
    rig = rigs(routes=({**LAN, "daily_token_budget": 0},))
    rig.turn("ciao robot", language="en", keywords=("greeting", "wake"))
    assert rig.speech.texts() == ["Sorry, I can't think right now."]


async def test_any_tts_backend_may_speak_every_sentence():
    server = SpeechServer()
    remote, _ = openai_backend(server)
    assert remote.available() and list(inspect.signature(remote.available).parameters) == []
    await collect(remote, "Ciao")
    assert len(server.requests) == 1
