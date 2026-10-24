"""Owner decision 2026-10-23 (SPEC 5, 7, 8, 10, 12): one behaviour for everyone.

A face whose age is unknown (side-on, or a face-like object at a fixed spot) once kept the robot
in minor mode all day: damped motion, no turn toward a voice, child lines instead of answers.
Now such a face, or a child's, changes nothing. Here, the register: the one register with
anybody in view. The motion amplitude, the turn toward a voice, the spoken lines and the TTS
backend are checked through the real app with such a face in view
(``roboemot-reachymini/tests/test_app_one_behaviour.py``).
"""

from __future__ import annotations

import dataclasses

import pytest
from roboemot.affect.register import RegisterTracker
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from test_reflex import face as reflex_face

CFG = RoboEmotConfig()
AGES = [("unknown", "none"), ("child", "declared"), ("teen", "estimated")]


def face(stamp: float, group: str, source: str, height: float = 0.1):
    """A small face (no close-face withdraw) of the given age group."""
    return dataclasses.replace(reflex_face(stamp, height), age_group=group, age_source=source)


@pytest.mark.parametrize(("group", "source"), AGES)
def test_the_register_is_the_one_register_with_any_face_in_view(group, source):
    bus = Bus()
    tracker = RegisterTracker(bus, CFG.expression)
    bus.publish("percept.face", face(0.0, group, source), 0.0)
    got = tracker.tick(0.1, CFG.register, CFG.persona)
    assert got.people == 1 and got.params == CFG.register.params
    assert got.params.approach_allowed and got.params.motion_amplitude == 1.0
