"""Owner decision 2026-10-23 (SPEC 5, 7, 8, 10, 12): one behaviour for everyone.

A face whose age is unknown (side-on, or a face-like object at a fixed spot) once kept the robot
in minor mode all day: damped motion, no turn toward a voice, child lines instead of answers.
Now no age is estimated at all (2026-10-23), so no face changes anything. Here, the register:
the one register with anybody in view. The motion amplitude, the turn toward a voice, the spoken
lines and the TTS backend are checked through the real app with a face in view
(``roboemot-reachymini/tests/test_app_one_behaviour.py``).
"""

from __future__ import annotations

from roboemot.affect.register import RegisterTracker
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from test_reflex import face as reflex_face

CFG = RoboEmotConfig()


def test_the_register_is_the_one_register_with_any_face_in_view():
    bus = Bus()
    tracker = RegisterTracker(bus, CFG.expression)
    bus.publish("percept.face", reflex_face(0.0, 0.1), 0.0)  # small: no close-face withdraw
    got = tracker.tick(0.1, CFG.register, CFG.persona)
    assert got.people == 1 and got.params == CFG.register.params
    assert got.params.approach_allowed and got.params.motion_amplitude == 1.0
