"""Reachy Mini gaze and frame definitions (SPEC 4A, 5A Frame).

One place for both, so the world model's frame and the attention gaze cannot drift. Head yaw
is a base-frame angle on this robot (the SDK reports the head pose in the base frame and the
manifest constrains ``head.yaw - body_yaw``), so the body yaw is not part of the frame spec.
"""

from roboemot.attention import GazeMap
from roboemot.reflex import ReflexMap
from roboemot.world import FrameSpec

GAZE = GazeMap(head_yaw="head.yaw", head_pitch="head.pitch", body_yaw="body_yaw")
FRAME = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",), roll_dofs=("head.roll",))

# Reflexes (SPEC 10): a positive head pitch points the nose down, so tipping back is negative; an
# antenna lift toward vertical is right +, left - (SDK order [right, left], signs as in the
# expression map, not yet checked on the hardware); "doa" is the world plugin's sound source.
REFLEX = ReflexMap(
    head_yaw="head.yaw",
    head_pitch="head.pitch",
    body_yaw="body_yaw",
    startle_lift={"antenna_right": 1.0, "antenna_left": -1.0},
    pitch_away_sign=-1.0,
    voice_sources=("doa",),
)
