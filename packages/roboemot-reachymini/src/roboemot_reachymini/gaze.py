"""Reachy Mini gaze and frame definitions (SPEC 4A, 5A Frame).

One place for both, so the world model's frame and the attention gaze cannot drift. Head yaw
is a base-frame angle on this robot (the SDK reports the head pose in the base frame and the
manifest constrains ``head.yaw - body_yaw``), so the body yaw is not part of the frame spec.
"""

from roboemot.attention import GazeMap
from roboemot.world import FrameSpec

GAZE = GazeMap(head_yaw="head.yaw", head_pitch="head.pitch", body_yaw="body_yaw")
FRAME = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",), roll_dofs=("head.roll",))
