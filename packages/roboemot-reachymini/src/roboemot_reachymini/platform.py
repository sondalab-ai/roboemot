"""Entry point ``roboemot.robots``: the Reachy Mini platform (SPEC 4A, 11, 12A).

Builds the robot's parts for core from the connected ``ReachyMini`` the daemon hands to the app.
The SPEC 4A rules stay in the parts it builds (``set_target`` only, ``automatic_body_yaw=False``,
motors enabled before targets, antenna order ``[right, left]``, one process owns the SDK client).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from roboemot.attention import GazeMap
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import PoseGroupDriver
from roboemot.reflex import ReflexMap
from roboemot.vision.runner import FrameSource
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import PlaybackReference
from roboemot.world import FrameSpec
from roboemot.world.sources import JointStateFn

from .audio import ReachyAudio
from .camera import ReachyCamera
from .driver import ReachyMiniDriver
from .gaze import FRAME, GAZE, REFLEX
from .sdk import installed_version, require_sdk
from .speaker import ReachySpeaker
from .voice import LatestBearing

MS_PER_S = 1000.0


class ReachyMiniSession:
    """``RobotSession`` over the connected ``ReachyMini`` the daemon passed to the app. ``client``
    is public for the Reachy world plugin, which reads the microphone array through it."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def driver(self) -> PoseGroupDriver:
        """The pose driver over the SDK client."""
        return ReachyMiniDriver(self.client)

    def camera(self, joint_state: JointStateFn) -> FrameSource:
        """The SDK media camera, each frame paired with the measured joint state."""
        return ReachyCamera(self.client, joint_state)

    def audio(
        self, cfg: RoboEmotConfig, on_level: Callable[[float, float], None] | None
    ) -> ReachyAudio:
        """The microphone hub; the voice queue is on when ``voice.enabled``."""
        return ReachyAudio(
            self.client,
            on_level,
            cfg.reflexes.audio.poll_hz,
            max_gap_s=cfg.reflexes.max_age_ms / MS_PER_S,
            voice=cfg.voice.enabled,
            voice_rate_hz=cfg.voice.sample_rate_hz,
            voice_queue_s=cfg.voice.capture_queue_s,
            level_block_s=cfg.reflexes.audio.level_block_s,
            level_window=cfg.reflexes.audio.level_window,
        )

    def speaker(
        self, cfg: RoboEmotConfig, activity: SpeechActivity, reference: PlaybackReference | None
    ) -> ReachySpeaker:
        """The SDK media player, chunked at ``voice.playback_chunk_ms``."""
        return ReachySpeaker(
            self.client, cfg.voice.playback_chunk_ms, activity, reference=reference
        )

    def bearing_source(self, bus: Bus) -> LatestBearing:
        """The DoA source's newest bearing, rotated into the base frame like the world model."""
        return LatestBearing(bus, FRAME)


class ReachyMiniPlatform:
    """``RobotPlatform`` of ``reachy-mini-wireless``: stateless description of the robot."""

    def require_runtime(self) -> None:
        """Refuse to start without a ``reachy_mini`` SDK of the supported version."""
        require_sdk(installed_version())

    def bind(self, client: Any) -> ReachyMiniSession:
        """The session over the connected ``ReachyMini``."""
        return ReachyMiniSession(client)

    def gaze_map(self) -> GazeMap:
        """Head yaw and pitch aim the gaze; the body turns with ``body_yaw``."""
        return GAZE

    def frame_spec(self) -> FrameSpec:
        """The head pose in the base frame gives the camera's pointing direction."""
        return FRAME

    def reflex_map(self) -> ReflexMap:
        """Antenna startle, head pitch away, and the DoA sound source."""
        return REFLEX
