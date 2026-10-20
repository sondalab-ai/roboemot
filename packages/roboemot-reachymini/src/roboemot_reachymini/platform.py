"""Entry point ``roboemot.robots``: the Reachy Mini platform (SPEC 4A, 11, 12A).

Builds the robot's parts for core from the connected ``ReachyMini`` the daemon hands to the app.
The SPEC 4A rules stay in the parts it builds (``set_target`` only, ``automatic_body_yaw=False``,
motors enabled before targets, antenna order ``[right, left]``, one process owns the SDK client).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from roboemot.attention import AttentionSelector, GazeMap
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import PoseGroupDriver
from roboemot.reflex import ReflexMap
from roboemot.robot_platform import VoicePipeline
from roboemot.vision.runner import FrameSource
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import PlaybackReference
from roboemot.voice.fastpath import Emit
from roboemot.voice.listener import Utterance
from roboemot.voice.robot_name import RobotName
from roboemot.world import FrameSpec, WorldModel
from roboemot.world.sources import JointStateFn

from .audio import ReachyAudio
from .camera import ReachyCamera
from .driver import ReachyMiniDriver
from .gaze import FRAME, GAZE, REFLEX
from .sdk import installed_version, require_sdk
from .speaker import ReachySpeaker
from .voice import build_listener

MS_PER_S = 1000.0


class ReachyMiniPlatform:
    """``RobotPlatform`` of ``reachy-mini-wireless``; ``client`` is the connected ``ReachyMini``."""

    def require_runtime(self) -> None:
        """Refuse to start without a ``reachy_mini`` SDK of the supported version."""
        require_sdk(installed_version())

    def driver(self, client: Any) -> PoseGroupDriver:
        """The pose driver over the SDK client."""
        return ReachyMiniDriver(client)

    def camera(self, client: Any, joint_state: JointStateFn) -> FrameSource:
        """The SDK media camera, each frame paired with the measured joint state."""
        return ReachyCamera(client, joint_state)

    def audio(
        self, client: Any, cfg: RoboEmotConfig, on_level: Callable[[float, float], None] | None
    ) -> ReachyAudio:
        """The microphone hub; the voice queue is on when ``voice.enabled``."""
        return ReachyAudio(
            client,
            on_level,
            cfg.reflexes.audio.poll_hz,
            max_gap_s=cfg.reflexes.max_age_ms / MS_PER_S,
            voice=cfg.voice.enabled,
            voice_rate_hz=cfg.voice.sample_rate_hz,
            voice_queue_s=cfg.voice.capture_queue_s,
        )

    def speaker(
        self,
        client: Any,
        cfg: RoboEmotConfig,
        activity: SpeechActivity,
        reference: PlaybackReference | None,
    ) -> ReachySpeaker:
        """The SDK media player, chunked at ``voice.playback_chunk_ms``."""
        return ReachySpeaker(client, cfg.voice.playback_chunk_ms, activity, reference=reference)

    def gaze_map(self) -> GazeMap:
        """Head yaw and pitch aim the gaze; the body turns with ``body_yaw``."""
        return GAZE

    def frame_spec(self) -> FrameSpec:
        """The head pose in the base frame gives the camera's pointing direction."""
        return FRAME

    def reflex_map(self) -> ReflexMap:
        """Antenna startle, head pitch away, and the DoA sound source."""
        return REFLEX

    def listener(
        self,
        cfg: RoboEmotConfig,
        audio: ReachyAudio,
        speech: SpeechActivity,
        bus: Bus,
        emit: Emit,
        selector: AttentionSelector,
        world: WorldModel,
        registry: Any,
        on_utterance: Callable[[Utterance], None],
        reference: PlaybackReference | None,
        names: RobotName,
        on_answer: Callable[[str, float], None] | None,
    ) -> VoicePipeline | None:
        """The voice listener with the Reachy DoA bearing in its fast path (see ``voice``)."""
        return build_listener(
            cfg,
            audio,
            speech,
            bus,
            emit,
            selector,
            world,
            registry,
            self.frame_spec(),
            on_utterance,
            reference,
            names,
            on_answer,
        )
