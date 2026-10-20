"""The robot platform: what core needs a robot package to build (SPEC 4A, 11, 12A).

A robot package registers one class in the ``roboemot.robots`` entry-point group, keyed by the
manifest ``robot.id`` (same pattern as ``roboemot.world`` and ``roboemot.settings``). Core loads it
with :func:`load_robot_platform`, which fails loud: no entry point for the id, no robot, no run.
The platform is stateless and describes the robot (maps, runtime check); ``bind(client)`` returns
the per-run :class:`RobotSession`, which owns the handle the robot's own launcher passed to the app
(the connected robot) and builds every part that needs it. Core never sees the handle's type. Every
factory returns a core protocol, so the runtime builds the whole pipeline without importing the
robot package.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from roboemot.attention import GazeMap
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import PoseGroupDriver
from roboemot.reflex import ReflexMap
from roboemot.vision.runner import FrameSource
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import PlaybackReference
from roboemot.voice.listener import VoiceChunkLike
from roboemot.world import FrameSpec
from roboemot.world.sources import JointStateFn

ENTRY_POINT_GROUP = "roboemot.robots"


class RobotPlatformError(RuntimeError):
    """No usable platform for the manifest's robot id (nothing is run without one)."""


@runtime_checkable
class AudioHub(Protocol):
    """The one reader of the microphone: feeds the reflex level sink and the voice queue.

    ``start`` and ``stop`` bracket the capture thread (``start`` degrades to an ``unavailable``
    snapshot when there is no microphone, never raises); ``read`` drains the voice queue for the
    listener (None on timeout); ``set_voice`` turns that queue off before ``start`` when no
    listener exists; ``snapshot`` is the debug-report state (counts and states, never samples).
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_voice(self, enabled: bool) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...
    def read(self, timeout: float = 0.0) -> VoiceChunkLike | None: ...


@runtime_checkable
class Speaker(Protocol):
    """Playback of the robot's voice. ``push_audio_sample`` and ``stop`` are the speech output's
    ``AudioSink`` (any sample rate, the speaker resamples; ``stop`` flushes what is queued).
    ``start_playing`` opens the output and is False when the robot has no usable audio output;
    ``close`` flushes and releases it before the robot link goes away."""

    def push_audio_sample(self, samples: Any, rate_hz: int) -> float: ...
    def stop(self) -> None: ...
    def start_playing(self) -> bool: ...
    def close(self) -> None: ...


@runtime_checkable
class BearingSource(Protocol):
    """The newest sound direction in the base frame (the robot's own source, e.g. a microphone
    array's direction of arrival). ``get`` is the azimuth if the newest reading is at most
    ``max_age_s`` old, else None; ``close`` stops listening to whatever feeds it."""

    def get(self, now: float, max_age_s: float) -> float | None: ...
    def close(self) -> None: ...


@runtime_checkable
class RobotSession(Protocol):
    """One run of the app on one robot: the factories that need the launcher's robot handle."""

    def driver(self) -> PoseGroupDriver:
        """The motion driver (core's motion loop owns and calls it)."""
        ...

    def camera(self, joint_state: JointStateFn) -> FrameSource:
        """The camera the vision pump reads; each frame is paired with ``joint_state`` at
        capture so core can place what it sees in the base frame."""
        ...

    def audio(
        self, cfg: RoboEmotConfig, on_level: Callable[[float, float], None] | None
    ) -> AudioHub:
        """The microphone hub, configured from ``cfg`` (reflexes and voice sections).
        ``on_level(rms, stamp)`` is the reflex engine's audio-level sink, None when off."""
        ...

    def speaker(
        self, cfg: RoboEmotConfig, activity: SpeechActivity, reference: PlaybackReference | None
    ) -> Speaker:
        """The playback device. It marks ``activity`` while it plays and records what it plays
        in ``reference`` (the barge-in echo reference). Only built when ``voice.enabled``."""
        ...

    def bearing_source(self, bus: Bus) -> BearingSource:
        """The sound direction in the base frame, read from the observations the robot's world
        sources publish on ``bus``. Core's voice listener uses it to hint who is speaking."""
        ...


@runtime_checkable
class RobotPlatform(Protocol):
    """What a robot package provides to core, keyed by manifest ``robot.id``."""

    def require_runtime(self) -> None:
        """Raise ``RuntimeError`` with the remedy when what the robot needs at run time (its SDK
        and version) is missing. Called once at app start, before anything is built."""
        ...

    def bind(self, client: Any) -> RobotSession:
        """The session over ``client``, the handle the robot's launcher passed to the app."""
        ...

    def gaze_map(self) -> GazeMap:
        """Manifest dof names that aim the gaze."""
        ...

    def frame_spec(self) -> FrameSpec:
        """How the world model turns joint state into the camera's pointing direction."""
        ...

    def reflex_map(self) -> ReflexMap:
        """Manifest dofs the reflex primitives move, and the robot's voice-direction sources."""
        ...


def installed_robots() -> list[str]:
    """Robot ids that have a ``roboemot.robots`` entry point, sorted."""
    return sorted(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))


def load_robot_platform(robot_id: str) -> RobotPlatform:
    """The platform registered for ``robot_id``, instantiated. Fails loud (``RobotPlatformError``)
    when there is none, when two distributions register the same id, when it cannot be loaded
    (a missing SDK gets an install hint) or when it does not satisfy :class:`RobotPlatform`:
    there is no fallback and the app does not run."""
    matches = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == robot_id]
    if not matches:
        installed = ", ".join(installed_robots()) or "none"
        raise RobotPlatformError(
            f"no robot platform installed for robot id {robot_id!r} (installed: {installed}); "
            "install the robot's package"
        )
    if len(matches) > 1:
        dists = ", ".join(sorted(ep.dist.name if ep.dist else "unknown" for ep in matches))
        raise RobotPlatformError(
            f"robot id {robot_id!r} is registered by more than one package ({dists}); "
            "uninstall all but one"
        )
    try:
        platform = matches[0].load()()
    except ImportError as exc:
        raise RobotPlatformError(
            f"robot platform {robot_id!r} cannot be loaded: {type(exc).__name__}: {exc}; "
            "install the robot package's SDK extra"
        ) from exc
    except Exception as exc:
        raise RobotPlatformError(
            f"robot platform {robot_id!r} cannot be loaded: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(platform, RobotPlatform):
        raise RobotPlatformError(f"robot platform {robot_id!r} does not satisfy RobotPlatform")
    return platform
