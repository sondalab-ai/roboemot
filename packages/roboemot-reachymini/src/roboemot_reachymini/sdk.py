"""The slice of the ``reachy_mini`` SDK this package touches, as typed protocols (SPEC 4A, 12A).

Every member is read straight from the SDK source of the version the package is tested against
(1.12, the robot's; the package requires at least that, ``MIN_SDK``, and the simulation CI job
installs 1.12.x); nothing is probed with ``hasattr``/``getattr``. Where a name is private to the
SDK (``_is_alive``, ``_last_status``) the SDK line it comes from is cited here, so an SDK upgrade
that moves it fails loudly instead of degrading silently; a newer SDK than ``TESTED_SDK`` is
logged at start (``newer_than_tested``). Paths are relative to ``reachy_mini/`` of the 1.12.0
wheel.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

MIN_SDK = "1.12"
"""Oldest SDK the package is written for: ``AudioBase.clear_player`` (barge-in) and the private
websocket state below are read as in 1.12.0, the version of the daemon and SDK on the robot."""

TESTED_SDK = "1.12"
"""Newest SDK major.minor the package was checked against (the sim CI job pins it). A newer one
may have moved a private name read here: the app warns at start but runs (a daemon update must
not stop it)."""


def _numbers(text: str) -> tuple[int, ...]:
    """The leading numeric release segments of a version string (``1.12.0rc1`` -> (1, 12, 0))."""
    out: list[int] = []
    for part in text.split("."):
        match = re.match(r"\d+", part)
        if match is None:
            break
        out.append(int(match.group()))
        if match.end() != len(part):
            break
    return tuple(out)


def installed_version() -> str | None:
    """The installed ``reachy_mini`` distribution version, None when it is not installed."""
    try:
        return version("reachy_mini")
    except PackageNotFoundError:
        return None


def require_sdk(installed: str | None) -> None:
    """Raise ``RuntimeError`` unless ``installed`` is at least ``MIN_SDK``: an older SDK would
    lose barge-in (no ``clear_player``) without any sign, so the app refuses to start."""
    if installed is None:
        raise RuntimeError(f"reachy_mini is not installed; this app needs {MIN_SDK} or newer")
    if _numbers(installed) < _numbers(MIN_SDK):
        raise RuntimeError(
            f"reachy_mini {installed} is older than {MIN_SDK}, which this app needs "
            "(AudioBase.clear_player for barge-in, the websocket status fields); "
            "update the robot's Reachy Mini software"
        )


def newer_than_tested(installed: str | None) -> bool:
    """Whether ``installed`` has a higher major.minor than ``TESTED_SDK``."""
    return installed is not None and _numbers(installed)[:2] > _numbers(TESTED_SDK)[:2]


class AudioPort(Protocol):
    """``media.audio`` (``GStreamerAudio`` or ``GstWebRTCClient``, both ``AudioBase``)."""

    def clear_player(self) -> None:
        """Drop queued playback audio now (``media/audio_base.py:319``, abstract there, so every
        audio backend has it)."""
        ...


class CameraPort(Protocol):
    """``media.camera`` (``CameraBase``)."""

    @property
    def K(self) -> npt.NDArray[np.float64] | None:  # noqa: N802
        """Intrinsics rescaled to the streamed size, None until a resolution is set
        (``media/camera_base.py:90-96``)."""
        ...


class MediaPort(Protocol):
    """``ReachyMini.media``, a ``MediaManager`` (``reachy_mini.py:194``, always present).

    Methods, all in ``media/media_manager.py``: ``get_frame`` :243, ``start_recording`` :280,
    ``get_audio_sample`` :287, ``get_input_audio_samplerate`` :299,
    ``get_output_audio_samplerate`` :306, ``get_input_channels`` :313, ``stop_recording`` :327,
    ``start_playing`` :334, ``push_audio_sample`` :341, ``stop_playing`` :380, ``get_DoA`` :418.
    """

    @property
    def audio(self) -> AudioPort | None:
        """None when the robot has no audio system (``media/media_manager.py:156``)."""
        ...

    @property
    def camera(self) -> CameraPort | None:
        """None when the robot has no camera (``media/media_manager.py:155``)."""
        ...

    def get_frame(self) -> npt.NDArray[np.uint8] | None: ...

    def start_recording(self) -> None: ...

    def get_audio_sample(self) -> npt.NDArray[np.float32] | None: ...

    def stop_recording(self) -> None: ...

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None: ...

    def get_DoA(self) -> tuple[float, bool] | None:  # noqa: N802
        """Direction of arrival in radians and whether speech is detected, None without it
        (``media/media_manager.py:418``)."""
        ...

    def get_input_audio_samplerate(self) -> int: ...

    def get_output_audio_samplerate(self) -> int: ...

    def get_input_channels(self) -> int: ...

    def start_playing(self) -> None: ...

    def stop_playing(self) -> None: ...


class ClientPort(Protocol):
    """``ReachyMini.client``, a ``WSClient`` (``io/ws_client.py``)."""

    host: str
    port: int

    @property
    def _is_alive(self) -> bool:
        """False once the state stream stops (``io/ws_client.py:74``, refreshed at ``:126``)."""
        ...

    _last_status: Any
    """The last daemon status, None until one arrives (``io/ws_client.py:75``, set at ``:195``)."""

    def get_status(self, wait: bool = True, timeout: float = 5.0) -> Any:
        """``io/ws_client.py:228``; with ``wait=False`` it asserts ``_last_status`` is set
        (``:233``), hence the guard in the callers."""
        ...

    _last_head_pose: Any
    """The last ``HeadPoseMsg``; a new object per message (``io/ws_client.py:70``, ``:189``)."""

    _last_joint_positions: Any
    """The last ``JointPositionsMsg``; a new object per message (``:69``, set at ``:186``)."""


class MiniPort(Protocol):
    """The ``ReachyMini`` instance the daemon hands to ``run``."""

    client: ClientPort

    @property
    def media(self) -> MediaPort: ...
