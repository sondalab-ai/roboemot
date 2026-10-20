"""The slice of the ``reachy_mini`` SDK this package touches, as typed protocols (SPEC 4A, 12A).

Every member is read straight from the SDK source of the version the package pins (1.12, the
robot's); nothing is probed with ``hasattr``/``getattr``. Where a name is private to the SDK
(``_is_alive``, ``_last_status``) the SDK line it comes from is cited here, so an SDK upgrade
that moves it fails loudly instead of degrading silently. Paths are relative to ``reachy_mini/``
of the 1.12.0 wheel.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


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
    """``ReachyMini.media``, a ``MediaManager`` (``reachy_mini.py:194``, always present)."""

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

    def get_input_audio_samplerate(self) -> int: ...

    def get_output_audio_samplerate(self) -> int: ...

    def get_input_channels(self) -> int: ...

    def start_playing(self) -> None: ...

    def stop_playing(self) -> None: ...


class ClientPort(Protocol):
    """``ReachyMini.client``, a ``WSClient`` (``io/ws_client.py``)."""

    host: str
    port: int
    _is_alive: bool
    """False once the state stream stops (``io/ws_client.py:74``, refreshed at ``:126``)."""
    _last_status: Any
    """The last daemon status, None until one arrives (``io/ws_client.py:75``, set at ``:195``)."""

    def get_status(self, wait: bool = True, timeout: float = 5.0) -> Any:
        """``io/ws_client.py:228``; with ``wait=False`` it asserts ``_last_status`` is set
        (``:233``), hence the guard in the callers."""
        ...


class MiniPort(Protocol):
    """The ``ReachyMini`` instance the daemon hands to ``run``."""

    client: ClientPort

    @property
    def media(self) -> MediaPort: ...
