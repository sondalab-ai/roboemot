"""Optional capabilities are declared members of the core protocols, not probed (SPEC 4A, 12A)."""

from __future__ import annotations

import inspect

from roboemot.vision.runner import FrameSource, LockedSource
from roboemot.voice.asr import ASRBackend, VoskASR
from roboemot.world.sources import WorldSource


def test_protocols_declare_their_optional_members() -> None:
    assert "camera_model" in dir(FrameSource)
    assert "missing_words" in dir(ASRBackend)
    assert "speaking" in WorldSource.__annotations__


def test_locked_source_delegates_camera_model_explicitly() -> None:
    class Camera:
        def read(self):
            return None

        def camera_model(self, width: int, height: int):
            return (width, height)

    assert LockedSource(Camera()).camera_model(8, 6) == (8, 6)
    assert "camera_model" in vars(LockedSource)


def test_vosk_asr_signature_matches_the_protocol() -> None:
    declared = inspect.signature(ASRBackend.missing_words)
    assert list(inspect.signature(VoskASR.missing_words).parameters) == list(declared.parameters)
