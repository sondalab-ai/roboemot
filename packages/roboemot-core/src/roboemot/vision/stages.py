"""Stage interfaces of the vision pipeline (SPEC 5, 11 Plugin interfaces).

Each stage is a small protocol so backends (ONNX on CPU, Hailo, RKNN, mocks) are swappable.
Backends own all image preprocessing (resize, letterbox, alignment); the pipeline hands over
the original frame and boxes in original pixels and receives results in the same space.
Crops made inside a stage are never kept, logged or written to disk (SPEC 12).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Box, Detection, Frame


class FaceDetector(Protocol):
    """Finds faces: boxes, score and 5 landmarks in original pixels."""

    def detect(self, image: np.ndarray) -> list[Detection]: ...


class FaceEmbedder(Protocol):
    """Identity embedding of one detected face; ``model_id`` keys the registry gallery."""

    model_id: str

    def embed(self, image: np.ndarray, face: Detection) -> np.ndarray: ...


class AppearanceEmbedder(Protocol):
    """Appearance embedding of an animal or object box, for matching and the world model."""

    model_id: str

    def embed(self, image: np.ndarray, box: Box) -> np.ndarray: ...


class ObjectDetector(Protocol):
    """Labelled boxes of animals and objects; ``Detection.label`` is the detector class name."""

    def detect(self, image: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class PerceptionModule(Protocol):
    """A perception plugin: consumes frames, publishes percepts on the bus (SPEC 11)."""

    name: str

    def process(self, frame: Frame) -> object: ...
