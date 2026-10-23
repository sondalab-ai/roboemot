"""Vision: detect, track, embed, match and publish percepts (SPEC 5, 5A, 6, 13).

Importing this package needs only numpy: the ONNX backends (``roboemot.vision.onnx``) need
the optional ``vision`` extra and are imported explicitly.
"""

from .attributes import AttributeResult, AttributeWorker, InlineExecutor
from .geometry import CameraModel
from .governor import CIRCADIAN, CONVERSATION, LOAD, RateGovernor
from .pipeline import VisionPipeline
from .runner import FrameSource, LatestFrameRunner, LockedSource, pump
from .stages import (
    AppearanceEmbedder,
    AttributeEstimator,
    FaceDetector,
    FaceEmbedder,
    ObjectDetector,
    PerceptionModule,
)
from .tracker import Track, Tracker, TrackerResult
from .types import (
    AttributeSample,
    Box,
    Detection,
    FacePercept,
    Frame,
    FrameReport,
    PersonRecognized,
)

__all__ = [
    "CIRCADIAN",
    "CONVERSATION",
    "LOAD",
    "AppearanceEmbedder",
    "AttributeEstimator",
    "AttributeResult",
    "AttributeSample",
    "AttributeWorker",
    "Box",
    "CameraModel",
    "Detection",
    "FaceDetector",
    "FaceEmbedder",
    "FacePercept",
    "Frame",
    "FrameReport",
    "FrameSource",
    "LockedSource",
    "InlineExecutor",
    "LatestFrameRunner",
    "ObjectDetector",
    "PerceptionModule",
    "PersonRecognized",
    "RateGovernor",
    "Track",
    "Tracker",
    "TrackerResult",
    "VisionPipeline",
    "pump",
]
