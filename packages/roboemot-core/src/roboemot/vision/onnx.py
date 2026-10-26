"""ONNX backends for the vision stages (SPEC 5, 13): needs the ``vision`` extra.

YuNet face detector (MIT), SFace face embedder (Apache-2.0), DINOv2-S appearance embedder
(Apache-2.0) and NanoDet-Plus-m object detector (Apache-2.0), with the preprocessing verified
in the M1 spike (docs/spikes/2026-10-m1-models.md). Everything runs on CPU with two threads;
the cm4 budget leaves two cores to vision, audio and core together (SPEC 13). All resizing,
alignment and cropping happens here: crops live only inside a call (SPEC 12).
Weights are never downloaded here, see ``roboemot models fetch``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..config.schema import VisionConfig
from . import models
from .models import ModelError
from .types import Box, Detection

log = logging.getLogger(__name__)

LOG_ERROR = 4
CHANNELS = 3

YUNET_NMS_IOU = 0.3
YUNET_TOP_K = 5000
YUNET_BOX_END = 4
YUNET_KPS_END = 14
YUNET_SCORE_COLUMN = 14

SFACE_MODEL_ID = "sface-2021dec"
SFACE_SIZE = 112
# ArcFace 5-point template for a 112x112 crop (what OpenCV SFace uses).
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

PIXEL_MAX = 255.0

# NanoDet-Plus-m 320 (config nanodet-plus-m_320.yml of the release): plain resize without
# keeping the aspect ratio, BGR mean/std, four strides, distance distribution of 8 bins per side.
NANODET_NAME = "nanodet-plus-m-320"
NANODET_SIZE = 320
NANODET_MEAN = (103.53, 116.28, 123.675)
NANODET_STD = (57.375, 57.12, 58.395)
NANODET_STRIDES = (8, 16, 32, 64)
NANODET_REG_BINS = 8
NANODET_SIDES = 4
NANODET_MAX_DETECTIONS = 100
IOU_EPS = 1e-9

DINO_MODEL_ID = "dinov2-small"
DINO_SIZE = 224
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def configure(threads: int) -> None:
    """Process-wide ONNX Runtime log level and OpenCV thread count; called by ``load_stages``."""
    ort.set_default_logger_severity(LOG_ERROR)
    cv2.setNumThreads(threads)


def _session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = LOG_ERROR
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x)).astype(np.float32)


class YuNetDetector:
    """YuNet through ``cv2.FaceDetectorYN``, letterboxed top-left into the input size."""

    def __init__(self, path: Path, score_min: float, input_size: tuple[int, int]):
        self._size = input_size
        self._det = cv2.FaceDetectorYN.create(
            str(path), "", input_size, score_min, YUNET_NMS_IOU, YUNET_TOP_K
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Faces with score at least ``score_min``, in pixels of ``image``.

        The frame is letterboxed into the detector input (aspect ratio kept) and the boxes and
        landmarks are scaled back to the original pixels. The lowest scores feed the tracker.
        """
        w, h = self._size
        scale = min(w / image.shape[1], h / image.shape[0])
        nw, nh = int(image.shape[1] * scale), int(image.shape[0] * scale)
        canvas = np.zeros((h, w, CHANNELS), np.uint8)
        canvas[:nh, :nw] = cv2.resize(image, (nw, nh))
        self._det.setInputSize((w, h))
        _, rows = self._det.detect(canvas)
        if rows is None:
            return []
        out = []
        for r in rows:
            x, y, bw, bh = r[:YUNET_BOX_END] / scale
            kps = r[YUNET_BOX_END:YUNET_KPS_END].reshape(-1, 2) / scale
            out.append(Detection((x, y, x + bw, y + bh), float(r[YUNET_SCORE_COLUMN]), "face", kps))
        return out


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = SFACE_SIZE) -> np.ndarray:
    """Similarity-warp to the ArcFace template (least squares, Umeyama).

    ``landmarks`` are the 5 detector points (eyes, nose, mouth corners) in image pixels.
    Returns the aligned ``size`` x ``size`` crop that SFace expects.
    """
    src = landmarks.astype(np.float64)
    dst = ARCFACE_TEMPLATE.astype(np.float64) * (size / SFACE_SIZE)
    sm, dm = src.mean(0), dst.mean(0)
    s0, d0 = src - sm, dst - dm
    cov = d0.T @ s0 / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    scale = (sig * d).sum() / s0.var(0).sum()
    t = dm - scale * r @ sm
    m = np.hstack([scale * r, t[:, None]]).astype(np.float32)
    return cv2.warpAffine(image, m, (size, size), borderValue=0)


class SFaceEmbedder:
    """SFace through ``cv2.FaceRecognizerSF`` on the aligned 112x112 crop."""

    model_id = SFACE_MODEL_ID

    def __init__(self, path: Path):
        self._rec = cv2.FaceRecognizerSF.create(str(path), "")

    def embed(self, image: np.ndarray, face: Detection) -> np.ndarray:
        """Unit-length embedding of one face; the aligned crop is dropped on return."""
        if face.landmarks is None:
            raise ValueError("SFace needs the detector's landmarks")
        return _unit(self._rec.feature(align_face(image, face.landmarks))[0])


class DinoV2Embedder:
    """DINOv2-S/14 CLS token of the whole box, squashed to 224x224 (no margin).

    ``embed`` is serialised by a lock: the vision worker and the catalogue import (web UI
    thread) share one session, which the project keeps single-threaded.
    """

    model_id = DINO_MODEL_ID

    def __init__(self, path: Path, threads: int):
        self._lock = threading.Lock()
        self._sess = _session(path, threads)
        self._input = self._sess.get_inputs()[0].name

    def embed(self, image: np.ndarray, box: Box) -> np.ndarray:
        """Unit-length appearance embedding of the box; the crop is dropped on return."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        crop = image[max(0, int(y1)) : min(height, int(y2)), max(0, int(x1)) : min(width, int(x2))]
        if crop.size == 0:
            raise ValueError("empty crop")
        rgb = cv2.cvtColor(
            cv2.resize(crop, (DINO_SIZE, DINO_SIZE), interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_BGR2RGB,
        )
        x = (rgb.astype(np.float32) / PIXEL_MAX - np.array(DINO_MEAN, np.float32)) / np.array(
            DINO_STD, np.float32
        )
        with self._lock:
            tokens = self._sess.run(None, {self._input: x.transpose(2, 0, 1)[None]})[0][0]
        return _unit(tokens[0])


def nanodet_priors(size: int = NANODET_SIZE) -> np.ndarray:
    """Grid-cell centres of the four feature maps as ``(N, 3)`` rows ``x, y, stride`` in pixels.

    Row-major per level (x fastest), levels concatenated, which is the order of the model
    output rows (2125 rows at 320x320).
    """
    rows = []
    for stride in NANODET_STRIDES:
        n = -(-size // stride)
        ys, xs = np.mgrid[0:n, 0:n]
        rows.append(
            np.stack([xs.ravel() * stride, ys.ravel() * stride, np.full(n * n, stride)], axis=1)
        )
    return np.concatenate(rows).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_max: float) -> list[int]:
    """Greedy non-maximum suppression; indices kept, best score first.

    ``boxes`` are ``(N, 4)`` xyxy. A box is dropped when its IoU with a better kept box exceeds
    ``iou_max``.
    """
    order = np.argsort(-scores, kind="stable")
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.prod(np.clip(rb - lt, 0, None), axis=1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, IOU_EPS)
        order = rest[iou <= iou_max]
    return keep


def decode_nanodet(
    output: np.ndarray,
    priors: np.ndarray,
    columns: np.ndarray,
    score_min: float,
    iou_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boxes, scores and class columns from one NanoDet-Plus output, in input-image pixels.

    ``output`` is ``(N, classes + 4 * bins)``: class scores (already sigmoid) then, per side
    (left, top, right, bottom), a distribution over ``NANODET_REG_BINS`` distances in strides.
    Only the class ``columns`` are considered. Candidates at or above ``score_min`` are
    decoded (softmax expectation times stride, around the prior centre, clipped to the input),
    merged per class by NMS (each class on its own) and capped at ``NANODET_MAX_DETECTIONS``.
    """
    scores = output[:, columns]
    boxes_none = np.zeros((0, NANODET_SIDES), np.float32)
    anchors, picked = np.nonzero(scores >= score_min)
    if anchors.size == 0:
        return boxes_none, np.zeros(0, np.float32), np.zeros(0, np.int64)
    classes = columns[picked]
    conf = scores[anchors, picked]
    reg = output[anchors, -NANODET_SIDES * NANODET_REG_BINS :]
    reg = reg.reshape(-1, NANODET_SIDES, NANODET_REG_BINS)
    e = np.exp(reg - reg.max(axis=2, keepdims=True))
    dist = (e / e.sum(axis=2, keepdims=True) * np.arange(NANODET_REG_BINS)).sum(axis=2)
    cx, cy, stride = priors[anchors].T
    left, top, right, bottom = (dist * stride[:, None]).T
    boxes = np.stack([cx - left, cy - top, cx + right, cy + bottom], axis=1)
    boxes = np.clip(boxes, 0, NANODET_SIZE)
    kept: list[int] = []
    for c in np.unique(classes):
        same = np.flatnonzero(classes == c)
        kept.extend(int(same[i]) for i in nms(boxes[same], conf[same], iou_max))
    keep = sorted(kept, key=lambda i: -conf[i])[:NANODET_MAX_DETECTIONS]
    return boxes[keep], conf[keep], classes[keep]


class NanoDetDetector:
    """NanoDet-Plus-m 320 (Apache-2.0) object detector for the animal and object stage.

    Implements the ``ObjectDetector`` protocol: ``Detection.label`` is the class name from the
    manifest labels. Only the classes in ``classes`` are returned (person and the rest are never
    scored, decoded or merged), so the cost does not grow with the 80-class output.
    """

    def __init__(
        self,
        path: Path,
        labels: list[str],
        classes: list[str],
        score_min: float,
        nms_iou: float,
        threads: int,
    ):
        unknown = sorted(set(classes) - set(labels))
        if unknown:
            raise ModelError(f"object classes not known to {NANODET_NAME}: {unknown}")
        self._labels = labels
        self._columns = np.array([labels.index(c) for c in dict.fromkeys(classes)], dtype=np.int64)
        self._score_min = score_min
        self._nms_iou = nms_iou
        self._priors = nanodet_priors()
        self._sess = _session(path, threads)
        self._input = self._sess.get_inputs()[0].name
        self._mean = np.array(NANODET_MEAN, np.float32)
        self._std = np.array(NANODET_STD, np.float32)
        shape = self._sess.get_inputs()[0].shape
        if shape[2:] != [NANODET_SIZE, NANODET_SIZE]:
            raise ModelError(f"{path.name}: input {shape}, expected {NANODET_SIZE}x{NANODET_SIZE}")
        columns = len(labels) + NANODET_SIDES * NANODET_REG_BINS
        out_shape = self._sess.get_outputs()[0].shape
        if out_shape[1:] != [len(self._priors), columns]:
            raise ModelError(
                f"{path.name}: output {out_shape}, expected rows {len(self._priors)} "
                f"x {columns} columns ({len(labels)} classes)"
            )

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Labelled boxes (original pixels) of the configured classes at or above the floor.

        The frame is squashed to 320x320 (the model was trained without keeping the aspect
        ratio), run, decoded and merged per class; boxes are scaled back to ``image``.
        """
        height, width = image.shape[:2]
        resized = cv2.resize(image, (NANODET_SIZE, NANODET_SIZE), interpolation=cv2.INTER_LINEAR)
        x = ((resized.astype(np.float32) - self._mean) / self._std).transpose(2, 0, 1)[None]
        output = self._sess.run(None, {self._input: np.ascontiguousarray(x)})[0][0]
        boxes, scores, classes = decode_nanodet(
            output, self._priors, self._columns, self._score_min, self._nms_iou
        )
        scale = np.array([width, height, width, height], np.float32) / NANODET_SIZE
        return [
            Detection(tuple(float(v) for v in box * scale), float(s), self._labels[c])
            for box, s, c in zip(boxes, scores, classes, strict=True)
        ]


@dataclass
class Stages:
    """The ONNX backends built from a models folder; optional ones are None when skipped."""

    face_detector: YuNetDetector
    face_embedder: SFaceEmbedder
    appearance_embedder: DinoV2Embedder | None
    object_detector: NanoDetDetector | None = None
    objects_unavailable: str | None = None  # why the animal and object stage was dropped


def _file(root: Path, name: str, index: int = 0) -> Path:
    spec = models.load_manifest()[name]
    return models.check_file(root, spec.files[index])


def load_object_detector(root: Path, cfg: VisionConfig) -> NanoDetDetector:
    """The NanoDet-Plus-m detector for the configured animal and object classes.

    Raises ``ModelError`` when the file is missing or unverified. Callers that can run without
    objects (the app) catch it so the rest of vision keeps working.
    """
    spec = models.load_manifest()[NANODET_NAME]
    if spec.labels is None:
        raise ModelError("models.yaml lacks the NanoDet class labels")
    return NanoDetDetector(
        _file(root, NANODET_NAME),
        spec.labels,
        [*cfg.animal_categories, *cfg.object_categories],
        cfg.object_score_min,
        cfg.object_nms_iou,
        cfg.threads,
    )


def load_stages(
    root: Path,
    cfg: VisionConfig,
    *,
    profile: str | None = None,
    appearance: bool = True,
    objects: bool = True,
) -> Stages:
    """Load the backends from ``root``; DINOv2 and the object detector can be skipped.

    The face stack (YuNet, SFace) is required: a missing or unverified file raises
    ``ModelError``. The animal and object stage is optional: the NanoDet-Plus-m 320 detector
    (when ``objects`` is true and ``vision.objects_enabled`` is on; it returns the
    ``vision.animal_categories`` and ``vision.object_categories`` classes only) and the DINOv2-S
    embedder (when ``appearance``) are loaded together, and if either fails both are dropped
    (no 88 MB embedder kept for nothing), ``Stages.objects_unavailable`` holds the reason and
    faces keep working.

    Every file must carry a valid ``.verified`` marker (``roboemot models fetch`` writes it);
    otherwise ``ModelError`` names the file. Sessions use ``vision.threads`` CPU threads (SPEC 13:
    two cores are shared by vision, audio and core). ``profile`` (the robot manifest's) is
    accepted for callers and not used. Sets the OpenCV and ONNX Runtime globals once.
    """
    configure(cfg.threads)
    detector, embedder, unavailable = None, None, None
    wanted = objects and cfg.objects_enabled
    if wanted or appearance:
        try:
            detector = load_object_detector(root, cfg) if wanted else None
            embedder = (
                DinoV2Embedder(_file(root, "dinov2-small"), cfg.threads) if appearance else None
            )
        except Exception as exc:
            detector = embedder = None
            unavailable = f"{type(exc).__name__}: {exc}"
            log.warning("animal and object stage unavailable, running without it (%s)", exc)
    return Stages(
        YuNetDetector(
            _file(root, "yunet"),
            cfg.tracker.low_score,
            (cfg.detector_width_px, cfg.detector_height_px),
        ),
        SFaceEmbedder(_file(root, "sface")),
        embedder,
        detector,
        unavailable,
    )
