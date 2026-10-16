"""Object detector candidates for the M1 spike (SPEC 5, 13, 15): YOLOX-Nano and NanoDet-Plus-m.

Reference implementations of preprocessing, decoding and NMS written to follow the upstream demo
code (YOLOX ``demo_utils.demo_postprocess``, NanoDet ``nanodet_plus_head.get_bboxes``), with
loops and ``cv2.dnn.NMSBoxes`` on purpose: ``tests/test_vision_objects.py`` checks the shipped
numpy backend (``roboemot.vision.onnx.OnnxObjectDetector``) against these. Only loads ONNX
weights and image data from the cache; nothing downloaded is executed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from bench_lib import DATASETS, MODELS

COCO_ANNOTATIONS = DATASETS / "coco" / "instances_val2017.json"
YOLOX_FILE = MODELS / "yolox_nano" / "yolox_nano.onnx"
NANODET_FILE = MODELS / "nanodet_plus" / "nanodet-plus-m_320.onnx"

# The 16 household-relevant COCO classes of the spike (name -> reported as animal or object).
RELEVANT = (
    "person",
    "cat",
    "dog",
    "bird",
    "cup",
    "bottle",
    "chair",
    "couch",
    "potted plant",
    "tv",
    "laptop",
    "book",
    "remote",
    "cell phone",
    "bowl",
    "teddy bear",
)

# NMS IoU of each upstream's own evaluation setting (YOLOX tools/eval, NanoDet head).
YOLOX_NMS_IOU = 0.65
NANODET_NMS_IOU = 0.6
NANODET_REG_MAX = 7
NANODET_STRIDES = (8, 16, 32, 64)
MAX_DETECTIONS = 100


def coco_names() -> list[str]:
    """The 80 COCO class names in detector output order (annotation categories by id)."""
    cats = json.loads(COCO_ANNOTATIONS.read_text())["categories"]
    return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]


def _session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def classwise_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, nms_iou: float
) -> np.ndarray:
    """Indices kept by per-class NMS (``cv2.dnn.NMSBoxes`` on x, y, w, h), best score first."""
    keep: list[int] = []
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        xywh = np.column_stack([boxes[idx, :2], boxes[idx, 2:] - boxes[idx, :2]])
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores[idx].tolist(), 0.0, nms_iou)
        keep.extend(idx[np.asarray(kept, dtype=int).reshape(-1)].tolist())
    order = sorted(keep, key=lambda i: -scores[i])[:MAX_DETECTIONS]
    return np.asarray(order, dtype=int)


class YoloxNano:
    """YOLOX-Nano 416x416: letterbox top-left with 114, BGR 0..255, grid decode, obj x cls."""

    name = "yolox_nano_416"
    size = 416
    nms_iou = YOLOX_NMS_IOU

    def __init__(self, path: Path = YOLOX_FILE, threads: int = 2):
        self.sess = _session(path, threads)
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        r = min(self.size / img.shape[0], self.size / img.shape[1])
        resized = cv2.resize(
            img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        return canvas.transpose(2, 0, 1)[None].astype(np.float32), r

    def decode(self, out: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (xyxy, original pixels), scores and class ids of every anchor and class."""
        rows = []
        for stride in (8, 16, 32):
            n = self.size // stride
            for y in range(n):
                for x in range(n):
                    rows.append((x, y, stride))
        grid = np.asarray(rows, dtype=np.float32)
        cx = (out[:, 0] + grid[:, 0]) * grid[:, 2]
        cy = (out[:, 1] + grid[:, 1]) * grid[:, 2]
        w = np.exp(out[:, 2]) * grid[:, 2]
        h = np.exp(out[:, 3]) * grid[:, 2]
        xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) / ratio
        scores = out[:, 4:5] * out[:, 5:]
        return xyxy, scores, np.arange(scores.shape[1])

    def detect(
        self, img: np.ndarray, score_min: float, nms_iou: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(boxes, scores, class ids) above ``score_min`` after per-class NMS."""
        x, ratio = self.preprocess(img)
        out = self.sess.run(None, {self.input_name: x})[0][0]
        xyxy, scores, _ = self.decode(out, ratio)
        a, c = np.nonzero(scores >= score_min)
        boxes, sc, cl = xyxy[a], scores[a, c], c
        keep = classwise_nms(boxes, sc, cl, self.nms_iou if nms_iou is None else nms_iou)
        return boxes[keep], sc[keep], cl[keep]


class NanodetPlus:
    """NanoDet-Plus-m 320x320: plain resize (keep_ratio False), BGR mean/std, DFL decode."""

    name = "nanodet_plus_m_320"
    size = 320
    nms_iou = NANODET_NMS_IOU
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)

    def __init__(self, path: Path = NANODET_FILE, threads: int = 2):
        self.sess = _session(path, threads)
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        resized = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) - self.mean) / self.std
        return x.transpose(2, 0, 1)[None], (self.size / img.shape[1], self.size / img.shape[0])

    def decode(self, out: np.ndarray, scale: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Boxes (xyxy, original pixels) and class scores of every prior."""
        priors = []
        for stride in NANODET_STRIDES:
            n = math.ceil(self.size / stride)
            for y in range(n):
                for x in range(n):
                    priors.append((x * stride, y * stride, stride))
        p = np.asarray(priors, dtype=np.float32)
        scores = out[:, :80]
        dist = out[:, 80:].reshape(-1, 4, NANODET_REG_MAX + 1)
        e = np.exp(dist - dist.max(axis=2, keepdims=True))
        dist = (e / e.sum(axis=2, keepdims=True) * np.arange(NANODET_REG_MAX + 1)).sum(axis=2)
        dist = dist * p[:, 2:3]
        boxes = np.column_stack(
            [p[:, 0] - dist[:, 0], p[:, 1] - dist[:, 1], p[:, 0] + dist[:, 2], p[:, 1] + dist[:, 3]]
        )
        boxes = np.clip(boxes, 0, self.size)
        boxes[:, [0, 2]] /= scale[0]
        boxes[:, [1, 3]] /= scale[1]
        return boxes, scores

    def detect(
        self, img: np.ndarray, score_min: float, nms_iou: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(boxes, scores, class ids) above ``score_min`` after per-class NMS."""
        x, scale = self.preprocess(img)
        out = self.sess.run(None, {self.input_name: x})[0][0]
        xyxy, scores = self.decode(out, scale)
        a, c = np.nonzero(scores >= score_min)
        boxes, sc, cl = xyxy[a], scores[a, c], c
        keep = classwise_nms(boxes, sc, cl, self.nms_iou if nms_iou is None else nms_iou)
        return boxes[keep], sc[keep], cl[keep]
