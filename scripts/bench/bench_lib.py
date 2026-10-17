"""Shared helpers for the M1 model spike (SPEC 5, 13, 15): detectors, alignment, embedders, metrics.

Only loads ONNX weights and image data from the cache; nothing downloaded is executed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CACHE = Path(os.environ.get("ROBOEMOT_BENCH_CACHE", Path.home() / ".cache" / "roboemot-bench"))
MODELS = CACHE / "models"
DATASETS = CACHE / "datasets"

ort.set_default_logger_severity(4)
cv2.setNumThreads(2)

# ArcFace 5-point template for a 112x112 crop (also what OpenCV SFace uses).
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

MODEL_FILES = {
    "scrfd_500m": MODELS / "insightface/buffalo_s/det_500m.onnx",
    "scrfd_2.5g": MODELS / "insightface/scrfd_2.5g_bnkps.onnx",
    "yunet": MODELS / "opencv_zoo/face_detection_yunet_2023mar.onnx",
    "mbf": MODELS / "insightface/buffalo_s/w600k_mbf.onnx",
    "r50": MODELS / "insightface/buffalo_l/w600k_r50.onnx",
    "sface": MODELS / "opencv_zoo/face_recognition_sface_2021dec.onnx",
    "genderage": MODELS / "insightface/buffalo_l/genderage.onnx",
    "dinov2_s": MODELS / "dinov2_small/model.onnx",
    "clip_b32": MODELS / "clip_b32/vision_model.onnx",
    "mobileclip_s0": MODELS / "mobileclip_s0/vision_model.onnx",
    "fairface_age_vit": MODELS / "fairface/age_model.onnx",
    "fairface_gender_vit": MODELS / "fairface/gender_model.onnx",
}


def session(path: Path, threads: int = 0) -> ort.InferenceSession:
    """Create a CPU ONNX Runtime session; threads=0 means the default of 2 (shared machine)."""
    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.intra_op_num_threads = threads or 2
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


@dataclass
class Face:
    box: np.ndarray  # x1, y1, x2, y2
    kps: np.ndarray  # 5x2, image-left eye first
    score: float


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thr]
    return keep


class Scrfd:
    """InsightFace SCRFD (bnkps variants: 3 strides, 2 anchors, 5 landmarks)."""

    def __init__(self, path: Path, size: tuple[int, int] = (640, 640), threads: int = 0):
        self.sess = session(path, threads)
        self.input = self.sess.get_inputs()[0].name
        self.size = size  # (w, h)

    def detect(self, bgr: np.ndarray, thresh: float = 0.5) -> list[Face]:
        w, h = self.size
        ratio = bgr.shape[0] / bgr.shape[1]
        if ratio > h / w:
            nh, nw = h, int(h / ratio)
        else:
            nw, nh = w, int(w * ratio)
        scale = nh / bgr.shape[0]
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (w, h), (127.5,) * 3, swapRB=True)
        outs = self.sess.run(None, {self.input: blob})
        boxes_all, kps_all, scores_all = [], [], []
        for idx, stride in enumerate((8, 16, 32)):
            scores = outs[idx].reshape(-1)
            bbox = outs[idx + 3].reshape(-1, 4) * stride
            kps = outs[idx + 6].reshape(-1, 10) * stride
            gh, gw = -(-h // stride), -(-w // stride)  # ceil: 240 is not a multiple of 32
            centers = np.stack(np.mgrid[:gh, :gw][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            centers = np.stack([centers] * 2, axis=1).reshape(-1, 2)
            pos = scores >= thresh
            c = centers[pos]
            b = np.concatenate([c - bbox[pos][:, :2], c + bbox[pos][:, 2:]], axis=1)
            k = kps[pos].reshape(-1, 5, 2) + c[:, None, :]
            # kps are offsets from the anchor centre: x = cx + d0 ...
            boxes_all.append(b)
            kps_all.append(k)
            scores_all.append(scores[pos])
        if not scores_all or sum(len(s) for s in scores_all) == 0:
            return []
        boxes = np.concatenate(boxes_all) / scale
        kps = np.concatenate(kps_all) / scale
        scores = np.concatenate(scores_all)
        keep = _nms(boxes, scores, 0.4)
        return [Face(boxes[i], kps[i], float(scores[i])) for i in keep]


class YuNet:
    """OpenCV Zoo YuNet through cv2.FaceDetectorYN (the model's reference runtime)."""

    def __init__(self, path: Path, size: tuple[int, int] = (640, 640), score: float = 0.5):
        self.det = cv2.FaceDetectorYN.create(str(path), "", size, score, 0.3, 5000)
        self.size = size

    def detect(self, bgr: np.ndarray, thresh: float = 0.5) -> list[Face]:
        w, h = self.size
        ratio = bgr.shape[0] / bgr.shape[1]
        if ratio > h / w:
            nh, nw = h, int(h / ratio)
        else:
            nw, nh = w, int(w * ratio)
        scale = nh / bgr.shape[0]
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        self.det.setInputSize((w, h))
        _, rows = self.det.detect(canvas)
        if rows is None:
            return []
        faces = []
        for r in rows:
            if r[14] < thresh:
                continue
            x, y, bw, bh = r[:4]
            box = np.array([x, y, x + bw, y + bh]) / scale
            kps = r[4:14].reshape(5, 2) / scale
            faces.append(Face(box, kps, float(r[14])))
        return faces


def align(bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-warp to the ArcFace template (least squares, Umeyama)."""
    src = kps.astype(np.float64)
    dst = ARCFACE_TEMPLATE.astype(np.float64) * (size / 112)
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
    return cv2.warpAffine(bgr, m, (size, size), borderValue=0)


def largest(faces: list[Face]) -> Face | None:
    if not faces:
        return None
    return max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))


class ArcEmbedder:
    """InsightFace ArcFace-style ONNX (w600k_mbf / w600k_r50): RGB, (x-127.5)/127.5."""

    def __init__(self, path: Path, threads: int = 0):
        self.sess = session(path, threads)
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, aligned_bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(aligned_bgr, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        e = self.sess.run(None, {self.input: blob})[0][0]
        return e / np.linalg.norm(e)


class SFace:
    """OpenCV Zoo SFace via cv2.FaceRecognizerSF (reference preprocessing)."""

    def __init__(self, path: Path):
        self.rec = cv2.FaceRecognizerSF.create(str(path), "")

    def __call__(self, aligned_bgr: np.ndarray) -> np.ndarray:
        e = self.rec.feature(aligned_bgr)[0]
        return e / np.linalg.norm(e)


# ---- generic image embedders (individuals spike) ----------------------------------------


def _prep(bgr: np.ndarray, size: int, mean, std, rescale: bool = True) -> np.ndarray:
    img = cv2.cvtColor(
        cv2.resize(bgr, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB
    )
    x = img.astype(np.float32) / (255.0 if rescale else 1.0)
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    return x.transpose(2, 0, 1)[None]


class ImageEmbedder:
    """DINOv2-S/14 (CLS), CLIP ViT-B/32 and MobileCLIP-S0 image towers.

    Input is the whole crop squashed to the model size (the robot feeds detector boxes).
    """

    SPECS = {
        "dinov2_s": (224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "clip_b32": (
            224,
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
        "mobileclip_s0": (256, (0, 0, 0), (1, 1, 1)),
    }

    def __init__(self, name: str, threads: int = 0, variant: str = "cls"):
        self.name = name
        self.size, self.mean, self.std = self.SPECS[name]
        self.sess = session(MODEL_FILES[name], threads)
        self.input = self.sess.get_inputs()[0].name
        self.variant = variant

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        x = _prep(bgr, self.size, self.mean, self.std)
        out = self.sess.run(None, {self.input: x})[0]
        if self.name == "dinov2_s":
            tokens = out[0]
            e = (
                tokens[0]
                if self.variant == "cls"
                else np.concatenate([tokens[0], tokens[1:].mean(0)])
            )
        else:
            e = out[0]
        return (e / np.linalg.norm(e)).astype(np.float32)


# ---- metrics --------------------------------------------------------------------------


def roc_threshold(neg: np.ndarray, far: float) -> float:
    """Cosine threshold at which a fraction `far` of impostor scores is accepted."""
    return float(np.quantile(neg, 1.0 - far))


def tar_at_far(pos: np.ndarray, neg: np.ndarray, far: float) -> tuple[float, float]:
    thr = roc_threshold(neg, far)
    return float((pos > thr).mean()), thr


def dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    return float((pos.mean() - neg.mean()) / np.sqrt(0.5 * (pos.var() + neg.var())))


def eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold (sweeps all observed scores)."""
    thr = np.unique(np.concatenate([pos, neg]))
    if len(thr) > 4000:
        thr = np.quantile(np.concatenate([pos, neg]), np.linspace(0, 1, 4000))
    pos_s, neg_s = np.sort(pos), np.sort(neg)
    frr = np.searchsorted(pos_s, thr, side="left") / len(pos_s)
    far = 1.0 - np.searchsorted(neg_s, thr, side="left") / len(neg_s)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(thr[i])
