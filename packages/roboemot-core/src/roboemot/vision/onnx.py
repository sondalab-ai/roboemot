"""ONNX backends for the vision stages (SPEC 5, 13): needs the ``vision`` extra.

YuNet face detector (MIT), SFace face embedder (Apache-2.0), FairFace ViT-B age and gender
(Apache-2.0) and DINOv2-S appearance embedder (Apache-2.0), with the preprocessing verified
in the M1 spike (docs/spikes/2026-10-m1-models.md). Everything runs on CPU with two threads;
the cm4 budget leaves two cores to vision, audio and core together (SPEC 13). All resizing,
letterboxing, alignment and cropping happens here: crops live only inside a call (SPEC 12).
Weights are never downloaded here, see ``roboemot models fetch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..config.schema import VisionConfig
from . import models
from .models import ModelError
from .types import AttributeSample, Box, Detection

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

VIT_SIZE = 224
VIT_MEAN = 0.5
VIT_STD = 0.5
PIXEL_MAX = 255.0
# FairFace age bins by label. The bin 10-19 cannot be split at 13, so it maps to teen (SPEC 5
# groups: child, teen, adult, senior). The output index order comes from models.yaml.
AGE_LABEL_GROUPS = {
    "0-2": "child",
    "3-9": "child",
    "10-19": "teen",
    "20-29": "adult",
    "30-39": "adult",
    "40-49": "adult",
    "50-59": "adult",
    "60-69": "senior",
    "more than 70": "senior",
}
GENDER_LABEL_VALUES = {"Female": "female", "Male": "male"}

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


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


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


class FairFaceEstimator:
    """FairFace ViT-B age and gender classifiers on a padded face crop.

    Per crop: the argmax age group (bins summed into child, teen, adult, senior) and gender
    with their probabilities. Thresholding and aggregation over crops are done by the registry
    estimate helpers, so the 0.8 rule is applied once.
    """

    def __init__(
        self,
        age_path: Path,
        gender_path: Path,
        age_labels: list[str],
        gender_labels: list[str],
        threads: int,
    ):
        unknown = [x for x in age_labels if x not in AGE_LABEL_GROUPS] + [
            x for x in gender_labels if x not in GENDER_LABEL_VALUES
        ]
        if unknown:
            raise ModelError(f"unexpected FairFace labels {unknown}")
        self._age_groups = [AGE_LABEL_GROUPS[x] for x in age_labels]
        self._genders = [GENDER_LABEL_VALUES[x] for x in gender_labels]
        self._age = _session(age_path, threads)
        self._gender = _session(gender_path, threads)

    @staticmethod
    def _input(crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(crop, (VIT_SIZE, VIT_SIZE)), cv2.COLOR_BGR2RGB)
        return ((rgb.astype(np.float32) / PIXEL_MAX - VIT_MEAN) / VIT_STD).transpose(2, 0, 1)[None]

    def estimate(self, crop: np.ndarray) -> AttributeSample:
        """Age group and gender of one crop (BGR).

        The crop is resized to 224x224 and normalised as the FairFace ViT expects. Returns the
        most likely age group and gender with their probabilities, no thresholding.
        """
        x = self._input(crop)
        age = _softmax(self._age.run(None, {self._age.get_inputs()[0].name: x})[0][0])
        gender = _softmax(self._gender.run(None, {self._gender.get_inputs()[0].name: x})[0][0])
        totals: dict[str, float] = {}
        for group, p in zip(self._age_groups, age, strict=True):
            totals[group] = totals.get(group, 0.0) + float(p)
        age_group, age_conf = max(totals.items(), key=lambda kv: kv[1])
        g = int(gender.argmax())
        return AttributeSample(age_group, age_conf, self._genders[g], float(gender[g]))


class DinoV2Embedder:
    """DINOv2-S/14 CLS token of the whole box, squashed to 224x224 (no margin)."""

    model_id = DINO_MODEL_ID

    def __init__(self, path: Path, threads: int):
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
        tokens = self._sess.run(None, {self._input: x.transpose(2, 0, 1)[None]})[0][0]
        return _unit(tokens[0])


@dataclass
class Stages:
    """The ONNX backends built from a models folder; optional ones are None when skipped."""

    face_detector: YuNetDetector
    face_embedder: SFaceEmbedder
    attribute_estimator: FairFaceEstimator | None
    appearance_embedder: DinoV2Embedder | None


def _file(root: Path, name: str, index: int = 0) -> Path:
    spec = models.load_manifest()[name]
    return models.check_file(root, spec.files[index])


def load_stages(
    root: Path,
    cfg: VisionConfig,
    *,
    profile: str | None = None,
    attributes: bool = True,
    appearance: bool = True,
) -> Stages:
    """Load the backends from ``root``; the ViTs and DINOv2 can be left out.

    Every file must carry a valid ``.verified`` marker (``roboemot models fetch`` writes it);
    otherwise ``ModelError`` names the file. Frame-path sessions use ``vision.threads`` CPU
    threads, the age and gender sessions ``vision.attribute_threads`` (SPEC 13: two cores are
    shared by vision, audio and core). ``vision.attribute_precision`` picks the INT8 files or the
    343 MB FP32 files of the age and gender models; ``auto`` follows ``profile``, the robot
    manifest's profile (``cm4`` gives INT8, anything else FP32). Sets the OpenCV and ONNX
    Runtime globals once.
    """
    configure(cfg.threads)
    specs = models.load_manifest()
    estimator = None
    if attributes:
        precision = models.resolve_precision(cfg.attribute_precision, profile)
        suffix = "-int8" if precision == "int8" else ""
        age_name, gender_name = f"fairface-age{suffix}", f"fairface-gender{suffix}"
        age, gender = specs[age_name], specs[gender_name]
        if age.labels is None or gender.labels is None:
            raise ModelError("models.yaml lacks the FairFace label order")
        estimator = FairFaceEstimator(
            _file(root, age_name),
            _file(root, gender_name),
            age.labels,
            gender.labels,
            cfg.attribute_threads,
        )
    return Stages(
        YuNetDetector(
            _file(root, "yunet"),
            cfg.tracker.low_score,
            (cfg.detector_width_px, cfg.detector_height_px),
        ),
        SFaceEmbedder(_file(root, "sface")),
        estimator,
        DinoV2Embedder(_file(root, "dinov2-small"), cfg.threads) if appearance else None,
    )
