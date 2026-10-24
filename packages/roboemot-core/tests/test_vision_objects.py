"""NanoDet-Plus-m object detector backend (SPEC 5, 13): decode and NMS on synthetic outputs,
class filter and threshold, and (opt in, ``-m models``) the real model on COCO images, with a
parity check against the independent bench implementation in ``scripts/bench/objects_lib.py``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision import models

pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
from roboemot.vision import onnx  # noqa: E402
from roboemot.vision.models import ModelError, load_manifest  # noqa: E402

LABELS = load_manifest()["nanodet-plus-m-320"].labels
CAT, CUP, PERSON, DOG = (LABELS.index(x) for x in ("cat", "cup", "person", "dog"))
ROWS = 2125
LEVEL1_START = 1600  # first row of the stride-16 map (40 * 40 rows come from stride 8)
BINS = onnx.NANODET_REG_BINS


def raw_output() -> np.ndarray:
    """A model output with no confident class anywhere and flat distances."""
    out = np.zeros((ROWS, len(LABELS) + onnx.NANODET_SIDES * BINS), np.float32)
    out[:, : len(LABELS)] = 0.01
    return out


def put(out: np.ndarray, row: int, column: int, score: float, bins=(2, 3, 4, 1)) -> None:
    """Set one confident prediction; ``bins`` is the peaked bin of left, top, right, bottom."""
    out[row, column] = score
    reg = np.full((onnx.NANODET_SIDES, BINS), -20.0, np.float32)
    for side, b in enumerate(bins):
        reg[side, b] = 20.0
    out[row, len(LABELS) :] = reg.ravel()


def decode(out, classes=("cat", "cup"), score_min=0.4, iou=0.6):
    columns = np.array([LABELS.index(c) for c in classes])
    return onnx.decode_nanodet(out, onnx.nanodet_priors(), columns, score_min, iou)


# ------------------------------------------------------------------------------ unit


def test_priors_follow_the_output_row_order():
    p = onnx.nanodet_priors()
    assert p.shape == (ROWS, 3)
    assert p[0].tolist() == [0, 0, 8] and p[1].tolist() == [8, 0, 8] and p[40].tolist() == [0, 8, 8]
    assert p[LEVEL1_START].tolist() == [0, 0, 16]
    assert p[-1].tolist() == [256, 256, 64]
    assert set(p[:, 2].astype(int)) == set(onnx.NANODET_STRIDES)


def test_decode_uses_the_expected_distance_times_stride_around_the_prior():
    out = raw_output()
    row = LEVEL1_START + 5 * 20 + 7  # stride 16: cell x=7, y=5 -> centre (112, 80)
    put(out, row, CAT, 0.9, bins=(2, 3, 4, 1))
    boxes, scores, classes = decode(out)
    assert boxes.shape == (1, 4) and classes.tolist() == [CAT]
    assert scores[0] == pytest.approx(0.9)
    assert boxes[0].tolist() == pytest.approx(
        [112 - 2 * 16, 80 - 3 * 16, 112 + 4 * 16, 80 + 1 * 16]
    )


def test_decode_clips_to_the_input_square():
    out = raw_output()
    put(out, 0, CAT, 0.9, bins=(7, 7, 7, 7))
    boxes, _, _ = decode(out)
    assert boxes[0].tolist() == [0.0, 0.0, 56.0, 56.0]
    out = raw_output()
    put(out, ROWS - 1, CAT, 0.9, bins=(0, 0, 7, 7))
    assert decode(out)[0][0].tolist() == [256.0, 256.0, 320.0, 320.0]


def test_only_the_configured_classes_above_the_threshold_are_returned():
    out = raw_output()
    put(out, LEVEL1_START + 3, PERSON, 0.95)
    put(out, LEVEL1_START + 50, DOG, 0.95)
    put(out, LEVEL1_START + 90, CAT, 0.39)
    put(out, LEVEL1_START + 130, CUP, 0.41)
    boxes, scores, classes = decode(out)
    assert classes.tolist() == [CUP] and scores[0] == pytest.approx(0.41)
    assert decode(out, classes=("person",))[2].tolist() == [PERSON]
    assert sorted(decode(out, score_min=0.3)[2].tolist()) == sorted([CAT, CUP])


def test_uniform_distance_distribution_decodes_to_the_expectation_of_3_5_strides():
    out = raw_output()
    out[LEVEL1_START + 5 * 20 + 7, CAT] = 0.9  # regression logits stay 0: uniform over 8 bins
    (box,) = decode(out)[0]
    assert box.tolist() == pytest.approx([112 - 56, 80 - 56, 112 + 56, 80 + 56])


def test_nothing_above_the_threshold_gives_empty_arrays():
    boxes, scores, classes = decode(raw_output())
    assert boxes.shape == (0, 4) and scores.shape == (0,) and classes.shape == (0,)


def test_nms_merges_overlap_within_a_class_but_not_across_classes():
    out = raw_output()
    put(out, LEVEL1_START + 5 * 20 + 7, CAT, 0.9)
    put(out, LEVEL1_START + 5 * 20 + 8, CAT, 0.7, bins=(2, 3, 4, 1))  # shifted by one cell
    put(out, LEVEL1_START + 5 * 20 + 7, CUP, 0.6)
    boxes, scores, classes = decode(out)
    assert sorted(classes.tolist()) == sorted([CAT, CUP])
    assert scores[classes == CAT].tolist() == pytest.approx([0.9])
    loose = decode(out, iou=0.95)
    assert (loose[2] == CAT).sum() == 2


def test_nms_function_order_and_threshold():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.5, 0.9, 0.4], np.float32)
    assert onnx.nms(boxes, scores, 0.6) == [1, 2]
    assert onnx.nms(boxes, scores, 0.9) == [1, 0, 2]


def test_detections_are_capped():
    out = raw_output()
    out[:, CAT] = 0.5
    reg = np.full((onnx.NANODET_SIDES, BINS), -20.0, np.float32)
    reg[:, 0] = 20.0  # zero-size boxes never overlap, so NMS keeps all of them
    out[:, len(LABELS) :] = reg.ravel()
    boxes, _, _ = decode(out, iou=0.99)
    assert len(boxes) == onnx.NANODET_MAX_DETECTIONS


class FakeSession:
    """Stands in for the ONNX session: returns a prepared output and records the input."""

    def __init__(self, output, shape=(1, 3, 320, 320)):
        self.output, self.shape, self.seen = output, list(shape), None

    def get_inputs(self):
        return [SimpleNamespace(name="data", shape=self.shape)]

    def get_outputs(self):
        return [SimpleNamespace(name="output", shape=[1, *self.output.shape])]

    def run(self, names, feed):
        self.seen = feed["data"]
        return [self.output[None]]


def detector(monkeypatch, output, classes=("cat", "cup"), **kw):
    session = FakeSession(output, **kw)
    monkeypatch.setattr(onnx, "_session", lambda path, threads: session)
    det = onnx.NanoDetDetector(Path("m.onnx"), LABELS, list(classes), 0.4, 0.6, 2)
    return det, session


def test_detect_scales_boxes_to_the_original_image_and_names_the_class(monkeypatch):
    out = raw_output()
    put(out, LEVEL1_START + 5 * 20 + 7, CAT, 0.9)  # 320x320 box (80, 32, 176, 96)
    det, session = detector(monkeypatch, out)
    image = np.full((480, 640, 3), 255, np.uint8)
    (d,) = det.detect(image)
    assert d.label == "cat" and d.score == pytest.approx(0.9)
    assert d.box == pytest.approx((160.0, 48.0, 352.0, 144.0))
    assert session.seen.shape == (1, 3, 320, 320) and session.seen.dtype == np.float32
    expected = (255 - np.array(onnx.NANODET_MEAN)) / np.array(onnx.NANODET_STD)
    assert session.seen[0, :, 10, 10] == pytest.approx(expected, rel=1e-5)


def test_detect_returns_nothing_for_a_quiet_frame(monkeypatch):
    det, _ = detector(monkeypatch, raw_output())
    assert det.detect(np.zeros((240, 320, 3), np.uint8)) == []


def test_detector_rejects_an_output_that_does_not_match_the_labels(monkeypatch):
    wrong_rows = raw_output()[:-1]
    with pytest.raises(ModelError, match="output"):
        detector(monkeypatch, wrong_rows)
    with pytest.raises(ModelError, match="output"):
        detector(monkeypatch, raw_output()[:, :-1])


def test_detector_rejects_unknown_classes_and_a_wrong_input_size(monkeypatch):
    with pytest.raises(ModelError, match="not known"):
        detector(monkeypatch, raw_output(), classes=("cat", "unicorn"))
    with pytest.raises(ModelError, match="expected 320x320"):
        detector(monkeypatch, raw_output(), shape=(1, 3, 416, 416))


# ---------------------------------------------------------------- real model (opt in)

CACHE = models.default_dir()
COCO_DIR = Path.home() / ".cache" / "roboemot-bench" / "datasets" / "coco" / "val2017"
# COCO val2017 images (fetch with `scripts/bench/fetch.py coco-samples`), pinned by sha256.
SAMPLES = {
    "000000078426.jpg": "475db6e34f57f03cf611cc4c359d8707df49628491adc27d6b5209607284778b",
    "000000002592.jpg": "ffa3d472d44184b1f48878f7b097189723e57a24f36c44c29e4bc561110fbf41",
}
# COCO val2017 ground truth (x1, y1, x2, y2) from instances_val2017.json.
COCO_BOXES = {
    "000000078426.jpg": {"cat": (0.0, 64.4, 275.7, 319.2), "cup": (267.7, 76.1, 356.4, 166.0)},
    "000000002592.jpg": {"cup": (222.9, 34.5, 432.6, 247.5)},
}
BENCH = Path(__file__).resolve().parents[3] / "scripts" / "bench"


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0.0) * max(h, 0.0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def sample(name: str) -> np.ndarray:
    import cv2

    path = COCO_DIR / name
    if not path.exists():
        pytest.skip(f"{path} missing (python scripts/bench/fetch.py coco-samples)")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SAMPLES[name]
    return cv2.imread(str(path))


@pytest.fixture(scope="module")
def real_stages():
    specs = load_manifest()
    if any(models.status(specs[n], CACHE) != "ok" for n in ("nanodet-plus-m-320", "dinov2-small")):
        pytest.skip(f"model files missing in {CACHE} (roboemot models fetch --all)")
    return onnx.load_stages(CACHE, RoboEmotConfig().vision)


@pytest.mark.models
def test_real_detector_finds_the_cat_and_the_cup(real_stages):
    found = {d.label: d for d in real_stages.object_detector.detect(sample("000000078426.jpg"))}
    assert {"cat", "cup"} <= set(found)
    cat, cup = found["cat"], found["cup"]
    assert cat.score > 0.6 and cup.score > 0.4
    assert iou(cat.box, COCO_BOXES["000000078426.jpg"]["cat"]) > 0.5
    assert iou(cup.box, COCO_BOXES["000000078426.jpg"]["cup"]) > 0.5
    only_cup = {d.label: d for d in real_stages.object_detector.detect(sample("000000002592.jpg"))}
    assert iou(only_cup["cup"].box, COCO_BOXES["000000002592.jpg"]["cup"]) > 0.5


@pytest.mark.models
def test_real_detector_feeds_percept_entity_with_embeddings(real_stages, tmp_path):
    from roboemot.bus import Bus
    from roboemot.registry import FileStore, Matcher, Registry
    from roboemot.vision import Frame, VisionPipeline

    cfg = RoboEmotConfig()
    reg = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), cfg.identity)
    bus = Bus()
    seen = []
    bus.subscribe("percept.entity", lambda m: seen.append(m.payload))

    class NoFaces:
        def detect(self, image):
            return []

    pipe = VisionPipeline(
        cfg.vision,
        cfg.identity,
        bus,
        Matcher(reg, cfg.identity),
        face_detector=NoFaces(),
        face_embedder=None,
        appearance_embedder=real_stages.appearance_embedder,
        object_detector=real_stages.object_detector,
    )
    image = sample("000000078426.jpg")
    for i in range(4):
        pipe.process(Frame(image, float(i)))
    kinds = {o.category: o for o in seen}
    assert kinds["cat"].kind == "animal" and kinds["cup"].kind == "object"
    assert kinds["cat"].embedding is not None and kinds["cat"].embedding.shape == (384,)
    assert 0 < kinds["cup"].confidence <= 1


@pytest.mark.models
def test_numpy_backend_matches_the_bench_reference(monkeypatch):
    specs = load_manifest()
    if models.status(specs["nanodet-plus-m-320"], CACHE) != "ok":
        pytest.skip(f"nanodet-plus-m-320 missing in {CACHE}")
    for name in ("objects_lib", "bench_lib"):
        monkeypatch.delitem(sys.modules, name, raising=False)  # restored when the test ends
    monkeypatch.syspath_prepend(str(BENCH))
    ref = pytest.importorskip("objects_lib").NanodetPlus(
        CACHE / specs["nanodet-plus-m-320"].files[0].path, threads=2
    )
    mine = onnx.NanoDetDetector(
        CACHE / specs["nanodet-plus-m-320"].files[0].path, LABELS, LABELS, 0.05, 0.6, 2
    )
    rng = np.random.default_rng(0)
    images = [sample(n) for n in SAMPLES] + [
        (rng.random((300, 500, 3)) * 255).astype(np.uint8),
        sample("000000078426.jpg")[:200, :250].copy(),
    ]
    total = 0
    for img in images:
        boxes, scores, classes = ref.detect(img, 0.05, 0.6)
        got = mine.detect(img)
        assert [d.label for d in got] == [LABELS[c] for c in classes]
        assert np.array([d.score for d in got]) == pytest.approx(scores, abs=1e-5)
        assert np.array([d.box for d in got]).reshape(-1, 4) == pytest.approx(boxes, abs=1e-2)
        total += len(got)
    assert total >= 4


@pytest.mark.models
def test_pinned_labels_are_the_class_list_of_the_pinned_source_file():
    import yaml

    spec = load_manifest()["nanodet-plus-m-320"]
    source = CACHE / "nanodet_plus" / "nanodet-plus-m_320.yml"
    if not source.exists():
        pytest.skip(f"{source} missing (python scripts/bench/fetch.py nanodet-plus)")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == spec.labels_sha256
    assert yaml.safe_load(source.read_text())["class_names"] == spec.labels
