"""Age from yolo11n-face-age (SPEC 5, Rules for estimated attributes; 13).

The decode is checked on hand-made outputs and, end to end, on a tiny synthetic ONNX file that
returns a fixed ``1 x 7 x anchors`` tensor and carries the same metadata keys as the real file.
"""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.config.schema import IdentityConfig, RoboEmotConfig
from roboemot.registry import estimate_age_group
from roboemot.vision import AttributeWorker, InlineExecutor, models
from roboemot.vision.models import ModelError, load_manifest
from roboemot.vision.types import AttributeSample

LABELS = ["0-14", "15-22", "22+"]
GROUPS = {"0-14": "child", "15-22": "teen", "22+": "adult"}
VISION = RoboEmotConfig().vision
CACHE = models.default_dir()  # read at import, before the per-test data folder (conftest)


def output(scores: list[list[float]]) -> np.ndarray:
    """``(4 + classes, anchors)``: box rows (ignored by the decode), then one row per class."""
    cls = np.array(scores, np.float32).T
    return np.concatenate([np.full((4, cls.shape[1]), 50.0, np.float32), cls])


def test_decode_takes_the_best_anchor_and_its_class():
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision.onnx import decode_age_classes

    out = output([[0.1, 0.2, 0.3], [0.05, 0.9, 0.4], [0.7, 0.1, 0.2]])
    assert decode_age_classes(out) == (1, pytest.approx(0.9))
    assert decode_age_classes(output([[0.0, 0.0, 0.0]]))[1] == 0.0


def test_manifest_entry_pins_the_age_model_and_its_groups():
    spec = load_manifest()["yolo11n-face-age"]
    assert spec.role == "attribute_estimator" and spec.licence == "CC-BY-NC-4.0"
    assert "AGPL-3.0" in spec.notes
    assert spec.labels == LABELS and spec.groups == GROUPS
    (f,) = spec.files
    assert f.url == (
        "https://huggingface.co/AdamCodd/yolo11n-face-age/resolve/"
        "410bdd84fa762b6aa550ef73bd57865a2d30be37/model.onnx"
    )
    assert f.size_bytes == 10_513_214
    assert f.sha256 == "e37945afcbc1b855f4a66bbf1e13db9d19d5ca2e8ddb2ffe2be2906d53f5ff93"


def test_manifest_groups_must_cover_the_labels():
    from roboemot.vision.models import ModelSpec

    with pytest.raises(ValueError, match="groups"):
        ModelSpec(
            name="m", role="r", licence="x", source="s", files=[],
            labels=LABELS, groups={"0-14": "child"},
        )  # fmt: skip


def test_schema_defaults_and_ranges():
    assert VISION.attribute_input_px == 160 and VISION.attribute_score_min == 0.5
    assert VISION.attribute_crops == 1 and RoboEmotConfig().identity.estimate_frames_min == 1
    for px in (96, 256, 150):
        with pytest.raises(ValueError):
            RoboEmotConfig(vision={"attribute_input_px": px})
    for px in (128, 192, 224):
        RoboEmotConfig(vision={"attribute_input_px": px})
    with pytest.raises(ValueError):
        RoboEmotConfig(vision={"attribute_precision": "int8"})
    # Both are copied into the estimator and the worker at start, so a hot edit must say restart.
    fields = type(VISION).model_fields
    for name in ("attribute_input_px", "attribute_score_min"):
        assert fields[name].json_schema_extra["restart"] is True


# ------------------------------------------------------------- aggregation with the score floor


def test_one_crop_above_the_floor_decides_and_below_is_a_possible_minor():
    cfg = IdentityConfig()
    adult = estimate_age_group([("adult", 0.6)], cfg, confidence_min=0.5)
    assert adult.value == "adult" and not adult.possible_minor
    weak = estimate_age_group([("unknown", 0.4)], cfg, confidence_min=0.5)
    assert weak.value == "unknown" and weak.possible_minor
    # The 0.8 rule still applies when no floor is given (gender, other estimators).
    assert estimate_age_group([("adult", 0.6)], cfg).value == "unknown"


def test_median_over_crops_still_leans_to_unknown():
    cfg = IdentityConfig(estimate_frames_min=3)
    frames = [("adult", 0.9), ("unknown", 0.3), ("unknown", 0.3)]
    assert estimate_age_group(frames, cfg, confidence_min=0.5).value == "unknown"
    frames = [("adult", 0.9), ("adult", 0.6), ("unknown", 0.3)]
    assert estimate_age_group(frames, cfg, confidence_min=0.5).value == "adult"


class Fixed:
    def __init__(self, sample: AttributeSample):
        self.sample = sample

    def estimate(self, crop):
        return self.sample


def run_worker(sample: AttributeSample):
    worker = AttributeWorker(
        Fixed(sample), IdentityConfig(), 2, InlineExecutor(), age_confidence_min=0.5
    )
    assert worker.submit(1, 1, [(0.0, np.zeros((4, 4, 3), np.uint8))])
    (result,) = worker.poll()
    return result


def test_worker_uses_the_age_floor_and_gender_stays_unknown():
    result = run_worker(AttributeSample("adult", 0.6, "unknown", 0.0))
    assert result.age.value == "adult" and result.age.source == "estimated"
    assert result.gender.value == "unknown" and result.gender.source == "none"
    assert run_worker(AttributeSample("unknown", 0.3, "unknown", 0.0)).age.possible_minor


def test_pipeline_publishes_unknown_gender_and_a_declared_one_still_wins(tmp_path):
    from test_vision import CONSENT, FACE_MODEL, axis, face, rig, run

    est = Fixed(AttributeSample("adult", 0.9, "unknown", 0.0))
    _, _, pipe, _, got = rig(tmp_path / "a", lambda i: [face(100)], estimator=est)
    run(pipe, 20)
    last = got["percept.face"][-1].payload
    assert (last.age_group, last.gender, last.gender_source) == ("adult", "unknown", "none")
    reg, _, pipe, _, got = rig(tmp_path / "b", lambda i: [face(100)], estimator=est)
    faces = np.stack([axis(0)] * 5)
    reg.enroll("Ada", faces, FACE_MODEL, consent=CONSENT, age_group="adult", gender="female")
    run(pipe, 40)
    last = got["percept.face"][-1].payload
    assert (last.gender, last.gender_source) == ("female", "declared")
    assert (last.age_group, last.age_source) == ("adult", "declared")


# ------------------------------------------------------------- estimator on a synthetic ONNX


def synthetic_model(path, scores, *, names=None, stride="32"):
    """A one-node ONNX whose output is the fixed ``1 x 7 x anchors`` tensor of ``scores``."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    value = output(scores)[None]
    const = helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(value, "v"))
    graph = helper.make_graph(
        [const],
        "age",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, "h", "w"])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(value.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 7
    names = names if names is not None else "{0: '0-14', 1: '15-22', 2: '22+'}"
    for key, val in {"names": names, "stride": stride}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, val
    onnx.save(model, str(path))
    return path


def estimator(path, **kw):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision.onnx import YoloAgeEstimator

    args = {"input_px": 160, "score_min": 0.5, "threads": 1, **kw}
    return YoloAgeEstimator(path, LABELS, GROUPS, **args)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([[0.1, 0.2, 0.8], [0.0, 0.0, 0.1]], ("adult", 0.8)),
        ([[0.0, 0.7, 0.1], [0.2, 0.0, 0.0]], ("teen", 0.7)),
        ([[0.95, 0.0, 0.3]], ("child", 0.95)),
        ([[0.1, 0.2, 0.45]], ("unknown", 0.45)),
    ],
)
def test_estimator_maps_classes_to_groups_with_its_own_floor(tmp_path, scores, expected):
    est = estimator(synthetic_model(tmp_path / "m.onnx", scores))
    sample = est.estimate(np.full((90, 70, 3), 128, np.uint8))
    assert (sample.age_group, sample.age_confidence) == (expected[0], pytest.approx(expected[1]))
    assert (sample.gender, sample.gender_confidence) == ("unknown", 0.0)


def test_estimator_checks_the_file_metadata(tmp_path):
    scores = [[0.1, 0.2, 0.8]]
    wrong = synthetic_model(tmp_path / "a.onnx", scores, names="{0: '22+', 1: '15-22', 2: '0-14'}")
    with pytest.raises(ModelError, match="labels"):
        estimator(wrong)
    evil = synthetic_model(tmp_path / "b.onnx", scores, names="__import__('os')")
    with pytest.raises(ModelError, match="labels"):
        estimator(evil)
    good = synthetic_model(tmp_path / "c.onnx", scores)
    with pytest.raises(ModelError, match="stride"):
        estimator(good, input_px=150)
    with pytest.raises(ModelError, match="age group"):
        pytest.importorskip("onnxruntime")
        from roboemot.vision.onnx import YoloAgeEstimator

        YoloAgeEstimator(good, LABELS, {**GROUPS, "22+": "grown-up"}, 160, 0.5, 1)


# ------------------------------------------------------------- the real file (opt in)


@pytest.mark.models
@pytest.mark.parametrize("input_px", [128, 160, 224])
def test_real_age_model_runs_on_a_crop(input_px):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    root = CACHE
    spec = load_manifest()["yolo11n-face-age"]
    if models.status(spec, root) != "ok":
        pytest.skip(f"yolo11n-face-age missing in {root} (roboemot models fetch yolo11n-face-age)")
    cfg = RoboEmotConfig(vision={"attribute_input_px": input_px}).vision
    stages = onnx.load_stages(root, cfg, appearance=False, objects=False)
    crop = (np.random.default_rng(0).random((150, 120, 3)) * 255).astype(np.uint8)
    sample = stages.attribute_estimator.estimate(crop)
    assert sample.age_group in {"child", "teen", "adult", "unknown"}
    assert 0 <= sample.age_confidence <= 1
    assert (sample.gender, sample.gender_confidence) == ("unknown", 0.0)
    assert (sample.age_group == "unknown") == (sample.age_confidence < cfg.attribute_score_min)
