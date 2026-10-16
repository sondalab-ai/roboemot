"""Model manifest, fetcher and CLI (SPEC 5, 13), plus the optional real-model tests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from roboemot import cli
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision import models
from roboemot.vision.models import (
    ModelError,
    ModelFile,
    fetch,
    fetch_file,
    load_manifest,
    sha256_of,
)

BODY = b"onnx-bytes" * 100
GOOD = ModelFile(
    path="sub/model.onnx",
    url="https://example.test/m.onnx",
    sha256=hashlib.sha256(BODY).hexdigest(),
)


def client(body: bytes = BODY, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(status, content=body))
    )


def test_manifest_is_complete_and_pinned():
    specs = load_manifest()
    assert set(specs) == {
        "yunet",
        "sface",
        "fairface-age",
        "fairface-gender",
        "fairface-age-int8",
        "fairface-gender-int8",
        "dinov2-small",
    }
    paths = [f.path for s in specs.values() for f in s.files]
    assert len(paths) == len(set(paths))
    for spec in specs.values():
        assert spec.licence
        for f in spec.files:
            assert not f.path.startswith("/")
            if f.quantize is None:
                assert re.fullmatch(r"[0-9a-f]{64}", f.sha256)
                assert f.url.startswith("https://")
            else:
                assert f.sha256 is None and f.url is None
    for fp32, int8 in (
        ("fairface-age", "fairface-age-int8"),
        ("fairface-gender", "fairface-gender-int8"),
    ):
        assert specs[int8].labels == specs[fp32].labels
        assert specs[int8].files[0].quantize.source == specs[fp32].files[0].path


SOURCE_BODY = b"fp32-weights" * 50
SOURCE = ModelFile(
    path="fp/m.onnx",
    url="https://example.test/m.onnx",
    sha256=hashlib.sha256(SOURCE_BODY).hexdigest(),
)
DERIVED = ModelFile(path="q/m.onnx", quantize={"source": "fp/m.onnx", "weight_type": "QInt8"})


@pytest.fixture
def derived_manifest(monkeypatch):
    def spec(name, f):
        return models.ModelSpec(name=name, role="r", licence="x", source="s", files=[f])

    only = {"src": spec("src", SOURCE), "q": spec("q", DERIVED)}
    monkeypatch.setattr(models, "load_manifest", lambda path=models.MANIFEST: only)
    return only


class FakeQuantizer:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, source: Path, dest: Path, weight_type: str) -> None:
        self.calls.append((source.name, weight_type))
        dest.write_bytes(b"int8:" + source.read_bytes()[:20])


def produced(tmp_path, quant=None):
    quant = quant or FakeQuantizer()
    fetch(["q"], tmp_path, client=client(SOURCE_BODY), quantizer=quant)
    return quant


def failing(source, dest, weight_type):
    dest.write_bytes(b"half")
    raise ModelError("quantizer failed")


def test_derived_file_is_produced_from_the_verified_source(tmp_path, derived_manifest):
    quant = FakeQuantizer()
    report = fetch(["q"], tmp_path, client=client(SOURCE_BODY), quantizer=quant)
    assert report.files == {"fp/m.onnx": "downloaded", "q/m.onnx": "produced"}
    assert quant.calls == [("m.onnx", "QInt8")]
    target = tmp_path / "q/m.onnx"
    marker = json.loads((tmp_path / "q/m.onnx.verified").read_text())
    assert marker["source_sha256"] == SOURCE.sha256 and marker["weight_type"] == "QInt8"
    assert marker["sha256"] == sha256_of(target) and marker["size"] == target.stat().st_size
    assert marker["mtime_ns"] == target.stat().st_mtime_ns and marker["ort_version"]
    assert models.check_file(tmp_path, DERIVED) == target
    assert models.status(derived_manifest["q"], tmp_path, verify=True) == "ok"
    assert not list(tmp_path.rglob("*.part"))
    assert (target.stat().st_mode & 0o777) == 0o600
    again = fetch(["q"], tmp_path, client=client(b"x"), quantizer=quant)
    assert again.files["q/m.onnx"] == "present" and len(quant.calls) == 1


def test_fetch_downloads_before_producing_and_hashes_a_shared_source_once(tmp_path, monkeypatch):
    other = ModelFile(path="q/n.onnx", quantize={"source": "fp/m.onnx", "weight_type": "QInt8"})
    plain = ModelFile(path="p.onnx", url="https://example.test/p", sha256=SOURCE.sha256)
    spec = models.ModelSpec(
        name="s", role="r", licence="x", source="s", files=[DERIVED, other, SOURCE, plain]
    )
    monkeypatch.setattr(models, "load_manifest", lambda path=models.MANIFEST: {"s": spec})
    order: list[str] = []
    real_fetch_file = models.fetch_file

    def spy(f, root, http):
        order.append("get " + f.path)
        return real_fetch_file(f, root, http)

    monkeypatch.setattr(models, "fetch_file", spy)
    fake = FakeQuantizer()

    def quantizer(src, dst, weight_type):
        order.append("quantize " + dst.name)
        fake(src, dst, weight_type)

    fetch(["s"], tmp_path, client=client(SOURCE_BODY), quantizer=quantizer)
    assert order.count("get fp/m.onnx") == 1
    last_get = max(i for i, x in enumerate(order) if x.startswith("get"))
    assert min(i for i, x in enumerate(order) if x.startswith("quantize")) > last_get
    assert len(fake.calls) == 2


def test_derived_file_loads_without_its_source_after_prune(tmp_path, derived_manifest):
    produced(tmp_path)
    assert models.prune_sources(tmp_path) == [tmp_path / "fp/m.onnx"]
    assert not (tmp_path / "fp/m.onnx").exists()
    assert not (tmp_path / "fp/m.onnx.verified").exists()
    assert models.check_file(tmp_path, DERIVED) == tmp_path / "q/m.onnx"
    assert models.status(derived_manifest["q"], tmp_path, verify=True) == "ok"
    assert models.prune_sources(tmp_path) == []
    report = fetch(["q"], tmp_path, client=client(b"unused"), quantizer=failing)
    assert report.files["q/m.onnx"] == "present"


def test_prune_keeps_a_source_whose_derived_file_is_not_verified(tmp_path, derived_manifest):
    produced(tmp_path)
    (tmp_path / "q/m.onnx").write_bytes(b"tampered")
    assert models.prune_sources(tmp_path) == []
    assert (tmp_path / "fp/m.onnx").exists()


def test_cli_prune_needs_sources_flag(tmp_path, derived_manifest, capsys):
    produced(tmp_path)
    assert cli.main(["models", "prune", "--cache-dir", str(tmp_path)]) == 1
    assert (tmp_path / "fp/m.onnx").exists()
    assert cli.main(["models", "prune", "--sources", "--cache-dir", str(tmp_path)]) == 0
    assert not (tmp_path / "fp/m.onnx").exists()
    assert "1 source file(s) deleted" in capsys.readouterr().out


def test_derived_marker_must_match_the_manifest_and_the_file(tmp_path, derived_manifest):
    produced(tmp_path)
    target = tmp_path / "q/m.onnx"
    marker_path = tmp_path / "q/m.onnx.verified"
    good = marker_path.read_text()

    def tamper(**changes):
        marker_path.write_text(json.dumps({**json.loads(good), **changes}))
        with pytest.raises(ModelError, match="changed since"):
            models.check_file(tmp_path, DERIVED)

    tamper(source_sha256="0" * 64)
    tamper(weight_type="QUInt8")
    tamper(size=1)
    tamper(sha256=None)
    marker_path.write_text(good)
    assert models.check_file(tmp_path, DERIVED) == target

    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ModelError, match="changed since"):
        models.check_file(tmp_path, DERIVED)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    target.write_bytes(b"swapped!")
    with pytest.raises(ModelError, match="changed since"):
        models.check_file(tmp_path, DERIVED)


def test_ort_version_policy_stale_loads_requantize_rebuilds(
    tmp_path, derived_manifest, monkeypatch
):
    quant = produced(tmp_path)
    monkeypatch.setattr(models, "ort_version", lambda: "99.0.0")
    assert models.check_file(tmp_path, DERIVED)
    assert models.status(derived_manifest["q"], tmp_path) == "stale-ort"
    report = fetch(["q"], tmp_path, client=client(SOURCE_BODY), quantizer=quant)
    assert report.files["q/m.onnx"] == "present" and len(quant.calls) == 1
    report = fetch(["q"], tmp_path, client=client(SOURCE_BODY), quantizer=quant, requantize=True)
    assert report.files["q/m.onnx"] == "produced" and len(quant.calls) == 2
    assert models.status(derived_manifest["q"], tmp_path) == "ok"
    assert json.loads((tmp_path / "q/m.onnx.verified").read_text())["ort_version"] == "99.0.0"


def test_failed_requantization_keeps_the_old_file_and_marker(tmp_path, derived_manifest):
    produced(tmp_path)
    target = tmp_path / "q/m.onnx"
    before = (target.read_bytes(), (tmp_path / "q/m.onnx.verified").read_text())
    with pytest.raises(ModelError, match="quantizer failed"):
        fetch(["q"], tmp_path, client=client(SOURCE_BODY), quantizer=failing, requantize=True)
    assert (target.read_bytes(), (tmp_path / "q/m.onnx.verified").read_text()) == before
    assert models.check_file(tmp_path, DERIVED) == target
    assert not list(tmp_path.rglob("*.part"))


def test_failed_first_quantization_leaves_no_file(tmp_path, derived_manifest):
    with pytest.raises(ModelError, match="quantizer failed"):
        fetch(["q"], tmp_path, client=client(SOURCE_BODY), quantizer=failing)
    assert not (tmp_path / "q/m.onnx").exists() and not list(tmp_path.rglob("*.part"))


def test_quantize_dynamic_file_passes_the_weight_type(monkeypatch):
    qmod = pytest.importorskip("onnxruntime.quantization")
    seen = {}
    monkeypatch.setattr(
        qmod, "quantize_dynamic", lambda src, dst, **kw: seen.update(src=src, dst=dst, **kw)
    )
    models.quantize_dynamic_file(Path("a.onnx"), Path("b.onnx"), "QInt8")
    assert seen["weight_type"] == qmod.QuantType.QInt8
    assert (seen["src"], seen["dst"]) == (Path("a.onnx"), Path("b.onnx"))
    models.quantize_dynamic_file(Path("a.onnx"), Path("b.onnx"), "QUInt8")
    assert seen["weight_type"] == qmod.QuantType.QUInt8


INT8 = {"source": "b", "weight_type": "QInt8"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"url": "https://x"},
        {"sha256": "0" * 64},
        {"url": "https://x", "sha256": "0" * 64, "quantize": INT8},
        {"url": "https://x", "quantize": INT8},
        {"size_bytes": 5, "quantize": INT8},
    ],
)
def test_manifest_file_needs_exactly_one_origin(kwargs):
    with pytest.raises(ValueError):
        ModelFile(path="a.onnx", **kwargs)


def test_quantize_weight_type_is_required_and_restricted():
    with pytest.raises(ValueError):
        ModelFile(path="a.onnx", quantize={"source": "b"})
    with pytest.raises(ValueError):
        ModelFile(path="a.onnx", quantize={"source": "b", "weight_type": "int4"})


@pytest.mark.parametrize(
    ("configured", "profile", "expected"),
    [
        ("auto", "cm4", "int8"),
        ("auto", "npu", "fp32"),
        ("auto", None, "fp32"),
        ("int8", "npu", "int8"),
        ("fp32", "cm4", "fp32"),
    ],
)
def test_resolve_precision(configured, profile, expected):
    assert models.resolve_precision(configured, profile) == expected


def test_auto_precision_follows_the_manifest_profile(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.manifest import load_manifest as load_robot
    from roboemot.vision import onnx

    asked: list[str] = []
    monkeypatch.setattr(onnx, "configure", lambda threads: None)
    monkeypatch.setattr(onnx, "_file", lambda root, name, index=0: asked.append(name) or tmp_path)
    monkeypatch.setattr(onnx, "FairFaceEstimator", lambda *a: "estimator")
    monkeypatch.setattr(onnx, "YuNetDetector", lambda *a: None)
    monkeypatch.setattr(onnx, "SFaceEmbedder", lambda *a: None)
    cfg = RoboEmotConfig().vision
    assert cfg.attribute_precision == "auto"
    manifest_dir = Path(__file__).resolve().parents[3] / "configs" / "manifests"
    cm4 = load_robot(manifest_dir / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)
    assert cm4.robot.profile == "cm4"
    onnx.load_stages(tmp_path, cfg, profile=cm4.robot.profile, appearance=False)
    assert "fairface-age-int8" in asked and "fairface-gender-int8" in asked
    asked.clear()
    onnx.load_stages(tmp_path, cfg, profile="npu", appearance=False)
    assert "fairface-age" in asked and not [n for n in asked if n.endswith("int8")]
    asked.clear()
    onnx.load_stages(tmp_path, cfg, appearance=False)
    assert "fairface-age" in asked
    asked.clear()
    forced = cfg.model_copy(update={"attribute_precision": "int8"})
    onnx.load_stages(tmp_path, forced, profile="npu", appearance=False)
    assert "fairface-age-int8" in asked
    with pytest.raises(ValueError):
        RoboEmotConfig(vision={"attribute_precision": "int4"})


def test_load_stages_int8_pair_uses_the_real_marker_check(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    specs = load_manifest()
    for name in ("yunet", "sface"):
        f = specs[name].files[0]
        target = tmp_path / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        models._write_marker(target, f.sha256)
    for name in ("fairface-age-int8", "fairface-gender-int8"):
        f = specs[name].files[0]
        target = tmp_path / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"int8")
        models._write_marker(
            target,
            hashlib.sha256(b"int8").hexdigest(),
            {
                "source_sha256": models._source_of(f).sha256,
                "weight_type": "QInt8",
                "ort_version": "1.0",
            },
        )
    opened: list[str] = []
    monkeypatch.setattr(onnx, "configure", lambda threads: None)
    monkeypatch.setattr(onnx, "_session", lambda path, threads: opened.append(path.name))
    monkeypatch.setattr(onnx, "YuNetDetector", lambda *a: None)
    monkeypatch.setattr(onnx, "SFaceEmbedder", lambda *a: None)
    cfg = RoboEmotConfig().vision
    stages = onnx.load_stages(tmp_path, cfg, profile="cm4", appearance=False)
    assert stages.attribute_estimator is not None
    assert opened == ["age_model.onnx", "gender_model.onnx"]
    (tmp_path / "fairface_int8/age_model.onnx").write_bytes(b"swapped")
    with pytest.raises(ModelError, match="changed since"):
        onnx.load_stages(tmp_path, cfg, profile="cm4", appearance=False)
    with pytest.raises(ModelError, match="not verified"):
        onnx.load_stages(tmp_path, cfg, profile="npu", appearance=False)


def test_fetch_downloads_verifies_and_is_idempotent(tmp_path):
    assert fetch_file(GOOD, tmp_path, client()) == "downloaded"
    assert (tmp_path / "sub/model.onnx").read_bytes() == BODY
    assert not list(tmp_path.rglob("*.part"))
    assert fetch_file(GOOD, tmp_path, client(b"changed upstream")) == "present"


def test_fetch_discards_a_download_with_the_wrong_digest(tmp_path):
    with pytest.raises(ModelError, match="sha256 mismatch"):
        fetch_file(GOOD, tmp_path, client(b"tampered"))
    assert list(tmp_path.rglob("*")) == [tmp_path / "sub"]


def test_fetch_replaces_a_corrupt_file_only_after_verification(tmp_path):
    target = tmp_path / "sub/model.onnx"
    target.parent.mkdir()
    target.write_bytes(b"corrupt")
    with pytest.raises(ModelError):
        fetch_file(GOOD, tmp_path, client(b"tampered"))
    assert target.read_bytes() == b"corrupt"
    assert fetch_file(GOOD, tmp_path, client()) == "downloaded"
    assert target.read_bytes() == BODY


def test_fetch_refuses_http_traversal_and_http_errors(tmp_path):
    with pytest.raises(ModelError, match="https"):
        fetch_file(
            GOOD.model_copy(update={"url": "http://example.test/m.onnx"}), tmp_path, client()
        )
    with pytest.raises(ModelError, match="leaves"):
        fetch_file(GOOD.model_copy(update={"path": "../escape.onnx"}), tmp_path, client())
    with pytest.raises(ModelError, match="download failed"):
        fetch_file(GOOD, tmp_path, client(status=404))
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_unknown_name_and_status(tmp_path):
    with pytest.raises(ModelError, match="unknown"):
        fetch(["nope"], tmp_path)
    spec = load_manifest()["yunet"]
    assert models.status(spec, tmp_path) == "missing"
    target = tmp_path / spec.files[0].path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert models.status(spec, tmp_path, verify=True) == "corrupt"


def test_models_dir_default_and_config(tmp_path, monkeypatch):
    monkeypatch.delenv(models.ENV_DIR, raising=False)
    monkeypatch.delenv("ROBOEMOT_DATA_DIR", raising=False)
    assert models.models_dir("") == Path.home() / ".local" / "share" / "roboemot" / "models"
    monkeypatch.setenv(models.ENV_DIR, str(tmp_path))
    assert models.models_dir("") == tmp_path
    assert models.models_dir("/x/y") == Path("/x/y")


def test_cli_models_list_and_fetch_needs_names(tmp_path, capsys):
    assert cli.main(["models", "list", "--cache-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "yunet" in out and "missing" in out
    assert cli.main(["models", "fetch", "--cache-dir", str(tmp_path)]) == 1
    assert cli.main(["models", "fetch", "nope", "--cache-dir", str(tmp_path)]) == 1


def test_core_imports_do_not_need_the_vision_extra():
    code = (
        "import sys; sys.modules['cv2'] = None; sys.modules['onnxruntime'] = None; "
        "import roboemot.vision, roboemot.vision.replay, roboemot.vision.models, roboemot.cli"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------- real models (opt in)

CACHE = models.default_dir()
NEEDED = ["yunet", "sface", "fairface-age-int8", "fairface-gender-int8", "dinov2-small"]


@pytest.fixture(scope="module")
def stages():
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    specs = load_manifest()
    if any(models.status(specs[n], CACHE) != "ok" for n in NEEDED):
        pytest.skip(f"model files missing or corrupt in {CACHE} (roboemot models fetch --all)")
    from roboemot.vision import onnx

    return onnx.load_stages(CACHE, RoboEmotConfig().vision, profile="cm4")


def textured_scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((480, 640, 3)) * 255).astype(np.uint8)


@pytest.mark.models
def test_real_backends_run_end_to_end_on_a_synthetic_image(stages):
    img = textured_scene()
    assert stages.face_detector.detect(img) == []
    crop = img[100:300, 100:300]
    sample = stages.attribute_estimator.estimate(crop)
    assert sample.age_group in {"child", "teen", "adult", "senior"}
    assert 0 < sample.age_confidence <= 1 and sample.gender in {"female", "male"}
    app = stages.appearance_embedder.embed(img, (100.0, 100.0, 300.0, 300.0))
    assert app.shape == (384,) and np.linalg.norm(app) == pytest.approx(1.0, abs=1e-4)
    same = stages.appearance_embedder.embed(img, (100.0, 100.0, 300.0, 300.0))
    other = stages.appearance_embedder.embed(img, (300.0, 200.0, 600.0, 450.0))
    assert float(app @ same) > float(app @ other)


@pytest.mark.models
def test_real_face_pipeline_on_a_face_image(stages):
    path = os.environ.get("ROBOEMOT_TEST_FACE")
    if not path:
        pytest.skip("set ROBOEMOT_TEST_FACE to a portrait image to run the face path")
    import cv2

    img = cv2.imread(path)
    faces = stages.face_detector.detect(img)
    assert faces and faces[0].landmarks.shape == (5, 2)
    start = time.perf_counter()
    e1 = stages.face_embedder.embed(img, faces[0])
    e2 = stages.face_embedder.embed(
        cv2.flip(img, 1), stages.face_detector.detect(cv2.flip(img, 1))[0]
    )
    assert e1.shape == (128,) and float(e1 @ e2) > 0.45
    assert time.perf_counter() - start < 5
