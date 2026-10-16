"""Model manifest, fetcher and CLI (SPEC 5, 13), plus the optional real-model tests."""

from __future__ import annotations

import hashlib
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
from roboemot.vision.models import ModelError, ModelFile, fetch, fetch_file, load_manifest

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
    assert set(specs) == {"yunet", "sface", "fairface-age", "fairface-gender", "dinov2-small"}
    paths = [f.path for s in specs.values() for f in s.files]
    assert len(paths) == len(set(paths))
    for spec in specs.values():
        assert spec.licence
        for f in spec.files:
            assert re.fullmatch(r"[0-9a-f]{64}", f.sha256)
            assert f.url.startswith("https://") and not f.path.startswith("/")


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
    assert models.models_dir("") == Path.home() / ".cache" / "roboemot" / "models"
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
NEEDED = ["yunet", "sface", "fairface-age", "fairface-gender", "dinov2-small"]


@pytest.fixture(scope="module")
def stages():
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    specs = load_manifest()
    if any(models.status(specs[n], CACHE) != "ok" for n in NEEDED):
        pytest.skip(f"model files missing or corrupt in {CACHE} (roboemot models fetch --all)")
    from roboemot.vision import onnx

    return onnx.load_stages(CACHE, RoboEmotConfig().vision)


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
