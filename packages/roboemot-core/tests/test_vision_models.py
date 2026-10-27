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
        "dinov2-small",
        "nanodet-plus-m-320",
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
    assert not [n for n in specs if "fairface" in n]
    retired = models.retired_paths()
    assert sorted(retired) == [
        "fairface/age_model.onnx",
        "fairface/gender_model.onnx",
        "fairface_int8/age_model.onnx",
        "fairface_int8/gender_model.onnx",
        "yolo11n_face_age/model.onnx",
    ]
    assert not set(retired) & set(paths)
    assert not [n for n in specs if "age" in n.split("-")]


def test_prune_deletes_only_the_retired_files_and_their_empty_folders(tmp_path):
    for rel in [*models.retired_paths(), "fairface/notes.txt", "clip_b32/model.onnx"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        (tmp_path / (rel + ".verified")).write_text("{}")
    deleted = models.prune_retired(tmp_path)
    assert sorted(deleted) == sorted(tmp_path / r for r in models.retired_paths())
    assert not (tmp_path / "fairface_int8").exists()  # emptied, so removed
    assert not (tmp_path / "yolo11n_face_age").exists()  # the age model (2026-10-23) too
    assert (tmp_path / "fairface/notes.txt").exists()  # not listed: kept, folder kept
    assert not (tmp_path / "fairface/age_model.onnx.verified").exists()
    assert (tmp_path / "clip_b32/model.onnx").exists()
    assert models.prune_retired(tmp_path) == []
    assert models.prune_retired(tmp_path / "absent") == []


def test_cli_list_names_retired_files_on_disk_and_prune_deletes_them(tmp_path, capsys):
    """Review ZN S2: `models list` shows each retired file still in the folder (the age model
    included) and points at `models prune`, which then deletes it."""
    age = tmp_path / "yolo11n_face_age/model.onnx"
    age.parent.mkdir()
    age.write_bytes(b"x" * 2_000_000)
    assert models.retired_on_disk(tmp_path) == [(age, 2_000_000)]
    assert cli.main(["models", "list", "--cache-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"retired, unused: {age} (2.0 MB); run `roboemot models prune`" in out
    assert cli.main(["models", "prune", "--cache-dir", str(tmp_path)]) == 0
    assert str(age) in capsys.readouterr().out and not age.exists()
    assert cli.main(["models", "list", "--cache-dir", str(tmp_path)]) == 0
    assert "retired, unused" not in capsys.readouterr().out


def test_prune_does_not_follow_a_symlink_out_of_the_folder(tmp_path):
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"keep")
    root = tmp_path / "models"
    (root / "fairface").mkdir(parents=True)
    (root / "fairface/age_model.onnx").symlink_to(outside)
    assert models.prune_retired(root) == []
    assert outside.read_bytes() == b"keep"


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


def test_cli_prune_deletes_retired_files_and_sources_only_on_request(
    tmp_path, derived_manifest, capsys, monkeypatch
):
    produced(tmp_path)
    old = tmp_path / "fairface/age_model.onnx"
    old.parent.mkdir()
    old.write_bytes(b"old")
    retired = ["fairface/age_model.onnx"]
    monkeypatch.setattr(models, "retired_paths", lambda path=models.MANIFEST: retired)
    assert cli.main(["models", "prune", "--cache-dir", str(tmp_path)]) == 0
    assert not old.exists() and (tmp_path / "fp/m.onnx").exists()
    assert "1 file(s) deleted" in capsys.readouterr().out
    assert cli.main(["models", "prune", "--sources", "--cache-dir", str(tmp_path)]) == 0
    assert not (tmp_path / "fp/m.onnx").exists()


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


def test_nanodet_manifest_pins_the_class_order_and_covers_the_default_mapping():
    spec = load_manifest()["nanodet-plus-m-320"]
    cfg = RoboEmotConfig().vision
    assert spec.role == "object_detector" and spec.licence == "Apache-2.0"
    assert len(spec.labels) == 80 and len(set(spec.labels)) == 80
    assert spec.labels[:3] == ["person", "bicycle", "car"] and spec.labels[-1] == "toothbrush"
    assert re.fullmatch(r"[0-9a-f]{64}", spec.labels_sha256) and spec.labels_source.startswith(
        "https://"
    )
    assert set(cfg.animal_categories + cfg.object_categories) <= set(spec.labels)


def test_load_stages_builds_the_object_detector_from_the_class_mapping(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    built: list[tuple] = []
    monkeypatch.setattr(onnx, "configure", lambda threads: None)
    monkeypatch.setattr(onnx, "_file", lambda root, name, index=0: tmp_path / name)
    monkeypatch.setattr(onnx, "YuNetDetector", lambda *a: None)
    monkeypatch.setattr(onnx, "SFaceEmbedder", lambda *a: None)
    monkeypatch.setattr(onnx, "NanoDetDetector", lambda *a: built.append(a) or "detector")
    cfg = RoboEmotConfig().vision
    stages = onnx.load_stages(tmp_path, cfg, appearance=False)
    path, labels, classes, score_min, nms_iou, threads = built[0]
    assert stages.object_detector == "detector" and path.name == "nanodet-plus-m-320"
    assert len(labels) == 80 and classes == [*cfg.animal_categories, *cfg.object_categories]
    assert (score_min, nms_iou, threads) == (cfg.object_score_min, cfg.object_nms_iou, cfg.threads)
    built.clear()
    off = cfg.model_copy(update={"objects_enabled": False})
    assert onnx.load_stages(tmp_path, off, appearance=False).object_detector is None
    assert onnx.load_stages(tmp_path, cfg, appearance=False, objects=False).object_detector is None
    assert built == []


def _optional_stage_rig(monkeypatch, tmp_path, missing):
    from roboemot.vision import onnx

    built: list[str] = []

    def file(root, name, index=0):
        if name == missing:
            raise ModelError(f"{name}: not verified in {root}; run `roboemot models fetch`")
        return tmp_path / name

    monkeypatch.setattr(onnx, "configure", lambda threads: None)
    monkeypatch.setattr(onnx, "_file", file)
    monkeypatch.setattr(onnx, "YuNetDetector", lambda *a: "yunet")
    monkeypatch.setattr(onnx, "SFaceEmbedder", lambda *a: "sface")
    monkeypatch.setattr(onnx, "NanoDetDetector", lambda *a: built.append("nanodet") or "nanodet")
    monkeypatch.setattr(onnx, "DinoV2Embedder", lambda *a: built.append("dino") or "dino")
    return onnx, built


def test_missing_dinov2_keeps_faces_and_drops_the_object_stage(monkeypatch, tmp_path, caplog):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    onnx, built = _optional_stage_rig(monkeypatch, tmp_path, "dinov2-small")
    stages = onnx.load_stages(tmp_path, RoboEmotConfig().vision)
    assert stages.face_detector == "yunet" and stages.face_embedder == "sface"
    assert stages.appearance_embedder is None and stages.object_detector is None
    assert (
        "ModelError" in stages.objects_unavailable and "dinov2-small" in stages.objects_unavailable
    )
    assert "unavailable" in caplog.text


def test_a_failing_detector_drops_dinov2_too_and_never_loads_it(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    onnx, built = _optional_stage_rig(monkeypatch, tmp_path, "nanodet-plus-m-320")
    stages = onnx.load_stages(tmp_path, RoboEmotConfig().vision)
    assert stages.face_detector == "yunet"
    assert stages.appearance_embedder is None and stages.object_detector is None
    assert "nanodet-plus-m-320" in stages.objects_unavailable
    assert "dino" not in built


def test_the_object_stage_loads_whole_or_not_at_all_and_can_be_off(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    onnx, built = _optional_stage_rig(monkeypatch, tmp_path, None)
    cfg = RoboEmotConfig().vision
    ok = onnx.load_stages(tmp_path, cfg)
    assert (ok.object_detector, ok.appearance_embedder, ok.objects_unavailable) == (
        "nanodet",
        "dino",
        None,
    )
    built.clear()
    off = cfg.model_copy(update={"objects_enabled": False})
    stages = onnx.load_stages(tmp_path, off, appearance=False)
    assert stages.object_detector is None and stages.appearance_embedder is None and built == []


def test_load_stages_needs_only_the_face_stack_and_uses_the_real_marker_check(
    monkeypatch, tmp_path
):
    """No age model since 2026-10-23: YuNet and SFace alone make the required stages."""
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    specs = load_manifest()
    built: list[tuple] = []
    for name in ("yunet", "sface"):
        f = specs[name].files[0]
        target = tmp_path / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        models._write_marker(target, f.sha256)
    monkeypatch.setattr(onnx, "configure", lambda threads: None)
    monkeypatch.setattr(onnx, "YuNetDetector", lambda *a: built.append(a) or "detector")
    monkeypatch.setattr(onnx, "SFaceEmbedder", lambda *a: "embedder")
    cfg = RoboEmotConfig().vision
    stages = onnx.load_stages(tmp_path, cfg, profile="cm4", appearance=False, objects=False)
    assert (stages.face_detector, stages.face_embedder) == ("detector", "embedder")
    assert not hasattr(stages, "attribute_estimator")
    assert built[0][0] == tmp_path / specs["yunet"].files[0].path
    (tmp_path / specs["sface"].files[0].path).write_bytes(b"swapped")
    with pytest.raises(ModelError, match="changed since"):
        onnx.load_stages(tmp_path, cfg, appearance=False, objects=False)


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


class Flaky:
    """Serves ``BODY``, dropping the connection after ``cut`` bytes for the first ``drops`` GETs."""

    def __init__(self, drops: int, cut: int = 300, honour_range: bool = True, offset_bug: int = 0):
        self.drops, self.cut, self.honour_range, self.offset_bug = (
            drops,
            cut,
            honour_range,
            offset_bug,
        )
        self.ranges: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.ranges.append(request.headers.get("range"))
        start = 0
        if request.headers.get("range") and self.honour_range:
            start = int(request.headers["range"].removeprefix("bytes=").rstrip("-"))
        body = BODY[start:]
        status, headers = 200, {}
        if start:
            status = 206
            shown = start + self.offset_bug
            headers = {"content-range": f"bytes {shown}-{len(BODY) - 1}/{len(BODY)}"}
        if len(self.ranges) <= self.drops:
            return httpx.Response(status, headers=headers, content=self._cut(body))
        return httpx.Response(status, headers=headers, content=body)

    def _cut(self, body: bytes):
        yield body[: self.cut]
        raise httpx.RemoteProtocolError("peer closed connection")


@pytest.fixture
def waits(monkeypatch):
    """Records retry waits; small chunks so bytes before a drop reach the file."""
    monkeypatch.setattr(models, "CHUNK_BYTES", 100)
    slept: list[float] = []
    monkeypatch.setattr(models.time, "sleep", slept.append)
    return slept


def flaky(server: Flaky) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server))


def test_fetch_resumes_a_dropped_download_from_the_bytes_written(tmp_path, waits):
    server = Flaky(drops=2)
    assert fetch_file(GOOD, tmp_path, flaky(server)) == "downloaded"
    assert (tmp_path / "sub/model.onnx").read_bytes() == BODY
    assert server.ranges == [None, "bytes=300-", "bytes=600-"]
    assert waits == [models.RETRY_WAIT_S, models.RETRY_WAIT_S * 2]
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_restarts_when_the_server_ignores_the_range(tmp_path, waits):
    sized = GOOD.model_copy(update={"size_bytes": len(BODY)})
    server = Flaky(drops=1, honour_range=False)
    assert fetch_file(sized, tmp_path, flaky(server)) == "downloaded"
    assert (tmp_path / "sub/model.onnx").read_bytes() == BODY


def test_fetch_gives_up_after_the_last_attempt_and_leaves_no_orphans(tmp_path, waits):
    server = Flaky(drops=models.DOWNLOAD_ATTEMPTS)
    with pytest.raises(ModelError, match=r"download failed \(RemoteProtocolError\)"):
        fetch_file(GOOD, tmp_path, flaky(server))
    assert len(server.ranges) == models.DOWNLOAD_ATTEMPTS
    assert len(waits) == models.DOWNLOAD_ATTEMPTS - 1
    assert list(tmp_path.rglob("*")) == [tmp_path / "sub"]


def test_fetch_refuses_a_resume_at_the_wrong_offset(tmp_path, waits):
    with pytest.raises(ModelError, match="wrong offset"):
        fetch_file(GOOD, tmp_path, flaky(Flaky(drops=1, offset_bug=1)))
    assert list(tmp_path.rglob("*")) == [tmp_path / "sub"]


def test_a_resume_resends_the_range_on_every_redirect_hop(tmp_path, waits):
    origin = Flaky(drops=1)
    hops: list[tuple[str, str | None]] = []

    def serve(request: httpx.Request) -> httpx.Response:
        hops.append((request.url.host, request.headers.get("range")))
        if request.url.host == "example.test":  # the manifest host redirects to a CDN
            return httpx.Response(302, headers={"location": "https://cdn.example.test/m.onnx"})
        return origin(request)

    client = httpx.Client(transport=httpx.MockTransport(serve))
    assert fetch_file(GOOD, tmp_path, client) == "downloaded"
    assert (tmp_path / "sub/model.onnx").read_bytes() == BODY
    assert hops == [
        ("example.test", None),
        ("cdn.example.test", None),
        ("example.test", "bytes=300-"),
        ("cdn.example.test", "bytes=300-"),
    ]


def test_a_drop_after_the_last_byte_is_not_resumed_into_a_416(tmp_path, waits):
    sized = GOOD.model_copy(update={"size_bytes": len(BODY)})

    class AtEof(Flaky):
        def __call__(self, request):
            if request.headers.get("range") == f"bytes={len(BODY)}-":
                self.ranges.append(request.headers["range"])
                return httpx.Response(416)  # a compliant server: nothing left to send
            return super().__call__(request)

    server = AtEof(drops=1, cut=len(BODY))  # the whole body arrived, then the close failed
    assert fetch_file(sized, tmp_path, flaky(server)) == "downloaded"
    assert (tmp_path / "sub/model.onnx").read_bytes() == BODY
    assert server.ranges == [None]  # the digest decides; no resume was asked


def test_fetch_does_not_retry_http_errors(tmp_path, waits):
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda req: seen.append(req) or httpx.Response(503))
    with pytest.raises(ModelError, match="download failed"):
        fetch_file(GOOD, tmp_path, httpx.Client(transport=transport))
    assert len(seen) == 1 and waits == []


def test_fetch_replaces_a_part_file_left_by_a_killed_run(tmp_path):
    part = tmp_path / "sub/model.onnx.part"
    part.parent.mkdir()
    part.write_bytes(b"left over")
    assert fetch_file(GOOD, tmp_path, client()) == "downloaded"
    assert not part.exists()


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
NEEDED = [
    "yunet",
    "sface",
    "dinov2-small",
    "nanodet-plus-m-320",
]


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
