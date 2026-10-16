"""Catalogue photos: decoding, region embeddings, idempotence, budget (SPEC 6, 12 Data).

Needs OpenCV (the ``vision`` extra); the CI job with that extra runs it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from roboemot.config.schema import CatalogueConfig
from roboemot.registry.catalogue import CatalogueError, CatalogueService
from roboemot.registry.images import ImageError, decode_bgr
from test_catalogue import (
    DIM,
    MODEL,
    FakeEmbedder,
    live_world,
    make_registry,
    make_service,
)
from test_xmp import ATTR_FORM, HEAD, TAIL, subjects

cv2 = pytest.importorskip("cv2")


def jpeg(w=400, h=300) -> bytes:
    return cv2.imencode(".jpg", np.full((h, w, 3), 127, np.uint8))[1].tobytes()


def png(w=400, h=300) -> bytes:
    return cv2.imencode(".png", np.full((h, w, 3), 90, np.uint8))[1].tobytes()


def three_regions() -> str:
    areas = "".join(
        f'<rdf:li><rdf:Description mwg-rs:Name="Mia" mwg-rs:Type="Pet"><mwg-rs:Area '
        f'stArea:x="{x}" stArea:y="0.5" stArea:w="0.2" stArea:h="0.3" stArea:unit="normalized"/>'
        f"</rdf:Description></rdf:li>"
        for x in (0.2, 0.5, 0.8)
    )
    return (
        HEAD
        + "<rdf:Description><mwg-rs:Regions rdf:parseType='Resource'>"
        + "<mwg-rs:RegionList><rdf:Bag>"
        + areas
        + "</rdf:Bag></mwg-rs:RegionList></mwg-rs:Regions></rdf:Description>"
        + TAIL
    )


def test_decode_valid_and_corrupt():
    assert decode_bgr(png(50, 20), max_bytes=10**6, max_pixels=10**6).shape == (20, 50, 3)
    assert decode_bgr(jpeg(50, 20), max_bytes=10**6, max_pixels=10**6).shape == (20, 50, 3)
    data = bytearray(png())
    data[40:80] = b"\x00" * 40
    with pytest.raises(ImageError):
        decode_bgr(bytes(data), max_bytes=10**6, max_pixels=10**6)


def test_photo_embeds_animal_regions_only_and_keeps_them_pending(tmp_path):
    svc, reg, emb = make_service(tmp_path)
    report = svc.import_sidecar(ATTR_FORM, jpeg(400, 300))
    assert report.photo == "used"
    assert len(emb.boxes) == 1  # the Face region (Anna) is never embedded
    assert emb.boxes[0] == pytest.approx((160.0, 75.0, 240.0, 165.0))  # centre (0.5, 0.4)
    mia = next(i for i in report.items if i.display_name == "Mia")
    assert mia.embeddings == 1
    record = reg.get_entity(mia.id)
    assert record.status == "pending" and len(record.embeddings) == 1
    assert record.embedding_model_id == MODEL
    assert reg.gallery("entity", MODEL) == {}  # inert until the owner confirms
    anna = next(i for i in report.items if i.display_name == "Anna")
    assert reg.get_person(anna.id).face_embeddings.size == 0


def test_photo_never_reaches_disk(tmp_path):
    svc, _, _ = make_service(tmp_path)
    image = png(400, 300)
    svc.import_sidecar(ATTR_FORM, image)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.suffix not in (".png", ".jpg", ".jpeg")
            assert image[:16] not in path.read_bytes()


def test_bad_photo_rejects_the_whole_import(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    with pytest.raises(CatalogueError) as exc:
        svc.import_sidecar(ATTR_FORM, b"GIF89a....")
    assert exc.value.code == "image_format"
    assert reg.list() == []


def test_photo_with_other_aspect_ratio_than_the_regions_is_skipped(tmp_path):
    svc, reg, emb = make_service(tmp_path)
    report = svc.import_sidecar(ATTR_FORM, jpeg(300, 300))  # sidecar says 4000x3000
    assert report.photo == "dims_mismatch" and emb.boxes == []
    assert len(report.items) == 2


def test_small_regions_are_skipped(tmp_path):
    svc, _, emb = make_service(tmp_path, min_region_px=500)
    report = svc.import_sidecar(ATTR_FORM, jpeg())
    assert emb.boxes == [] and report.skipped.get("region_small") == 1


def test_embedder_failure_is_counted_not_raised(tmp_path):
    class Boom(FakeEmbedder):
        def embed(self, image, box):
            raise RuntimeError("secret detail")

    svc, _, _ = make_service(tmp_path, embedder=Boom())
    report = svc.import_sidecar(ATTR_FORM, jpeg())
    assert report.skipped.get("embed_failed") == 1 and len(report.items) == 2


def test_embeddings_per_entity_are_capped(tmp_path):
    svc, reg, emb = make_service(tmp_path, max_embeddings_per_entity=2)
    first = svc.import_sidecar(three_regions(), jpeg(400, 300))
    assert len(reg.entities()[0].embeddings) == 2 and len(emb.boxes) == 2
    assert first.items[0].embeddings == 2 and first.skipped.get("cap") == 1


def test_repeating_an_import_does_not_duplicate_rows(tmp_path):
    svc, reg, emb = make_service(tmp_path)
    photo = png(400, 300)
    first = svc.import_sidecar(ATTR_FORM, photo)
    second = svc.import_sidecar(ATTR_FORM, photo)
    assert first.items[0].embeddings == 1 and second.items[0].embeddings == 0
    assert len(emb.boxes) == 1  # the second run did not even embed again
    (mia,) = [e for e in reg.entities() if e.display_name == "Mia"]
    assert len(mia.embeddings) == 1
    other = svc.import_sidecar(ATTR_FORM, png(800, 600))  # another photo: new rows
    assert other.items[0].embeddings == 1
    assert len(reg.get_entity(mia.entity_id).embeddings) == 2


def test_the_budget_stops_embedding_but_the_records_are_kept_and_a_retry_finishes(tmp_path):
    now = [0.0]

    class Slow(FakeEmbedder):
        def embed(self, image, box):
            now[0] += 5.0
            return super().embed(image, box)

    reg = make_registry(tmp_path)
    svc = CatalogueService(
        reg,
        CatalogueConfig(import_budget_s=8.0),
        embedder=lambda: Slow(),
        clock=lambda: now[0],
    )
    photo = png(400, 300)
    first = svc.import_sidecar(three_regions(), photo)
    assert first.photo == "partial" and first.skipped.get("budget") == 1
    (mia,) = reg.entities()
    assert first.items[0].created and len(mia.embeddings) == 2
    now[0] = 100.0
    second = svc.import_sidecar(three_regions(), photo)
    assert second.photo == "used" and second.items[0].embeddings == 1
    assert len(reg.get_entity(mia.entity_id).embeddings) == 3  # no duplicates


def test_photo_rows_pool_with_the_live_one_on_confirm(tmp_path):
    world, entry_id = live_world()
    svc, reg, _ = make_service(tmp_path, world=world)
    svc.import_sidecar(ATTR_FORM, jpeg())
    mia = next(e for e in reg.entities() if e.display_name == "Mia")
    record = svc.confirm(mia.entity_id, entry_id)
    assert len(record.embeddings) == 2 and record.category == "cat"


def test_activate_from_photo_rows_without_an_entry(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(ATTR_FORM, jpeg())
    mia = next(e for e in reg.entities() if e.display_name == "Mia")
    assert svc.confirm(mia.entity_id).status == "active"


def test_listing_shows_photo_rows_exist_without_exposing_them(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.import_sidecar(ATTR_FORM, jpeg())
    (summary,) = svc.list_entities()
    assert summary.has_appearance and not hasattr(summary, "embeddings")
    assert subjects("Animals|Cat|X") and DIM


# ------------------------------------------------------------------ real DINOv2 (opt in)

BENCH_MODELS = Path(os.environ.get("ROBOEMOT_BENCH_MODELS", "~/.cache/roboemot-bench/models"))


@pytest.fixture(scope="module")
def dino():
    pytest.importorskip("onnxruntime")
    from roboemot.vision import models, onnx

    spec = models.load_manifest()["dinov2-small"]
    for root in (BENCH_MODELS.expanduser(), models.default_dir()):
        if models.status(spec, root) == "ok":
            return onnx.DinoV2Embedder(models.check_file(root, spec.files[0]), threads=2)
    pytest.skip("DINOv2-S is not in the model cache (roboemot models fetch dinov2-small)")


@pytest.mark.models
def test_real_dinov2_embeds_photo_regions_and_keeps_them_pending(tmp_path, dino):
    rng = np.random.default_rng(3)
    scene = (rng.random((300, 400, 3)) * 255).astype(np.uint8)
    photo = cv2.imencode(".png", scene)[1].tobytes()
    reg = make_registry(tmp_path)
    svc = CatalogueService(reg, CatalogueConfig(), embedder=lambda: dino)
    report = svc.import_sidecar(ATTR_FORM, photo)
    assert report.photo == "used"
    mia = next(e for e in reg.entities() if e.display_name == "Mia")
    assert mia.status == "pending" and mia.embedding_model_id == dino.model_id
    assert mia.embeddings.shape == (1, 384)
    assert float(np.linalg.norm(mia.embeddings[0])) == pytest.approx(1.0, abs=1e-4)
    same = dino.embed(scene, (160.0, 75.0, 240.0, 165.0))
    other = dino.embed(scene, (0.0, 150.0, 120.0, 300.0))
    assert float(mia.embeddings[0] @ same) > 0.999 > float(mia.embeddings[0] @ other)
    assert reg.gallery("entity", dino.model_id) == {}
    done = svc.confirm(mia.entity_id)
    assert done.status == "active" and mia.entity_id in reg.gallery("entity", dino.model_id)
