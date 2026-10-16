"""Catalogue import, photos and confirmation (SPEC 6, Catalogue import; SPEC 12 Data)."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest
from roboemot.config.schema import CatalogueConfig, IdentityConfig, WorldConfig
from roboemot.registry import FileStore, Registry
from roboemot.registry.catalogue import CatalogueError, CatalogueService
from roboemot.registry.images import ImageError, decode_bgr, image_size
from roboemot.world import Observation, WorldModel
from test_xmp import ATTR_FORM, HEAD, TAIL, subjects

cv2 = pytest.importorskip("cv2")

MODEL = "fake-dino"
DIM = 8


class FakeEmbedder:
    model_id = MODEL

    def __init__(self) -> None:
        self.boxes: list[tuple] = []

    def embed(self, image, box):
        self.boxes.append(box)
        v = np.zeros(DIM, np.float32)
        v[len(self.boxes) % DIM] = 1.0
        return v


def jpeg(w=400, h=300) -> bytes:
    img = np.full((h, w, 3), 127, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def png(w=400, h=300) -> bytes:
    return cv2.imencode(".png", np.full((h, w, 3), 90, np.uint8))[1].tobytes()


def make_registry(tmp_path):
    return Registry(FileStore(tmp_path / "reg", allow_plaintext=True), IdentityConfig())


def make_service(tmp_path, embedder="default", world=None, **cfg):
    reg = make_registry(tmp_path)
    emb = FakeEmbedder() if embedder == "default" else embedder
    svc = CatalogueService(
        reg, CatalogueConfig(**cfg), embedder=lambda: emb, world=world, now=lambda: 100.0
    )
    return svc, reg, emb


# ------------------------------------------------------------------ images


def test_images_decode_and_report_size():
    assert image_size(jpeg(40, 30)) == (40, 30)
    assert image_size(png(50, 20)) == (50, 20)
    img = decode_bgr(png(50, 20), max_bytes=10**6, max_pixels=10**6)
    assert img.shape == (20, 50, 3)


def test_png_bomb_is_refused_by_header_before_decoding():
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">II", 60000, 60000) + b"\x08\x02\x00\x00\x00"
    bomb = sig + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"
    with pytest.raises(ImageError) as exc:
        decode_bgr(bomb, max_bytes=10**6, max_pixels=25_000_000)
    assert exc.value.code == "image_too_many_pixels"


def test_jpeg_bomb_is_refused_by_header():
    data = bytearray(jpeg())
    sof = data.index(b"\xff\xc0")
    data[sof + 5 : sof + 9] = struct.pack(">HH", 65000, 65000)
    with pytest.raises(ImageError) as exc:
        decode_bgr(bytes(data), max_bytes=10**6, max_pixels=25_000_000)
    assert exc.value.code == "image_too_many_pixels"


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"GIF89a" + b"\x00" * 50, "image_format"),
        (b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image_format"),
        (b"\xff\xd8\xff\xe0\x00\x02", "image_corrupt"),
        (b"\x89PNG\r\n\x1a\n", "image_corrupt"),
    ],
)
def test_other_formats_and_truncated_headers_are_refused(data, code):
    with pytest.raises(ImageError) as exc:
        decode_bgr(data, max_bytes=10**6, max_pixels=10**6)
    assert exc.value.code == code


def test_oversized_bytes_are_refused():
    with pytest.raises(ImageError) as exc:
        decode_bgr(jpeg(), max_bytes=100, max_pixels=10**6)
    assert exc.value.code == "image_too_large"


def test_corrupt_body_with_good_header_is_refused():
    data = bytearray(png())
    data[40:80] = b"\x00" * 40
    with pytest.raises(ImageError):
        decode_bgr(bytes(data), max_bytes=10**6, max_pixels=10**6)


# ------------------------------------------------------------------ import


def test_import_creates_pending_records_without_biometrics(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    report = svc.import_sidecar(ATTR_FORM)
    by = {i.display_name: i for i in report.items}
    assert set(by) == {"Mia", "Anna"} and all(i.created for i in by.values())
    anna = reg.get_person(by["Anna"].id)
    assert anna.status == "pending" and anna.face_embeddings.size == 0
    assert anna.consent is None and anna.embedding_model_id is None
    mia = reg.get_entity(by["Mia"].id)
    assert (mia.status, mia.origin, mia.kind) == ("pending", "xmp", "animal")
    assert reg.gallery("entity", MODEL) == {} and reg.gallery("face", MODEL) == {}


def test_reimport_does_not_duplicate_and_ignores_case(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(subjects("Animals|Cat|Mia", "People|Anna"))
    again = svc.import_sidecar(subjects("Animals|Cat|MIA", "People|anna"))
    assert [i.created for i in again.items] == [False, False]
    assert len(reg.list()) == 2


def test_category_less_existing_record_matches_later_category(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(ATTR_FORM)  # Mia, no category
    out = svc.import_sidecar(subjects("Animals|Cat|Mia"))
    assert not out.items[0].created and len(reg.entities()) == 1


def test_photo_embeds_animal_regions_only_and_keeps_them_pending(tmp_path):
    svc, reg, emb = make_service(tmp_path)
    report = svc.import_sidecar(ATTR_FORM, jpeg(400, 300))
    assert report.photo == "used"
    assert len(emb.boxes) == 1  # the Face region (Anna) is never embedded
    x1, y1, x2, y2 = emb.boxes[0]
    # centre (0.5, 0.4), size (0.2, 0.3) on 400x300
    assert (x1, y1, x2, y2) == pytest.approx((160.0, 75.0, 240.0, 165.0))
    mia = next(i for i in report.items if i.display_name == "Mia")
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


def test_bad_sidecar_is_a_clean_error(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    for payload, code in (
        (b"nope", "xmp_not_xml"),
        (b"<!DOCTYPE a [<!ENTITY b 'c'>]><a>&b;</a>", "xmp_forbidden_xml"),
        (b"<a>" + b"x" * 300000 + b"</a>", "xmp_too_large"),
    ):
        with pytest.raises(CatalogueError) as exc:
            svc.import_sidecar(payload)
        assert exc.value.code == code
    assert reg.list() == []


def test_without_embedder_records_are_made_and_photo_is_reported_skipped(tmp_path):
    svc, reg, _ = make_service(tmp_path, embedder=None)
    report = svc.import_sidecar(ATTR_FORM, jpeg())
    assert report.photo == "no_embedder" and len(report.items) == 2
    assert all(len(e.embeddings) == 0 for e in reg.entities())


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

    svc, reg, _ = make_service(tmp_path, embedder=Boom())
    report = svc.import_sidecar(ATTR_FORM, jpeg())
    assert report.skipped.get("embed_failed") == 1 and len(report.items) == 2


def test_embeddings_per_entity_are_capped(tmp_path):
    xml = ATTR_FORM
    svc, reg, _ = make_service(tmp_path, max_embeddings_per_entity=2)
    for _ in range(4):
        svc.import_sidecar(xml, jpeg())
    mia = next(e for e in reg.entities() if e.display_name == "Mia")
    assert len(mia.embeddings) == 2


# ------------------------------------------------------------------ confirmation


def live_world(category="cat", vec=None):
    world = WorldModel(WorldConfig())
    emb = np.eye(DIM, dtype=np.float32)[3] if vec is None else vec
    result = world.observe(
        Observation("animal", 1.0, "vision", 0.0, 0.0, category=category, embedding=emb)
    )
    return world, result.entry.entry_id


def test_confirm_from_live_entry_activates_and_attaches_identity(tmp_path):
    world, entry_id = live_world()
    svc, reg, _ = make_service(tmp_path, world=world)
    svc.import_sidecar(subjects("Animals|Cat|Mia"))
    (mia,) = reg.entities()
    record = svc.confirm(mia.entity_id, entry_id)
    assert record.status == "active" and record.embedding_model_id == MODEL
    assert len(record.embeddings) == 1
    assert world.get(entry_id).identity == mia.entity_id
    assert mia.entity_id in reg.gallery("entity", MODEL)


def test_confirm_pools_photo_rows_with_the_live_one(tmp_path):
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


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        ("gone", "entry_gone"),
        ("dog", "category_mismatch"),
        ("person", "kind_mismatch"),
        ("active", "not_pending"),
    ],
)
def test_confirm_refusals(tmp_path, setup, code):
    world, entry_id = live_world("dog" if setup == "dog" else "cat")
    svc, reg, _ = make_service(tmp_path, world=world)
    svc.import_sidecar(subjects("Animals|Cat|Mia"))
    (mia,) = reg.entities()
    target = entry_id
    if setup == "gone":
        target = "animal-999"
    if setup == "person":
        reg.delete_entity(mia.entity_id)
        mia = reg.create_pending_entity("object", "Mug", category="cat")
    if setup == "active":
        svc.confirm(mia.entity_id, entry_id)
    with pytest.raises(CatalogueError) as exc:
        svc.confirm(mia.entity_id, target)
    assert exc.value.code == code


def test_confirm_without_any_appearance_is_refused(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(subjects("Animals|Cat|Mia"))
    (mia,) = reg.entities()
    with pytest.raises(CatalogueError) as exc:
        svc.confirm(mia.entity_id)
    assert exc.value.code == "no_appearance"
    with pytest.raises(KeyError):
        svc.confirm("nope")


def test_listings_carry_no_embeddings(tmp_path):
    world, _ = live_world()
    svc, reg, _ = make_service(tmp_path, world=world)
    svc.import_sidecar(ATTR_FORM, jpeg())
    reg.enroll_entity("animal", "cat 1", np.eye(DIM)[:3], MODEL, category="cat", origin="auto")
    reg.enroll_entity("animal", "Rex", np.eye(DIM)[:3], MODEL, category="dog")
    names = {s.display_name for s in svc.list_entities()}
    assert names == {"Mia", "cat 1"}  # pending or auto only
    assert all(not hasattr(s, "embeddings") for s in svc.list_entities())
    (live,) = svc.live_entries()
    assert (live.kind, live.category, live.identified) == ("animal", "cat", False)
    assert HEAD and TAIL


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
