"""Catalogue import without photo decoding, confirmation and listings (SPEC 6, 5A, 12).

Nothing here needs OpenCV, so it runs in the plain CI job; the tests that decode photos are in
``test_catalogue_photos.py``.
"""

from __future__ import annotations

import struct
import sys
import threading
import zlib
from datetime import UTC, datetime

import numpy as np
import pytest
from roboemot.config.schema import (
    CatalogueConfig,
    EnrollmentConfig,
    IdentityConfig,
    WorldConfig,
)
from roboemot.registry import Consent, FileStore, Registry
from roboemot.registry import images as images_module
from roboemot.registry.catalogue import CatalogueError, CatalogueService
from roboemot.registry.enrollment import EnrollmentService
from roboemot.registry.images import ImageError, decode_bgr, image_size
from roboemot.world import Observation, WorldModel
from test_xmp import ATTR_FORM, subjects

MODEL = "fake-dino"
DIM = 8
PEOPLE_ONLY = subjects("People|Anna")


class FakeEmbedder:
    model_id = MODEL

    def __init__(self) -> None:
        self.boxes: list[tuple] = []

    def embed(self, image, box):
        self.boxes.append(box)
        v = np.zeros(DIM, np.float32)
        v[len(self.boxes) % DIM] = 1.0
        return v


def png_bytes(w=40, h=30) -> bytes:
    """A real, tiny PNG built by hand (no imaging library needed)."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body)
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x7f" * (w * 3) for _ in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def jpeg_header(w=40, h=30) -> bytes:
    """SOI, an APP0 segment and an SOF0 with the size: all the header parser reads."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def make_registry(tmp_path):
    return Registry(FileStore(tmp_path / "reg", allow_plaintext=True), IdentityConfig())


def make_service(tmp_path, embedder="default", world=None, **cfg):
    reg = make_registry(tmp_path)
    emb = FakeEmbedder() if embedder == "default" else embedder
    svc = CatalogueService(
        reg, CatalogueConfig(**cfg), embedder=lambda: emb, world=world, now=lambda: 100.0
    )
    return svc, reg, emb


# ------------------------------------------------------------------ image headers


def test_header_sizes():
    assert image_size(jpeg_header(40, 30)) == (40, 30)
    assert image_size(png_bytes(50, 20)) == (50, 20)


def test_png_bomb_is_refused_by_header_before_decoding():
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">II", 60000, 60000) + b"\x08\x02\x00\x00\x00"
    bomb = sig + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"
    with pytest.raises(ImageError) as exc:
        decode_bgr(bomb, max_bytes=10**6, max_pixels=25_000_000)
    assert exc.value.code == "image_too_many_pixels"


def test_jpeg_bomb_is_refused_by_header():
    with pytest.raises(ImageError) as exc:
        decode_bgr(jpeg_header(65000, 65000), max_bytes=10**6, max_pixels=25_000_000)
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
        decode_bgr(png_bytes(), max_bytes=10, max_pixels=10**6)
    assert exc.value.code == "image_too_large"


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


def test_bad_sidecar_is_a_clean_error(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    for payload, code in (
        (b"nope", "xmp_not_xml"),
        (b"<!DOCTYPE a [<!ENTITY b 'c'>]><a>&b;</a>", "xmp_forbidden_xml"),
        (b"<a>" + b"x" * 300000 + b"</a>", "xmp_too_large"),
        ("<a>\ud800</a>", "xmp_not_xml"),  # a lone surrogate cannot be encoded
    ):
        with pytest.raises(CatalogueError) as exc:
            svc.import_sidecar(payload)
        assert exc.value.code == code
    assert reg.list() == []


def test_without_embedder_records_are_made_and_photo_is_reported_skipped(tmp_path):
    svc, reg, _ = make_service(tmp_path, embedder=None)
    report = svc.import_sidecar(ATTR_FORM, png_bytes())
    assert report.photo == "no_embedder" and len(report.items) == 2
    assert all(len(e.embeddings) == 0 for e in reg.entities())


def test_a_people_only_sidecar_never_decodes_the_photo(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the photo was decoded")

    monkeypatch.setattr("roboemot.registry.catalogue.decode_bgr", boom)
    svc, reg, emb = make_service(tmp_path)
    report = svc.import_sidecar(PEOPLE_ONLY, png_bytes())
    assert report.photo == "no_regions" and emb.boxes == []
    assert [r.kind for r in reg.list()] == ["person"]


def test_a_sidecar_without_regions_does_not_decode_either(tmp_path, monkeypatch):
    monkeypatch.setattr("roboemot.registry.catalogue.decode_bgr", lambda *a, **k: 1 / 0)
    svc, _, emb = make_service(tmp_path)
    report = svc.import_sidecar(subjects("Animals|Cat|Mia"), png_bytes())
    assert report.photo == "no_regions" and emb.boxes == []


def test_opencv_missing_still_imports_the_records(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # `import cv2` now raises ImportError
    svc, reg, emb = make_service(tmp_path)
    report = svc.import_sidecar(ATTR_FORM, png_bytes(400, 300))
    assert report.photo == "no_embedder" and emb.boxes == []
    assert {i.display_name for i in report.items} == {"Mia", "Anna"}
    assert len(reg.list()) == 2
    with pytest.raises(ImageError) as exc:
        images_module.decode_bgr(png_bytes(), max_bytes=10**6, max_pixels=10**6)
    assert exc.value.code == "image_unsupported"


def test_only_one_import_runs_at_a_time(tmp_path):
    inside, release = threading.Event(), threading.Event()

    class Slow(FakeEmbedder):
        def embed(self, image, box):
            inside.set()
            release.wait(10)
            return super().embed(image, box)

    svc, _, _ = make_service(tmp_path, embedder=Slow())
    monkey_pixels = np.zeros((300, 400, 3), np.uint8)
    svc._open_photo = lambda *a, **k: _fake_photo(monkey_pixels)  # skip decoding
    first = threading.Thread(target=svc.import_sidecar, args=(ATTR_FORM, b"x"))
    first.start()
    try:
        assert inside.wait(10)
        with pytest.raises(CatalogueError) as exc:
            svc.import_sidecar(subjects("Animals|Cat|Rex"))
        assert exc.value.code == "import_busy"
    finally:
        release.set()
        first.join(10)
    assert svc.import_sidecar(subjects("Animals|Cat|Rex")).items[0].created


def _fake_photo(pixels):
    from roboemot.registry.catalogue import _Photo

    return _Photo(pixels, "abc", float("inf"))


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
    svc.import_sidecar(subjects("Animals|Cat|Mia"))
    reg.enroll_entity("animal", "cat 1", np.eye(DIM)[:3], MODEL, category="cat", origin="auto")
    reg.enroll_entity("animal", "Rex", np.eye(DIM)[:3], MODEL, category="dog")
    names = {s.display_name for s in svc.list_entities()}
    assert names == {"Mia", "cat 1"}  # pending or auto only
    assert all(not hasattr(s, "embeddings") for s in svc.list_entities())
    (live,) = svc.live_entries()
    assert (live.kind, live.category, live.identified) == ("animal", "cat", False)


def test_deleting_an_entity_tells_the_history_links(tmp_path):
    reg = make_registry(tmp_path)
    forgotten: list[str] = []

    class Link:
        def export(self, owner_id):
            return None

        def forget(self, owner_id):
            forgotten.append(owner_id)

    reg.attach_history(Link())
    entity = reg.create_pending_entity("animal", "Mia")
    reg.delete_entity(entity.entity_id)
    assert forgotten == [entity.entity_id]


def test_pending_embeddings_need_one_key_per_row_and_a_pending_record(tmp_path):
    reg = make_registry(tmp_path)
    entity = reg.create_pending_entity("animal", "Mia")
    with pytest.raises(ValueError):
        reg.attach_pending_embeddings(entity.entity_id, np.eye(2, DIM), MODEL, 5, ["only-one"])
    assert (
        reg.attach_pending_embeddings(entity.entity_id, np.eye(2, DIM), MODEL, 5, ["a", "b"]) == 2
    )
    assert (
        reg.attach_pending_embeddings(entity.entity_id, np.eye(2, DIM), MODEL, 5, ["a", "c"]) == 1
    )
    assert reg.get_entity(entity.entity_id).photo_keys == ["a", "b", "c"]
    reg.confirm_entity(entity.entity_id, np.eye(1, DIM), MODEL, keep_existing=True)
    with pytest.raises(ValueError):
        reg.attach_pending_embeddings(entity.entity_id, np.eye(1, DIM), MODEL, 5, ["z"])


# ------------------------------------------------------------------ tombstones (SPEC 6, 12)


def enrollment_for(reg):
    return EnrollmentService(reg, EnrollmentConfig(), IdentityConfig())


def test_a_deleted_pending_entity_does_not_come_back_on_reimport(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    xmp = subjects("Animals|Cat|Mia", "Objects|Cup|Mug")
    first = svc.import_sidecar(xmp)
    assert [i.created for i in first.items] == [True, True] and len(reg.list()) == 2
    enrollment = enrollment_for(reg)
    for record in reg.list():
        enrollment.delete(record.id)
    assert reg.list() == []

    again = svc.import_sidecar(xmp)
    assert again.items == [] and again.skipped == {"deleted": 2}
    assert reg.list() == []
    assert svc.tombstone_count() == 2

    assert svc.clear_tombstones() == 2 and svc.tombstone_count() == 0
    assert len(svc.import_sidecar(xmp).items) == 2  # the owner allowed it again


def test_the_tombstone_is_keyed_by_category_so_another_luna_still_imports(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(subjects("Animals|Cat|Luna"))
    enrollment_for(reg).delete(reg.list()[0].id)
    dog = svc.import_sidecar(subjects("Animals|Dog|Luna"))
    assert [i.created for i in dog.items] == [True]
    assert svc.import_sidecar(subjects("Animals|Cat|Luna")).skipped == {"deleted": 1}


def test_tombstones_hold_no_name_and_survive_a_restart(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(subjects("Animals|Cat|Zelda"))
    enrollment_for(reg).delete(reg.list()[0].id)
    files = [p for p in (tmp_path / "reg").rglob("*") if p.is_file()]
    blob = b"".join(p.read_bytes() + p.name.encode() for p in files)
    assert b"elda" not in blob
    reopened = make_registry(tmp_path)
    assert reopened.tombstones.has("animal", "cat", "ZELDA")  # case-insensitive, like the import
    assert not reopened.tombstones.has("animal", "dog", "Zelda")


def test_forgetting_a_person_leaves_no_tombstone_and_the_name_can_be_imported(tmp_path):
    """Erasure leaves nothing (SPEC 12): people are never tombstoned, pending or enrolled."""
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(PEOPLE_ONLY)
    enrollment_for(reg).delete(reg.list()[0].id)
    assert svc.tombstone_count() == 0
    assert [i.created for i in svc.import_sidecar(PEOPLE_ONLY).items] == [True]
    consent = Consent("Carl", datetime(2026, 1, 1, tzinfo=UTC), ("face",))
    vec = [[1.0] + [0.01 * i] * (DIM - 1) for i in range(5)]
    active = reg.enroll("Carl", vec, MODEL, consent=consent, age_group="adult")
    enrollment_for(reg).delete(active.person_id)
    assert reg.tombstones.count() == 0


def test_the_tombstone_is_written_before_the_record_is_deleted(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(subjects("Animals|Cat|Mia"))
    order = []
    real_delete = reg.delete_entity
    reg.delete_entity = lambda eid: (order.append(reg.tombstones.count()), real_delete(eid))
    enrollment_for(reg).delete(reg.list()[0].id)
    assert order == [1]


def test_a_corrupt_salt_does_not_stop_the_registry_and_drops_the_orphans(tmp_path):
    svc, reg, _ = make_service(tmp_path)
    svc.import_sidecar(subjects("Animals|Cat|Mia"))
    enrollment_for(reg).delete(reg.list()[0].id)
    (tmp_path / "reg" / "tombstone_salt" / "salt.json").write_bytes(b"\x01{not json")
    reopened = make_registry(tmp_path)  # must not raise
    assert reopened.tombstones.count() == 0
    assert not reopened.tombstones.has("animal", "cat", "Mia")
    reopened.tombstones.add("animal", "cat", "Mia")  # and it works again with a new salt
    assert reopened.tombstones.has("animal", "cat", "Mia")
