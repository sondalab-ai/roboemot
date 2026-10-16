"""Catalogue endpoints of the web UI: auth, CSRF, size caps, shapes, errors (SPEC 6, 12)."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import CatalogueConfig, IdentityConfig, RoboEmotConfig, WorldConfig
from roboemot.diagnostics import Diagnostics
from roboemot.registry import FileStore, Registry
from roboemot.registry.catalogue import CatalogueError, CatalogueService
from roboemot.webui import make_server, validation
from roboemot.webui.strings import STRINGS
from roboemot.world import Observation, WorldModel
from test_webui import PIN, Client, FakeService
from test_xmp import ATTR_FORM, subjects

WHEN = datetime(2026, 1, 1, tzinfo=UTC)


class FakeCatalogue:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail: Exception | None = None

    def import_sidecar(self, xmp, image=None):
        self.calls.append(("import", xmp, image))
        if self.fail:
            raise self.fail
        item = SimpleNamespace(
            id="e1",
            display_name="Mia",
            kind="animal",
            category="cat",
            created=True,
            embeddings=1,
            embedding=[0.1] * 64,
            path="/home/pollen/x.jpg",
        )
        return SimpleNamespace(
            items=[item],
            conflicts=[SimpleNamespace(name="Bob", code="unknown_category")],
            skipped={"region_small": 2},
            photo="used",
            truncated=False,
            image=b"\xff\xd8",
        )

    def list_entities(self):
        return [
            SimpleNamespace(
                id="e1",
                display_name="Mia",
                kind="animal",
                category="cat",
                status="pending",
                origin="xmp",
                has_appearance=True,
                last_seen=WHEN,
                embeddings=[[0.5] * 64],
            )
        ]

    def live_entries(self):
        return [
            SimpleNamespace(
                entry_id="animal-3",
                kind="animal",
                category="cat",
                age_s=4.26,
                identified=False,
                embedding=[0.2] * 64,
            )
        ]

    def confirm(self, record_id, entry_id=None):
        self.calls.append(("confirm", record_id, entry_id))
        if self.fail:
            raise self.fail

    def purge_candidates(self):
        self.calls.append(("purge",))
        return 3


@pytest.fixture
def catalogue():
    return FakeCatalogue()


def build(tmp_path, catalogue, clock=None, **web):
    cfg = RoboEmotConfig(
        webui={"bind": "127.0.0.1", "port": 0, "catalogue_max_body_bytes": 40000, **web},
        catalogue={"max_xmp_bytes": 4096, "max_image_bytes": 4096},
    )
    srv = make_server(
        FakeService(),
        Diagnostics(),
        cfg,
        tmp_path,
        catalogue=catalogue,
        **({"clock": clock} if clock else {}),
    )
    srv.set_pin(PIN)
    srv.start()
    return srv


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def server(tmp_path, catalogue, clock):
    srv = build(tmp_path, catalogue, clock)
    yield srv
    srv.stop()


@pytest.fixture
def anon(server):
    return Client(server.port)


@pytest.fixture
def authed(anon):
    assert anon.login().status == 200
    return anon


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


PROTECTED = [
    ("GET", "/api/catalogue"),
    ("POST", "/api/catalogue/import"),
    ("POST", "/api/catalogue/e1/confirm"),
    ("POST", "/api/catalogue/candidates/clear"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_routes_need_login(anon, method, path):
    assert anon.request(method, path, {} if method == "POST" else None, csrf=False).status == 401


@pytest.mark.parametrize(("method", "path"), PROTECTED[1:])
def test_posts_need_csrf(authed, method, path):
    resp = authed.request(method, path, {"xmp": "x"}, csrf=False)
    assert resp.status == 403 and resp.json == {"error": "csrf"}


def test_catalogue_unavailable_without_a_service(tmp_path):
    srv = build(tmp_path, None)
    try:
        client = Client(srv.port)
        client.login()
        assert client.request("GET", "/api/catalogue").status == 503
    finally:
        srv.stop()


def test_listing_is_whitelisted_and_free_of_vectors(authed):
    resp = authed.request("GET", "/api/catalogue")
    assert resp.status == 200
    assert resp.json["records"] == [
        {
            "id": "e1",
            "display_name": "Mia",
            "kind": "animal",
            "category": "cat",
            "status": "pending",
            "origin": "xmp",
            "has_appearance": True,
            "last_seen": WHEN.isoformat(),
        }
    ]
    assert resp.json["live"] == [
        {
            "entry_id": "animal-3",
            "kind": "animal",
            "category": "cat",
            "age_s": 4.3,
            "identified": False,
        }
    ]
    assert resp.json["limits"]["max_xmp_bytes"] == 4096
    assert b"0.5" not in resp.body and b"0.2" not in resp.body


def test_import_returns_only_names_and_counts(authed, catalogue):
    resp = authed.request(
        "POST", "/api/catalogue/import", {"xmp": "<x/>", "image_b64": b64(b"abc")}
    )
    assert resp.status == 200
    assert resp.json == {
        "items": [
            {
                "id": "e1",
                "display_name": "Mia",
                "kind": "animal",
                "category": "cat",
                "created": True,
                "embeddings": 1,
            }
        ],
        "conflicts": [{"name": "Bob", "code": "unknown_category"}],
        "skipped": {"region_small": 2},
        "photo": "used",
        "truncated": False,
    }
    assert catalogue.calls == [("import", "<x/>", b"abc")]


def test_import_without_photo(authed, catalogue):
    assert authed.request("POST", "/api/catalogue/import", {"xmp": "<x/>"}).status == 200
    assert catalogue.calls[-1] == ("import", "<x/>", None)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({}, "xmp_required"),
        ({"xmp": ""}, "xmp_required"),
        ({"xmp": 5}, "xmp_required"),
        ({"xmp": "x" * 5000}, "xmp_too_large"),
        ({"xmp": "<x/>", "image_b64": "!!!not base64!!!"}, "image_corrupt"),
        ({"xmp": "<x/>", "image_b64": 7}, "image_corrupt"),
        ({"xmp": "<x/>", "image_b64": "A" * 8000}, "image_too_large"),
        ({"xmp": "<x/>", "extra": 1}, "unknown_field"),
    ],
)
def test_import_validation(authed, catalogue, body, code):
    resp = authed.request("POST", "/api/catalogue/import", body)
    assert resp.status == 422 and resp.json["error"] == code
    assert catalogue.calls == []


def test_anonymous_bodies_stay_small_but_sessions_may_upload_photos(server, anon):
    big = b'{"xmp": "' + b"x" * 30000 + b'"}'
    resp = anon.request("POST", "/api/catalogue/import", raw=big, csrf=False)
    assert resp.status == 413  # no session: the default body limit applies
    anon.login()
    too_big = b'{"xmp": "x", "pad": "' + b"y" * 41000 + b'"}'
    assert anon.request("POST", "/api/catalogue/import", raw=too_big).status == 413
    ok = b'{"xmp": "' + b"x" * 3000 + b'", "image_b64": "' + b"A" * 4000 + b'"}'
    assert anon.request("POST", "/api/catalogue/import", raw=ok).status == 200


def test_other_posts_keep_the_small_limit_even_when_logged_in(authed):
    big = b'{"display_name": "' + b"x" * 30000 + b'"}'
    assert authed.request("POST", "/api/registry/e1/rename", raw=big).status == 413


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("import_busy", 503),
        ("xmp_not_xml", 422),
        ("xmp_forbidden_xml", 422),
        ("image_format", 422),
        ("image_too_many_pixels", 422),
        ("not_pending", 409),
        ("entry_gone", 409),
        ("no_appearance", 409),
        ("kind_mismatch", 422),
        ("category_mismatch", 422),
    ],
)
def test_refusals_map_to_stable_codes(authed, catalogue, code, status):
    catalogue.fail = CatalogueError(code)
    resp = authed.request("POST", "/api/catalogue/import", {"xmp": "<x/>"})
    assert resp.status == status and resp.json == {"error": code}
    resp = authed.request("POST", "/api/catalogue/e1/confirm", {"entry_id": "animal-3"})
    assert resp.status == status and resp.json == {"error": code}


def test_every_catalogue_error_has_text_in_both_languages():
    for code in (
        "xmp_not_xml",
        "xmp_forbidden_xml",
        "xmp_too_large",
        "xmp_required",
        "image_format",
        "image_corrupt",
        "image_too_large",
        "image_too_many_pixels",
        "image_unsupported",
        "import_busy",
        "pending_invalid",
        "not_pending",
        "entry_gone",
        "no_appearance",
        "kind_mismatch",
        "category_mismatch",
        "catalogue_unavailable",
    ):
        for lang in STRINGS:
            assert "err_" + code in STRINGS[lang], (lang, code)


def test_unexpected_failure_is_a_generic_500_without_detail(authed, catalogue):
    catalogue.fail = RuntimeError("/secret/path exploded")
    resp = authed.request("POST", "/api/catalogue/import", {"xmp": "<x/>"})
    assert resp.status == 500 and resp.json == {"error": "generic"}
    assert b"secret" not in resp.body


def test_confirm(authed, catalogue):
    assert (
        authed.request("POST", "/api/catalogue/e1/confirm", {"entry_id": "animal-3"}).status == 200
    )
    assert authed.request("POST", "/api/catalogue/e1/confirm", {}).status == 200
    assert catalogue.calls == [("confirm", "e1", "animal-3"), ("confirm", "e1", None)]
    assert authed.request("POST", "/api/catalogue/e1/confirm", {"entry_id": "../x"}).status == 422
    assert authed.request("POST", "/api/catalogue/e1/confirm", {"x": 1}).status == 422
    catalogue.fail = KeyError("e1")
    assert authed.request("POST", "/api/catalogue/e1/confirm", {}).status == 404


def test_clearing_candidates_needs_confirmation_and_the_pin(authed, catalogue, clock):
    path = "/api/catalogue/candidates/clear"
    assert authed.request("POST", path, {"pin": PIN}).status == 409
    assert authed.request("POST", path, {"confirm": True, "pin": "000000"}).status == 401
    assert catalogue.calls == []
    clock.now += 5
    resp = authed.request("POST", path, {"confirm": True, "pin": PIN})
    assert resp.status == 200 and resp.json == {"cleared": 3}


# ------------------------------------------------------------------ with the real service


@pytest.fixture
def real(tmp_path):
    registry = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), IdentityConfig())
    world = WorldModel(WorldConfig())
    world.observe(
        Observation(
            "animal", 1.0, "v", 0.0, 0.0, category="cat", embedding=np.eye(8, dtype=np.float32)[2]
        )
    )

    class Embedder:
        model_id = "fake"

        def embed(self, image, box):
            return np.eye(8, dtype=np.float32)[1]

    service = CatalogueService(
        registry,
        CatalogueConfig(max_xmp_bytes=4096),
        embedder=lambda: Embedder(),
        world=world,
        now=lambda: 2.0,
    )
    srv = build(tmp_path / "ui", service)
    client = Client(srv.port)
    client.login()
    yield SimpleNamespace(client=client, registry=registry, world=world)
    srv.stop()


def test_real_service_import_and_confirm_round_trip(real):
    resp = real.client.request(
        "POST", "/api/catalogue/import", {"xmp": subjects("Animals|Cat|Mia")}
    )
    assert resp.status == 200 and resp.json["items"][0]["created"] is True
    listing = real.client.request("GET", "/api/catalogue").json
    (record,) = listing["records"]
    assert record["status"] == "pending" and record["has_appearance"] is False
    (live,) = listing["live"]
    done = real.client.request(
        "POST", f"/api/catalogue/{record['id']}/confirm", {"entry_id": live["entry_id"]}
    )
    assert done.status == 200
    after = real.client.request("GET", "/api/catalogue").json
    assert after["records"] == [] and after["live"][0]["identified"] is True


def test_real_service_rejects_hostile_files_cleanly(real):
    bomb = '<!DOCTYPE a [<!ENTITY b "c">]><a>&b;</a>'
    for xmp, code in ((bomb, "xmp_forbidden_xml"), ("not xml", "xmp_not_xml")):
        resp = real.client.request("POST", "/api/catalogue/import", {"xmp": xmp})
        assert (resp.status, resp.json) == (422, {"error": code})
    gif = b64(b"GIF89a" + b"\x00" * 32)
    resp = real.client.request(
        "POST", "/api/catalogue/import", {"xmp": ATTR_FORM, "image_b64": gif}
    )
    assert resp.status == 422 and resp.json == {"error": "image_format"}
    assert real.registry.list() == []


def test_a_lone_surrogate_in_the_sidecar_is_a_clean_422(real):
    resp = real.client.request("POST", "/api/catalogue/import", raw=b'{"xmp": "<a>\\ud800</a>"}')
    assert (resp.status, resp.json) == (422, {"error": "xmp_not_xml"})


def test_names_are_nfc_normalised_before_validation():
    cfg = RoboEmotConfig().webui
    assert validation.name("Re\u0301", "n", cfg) == "R\u00e9"
    assert validation.name(" Cafe\u0301 ", "n", cfg) == "Caf\u00e9"


def test_nfd_and_nfc_spellings_of_a_name_are_one_record(real):
    for spelling in ("Cle\u0301o", "Cl\u00e9o"):
        resp = real.client.request(
            "POST", "/api/catalogue/import", {"xmp": subjects(f"Animals|Cat|{spelling}")}
        )
        assert resp.status == 200
    assert len(real.registry.entities()) == 1


def test_config_checks_the_catalogue_against_the_web_limits():
    RoboEmotConfig()  # the defaults agree
    with pytest.raises(ValueError, match="catalogue_max_body_bytes"):
        RoboEmotConfig(
            catalogue={"max_image_bytes": 4_000_000}, webui={"catalogue_max_body_bytes": 1_000_000}
        )


def test_the_import_budget_is_capped_by_half_the_request_deadline(tmp_path):
    from roboemot.webui.catalogue import open_catalogue

    reg = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), IdentityConfig())
    cfg = RoboEmotConfig(webui={"request_deadline_s": 4}, catalogue={"import_budget_s": 30})
    service = open_catalogue(reg, lambda: cfg)
    assert service._budget_cap() == 2.0 and service.cfg.import_budget_s == 30


def test_default_pixel_cap_is_the_camera_size():
    assert RoboEmotConfig().catalogue.max_image_pixels == 12_000_000
