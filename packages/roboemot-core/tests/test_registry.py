"""Registry, matcher, store and estimates (SPEC 6, 12 Data, 5 Rules for estimated attributes)."""

import stat
from datetime import UTC, datetime

import numpy as np
import pytest
from roboemot.config.schema import IdentityConfig
from roboemot.registry import (
    Consent,
    ConsentError,
    FileStore,
    Matcher,
    Registry,
    estimate_age_group,
    estimate_gender,
)

DIM = 8
MODEL = "face-v1"
VOICE_MODEL = "voice-v1"
WHEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
CONSENT = Consent("Marcello", WHEN, ("recognition", "profile"))


def axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def blend(i: int, j: int, angle: float) -> np.ndarray:
    return np.cos(angle) * axis(i) + np.sin(angle) * axis(j)


def cluster(i: int, n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(i)
    return np.stack([axis(i) + 0.01 * rng.standard_normal(DIM) for _ in range(n)]).astype(
        np.float32
    )


def make(tmp_path, **cfg) -> Registry:
    return Registry(FileStore(tmp_path / "reg"), IdentityConfig(**cfg), clock=lambda: WHEN)


def enroll(reg: Registry, axis_i: int, name: str = "A", **kw):
    return reg.enroll(name, cluster(axis_i), MODEL, consent=CONSENT, **kw)


# ----------------------------------------------------------------------------- matching


def test_voting_needs_three_of_last_five(tmp_path):
    reg = make(tmp_path)
    person = enroll(reg, 0)
    m = Matcher(reg)
    hit, miss = axis(0), axis(5)
    seen = [m.observe(1, e, MODEL, refresh=False) for e in (hit, miss, hit, miss, hit)]
    assert [i.known for i in seen] == [False, False, False, False, True]
    assert seen[-1].owner_id == person.person_id
    assert seen[-1].anonymous_id is None


def test_old_votes_leave_the_window(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0)
    m = Matcher(reg)
    for e in (axis(0), axis(0), axis(5), axis(5), axis(5)):
        last = m.observe(1, e, MODEL, refresh=False)
    assert not last.known


def test_margin_over_second_best_is_required(tmp_path):
    reg = make(tmp_path)
    a = enroll(reg, 0, "A")
    reg.enroll("B", np.stack([blend(0, 1, 0.3)] * 5), MODEL, consent=CONSENT)
    m = Matcher(reg)
    assert not any(m.observe(1, axis(0), MODEL, refresh=False).known for _ in range(5))
    reg.forget(next(r.person_id for r in reg.list() if r.display_name == "B"))
    assert any(m.observe(2, axis(0), MODEL, refresh=False).known for _ in range(5))
    assert m.observe(2, axis(0), MODEL, refresh=False).owner_id == a.person_id


def test_below_threshold_is_unknown(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0)
    m = Matcher(reg)
    query = blend(0, 1, 1.2)  # cosine about 0.36
    assert not any(m.observe(1, query, MODEL, refresh=False).known for _ in range(5))


def test_embeddings_of_another_model_are_ignored(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0)
    m = Matcher(reg)
    assert not any(m.observe(1, axis(0), "face-v2", refresh=False).known for _ in range(5))


def test_anonymous_id_is_session_only_and_never_persisted(tmp_path):
    reg = make(tmp_path)
    m = Matcher(reg)
    first = m.observe(1, axis(3), MODEL)
    other = m.observe(2, axis(4), MODEL)
    assert first.anonymous_id == "anon-1" and other.anonymous_id == "anon-2"
    assert m.observe(1, axis(3), MODEL).anonymous_id == "anon-1"
    for path in (tmp_path / "reg").rglob("*"):
        assert path.is_dir() or b"anon-" not in path.read_bytes()
    assert reg.list() == []
    m.new_session()
    assert m.observe(7, axis(3), MODEL).anonymous_id == "anon-1"


def test_face_voice_disagreement_drops_to_unknown(tmp_path):
    reg = make(tmp_path)
    a = reg.enroll(
        "A", cluster(0), MODEL, consent=CONSENT, voice_embedding=axis(0), voice_model_id=VOICE_MODEL
    )
    reg.enroll(
        "B", cluster(1), MODEL, consent=CONSENT, voice_embedding=axis(1), voice_model_id=VOICE_MODEL
    )
    m = Matcher(reg)
    for _ in range(3):
        face = m.observe(1, axis(0), MODEL, refresh=False)
    assert face.owner_id == a.person_id
    for _ in range(3):
        fused = m.observe_voice(1, axis(1), VOICE_MODEL)
    assert not fused.known and fused.disagreement and fused.anonymous_id


def test_face_and_voice_agreeing_keep_the_identity(tmp_path):
    reg = make(tmp_path)
    a = reg.enroll(
        "A", cluster(0), MODEL, consent=CONSENT, voice_embedding=axis(0), voice_model_id=VOICE_MODEL
    )
    m = Matcher(reg)
    for _ in range(3):
        m.observe(1, axis(0), MODEL, refresh=False)
        fused = m.observe_voice(1, axis(0), VOICE_MODEL)
    assert fused.owner_id == a.person_id and not fused.disagreement


def test_voice_alone_can_identify(tmp_path):
    reg = make(tmp_path)
    a = reg.enroll(
        "A", cluster(0), MODEL, consent=CONSENT, voice_embedding=axis(0), voice_model_id=VOICE_MODEL
    )
    m = Matcher(reg)
    for _ in range(3):
        fused = m.observe_voice(1, axis(0), VOICE_MODEL)
    assert fused.owner_id == a.person_id


def test_entities_are_matched_by_appearance(tmp_path):
    reg = make(tmp_path)
    cat = reg.enroll_entity("animal", "Mia", cluster(2), MODEL, category="cat")
    m = Matcher(reg)
    for _ in range(3):
        found = m.observe(1, axis(2), MODEL, kind="entity")
    assert found.owner_id == cat.entity_id and found.kind == "entity"


def test_pending_records_are_never_matched(tmp_path):
    reg = make(tmp_path)
    reg.create_pending_entity("animal", "Mia")
    reg.enroll("P", cluster(0), MODEL)
    m = Matcher(reg)
    assert not any(m.observe(1, axis(0), MODEL).known for _ in range(5))


# ------------------------------------------------------------------------------ refresh


def test_refresh_adds_after_high_confidence_match_and_respects_the_cap(tmp_path):
    reg = make(tmp_path, max_embeddings_per_person=7)
    person = enroll(reg, 0)
    m = Matcher(reg)
    rows = [len(reg.get_person(person.person_id).face_embeddings)]
    for j in (1, 2, 3, 4):
        for _ in range(3):
            m.observe(1, blend(0, j, 0.5), MODEL)
        rows.append(len(reg.get_person(person.person_id).face_embeddings))
    stored = reg.get_person(person.person_id)
    assert rows[0] == 5 and max(rows) == 7
    assert len(stored.face_embeddings) == 7 and stored.n_enrolled == 5
    np.testing.assert_allclose(stored.face_embeddings[:5], cluster(0), atol=1e-6)


def test_refresh_never_replaces_enrollment_embeddings(tmp_path):
    reg = make(tmp_path, max_embeddings_per_person=5)
    person = enroll(reg, 0)
    assert not reg.add_refresh_embedding(person.person_id, blend(0, 1, 0.5), MODEL)


def test_refresh_requires_high_similarity_and_skips_duplicates(tmp_path):
    reg = make(tmp_path)
    person = enroll(reg, 0)
    m = Matcher(reg)
    for _ in range(5):
        m.observe(1, blend(0, 1, 0.9), MODEL)  # cosine 0.62: matches, below refresh_min
    assert len(reg.get_person(person.person_id).face_embeddings) == 5
    assert not reg.add_refresh_embedding(person.person_id, cluster(0)[0], MODEL)
    assert not reg.add_refresh_embedding(person.person_id, blend(0, 1, 0.5), "face-v2")


# ----------------------------------------------------------------------------- registry


def test_pending_without_consent_holds_no_biometrics(tmp_path):
    reg = make(tmp_path)
    p = reg.enroll("Nobody", cluster(0), MODEL)
    assert p.status == "pending" and p.consent is None and not len(p.face_embeddings)
    folder = tmp_path / "reg" / "people"
    assert [f.suffix for f in folder.iterdir()] == [".json"]


def test_minor_needs_a_guardian(tmp_path):
    reg = make(tmp_path)
    assert reg.enroll("Kid", cluster(0), MODEL, consent=CONSENT, age_group="child").status == (
        "pending"
    )
    kid = reg.enroll(
        "Kid",
        cluster(0),
        MODEL,
        consent=Consent("Parent", WHEN, ("recognition",), guardian="Parent"),
        age_group="child",
    )
    assert kid.status == "active" and kid.age_group.source == "declared"


def test_pending_person_activates_with_consent(tmp_path):
    reg = make(tmp_path)
    p = reg.create_pending_person("From catalogue")
    with pytest.raises(ConsentError):
        reg.complete_enrollment(p.person_id, cluster(0), MODEL, None)
    done = reg.complete_enrollment(p.person_id, cluster(0), MODEL, CONSENT)
    assert done.status == "active" and done.n_enrolled == 5


def test_enrollment_embedding_count_is_bounded(tmp_path):
    reg = make(tmp_path)
    with pytest.raises(ValueError):
        reg.enroll("A", cluster(0, 4), MODEL, consent=CONSENT)
    with pytest.raises(ValueError):
        reg.enroll("A", cluster(0, 21), MODEL, consent=CONSENT)


def test_forget_removes_every_file_and_field(tmp_path):
    reg = make(tmp_path)
    a = enroll(reg, 0, voice_embedding=axis(0), voice_model_id=VOICE_MODEL)
    b = enroll(reg, 1, "B")
    seen = []

    class Hist:
        def export(self, person_id):
            return {"events": 3}

        def forget(self, person_id):
            seen.append(person_id)

    reg.attach_history(Hist())
    folder = tmp_path / "reg" / "people"
    assert len(list(folder.glob(f"{a.person_id}.*"))) == 3
    assert reg.forget(a.person_id) is True
    assert not list(folder.glob(f"{a.person_id}.*"))
    assert len(list(folder.glob(f"{b.person_id}.*"))) == 2
    assert seen == [a.person_id]
    assert [r.person_id for r in reg.list()] == [b.person_id]
    assert Registry(FileStore(tmp_path / "reg"), IdentityConfig()).list()[0].person_id == (
        b.person_id
    )
    assert reg.forget(a.person_id) is False
    assert not Matcher(reg).observe(1, axis(0), MODEL).known


def test_export_has_the_whole_record(tmp_path):
    reg = make(tmp_path)
    p = enroll(
        reg,
        0,
        voice_embedding=axis(0),
        voice_model_id=VOICE_MODEL,
        age_group="adult",
        preferences={"language": "it"},
    )

    class Hist:
        def export(self, person_id):
            return {"events": 3}

        def forget(self, person_id):
            pass

    reg.attach_history(Hist())
    data = reg.export(p.person_id)
    assert data["display_name"] == "A" and data["embedding_model_id"] == MODEL
    assert len(data["face_embeddings"]) == 5 and len(data["voice_embedding"]) == DIM
    assert data["age_group"] == {"value": "adult", "source": "declared"}
    assert data["consent"]["who"] == "Marcello" and data["preferences"] == {"language": "it"}
    assert data["history"] == [{"events": 3}]


def test_entity_rename_merge_delete_and_pending(tmp_path):
    reg = make(tmp_path)
    placeholder = reg.enroll_entity("animal", "cat 1", cluster(2), MODEL, origin="auto")
    pending = reg.create_pending_entity("animal", "Mia", category="cat")
    assert pending.status == "pending" and pending.origin == "xmp"
    reg.merge_entities(pending.entity_id, placeholder.entity_id)
    merged = reg.get_entity(pending.entity_id)
    assert merged.status == "active" and len(merged.embeddings) == 5
    assert [r.entity_id for r in reg.list()] == [pending.entity_id]
    assert reg.rename_entity(pending.entity_id, "Mia the cat").display_name == "Mia the cat"
    other = reg.enroll_entity("object", "mug", cluster(3), "face-v2")
    with pytest.raises(ValueError):
        reg.merge_entities(pending.entity_id, other.entity_id)
    reg.delete_entity(other.entity_id)
    assert not list((tmp_path / "reg" / "entities").glob(f"{other.entity_id}.*"))


def test_mark_seen_persists_last_seen(tmp_path):
    reg = make(tmp_path)
    p = enroll(reg, 0)
    reg.mark_seen(p.person_id)
    again = Registry(FileStore(tmp_path / "reg"), IdentityConfig())
    assert again.get_person(p.person_id).last_seen == WHEN


# --------------------------------------------------------------------------------- store


def test_store_round_trip_keeps_everything(tmp_path):
    reg = make(tmp_path)
    p = enroll(
        reg,
        0,
        voice_embedding=axis(1),
        voice_model_id=VOICE_MODEL,
        age_group="senior",
        gender="female",
        register_overrides={"vocabulary": "simple"},
    )
    back = Registry(FileStore(tmp_path / "reg"), IdentityConfig()).get_person(p.person_id)
    np.testing.assert_array_equal(back.face_embeddings, p.face_embeddings)
    np.testing.assert_array_equal(back.voice_embedding, p.voice_embedding)
    assert back.meta() == p.meta()
    assert back.consent == CONSENT and back.possible_minor is False


def test_failed_write_keeps_the_old_file_and_leaves_no_temp(tmp_path):
    class Boom:
        armed = False

        def encrypt(self, data):
            if self.armed:
                raise OSError("disk full")
            return data

        def decrypt(self, data):
            return data

    cipher = Boom()
    store = FileStore(tmp_path / "s", cipher)
    store.put("c", "k", {"v": 1}, {"a": np.ones(3, np.float32)})
    cipher.armed = True
    with pytest.raises(OSError):
        store.put("c", "k", {"v": 2}, {"a": np.zeros(3, np.float32)})
    meta, arrays = store.get("c", "k")
    assert meta == {"v": 1} and arrays["a"].tolist() == [1, 1, 1]
    assert sorted(p.name for p in (tmp_path / "s" / "c").iterdir()) == ["k.a.npy", "k.json"]


def test_put_drops_arrays_the_item_no_longer_has(tmp_path):
    store = FileStore(tmp_path)
    store.put("c", "k", {}, {"a": np.ones(2), "b": np.ones(2)})
    store.put("c", "k", {}, {"a": np.ones(2)})
    assert list(store.get("c", "k")[1]) == ["a"]


def test_cipher_hook_transforms_files_on_disk(tmp_path):
    class Xor:
        def encrypt(self, data):
            return bytes(b ^ 0x5A for b in data)

        decrypt = encrypt

    store = FileStore(tmp_path, Xor())
    store.put("c", "k", {"name": "secret"}, {"a": np.ones(2)})
    assert b"secret" not in (tmp_path / "c" / "k.json").read_bytes()
    assert store.get("c", "k")[0] == {"name": "secret"}


def test_files_are_private(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0, voice_embedding=axis(0), voice_model_id=VOICE_MODEL)
    root = tmp_path / "reg"
    for path in [root, *root.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), path


def test_store_rejects_path_traversal(tmp_path):
    store = FileStore(tmp_path)
    with pytest.raises(ValueError):
        store.put("c", "../evil", {}, {})
    with pytest.raises(KeyError):
        store.get("c", "missing")


# ------------------------------------------------------------------------------ estimates


def test_age_estimate_is_the_median_over_enough_frames():
    cfg = IdentityConfig()
    frames = [("adult", 0.9), ("adult", 0.95), ("teen", 0.9), ("adult", 0.9), ("senior", 0.85)]
    est = estimate_age_group(frames, cfg)
    assert (est.value, est.source) == ("adult", "estimated")
    assert est.possible_minor is False


def test_estimate_below_frame_or_confidence_floor_is_unknown_and_possibly_minor():
    cfg = IdentityConfig()
    few = estimate_age_group([("adult", 0.99)] * 4, cfg)
    weak = estimate_age_group([("adult", 0.5)] * 6, cfg)
    for est in (few, weak):
        assert est.value == "unknown" and est.source == "none" and est.possible_minor


def test_even_frame_count_takes_the_younger_median():
    frames = [("teen", 0.9)] * 3 + [("adult", 0.9)] * 3
    assert estimate_age_group(frames, IdentityConfig()).value == "teen"


def test_declared_value_always_overrides_the_estimate():
    cfg = IdentityConfig()
    est = estimate_age_group([("child", 0.99)] * 9, cfg, declared="adult")
    assert (est.value, est.source) == ("adult", "declared")
    assert estimate_gender([], cfg, declared="female").source == "declared"


def test_gender_estimate_uses_the_most_frequent_value_and_rejects_ties():
    cfg = IdentityConfig()
    assert estimate_gender([("female", 0.9)] * 5 + [("male", 0.9)], cfg).value == "female"
    tie = [("female", 0.9)] * 3 + [("male", 0.9)] * 3
    assert estimate_gender(tie, cfg).value == "unknown"
    assert estimate_gender([("female", 0.7)] * 6, cfg).value == "unknown"


def test_person_without_age_group_is_a_possible_minor(tmp_path):
    reg = make(tmp_path)
    assert enroll(reg, 0).possible_minor is True
    assert enroll(reg, 1, age_group="adult").possible_minor is False
