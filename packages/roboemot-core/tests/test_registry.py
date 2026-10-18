"""Registry, matcher, store and estimates (SPEC 6, 12 Data, 5 Rules for estimated attributes)."""

import json
import logging
import stat
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.config.schema import IdentityConfig
from roboemot.registry import (
    Consent,
    ConsentError,
    FileStore,
    Matcher,
    NullCipher,
    Registry,
    estimate_age_group,
    estimate_gender,
    possible_minor,
)
from roboemot.registry.records import RecordSummary

DIM = 8
MODEL = "face-v1"
VOICE_MODEL = "voice-v1"
WHEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
SCOPE = ("face", "voice", "refresh")
CONSENT = Consent("Marcello", WHEN, SCOPE)
GUARDIAN = Consent("Parent", WHEN, SCOPE, guardian="Parent")


class Clock:
    def __init__(self):
        self.now = WHEN

    def __call__(self):
        return self.now

    def advance(self, seconds: float):
        self.now += timedelta(seconds=seconds)


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


def file_store(tmp_path, **kw) -> FileStore:
    return FileStore(tmp_path / "reg", allow_plaintext=True, **kw)


def make(tmp_path, clock=None, **cfg) -> Registry:
    return Registry(file_store(tmp_path), IdentityConfig(**cfg), clock=clock or Clock())


def reopen(tmp_path, **cfg) -> Registry:
    return Registry(file_store(tmp_path), IdentityConfig(**cfg), clock=Clock())


def matcher(reg: Registry) -> Matcher:
    return Matcher(reg, reg.identity)


def enroll(reg: Registry, axis_i: int, name: str = "A", **kw):
    kw.setdefault("age_group", "adult")
    kw.setdefault("consent", CONSENT)
    return reg.enroll(name, cluster(axis_i), MODEL, **kw)


def people_files(tmp_path, person_id: str):
    return sorted((tmp_path / "reg" / "people").glob(f"{person_id}.*"))


# ----------------------------------------------------------------------------- matching


def test_voting_needs_three_of_last_five(tmp_path):
    reg = make(tmp_path)
    person = enroll(reg, 0)
    m = matcher(reg)
    hit, miss = axis(0), axis(5)
    seen = [m.observe(1, e, MODEL, 0.0, refresh=False) for e in (hit, miss, hit, miss, hit)]
    assert [i.known for i in seen] == [False, False, False, False, True]
    assert seen[-1].owner_id == person.person_id
    assert seen[-1].anonymous_id is None


def test_old_votes_leave_the_window(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0)
    m = matcher(reg)
    for e in (axis(0), axis(0), axis(5), axis(5), axis(5)):
        last = m.observe(1, e, MODEL, 0.0, refresh=False)
    assert not last.known


def test_margin_over_second_best_is_required(tmp_path):
    reg = make(tmp_path)
    a = enroll(reg, 0, "A")
    b = reg.enroll("B", np.stack([blend(0, 1, 0.3)] * 5), MODEL, consent=CONSENT, age_group="adult")
    m = matcher(reg)
    assert not any(m.observe(1, axis(0), MODEL, 0.0, refresh=False).known for _ in range(5))
    reg.forget(b.person_id)
    assert any(m.observe(2, axis(0), MODEL, 0.0, refresh=False).known for _ in range(5))
    assert m.observe(2, axis(0), MODEL, 0.0, refresh=False).owner_id == a.person_id


def test_single_enrolled_owner_has_no_margin_to_beat(tmp_path):
    reg = make(tmp_path)
    a = enroll(reg, 0)
    m = matcher(reg)
    query = blend(0, 1, 1.0)  # cosine about 0.54: above the threshold, nobody to beat
    for _ in range(3):
        found = m.observe(1, query, MODEL, 0.0, refresh=False)
    assert found.owner_id == a.person_id


def test_below_threshold_is_unknown(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0)
    m = matcher(reg)
    query = blend(0, 1, 1.2)  # cosine about 0.36
    assert not any(m.observe(1, query, MODEL, 0.0, refresh=False).known for _ in range(5))


def test_embeddings_of_another_model_are_ignored(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0)
    m = matcher(reg)
    assert not any(m.observe(1, axis(0), "face-v2", 0.0, refresh=False).known for _ in range(5))


def test_anonymous_id_is_random_session_only_and_never_persisted(tmp_path):
    reg = make(tmp_path)
    m = matcher(reg)
    first = m.observe(1, axis(3), MODEL, 0.0)
    other = m.observe(2, axis(4), MODEL, 0.0)
    assert first.anonymous_id.startswith("anon-") and first.anonymous_id != other.anonymous_id
    assert m.observe(1, axis(3), MODEL, 0.0).anonymous_id == first.anonymous_id
    for path in (tmp_path / "reg").rglob("*"):
        assert path.is_dir() or b"anon-" not in path.read_bytes()
    assert reg.list() == []
    m.new_session()
    assert m.observe(1, axis(3), MODEL, 0.0).anonymous_id != first.anonymous_id


def test_track_id_reuse_after_release_starts_clean(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0)
    m = matcher(reg)
    for _ in range(3):
        found = m.observe(1, axis(0), MODEL, 0.0, refresh=False)
    assert found.known
    m.release(1)
    assert not m.observe(1, axis(5), MODEL, 0.0, refresh=False).known


def enroll_with_voice(reg, i, name="A", **kw):
    return enroll(reg, i, name, voice_embedding=axis(i), voice_model_id=VOICE_MODEL, **kw)


def test_face_voice_disagreement_drops_to_unknown(tmp_path):
    reg = make(tmp_path)
    a = enroll_with_voice(reg, 0)
    enroll_with_voice(reg, 1, "B")
    m = matcher(reg)
    for _ in range(3):
        face = m.observe(1, axis(0), MODEL, 0.0, refresh=False)
    assert face.owner_id == a.person_id
    for _ in range(3):
        fused = m.observe_voice(1, axis(1), VOICE_MODEL, 0.0)
    assert not fused.known and fused.disagreement and fused.anonymous_id


def test_face_and_voice_agreeing_keep_the_identity(tmp_path):
    reg = make(tmp_path)
    a = enroll_with_voice(reg, 0)
    m = matcher(reg)
    for _ in range(3):
        m.observe(1, axis(0), MODEL, 0.0, refresh=False)
        fused = m.observe_voice(1, axis(0), VOICE_MODEL, 0.0)
    assert fused.owner_id == a.person_id and not fused.disagreement


def test_voice_alone_can_identify(tmp_path):
    reg = make(tmp_path)
    a = enroll_with_voice(reg, 0)
    m = matcher(reg)
    for _ in range(3):
        fused = m.observe_voice(1, axis(0), VOICE_MODEL, 0.0)
    assert fused.owner_id == a.person_id


def switched_to_y(tmp_path):
    reg = make(tmp_path)
    x = enroll_with_voice(reg, 0, "X")
    y = enroll_with_voice(reg, 1, "Y")
    m = matcher(reg)
    for _ in range(3):
        m.observe(1, axis(0), MODEL, 0.0, refresh=False)
        m.observe_voice(1, axis(0), VOICE_MODEL, 0.0)
    for _ in range(3):
        last = m.observe(1, axis(1), MODEL, 0.0, refresh=False)
    return None, m, x, y, last


def test_a_face_switch_does_not_erase_voice_evidence(tmp_path):
    _, _, _, _, last = switched_to_y(tmp_path)
    assert not last.known and last.disagreement


def test_disagreement_ends_when_the_stale_votes_age_out(tmp_path):
    _, m, _, y, _ = switched_to_y(tmp_path)
    later = IdentityConfig().vote_max_age_s + 1
    for _ in range(3):
        last = m.observe(1, axis(1), MODEL, later, refresh=False)
    assert last.owner_id == y.person_id and not last.disagreement


def test_fresh_agreeing_voice_votes_resolve_the_disagreement(tmp_path):
    _, m, _, y, _ = switched_to_y(tmp_path)
    for _ in range(3):
        last = m.observe_voice(1, axis(1), VOICE_MODEL, 1.0)
    assert last.owner_id == y.person_id and not last.disagreement


def test_refresh_resumes_once_the_conflicting_vote_expires(tmp_path):
    reg = make(tmp_path, refresh_min_interval_s=0)
    a = enroll_with_voice(reg, 0, "A")
    enroll_with_voice(reg, 1, "B")
    m = matcher(reg)
    m.observe_voice(1, axis(1), VOICE_MODEL, 0.0)
    later = IdentityConfig().vote_max_age_s + 1
    for _ in range(3):
        m.observe(1, blend(0, 2, 0.5), MODEL, later)
    assert len(reg.get_person(a.person_id).face_embeddings) == 6


def test_entities_are_matched_by_appearance(tmp_path):
    reg = make(tmp_path)
    cat = reg.enroll_entity("animal", "Mia", cluster(2), MODEL, category="cat")
    m = matcher(reg)
    for _ in range(3):
        found = m.observe(1, axis(2), MODEL, 0.0, kind="entity")
    assert found.owner_id == cat.entity_id and found.kind == "entity"


def test_pending_records_are_never_matched(tmp_path):
    reg = make(tmp_path)
    reg.create_pending_entity("animal", "Mia")
    reg.create_pending_person("P")
    m = matcher(reg)
    assert not any(m.observe(1, axis(0), MODEL, 0.0).known for _ in range(5))


def test_non_finite_queries_are_unknown(tmp_path):
    reg = make(tmp_path)
    enroll(reg, 0)
    m = matcher(reg)
    for bad in (np.full(DIM, np.nan), np.full(DIM, np.inf), np.zeros(DIM)):
        assert not any(m.observe(1, bad, MODEL, 0.0).known for _ in range(5))


def test_matcher_requires_its_config(tmp_path):
    with pytest.raises(TypeError):
        Matcher(make(tmp_path))


# ------------------------------------------------------------------------------ refresh


def test_refresh_adds_after_high_confidence_match_and_respects_the_cap(tmp_path):
    reg = make(tmp_path, max_embeddings_per_person=7, refresh_min_interval_s=0)
    person = enroll(reg, 0)
    m = matcher(reg)
    rows = [len(reg.get_person(person.person_id).face_embeddings)]
    for j in (1, 2, 3, 4):
        for _ in range(3):
            m.observe(1, blend(0, j, 0.5), MODEL, 0.0)
        rows.append(len(reg.get_person(person.person_id).face_embeddings))
    stored = reg.get_person(person.person_id)
    assert rows[0] == 5 and max(rows) == 7
    assert len(stored.face_embeddings) == 7 and stored.n_enrolled == 5
    np.testing.assert_allclose(stored.face_embeddings[:5], cluster(0), atol=1e-6)


def test_refresh_never_replaces_enrollment_embeddings(tmp_path):
    reg = make(tmp_path, max_embeddings_per_person=5, refresh_min_interval_s=0)
    person = enroll(reg, 0)
    assert not reg.add_refresh_embedding(person.person_id, blend(0, 1, 0.5), MODEL)


def test_refresh_requires_high_similarity_and_skips_duplicates(tmp_path):
    reg = make(tmp_path)
    person = enroll(reg, 0)
    m = matcher(reg)
    for _ in range(5):
        m.observe(1, blend(0, 1, 0.9), MODEL, 0.0)  # cosine 0.62: matches, below refresh_min
    assert len(reg.get_person(person.person_id).face_embeddings) == 5
    assert not reg.add_refresh_embedding(person.person_id, cluster(0)[0], MODEL)
    assert not reg.add_refresh_embedding(person.person_id, blend(0, 1, 0.5), "face-v2")


def test_refresh_similarity_is_measured_against_enrollment_embeddings_only(tmp_path):
    reg = make(tmp_path, refresh_min_interval_s=0)
    p = enroll(reg, 0)
    assert reg.add_refresh_embedding(p.person_id, blend(0, 1, 0.5), MODEL)
    # close to the refreshed embedding (0.4 rad away) but 0.9 rad from enrollment: drift
    assert not reg.add_refresh_embedding(p.person_id, blend(0, 1, 0.9), MODEL)


def test_refresh_respects_the_minimum_interval(tmp_path):
    clock = Clock()
    reg = make(tmp_path, clock, refresh_min_interval_s=60)
    p = enroll(reg, 0)
    assert reg.add_refresh_embedding(p.person_id, blend(0, 1, 0.4), MODEL)
    assert not reg.add_refresh_embedding(p.person_id, blend(0, 2, 0.4), MODEL)
    clock.advance(61)
    assert reg.add_refresh_embedding(p.person_id, blend(0, 2, 0.4), MODEL)


def test_refresh_needs_the_refresh_scope(tmp_path):
    reg = make(tmp_path, refresh_min_interval_s=0)
    p = enroll(reg, 0, consent=Consent("M", WHEN, ("face",)))
    assert not reg.add_refresh_embedding(p.person_id, blend(0, 1, 0.5), MODEL)


def test_refresh_is_blocked_while_voice_conflicts(tmp_path):
    reg = make(tmp_path, refresh_min_interval_s=0)
    a = enroll_with_voice(reg, 0, "A")
    enroll_with_voice(reg, 1, "B")
    m = matcher(reg)
    m.observe_voice(1, axis(1), VOICE_MODEL, 0.0)
    for _ in range(3):
        found = m.observe(1, blend(0, 2, 0.5), MODEL, 0.0)
    assert found.owner_id == a.person_id
    assert len(reg.get_person(a.person_id).face_embeddings) == 5


def test_refresh_ignores_non_finite_embeddings(tmp_path):
    reg = make(tmp_path, refresh_min_interval_s=0)
    p = enroll(reg, 0)
    assert not reg.add_refresh_embedding(p.person_id, np.full(DIM, np.nan), MODEL)
    assert not reg.add_refresh_embedding(p.person_id, np.zeros(DIM), MODEL)


# ----------------------------------------------------------------------------- registry


def test_enroll_without_consent_raises_and_stores_nothing(tmp_path):
    reg = make(tmp_path)
    with pytest.raises(ConsentError):
        reg.enroll("Nobody", cluster(0), MODEL, age_group="adult")
    assert reg.list() == [] and not list((tmp_path / "reg" / "people").glob("*"))


def test_possible_minors_need_a_guardian(tmp_path):
    reg = make(tmp_path)
    for age in (None, "unknown", "child", "teen"):
        with pytest.raises(ConsentError):
            reg.enroll("Kid", cluster(0), MODEL, consent=CONSENT, age_group=age)
    kid = reg.enroll("Kid", cluster(0), MODEL, consent=GUARDIAN, age_group="child")
    assert kid.status == "active" and kid.age_group.source == "declared"
    assert reg.enroll("Gran", cluster(1), MODEL, consent=CONSENT, age_group="senior").status == (
        "active"
    )
    unknown = reg.enroll("Anon", cluster(2), MODEL, consent=GUARDIAN)
    assert unknown.possible_minor


def test_possible_minor_is_true_for_anything_not_adult_or_senior():
    assert [possible_minor(a) for a in ("adult", "senior")] == [False, False]
    assert all(possible_minor(a) for a in (None, "unknown", "child", "teen", "bogus", ""))


def test_age_group_and_gender_are_validated(tmp_path):
    reg = make(tmp_path)
    with pytest.raises(ValueError):
        reg.enroll("A", cluster(0), MODEL, consent=CONSENT, age_group="grown-up")
    for gender in ("other", "x"):
        with pytest.raises(ValueError):
            reg.enroll("A", cluster(0), MODEL, consent=CONSENT, age_group="adult", gender=gender)
    ok = reg.enroll("A", cluster(0), MODEL, consent=CONSENT, age_group="adult", gender="female")
    assert ok.gender.value == "female"
    assert reg.list()[0].id == ok.person_id and len(reg.list()) == 1


def test_consent_contents_are_validated():
    with pytest.raises(ConsentError):
        Consent("  ", WHEN, SCOPE)
    with pytest.raises(ConsentError):
        Consent("M", WHEN, ())
    with pytest.raises(ConsentError):
        Consent("M", WHEN, ("face", "everything"))
    with pytest.raises(ConsentError):
        Consent("M", datetime(2026, 1, 1), SCOPE)
    with pytest.raises(ConsentError):
        Consent("M", WHEN, SCOPE, guardian=" ")


def test_blank_guardian_does_not_satisfy_a_minor(tmp_path):
    reg = make(tmp_path)
    with pytest.raises(ConsentError):
        reg.enroll("K", cluster(0), MODEL, consent=Consent("P", WHEN, SCOPE), age_group="child")


def test_voice_and_face_are_stored_only_within_scope(tmp_path):
    reg = make(tmp_path)
    with pytest.raises(ConsentError):
        enroll_with_voice(reg, 0, consent=Consent("M", WHEN, ("face",)))
    with pytest.raises(ConsentError):
        enroll(reg, 0, consent=Consent("M", WHEN, ("voice",)))
    assert reg.list() == []


def test_pending_person_needs_a_guardian_unless_declared_adult(tmp_path):
    reg = make(tmp_path)
    p = reg.create_pending_person("From catalogue")
    assert p.status == "pending" and not len(p.face_embeddings)
    with pytest.raises(ConsentError):
        reg.complete_enrollment(p.person_id, cluster(0), MODEL, CONSENT)
    done = reg.complete_enrollment(p.person_id, cluster(0), MODEL, GUARDIAN)
    assert done.status == "active" and done.n_enrolled == 5
    assert sorted(f.suffix for f in people_files(tmp_path, p.person_id)) == [".json", ".npy"]


def test_complete_enrollment_validates_inputs(tmp_path):
    reg = make(tmp_path)
    p = reg.create_pending_person("X")
    with pytest.raises(ValueError):
        reg.complete_enrollment(p.person_id, cluster(0, 4), MODEL, GUARDIAN)
    with pytest.raises(ValueError):
        reg.complete_enrollment(p.person_id, np.full((5, DIM), np.nan), MODEL, GUARDIAN)
    assert reg.get_person(p.person_id).status == "pending"


def test_enrollment_embedding_count_is_bounded(tmp_path):
    reg = make(tmp_path)
    with pytest.raises(ValueError):
        reg.enroll("A", cluster(0, 4), MODEL, consent=CONSENT, age_group="adult")
    with pytest.raises(ValueError):
        reg.enroll("A", cluster(0, 21), MODEL, consent=CONSENT, age_group="adult")


def test_non_finite_and_zero_embeddings_are_rejected_at_enrollment(tmp_path):
    reg = make(tmp_path)
    for bad in (np.nan, np.inf):
        faces = cluster(0).copy()
        faces[2, 3] = bad
        with pytest.raises(ValueError):
            reg.enroll("A", faces, MODEL, consent=CONSENT, age_group="adult")
    zero = cluster(0).copy()
    zero[1] = 0
    with pytest.raises(ValueError):
        reg.enroll("A", zero, MODEL, consent=CONSENT, age_group="adult")
    with pytest.raises(ValueError):
        enroll(reg, 0, voice_embedding=np.full(DIM, np.nan), voice_model_id=VOICE_MODEL)
    assert reg.list() == []


def test_failed_write_leaves_no_file_and_no_record(tmp_path):
    reg = make(tmp_path)
    with pytest.raises(TypeError):
        enroll(reg, 0, preferences={"t": datetime.now()})
    assert not list((tmp_path / "reg" / "people").glob("*")) and reg.list() == []


def test_forget_removes_every_file_and_field(tmp_path):
    reg = make(tmp_path)
    a = enroll_with_voice(reg, 0)
    b = enroll(reg, 1, "B")
    seen = []

    class Hist:
        def export(self, person_id):
            return {"events": 3}

        def forget(self, person_id):
            seen.append(person_id)

    reg.attach_history(Hist())
    assert len(people_files(tmp_path, a.person_id)) == 3
    assert reg.forget(a.person_id) is True
    assert not people_files(tmp_path, a.person_id)
    assert len(people_files(tmp_path, b.person_id)) == 2
    assert seen == [a.person_id]
    assert [r.id for r in reg.list()] == [b.person_id]
    assert [r.id for r in reopen(tmp_path).list()] == [b.person_id]
    assert reg.forget(a.person_id) is False


def test_forget_purges_matcher_votes(tmp_path):
    reg = make(tmp_path)
    a = enroll(reg, 0)
    m = matcher(reg)
    for _ in range(3):
        found = m.observe(1, axis(0), MODEL, 0.0, refresh=False)
    assert found.owner_id == a.person_id
    reg.forget(a.person_id)
    assert not m.observe(1, axis(0), MODEL, 0.0, refresh=False).known


def test_forget_purges_voice_votes_too(tmp_path):
    reg = make(tmp_path)
    a = enroll_with_voice(reg, 0)
    m = matcher(reg)
    for _ in range(3):
        m.observe_voice(1, axis(0), VOICE_MODEL, 0.0)
    reg.forget(a.person_id)
    assert not m.observe_voice(1, axis(0), VOICE_MODEL, 0.0).known


def test_forget_survives_a_failing_history_link(tmp_path):
    reg = make(tmp_path)
    a = enroll(reg, 0)
    calls = []

    class Bad:
        def export(self, person_id):
            return None

        def forget(self, person_id):
            raise RuntimeError("history down")

    class Good(Bad):
        def forget(self, person_id):
            calls.append(person_id)

    reg.attach_history(Bad())
    reg.attach_history(Good())
    with pytest.raises(ExceptionGroup):
        reg.forget(a.person_id)
    assert calls == [a.person_id]
    assert not people_files(tmp_path, a.person_id)
    assert reg.gallery("face", MODEL) == {}


def test_forget_clears_memory_even_when_the_disk_delete_fails(tmp_path):
    class Stuck(FileStore):
        def delete(self, collection, key):
            raise OSError("read-only")

    reg = Registry(Stuck(tmp_path / "reg", allow_plaintext=True), IdentityConfig(), clock=Clock())
    a = enroll(reg, 0)
    with pytest.raises(ExceptionGroup):
        reg.forget(a.person_id)
    assert reg.gallery("face", MODEL) == {} and reg.list() == []


def test_a_stale_record_cannot_resurrect_a_forgotten_person(tmp_path):
    reg = make(tmp_path)
    a = enroll(reg, 0)
    stale = reg.get_person(a.person_id)
    reg.forget(a.person_id)
    with pytest.raises(KeyError):
        reg._commit_person(stale)
    assert not people_files(tmp_path, a.person_id)
    assert not reg.add_refresh_embedding(a.person_id, blend(0, 1, 0.5), MODEL)


def test_export_has_the_whole_record(tmp_path):
    reg = make(tmp_path)
    p = enroll_with_voice(reg, 0, preferences={"language": "it"})

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


def test_list_returns_summaries_without_embeddings(tmp_path):
    reg = make(tmp_path)
    p = enroll(reg, 0, "Zed")
    e = reg.enroll_entity("animal", "Mia", cluster(2), MODEL)
    q = reg.create_pending_person("Abe")
    rows = reg.list()
    assert all(isinstance(r, RecordSummary) for r in rows)
    assert [r.id for r in rows] == [q.person_id, p.person_id, e.entity_id]
    assert [r.kind for r in rows] == ["person", "person", "animal"]
    assert rows[1].age_group == "adult" and rows[0].status == "pending"
    assert not hasattr(rows[1], "face_embeddings")


def test_records_do_not_show_biometrics_or_consent_in_repr(tmp_path):
    reg = make(tmp_path)
    p = reg.enroll(
        "A",
        np.full((5, DIM), 0.123456),
        MODEL,
        consent=CONSENT,
        age_group="adult",
        voice_embedding=np.full(DIM, 0.654321),
        voice_model_id=VOICE_MODEL,
        preferences={"topic": "secret-topic"},
    )
    e = reg.enroll_entity("animal", "Mia", np.full((2, DIM), 0.777), MODEL)
    text = repr(p) + repr(e)
    for needle in ("0.123456", "0.654321", "secret-topic", "Marcello", "0.777"):
        assert needle not in text


def test_entity_rename_merge_delete_and_pending(tmp_path):
    reg = make(tmp_path)
    placeholder = reg.enroll_entity("animal", "cat 1", cluster(2), MODEL, category="cat")
    pending = reg.create_pending_entity("animal", "Mia", category="cat", description="tabby")
    assert pending.status == "pending" and pending.origin == "xmp"
    reg.mark_seen(placeholder.entity_id)
    reg.merge_entities(pending.entity_id, placeholder.entity_id)
    merged = reg.get_entity(pending.entity_id)
    assert merged.status == "active" and len(merged.embeddings) == 5
    assert merged.last_seen == WHEN and merged.description == "tabby"
    assert [r.id for r in reg.list()] == [pending.entity_id]
    assert reg.rename_entity(pending.entity_id, "Mia the cat").display_name == "Mia the cat"
    other = reg.enroll_entity("animal", "Rex", cluster(3), "face-v2", category="cat")
    with pytest.raises(ValueError):
        reg.merge_entities(pending.entity_id, other.entity_id)
    reg.delete_entity(other.entity_id)
    assert not list((tmp_path / "reg" / "entities").glob(f"{other.entity_id}.*"))


def test_merge_rejects_self_and_mismatched_kind_or_category(tmp_path):
    reg = make(tmp_path)
    cat = reg.enroll_entity("animal", "Mia", cluster(2), MODEL, category="cat")
    dog = reg.enroll_entity("animal", "Rex", cluster(3), MODEL, category="dog")
    mug = reg.enroll_entity("object", "mug", cluster(4), MODEL, category="cat")
    for keep, drop in ((cat, cat), (cat, dog), (cat, mug)):
        with pytest.raises(ValueError):
            reg.merge_entities(keep.entity_id, drop.entity_id)
    assert len(reg.list()) == 3


def test_merge_keeps_the_latest_last_seen(tmp_path):
    clock = Clock()
    reg = make(tmp_path, clock)
    a = reg.enroll_entity("animal", "a", cluster(2), MODEL, category="cat")
    b = reg.enroll_entity("animal", "b", cluster(3), MODEL, category="cat")
    reg.mark_seen(a.entity_id)
    clock.advance(10)
    reg.mark_seen(b.entity_id)
    assert reg.merge_entities(a.entity_id, b.entity_id).last_seen == clock.now


def test_entity_kind_is_validated(tmp_path):
    reg = make(tmp_path)
    with pytest.raises(ValueError):
        reg.enroll_entity("person", "x", cluster(2), MODEL)
    with pytest.raises(ValueError):
        reg.create_pending_entity("robot", "x")
    ok = reg.create_pending_entity("object", "x")
    assert reg.get_entity(ok.entity_id).kind == "object"


def test_mark_seen_is_buffered_until_flush_or_interval(tmp_path):
    clock = Clock()
    reg = make(tmp_path, clock)
    p = enroll(reg, 0)
    version = reg.version
    reg.mark_seen(p.person_id)
    assert reg.version == version
    assert reopen(tmp_path).get_person(p.person_id).last_seen is None
    reg.flush()
    assert reopen(tmp_path).get_person(p.person_id).last_seen == WHEN
    clock.advance(120)
    reg.mark_seen(p.person_id)
    assert reopen(tmp_path).get_person(p.person_id).last_seen == clock.now


def test_corrupt_item_is_quarantined_not_fatal(tmp_path, caplog):
    reg = make(tmp_path)
    good = enroll(reg, 1, "Good")
    bad = enroll(reg, 0, "Bad")
    (tmp_path / "reg" / "people" / f"{bad.person_id}.json").write_text("{not json Bad-secret")
    with caplog.at_level(logging.ERROR):
        again = reopen(tmp_path)
    assert [r.id for r in again.list()] == [good.person_id]
    names = [p.name for p in people_files(tmp_path, bad.person_id)]
    assert all(n.endswith(".corrupt") for n in names) and names
    assert bad.person_id in caplog.text and "Bad-secret" not in caplog.text


# --------------------------------------------------------------------------------- store


def test_store_round_trip_keeps_everything(tmp_path):
    reg = make(tmp_path)
    p = enroll_with_voice(
        reg, 0, age_group="senior", gender="female", register_overrides={"vocabulary": "simple"}
    )
    back = reopen(tmp_path).get_person(p.person_id)
    np.testing.assert_array_equal(back.face_embeddings, p.face_embeddings)
    np.testing.assert_array_equal(back.voice_embedding, p.voice_embedding)
    assert back.meta() == p.meta()
    assert back.consent == CONSENT and back.possible_minor is False


def test_stored_json_carries_a_schema_version(tmp_path):
    reg = make(tmp_path)
    p = enroll(reg, 0)
    raw = json.loads((tmp_path / "reg" / "people" / f"{p.person_id}.json").read_bytes()[1:])
    assert raw["schema_version"] == 1


class Boom:
    armed = False

    def encrypt(self, data, aad):
        if self.armed and aad.endswith(b".json"):
            raise OSError("disk full")
        return data

    def decrypt(self, data, aad):
        return data


def test_failed_write_keeps_the_old_item_and_leaves_no_temp_or_orphan(tmp_path):
    cipher = Boom()
    store = FileStore(tmp_path / "s", cipher)
    store.put("c", "k", {"v": 1}, {"a": np.ones(3, np.float32)})
    cipher.armed = True
    with pytest.raises(OSError):
        store.put("c", "k", {"v": 2}, {"a": np.zeros(3, np.float32)})
    meta, arrays = store.get("c", "k")
    assert meta == {"v": 1} and arrays["a"].tolist() == [1, 1, 1]
    assert sorted(p.name for p in (tmp_path / "s" / "c").iterdir()) == ["k.a.npy", "k.json"]
    with pytest.raises(OSError):
        store.put("c", "new", {"v": 1}, {"a": np.zeros(3, np.float32)})
    assert sorted(p.name for p in (tmp_path / "s" / "c").iterdir()) == ["k.a.npy", "k.json"]


def test_unserialisable_meta_fails_before_touching_the_disk(tmp_path):
    store = file_store(tmp_path)
    with pytest.raises(TypeError):
        store.put("c", "k", {"t": datetime.now()}, {"a": np.ones(2)})
    assert not (tmp_path / "reg" / "c").exists()


def test_startup_sweeps_temp_files_and_orphan_arrays(tmp_path):
    store = file_store(tmp_path)
    store.put("c", "k", {}, {"a": np.ones(2)})
    folder = tmp_path / "reg" / "c"
    (folder / ".tmp-abc").write_bytes(b"x")
    (folder / "ghost.face.npy").write_bytes(b"x")
    file_store(tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == ["k.a.npy", "k.json"]


def test_delete_removes_json_then_arrays(tmp_path):
    store = file_store(tmp_path)
    store.put("c", "k", {}, {"a": np.ones(2), "b": np.ones(2)})
    store.delete("c", "k")
    assert list((tmp_path / "reg" / "c").iterdir()) == []


def test_put_drops_arrays_the_item_no_longer_has(tmp_path):
    store = file_store(tmp_path)
    store.put("c", "k", {}, {"a": np.ones(2), "b": np.ones(2)})
    store.put("c", "k", {}, {"a": np.ones(2)})
    assert list(store.get("c", "k")[1]) == ["a"]


def test_cipher_gets_the_path_as_aad_and_files_carry_a_format_byte(tmp_path):
    seen = []

    class Xor:
        def encrypt(self, data, aad):
            seen.append(aad)
            return bytes(b ^ 0x5A for b in data)

        def decrypt(self, data, aad):
            return bytes(b ^ 0x5A for b in data)

    store = FileStore(tmp_path, Xor())
    store.put("c", "k", {"name": "secret"}, {"a": np.ones(2)})
    raw = (tmp_path / "c" / "k.json").read_bytes()
    assert raw[0] == 1 and b"secret" not in raw
    assert store.get("c", "k")[0] == {"name": "secret"}
    assert sorted(seen) == [b"c/k.a.npy", b"c/k.json"]


def test_plaintext_must_be_explicit_and_warns_once(tmp_path, caplog):
    with pytest.raises(ValueError):
        FileStore(tmp_path / "a")
    with caplog.at_level(logging.WARNING):
        FileStore(tmp_path / "b", allow_plaintext=True)
    assert len([r for r in caplog.records if "plaintext" in r.getMessage()]) == 1
    FileStore(tmp_path / "c", NullCipher())


def test_files_are_private(tmp_path):
    reg = make(tmp_path)
    enroll_with_voice(reg, 0)
    root = tmp_path / "reg"
    for path in [root, *root.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), path


def test_existing_directories_keep_their_mode(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    FileStore(shared, allow_plaintext=True).put("c", "k", {}, {})
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755
    assert stat.S_IMODE((shared / "c").stat().st_mode) == 0o700


def test_reads_and_deletes_do_not_create_directories(tmp_path):
    store = file_store(tmp_path)
    assert store.keys("c") == []
    store.delete("c", "k")
    with pytest.raises(KeyError):
        store.get("c", "k")
    assert not (tmp_path / "reg" / "c").exists()


def test_store_rejects_unsafe_names_and_skips_foreign_files(tmp_path):
    store = file_store(tmp_path)
    for bad in ("../evil", "k\n", "", "a b"):
        with pytest.raises(ValueError):
            store.put("c", bad, {}, {})
    store.put("c", "ok", {}, {})
    (tmp_path / "reg" / "c" / "not safe.json").write_bytes(b"x")
    assert store.keys("c") == ["ok"]
    with pytest.raises(KeyError):
        store.get("c", "missing")


# ------------------------------------------------------------------------------- config


def test_identity_config_rejects_inconsistent_values():
    with pytest.raises(ValidationError):
        IdentityConfig(votes_required=6, votes_window=5)
    with pytest.raises(ValidationError):
        IdentityConfig(enroll_min_embeddings=4)
    with pytest.raises(ValidationError):
        IdentityConfig(enroll_min_embeddings=10, max_embeddings_per_person=6)
    with pytest.raises(ValidationError):
        IdentityConfig(match_threshold=0.7, refresh_min_similarity=0.6)


# ------------------------------------------------------------------------------ estimates


def test_age_estimate_is_the_median_over_enough_frames():
    cfg = IdentityConfig()
    frames = [("adult", 0.9), ("adult", 0.95), ("teen", 0.9), ("adult", 0.9), ("senior", 0.85)]
    est = estimate_age_group(frames, cfg)
    assert (est.value, est.source) == ("adult", "estimated")
    assert est.possible_minor is False


def test_estimate_below_frame_or_confidence_floor_is_unknown_and_possibly_minor():
    cfg = IdentityConfig()
    few = estimate_age_group([("adult", 0.99)] * (cfg.estimate_frames_min - 1), cfg)
    weak = estimate_age_group([("adult", 0.5)] * 6, cfg)
    for est in (few, weak):
        assert est.value == "unknown" and est.source == "none" and est.possible_minor


def test_unknown_frames_count_toward_unknown():
    cfg = IdentityConfig()
    mostly_unknown = [("adult", 0.9)] * 5 + [("unknown", 0.9)] * 6
    assert estimate_age_group(mostly_unknown, cfg).value == "unknown"
    mostly_adult = [("adult", 0.9)] * 6 + [("unknown", 0.9)] * 2
    assert estimate_age_group(mostly_adult, cfg).value == "adult"
    assert estimate_gender([("female", 0.9)] * 3 + [("unknown", 0.9)] * 4, cfg).value == "unknown"


def test_even_frame_count_takes_the_younger_median():
    frames = [("teen", 0.9)] * 3 + [("adult", 0.9)] * 3
    assert estimate_age_group(frames, IdentityConfig()).value == "teen"


def test_declared_value_always_overrides_the_estimate():
    cfg = IdentityConfig()
    est = estimate_age_group([("child", 0.99)] * 9, cfg, declared="adult")
    assert (est.value, est.source) == ("adult", "declared")
    assert estimate_gender([], cfg, declared="female").source == "declared"


def test_declared_and_sample_values_are_validated():
    cfg = IdentityConfig()
    with pytest.raises(ValueError):
        estimate_age_group([], cfg, declared="grown-up")
    with pytest.raises(ValueError):
        estimate_gender([], cfg, declared="other")
    with pytest.raises(ValueError):
        estimate_age_group([("ancient", 0.9)] * 6, cfg)


def test_gender_estimate_uses_the_most_frequent_value_and_rejects_ties():
    cfg = IdentityConfig()
    assert estimate_gender([("female", 0.9)] * 5 + [("male", 0.9)], cfg).value == "female"
    tie = [("female", 0.9)] * 3 + [("male", 0.9)] * 3
    assert estimate_gender(tie, cfg).value == "unknown"
    assert estimate_gender([("female", 0.7)] * 6, cfg).value == "unknown"


def test_person_without_age_group_is_a_possible_minor(tmp_path):
    reg = make(tmp_path)
    p = reg.create_pending_person("X")
    assert p.possible_minor is True
    assert enroll(reg, 1, age_group="adult").possible_minor is False


def test_forget_purges_quarantined_files_too(tmp_path):
    reg = make(tmp_path)
    bad = enroll(reg, 0, "Bad")
    (tmp_path / "reg" / "people" / f"{bad.person_id}.json").write_text("{broken")
    again = reopen(tmp_path)
    assert again.quarantined() == [bad.person_id] and again.list() == []
    assert people_files(tmp_path, bad.person_id)
    assert again.forget(bad.person_id) is True
    assert not people_files(tmp_path, bad.person_id)
    assert again.quarantined() == [] and again.forget(bad.person_id) is False


def test_vote_age_follows_capture_stamps_not_the_wall_clock(tmp_path):
    clock = Clock()
    reg = make(tmp_path, clock)
    enroll_with_voice(reg, 0, "X")
    m = matcher(reg)
    for _ in range(3):
        m.observe_voice(1, axis(0), VOICE_MODEL, 100.0)
    clock.advance(3600)
    assert m.observe_voice(1, axis(0), VOICE_MODEL, 100.5).known
    assert not m.observe(1, axis(5), MODEL, 100.5 + IdentityConfig().vote_max_age_s + 1).known


def test_stamp_is_required(tmp_path):
    m = matcher(make(tmp_path))
    with pytest.raises(TypeError):
        m.observe(1, axis(0), MODEL)


def test_a_confirmed_preference_is_stored_for_an_active_person_only(tmp_path):
    reg = make(tmp_path)
    person = enroll(reg, 0)
    updated = reg.set_preference(person.person_id, "colore", "blu", max_items=2)
    assert updated.preferences == {"colore": "blu"}
    assert reopen(tmp_path).get_person(person.person_id).preferences == {"colore": "blu"}
    reg.set_preference(person.person_id, "colore", "rosso", max_items=2)  # same key: replaced
    reg.set_preference(person.person_id, "cibo", "pasta", max_items=2)
    with pytest.raises(ValueError):  # a third new key
        reg.set_preference(person.person_id, "gioco", "palla", max_items=2)
    reg.set_preference(person.person_id, "cibo", "riso", max_items=2)  # an existing one still goes
    pending = reg.create_pending_person("Ospite")
    with pytest.raises(ValueError):
        reg.set_preference(pending.person_id, "colore", "blu", max_items=2)
    with pytest.raises(KeyError):
        reg.set_preference("nobody", "colore", "blu", max_items=2)
