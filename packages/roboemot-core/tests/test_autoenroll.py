"""Auto-enrollment of recurring animals and objects (SPEC 5A, Known individuals; SPEC 12)."""

from __future__ import annotations

import stat
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from roboemot.config.schema import IdentityConfig, WorldAutoenrollConfig, WorldConfig
from roboemot.registry import FileStore, Registry
from roboemot.world import Observation, WorldModel
from roboemot.world.autoenroll import COLLECTION, AutoEnroller, display_category

MODEL = "fake-dino"
DIM = 16
START = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


class Rig:
    def __init__(self, tmp_path, model=MODEL, language="en", **cfg):
        self.tmp = tmp_path
        self.store = FileStore(tmp_path / "reg", allow_plaintext=True)
        self.registry = Registry(self.store, IdentityConfig())
        self.world = WorldModel(WorldConfig())
        self.now = START
        self.mono = 0.0
        self.stamp = 0.0
        self.rng = np.random.default_rng(1)
        self.language = language
        self.cfg = WorldAutoenrollConfig(**{"sighting_min_interval_s": 1, **cfg})
        self.enroller = self.make(model)

    def make(self, model=MODEL) -> AutoEnroller:
        return AutoEnroller(
            self.registry,
            self.store,
            self.cfg,
            model,
            world=self.world,
            language=lambda: self.language,
            clock=lambda: self.now,
            monotonic=lambda: self.mono,
        )

    def vec(self, axis=0, noise=0.05):
        v = np.zeros(DIM, np.float32)
        v[axis] = 1.0
        return v + self.rng.normal(0, noise, DIM).astype(np.float32)

    def see(self, embedding=None, *, kind="animal", category="cat", az=0.0, identity=None):
        """One observation through the world model and the enroller; returns the outcome."""
        self.stamp += 10.0
        emb = self.vec() if embedding is None else embedding
        obs = Observation(
            kind, self.stamp, "vision", az, 0.0, category=category, embedding=emb, identity=identity
        )
        result = self.world.observe(obs)
        return self.enroller.observe(obs, result), result

    def day(self, n):
        self.now = START + timedelta(days=n)

    def files(self):
        return sorted((self.tmp / "reg" / COLLECTION).glob("*.json")) if (
            self.tmp / "reg" / COLLECTION
        ).exists() else []


def sights(rig, count, az_step=1.0, **kw):
    """``count`` sightings, each a different world entry (far apart in bearing)."""
    out = None
    for i in range(count):
        out = rig.see(az=az_step * (i + 1), **kw)
    return out


def test_enrolls_after_enough_days_and_sightings(tmp_path):
    rig = Rig(tmp_path)
    for day, n in ((0, 2), (1, 2)):
        rig.day(day)
        sights(rig, n)
    assert rig.registry.entities() == []
    rig.day(2)
    owner, result = rig.see(az=2.5)
    assert owner is not None
    (record,) = rig.registry.entities()
    assert (record.origin, record.status, record.kind, record.category) == (
        "auto",
        "active",
        "animal",
        "cat",
    )
    assert record.display_name == "cat 1" and len(record.embeddings) == 5
    assert rig.world.get(result.entry.entry_id).identity == owner
    assert rig.enroller.stats() == {"clusters": 0, "sightings": 0}
    assert rig.files() == []


def test_two_days_are_not_enough_however_many_sightings(tmp_path):
    rig = Rig(tmp_path)
    for day in (0, 1):
        rig.day(day)
        sights(rig, 10)
    assert rig.registry.entities() == []
    assert rig.enroller.stats()["sightings"] == 20


def test_three_days_with_too_few_sightings(tmp_path):
    rig = Rig(tmp_path)
    for day in (0, 1, 2):
        rig.day(day)
        sights(rig, 1)
    assert rig.registry.entities() == []


def test_a_wide_cluster_never_enrolls(tmp_path):
    rig = Rig(tmp_path)
    rig.day(0)
    wide = np.zeros((4, DIM), np.float32)
    for i, angle in enumerate((-0.9, -0.3, 0.3, 0.9)):
        wide[i, 0], wide[i, 1] = np.cos(angle), np.sin(angle)
    rig.enroller.purge()
    # Seed a persisted, already-qualifying but spread-out cluster, then reload it.
    rig.store.put(
        COLLECTION,
        "wide",
        {
            "schema_version": 1,
            "cluster_id": "wide",
            "kind": "animal",
            "category": "cat",
            "model_id": MODEL,
            "sightings": 9,
            "days": [1, 2, 3],
            "last_day": START.toordinal(),
        },
        {"members": wide},
    )
    enroller = rig.enroller = rig.make()
    assert enroller.stats()["clusters"] == 1
    centre = np.zeros(DIM, np.float32)
    centre[0] = 1.0
    for i in range(3):
        rig.day(i + 5)
        assert rig.see(centre, az=float(i + 1))[0] is None
    assert rig.registry.entities() == []


def test_people_are_refused_at_runtime(tmp_path):
    rig = Rig(tmp_path)
    for _ in range(12):
        rig.see(kind="person", category="person")
    rig.see(kind="animal", category="person")
    assert rig.enroller.stats() == {"clusters": 0, "sightings": 0}
    assert rig.registry.entities() == [] and rig.files() == []
    from roboemot.world.autoenroll import _Cluster

    with pytest.raises(ValueError, match="never"):
        rig.enroller._enroll(_Cluster("x", "person", "", np.eye(2, DIM, dtype=np.float32)), "e")


def test_kinds_and_enabled_switches(tmp_path):
    rig = Rig(tmp_path, kinds=["animal"])
    rig.see(kind="object", category="cup")
    assert rig.enroller.stats()["clusters"] == 0
    rig.enroller.apply_config(WorldAutoenrollConfig(enabled=False))
    rig.see()
    assert rig.enroller.stats()["clusters"] == 0


def test_unusable_observations_are_ignored(tmp_path):
    rig = Rig(tmp_path)
    for emb in (np.zeros(DIM), np.full(DIM, np.nan), np.ones((2, 2))):
        rig.see(emb)
    rig.see(identity="some-registry-id")
    assert rig.enroller.stats()["clusters"] == 0


def test_one_entry_gives_one_sighting_per_interval(tmp_path):
    rig = Rig(tmp_path, sighting_min_interval_s=300)
    for _ in range(40):
        rig.see()  # the same world entry every 10 s
    assert rig.enroller.stats()["sightings"] == 2  # at 10 s and 310 s... not 40


def test_known_individual_gets_identity_instead_of_a_candidate(tmp_path):
    rig = Rig(tmp_path)
    base = np.zeros(DIM, np.float32)
    base[0] = 1.0
    mia = rig.registry.enroll_entity("animal", "Mia", np.vstack([base, base]), MODEL, category="cat")
    owner, result = rig.see(rig.vec(noise=0.02))
    assert owner == mia.entity_id
    assert rig.world.get(result.entry.entry_id).identity == mia.entity_id
    assert rig.enroller.stats()["clusters"] == 0
    assert rig.registry.get_entity(mia.entity_id).last_seen is not None
    other_kind, _ = rig.see(rig.vec(), kind="object", category="cup", az=3.0)
    assert other_kind is None  # an animal record never claims an object


def test_clusters_are_kept_apart_by_category_and_by_appearance(tmp_path):
    rig = Rig(tmp_path)
    sights(rig, 2, category="cat")
    sights(rig, 2, category="dog")
    for i in range(2):
        rig.see(rig.vec(axis=5), az=20.0 + i)
    assert rig.enroller.stats()["clusters"] == 3


def test_candidates_persist_across_restart_with_private_files(tmp_path):
    rig = Rig(tmp_path)
    rig.day(0)
    sights(rig, 3)
    rig.enroller.flush()
    (path,) = rig.files()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    again = rig.enroller = rig.make()
    assert again.stats() == {"clusters": 1, "sightings": 3}
    for day in (1, 2):
        rig.day(day)
        sights(rig, 1, az_step=7.0)
    assert len(rig.registry.entities()) == 1


def test_candidates_of_another_model_are_dropped_on_load(tmp_path):
    rig = Rig(tmp_path)
    sights(rig, 2)
    rig.enroller.flush()
    assert len(rig.files()) == 1
    other = rig.make("another-model")
    assert other.stats()["clusters"] == 0 and rig.files() == []


def test_retention_purges_old_candidates(tmp_path):
    rig = Rig(tmp_path, retention_days=10, persist_interval_s=1)
    sights(rig, 2, category="cat")
    rig.enroller.flush()
    rig.day(40)
    rig.mono += 5
    rig.see(rig.vec(axis=3), category="dog", az=30.0)
    assert rig.enroller.stats()["clusters"] == 1  # only the dog
    assert len(rig.files()) == 1


def test_purge_removes_candidates_from_memory_and_disk(tmp_path):
    rig = Rig(tmp_path)
    sights(rig, 2, category="cat")
    sights(rig, 2, category="dog")
    rig.enroller.flush()
    assert rig.enroller.purge("cat") == 1
    assert len(rig.files()) == 1
    assert rig.enroller.purge() == 1 and rig.files() == []


def test_cluster_and_member_caps(tmp_path):
    rig = Rig(tmp_path, max_clusters=3, max_members=2)
    for axis in range(6):
        rig.see(rig.vec(axis=axis), az=float(axis * 5))
    assert rig.enroller.stats()["clusters"] == 3
    rig2 = Rig(tmp_path / "b", max_members=2)
    sights(rig2, 6)
    (cluster,) = rig2.enroller._clusters.values()
    assert len(cluster.members) == 2 and cluster.sightings == 6


def test_placeholder_never_collides_after_a_delete(tmp_path):
    rig = Rig(tmp_path)
    reg = rig.registry
    reg.enroll_entity("animal", "cat 1", np.eye(2, DIM), MODEL, category="cat", origin="auto")
    two = reg.enroll_entity("animal", "cat 2", np.eye(2, DIM), MODEL, category="cat", origin="auto")
    reg.delete_entity(reg.entities()[0].entity_id)
    from roboemot.world.autoenroll import _Cluster

    cluster = _Cluster("c", "animal", "cat", np.eye(2, DIM, dtype=np.float32))
    assert rig.enroller._placeholder(cluster) == "cat 3"
    reg.delete_entity(two.entity_id)
    assert rig.enroller._placeholder(cluster) == "cat 1"


def test_placeholder_is_localized(tmp_path):
    rig = Rig(tmp_path, language="it")
    from roboemot.world.autoenroll import _Cluster

    cluster = _Cluster("c", "animal", "cat", np.eye(2, DIM, dtype=np.float32))
    assert rig.enroller._placeholder(cluster) == "gatto 1"
    assert display_category("potted plant", "en") == "potted plant"
    assert display_category("sports_ball", "it") == "sports ball"
    assert display_category("", "en") == "item"
