"""SPEC 5, 13: animal and object embeddings (DINOv2-S) run only when they add information.

Mock stages, one object run per second of capture time (the default ``objects_hz``), the
scene-change gate off so every frame is an object run.
"""

from __future__ import annotations

import numpy as np
from roboemot.config.schema import CatalogueConfig, WorldConfig
from roboemot.registry.catalogue import CatalogueService
from roboemot.vision import Detection, Frame
from roboemot.world import FrameSpec, WorldModel, feed_world
from test_vision import APP_MODEL, CFG, axis, crowd, entity_rig, image

NO_GATE = CFG.vision.object_gate.model_copy(update={"enabled": False})
CUP = Detection((100, 100, 160, 160), 0.8, "cup")
NO_AUTOENROLL = WorldConfig().model_copy(
    update={"autoenroll": WorldConfig().autoenroll.model_copy(update={"enabled": False})}
)


def rig(tmp_path, dets=(CUP,), **kw):
    return entity_rig(tmp_path, list(dets), object_gate=NO_GATE, **kw)


def run(pipe, start, stop):
    """Process frames ``start`` .. ``stop - 1``, one second apart; returns the embed count."""
    calls = 0
    for i in range(start, stop):
        calls += pipe.process(Frame(image(i), float(i))).embeddings - 1  # minus the face
    return calls


def runs(pipe):
    return pipe.stats()["entity_embeds"]["runs"]


def skips(pipe):
    return pipe.stats()["entity_embeds"]["skips"]


def test_an_unknown_still_object_is_embedded_once_then_only_at_the_floor(tmp_path):
    pipe, app = rig(tmp_path)
    floor = int(CFG.vision.entity_reembed_floor_s)
    run(pipe, 0, floor)
    assert len(app.boxes) == 1 and runs(pipe)["new"] == 1
    assert runs(pipe)["burst"] == 0  # no record to vote against
    assert skips(pipe)["settled"] > 0
    run(pipe, floor, floor + 3)
    assert len(app.boxes) == 2 and runs(pipe)["floor"] == 1


def test_the_floor_is_configurable(tmp_path):
    pipe, app = rig(tmp_path, entity_reembed_floor_s=10.0)
    run(pipe, 0, 23)
    assert runs(pipe)["floor"] == 2 and len(app.boxes) == 3


def test_a_record_of_the_category_opens_a_burst_of_votes(tmp_path):
    pipe, app = rig(tmp_path, record="active")
    run(pipe, 0, 10)
    assert runs(pipe)["new"] == 1
    assert runs(pipe)["burst"] == CFG.vision.embedding_burst - 1
    run(pipe, 10, 20)
    assert len(app.boxes) == CFG.vision.embedding_burst  # settled after the burst


def test_a_record_of_another_kind_opens_no_burst(tmp_path):
    pipe, app = rig(tmp_path)
    reg = pipe.matcher.registry
    reg.enroll_entity("animal", "Tom", np.stack([axis(5)] * 5), APP_MODEL, category="cat")
    run(pipe, 0, 10)
    assert len(app.boxes) == 1 and runs(pipe)["burst"] == 0


def test_a_record_of_the_same_kind_votes_whatever_its_category(tmp_path):
    """The matcher votes by kind (Registry.gallery), so a label flicker still matches."""
    pipe, app = rig(
        tmp_path, dets=crowd(1, "bottle"), record="active", object_categories=["cup", "bottle"]
    )
    run(pipe, 0, 10)
    assert runs(pipe)["burst"] == CFG.vision.embedding_burst - 1


def test_a_recognised_object_keeps_the_steady_rate(tmp_path):
    pipe, app = rig(tmp_path)
    reg = pipe.matcher.registry
    mug = reg.enroll_entity("object", "Mug", np.stack([axis(3)] * 5), APP_MODEL, category="cup")
    stamps = []
    for i in range(20):
        before = len(app.boxes)
        run(pipe, i, i + 1)
        if len(app.boxes) > before:
            stamps.append(float(i))
    (state,) = pipe._entities.values()
    assert state.owner_id == mug.entity_id
    votes = CFG.identity.votes_required
    assert runs(pipe)["new"] == 1 and runs(pipe)["burst"] == votes - 1
    period = 1 / CFG.vision.rates.appearance_hz
    assert np.diff(stamps[votes - 1 :]).tolist() == [period] * (len(stamps) - votes)
    assert runs(pipe)["recognised"] == len(stamps) - votes
    assert runs(pipe)["floor"] == 0 and runs(pipe)["recognised"] >= 7


def test_an_unknown_track_is_recognised_by_a_record_enrolled_later(tmp_path):
    """Regression: a track that used its burst against record A votes again once B exists."""
    pipe, app = rig(tmp_path, record="active")  # A: never matches
    run(pipe, 0, 15)
    (state,) = pipe._entities.values()
    assert state.owner_id is None and state.votes >= CFG.vision.embedding_burst
    reg = pipe.matcher.registry
    mug = reg.enroll_entity("object", "Mug", np.stack([axis(3)] * 5), APP_MODEL, category="cup")
    run(pipe, 15, 15 + CFG.identity.votes_required + 1)
    assert state.owner_id == mug.entity_id


def test_a_record_confirmed_later_opens_a_new_burst(tmp_path):
    pipe, app = rig(tmp_path, record="pending")
    run(pipe, 0, 15)
    (state,) = pipe._entities.values()
    assert len(app.boxes) == 1  # a pending record does not vote
    reg = pipe.matcher.registry
    (mia,) = reg.entities()
    reg.confirm_entity(mia.entity_id, np.stack([axis(3)] * 5), APP_MODEL)
    run(pipe, 15, 15 + CFG.identity.votes_required + 1)
    assert state.owner_id == mia.entity_id


def test_floor_reopens_the_burst_while_a_record_could_match(tmp_path):
    pipe, app = rig(tmp_path, record="active", entity_reembed_floor_s=10.0)
    run(pipe, 0, 22)
    burst = CFG.vision.embedding_burst
    assert runs(pipe)["floor"] == 1 and runs(pipe)["burst"] >= 2 * (burst - 1) - 1


def test_a_record_added_mid_run_gives_an_unwanted_track_its_first_embedding(tmp_path):
    for status in ("pending", "active"):
        pipe, app = rig(tmp_path / status, world=NO_AUTOENROLL)
        run(pipe, 0, 5)
        assert app.boxes == []
        reg = pipe.matcher.registry
        if status == "pending":
            reg.create_pending_entity("object", "Mug", category="cup")
        else:
            reg.enroll_entity("object", "Mug", np.stack([axis(5)] * 5), APP_MODEL, category="cup")
        run(pipe, 5, 7)
        assert runs(pipe)["new"] == 1, status


def test_catalogue_confirm_works_with_auto_enrollment_off(tmp_path):
    pipe, app = rig(tmp_path, world=NO_AUTOENROLL, record="pending")
    world = WorldModel(CFG.world, FrameSpec(yaw_dofs=("head_yaw",)), pipe.bus)
    feed_world(pipe.bus, world)
    run(pipe, 0, 5)
    (entry,) = [e for e in world.entries if e.kind == "object"]
    assert world.embedding_of(entry.entry_id) is not None
    reg = pipe.matcher.registry
    (mia,) = reg.entities()
    svc = CatalogueService(
        reg, CatalogueConfig(), embedder=lambda: pipe.appearance_embedder, world=world
    )
    assert svc.confirm(mia.entity_id, entry.entry_id).status == "active"


def test_a_deleted_record_releases_its_track_and_a_new_burst_runs(tmp_path):
    pipe, app = rig(tmp_path, record="active")  # A stays, never matches
    reg = pipe.matcher.registry
    mug = reg.enroll_entity("object", "Mug", np.stack([axis(3)] * 5), APP_MODEL, category="cup")
    run(pipe, 0, 10)
    (state,) = pipe._entities.values()
    assert state.owner_id == mug.entity_id
    bursts = runs(pipe)["burst"]
    reg.delete_entity(mug.entity_id)
    run(pipe, 10, 11)
    assert state.owner_id is None and state.votes <= 1
    run(pipe, 11, 20)
    assert runs(pipe)["burst"] >= bursts + CFG.vision.embedding_burst - 1
    assert state.owner_id is None


def test_recognition_lost_votes_again(tmp_path):
    pipe, app = rig(tmp_path, record="active")
    reg = pipe.matcher.registry
    mug = reg.enroll_entity("object", "Mug", np.stack([axis(3)] * 5), APP_MODEL, category="cup")
    run(pipe, 0, 10)
    (state,) = pipe._entities.values()
    assert state.owner_id == mug.entity_id
    pipe._on_forget(mug.entity_id)
    assert state.owner_id is None and state.votes == 0


def test_a_box_that_changes_size_is_embedded_again(tmp_path):
    pipe, app = rig(tmp_path)
    run(pipe, 0, 5)
    pipe.object_detector.dets = [Detection((90, 90, 170, 170), 0.8, "cup")]  # area x1.8
    run(pipe, 5, 12)
    assert runs(pipe)["changed"] == 1 and len(app.boxes) == 2


def test_a_box_that_changes_shape_is_embedded_again(tmp_path):
    pipe, app = rig(tmp_path)
    run(pipe, 0, 5)
    pipe.object_detector.dets = [Detection((85, 112, 175, 152), 0.8, "cup")]  # aspect x2.25
    run(pipe, 5, 12)
    assert runs(pipe)["changed"] == 1


def test_box_jitter_below_the_ratios_does_not_re_embed(tmp_path):
    pipe, app = rig(tmp_path)
    run(pipe, 0, 5)
    pipe.object_detector.dets = [Detection((97, 97, 163, 163), 0.8, "cup")]  # area x1.21
    run(pipe, 5, 15)
    assert runs(pipe)["changed"] == 0 and len(app.boxes) == 1


def test_a_changed_box_restarts_the_burst_while_a_record_could_match(tmp_path):
    pipe, app = rig(tmp_path, record="active")
    run(pipe, 0, 12)
    assert len(app.boxes) == CFG.vision.embedding_burst
    pipe.object_detector.dets = [Detection((90, 90, 170, 170), 0.8, "cup")]
    run(pipe, 12, 24)
    assert runs(pipe)["changed"] == 1
    assert runs(pipe)["burst"] == 2 * (CFG.vision.embedding_burst - 1)


def test_a_track_seen_again_after_a_missed_object_run_is_embedded_again(tmp_path):
    pipe, app = rig(tmp_path)
    run(pipe, 0, 5)
    pipe.object_detector.dets = []
    run(pipe, 5, 6)
    pipe.object_detector.dets = [CUP]
    run(pipe, 6, 10)
    assert len(pipe._entities) == 1  # the same track
    assert runs(pipe)["reacquired"] == 1 and len(app.boxes) == 2


def test_no_consumer_no_embedding(tmp_path):
    off = WorldConfig().model_copy(
        update={"autoenroll": WorldConfig().autoenroll.model_copy(update={"enabled": False})}
    )
    pipe, app = rig(tmp_path, world=off)
    run(pipe, 0, 5)
    assert app.boxes == [] and skips(pipe)["no_consumer"] > 0
    obs = []
    pipe.bus.subscribe("percept.entity", lambda m: obs.append(m.payload))
    run(pipe, 5, 6)
    assert [o.category for o in obs if o.kind == "object"] == ["cup"]  # still published


def test_auto_enrollment_of_animals_only_leaves_objects_unembedded(tmp_path):
    animals = WorldConfig().autoenroll.model_copy(update={"kinds": ["animal"]})
    world = WorldConfig().model_copy(update={"autoenroll": animals})
    pipe, app = rig(tmp_path, dets=[CUP, Detection((200, 100, 260, 160), 0.8, "cat")], world=world)
    run(pipe, 0, 5)
    assert len(app.boxes) == 1 and app.boxes[0][0] > 150  # the cat


def test_a_pending_record_keeps_its_category_embedded(tmp_path):
    off = WorldConfig().autoenroll.model_copy(update={"enabled": False})
    world = WorldConfig().model_copy(update={"autoenroll": off})
    pipe, app = rig(tmp_path, world=world, record="pending")
    run(pipe, 0, 5)
    assert len(app.boxes) == 1 and runs(pipe)["burst"] == 0  # pending records do not vote


def test_a_due_track_over_the_frame_budget_counts_as_a_budget_skip(tmp_path):
    pipe, app = rig(tmp_path, dets=crowd(3))
    run(pipe, 0, 4)
    assert runs(pipe)["new"] == 3 and skips(pipe)["budget"] >= 2


def test_counts_are_in_the_report_with_every_reason(tmp_path):
    pipe, _ = rig(tmp_path)
    report = pipe.stats()["entity_embeds"]
    assert set(report["runs"]) == {"new", "burst", "recognised", "changed", "reacquired", "floor"}
    assert set(report["skips"]) == {"no_consumer", "spacing", "settled", "budget"}
    assert all(v == 0 for part in report.values() for v in part.values())
