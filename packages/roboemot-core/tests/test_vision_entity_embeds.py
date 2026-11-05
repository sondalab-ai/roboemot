"""SPEC 5, 13: animal and object embeddings (DINOv2-S) run only when they add information.

Mock stages, one object run per second of capture time (the default ``objects_hz``), the
scene-change gate off so every frame is an object run.
"""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.config.schema import CatalogueConfig, WorldConfig
from roboemot.registry.catalogue import CatalogueService
from roboemot.vision import Detection, Frame
from roboemot.world import FrameSpec, WorldModel, feed_world
from test_vision import (
    APP_MODEL,
    CFG,
    H,
    RecordingAppearance,
    W,
    axis,
    crowd,
    entity_rig,
    image,
)
from test_vision import rig as vision_rig
from test_vision_object_crop import CropAwareDetector, marked_image
from test_vision_object_gate import DT

NO_GATE = CFG.vision.object_gate.model_copy(update={"enabled": False})
NO_CROP = CFG.vision.object_crop.model_copy(update={"enabled": False})  # boxes stay put
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
    first = int(CFG.vision.entity_embed_min_age_s)  # the first embedding, at the minimum age
    run(pipe, 0, first + floor)
    assert len(app.boxes) == 1 and runs(pipe)["new"] == 1
    assert runs(pipe)["burst"] == 0  # no record to vote against
    assert skips(pipe)["settled"] > 0
    run(pipe, first + floor, first + floor + 3)
    assert len(app.boxes) == 2 and runs(pipe)["floor"] == 1


def test_the_floor_is_configurable(tmp_path):
    pipe, app = rig(tmp_path, entity_reembed_floor_s=10.0)
    run(pipe, 0, int(CFG.vision.entity_embed_min_age_s) + 21)
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
    run(pipe, 0, int(CFG.vision.entity_embed_min_age_s) + 3)
    assert runs(pipe)["new"] == 3 and skips(pipe)["budget"] >= 2


def test_counts_are_in_the_report_with_every_reason(tmp_path):
    pipe, _ = rig(tmp_path)
    report = pipe.stats()["entity_embeds"]
    assert set(report["runs"]) == {"new", "burst", "recognised", "changed", "reacquired", "floor"}
    assert set(report["skips"]) == {
        "no_consumer",
        "episode",
        "young",
        "spacing",
        "settled",
        "budget",
        "pending",
    }
    assert all(v == 0 for part in report.values() for v in part.values())


# ---------------------------------------------------- track age and shown-object episodes

MIN_AGE = CFG.vision.entity_embed_min_age_s
CENTRE_CUP = Detection((140, 100, 180, 140), 0.8, "cup")  # the frame centre is (160, 120)
SIDE_CUP = Detection((10, 100, 40, 140), 0.8, "cup")
SIDE_CUP_NEAR = Detection((0, 90, 60, 150), 0.8, "cup")  # the side cup, area x3
SPOTS = 10  # a flicker spot is used again only after SPOTS x 2 s, longer than a track lives


def flicker(i):
    """A short-lived thing of a cluttered room: one new spot every 2 s, seen on two object
    runs (the robot's object tracks lived 1.8 s at the median, 2026-11-05)."""
    k = (i // 2) % SPOTS
    return Detection((10 + 30 * k, 180, 30 + 30 * k, 200), 0.8, "bottle")


def test_a_cluttered_room_embeds_only_the_thing_that_stays(tmp_path):
    pipe, app = rig(tmp_path, record="active", object_categories=["cup", "bottle"])
    det = pipe.object_detector
    first = {}
    for i in range(40):
        det.dets = [CUP, flicker(i)]
        before = len(app.boxes)
        run(pipe, i, i + 1)
        if len(app.boxes) > before:
            first.setdefault(app.boxes[-1], float(i))
    assert set(app.boxes) == {CUP.box}  # no short-lived track got an embedding
    assert first[CUP.box] == MIN_AGE  # first sighting at 0
    assert skips(pipe)["young"] >= 15
    # A young track sorted first (never embedded) leaves the frame's turn to the next one:
    # the cup's burst runs while new flickers keep coming.
    assert runs(pipe)["new"] == 1 and runs(pipe)["burst"] == CFG.vision.embedding_burst - 1


def test_the_minimum_age_counts_from_the_first_detection(tmp_path):
    pipe, app = rig(tmp_path)
    run(pipe, 0, int(MIN_AGE))
    assert app.boxes == [] and skips(pipe)["young"] == int(MIN_AGE) - 1  # confirmed at 1 s
    run(pipe, int(MIN_AGE), int(MIN_AGE) + 1)
    assert runs(pipe)["new"] == 1


def test_a_minimum_age_of_zero_embeds_at_the_first_confirmed_sighting(tmp_path):
    pipe, app = rig(tmp_path, entity_embed_min_age_s=0.0)
    run(pipe, 0, 2)
    assert runs(pipe)["new"] == 1 and skips(pipe)["young"] == 0


def enroll_mug(pipe):
    reg = pipe.matcher.registry
    return reg.enroll_entity("object", "Mug", np.stack([axis(3)] * 5), APP_MODEL, category="cup")


def recognition_age(pipe, stamps):
    """Capture time from the cup's first detection to its identity decision."""
    for stamp in stamps:
        pipe.process(Frame(image(0), stamp))
        if any(s.owner_id is not None for s in pipe._entities.values()):
            return stamp - stamps[0]
    return None


def test_a_known_object_is_still_recognised_after_the_minimum_age(tmp_path):
    """SPEC 13: the decision of a known thing takes the minimum age plus the burst gaps of the
    votes still needed; without the age it took the confirm time plus those gaps."""
    gaps = (CFG.identity.votes_required - 1) * CFG.vision.entity_burst_min_gap_s
    active_hz = CFG.vision.rates.objects_active_hz
    for hz in (CFG.vision.rates.objects_hz, active_hz):
        stamps = [k / hz for k in range(round(20 * hz))]
        confirm = (CFG.vision.object_confirm_hits - 1) / hz
        for min_age, want in ((MIN_AGE, MIN_AGE + gaps), (0.0, confirm + gaps)):
            pipe, _ = rig(
                tmp_path / f"{hz}-{min_age}", entity_embed_min_age_s=min_age, object_crop=NO_CROP
            )
            mug = enroll_mug(pipe)
            pipe.governor.set_active("objects", hz == active_hz)  # 3 Hz only in an episode
            assert recognition_age(pipe, stamps) == pytest.approx(want), (hz, min_age)
            (state,) = pipe._entities.values()
            assert state.owner_id == mug.entity_id
            assert state.votes == CFG.identity.votes_required


def test_during_an_episode_only_the_shown_track_embeds(tmp_path):
    pipe, app = rig(tmp_path, dets=[SIDE_CUP], object_crop=NO_CROP)
    run(pipe, 0, 10)
    assert app.boxes == [SIDE_CUP.box]
    pipe.object_detector.dets = [SIDE_CUP, CENTRE_CUP]  # the person holds up a cup
    run(pipe, 10, 12)  # confirmed: attention can start a look at it
    pipe.governor.set_active("objects", True)
    pipe.object_detector.dets = [SIDE_CUP_NEAR, CENTRE_CUP]  # the side cup's box changes too
    run(pipe, 12, 20)
    assert app.boxes == [SIDE_CUP.box, CENTRE_CUP.box]  # the side cup's change waits
    assert skips(pipe)["episode"] >= 7
    pipe.governor.set_active("objects", False)
    run(pipe, 20, 23)
    assert len(app.boxes) == 3 and runs(pipe)["changed"] == 1


def test_after_an_episode_the_others_are_served_oldest_first(tmp_path):
    later = [Detection((10 + 50 * k, 180, 40 + 50 * k, 220), 0.8, "cup") for k in range(3)]
    pipe, app = rig(tmp_path, dets=[CENTRE_CUP], object_crop=NO_CROP)
    pipe.governor.set_active("objects", True)
    run(pipe, 0, 2)
    for k in range(len(later)):  # they come into view one run apart
        pipe.object_detector.dets = [CENTRE_CUP, *later[: k + 1]]
        run(pipe, 2 + k, 3 + k)
    run(pipe, 5, 12)
    assert app.boxes == [CENTRE_CUP.box]  # the others waited, past their minimum age
    pipe.governor.set_active("objects", False)
    run(pipe, 12, 15)
    assert app.boxes[1:] == [d.box for d in later]  # one per run, oldest track first


def test_the_shown_track_runs_only_its_votes_and_keeps_its_owner(tmp_path):
    """A recognised shown thing keeps its owner (sticky identity, SPEC 6) with no embedding
    while the gaze follows it, even when it is waved (its box changes)."""
    pipe, app = rig(tmp_path, dets=[CENTRE_CUP], object_crop=NO_CROP)
    mug = enroll_mug(pipe)
    pipe.governor.set_active("objects", True)
    run(pipe, 0, 10)
    (state,) = pipe._entities.values()
    assert state.owner_id == mug.entity_id
    assert len(app.boxes) == CFG.identity.votes_required  # new and burst: the votes only
    waved = [CENTRE_CUP, Detection((130, 90, 190, 150), 0.8, "cup")]  # area x2.25 and back
    for i in range(10, 40):
        pipe.object_detector.dets = [waved[i % 2]]
        run(pipe, i, i + 1)
    assert len(app.boxes) == CFG.identity.votes_required
    assert state.owner_id == mug.entity_id and skips(pipe)["episode"] >= 25
    pipe.governor.set_active("objects", False)
    run(pipe, 40, 43)
    assert runs(pipe)["recognised"] >= 1  # the steady rate is back


COUCH = Detection((60, 40, 260, 240), 0.8, "couch")  # behind the person, centre (160, 140)
COUCH_TURNED = Detection((120, 40, 320, 240), 0.8, "couch")  # the head turned to the cup
HELD_CUP = Detection((120, 90, 160, 130), 0.8, "cup")  # centre (140, 110): farther than the couch


def test_a_wrong_first_pick_of_the_shown_track_corrects_itself(tmp_path):
    """A large couch near the frame centre is picked when the look starts; once the head turns
    to the held cup, the cup is the shown track and gets its votes, and the couch none."""
    pipe, app = rig(
        tmp_path,
        dets=[COUCH],
        record="active",  # an object record that never matches: bursts stay open
        object_categories=["cup", "couch"],
        object_crop=NO_CROP,
    )
    burst = CFG.vision.embedding_burst
    run(pipe, 0, 10)
    assert len(app.boxes) == burst  # the couch's new embedding and burst, before the look
    pipe.governor.set_active("objects", True)
    pipe.object_detector.dets = [COUCH, HELD_CUP]
    run(pipe, 10, 12)
    camera = pipe._camera(W, H)
    pick = pipe._shown_track(pipe._object_visible, Frame(image(0), 11.0), camera)
    assert pick == min(pipe._entities)  # no bearing: the centre fallback takes the couch
    pipe.object_detector.dets = [COUCH_TURNED, CENTRE_CUP]
    run(pipe, 12, 20)
    during = app.boxes[burst:]
    assert len(during) == burst  # the cup's new embedding and burst votes
    assert all(x2 - x1 < 100 for x1, _, x2, _ in during)  # none of them the couch
    assert len(pipe._entities) == 2  # the cup kept its track when the head turned


def bearing_of(pipe, det):
    """The base-frame bearing of a box centre with the head at rest (the attention target)."""
    x1, y1, x2, y2 = det.box
    return pipe._camera(W, H).bearing((x1 + x2) / 2, (y1 + y2) / 2)


def test_with_crops_the_attention_bearing_picks_the_held_cup_not_the_couch(tmp_path):
    """Review of PR 188: on look-down crop runs around a wrongly picked couch, the held cup was
    left aside and never embedded. The target's bearing picks the cup, and the crop follows
    it."""
    cfg = CFG.vision.model_copy(
        update={
            "object_gate": NO_GATE,
            "object_categories": ["cup", "couch"],
        }
    )
    det = CropAwareDetector([COUCH])
    app = RecordingAppearance()
    _, _, pipe, _, _ = vision_rig(
        tmp_path, lambda i: [], vision=cfg, object_detector=det, appearance_embedder=app
    )
    img = marked_image()
    for i in range(round(6 / DT)):  # the couch alone, embedded once
        pipe.process(Frame(img, i * DT))
    assert len(app.boxes) == 1
    det.things = [COUCH, HELD_CUP]
    pipe.governor.set_active("objects", True, bearing_of(pipe, HELD_CUP))
    start = len(det.shapes)
    for i in range(round(6 / DT), round(16 / DT)):
        pipe.process(Frame(img, i * DT))
    during = app.boxes[1:]
    assert len(during) == 1 and during[0][2] - during[0][0] < 100  # the cup, not the couch
    assert pipe.stats()["object_input"]["crop_by"]["shown"] >= 1
    assert any(shape != (H, W) for shape in det.shapes[start:])  # crop runs happened
    assert skips(pipe)["episode"] > 0  # the couch waited
    pipe.governor.set_active("objects", False)


def test_a_flickering_track_keeps_its_age_across_rebirths(tmp_path):
    """Review of PR 188: a cup whose track is reborn every 2 s (1.8 s median life on the robot)
    never reached the minimum age. A rebirth carries the age, not the identity or the votes."""
    pipe, app = rig(tmp_path, object_lost_s=0.5, object_lost_periods=1.0)
    created, first = set(), None
    for i in range(12):
        pipe.object_detector.dets = [] if i % 3 == 2 else [CUP]  # seen twice, missed once
        run(pipe, i, i + 1)
        created |= set(pipe._entities)
        if app.boxes and first is None:
            first = i
    assert len(created) >= 3  # the track was reborn
    # No track lived past 1 s, but the reborn one at 3 s carries the age from 0 s: embedded
    # at its first confirmed run.
    assert first == 4 and runs(pipe)["new"] == 1
    assert all(s.born == 0.0 for s in pipe._entities.values())


def flicker_rig(tmp_path, **kw):
    """A cup whose track is reborn every 3 runs (seen twice, missed once), at 1 Hz."""
    return rig(tmp_path, object_lost_s=0.5, object_lost_periods=1.0, **kw)


def flicker_run(pipe, start, stop, det=CUP):
    created = set()
    for i in range(start, stop):
        pipe.object_detector.dets = [] if i % 3 == 2 else [det]
        run(pipe, i, i + 1)
        created |= set(pipe._entities)
    return created


def test_a_track_reborn_again_and_again_costs_one_embedding(tmp_path):
    """Owner decision 2026-11-05: a rebirth carries the last embedding too, so an unchanged
    box is not embedded again at each rebirth; the world gets the carried embedding."""
    pipe, app = flicker_rig(tmp_path)
    got = []
    pipe.bus.subscribe("percept.entity", lambda m: got.append(m.payload))
    created = flicker_run(pipe, 0, 30)
    assert len(created) >= 8 and len(app.boxes) == 1
    embedded = [o for o in got if o.kind == "object" and o.stamp > MIN_AGE + 2]
    assert embedded and all(o.embedding is not None for o in embedded)


def test_a_reborn_track_whose_box_changed_shape_is_embedded_again(tmp_path):
    pipe, app = flicker_rig(tmp_path)
    flicker_run(pipe, 0, 9)
    assert len(app.boxes) == 1
    wider = Detection((85, 112, 175, 152), 0.8, "cup")  # same centre, aspect x2.25
    flicker_run(pipe, 9, 12, det=wider)
    assert runs(pipe)["changed"] == 1 and len(app.boxes) == 2


def test_a_rebirth_carries_no_identity_and_no_votes(tmp_path):
    pipe, app = flicker_rig(tmp_path)
    mug = enroll_mug(pipe)
    run(pipe, 0, 10)  # seen on every run: recognised
    ((old_id, old),) = pipe._entities.items()
    assert old.owner_id == mug.entity_id and old.votes >= CFG.identity.votes_required
    pipe.object_detector.dets = []
    run(pipe, 10, 11)  # missed: the track ends
    pipe.object_detector.dets = [CUP]
    run(pipe, 11, 13)  # reborn (tentative), then confirmed with the carried state
    ((track_id, reborn),) = pipe._entities.items()
    assert track_id != old_id
    assert reborn.born == 0.0 and reborn.embedding is not None  # age and embedding carried
    assert reborn.owner_id is None  # the identity is not: it votes again
    assert reborn.votes <= 1  # this run's burst vote at most, none carried


def test_a_rebirth_far_away_or_too_late_starts_a_new_age(tmp_path):
    pipe, _ = rig(tmp_path, object_lost_s=0.5, object_lost_periods=1.0)
    run(pipe, 0, 2)
    pipe.object_detector.dets = []
    run(pipe, 2, 3)
    pipe.object_detector.dets = [SIDE_CUP]  # more than rebirth_distance_heights away
    run(pipe, 3, 5)
    (state,) = pipe._entities.values()
    assert state.born == 3.0
    late = int(CFG.vision.rebirth_window_s) + 2
    pipe.object_detector.dets = []
    run(pipe, 5, 5 + late)
    pipe.object_detector.dets = [SIDE_CUP]
    run(pipe, 5 + late, 7 + late)
    (state,) = pipe._entities.values()
    assert state.born == 5.0 + late


def test_a_burst_stalled_past_the_vote_age_starts_again(tmp_path):
    """Review of PR 188: a track whose burst was cut by an episode on another track kept
    counting its old votes, which the matcher had already dropped."""
    pipe, app = rig(tmp_path, dets=[SIDE_CUP], record="active", object_crop=NO_CROP)
    run(pipe, 0, int(MIN_AGE) + 2)
    (side,) = pipe._entities.values()
    assert 0 < side.votes < CFG.vision.embedding_burst  # mid-burst
    pipe.object_detector.dets = [SIDE_CUP, CENTRE_CUP]
    pipe.governor.set_active("objects", True)  # a look at the centre cup
    stall = int(CFG.identity.vote_max_age_s) + 2
    run(pipe, int(MIN_AGE) + 2, int(MIN_AGE) + 2 + stall)
    pipe.governor.set_active("objects", False)
    side_before = side.last_embed
    count = sum(1 for b in app.boxes if b[0] < 50)
    run(pipe, int(MIN_AGE) + 2 + stall, int(MIN_AGE) + 2 + stall + 20)
    assert side.last_embed > side_before
    after = sum(1 for b in app.boxes if b[0] < 50) - count
    assert after == CFG.vision.embedding_burst  # a whole new burst, not the remainder


def test_a_shown_track_young_at_the_start_of_the_look_waits_for_its_age(tmp_path):
    pipe, app = rig(tmp_path, dets=[CENTRE_CUP], object_crop=NO_CROP)
    run(pipe, 0, 2)  # confirmed at 1 s: attention can start the look
    pipe.governor.set_active("objects", True)
    run(pipe, 2, int(MIN_AGE))
    assert app.boxes == [] and skips(pipe)["young"] >= 2
    run(pipe, int(MIN_AGE), int(MIN_AGE) + 1)
    assert runs(pipe)["new"] == 1  # during the look, at the minimum age
    pipe.governor.set_active("objects", False)


def test_the_same_stamps_give_the_same_counts_twice(tmp_path):
    """Replay determinism (SPEC 14): the age and the rebirth carry use capture stamps only."""

    def once(path):
        pipe, app = rig(path, record="active", object_categories=["cup", "bottle"])
        for i in range(30):
            pipe.object_detector.dets = [CUP, flicker(i)]
            run(pipe, i, i + 1)
        return skips(pipe), runs(pipe), app.boxes

    first = once(tmp_path / "a")
    assert first == once(tmp_path / "b")
    assert first[0]["young"] > 0 and first[1]["new"] == 1
