"""SPEC 5, 6: a recognised still object keeps its identity at the scene-change gate's cadence.

Defaults throughout (gate and crop on, ``identity.votes_required`` 3, ``vote_max_age_s`` 10 s),
8 Hz frames with a still head. A cat in view for the first 10 s keeps the gate on its fixed
schedule, so the cup on the table gets its burst of votes and is recognised; then the cat leaves
and the cup's object runs fall to the gate floor, about 10 s apart.
"""

from __future__ import annotations

import numpy as np
from roboemot.attention.interest import _known
from roboemot.vision import Detection, Frame
from roboemot.world import WorldModel, feed_world
from test_vision import APP_MODEL, CFG, H, W, axis, rig
from test_vision_object_crop import CropAwareDetector, face_script, marked_image
from test_vision_object_gate import DT, SPEC, joint

# Cosine 0.7 to the record: a match (above identity.match_threshold) that auto-enrollment's
# stricter known_match_threshold would not attach on its own, as for a real DINOv2 view.
LOOK = 0.7 * axis(3) + np.sqrt(1 - 0.7**2) * axis(4)
TABLE = Detection((60, 130, 100, 170), 0.8, "cup")  # left of the face crop
LIFTED = Detection((130, 60, 170, 100), 0.8, "cup")  # in front of the chest, inside the crop
CAT = Detection((200, 150, 300, 230), 0.8, "cat")
CAT_LEAVES_S = 10.0
STILL_UNTIL_S = 44.0  # more than three gate floors after the cat left
LIFT_S = 44.3  # the cup is picked up just before a crop run of the look


class SameMug:
    model_id = APP_MODEL

    def __init__(self, look=LOOK):
        self.look = look

    def embed(self, img, box):
        return self.look


def scene(tmp_path):
    det = CropAwareDetector([TABLE, CAT])
    app = SameMug()
    reg, bus, pipe, _, _ = rig(
        tmp_path, face_script, object_detector=det, appearance_embedder=app, frame_spec=SPEC
    )
    world = WorldModel(CFG.world, SPEC, bus)
    landed = []
    feed_world(bus, world, lambda obs, res: landed.append((obs, res)))
    mug = reg.enroll_entity("object", "Mug", np.stack([axis(3)] * 5), APP_MODEL, category="cup")
    return pipe, det, app, world, landed, mug


def cup_entries(landed):
    return [res.entry for obs, res in landed if obs.kind == "object" and res.entry is not None]


def drive(pipe, det, start_s, stop_s):
    """Process 8 Hz frames in ``[start_s, stop_s)``; the cat leaves at ``CAT_LEAVES_S``."""
    img = marked_image()
    for i in range(round(start_s / DT), round(stop_s / DT)):
        stamp = i * DT
        if stamp >= CAT_LEAVES_S and CAT in det.things:
            det.things = [d for d in det.things if d != CAT]
        pipe.process(Frame(img, stamp, joint()))


def test_a_still_recognised_cup_keeps_its_identity_across_gate_floors(tmp_path):
    pipe, det, _, _, landed, mug = scene(tmp_path)
    drive(pipe, det, 0.0, CAT_LEAVES_S)
    (cup,) = [s for s in pipe._entities.values() if s.kind == "object"]
    assert cup.owner_id == mug.entity_id
    landed.clear()
    drive(pipe, det, CAT_LEAVES_S, STILL_UNTIL_S)
    assert pipe.stats()["object_gate"]["triggered"]["floor"] >= 3
    assert cup.owner_id == mug.entity_id
    assert {obs.identity for obs, _ in landed if obs.kind == "object"} == {mug.entity_id}


def test_a_still_cup_picked_up_and_shown_is_still_the_known_mug(tmp_path):
    """Regression (handoff 2026-11-03, section 3): the lapsed cup was published unknown, so the
    lifted cup, past the world's match gate, became a new unknown entry: novelty 1 instead of
    ``familiar_novelty``, a long look at a mug the robot knows."""
    pipe, det, _, world, landed, mug = scene(tmp_path)
    drive(pipe, det, 0.0, STILL_UNTIL_S)
    pipe.governor.set_active("objects", True)  # the person shows it: a look starts
    drive(pipe, det, STILL_UNTIL_S, LIFT_S)
    det.things = [LIFTED]
    landed.clear()
    first = len(det.shapes)
    drive(pipe, det, LIFT_S, LIFT_S + 1.0)
    assert det.shapes[first] != (H, W)  # the first run after the lift is a crop run (pickup)
    shown = cup_entries(landed)
    assert shown and all(_known(e) for e in shown)
    assert all(_known(e) for e in world.entries if e.kind == "object")
    pipe.governor.set_active("objects", False)


def test_another_enrolled_object_in_the_same_place_ends_the_identity_at_gate_cadence(tmp_path):
    pipe, det, app, _, _, mug = scene(tmp_path)
    reg = pipe.matcher.registry
    other = reg.enroll_entity("object", "Cup", np.stack([axis(6)] * 5), APP_MODEL, category="cup")
    drive(pipe, det, 0.0, CAT_LEAVES_S + 15.0)
    (cup,) = [s for s in pipe._entities.values() if s.kind == "object"]
    assert cup.owner_id == mug.entity_id
    app.look = axis(6)  # swapped for the other enrolled cup, same box
    drive(pipe, det, CAT_LEAVES_S + 15.0, CAT_LEAVES_S + 60.0)
    assert cup.owner_id in (None, other.entity_id)


def test_a_no_match_vote_does_not_end_a_still_cups_identity(tmp_path):
    pipe, det, app, _, _, mug = scene(tmp_path)
    drive(pipe, det, 0.0, CAT_LEAVES_S)
    (cup,) = [s for s in pipe._entities.values() if s.kind == "object"]
    app.look = axis(7)  # a lighting change: no record matches
    drive(pipe, det, CAT_LEAVES_S, STILL_UNTIL_S)
    assert cup.owner_id == mug.entity_id
