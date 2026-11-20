"""SPEC 5A, 7, 8 (Robot expression): a thing shown to the robot is commented on, only while it
is still shown, at most once per category per cooldown, and with a gesture alone under music."""

from __future__ import annotations

from expressor_rig import Rig
from roboemot.attention.types import SHOWN, AttentionTarget


def target(entry_id: str, category: str, *, priority: str = SHOWN, kind: str = "object"):
    return AttentionTarget(entry_id, kind, 0.0, 0.0, priority, "new", 0.0, category)


def attached() -> Rig:
    rig = Rig()
    rig.expressor.attach()
    return rig


def show(rig: Rig, entry_id: str, category: str) -> None:
    rig.host.bus.publish("attention.target", target(entry_id, category), rig.clock())


def test_shown_object_spoken_with_category():
    rig = attached()
    show(rig, "obj-7c1d", "cup")
    assert rig.expressor.handle(rig.expressor._take()) == "spoken"
    text = rig.worker.turns[0]["text"]
    assert "cup" in text and "obj-7c1d" not in text
    assert rig.worker.turns[0]["profile"] is True


def test_the_same_look_offers_once():
    rig = attached()
    show(rig, "e1", "cup")
    show(rig, "e1", "cup")  # the target held: the same look
    assert rig.counts("shown_object") == {}
    assert rig.expressor.snapshot()["queued"] == 1


def test_look_ended_before_idle_drops_stale():
    rig = attached()
    rig.worker.busy_until = 3.0
    show(rig, "e1", "cup")

    def look_away(t):
        if t >= 2.0:
            person = target("p1", "", priority="person", kind="person")
            rig.host.bus.publish("attention.target", person, t)

    rig.clock.waits.append(look_away)
    assert rig.expressor.handle(rig.expressor._take()) == "dropped_stale"
    assert rig.worker.turns == []


def test_a_cleared_target_ends_the_look():
    rig = attached()
    show(rig, "e1", "cup")
    rig.host.bus.publish("attention.target", None, 0.0)
    assert rig.expressor.handle(rig.expressor._take()) == "dropped_stale"


def test_burst_of_shown_objects_speaks_at_most_once_about_the_last():
    """Review Focus 4: five things shown within 2 s while a turn runs: one comment at most,
    about the thing still shown."""
    rig = attached()
    rig.worker.busy_until = 4.0
    for i, category in enumerate(("cup", "book", "cat", "cup", "phone")):
        rig.clock.t = i * 0.5
        show(rig, f"e{i + 1}", category)
    assert rig.expressor.snapshot()["queued"] == 1
    assert rig.expressor.handle(rig.expressor._take()) == "spoken"
    assert len(rig.worker.turns) == 1 and "phone" in rig.worker.turns[0]["text"]
    counts = rig.counts("shown_object")
    assert counts == {"dropped_replaced": 3, "dropped_cooldown": 1, "spoken": 1}


def test_same_category_once_per_cooldown():
    rig = attached()
    show(rig, "e1", "cup")
    assert rig.expressor.handle(rig.expressor._take()) == "spoken"
    rig.clock.t = 60.0
    rig.host.bus.publish("attention.target", None, 60.0)
    show(rig, "e2", "cup")
    assert rig.expressor.snapshot()["queued"] == 0
    assert len(rig.worker.turns) == 1


def test_existing_reaction_is_whole_response_under_music():
    rig = attached()
    rig.media.is_playing = True
    show(rig, "e1", "cup")
    assert rig.host.expression.requests == []  # shown_object.gesture is ""
    assert rig.counts("shown_object") == {"gesture_only_music": 1}
    assert rig.worker.turns == [] and rig.media.calls == []
