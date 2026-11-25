"""The head holds still while a surroundings snapshot is taken (SPEC 8, Surroundings
snapshots; owner 2026-11-25): the gaze dofs pinned at EXPRESSION, above gaze, breathing and
sweeps (IDLE), below reflexes and safety; released when the hold ends."""

from __future__ import annotations

from roboemot.motion import Priority
from roboemot.runtime.ticks import HOLD_SOURCE, SnapshotHold


class Loop:
    def __init__(self) -> None:
        self.fed, self.cleared = [], []
        self.now_pose = {"head.yaw": 0.3, "head.pitch": -0.1, "body.yaw": 0.5, "antenna.l": 1.0}

    def pose(self):
        return dict(self.now_pose)

    def feed(self, cmd):
        self.fed.append(cmd)

    def clear(self, source):
        self.cleared.append(source)


DOFS = ["head.yaw", "head.pitch", "body.yaw"]


def test_the_hold_pins_the_gaze_dofs_where_they_were_and_releases_them():
    loop, hold = Loop(), SnapshotHold()
    assert hold.tick(10.0, 12.0, loop, DOFS)
    loop.now_pose["head.yaw"] = 0.9  # whatever moves later, the hold keeps the first pose
    assert hold.tick(10.1, 12.0, loop, DOFS)
    first, second = loop.fed
    assert first.source == HOLD_SOURCE and first.priority == Priority.EXPRESSION
    assert first.targets == {"head.yaw": 0.3, "head.pitch": -0.1, "body.yaw": 0.5}
    assert second.targets == first.targets and first.expires == 12.0
    assert not hold.tick(12.5, None, loop, DOFS)
    assert loop.cleared == [HOLD_SOURCE]
    assert not hold.tick(12.6, None, loop, DOFS) and loop.cleared == [HOLD_SOURCE]
