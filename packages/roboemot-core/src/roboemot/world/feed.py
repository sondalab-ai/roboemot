"""Bus adapter: folds ``percept.entity`` observations into a world model (SPEC 5A, 11).

Sources publish ``Observation`` payloads in the sensor frame with the joint state at capture
in the message ``meta`` (``joint_state``); the world model rotates them into the base frame.
Subscribers run synchronously in publish order, so replays stay deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

from ..bus import Bus, Message
from .model import WorldModel
from .types import Observation


def feed_world(bus: Bus, world: WorldModel) -> Callable[[], None]:
    """Subscribe ``world`` to ``percept.entity``; returns the unsubscribe function.

    Each ``Observation`` payload goes to ``world.observe`` with the ``joint_state`` of the
    message meta, so the model rotates it into the base frame. Other payloads are ignored.
    """

    def on_entity(msg: Message) -> None:
        if isinstance(msg.payload, Observation):
            world.observe(msg.payload, msg.meta.get("joint_state"))

    return bus.subscribe("percept.entity", on_entity)
