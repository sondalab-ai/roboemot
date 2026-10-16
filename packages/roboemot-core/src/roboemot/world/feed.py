"""Bus adapter: folds ``percept.entity`` observations into a world model (SPEC 5A, 11).

Sources publish ``Observation`` payloads in the sensor frame with the joint state at capture
in the message ``meta`` (``joint_state``); the world model rotates them into the base frame.
Subscribers run synchronously in publish order, so replays stay deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

from ..bus import Bus, Message
from .model import WorldModel
from .types import Observation, ObserveResult


def feed_world(
    bus: Bus,
    world: WorldModel,
    on_result: Callable[[Observation, ObserveResult], None] | None = None,
) -> Callable[[], None]:
    """Subscribe ``world`` to ``percept.entity``; returns the unsubscribe function.

    Each ``Observation`` payload goes to ``world.observe`` with the ``joint_state`` of the
    message meta, so the model rotates it into the base frame. ``on_result`` sees every
    observation with its outcome (scanning uses it for unexplained bearings); it runs in the
    publisher's thread. Other payloads are ignored.
    """

    def on_entity(msg: Message) -> None:
        if isinstance(msg.payload, Observation):
            result = world.observe(msg.payload, msg.meta.get("joint_state"))
            if on_result is not None:
                on_result(msg.payload, result)

    return bus.subscribe("percept.entity", on_entity)
