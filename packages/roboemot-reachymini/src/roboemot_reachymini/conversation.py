"""App-side lookups for the conversation worker (SPEC 8 stage 6, 12).

The worker lives in core and stays robot-agnostic; these closures read the world model, the
attention selector and the registry for the two summaries in the system prompt: who is talking
(display name and confirmed preferences) and what is in view. A missing record or a registry
error gives no summary rather than an error. Nothing here returns an id, an embedding or an
estimate.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from roboemot.attention import AttentionSelector
from roboemot.conversation import profile_summary, scene_summary
from roboemot.world import Entry, WorldModel
from roboemot.world.types import UNKNOWN_IDENTITY

log = logging.getLogger(__name__)


def attended_person(selector: AttentionSelector, world: WorldModel) -> Entry | None:
    """The world entry of the person the robot is attending to, or None."""
    target = selector.target
    entry = world.get(target.entry_id) if target is not None and target.entry_id else None
    return entry if entry is not None and entry.kind == "person" else None


def profile_source(
    selector: AttentionSelector,
    world: WorldModel,
    registry: Any,
    value_cap: Callable[[], int],
) -> Callable[[int], str | None]:
    """``profile(max_chars)``: the attended, recognised person's name and preferences. A
    record still pending (not confirmed by the owner) is not described."""

    def profile(max_chars: int) -> str | None:
        entry = attended_person(selector, world)
        if entry is None or entry.identity in ("", UNKNOWN_IDENTITY) or registry is None:
            return None
        try:
            record = registry.get_person(entry.identity)
        except Exception as exc:
            log.debug("conversation: no profile (%s)", type(exc).__name__)
            return None
        if record.status != "active":
            return None
        return profile_summary(record.display_name, record.preferences, max_chars, value_cap())

    return profile


def scene_source(
    selector: AttentionSelector,
    world: WorldModel,
    registry: Any,
    stale_s: Callable[[], float],
    value_cap: Callable[[], int],
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[int], str | None]:
    """``scene(max_chars)``: what is in view, the attention target first."""

    def name_of(entry: Entry) -> str | None:
        if registry is None:
            return None
        try:
            if entry.kind == "person":
                return registry.get_person(entry.identity).display_name
            return registry.get_entity(entry.identity).display_name
        except Exception:  # a record that is gone: describe the entry by its kind
            return None

    def scene(max_chars: int) -> str | None:
        target = selector.target
        return scene_summary(
            world.entries,
            now=clock(),
            stale_s=stale_s(),
            target_id=target.entry_id if target is not None else None,
            name_of=name_of,
            max_chars=max_chars,
            value_max_chars=value_cap(),
        )

    return scene


def speaker_source(selector: AttentionSelector, world: WorldModel) -> Callable[[], str | None]:
    """``speaker()``: the registry id of the attended person when recognised, else None."""

    def speaker() -> str | None:
        entry = attended_person(selector, world)
        return None if entry is None or entry.identity in ("", UNKNOWN_IDENTITY) else entry.identity

    return speaker
