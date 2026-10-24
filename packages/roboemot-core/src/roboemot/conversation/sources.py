"""Runtime-side lookups for the conversation worker (SPEC 8 stage 6, 12).

The worker lives in core and stays robot-agnostic; these closures read the world model, the
attention selector and the registry for the two summaries in the system prompt: who is talking
(display name and confirmed preferences) and what is in view. A missing record or a registry
error gives no summary rather than an error. Nothing here returns an id or an embedding.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from roboemot.attention import AttentionSelector
from roboemot.conversation.summaries import one_line, profile_summary, scene_summary
from roboemot.world import Entry, WorldModel
from roboemot.world.types import UNKNOWN_IDENTITY

log = logging.getLogger(__name__)


def attended_person(selector: AttentionSelector, world: WorldModel) -> Entry | None:
    """The world entry of the person the robot is attending to, or None."""
    target = selector.target
    entry = world.get(target.entry_id) if target is not None and target.entry_id else None
    return entry if entry is not None and entry.kind == "person" else None


def entry_name(registry: Any, entry: Entry) -> str | None:
    """The registry display name of a world entry, only for a record the owner confirmed
    (``status == "active"``): a pending record's name was typed or imported and nobody vouched
    for it. None for an unknown, missing or pending record, or without a registry. The one
    lookup behind the scene summary and the ``look_at`` choices."""
    if entry.kind == "person":
        return person_name(registry, entry.identity)
    if registry is None or entry.identity in ("", UNKNOWN_IDENTITY):
        return None
    try:
        record = registry.get_entity(entry.identity)
    except Exception:  # a record that is gone: describe the entry by its kind
        return None
    return record.display_name if record.status == "active" else None


def person_name(registry: Any, person_id: str | None) -> str | None:
    """The display name of a confirmed (``active``) person record, else None. The same rule as
    ``entry_name``, for callers that hold a registry id instead of a world entry."""
    if registry is None or person_id in (None, "", UNKNOWN_IDENTITY):
        return None
    try:
        record = registry.get_person(person_id)
    except Exception:
        return None
    return record.display_name if record.status == "active" else None


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
        return entry_name(registry, entry)

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


def registry_names(registry: Any, value_cap: Callable[[], int]) -> Callable[[], list[str]]:
    """``known_names()``: every display name in the registry (people, animals, objects, any
    status), plus its one-line form cut to the summary cap, which is how it reads in the
    summaries. For ``privacy.pseudonyms_to_cloud``; an error propagates, so the worker never
    sends without the names it should have replaced."""

    def names() -> list[str]:
        if registry is None:
            return []
        cap = value_cap()
        out: dict[str, None] = {}
        for row in registry.list():
            out.setdefault(row.display_name)
            out.setdefault(one_line(row.display_name, cap))
        return [n for n in out if n]

    return names


def speaker_source(selector: AttentionSelector, world: WorldModel) -> Callable[[], str | None]:
    """``speaker()``: the registry id of the attended person when recognised, else None."""

    def speaker() -> str | None:
        entry = attended_person(selector, world)
        return None if entry is None or entry.identity in ("", UNKNOWN_IDENTITY) else entry.identity

    return speaker
