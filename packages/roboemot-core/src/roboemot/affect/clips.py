"""Recorded clip libraries served by robot plugins (SPEC 7, Recorded clips; SPEC 4A).

A robot package registers a factory in the ``roboemot.expression_clips`` entry-point group under
its manifest ``robot.id``. The factory takes nothing or the manifest and returns a read-only
mapping of clip name to an entry in the expression keyframe format. :func:`load_clip_library`
loads it once and validates every clip against the entry rules and the manifest's angle dofs;
a robot without a library gets an empty one.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Collection, Mapping
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..config.schema import ExpressionEntry, check_keyframes

if TYPE_CHECKING:
    from ..manifest import Manifest

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "roboemot.expression_clips"

ClipLibrary = Mapping[str, ExpressionEntry]


class ClipLibraryError(ValueError):
    """A clip library that cannot be built or breaks the entry rules; refused with the manifest."""


def validate_clip_library(
    raw: Mapping[str, Any], channels: Collection[str]
) -> dict[str, ExpressionEntry]:
    """Every clip of ``raw`` as an :class:`ExpressionEntry` with channels over ``channels``.

    Items may be entries already or plain dicts. A clip needs channels (no variants), obeys the
    keyframe rules (start and end at 0, one end time, at most ``EXPRESSION_MAX_CYCLE_S``) and
    drives only the manifest's angle dofs. Raises :class:`ClipLibraryError` naming the clip.
    """
    out: dict[str, ExpressionEntry] = {}
    for name, item in raw.items():
        try:
            entry = (
                item if isinstance(item, ExpressionEntry) else ExpressionEntry.model_validate(item)
            )
            if entry.variants:
                raise ValueError("a clip is channels, not variants")
            check_keyframes(entry.channels)
        except (ValidationError, ValueError) as exc:
            raise ClipLibraryError(f"clip {name!r}: {exc}") from exc
        for channel in entry.channels:
            if channel not in channels:
                raise ClipLibraryError(
                    f"clip {name!r}: unknown channel {channel!r}, expected one of "
                    f"{sorted(channels)}"
                )
        out[name] = entry
    return out


def load_clip_library(manifest: Manifest) -> dict[str, ExpressionEntry]:
    """The validated clip library registered for ``manifest.robot.id``, or an empty one.

    The factory is called with the manifest when it accepts an argument, else with none. A
    plugin that fails to build or whose clips break the rules raises :class:`ClipLibraryError`.
    """
    robot_id = manifest.robot.id
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name != robot_id:
            continue
        factory = ep.load()
        try:
            takes_manifest = bool(inspect.signature(factory).parameters)
            raw = factory(manifest) if takes_manifest else factory()
        except Exception as exc:
            raise ClipLibraryError(
                f"clip library of {robot_id}: {type(exc).__name__}: {exc}"
            ) from exc
        return validate_clip_library(raw, manifest.channels)
    log.info("no clip library installed for %s", robot_id)
    return {}


__all__ = [
    "ENTRY_POINT_GROUP",
    "ClipLibrary",
    "ClipLibraryError",
    "load_clip_library",
    "validate_clip_library",
]
