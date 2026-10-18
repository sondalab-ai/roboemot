"""The robot's own settings section: entry-point plugins for the web UI (SPEC 4A, 11A, 12).

A robot package registers one object in the ``roboemot.settings`` entry-point group, keyed by the
manifest ``robot.id`` (same pattern as ``roboemot.diagnostics`` and ``roboemot.world``). Its
``section(stack, manifest)`` returns a ``RobotSection``: read-only hardware rows it builds with
``config.catalogue.descriptor`` and the config paths it *claims*, which the web UI then lists under
the robot instead of the core sections. Core stays robot-agnostic and never writes a plugin row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from roboemot.config.catalogue import FieldDescriptor

if TYPE_CHECKING:
    from roboemot.manifest import Manifest

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "roboemot.settings"
PATH_ROOT = "robot"


class RobotBlock(TypedDict):
    """The ``robot`` key of ``GET /api/settings``."""

    id: str
    title: str
    fields: list[FieldDescriptor]


@dataclass(frozen=True)
class RobotSection:
    """What a plugin contributes. ``fields`` are read-only rows under ``robot.<id>.``; ``claims``
    are config paths (a path and everything below it) that move from the core list into the
    robot's, still written by path through the normal settings endpoints."""

    title: str
    fields: list[FieldDescriptor] = field(default_factory=list)
    claims: tuple[str, ...] = ()


class SettingsPlugin(Protocol):
    """The object behind a ``roboemot.settings`` entry point."""

    def section(self, stack: Any, manifest: Manifest) -> RobotSection:
        """The robot's section for the current configuration; called on every listing."""
        ...


def load_settings_plugin(robot_id: str) -> SettingsPlugin | None:
    """The plugin registered for ``robot_id``, or ``None``. A missing or broken plugin is logged
    (type only, a message could carry a value) and gives no robot section."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == robot_id:
            try:
                return ep.load()  # type: ignore[no-any-return]
            except Exception as exc:
                log.warning("settings plugin %s: %s", robot_id, type(exc).__name__)
                return None
    log.info("no settings plugin installed for %s", robot_id)
    return None


def claimed(path: str, claims: tuple[str, ...]) -> bool:
    """Whether ``path`` is one of ``claims`` or below one."""
    return any(path == c or path.startswith(c + ".") for c in claims)


def build_section(
    plugin: SettingsPlugin | None, stack: Any, manifest: Manifest | None
) -> tuple[RobotSection, str] | None:
    """The plugin's section and the robot id, or ``None`` (no plugin, no manifest, or a section
    that raised: logged by type only). Rows outside ``robot.<id>.`` are dropped and every kept
    row is forced read-only, so a plugin cannot offer a write core would not route."""
    if plugin is None or manifest is None:
        return None
    robot_id = manifest.robot.id
    try:
        section = plugin.section(stack, manifest)
        prefix = f"{PATH_ROOT}.{robot_id}."
        rows = [r for r in section.fields if r["path"].startswith(prefix)]
        if len(rows) != len(section.fields):
            log.warning("settings plugin %s: rows outside %s dropped", robot_id, prefix)
        for row in rows:
            row["writable"] = False
        return RobotSection(section.title, rows, section.claims), robot_id
    except Exception as exc:
        log.warning("settings plugin %s section: %s", robot_id, type(exc).__name__)
        return None
