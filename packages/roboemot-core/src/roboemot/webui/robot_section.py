"""The robot's own settings section: entry-point plugins for the web UI (SPEC 4A, 11A, 12).

A robot package registers one object in the ``roboemot.settings`` entry-point group, keyed by the
manifest ``robot.id`` (same pattern as ``roboemot.diagnostics`` and ``roboemot.world``). Its
``section(stack, manifest)`` returns a ``RobotSection``: read-only hardware rows it builds with
``config.catalogue.descriptor`` and the config paths it *claims*, which the web UI then lists under
the robot instead of the core sections. Core stays robot-agnostic and never writes a plugin row.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol, TypedDict, cast, runtime_checkable

from roboemot.config.catalogue import CONTAINERS, DESCRIPTOR_KEYS, FieldDescriptor, seal
from roboemot.redact import REDACTED

if TYPE_CHECKING:
    from roboemot.manifest import Manifest

    from .settings import StackPort

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "roboemot.settings"
PATH_ROOT = "robot"
RESTART = "restart"
RESTART_EFFECTS = ("zero_pose", "webrtc_drop", "kill_on_timeout")
PROTECTED_CLAIMS = ("safety", "webui", "privacy", "llm_routes")


class RobotBlock(TypedDict):
    """The ``robot`` key of ``GET /api/settings``."""

    id: str
    title: str
    actions: list[str]
    restart_effects: list[str]
    fields: list[FieldDescriptor]


@dataclass(frozen=True)
class RobotSection:
    """What a plugin contributes. ``fields`` are read-only rows under ``robot.<id>.``; ``claims``
    are config paths (a path and everything below it) that move from the core list into the
    robot's, still written by path through the normal settings endpoints. ``actions`` is set by
    core from the plugin's capabilities, with ``restart_effects`` (ids from ``RESTART_EFFECTS``);
    a value a plugin sets for either is ignored."""

    title: str
    fields: list[FieldDescriptor] = field(default_factory=list)
    claims: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    restart_effects: tuple[str, ...] = ()


class SettingsPlugin(Protocol):
    """The object behind a ``roboemot.settings`` entry point."""

    def section(self, stack: StackPort, manifest: Manifest) -> RobotSection:
        """The robot's section for the current configuration; called on every listing."""
        ...


@runtime_checkable
class Restartable(Protocol):
    """Optional: a plugin that can restart RoboEmot through its robot's own supervisor. Core then
    offers the ``restart`` action (button, PIN, confirm dialog) while ``restart_available()``
    holds, lists ``restart_effects()`` in the dialog in its own words and calls ``restart()`` only
    after the web response has been sent, because the restart ends this very process."""

    def restart_available(self) -> bool:
        """Whether a restart can be done now (checked on every listing and again on the call)."""
        ...

    def restart_effects(self) -> tuple[str, ...]:
        """What the owner should know first: ids from ``RESTART_EFFECTS``; others are dropped."""
        ...

    def restart(self) -> None:
        """Ask the robot's supervisor to restart RoboEmot. May block, may raise."""
        ...


def load_settings_plugin(robot_id: str) -> SettingsPlugin | None:
    """The plugin registered for ``robot_id``, or ``None``. A missing or broken plugin is logged
    (type only, a message could carry a value) and gives no robot section."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == robot_id:
            try:
                return cast(SettingsPlugin, ep.load())
            except Exception as exc:
                log.warning("settings plugin %s: %s", robot_id, type(exc).__name__)
                return None
    log.info("no settings plugin installed for %s", robot_id)
    return None


def restart_hook(plugin: SettingsPlugin | None) -> Callable[[], None] | None:
    """The plugin's ``restart`` when it is ``Restartable`` and available now, else ``None``. A
    plugin whose availability check raises is treated as unavailable (logged by type only)."""
    if not isinstance(plugin, Restartable):
        return None
    try:
        return plugin.restart if plugin.restart_available() else None
    except Exception as exc:
        log.warning("settings plugin restart check: %s", type(exc).__name__)
        return None


def _restart_effects(plugin: Restartable) -> tuple[str, ...]:
    """The plugin's effect ids that core knows how to word; a failing plugin lists none."""
    try:
        return tuple(e for e in plugin.restart_effects() if e in RESTART_EFFECTS)
    except Exception as exc:
        log.warning("settings plugin restart effects: %s", type(exc).__name__)
        return ()


def claimed(path: str, claims: tuple[str, ...]) -> bool:
    """Whether ``path`` is one of ``claims`` or below one."""
    return any(path == c or path.startswith(c + ".") for c in claims)


def _vet_row(row: object, prefix: str) -> FieldDescriptor | None:
    """A copy of a plugin row, read-only and sealed again, or ``None`` when it is not a full
    descriptor under ``prefix`` (logged by path only). The plugin's own dict is never changed."""
    if not isinstance(row, dict) or set(row) != DESCRIPTOR_KEYS:
        path = row.get("path") if isinstance(row, dict) else None
        log.warning("settings plugin row dropped: not a full descriptor (%s)", path)
        return None
    if not isinstance(row["path"], str) or not row["path"].startswith(prefix):
        # also covers a row that repeats a core path, claimed or not: it would show twice
        log.warning("settings plugin row dropped: outside %s (%s)", prefix, row["path"])
        return None
    out = cast(FieldDescriptor, {**row, "writable": False})
    if out["kind"] not in CONTAINERS:
        seal(out, out["path"].rsplit(".", 1)[-1])
        if row["secret"]:  # a plugin's own secret flag only ever adds redaction
            out["secret"] = True
            out["value"] = REDACTED if out["value"] is not None else None
            out["default"] = REDACTED if out["default"] is not None else None
    return out


def build_section(
    plugin: SettingsPlugin | None, stack: StackPort, manifest: Manifest | None
) -> tuple[RobotSection, str] | None:
    """The plugin's section and the robot id, or ``None`` (no plugin, no manifest, or a section
    that raised: logged by type only). Rows that are not full descriptors under ``robot.<id>.``
    are dropped; kept rows are copies, read-only and redacted again by core. ``actions`` comes
    from the plugin's capabilities, not from what it returns."""
    if plugin is None or manifest is None:
        return None
    robot_id = manifest.robot.id
    try:
        section = plugin.section(stack, manifest)
        prefix = f"{PATH_ROOT}.{robot_id}."
        rows = [r for r in (_vet_row(row, prefix) for row in section.fields) if r is not None]
        hook = restart_hook(plugin)
        actions = (RESTART,) if hook else ()
        effects = _restart_effects(plugin) if hook else ()
        return RobotSection(section.title, rows, tuple(section.claims), actions, effects), robot_id
    except Exception as exc:
        log.warning("settings plugin %s section: %s", robot_id, type(exc).__name__)
        return None


def vet_claims(claims: tuple[str, ...], fields: list[FieldDescriptor]) -> tuple[str, ...]:
    """The claims core honours. A claim is refused when it is empty, at or under a protected
    section, or matches a private or admin-only row, so a plugin cannot move those out of core.
    A claim that matches nothing is dropped. Warnings carry the claim only."""
    kept = []
    for claim in claims:
        if not isinstance(claim, str) or not claim or claimed(claim, PROTECTED_CLAIMS):
            log.warning("settings claim refused: %r", claim)
            continue
        hits = [f for f in fields if claimed(f["path"], (claim,))]
        if any(f["private"] or f["admin_only"] for f in hits):
            log.warning("settings claim refused: %s holds admin or private rows", claim)
        elif not hits:
            log.warning("settings claim matches nothing: %s", claim)
        else:
            kept.append(claim)
    return tuple(kept)
