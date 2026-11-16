"""The ``roboemot.skills`` entry-point group: voice tools as plugins (SPEC 11, Voice tools).

Each entry point is a factory ``(ToolHost) -> list of @skill functions``. Unlike every other
``roboemot.*`` group it is not keyed by ``robot.id``: every entry point loads. Core's own tools
come through this group too. A factory that cannot load, raises or returns anything but
``@skill`` functions is logged by its entry-point name and skipped, so one broken plugin never
stops the run. Logs carry entry-point and skill names and exception types, never exception text.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from .defaults import SkillEnv, default_skills
from .host import ToolHost
from .registry import Skill, SkillRegistry, skill_of

log = logging.getLogger(__name__)

SKILLS_GROUP = "roboemot.skills"
CORE_DISTRIBUTION = "roboemot-core"


@dataclass(frozen=True)
class PluginSkills:
    """The skills one entry point gave: its name, the distribution that registered it (None
    when unknown) and its ``@skill`` functions."""

    name: str
    distribution: str | None
    skills: list[Callable[..., Any]]


def _normalized(dist: str) -> str:
    """A distribution name in its canonical form (``roboemot_core`` -> ``roboemot-core``)."""
    return re.sub(r"[-_.]+", "-", dist).lower()


def _distribution(ep: Any) -> str | None:
    """The canonical name of the distribution that registered ``ep``, or None."""
    return _normalized(ep.dist.name) if ep.dist is not None else None


def _skills_of(ep: Any, host: ToolHost) -> list[Callable[..., Any]] | None:
    """Call the factory of ``ep`` with ``host``; its skill functions, or None (logged) when it
    cannot load, raises or returns anything but a list of ``@skill`` functions."""
    try:
        made = ep.load()(host)
    except Exception as exc:  # the text may hold anything: type only
        log.warning("skill plugin %r skipped: %s", ep.name, type(exc).__name__)
        return None
    if not isinstance(made, list | tuple) or not all(
        callable(fn) and not isinstance(fn, Skill) and skill_of(fn) is not None for fn in made
    ):
        log.warning("skill plugin %r skipped: not a list of @skill functions", ep.name)
        return None
    return list(made)


def load_plugin_skill_sets(host: ToolHost) -> list[PluginSkills]:
    """Every entry point of :data:`SKILLS_GROUP` called with ``host``, sorted by entry-point
    name, each with the distribution that registered it. Broken ones are logged and left out.
    Load once per run and hand the result to :func:`build_skill_registry` and
    :func:`warn_hidden_core_skills`."""
    eps = sorted(importlib.metadata.entry_points(group=SKILLS_GROUP), key=lambda ep: ep.name)
    out = []
    for ep in eps:
        skills = _skills_of(ep, host)
        if skills is not None:
            out.append(PluginSkills(ep.name, _distribution(ep), skills))
    return out


def load_skill_plugins(host: ToolHost) -> list[Callable[..., Any]]:
    """The ``@skill`` functions of every ``roboemot.skills`` plugin, in entry-point name order;
    a broken plugin is logged and skipped (SPEC 11, Voice tools)."""
    return [fn for plugin in load_plugin_skill_sets(host) for fn in plugin.skills]


def build_skill_registry(
    env: SkillEnv, host: ToolHost, plugins: Sequence[PluginSkills] | None = None
) -> SkillRegistry:
    """The run's registry: the default skills over ``env`` plus every plugin skill. A plugin
    skill whose name is already taken is logged and skipped. ``plugins`` is what
    :func:`load_plugin_skill_sets` returned (loaded now when None)."""
    registry = default_skills(env)
    loaded = load_plugin_skill_sets(host) if plugins is None else plugins
    for plugin in loaded:
        for fn in plugin.skills:
            name = fn.skill.name
            if registry.get(name) is not None:
                log.warning("skill plugin %r: skill %s is already registered, skipped",
                            plugin.name, name)  # fmt: skip
                continue
            registry.add(fn)
    return registry


def warn_hidden_core_skills(
    plugins: Sequence[PluginSkills], llm_skills: Collection[str]
) -> list[str]:
    """Log one warning naming each skill of core's own ``roboemot.skills`` entry points that
    ``llm_skills`` leaves out: a saved ``conversation.llm_skills`` from before a tool shipped
    hides it. Returns those names (empty: no warning). Called once at startup."""
    hidden = [
        fn.skill.name
        for plugin in plugins
        if plugin.distribution == CORE_DISTRIBUTION
        for fn in plugin.skills
        if fn.skill.name not in llm_skills
    ]
    if hidden:
        log.warning(
            "conversation.llm_skills leaves out core voice tools, the model is not offered: %s",
            ", ".join(hidden),
        )
    return hidden
