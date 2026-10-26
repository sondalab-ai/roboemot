"""Registry of the ``roboemot.*`` entry-point groups and the core contract of each (SPEC 4A, 12A).

A robot package contributes a capability only through a group listed in :data:`GROUPS`, keyed by
manifest ``robot.id``. A group without a core contract cannot be registered:
``test_robot_hierarchy.py`` fails on any installed ``roboemot.*`` group missing here, and on any
registered object that does not satisfy its contract.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .affect.clips import ENTRY_POINT_GROUP as CLIPS_GROUP
from .diagnostics import ENTRY_POINT_GROUP as DIAGNOSTICS_GROUP
from .install import ENTRY_POINT_GROUP as INSTALLERS_GROUP
from .install import Platform
from .robot_platform import ENTRY_POINT_GROUP as ROBOTS_GROUP
from .robot_platform import RobotPlatform
from .webui.robot_section import ENTRY_POINT_GROUP as SETTINGS_GROUP
from .webui.robot_section import SettingsPlugin
from .world.sources import ENTRY_POINT_GROUP as WORLD_GROUP


@dataclass(frozen=True)
class GroupContract:
    """What an entry point of one group must be.

    ``protocol`` is a runtime_checkable Protocol the (optionally instantiated) object satisfies;
    ``factory_args`` is the number of positional arguments a factory callable must accept (a
    tuple: any one of those counts).
    ``instantiate`` calls the loaded object without arguments first (a class).
    """

    summary: str
    protocol: type | None = None
    factory_args: int | tuple[int, ...] | None = None
    instantiate: bool = False


GROUPS: dict[str, GroupContract] = {
    DIAGNOSTICS_GROUP: GroupContract(
        "factory (Target) -> list[Source]: debug-report sources of the platform (SPEC 12A)",
        factory_args=1,
    ),
    INSTALLERS_GROUP: GroupContract(
        "class implementing install.Platform (SPEC 4A, Packaging)",
        protocol=Platform,
        instantiate=True,
    ),
    WORLD_GROUP: GroupContract(
        "factory (manifest, cfg, session, joint_state) -> WorldPlugin (SPEC 5A)",
        factory_args=4,
    ),
    ROBOTS_GROUP: GroupContract(
        "class implementing RobotPlatform: the factories core builds the robot from (SPEC 4A)",
        protocol=RobotPlatform,
        instantiate=True,
    ),
    SETTINGS_GROUP: GroupContract(
        "object implementing SettingsPlugin: section(stack, manifest) (SPEC 12)",
        protocol=SettingsPlugin,
    ),
    CLIPS_GROUP: GroupContract(
        "factory () or (manifest) -> read-only mapping of clip name to keyframe entry (SPEC 7)",
        factory_args=(0, 1),
    ),
}


def _accepts_positional(fn: Callable[..., Any], count: int) -> bool:
    """Whether ``fn`` can be called with ``count`` positional arguments."""
    try:
        inspect.signature(fn).bind(*([object()] * count))
    except TypeError:
        return False
    except ValueError:
        return True
    return True


def contract_violation(group: str, loaded: Any) -> str | None:
    """Why ``loaded`` (the result of ``ep.load()``) breaks the contract of ``group``, or None."""
    contract = GROUPS.get(group)
    if contract is None:
        return f"group {group} has no core contract in roboemot.plugins.GROUPS"
    obj = loaded
    if contract.instantiate:
        try:
            obj = loaded()
        except Exception as exc:
            return f"cannot instantiate: {type(exc).__name__}"
    if contract.protocol is not None and not isinstance(obj, contract.protocol):
        return f"does not satisfy {contract.protocol.__name__}"
    if contract.factory_args is not None:
        if not callable(obj):
            return "not callable"
        counts = (
            (contract.factory_args,)
            if isinstance(contract.factory_args, int)
            else contract.factory_args
        )
        if not any(_accepts_positional(obj, n) for n in counts):
            return f"does not accept {contract.factory_args} positional arguments"
    return None
