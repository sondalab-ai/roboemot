"""What an install changed on the robot, and how to undo exactly that (SPEC 4A, Packaging).

Robot apps usually share one Python environment, so a clean uninstall cannot just restore a
full snapshot: other apps may have installed packages since. The record keeps only what this
installer changed: distributions it ``added`` and the original versions of those it
``replaced``, merged over every reinstall without ever overwriting an original. It is written
before pip runs (``pending_before``) and completed after, so an interrupted install still
leaves enough to undo it. The app data folder (``roboemot.paths``: registry, models, trait
cache) is recorded the same way, with the parents the installer had to create, but only when it
did not exist before the install: a folder that was already there is not the installer's to
remove.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

RECORD_VERSION = 1

Versions = dict[str, str]  # canonical distribution name -> version


def canonical(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def versions(pip_list: list[dict[str, str]]) -> Versions:
    """``pip list --format=json`` as canonical name -> version."""
    return {canonical(p["name"]): p["version"] for p in pip_list}


@dataclass
class Replaced:
    before: str  # the version found on the robot before the first install
    after: str  # the version this installer left


@dataclass
class InstallRecord:
    app: str
    platform: str
    packages: list[str]  # this app's own distributions
    added: list[str] = field(default_factory=list)
    replaced: dict[str, Replaced] = field(default_factory=dict)
    autostart_before: str | None = None
    autostart_recorded: bool = False
    pending_before: Versions | None = None  # set while pip runs
    data_dir: str | None = None  # app data folder this installer created (absolute path)
    data_dir_parents: list[str] = field(default_factory=list)  # parents it had to create too
    version: int = RECORD_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> InstallRecord:
        data = dict(data)
        data["replaced"] = {k: Replaced(**v) for k, v in data.get("replaced", {}).items()}
        return cls(**data)

    def merge(self, before: Versions, after: Versions) -> None:
        """Fold one pip run into the record; originals from earlier runs are kept."""
        own = {canonical(p) for p in self.packages}
        added = set(self.added)
        for name, version in after.items():
            if name in own or name in added:
                continue
            if name not in before:
                if name in self.replaced:  # removed and re-added: still a replacement
                    self.replaced[name].after = version
                else:
                    added.add(name)
            elif before[name] != version:
                if name in self.replaced:
                    self.replaced[name].after = version
                else:
                    self.replaced[name] = Replaced(before[name], version)
        self.added = sorted(added)
        self.pending_before = None


@dataclass
class UninstallPlan:
    remove: list[str]  # installed distributions to uninstall
    restore: dict[str, str]  # name -> original version to reinstall
    kept: dict[str, list[str]]  # added by us but now required by these other distributions
    drifted: dict[str, str]  # replaced by us, then changed again by someone else: left alone

    @property
    def empty(self) -> bool:
        return not (self.remove or self.restore)


def plan_uninstall(
    record: InstallRecord, installed: Versions, requires: dict[str, list[str]]
) -> UninstallPlan:
    """Undo only this installer's changes; keep what other distributions still need.

    ``requires`` maps every installed distribution to the canonical names it depends on.
    """
    own = {canonical(p) for p in record.packages}
    remove = {n for n in own | set(record.added) if n in installed}
    kept: dict[str, list[str]] = {}
    changed = True
    while changed:  # keeping one package can keep its own dependencies too
        changed = False
        for name in sorted(remove - own):
            users = sorted(d for d, reqs in requires.items() if name in reqs and d not in remove)
            if users:
                remove.discard(name)
                kept[name] = users
                changed = True
    restore, drifted = {}, {}
    for name, rep in record.replaced.items():
        current = installed.get(name)
        if current == rep.after:
            restore[name] = rep.before
        elif current != rep.before:
            drifted[name] = current or "missing"
    return UninstallPlan(sorted(remove), restore, kept, drifted)


__all__ = [
    "InstallRecord",
    "Replaced",
    "UninstallPlan",
    "Versions",
    "canonical",
    "plan_uninstall",
    "versions",
]
