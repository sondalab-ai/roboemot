"""What an install changed on the robot, and how to undo exactly that (SPEC 4A, Packaging).

Robot apps usually share one Python environment, so a clean uninstall cannot just restore a
full snapshot: other apps may have installed packages since. The record keeps only what this
installer changed: distributions it ``added`` and the original versions of those it
``replaced``, merged over every reinstall without ever overwriting an original. It is written
before pip runs (``pending_before``) and completed after, so an interrupted install still
leaves enough to undo it. The app data folder (``roboemot.paths``: registry, models, trait
cache) is recorded the same way, with the parents the installer had to create, but only when it
did not exist before the install: a folder that was already there is not the installer's to
remove. The Piper virtualenv (SPEC 8, TTS backends) is recorded on its own, so it is removed
even when the data folder predates the install; so are the model folders (voice, vision) the
installer created inside a data folder that was not its own, and the app config (``app.yaml``)
it wrote, which uninstall removes whole, and the app config keys of a profile (``--cloud-voice``)
it set, so a reinstall without the profile removes exactly those. Secrets (SPEC 8, 12 Data) are
recorded by name, never by value, and only the ones that were not in the store before;
uninstall removes those names and leaves the others. When the robot has no app environment
yet, the installer creates it the way the robot's own software would and records its path
before creating it (``env_created``) with the distributions it held afterwards
(``env_baseline``); uninstall removes it again unless another app or package has moved in.

The record carries a ``version``. A record from a newer installer (a higher version) is refused
with a message naming the cause, since this installer cannot honour fields it does not know;
keys this version does not know in a record of the same or an older version are ignored (the
caller warns). An older record loads with the missing fields at their defaults and is written
back at the current version.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .ui import WizardError

RECORD_VERSION = 3  # 2: models_dirs, env_created, env_baseline, app_config; 3: app_config_keys

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
    piper_venv: str | None = None  # Piper virtualenv this installer created (absolute path)
    secrets: list[str] = field(default_factory=list)  # secret names this installer added
    # model folders it created inside a data folder that was not its own
    models_dirs: list[str] = field(default_factory=list)
    env_created: str | None = None  # app environment this installer created (absolute path)
    env_baseline: list[str] = field(default_factory=list)  # distributions it held once created
    # app config (app.yaml) it wrote, created or edited: uninstall removes it whole
    app_config: str | None = None
    # app config keys a profile set (``--cloud-voice``), each a key path; ``["llm_routes",
    # name]`` is the route entry of that name: a reinstall without the profile removes them
    app_config_keys: list[list[str]] = field(default_factory=list)
    version: int = RECORD_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> InstallRecord:
        """The record from its JSON form; a newer ``version`` raises ``WizardError`` and keys
        this version does not know are dropped (``unknown_keys`` names them)."""
        version = data.get("version", 1)
        if not isinstance(version, int) or isinstance(version, bool):
            raise WizardError(f"the install record has an unreadable version {version!r}")
        if version > RECORD_VERSION:
            raise WizardError(
                f"the install record has version {version!r}, newer than this installer's "
                f"{RECORD_VERSION}: use the installer that wrote it (a newer checkout)"
            )
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        data["replaced"] = {k: Replaced(**v) for k, v in data.get("replaced", {}).items()}
        data["version"] = RECORD_VERSION
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


def unknown_keys(data: dict[str, Any]) -> list[str]:
    """Keys of a record's JSON form that this version does not know (``from_json`` drops them)."""
    known = {f.name for f in fields(InstallRecord)}
    return sorted(set(data) - known)


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
    "unknown_keys",
    "versions",
]
