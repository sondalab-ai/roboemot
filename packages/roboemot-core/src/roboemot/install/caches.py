"""Clear download caches that fill a small robot disk (SPEC 4A, Installer).

A platform names the caches that are safe to drop (``Platform.clearable_caches``, paths written
``~/...``). They are not part of the install record: a deleted cache cannot be restored, only
rebuilt by whatever filled it, so uninstall leaves them alone.
"""

from __future__ import annotations

import posixpath
import shlex

from .remote import Remote
from .ui import Ui, WizardError

HOME_PREFIX = "~/"


def cache_path(home: str, spec: str) -> str:
    """The absolute path of a ``~/...`` cache spec, strictly below ``home``."""
    path = posixpath.normpath(posixpath.join(home, spec.removeprefix(HOME_PREFIX)))
    if not spec.startswith(HOME_PREFIX) or not path.startswith(home + "/"):
        raise WizardError(f"cache {spec!r} must be a folder below the robot user's home")
    return path


def clear_caches(
    ui: Ui, remote: Remote, home: str, specs: tuple[str, ...], wanted: bool | None, dry_run: bool
) -> None:
    """Show the size of each existing cache and delete it when asked (``wanted`` None: ask; the
    default is yes on a terminal and no when unattended, where only ``-y`` or ``wanted`` True
    deletes). The folder is deleted only if its ``realpath`` is the expanded path itself, so
    a symlink never redirects the delete. The dry run only reports."""
    for spec in specs:
        path = cache_path(home, spec)
        if remote.run(f"test -e {shlex.quote(path)} && echo yes || echo no").strip() != "yes":
            continue
        ui.info(f"measuring {path}")
        size = remote.run(f"du -sh {shlex.quote(path)} 2>/dev/null | cut -f1").strip() or "?"
        label = f"{path} ({size}; uninstall cannot restore it, whatever filled it fills it again)"
        if wanted is False:
            ui.info(f"{label} kept")
            continue
        if dry_run:
            ui.info(f"would clear {label}")
            continue
        if wanted is None and not ui.confirm(f"{label}: clear it to free space?", ui.interactive):
            ui.info(f"{path} kept")
            continue
        real = remote.run(f"realpath {shlex.quote(path)} 2>/dev/null || true").strip()
        if real != path:
            ui.warn(f"{path} resolves to {real or 'nothing'}: symlink? left as it is")
            continue
        with ui.progress(f"clearing {path} ({size}), can take a minute", report=False, slow=False):
            remote.run(f"rm -rf {shlex.quote(path)}")
        ui.ok(f"cleared {path} ({size})")


__all__ = ["HOME_PREFIX", "cache_path", "clear_caches"]
