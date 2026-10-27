"""Clear download caches that fill a small robot disk (SPEC 4A, Installer).

A platform names the caches that are safe to drop (``Platform.clearable_caches``, paths written
``~/...``). They are not part of the install record: a deleted cache cannot be restored, only
rebuilt by whatever filled it, so uninstall leaves them alone. Because of that, ``-y`` alone
does not clear one: only ``--clear-caches`` or a yes on a terminal does. A cache that a
virtualenv links into (uv's ``--link-mode=symlink``) is not a cache any more but part of that
environment, so a symlink into it under the given folders keeps it.
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
        raise WizardError(
            "Invalid cache path", f"{spec!r} must be a folder below the robot user's home"
        )
    return path


def linked_from(remote: Remote, cache: str, guard_dirs: tuple[str, ...]) -> str | None:
    """A symlink under ``guard_dirs`` whose target lies inside ``cache`` (a virtualenv uv made
    with symlinks into its cache), or None."""
    pattern = shlex.quote(f"{cache}/*")
    for folder in guard_dirs:
        found = remote.run(
            f"find {shlex.quote(folder)} -lname {pattern} -print -quit 2>/dev/null || true"
        ).strip()
        if found:
            return found
    return None


def clear_caches(
    ui: Ui,
    remote: Remote,
    home: str,
    specs: tuple[str, ...],
    wanted: bool | None,
    dry_run: bool,
    guard_dirs: tuple[str, ...] = (),
) -> list[str]:
    """Show the size of each existing cache and delete it when asked (``wanted`` None: ask on a
    terminal, default no as it changes the robot; unattended or with ``-y`` it is kept, only
    ``wanted`` True deletes without asking). The folder is deleted only if its ``realpath`` is
    the expanded path itself, so a symlink never redirects the delete, and only if no symlink
    under ``guard_dirs`` points into it (``linked_from``). The dry run only reports, with the
    same decisions as the real run. Returns the caches left in place, with their size, for the
    free-space error."""
    kept: list[str] = []
    for spec in specs:
        path = cache_path(home, spec)
        if remote.run(f"test -e {shlex.quote(path)} && echo yes || echo no").strip() != "yes":
            continue
        with ui.running(f"Measuring {path}…"):
            size = remote.run(f"du -sh {shlex.quote(path)} 2>/dev/null | cut -f1").strip() or "?"
        label = f"{path} ({size}; uninstall cannot restore it, whatever filled it fills it again)"
        if not _clearable(ui, remote, path, label, wanted, dry_run, guard_dirs):
            kept.append(f"{path} ({size})")
            continue
        if dry_run:
            ui.would("Clear a cache", label)
            continue
        with ui.progress(f"clearing {path} ({size}), can take a minute", report=False, slow=False):
            remote.run(f"rm -rf {shlex.quote(path)}")
        ui.ok("Cleared a cache", f"{path} ({size})")
    return kept


def _clearable(
    ui: Ui,
    remote: Remote,
    path: str,
    label: str,
    wanted: bool | None,
    dry_run: bool,
    guard_dirs: tuple[str, ...],
) -> bool:
    """Decide, in the same order for the dry run and the real run, whether ``path`` goes: kept
    when declined, with ``-y`` alone, when an environment links into it or when it resolves
    elsewhere; else asked (a dry run says it would ask) unless ``wanted`` is True."""
    if wanted is False:
        ui.ok("Kept a cache", label)
        return False
    if wanted is None and ui.assume_yes:
        ui.ok("Kept a cache", f"{label}; -y does not clear it, --clear-caches does")
        return False
    if wanted is None and not ui.interactive:
        ui.ok("Kept a cache", f"{label}; unattended runs clear it only with --clear-caches")
        return False
    link = linked_from(remote, path, guard_dirs)
    if link:
        ui.warn("Kept a cache", f"{path}: {link} links into it (an environment depends on it)")
        return False
    real = remote.run(f"realpath {shlex.quote(path)} 2>/dev/null || true").strip()
    if real != path:
        ui.warn("Kept a cache", f"{path} resolves to {real or 'nothing'}: symlink?")
        return False
    if wanted is None:
        if dry_run:
            ui.would("Ask to clear a cache", f"{label} (default no)")
            return False
        if not ui.confirm(f"{label}: clear it to free space?", False, flag="--clear-caches"):
            ui.ok("Kept a cache", path)
            return False
    return True


__all__ = ["HOME_PREFIX", "cache_path", "clear_caches", "linked_from"]
