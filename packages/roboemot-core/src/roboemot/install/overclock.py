"""Opt-in CPU overclock through the robot's boot config (SPEC 4A, Installer).

Robot-agnostic: the platform says whether its hardware supports an overclock and how
(``Platform.overclock``, an :class:`Overclock`; None: no overclock). Its ``support`` runs the
read-only checks and gives an :class:`OverclockProfile` only when every check passes: the boot
config file, the block of lines with begin and end markers, and the trial config, a copy of the
boot config plus the block that the robot boots once (``Overclock.trial_reboot``) before the
block goes into the boot config itself. There is no direct write: a robot that cannot boot a
trial config is not supported (owner decision 2026-11-05).

Root and trust. Writes to the boot partition run as root through one ``sudo`` per step
(``Remote.run_root``, ``/usr/bin/sudo /bin/sh -c``): sudo asks on the owner's terminal and the
password never passes through the installer. The script root runs (``root_script``) reads no
file the robot user can write: each new text is a base64 literal in the script, decoded by root
into a temporary file next to its target, and checked against a sha256 computed on this PC;
each target's current text is checked against the sha256 of what was read (the boot config is
pinned that way even when only the trial config is written). It then syncs, renames, and syncs
again. Every text it writes is the boot config as read and pinned by its hash, with the
platform's current block added (the trial config, then the promote) or a known block taken
out (the revert); deleting the trial config writes nothing. The install record
(``record.OverclockRecord``) stays writable by the robot user, so it is never trusted for a
root write: its paths and block must be the platform's (``config_path``, ``trial_path``,
``known_blocks``) or the run stops before anything changes, and its backup of the boot config
is kept only for the owner's recovery by hand. These checks guard against a corrupt or stale
record and against files swapped while the owner types the password; they do not make the
installer safe against a fully hostile robot account, which controls the login shell that
runs sudo and the binaries the support checks call.

Every other byte of the boot config stays the same: the block is appended to a file that ends
in a newline (the platform refuses one that does not) and removed as that exact text. A block
edited by hand is refused and the difference printed. A reboot is only ever run after its own
question (default no; ``-y`` does not answer it, the ``--overclock-reboot`` flag does) and the
wait for the robot to come back is bounded.
"""

from __future__ import annotations

import base64
import binascii
import difflib
import hashlib
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

from .record import OverclockRecord
from .remote import SUDO, Remote
from .ui import Ui, WizardError

FLAG = "--overclock"
REBOOT_FLAG = "--overclock-reboot"
# A board booting its OS from eMMC is back on SSH in well under a minute; the first boot of a
# new config can add a file system check. Five minutes covers that and still ends the wait.
BOOT_TIMEOUT_S = 300.0
BOOT_POLL_S = 5.0
BOOT_ID = "cat /proc/sys/kernel/random/boot_id"  # changes on every boot (Linux)
REMOVED, GONE, EDITED = "removed", "gone", "edited"  # what ``remove`` found and did
OFF_NEXT_BOOT = "the overclock comes off on the next boot"
CHANGED_RC = 3  # the root script changed nothing: a target was not as read
PARTIAL_RC = 4  # the root script failed while renaming or deleting: maybe partly changed
SUDO_DENIED_RC = 1  # sudo's exit code when it could not authenticate (the script never uses it)
MISSING_RC = 5  # a binary the root script calls is not there: nothing ran
REBOOT_FAILED_RC = 6  # the reboot command itself failed (not sudo)
TOO_LONG_RC = 126  # the shell could not run the command: too long (E2BIG) or not executable
MAX_SCRIPT_BYTES = 95_000  # the boot config limit that keeps the root command runnable
NEW_SUFFIX = ".roboemot-new"
# Absolute binaries inside the root script (merged /usr, Debian 12 on): no PATH lookup as root.
ROOT_BINS = {
    "sha256sum": "/usr/bin/sha256sum",
    "base64": "/usr/bin/base64",
    "mv": "/usr/bin/mv",
    "rm": "/usr/bin/rm",
    "sync": "/usr/bin/sync",
}

# path, text it holds now (None: absent), new text (None: delete; the same text: check only)
Change = tuple[str, str | None, str | None]


@dataclass(frozen=True)
class Check:
    """One support or post-boot check: what was checked, whether it passed, and why."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class OverclockProfile:
    """What the platform writes for its overclock, once every support check passed."""

    label: str  # what it is, as the question names it (for example "1.8 GHz")
    config_path: str  # the boot config the block goes into
    trial_path: str  # the trial config booted once first
    block: str  # the exact lines, begin and end markers included, ending in a newline
    config: str  # the boot config as read during the checks


class Overclock(Protocol):
    """A platform's overclock (``Platform.overclock``)."""

    label: str  # what it is, as the keep-or-revert question names it (for example "1.8 GHz")
    warning: str  # said before the question: what a failed boot takes to recover
    reboot: str  # root command for a plain reboot
    trial_reboot: str  # root command that boots the trial config once
    reboot_warning: str  # what any reboot does to the robot, said before asking for one
    reboot_hint: str  # how the owner reboots it by hand
    # The only files root ever writes, the block it writes, and every block an earlier version
    # wrote (``block`` first), which it may remove: a record naming others is refused.
    config_path: str
    trial_path: str
    block: str
    known_blocks: tuple[str, ...]

    def support(self, remote: Remote) -> tuple[OverclockProfile | None, list[Check]]: ...
    def trial_booted(self, remote: Remote) -> bool: ...
    def verify(self, remote: Remote) -> list[Check]: ...


@dataclass
class Plan:
    """What the pre-change phase settled for the last step: the profile to write (a new
    overclock), whether root runs over the terminal, and whether root is needed at all."""

    profile: OverclockProfile | None = None
    tty: bool = False
    root: bool = False
    checks: list[Check] = field(default_factory=list)
    rerun: str = ""  # the installer command line that boots the trial config later
    found: OverclockRecord | None = None  # a block in the boot config that no record names
    forget: bool = False  # the recorded block is gone: ``apply`` drops the record entry


Save = Callable[[OverclockRecord | None], None]


# ---------------------------------------------------------------- text


def with_block(text: str, block: str) -> str:
    """``text`` with ``block`` appended. ``text`` must be empty or end in a newline (the
    platform's checks refuse any other), so removing the block gives back every byte."""
    if text and not text.endswith("\n"):
        raise WizardError("The boot config does not end in a newline", "nothing was changed")
    return text + block


def without_block(text: str, block: str) -> str | None:
    """``text`` with exactly ``block`` taken out, or None when it is not in it exactly once."""
    if text.count(block) != 1:
        return None
    return text.replace(block, "", 1)


def has_markers(text: str, block: str) -> bool:
    """Whether ``text`` still has the begin or the end marker line of ``block``."""
    lines = block.splitlines()
    return any(marker in text.splitlines() for marker in (lines[0], lines[-1]))


def diff(before: str, after: str, path: str) -> list[str]:
    """The unified diff from ``before`` to ``after`` as lines, ``path`` named on both sides."""
    out = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), path, path
    )
    return [line.rstrip("\n") for line in out]


def show_diff(ui: Ui, before: str, after: str, path: str) -> None:
    ui.verbatim(diff(before, after, path))


# ---------------------------------------------------------------- root


def root_mode(ui: Ui, remote: Remote, dry_run: bool) -> bool:
    """Whether root runs over the terminal (sudo asks for a password); False when sudo needs
    none. Off a terminal a sudo that needs a password stops the run here, before anything
    changes (the dry run warns instead)."""
    if remote.run(f"{SUDO} -n true 2>/dev/null && echo yes || echo no") == "yes":
        return False
    if ui.interactive:
        return True
    why = "sudo on the robot asks for a password and there is no terminal to type it"
    if dry_run:
        ui.warn("The real run would stop", why)
        return False
    raise WizardError("The overclock needs root", why, "run the installer on a terminal")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def root_script(changes: list[Change], bins: dict[str, str] = ROOT_BINS) -> str:
    """The shell script root runs for ``changes`` (path, text it must hold now or None for
    absent, new text or None to delete). Nothing in it is read from a file the robot user can
    write: each new text is a base64 literal in the script, decoded by root into a temporary
    file next to its target (on the boot partition, root-owned), and checked against its
    sha256. The phases keep the exit code honest:

    0. every binary it calls is there, else ``MISSING_RC``;
    1. every target holds what was read (its sha256, or absent), else ``CHANGED_RC``; a change
       whose new text is its current one is only this check (it pins a file it does not write);
    2. every new text is decoded and verified, else the temporary files go and ``CHANGED_RC``;
    3. ``sync``, then each rename or delete, then ``sync`` (FAT on the boot partition: the data
       is on disk before the rename); a failure here exits ``PARTIAL_RC``.

    Every command carries its own exit, so nothing depends on ``set -e``."""
    b = bins
    lines = ["set -u", *(f"[ -x {shlex.quote(p)} ] || exit {MISSING_RC}" for p in b.values())]
    changes = list(changes)
    writes = [c for c in changes if c[2] != c[1]]  # a change whose text stays is a check only
    for path, before, _ in changes:
        q = shlex.quote(path)
        if before is None:
            lines.append(f"if [ -e {q} ]; then exit {CHANGED_RC}; fi")
        else:
            lines += [
                f"h=$({b['sha256sum']} < {q}) || exit {CHANGED_RC}",
                f'[ "${{h%% *}}" = {sha256(before)} ] || exit {CHANGED_RC}',
            ]
    temps = [shlex.quote(path + NEW_SUFFIX) for path, _, after in writes if after is not None]
    drop = f"{b['rm']} -f {' '.join(temps)}" if temps else ":"
    for path, _, after in writes:
        if after is None:
            continue
        t = shlex.quote(path + NEW_SUFFIX)
        data = base64.b64encode(after.encode("utf-8")).decode("ascii")
        lines += [
            f"{b['rm']} -f {t} || exit {CHANGED_RC}",
            f"printf '%s' {data} | {b['base64']} -d > {t} || {{ {drop}; exit {CHANGED_RC}; }}",
            f"h=$({b['sha256sum']} < {t}) || {{ {drop}; exit {CHANGED_RC}; }}",
            f'[ "${{h%% *}}" = {sha256(after)} ] || {{ {drop}; exit {CHANGED_RC}; }}',
        ]
    lines.append(f"{b['sync']} || {{ {drop}; exit {CHANGED_RC}; }}")
    for path, _, after in writes:
        q = shlex.quote(path)
        if after is None:
            lines.append(f"{b['rm']} -f {q} || exit {PARTIAL_RC}")
        else:
            lines.append(f"{b['mv']} -f {shlex.quote(path + NEW_SUFFIX)} {q} || exit {PARTIAL_RC}")
    lines.append(f"{b['sync']} || exit {PARTIAL_RC}")
    return "\n".join(lines)


class RootUnchanged(WizardError):
    """A root write that changed nothing: sudo refused, or a target was not as read."""


class RootPartial(WizardError):
    """A root write that may have changed part of what it was asked to."""


def root_write(ui: Ui, remote: Remote, tty: bool, changes: list[Change]) -> None:
    """Apply ``changes`` as root in one ``sudo`` (``root_script``). A target that no longer
    holds what was read, or a refused sudo, changes nothing (``RootUnchanged``); any other
    failure may have changed part of it (``RootPartial``: rerun the installer)."""
    names = ", ".join(path for path, before, after in changes if after != before)
    script = root_script(changes)
    if len(script.encode()) > MAX_SCRIPT_BYTES:
        raise RootUnchanged(
            "The boot config is too large", f"over about {MAX_SCRIPT_BYTES // 1000} KB; nothing ran"
        )
    rc = remote.run_root(script, tty)
    if rc == MISSING_RC:
        raise RootUnchanged(
            "A binary is missing on the robot",
            f"one of {', '.join(ROOT_BINS.values())} is not there; nothing ran",
        )
    if rc == TOO_LONG_RC:
        raise RootUnchanged(
            "The root command could not run",
            f"exit {rc}: too long for the robot's shell, or not executable; nothing ran",
        )
    if rc == CHANGED_RC:
        raise RootUnchanged(
            "The boot config changed while the installer ran", f"{names}: unchanged"
        )
    if rc == SUDO_DENIED_RC:
        raise RootUnchanged("Root command failed", f"sudo exited {rc}; the robot is unchanged")
    if rc != 0:
        raise RootPartial(
            "Root command failed",
            f"exited {rc}; {names} may be partly changed",
            "rerun the installer: it checks the files and finishes or removes the change",
        )


def read(remote: Remote, path: str) -> str | None:
    """A file's exact text (world-readable), or None when it does not exist. It travels as
    base64 after a marker byte, so neither ``Remote.run``'s text mode (``\\r\\n``) nor its strip
    changes a byte. A file that exists but cannot be read stops the run (it never reads as
    empty), and so does one that is not UTF-8 text."""
    q = shlex.quote(path)
    out = remote.run(f"if [ -e {q} ]; then printf x; base64 -w0 < {q}; fi")
    if not out.startswith("x"):
        return None
    try:
        return base64.b64decode(out[1:], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as e:
        raise WizardError(f"Could not read {path}", f"not UTF-8 text: {e}") from None


# ---------------------------------------------------------------- reboot


def ask_reboot(ui: Ui, oc: Overclock, flag: bool | None, question: str) -> bool:
    """The separate reboot question (default no): ``--overclock-reboot`` answers it, ``-y``
    never does, and without a terminal and without the flag the reboot is refused."""
    if flag is not None:
        return flag
    if not ui.interactive:
        ui.warn("Did not reboot", f"without a terminal only {REBOOT_FLAG} reboots the robot")
        return False
    ui.info(oc.reboot_warning)
    return ui.confirm_opt_in(question, False, flag=REBOOT_FLAG)


def reboot_and_wait(ui: Ui, remote: Remote, tty: bool, command: str) -> bool:
    """Reboot with ``command`` as root and wait, up to ``BOOT_TIMEOUT_S``, until the robot is
    back on SSH with a new boot id. False when it did not come back in time. ssh losing the
    connection while the robot goes down is expected, not an error."""
    before = remote.run(BOOT_ID)
    rc = remote.run_root(f"{command} || exit {REBOOT_FAILED_RC}", tty)
    if rc == SUDO_DENIED_RC:
        raise RootUnchanged("Root command failed", "sudo refused the reboot; nothing rebooted")
    if rc == REBOOT_FAILED_RC:
        raise RootUnchanged("The reboot failed", f"{command} failed; nothing rebooted")
    ui.ok("Asked the robot to reboot", "the SSH connection drops now")
    remote.disconnect()
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        with ui.running("Waiting for the robot…", f"up to {BOOT_TIMEOUT_S:.0f} s"):
            while not remote.reachable():
                if time.monotonic() >= deadline:
                    return False
                time.sleep(BOOT_POLL_S)
        try:
            remote.connect(quiet=True)  # a refusal while it still goes down is expected
            now = remote.run(BOOT_ID)
        except WizardError:
            now = before  # it answered while still going down
        if now != before:
            ui.ok("The robot is back", "new boot")
            return True
        remote.disconnect()
        time.sleep(BOOT_POLL_S)
    return False


def offer_reboot(
    ui: Ui, remote: Remote, oc: Overclock | None, plan: Plan, flag: bool | None, why: str
) -> None:
    """Ask for a plain reboot that applies ``why`` (a removed block); without one, say how."""
    if oc is None:
        ui.hint("Reboot the robot to apply it", "a plain reboot")
        return
    if not ask_reboot(ui, oc, flag, f"Reboot the robot now ({why})?"):
        ui.hint(f"Reboot the robot to apply it ({why})", oc.reboot_hint)
        return
    if not reboot_and_wait(ui, remote, plan.tty, oc.reboot):
        ui.warn("Robot not back", f"not on SSH after {BOOT_TIMEOUT_S:.0f} s")


# ---------------------------------------------------------------- trust


def untrusted(oc: Overclock | None, rec: OverclockRecord) -> str | None:
    """Why the record's overclock entry cannot drive a root write, or None. The record is
    writable by the robot user: only the platform's own paths and known blocks are accepted."""
    if oc is None:
        return "this platform offers no overclock to check the recorded one against"
    if rec.path != oc.config_path:
        return f"it names {rec.path!r}, not {oc.config_path}"
    if rec.trial is not None and rec.trial != oc.trial_path:
        return f"its trial config is {rec.trial!r}, not {oc.trial_path}"
    if rec.block not in oc.known_blocks:
        return "its block is none the platform ever wrote"
    return None


def require_trusted(oc: Overclock | None, rec: OverclockRecord) -> Overclock:
    """The platform's overclock when the record's entry matches it; else stop, unchanged."""
    why = untrusted(oc, rec)
    if why is not None or oc is None:
        raise WizardError(
            "The recorded overclock does not match the platform",
            f"{why}; nothing was changed (the install record is not trusted for root writes)",
            "check the overclock entry of the install record by hand",
        )
    return oc


def own_trial(oc: Overclock, rec: OverclockRecord | None, trial: str, config: str | None) -> bool:
    """Whether ``trial`` is a trial config this installer wrote: the live boot config plus a
    block it knows, or (an older boot config) the recorded one plus the recorded block. Only
    used to decide that the file may be deleted; nothing of it is ever written back."""
    clean = config is not None and (not config or config.endswith("\n"))
    if clean and any(trial == config + b for b in oc.known_blocks):
        return True
    return rec is not None and trial == rec.applied


# ---------------------------------------------------------------- phases


def report_checks(ui: Ui, checks: list[Check]) -> None:
    for check in checks:
        (ui.ok if check.ok else ui.warn)(check.name, check.detail)


def applied_block(oc: Overclock, config: str) -> str | None:
    """The known block that ``config`` holds exactly once, or None."""
    return next((b for b in oc.known_blocks if config.count(b) == 1), None)


def find_unrecorded(remote: Remote, oc: Overclock) -> OverclockRecord | None:
    """An ``on`` record entry for a known block in the live boot config, for a robot whose
    install record no longer names it; None without such a block. The block is one of the
    platform's own, so the entry passes ``require_trusted``; ``backup`` is the live file
    without the block."""
    config = read(remote, oc.config_path) or ""
    block = applied_block(oc, config)
    if block is None:
        return None
    return OverclockRecord(oc.config_path, block, config.replace(block, "", 1), state="on")


def _keep(ui: Ui, question: str) -> bool:
    """The keep-or-revert question: default keep, and ``-y`` keeps and says how to remove it."""
    if ui.assume_yes:
        ui.ok("Kept the overclock", f"-y keeps it, --no-{FLAG[2:]} removes it")
    return ui.confirm(question, True, flag=FLAG)


def _ask_active(
    ui: Ui, remote: Remote, oc: Overclock, rec: OverclockRecord
) -> tuple[bool | None, bool]:
    """What the live boot config says for a run with no overclock flag: the choice (None: the
    normal opt-in decides) and whether the record entry still counts. A block in place gets
    only the keep-or-revert question (default keep; ``-y`` keeps), and so does a trial config
    the robot runs now, which is not saved yet (keep: ``apply`` checks and promotes it). A block
    edited by hand is kept, and ``apply`` prints the difference. A recorded block that is gone
    counts as no overclock."""
    config = read(remote, oc.config_path) or ""
    if applied_block(oc, config) is not None:
        return _keep(ui, f"Overclock is on ({oc.label}). Keep it (no reverts it)?"), True
    if rec.state != "on":
        if not oc.trial_booted(remote):
            return None, True
        question = f"Overclock is running from a trial boot ({oc.label}) and is not saved."
        return _keep(ui, f"{question} Keep it (no reverts it)?"), True
    return (True, True) if has_markers(config, rec.block) else (None, False)


def settle(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    rec: OverclockRecord | None,
    want: bool | None,
    dry_run: bool,
    adopt: bool = False,
) -> tuple[bool, Plan]:
    """Before anything on the robot changes: an overclock already in the boot config gets only
    the keep-or-revert question. Otherwise run the support checks when a new overclock may be
    written, ask the opt-in (after the recovery warning; default no, ``-y`` never answers it).
    Then settle root. ``adopt`` (install only) also treats a known block that no record names
    as applied. Returns the choice and the plan for ``apply``. An unsupported robot is a clean
    no: each check is printed, and the robot stays unchanged."""
    plan = Plan()
    if rec is not None:
        require_trusted(oc, rec)  # before anything changes
    elif adopt and oc is not None:
        rec = plan.found = find_unrecorded(remote, oc)
        if rec is not None:
            ui.ok("Found an overclock", f"{oc.config_path}: not in the install record")
    if want is False and rec is None:
        return False, plan
    if want is None and rec is not None and oc is not None:
        want, counts = _ask_active(ui, remote, oc, rec)
        plan.forget = not counts
        rec = rec if counts else None
    if oc is not None and rec is None:
        plan.profile, plan.checks = oc.support(remote)
        report_checks(ui, plan.checks)
        if plan.profile is None:
            failed = ", ".join(c.name for c in plan.checks if not c.ok)
            ui.warn("No overclock", f"{failed or 'not supported'}: the robot is unchanged")
            return False, plan
    if want is None and ui.assume_yes:
        ui.ok("Overclock off", f"-y does not turn it on, {FLAG} does")
        want = False
    if oc is not None and plan.profile is not None and want is not False:
        ui.info(oc.warning)
    if want is None:
        label = plan.profile.label if plan.profile else "the recorded overclock"
        want = ui.confirm_opt_in(f"Overclock the CPU ({label})?", False, flag=FLAG)
    keep = rec is not None and rec.state == "on" and want  # read back; root only if it is gone
    plan.root = (rec is not None and not keep) or (want and plan.profile is not None)
    if plan.root:
        plan.tty = root_mode(ui, remote, dry_run)
    return want, plan


def need_root(ui: Ui, remote: Remote, plan: Plan, dry_run: bool) -> None:
    """Settle root now, for a write the pre-change phase could not foresee (a kept overclock
    whose block turned out gone)."""
    if not plan.root:
        plan.tty, plan.root = root_mode(ui, remote, dry_run), True


def apply(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord | None,
    want: bool,
    reboot_flag: bool | None,
    dry_run: bool,
    save: Save,
) -> None:
    """The install's last step: write, promote or remove the block as ``want`` and the record
    say. ``save`` writes the record with its new overclock entry (None: none). A block found
    without a record entry is recorded first, and a recorded block that is gone is forgotten
    first, except in a dry run."""
    if rec is None and plan.found is not None:
        rec = plan.found
        if not dry_run:
            save(rec)
    if plan.forget:
        rec = None
        if not dry_run:
            save(None)
    if rec is not None and rec.state != "on":
        rec = finish_promote(ui, remote, oc, plan, rec, dry_run, save) or rec
    if rec is not None and rec.state != "on":
        _settle_trial(ui, remote, oc, plan, rec, want, reboot_flag, dry_run, save)
    elif rec is not None and want:
        _check_on(ui, remote, oc, plan, rec, reboot_flag, dry_run, save)
    elif rec is not None:
        if remove(ui, remote, oc, plan, rec, dry_run, save) == REMOVED:
            offer_reboot(ui, remote, oc, plan, reboot_flag, OFF_NEXT_BOOT)
    elif want and plan.profile is not None:
        _write(ui, remote, oc, plan, reboot_flag, dry_run, save)
    else:
        ui.ok("No overclock", "the boot config is not touched")


def _fresh(
    ui: Ui,
    remote: Remote,
    oc: Overclock,
    plan: Plan,
    reboot_flag: bool | None,
    dry_run: bool,
    save: Save,
) -> None:
    """Start over after the recorded overclock turned out gone: the support checks again,
    then a new write when they pass."""
    need_root(ui, remote, plan, dry_run)
    plan.profile, plan.checks = oc.support(remote)
    report_checks(ui, plan.checks)
    if plan.profile is None:
        failed = ", ".join(c.name for c in plan.checks if not c.ok)
        ui.warn("No overclock", f"{failed or 'not supported'}: the robot is unchanged")
        return
    _write(ui, remote, oc, plan, reboot_flag, dry_run, save)


def _check_on(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord,
    reboot_flag: bool | None,
    dry_run: bool,
    save: Save,
) -> None:
    """An overclock already on: the block is still there as written, or it was edited (left
    alone, the difference printed), or it is gone (removed by hand, or a write that never
    finished): the record forgets it and the checks run again."""
    oc = require_trusted(oc, rec)
    text = read(remote, oc.config_path) or ""
    if without_block(text, rec.block) is not None:
        ui.ok("Overclock on", f"the block is in {oc.config_path}")
    elif has_markers(text, rec.block):
        ui.warn("Overclock block edited", f"{oc.config_path}: left as it is")
        show_diff(ui, rec.block, _found(text, rec.block), oc.config_path)
    else:
        ui.warn("Overclock block gone", f"not in {oc.config_path}; checking again")
        if not dry_run:
            save(None)
        _fresh(ui, remote, oc, plan, reboot_flag, dry_run, save)


def _found(text: str, block: str) -> str:
    """The lines between (and with) ``block``'s markers in ``text``, or "" without them."""
    lines = text.splitlines(keepends=True)
    begin, end = block.splitlines()[0], block.splitlines()[-1]
    names = [ln.rstrip("\r\n") for ln in lines]
    start = names.index(begin) if begin in names else None
    stop = names.index(end) if end in names else None
    if start is None and stop is None:
        return ""
    return "".join(lines[start or 0 : (len(lines) if stop is None else stop + 1)])


def _write(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    reboot_flag: bool | None,
    dry_run: bool,
    save: Save,
) -> None:
    """A new overclock: the trial config, then the trial boot. The record is written first.
    Only a write that changed nothing (``RootUnchanged``) drops it again; anything else (a
    partial write, Ctrl-C, a lost connection) leaves it, so the next run checks the files and
    finishes or removes what is there."""
    p = plan.profile
    assert p is not None
    new = with_block(p.config, p.block)
    if dry_run:
        ui.would("Write the trial config", f"{p.trial_path}: {p.config_path} plus the block")
        show_diff(ui, p.config, new, p.config_path)
        ui.would("Ask for the trial boot", "then check it and promote the block")
        return
    rec = OverclockRecord(p.config_path, p.block, p.config, p.trial_path)
    save(rec)  # recorded before the write, so an interrupted run can still be undone
    leftover = read(remote, p.trial_path)
    if leftover == new:
        ui.ok("Found the trial config", f"{p.trial_path}, written by an earlier run")
    else:
        own = oc is not None and leftover is not None and own_trial(oc, None, leftover, p.config)
        if leftover is not None and not own:
            save(None)
            raise RootUnchanged(
                "Left the overclock alone", f"{p.trial_path} is not the installer's"
            )
        pinned = (p.config_path, p.config, p.config)  # checked by root, not written
        try:
            root_write(ui, remote, plan.tty, [(p.trial_path, leftover, new), pinned])
        except RootUnchanged:
            save(None)  # nothing was written
            raise
        ui.ok("Wrote the trial config", f"{p.trial_path}; {p.config_path} is unchanged")
    _trial_boot(ui, remote, oc, plan, rec, reboot_flag, dry_run, save)


def _trial_boot(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord,
    reboot_flag: bool | None,
    dry_run: bool,
    save: Save,
) -> None:
    """Ask for the trial boot; after it, check and promote, or remove the trial config. The
    trial config must still be the live boot config plus the block; one that is missing or
    stale is removed (when it is the installer's) or forgotten, and the checks run again."""
    oc = require_trusted(oc, rec)
    config, trial = read(remote, oc.config_path), read(remote, oc.trial_path)
    if config is None or (config and not config.endswith("\n")) or trial != config + oc.block:
        if trial is not None and not own_trial(oc, rec, trial, config):
            ui.warn("Left the overclock alone", f"{oc.trial_path} is not the installer's")
            return
        ui.warn("Trial config not current", f"{oc.trial_path} is missing or stale; checking again")
        if dry_run:
            if trial is not None:
                ui.would("Remove the trial config", oc.trial_path)
            return
        if trial is not None:
            root_write(ui, remote, plan.tty, [(oc.trial_path, trial, None)])
        save(None)
        _fresh(ui, remote, oc, plan, reboot_flag, dry_run, save)
        return
    if dry_run:
        ui.would("Ask for the trial boot", f"{oc.trial_path} is written")
        return
    question = "Reboot the robot now into the trial config?"
    command = plan.rerun or shlex.join(["python", "scripts/install_on_robot.py", FLAG, REBOOT_FLAG])
    if not ask_reboot(ui, oc, reboot_flag, question):
        ui.hint("Boot the trial config later", command)
        return
    rec.state = "trying"
    save(rec)  # before the reboot: the connection ends with it
    try:
        back = reboot_and_wait(ui, remote, plan.tty, oc.trial_reboot)
    except RootUnchanged:
        rec.state = "trial"  # sudo refused: nothing rebooted
        save(rec)
        raise
    if not back:
        ui.warn("Robot not back", f"not on SSH after {BOOT_TIMEOUT_S:.0f} s")
        ui.info(
            f"Power-cycle the robot: a power cycle boots the unchanged {oc.config_path}. The "
            "next run of the installer removes the trial config and says what happened."
        )
        return
    _promote_or_drop(ui, remote, oc, plan, rec, save)


def _promote_or_drop(
    ui: Ui,
    remote: Remote,
    oc: Overclock,
    plan: Plan,
    rec: OverclockRecord,
    save: Save,
) -> None:
    """On the trial boot: every check passes, then the block goes into the boot config (no
    second reboot: what runs and what the boot config says are the same); else the trial
    config goes and the next normal boot uses the unchanged boot config. Root writes only the
    platform's paths, with the live boot config plus the platform's block."""
    oc = require_trusted(oc, rec)
    checks = [Check("Booted the trial config", oc.trial_booted(remote), oc.trial_path)]
    checks += oc.verify(remote) if checks[0].ok else []
    report_checks(ui, checks)
    config, trial = read(remote, oc.config_path), read(remote, oc.trial_path)
    current = (
        config is not None and (not config or config.endswith("\n")) and trial == config + oc.block
    )
    if not current:
        checks.append(Check("Boot config unchanged", False, "changed since the trial"))
        ui.warn("Boot config unchanged", f"{oc.config_path} changed since the trial was written")
    failed = [c for c in checks if not c.ok]
    if failed:
        if trial is not None and own_trial(oc, rec, trial, config):
            root_write(ui, remote, plan.tty, [(oc.trial_path, trial, None)])
        save(None)
        ui.warn("Overclock not promoted", ", ".join(c.name for c in failed))
        ui.info(
            f"Removed the trial config. The robot runs it until its next boot, which uses the "
            f"unchanged {oc.config_path}."
        )
        return
    assert config is not None and trial is not None
    promote = [(oc.config_path, config, with_block(config, oc.block)), (oc.trial_path, trial, None)]
    rec.state = "promoting"
    save(rec)  # a promote cut short is then finished, never mistaken for an unbooted trial
    root_write(ui, remote, plan.tty, promote)
    rec.state, rec.trial, rec.block = "on", None, oc.block
    save(rec)
    ui.ok(
        "Promoted the overclock",
        f"the block is in {oc.config_path}; no second reboot: the running config is the same",
    )


def finish_promote(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord,
    dry_run: bool,
    save: Save,
) -> OverclockRecord | None:
    """A promote an earlier run began: the boot config already holds a known block exactly
    once (its rename went through, the trial config's delete may not have). Finish it: the
    installer's own trial config goes (the boot config as it now is, or the boot config before
    the block plus a known block), and the record says ``on`` with that block. Returns that
    record (not saved in a dry run), or None when no promote was begun."""
    oc = require_trusted(oc, rec)
    config = read(remote, oc.config_path) or ""
    block = applied_block(oc, config)
    if block is None:
        return None
    done = replace(rec, state="on", trial=None, block=block)
    trial = read(remote, oc.trial_path)
    before = config.replace(block, "", 1)
    own = trial is not None and (trial == config or own_trial(oc, rec, trial, before))
    if dry_run:
        ui.would("Finish the promote", f"the block is in {oc.config_path}")
        return done
    if own:
        need_root(ui, remote, plan, dry_run)
        root_write(ui, remote, plan.tty, [(oc.trial_path, trial, None)])
    elif trial is not None:
        ui.warn("Left the trial config alone", f"{oc.trial_path} is not the installer's")
    save(done)
    ui.ok("Promoted the overclock", f"the block is in {oc.config_path} (an earlier run's promote)")
    return done


def _settle_trial(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord,
    want: bool,
    reboot_flag: bool | None,
    dry_run: bool,
    save: Save,
) -> None:
    """A trial config from an earlier run: promote it when the robot runs it now, report a
    trial boot that did not happen (a power cycle, a hang), offer the trial boot again, or
    remove it when the overclock is no longer wanted."""
    oc = require_trusted(oc, rec)
    booted = oc.trial_booted(remote)
    if want and booted:
        if dry_run:
            ui.would("Check and promote", oc.config_path)
            return
        _promote_or_drop(ui, remote, oc, plan, rec, save)
        return
    if want and rec.state == "trial":
        _trial_boot(ui, remote, oc, plan, rec, reboot_flag, dry_run, save)
        return
    if rec.state in ("trying", "promoting") and not booted:
        ui.warn(
            "Trial boot not finished",
            f"the robot booted the unchanged {oc.config_path} (a power cycle, or the trial hung)",
        )
    if remove(ui, remote, oc, plan, rec, dry_run, save) == REMOVED and booted:
        offer_reboot(ui, remote, oc, plan, reboot_flag, "the robot runs the trial config now")


def remove(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord,
    dry_run: bool,
    save: Save,
) -> str:
    """Take out exactly what the record says: the trial config, or the block in the boot
    config (every other byte unchanged), and drop it from the record. A block or trial config
    edited by hand is refused and the difference printed; the record then keeps it. Returns
    ``REMOVED``, ``GONE`` (nothing was left) or ``EDITED``; the caller offers the reboot.
    Only the platform's paths and known blocks are used: the live file minus that exact
    block, or the installer's own trial config deleted; the record's backup is never written."""
    oc = require_trusted(oc, rec)
    on = rec.state == "on"
    path = oc.config_path if on else oc.trial_path
    text = read(remote, path)
    label = "overclock block" if on else "trial config"
    if on:
        after = without_block(text or "", rec.block)
        edited = after is None and has_markers(text or "", rec.block)
        gone = after is None and not edited
        expected, found = rec.block, _found(text or "", rec.block)
    else:
        after = None
        config = read(remote, oc.config_path)
        edited = text is not None and not own_trial(oc, rec, text, config)
        gone = text is None
        expected, found = (config or "") + rec.block, text or ""
    if edited:
        ui.warn("Left the overclock alone", f"{path} was edited since; fix it by hand")
        show_diff(ui, expected, found, path)
        return EDITED
    if gone:
        ui.ok(f"No {label} left", f"{path}: already removed")
    elif dry_run:
        ui.would(f"Remove {label}", path)
        if on:
            show_diff(ui, text or "", after or "", path)
    else:
        root_write(ui, remote, plan.tty, [(path, text, after)])
        ui.ok(f"Removed {label}", f"{path}; every other line is unchanged")
    if not dry_run:
        save(None)
    return GONE if gone else REMOVED


__all__ = [
    "BOOT_TIMEOUT_S",
    "FLAG",
    "OFF_NEXT_BOOT",
    "REBOOT_FLAG",
    "REMOVED",
    "ROOT_BINS",
    "Check",
    "Overclock",
    "OverclockProfile",
    "Plan",
    "RootPartial",
    "RootUnchanged",
    "apply",
    "diff",
    "finish_promote",
    "offer_reboot",
    "remove",
    "root_mode",
    "root_script",
    "settle",
    "with_block",
    "without_block",
]
