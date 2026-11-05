"""Opt-in CPU overclock through the robot's boot config (SPEC 4A, Installer).

Robot-agnostic: the platform says whether its hardware supports an overclock and how
(``Platform.overclock``, an :class:`Overclock`; None: no overclock). Its ``support`` runs the
read-only checks and gives an :class:`OverclockProfile` only when every check passes: the boot
config file, the block of lines with begin and end markers, and the trial config, a copy of the
boot config plus the block that the robot boots once (``Overclock.trial_reboot``) before the
block goes into the boot config itself. A platform without a trial boot (``trial_path`` None)
writes the boot config directly, and the installer never reboots it.

Every write is recorded first (``record.OverclockRecord``: the file, the block, a backup of the
file) and needs root: the files are staged in a temporary folder over the normal SSH channel,
then one ``sudo sh -c`` copies them into place after checking that each target still holds what
was read (``Remote.run_root``: sudo asks on the owner's terminal, the password never passes
through the installer). Every other line of the boot config stays byte-identical. Removing the
block takes out exactly the recorded text; a block edited by hand is refused and the difference
printed. A reboot is only ever run after its own question (default no; ``-y`` does not answer
it, the ``--overclock-reboot`` flag does) and the wait for the robot to come back is bounded.
"""

from __future__ import annotations

import difflib
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .record import OverclockRecord
from .remote import Remote
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
CHANGED_RC = 3  # the root script's exit code when a target no longer holds what was read
SUDO_DENIED_RC = 1  # sudo's exit code when it could not authenticate
NEW_SUFFIX = ".roboemot-new"


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
    trial_path: str | None  # the trial config booted once first; None: no trial boot
    block: str  # the exact lines, begin and end markers included, ending in a newline
    config: str  # the boot config as read during the checks


class Overclock(Protocol):
    """A platform's overclock (``Platform.overclock``)."""

    warning: str  # said before the question: what a failed boot takes to recover
    reboot: str  # root command for a plain reboot
    trial_reboot: str  # root command that boots the trial config once
    reboot_warning: str  # what any reboot does to the robot, said before asking for one
    reboot_hint: str  # how the owner reboots it by hand

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


# ---------------------------------------------------------------- text


def with_block(text: str, block: str) -> tuple[str, bool]:
    """``text`` with ``block`` appended, and whether a newline had to go in before it (the
    file did not end in one). Nothing else in ``text`` changes."""
    newline = bool(text) and not text.endswith("\n")
    return text + "\n" * newline + block, newline


def without_block(text: str, rec: OverclockRecord) -> str | None:
    """``text`` with exactly the recorded block (and the newline put before it) taken out, or
    None when that exact text is not in it once."""
    piece = "\n" * rec.newline + rec.block
    if text.count(piece) != 1:
        return None
    return text.replace(piece, "", 1)


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
    if remote.run("sudo -n true 2>/dev/null && echo yes || echo no") == "yes":
        return False
    if ui.interactive:
        return True
    why = "sudo on the robot asks for a password and there is no terminal to type it"
    if dry_run:
        ui.warn("The real run would stop", why)
        return False
    raise WizardError("The overclock needs root", why, "run the installer on a terminal")


def _stage(remote: Remote, texts: list[str]) -> tuple[str, list[str]]:
    """Copy ``texts`` into a new private temporary folder over the normal channel (stdin), and
    return the folder with each file's path."""
    folder = remote.run("mktemp -d /tmp/roboemot-install.XXXXXX")
    paths = [f"{folder}/{i}" for i in range(len(texts))]
    for path, text in zip(paths, texts, strict=True):
        remote.run(f"umask 077 && cat > {shlex.quote(path)}", stdin=text)
    return folder, paths


def root_write(
    ui: Ui, remote: Remote, tty: bool, changes: list[tuple[str, str | None, str | None]]
) -> None:
    """Apply ``changes`` (path, text it must hold now or None for absent, new text or None to
    delete) as root in one ``sudo``. Each target is compared with what was read first; a
    mismatch or a failed sudo changes nothing and raises ``WizardError``."""
    texts = [t for _, before, after in changes for t in (before, after) if t is not None]
    folder, staged = _stage(remote, texts)
    files = iter(staged)
    checks, writes = ["set -e"], []
    for path, before, after in changes:
        q = shlex.quote(path)
        if before is None:
            checks.append(f"test ! -e {q} || exit {CHANGED_RC}")
        else:
            checks.append(f"cmp -s {shlex.quote(next(files))} {q} || exit {CHANGED_RC}")
        if after is None:
            writes.append(f"rm -f {q}")
        else:
            tmp = shlex.quote(path + NEW_SUFFIX)
            writes.append(f"cp {shlex.quote(next(files))} {tmp} && mv -f {tmp} {q}")
    try:
        rc = remote.run_root("\n".join([*checks, *writes, "sync"]), tty)
    finally:
        remote.run(f"rm -rf {shlex.quote(folder)}")
    if rc == CHANGED_RC:
        names = ", ".join(path for path, _, _ in changes)
        raise WizardError("The boot config changed while the installer ran", f"{names}: unchanged")
    if rc != 0:
        raise WizardError("Root command failed", f"sudo exited {rc}; the robot is unchanged")


def read(remote: Remote, path: str) -> str | None:
    """A file's text (world-readable), or None when it does not exist. Read with ``cat`` as it
    is between two marker bytes, so ``Remote.run`` stripping its output keeps every byte."""
    q = shlex.quote(path)
    out = remote.run(f"test -e {q} && {{ printf x; cat {q}; printf x; }} || true")
    return out[1:-1] if len(out) >= 2 and out[0] == out[-1] == "x" else None


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
    rc = remote.run_root(command, tty)
    if rc == SUDO_DENIED_RC:
        raise WizardError("Root command failed", "sudo refused the reboot; the robot is unchanged")
    ui.ok("Asked the robot to reboot", "the SSH connection drops now")
    remote.disconnect()
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        with ui.running(f"Waiting for the robot (up to {BOOT_TIMEOUT_S:.0f} s)…"):
            while not remote.reachable():
                if time.monotonic() >= deadline:
                    return False
                time.sleep(BOOT_POLL_S)
        try:
            remote.connect()
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


# ---------------------------------------------------------------- phases


def report_checks(ui: Ui, checks: list[Check]) -> None:
    for check in checks:
        (ui.ok if check.ok else ui.warn)(check.name, check.detail)


def settle(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    rec: OverclockRecord | None,
    want: bool | None,
    dry_run: bool,
) -> tuple[bool, Plan]:
    """Before anything on the robot changes: run the support checks when a new overclock may
    be written, ask the opt-in (after the recovery warning; default no, ``-y`` never answers
    it), and settle root. Returns the choice and the plan for ``apply``. An unsupported robot is
    a clean no: each check is printed, and the robot stays unchanged."""
    plan = Plan()
    if want is False and rec is None:
        return False, plan
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
    keep = rec is not None and rec.state == "on" and want  # only read back, never written
    plan.root = (rec is not None and not keep) or (want and plan.profile is not None)
    if plan.root:
        plan.tty = root_mode(ui, remote, dry_run)
    return want, plan


def apply(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord | None,
    want: bool,
    reboot_flag: bool | None,
    dry_run: bool,
    save: Callable[[OverclockRecord | None], None],
) -> None:
    """The install's last step: write, promote or remove the block as ``want`` and the record
    say. ``save`` writes the record with its new overclock entry (None: none)."""
    if rec is not None and rec.state != "on":
        _settle_trial(ui, remote, oc, plan, rec, want, reboot_flag, dry_run, save)
    elif rec is not None and want:
        _check_on(ui, remote, rec, dry_run, save)
    elif rec is not None:
        if remove(ui, remote, plan, rec, dry_run, save) == REMOVED:
            offer_reboot(ui, remote, oc, plan, reboot_flag, OFF_NEXT_BOOT)
    elif want and plan.profile is not None:
        _write(ui, remote, oc, plan, reboot_flag, dry_run, save)
    else:
        ui.ok("No overclock", "the boot config is not touched")


def _check_on(
    ui: Ui,
    remote: Remote,
    rec: OverclockRecord,
    dry_run: bool,
    save: Callable[[OverclockRecord | None], None],
) -> None:
    """An overclock already on: the block is still there as written, or it was edited (left
    alone, the difference printed), or it was removed by hand (the record forgets it)."""
    text = read(remote, rec.path) or ""
    if without_block(text, rec) is not None:
        ui.ok("Overclock on", f"the block is in {rec.path}")
    elif has_markers(text, rec.block):
        ui.warn("Overclock block edited", f"{rec.path}: left as it is")
        show_diff(ui, rec.block, _found(text, rec), rec.path)
    else:
        ui.warn("Overclock block gone", f"removed from {rec.path} by hand; {FLAG} again")
        if not dry_run:
            save(None)


def _found(text: str, rec: OverclockRecord) -> str:
    """The lines between (and with) the recorded markers in ``text``, or "" without them."""
    lines = text.splitlines(keepends=True)
    begin, end = rec.block.splitlines()[0], rec.block.splitlines()[-1]
    names = [ln.rstrip("\n") for ln in lines]
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
    save: Callable[[OverclockRecord | None], None],
) -> None:
    """A new overclock: the trial config (then the trial boot), or the boot config directly."""
    p = plan.profile
    assert p is not None
    new, newline = with_block(p.config, p.block)
    target = p.trial_path or p.config_path
    if dry_run:
        if p.trial_path:
            ui.would("Write the trial config", f"{p.trial_path}: {p.config_path} plus the block")
            show_diff(ui, p.config, new, p.config_path)
            ui.would("Ask for the trial boot", "then check it and promote the block")
        else:
            ui.would("Write the overclock block", f"{p.config_path}; no trial boot on this robot")
            show_diff(ui, p.config, new, p.config_path)
        return
    rec = OverclockRecord(p.config_path, p.block, p.config, newline, p.trial_path)
    rec.state = "trial" if p.trial_path else "on"
    save(rec)  # recorded before the write, so an interrupted run can still be undone
    try:
        root_write(ui, remote, plan.tty, [(target, None if p.trial_path else p.config, new)])
    except WizardError:
        save(None)  # nothing was written
        raise
    if not p.trial_path:
        ui.ok("Wrote the overclock block", f"{p.config_path}; it applies on the next boot")
        if oc is not None:
            ui.hint("Reboot the robot to apply it", oc.reboot_hint)
        return
    ui.ok("Wrote the trial config", f"{p.trial_path}; {p.config_path} is unchanged")
    _trial_boot(ui, remote, oc, plan, rec, reboot_flag, save)


def _trial_boot(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord,
    reboot_flag: bool | None,
    save: Callable[[OverclockRecord | None], None],
) -> None:
    """Ask for the trial boot; after it, check and promote, or remove the trial config."""
    if oc is None or rec.trial is None:
        return
    question = "Reboot the robot now into the trial config?"
    command = shlex.join(["python", "scripts/install_on_robot.py", FLAG, REBOOT_FLAG])
    if not ask_reboot(ui, oc, reboot_flag, question):
        ui.hint("Boot the trial config later", command)
        return
    rec.state = "trying"
    save(rec)  # before the reboot: the connection ends with it
    try:
        back = reboot_and_wait(ui, remote, plan.tty, oc.trial_reboot)
    except WizardError:
        rec.state = "trial"  # sudo refused: nothing rebooted
        save(rec)
        raise
    if not back:
        ui.warn("Robot not back", f"not on SSH after {BOOT_TIMEOUT_S:.0f} s")
        ui.info(
            f"Power-cycle the robot: a power cycle boots the unchanged {rec.path}. The next run "
            "of the installer removes the trial config and says what happened."
        )
        return
    _promote_or_drop(ui, remote, oc, plan, rec, save)


def _promote_or_drop(
    ui: Ui,
    remote: Remote,
    oc: Overclock,
    plan: Plan,
    rec: OverclockRecord,
    save: Callable[[OverclockRecord | None], None],
) -> None:
    """On the trial boot: every check passes, then the block goes into the boot config (no
    second reboot: what runs and what the boot config says are the same); else the trial
    config goes and the next normal boot uses the unchanged boot config."""
    assert rec.trial is not None
    checks = [Check("Booted the trial config", oc.trial_booted(remote), rec.trial)]
    checks += oc.verify(remote) if checks[0].ok else []
    report_checks(ui, checks)
    config = read(remote, rec.path)
    if config != rec.backup:
        checks.append(Check("Boot config unchanged", False, f"{rec.path} changed since the trial"))
        ui.warn("Boot config unchanged", f"{rec.path} changed since the trial was written")
    failed = [c for c in checks if not c.ok]
    if failed:
        root_write(ui, remote, plan.tty, [(rec.trial, rec.applied, None)])
        save(None)
        ui.warn("Overclock not promoted", ", ".join(c.name for c in failed))
        ui.info(
            f"Removed the trial config. The robot runs it until its next boot, which uses the "
            f"unchanged {rec.path}."
        )
        return
    promote = [(rec.path, rec.backup, rec.applied), (rec.trial, rec.applied, None)]
    root_write(ui, remote, plan.tty, promote)
    rec.state, rec.trial = "on", None
    save(rec)
    ui.ok(
        "Promoted the overclock",
        f"the block is in {rec.path}; no second reboot: the running config is the same",
    )


def _settle_trial(
    ui: Ui,
    remote: Remote,
    oc: Overclock | None,
    plan: Plan,
    rec: OverclockRecord,
    want: bool,
    reboot_flag: bool | None,
    dry_run: bool,
    save: Callable[[OverclockRecord | None], None],
) -> None:
    """A trial config from an earlier run: promote it when the robot runs it now, report a
    trial boot that did not happen (a power cycle, a hang), offer the trial boot again, or
    remove it when the overclock is no longer wanted."""
    booted = oc is not None and oc.trial_booted(remote)
    if want and booted and oc is not None:
        if dry_run:
            ui.would("Check and promote", rec.path)
            return
        _promote_or_drop(ui, remote, oc, plan, rec, save)
        return
    if want and rec.state == "trial" and oc is not None:
        if dry_run:
            ui.would("Ask for the trial boot", f"{rec.trial} is written")
            return
        ui.ok("Found the trial config", f"{rec.trial}, not booted yet")
        _trial_boot(ui, remote, oc, plan, rec, reboot_flag, save)
        return
    if rec.state == "trying" and not booted:
        ui.warn(
            "Trial boot not finished",
            f"the robot booted the unchanged {rec.path} (a power cycle, or the trial hung)",
        )
    if remove(ui, remote, plan, rec, dry_run, save) == REMOVED and booted:
        offer_reboot(ui, remote, oc, plan, reboot_flag, "the robot runs the trial config now")


def remove(
    ui: Ui,
    remote: Remote,
    plan: Plan,
    rec: OverclockRecord,
    dry_run: bool,
    save: Callable[[OverclockRecord | None], None],
) -> str:
    """Take out exactly what the record says: the trial config, or the block in the boot
    config (every other line unchanged), and drop it from the record. A block or trial config
    edited by hand is refused and the difference printed; the record then keeps it. Returns
    ``REMOVED``, ``GONE`` (nothing was left) or ``EDITED``; the caller offers the reboot."""
    on = rec.state == "on"
    path = rec.path if on else rec.trial
    assert path is not None
    text = read(remote, path)
    label = "overclock block" if on else "trial config"
    if on:
        after = without_block(text or "", rec)
        edited = after is None and has_markers(text or "", rec.block)
        gone = after is None and not edited
        expected, found = rec.block, _found(text or "", rec)
    else:
        after = None
        edited = text is not None and text != rec.applied
        gone = text is None
        expected, found = rec.applied, text or ""
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
    "Check",
    "Overclock",
    "OverclockProfile",
    "Plan",
    "apply",
    "diff",
    "offer_reboot",
    "remove",
    "root_mode",
    "settle",
    "with_block",
    "without_block",
]
