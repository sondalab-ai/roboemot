"""Open a shell on the robot, run a command, install the SSH key, or fetch the debug report.

Runs on a PC on the robot's LAN (SPEC 4A, Packaging; SPEC 12A for ``--debug``). It reuses the
installer's robot choice (mDNS through ``roboemot.discovery``, then the platform's default host
and SSH user) and then replaces itself with ``ssh`` or ``ssh-copy-id``, so the terminal, the
password prompt and the exit code are the real ones. ``--debug`` runs ``roboemot debug --json``
with the ``roboemot`` script the installer put next to the platform's interpreter, so the
report is built on the robot and its ``local`` sections describe the robot.
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import os
import posixpath
import re
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import Options, Platform, installed_platforms, select_robot
from .remote import CONNECT_TIMEOUT_S
from .ui import Ui, WizardError

NOT_INSTALLED_EXIT = 127
CLI_NAME = "roboemot"

USER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9._-]*")
HOST_LABEL = re.compile(r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)")
HOST_MAX_CHARS = 253

Exec = Callable[[str, list[str]], None]


def robot_cli(platform: Platform) -> str:
    """Path of the ``roboemot`` script on the robot: next to the target environment's Python."""
    return posixpath.join(posixpath.dirname(platform.python), CLI_NAME)


def valid_user(user: str) -> str:
    """``user`` if it is a plain login name; it ends up in an ssh argument."""
    if not USER_PATTERN.fullmatch(user):
        raise WizardError(f"not a valid SSH user name: {user!r}")
    return user


def valid_host(host: str) -> str:
    """``host`` if it is an IP address or an RFC 1123 hostname (no ``-`` first, no ``@``/``%``)."""
    if host and not any(c in host for c in "@%"):
        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        name = host.rstrip(".")
        if (
            name
            and len(name) <= HOST_MAX_CHARS
            and all(HOST_LABEL.fullmatch(label) for label in name.split("."))
        ):
            return host
    raise WizardError(f"not a valid robot host name or IP address: {host!r}")


def debug_command(platform: Platform) -> str:
    """Remote shell line: run the report, or say clearly that the app is not installed."""
    cli = shlex.quote(robot_cli(platform))
    robot = shlex.quote(platform.manifest_id)
    missing = shlex.quote(
        "RoboEmot is not installed on the robot "
        f"({robot_cli(platform)} is missing): run scripts/install_on_robot.py"
    )
    return (
        f"test -x {cli} || {{ echo {missing} >&2; exit {NOT_INSTALLED_EXIT}; }}; "
        f"{cli} debug --robot {robot} --json"
    )


def build_argv(
    opts: Options, command: Sequence[str], copy_id: bool, debug: bool, ui: Ui, root: Path
) -> list[str]:
    """The ``ssh`` / ``ssh-copy-id`` command line for these options.

    The robot comes from ``--host``/``--robot`` or mDNS, the user from ``--user`` or the
    platform's usual one.
    """
    if opts.host:
        valid_host(opts.host)
    if opts.user:
        valid_user(opts.user)
    host, platform = select_robot(ui, opts, root)
    target = f"{valid_user(opts.user or platform.default_user)}@{valid_host(host)}"
    if copy_id:
        return ["ssh-copy-id", "--", target]
    argv = ["ssh", "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}", "--", target]
    if debug:
        return [*argv, debug_command(platform)]
    return [*argv, shlex.join(command)] if command else argv


def parse_args(argv: list[str] | None = None) -> tuple[Options, list[str], bool, bool]:
    """Flags, then the remote command after ``--`` (if any)."""
    args = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in args:
        cut = args.index("--")
        args, command = args[:cut], args[cut + 1 :]
    p = argparse.ArgumentParser(
        description="SSH into the robot (SPEC 4A). After `--`: a command to run there.",
        usage="%(prog)s [options] [-- command ...]",
    )
    p.add_argument("--host", help="robot hostname or IP (default: pick from robots found by mDNS)")
    p.add_argument("--user", help="SSH user (default: the platform's usual user)")
    p.add_argument(
        "--robot",
        help=f"manifest robot.id (installed: {', '.join(installed_platforms()) or 'none'})",
    )
    p.add_argument("--copy-id", action="store_true", help="install your SSH key (ssh-copy-id)")
    p.add_argument(
        "--debug", action="store_true", help="run `roboemot debug --json` on the robot (SPEC 12A)"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command, do not run it (discovery still runs)",
    )
    a = p.parse_args(args)
    if a.debug and a.copy_id:
        p.error("--debug and --copy-id exclude each other")
    if command and (a.debug or a.copy_id):
        p.error("a command after `--` cannot be combined with --debug or --copy-id")
    return (
        Options(host=a.host, user=a.user, robot=a.robot, dry_run=a.dry_run),
        command,
        a.copy_id,
        a.debug,
    )


def main(
    argv: list[str] | None = None,
    root: Path | None = None,
    execvp: Exec = os.execvp,
    ui: Ui | None = None,
) -> int:
    opts, command, copy_id, debug = parse_args(argv)
    ui = ui or Ui(interactive=sys.stdin.isatty() and not os.environ.get("CI"), assume_yes=False)
    try:
        # Prompts and discovery messages go to stderr: stdout stays the command's own output.
        with contextlib.redirect_stdout(sys.stderr):
            cmd = build_argv(opts, command, copy_id, debug, ui, root or Path.cwd())
    except WizardError as e:
        ui.fail(str(e))
        return 1
    except KeyboardInterrupt:
        ui.fail("interrupted")
        return 130
    if opts.dry_run:
        print(shlex.join(cmd))
        return 0
    try:
        execvp(cmd[0], cmd)
    except OSError as e:
        ui.fail(f"cannot run {cmd[0]}: {e}")
        return 1
    return 0  # only reached when execvp is replaced (tests); a real exec never returns


__all__ = ["build_argv", "debug_command", "main", "parse_args", "robot_cli"]
