"""``python -m roboemot.webui set-pin --data-dir DIR``: the only way to set the operator PIN.

The PIN is never an argument. On a terminal it is asked twice without echo; otherwise one line
is read from stdin (the installer pipes it over SSH). Replacing the PIN file also clears a login
lockout and ends every web session (SPEC 12).
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from roboemot.config.schema import RoboEmotConfig

from .auth import PinError, PinStore, check_pin_format


def read_pin(stdin=None) -> str | None:
    """The PIN from a terminal prompt (twice) or the first line of stdin; None if they differ."""
    stdin = stdin or sys.stdin
    if stdin.isatty():
        pin = getpass.getpass("New operator PIN: ")
        return pin if pin == getpass.getpass("Repeat: ") else None
    return stdin.readline().rstrip("\r\n")


def main(argv: list[str] | None = None, stdin=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m roboemot.webui")
    sub = parser.add_subparsers(dest="command", required=True)
    setter = sub.add_parser("set-pin", help="set or replace the operator PIN (read from stdin)")
    setter.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    pin = read_pin(stdin)
    if pin is None:
        print("The two entries differ.", file=sys.stderr)
        return 1
    try:
        PinStore(args.data_dir).set(check_pin_format(pin, RoboEmotConfig().webui))
    except PinError as exc:
        print(f"Rejected: {exc}", file=sys.stderr)
        return 1
    print("Operator PIN set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
