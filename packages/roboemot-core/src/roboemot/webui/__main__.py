"""``python -m roboemot.webui set-pin --data-dir DIR``: the admin path for the operator PIN."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from roboemot.config.schema import RoboEmotConfig

from .auth import PinError, PinStore, check_pin_format


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m roboemot.webui")
    sub = parser.add_subparsers(dest="command", required=True)
    setter = sub.add_parser("set-pin", help="set or replace the operator PIN")
    setter.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = RoboEmotConfig().webui
    pin = getpass.getpass("New operator PIN: ")
    if pin != getpass.getpass("Repeat: "):
        print("The two entries differ.", file=sys.stderr)
        return 1
    try:
        PinStore(args.data_dir).set(check_pin_format(pin, cfg))
    except PinError as exc:
        print(f"Rejected: {exc}", file=sys.stderr)
        return 1
    print("Operator PIN set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
