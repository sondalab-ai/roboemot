"""Command line: ``roboemot config {validate,schema,dump}`` and ``roboemot manifest check``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .config import ConfigError, ConfigStack, RoboEmotConfig


def _stack(args: argparse.Namespace) -> ConfigStack:
    return ConfigStack.from_files(
        defaults=Path(args.defaults) if args.defaults else None,
        preset=args.preset,
        preset_dirs=[Path(p) for p in args.preset_dir],
        app=Path(args.app) if args.app else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roboemot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cfg = sub.add_parser("config", help="configuration tools (SPEC 11A)")
    cfg.add_argument("action", choices=["validate", "schema", "dump"])
    cfg.add_argument("--app")
    cfg.add_argument("--defaults")
    cfg.add_argument("--preset")
    cfg.add_argument("--preset-dir", action="append", default=["configs/presets"])

    man = sub.add_parser("manifest", help="robot manifest tools (SPEC 4)")
    man.add_argument("action", choices=["check"])
    man.add_argument("path")
    man.add_argument("--dev", action="store_true", help="allow unmeasured dynamics")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "config":
            if args.action == "schema":
                print(json.dumps(RoboEmotConfig.model_json_schema(), indent=2))
                return 0
            effective = _stack(args).effective()
            if args.action == "dump":
                print(yaml.safe_dump(effective.model_dump(mode="json"), sort_keys=False))
            else:
                print("configuration valid")
            return 0
        if args.cmd == "manifest":
            from .manifest import load_manifest

            m = load_manifest(Path(args.path), allow_unmeasured_dynamics=args.dev)
            print(f"manifest {m.robot.id} valid; capabilities: {sorted(m.capabilities())}")
            return 0
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
