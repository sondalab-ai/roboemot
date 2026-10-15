"""Command line: ``roboemot config {validate,schema,dump}``, ``roboemot manifest check`` and
``roboemot discover``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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


def _discover(args: argparse.Namespace) -> int:
    """List robots answering on the LAN, matched to the manifests in ``--manifest-dir``."""
    from . import discovery

    hints = discovery.load_hints(sorted(Path(args.manifest_dir).glob("*.yaml")))
    timeout = args.timeout if args.timeout is not None else discovery.DEFAULT_TIMEOUT_S
    found = discovery.discover(hints, timeout)
    if args.json:
        rows = [{"manifest": f.manifest_id, "address": f.address} | asdict(f.record) for f in found]
        print(json.dumps(rows, indent=2))
    elif not found:
        looked_for = ", ".join(h.id for h in hints) or "no manifest declares discovery"
        print(f"no robot found ({looked_for})")
    for f in [] if args.json else found:
        r = f.record
        print(f"{r.name}  {f.manifest_id}  {f.address}:{r.port}  ({r.host})")
    return 0 if found else 1


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

    disc = sub.add_parser("discover", help="find robots on the LAN by mDNS (SPEC 4)")
    disc.add_argument("--manifest-dir", default="configs/manifests")
    disc.add_argument("--timeout", type=float, default=None, help="seconds to listen")
    disc.add_argument("--json", action="store_true")

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
        if args.cmd == "discover":
            return _discover(args)
    except (ConfigError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
