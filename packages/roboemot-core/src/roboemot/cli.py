"""Command line: ``roboemot config {validate,schema,dump}``, ``roboemot manifest check``,
``roboemot discover``, ``roboemot debug`` and ``roboemot models {list,fetch}``."""

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


def _debug(args: argparse.Namespace) -> int:
    """Print the debug report (SPEC 12A, Diagnostics) for this machine and its robot."""
    from . import diagnostics
    from .diagnostics.system import local_sources

    host, port, robot = args.host, args.port, args.robot
    if args.discover:
        from . import discovery

        hints = discovery.load_hints(sorted(Path(args.manifest_dir).glob("*.yaml")))
        found = discovery.discover(hints)
        if not found:
            raise RuntimeError("no robot found on the LAN")
        host, port, robot = found[0].address, found[0].record.port, robot or found[0].manifest_id
    installed = diagnostics.installed_platforms()
    if robot is None and len(installed) == 1:
        robot = installed[0]
    target = diagnostics.Target(host, port)
    notes, sources = [], []
    if target.remote:
        notes.append(
            f"system, temperatures and battery describe the machine that runs this command, "
            f"so they are left out for the remote robot {host}; run `roboemot debug` on the "
            f"robot (e.g. over ssh) to include them"
        )
    else:
        sources += local_sources(process_match=args.process)
    if robot is None:
        notes.append(f"no robot selected (--robot); installed platforms: {installed or 'none'}")
    else:
        sources += diagnostics.platform_sources(robot, target)
    timeout = args.timeout if args.timeout is not None else diagnostics.SECTION_TIMEOUT_S
    report = diagnostics.collect(sources, timeout_s=timeout, notes=notes)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(diagnostics.render_text(report))
    return 0 if all(s["ok"] for s in report["sections"].values()) else 1


def _models(args: argparse.Namespace) -> int:
    """List the vision models and their state, or download them with sha256 verification."""
    from .vision import models

    configured = args.cache_dir or _stack(args).effective().vision.models_dir
    root = models.models_dir(configured)
    specs = models.load_manifest()
    if args.action == "list":
        for spec in specs.values():
            state = models.status(spec, root, verify=args.verify)
            print(f"{spec.name:16} {spec.role:20} {spec.licence:11} {state}")
        print(f"folder: {root}")
        return 0
    if args.action == "prune":
        if not args.sources:
            raise ValueError(
                "prune needs --sources (delete FP32 files whose INT8 files are verified)"
            )
        removed = models.prune_sources(root)
        for path in removed:
            print(f"deleted    {path}")
        print(f"{len(removed)} source file(s) deleted")
        return 0
    names = list(specs) if args.all else args.names
    if not names:
        raise ValueError("name the models to fetch (see `roboemot models list`) or pass --all")

    report = models.fetch(names, root, requantize=args.requantize)
    for path, outcome in report.files.items():
        print(f"{outcome:10} {root / path}")
    return 0


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

    dbg = sub.add_parser("debug", help="robot, hardware, resources, system, battery (SPEC 12A)")
    dbg.add_argument("--robot", help="manifest robot.id (default: the only installed platform)")
    dbg.add_argument("--host", help="robot address (default: this machine)")
    dbg.add_argument("--port", type=int, default=None, help="platform API port")
    dbg.add_argument("--discover", action="store_true", help="find the robot by mDNS")
    dbg.add_argument("--manifest-dir", default="configs/manifests")
    dbg.add_argument("--process", default="roboemot", help="command-line match for the app")
    dbg.add_argument("--timeout", type=float, default=None, help="seconds per section")
    dbg.add_argument("--json", action="store_true")

    mod = sub.add_parser("models", help="vision model files: list, fetch, prune (SPEC 5, 13)")
    mod.add_argument("action", choices=["list", "fetch", "prune"])
    mod.add_argument("names", nargs="*", help="models to fetch (see `list`)")
    mod.add_argument(
        "--all",
        action="store_true",
        help="fetch every model (about 0.8 GB, plus 0.17 GB of INT8 files made from it)",
    )
    mod.add_argument(
        "--requantize", action="store_true", help="fetch: rebuild INT8 files from the FP32 sources"
    )
    mod.add_argument(
        "--sources",
        action="store_true",
        help="prune: delete FP32 files whose INT8 files are verified",
    )
    mod.add_argument(
        "--cache-dir", help="models folder (default: vision.models_dir, else ~/.cache)"
    )
    mod.add_argument("--verify", action="store_true", help="list: rehash the files (slow)")
    mod.add_argument("--app")
    mod.add_argument("--defaults")
    mod.add_argument("--preset")
    mod.add_argument("--preset-dir", action="append", default=["configs/presets"])

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
        if args.cmd == "debug":
            return _debug(args)
        if args.cmd == "models":
            return _models(args)
    except (ConfigError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
