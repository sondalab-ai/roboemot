"""Command line: ``roboemot config {validate,schema,dump}``, ``roboemot manifest check``,
``roboemot discover``, ``roboemot debug``, ``roboemot models {list,fetch}`` and
``roboemot llm check`` and ``roboemot secrets {set,remove,list}``."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml

from .config import ConfigError, ConfigStack, GuardrailError, RoboEmotConfig
from .redact import scrub_text

HTTP_CLIENT_FACTORY: Callable[[], httpx.AsyncClient] | None = None  # tests inject a transport


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
    """List the vision and voice models and their state, or download them with sha256 checks."""
    from .vision import models
    from .voice import models as voice_models

    cfg = _stack(args).effective()
    vision_root = models.models_dir(args.cache_dir or cfg.vision.models_dir, cfg.data_dir)
    voice_root = voice_models.models_dir(args.voice_dir or cfg.voice.models_dir, cfg.data_dir)
    groups = [
        (models.load_manifest(), vision_root, models.MANIFEST),
        (voice_models.load_manifest(), voice_root, voice_models.MANIFEST),
    ]
    if args.action == "list":
        for specs, root, _ in groups:
            rows = [
                (s.name, s.role, s.licence, models.status(s, root, verify=args.verify))
                for s in specs.values()
            ]
            widths = [max(len(r[i]) for r in rows) for i in range(3)]
            for name, role, licence, state in rows:
                print(f"{name:{widths[0]}}  {role:{widths[1]}}  {licence:{widths[2]}}  {state}")
            print(f"folder: {root}")
        legacy = models.legacy_dir()
        if vision_root != legacy and legacy.is_dir() and any(legacy.iterdir()):
            print(
                f"note: weights also exist in the old default folder {legacy}; move them to "
                f"{vision_root} or set ROBOEMOT_MODELS_DIR (the default moved under the data "
                f"folder)"
            )
        return 0
    if args.action == "prune":
        removed = models.prune_retired(vision_root)
        if args.sources:
            removed += models.prune_sources(vision_root)
        for path in removed:
            print(f"deleted    {path}")
        print(f"{len(removed)} file(s) deleted from {vision_root}")
        return 0
    wanted = set(args.names)
    if args.all:
        wanted |= {n for specs, _, _ in groups for n in specs}
    if args.voice:
        wanted |= set(groups[1][0])
    if not wanted:
        raise ValueError(
            "name the models to fetch (see `roboemot models list`) or pass --all or --voice"
        )
    known = {n for specs, _, _ in groups for n in specs}
    if wanted - known:
        raise models.ModelError(
            f"unknown model(s) {sorted(wanted - known)}; known: {sorted(known)}"
        )
    for specs, root, manifest in groups:
        names = [n for n in specs if n in wanted]
        if not names:
            continue
        report = models.fetch(names, root, requantize=args.requantize, manifest=manifest)
        for path, outcome in report.files.items():
            print(f"{outcome:10} {root / path}")
    return 0


def _llm(args: argparse.Namespace) -> int:
    """Run the conformance test (SPEC 8) against one configured route; exit 1 on any failure."""
    from .llm.conformance import check_route
    from .secret_store import SecretStore

    cfg = _stack(args).effective()
    route = next((r for r in cfg.llm_routes if r.name == args.route), None)
    if route is None:
        known = ", ".join(r.name for r in cfg.llm_routes) or "none configured"
        raise ValueError(f"no route named {args.route!r} (llm_routes: {known})")

    if cfg.privacy.pseudonyms_to_cloud:
        print(
            "warning: pseudonyms are on, but the CLI does not read the registry; the synthetic "
            "prompts are sent as they are (no registry names are in them)",
            file=sys.stderr,
        )

    async def run():
        return await asyncio.wait_for(
            check_route(
                route,
                cfg.safety,
                settings=cfg.llm_check,
                secrets=SecretStore.for_data_dir(cfg.data_dir).get,
                http_client_factory=HTTP_CLIENT_FACTORY,
            ),
            cfg.llm_check.deadline_s,
        )

    try:
        report = asyncio.run(run())
    except TimeoutError as exc:
        raise RuntimeError(f"check did not finish in {cfg.llm_check.deadline_s:g} s") from exc
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for c in report.checks:
            print(f"{c.status:5} {c.name:20} {c.reason}")
        print(f"route {report.route} ({report.model}): {'ok' if report.ok else 'FAILED'}")
    return 0 if report.ok else 1


def read_secret_value(stdin=None) -> str | None:
    """The value to store: asked twice without echo on a terminal, else the first line of stdin
    (the installer pipes it over SSH). Never an argument. None when the two entries differ."""
    stdin = stdin or sys.stdin
    if stdin.isatty():
        value = getpass.getpass("Secret value: ")
        return value if value == getpass.getpass("Repeat: ") else None
    return stdin.readline().rstrip("\r\n")


def _secrets(args: argparse.Namespace, stdin=None) -> int:
    """Set, remove or list (names only) the secrets of the store (SPEC 8, 12 Data)."""
    from .secret_store import SecretStore, SecretUnavailable

    configured = args.data_dir or _stack(args).effective().data_dir
    store = SecretStore.for_data_dir(configured)
    try:
        if args.action == "list":
            for name in store.names():
                print(name)
            return 0
        if not args.names:
            raise ValueError(f"secrets {args.action} needs a name")
        if args.action == "remove":
            for name in args.names:
                print(f"{'removed' if store.remove(name) else 'not set'}: {name}")
            return 0
        if len(args.names) != 1:
            raise ValueError("secrets set takes exactly one name")
        value = read_secret_value(stdin)
        if value is None:
            raise ValueError("the two entries differ")
        store.set(args.names[0], value)
        print(f"secret {args.names[0]} set")
        return 0
    except SecretUnavailable as exc:
        raise RuntimeError(str(exc)) from exc


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

    mod = sub.add_parser(
        "models",
        help="vision and voice model files: list, fetch, prune (delete the files of replaced "
        "models, such as the FairFace pair) (SPEC 5, 8, 13)",
    )
    mod.add_argument("action", choices=["list", "fetch", "prune"])
    mod.add_argument("names", nargs="*", help="models to fetch (see `list`)")
    mod.add_argument(
        "--all",
        action="store_true",
        help="fetch every vision and voice model (about 0.14 GB of vision and 0.3 GB of voice "
        "models)",
    )
    mod.add_argument("--voice", action="store_true", help="fetch every voice model (about 0.3 GB)")
    mod.add_argument(
        "--requantize",
        action="store_true",
        help="fetch: rebuild quantized files from their sources (none ship today)",
    )
    mod.add_argument(
        "--sources",
        action="store_true",
        help="prune: also delete source files whose quantized files are verified (none ship today)",
    )
    mod.add_argument(
        "--cache-dir", help="models folder (default: vision.models_dir, else the app data folder)"
    )
    mod.add_argument(
        "--voice-dir",
        help="voice models folder (default: voice.models_dir, else the app data folder)",
    )
    mod.add_argument("--verify", action="store_true", help="list: rehash the files (slow)")
    mod.add_argument("--app")
    mod.add_argument("--defaults")
    mod.add_argument("--preset")
    mod.add_argument("--preset-dir", action="append", default=["configs/presets"])

    llm = sub.add_parser("llm", help="LLM route tools (SPEC 8)")
    llm.add_argument("action", choices=["check"])
    llm.add_argument("route", help="route name from llm_routes")
    llm.add_argument("--json", action="store_true")
    llm.add_argument("--app")
    llm.add_argument("--defaults")
    llm.add_argument("--preset")
    llm.add_argument("--preset-dir", action="append", default=["configs/presets"])

    sec = sub.add_parser(
        "secrets",
        help="API keys for ${secret:name} routes: set (value read from the terminal or stdin, "
        "never an argument), remove, list names (SPEC 8, 12)",
    )
    sec.add_argument("action", choices=["set", "remove", "list"])
    sec.add_argument("names", nargs="*", help="secret name (A-Z a-z 0-9 _ . -)")
    sec.add_argument("--data-dir", help="app data folder (default: data_dir config, env, default)")
    sec.add_argument("--app")
    sec.add_argument("--defaults")
    sec.add_argument("--preset")
    sec.add_argument("--preset-dir", action="append", default=["configs/presets"])

    args = parser.parse_args(argv)
    try:
        if args.cmd == "secrets":
            return _secrets(args)
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
        if args.cmd == "llm":
            return _llm(args)
    except (ConfigError, GuardrailError, ValueError, RuntimeError) as exc:
        print(f"error: {scrub_text(str(exc))}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
