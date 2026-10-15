"""Robot installer: install, verify, start and cleanly uninstall the app (SPEC 4A, Packaging).

Runs on a PC on the robot's LAN, never on the robot. The flow is robot-agnostic; everything
that depends on the robot model sits behind a :class:`Platform`, registered by the platform
package in the ``roboemot.installers`` entry-point group under the manifest's ``robot.id``
(Reachy Mini: ``roboemot_reachymini.install``).

Install: find the robot (mDNS, every manifest that declares discovery), check SSH, the target
interpreter, architecture, Python and free space, build the wheels locally, install them with
the versions already on the robot pinned, record exactly what changed, verify, optionally
start the app and make it the autostart app.

Uninstall: stop the app, give back the previous autostart app (only if the setting still names
this app), uninstall what the record says this installer added, reinstall the original versions
it replaced, keep anything another distribution still needs, remove leftovers, verify, and only
then delete the record. Running it twice is harmless. Both flows have a read-only ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from .build import build_wheels, distribution_names, min_python, third_party_requirements
from .pyenv import RemotePython, report_changes
from .record import InstallRecord, canonical, plan_uninstall
from .remote import Remote
from .ui import Ui, WizardError, paint

ENTRY_POINT_GROUP = "roboemot.installers"
MIN_FREE_MB = 300  # wheels plus numpy/scipy/openai if the target environment lacks them
REMOTE_TMP_PREFIX = "/tmp/roboemot-install"
TMP_DIR_PATTERN = re.compile(re.escape(REMOTE_TMP_PREFIX) + r"\.[A-Za-z0-9]+")
INSTALL_STEPS = 6
UNINSTALL_STEPS = 6


class Platform(Protocol):
    """What the installer needs to know about one robot model."""

    manifest_id: str
    display_name: str
    app_name: str
    default_host: str
    default_user: str
    arch: str  # expected `platform.machine()` of the robot
    python: str  # interpreter of the environment the app is installed into
    record_path: str  # change record: outside that environment and outside /tmp
    space_paths: tuple[str, ...]  # filesystems that need MIN_FREE_MB
    distributions: tuple[str, ...]  # this app's own distributions
    modules: tuple[str, ...]  # their top-level import names (leftover check)

    def connect(self, ui: Ui, host: str) -> None: ...
    def probe(self, ui: Ui, py: RemotePython) -> None: ...
    def stage(self, root: Path, dest: Path, dev: bool) -> list[Path]: ...
    def verify(self, ui: Ui, py: RemotePython, dev: bool) -> None: ...
    def running_app(self) -> str | None: ...
    def stop_app(self) -> None: ...
    def start(self, ui: Ui) -> None: ...
    def autostart(self) -> str | None: ...
    def set_autostart(self, name: str | None) -> None: ...
    def cleanup(self, ui: Ui, py: RemotePython, dry_run: bool) -> None: ...
    def verify_removed(self, ui: Ui) -> None: ...


@dataclass
class Options:
    host: str | None = None
    user: str | None = None
    robot: str | None = None
    dev: bool | None = None
    start: bool | None = None
    startup: bool | None = None
    dry_run: bool = False
    yes: bool = False
    uninstall: bool = False


# ---------------------------------------------------------------- platforms


def installed_platforms() -> list[str]:
    return sorted({ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)})


def load_platform(manifest_id: str) -> Platform:
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == manifest_id:
            try:
                return ep.load()()
            except Exception as e:
                raise WizardError(f"installer for {manifest_id} failed to load: {e}") from e
    have = ", ".join(installed_platforms()) or "none"
    raise WizardError(f"no installer for {manifest_id} (installed: {have})")


def find_robots(root: Path) -> list[tuple[str, str, str]]:
    """Robots on the LAN that have an installer, as (label, address, manifest id)."""
    from ..discovery import discover, load_hints

    supported = set(installed_platforms())
    hints = [h for h in load_hints(sorted((root / "configs/manifests").glob("*.yaml")))]
    found = discover([h for h in hints if h.id in supported])
    return [
        (
            f"{f.record.name}  {f.address}  ({f.record.host}, {f.manifest_id})",
            f.address,
            f.manifest_id,
        )
        for f in found
    ]


def select_robot(ui: Ui, opts: Options, root: Path) -> tuple[str, Platform]:
    """Host and platform from the flags, else from mDNS, else by asking."""
    robot = opts.robot
    if opts.host and robot is None and len(installed_platforms()) == 1:
        robot = installed_platforms()[0]
    if opts.host and robot:
        return opts.host, load_platform(robot)
    ui.info("looking for robots on the network (mDNS)")
    try:
        found = [f for f in find_robots(root) if robot in (None, f[2])]
    except (ImportError, RuntimeError, OSError) as e:
        ui.warn(f"discovery unavailable: {e}")
        found = []
    if found:
        i = ui.choose("Which robot?", [label for label, _, _ in found] + ["another host or IP"])
        if i < len(found):
            return found[i][1], load_platform(found[i][2])
    if robot is None:
        options = installed_platforms()
        if not options:
            raise WizardError("no robot installer is installed")
        robot = (
            options[ui.choose("Which robot model?", options)] if len(options) > 1 else options[0]
        )
    platform = load_platform(robot)
    prompt = "No robot found. Robot hostname or IP" if not found else "Robot hostname or IP"
    return opts.host or ui.ask(prompt, platform.default_host), platform


# ---------------------------------------------------------------- shared steps


def check_ssh(
    ui: Ui, platform: Platform, remote: Remote, need: tuple[int, int] | None
) -> RemotePython:
    """Log in once; check the target interpreter, architecture, Python and free space."""
    ui.info(f"connecting to {remote.target} (ssh may ask for the password)")
    remote.connect()
    ui.ok(f"ssh {remote.target}")
    py = RemotePython(remote, platform.python)
    if not py.exists():
        raise WizardError(f"{platform.python} not found: is this a {platform.display_name}?")
    arch, version = py.py(
        "import sys, platform; print(platform.machine()); print('%d.%d' % sys.version_info[:2])"
    ).splitlines()
    if arch != platform.arch:
        raise WizardError(f"robot architecture is {arch}, expected {platform.arch}")
    if need and tuple(map(int, version.split("."))) < need:
        raise WizardError(f"target Python is {version}; the app needs {need[0]}.{need[1]}+")
    ui.ok(f"{arch}, Python {version} at {platform.python}")
    if need:
        for path in platform.space_paths:
            free_kb = int(remote.run(f"df -Pk {shlex.quote(path)} | tail -1").split()[3])
            if free_kb < MIN_FREE_MB * 1024:
                raise WizardError(
                    f"only {free_kb // 1024} MB free on {path}, need {MIN_FREE_MB} MB"
                )
            ui.ok(f"{free_kb // 1024} MB free on {path}")
    platform.probe(ui, py)
    return py


def read_record(remote: Remote, path: str) -> InstallRecord | None:
    raw = remote.run(f"cat {shlex.quote(path)} 2>/dev/null || true")
    return InstallRecord.from_json(json.loads(raw)) if raw else None


def write_record(remote: Remote, path: str, record: InstallRecord) -> None:
    data, tmp = json.dumps(record.to_json(), indent=1), f"{path}.tmp"
    remote.run(
        f"printf %s {shlex.quote(data)} > {shlex.quote(tmp)} "
        f"&& mv {shlex.quote(tmp)} {shlex.quote(path)}"
    )


def _with_remote(opts: Options, ui: Ui, platform: Platform, host: str, body) -> None:
    user = opts.user or ui.ask("SSH user", platform.default_user)
    # The ssh control socket path must stay under 104 bytes: macOS $TMPDIR is too long.
    with tempfile.TemporaryDirectory(prefix="re-ssh-", dir="/tmp") as sock_dir:
        remote = Remote(user, host, Path(sock_dir))
        try:
            body(remote)
        finally:
            remote.close()


# ---------------------------------------------------------------- install


def install_packages(
    ui: Ui,
    platform: Platform,
    py: RemotePython,
    remote: Remote,
    dirs: list[Path],
    wheels: list[Path],
    dry_run: bool,
) -> InstallRecord:
    own = distribution_names(dirs)
    before = py.installed()
    record = read_record(remote, platform.record_path) or InstallRecord(
        platform.app_name, platform.manifest_id, own
    )
    if record.pending_before is not None:
        ui.warn("an earlier install was interrupted; folding its changes into the record")
        record.merge(record.pending_before, before)
    record.packages = sorted(set(record.packages) | set(own))
    pins = py.constraints(before, set(own))
    if dry_run:
        reqs = third_party_requirements(dirs)
        plan = py.plan_install(reqs, pins)
        if plan.ok:
            added, _ = report_changes(_json(plan.output), before)
            ui.ok(
                f"would add {', '.join(f'{n} {v}' for n, v in sorted(added.items()))}"
                if added
                else "dependencies already satisfied"
            )
        else:
            free = py.plan_install(reqs, None)
            if not free.ok:
                raise WizardError(f"dependencies cannot be resolved on the robot:\n{free.output}")
            added, changed = report_changes(_json(free.output), before)
            ui.warn(
                "would need to change shared packages: "
                f"{', '.join(f'{n} {c}' for n, c in changed.items())} (asks first)"
            )
            if added:
                ui.ok(f"would add {', '.join(f'{n} {v}' for n, v in sorted(added.items()))}")
        ui.info(f"would record the changes in {platform.record_path}; nothing copied or installed")
        return record
    tmp = remote.run(f"mktemp -d {REMOTE_TMP_PREFIX}.XXXXXX")
    try:
        remote.copy(wheels, tmp)
        ui.ok(f"copied {', '.join(w.name for w in wheels)}")
        files = [f"{tmp}/{w.name}" for w in wheels]
        record.pending_before = before
        write_record(remote, platform.record_path, record)
        ui.info("installing (first time: numpy/scipy may take minutes)")
        result = py.install(files, pins)
        if not result.ok:
            ui.warn("the app does not fit the package versions already on the robot")
            free = py.plan_install(files, None)
            if not free.ok:
                raise WizardError(f"pip cannot resolve the dependencies:\n{free.output}")
            _, changed = report_changes(_json(free.output), before)
            for n, c in changed.items():
                ui.info(f"{n} {c}")
            question = "Change these shared packages? Uninstall restores them (needs internet)"
            if not ui.confirm(question, False):
                raise WizardError("install cancelled: shared packages left as they were")
            result = py.install(files, None)
            if not result.ok:
                raise WizardError(f"pip install failed on the robot:\n{result.output}")
        py.reinstall_own(files)
        after = py.installed()
        record.merge(before, after)
        write_record(remote, platform.record_path, record)
        added = ", ".join(record.added) or "none"
        ui.ok(
            f"installed; recorded in {platform.record_path} "
            f"(dependencies added by this installer: {added})"
        )
    finally:
        remote.run(f"rm -rf {shlex.quote(tmp)}", check=False)
    return record


def run_install(opts: Options, ui: Ui, root: Path) -> None:
    ui.steps = INSTALL_STEPS
    ui.title("installer", "dry run: the robot is not modified" if opts.dry_run else "")
    ui.step("Robot")
    host, platform = select_robot(ui, opts, root)
    ui.ok(platform.display_name)
    platform.connect(ui, host)

    def body(remote: Remote) -> None:
        with tempfile.TemporaryDirectory(prefix="roboemot-") as tmp:
            ui.step("Configuration")
            dev = opts.dev
            if dev is None:
                ui.info("manifest dynamics are not measured yet: dev mode lets the app run")
                ui.info(
                    "with conservative fallback limits (SPEC 12); without it, it refuses to start"
                )
                dev = ui.confirm("Install in dev mode?", True)
            ui.ok("dev mode (configs/dev-unmeasured.yaml)" if dev else "production config")
            dirs = platform.stage(root, Path(tmp) / "src", dev)

            ui.step("SSH access")
            py = check_ssh(ui, platform, remote, min_python(dirs))

            ui.step("Build")
            wheels = build_wheels(dirs, Path(tmp) / "wheels")
            ui.ok(f"built {', '.join(w.name for w in wheels)}")

            ui.step("Install (dependency check only)" if opts.dry_run else "Install")
            record = install_packages(ui, platform, py, remote, dirs, wheels, opts.dry_run)
            if not opts.dry_run:
                platform.verify(ui, py, dev)

            ui.step("Start")
            if opts.dry_run:
                running = platform.running_app()
                action = f"would stop {running}, " if running else "no app running; would "
                ui.info(f"{action}wake the robot and start {platform.app_name}")
                ui.info(f"autostart app now: {platform.autostart() or 'none'}")
                ui.title("dry run passed")
                return
            if (
                opts.start
                if opts.start is not None
                else ui.confirm(f"Start {platform.app_name} now?", True)
            ):
                platform.start(ui)
            startup = opts.startup
            if startup is None:
                startup = ui.confirm(
                    f"Start {platform.app_name} automatically when the robot wakes up?", False
                )
            if startup:
                if not record.autostart_recorded:
                    record.autostart_before, record.autostart_recorded = platform.autostart(), True
                    write_record(remote, platform.record_path, record)
                platform.set_autostart(platform.app_name)
                ui.ok(f"{platform.app_name} starts automatically on wake-up")
            ui.title("installed")
            ui.info("remove everything it changed: python scripts/install_on_robot.py --uninstall")

    _with_remote(opts, ui, platform, host, body)


# ---------------------------------------------------------------- uninstall


def run_uninstall(opts: Options, ui: Ui, root: Path) -> None:
    ui.steps = UNINSTALL_STEPS
    ui.title("uninstaller", "dry run: the robot is not modified" if opts.dry_run else "")
    ui.step("Robot")
    host, platform = select_robot(ui, opts, root)
    ui.ok(platform.display_name)
    platform.connect(ui, host)
    would = "would " if opts.dry_run else ""

    def body(remote: Remote) -> None:
        ui.step("SSH access")
        py = check_ssh(ui, platform, remote, None)

        ui.step("Install record")
        installed = py.installed()
        record = read_record(remote, platform.record_path)
        if record is None:
            ui.warn(
                f"no install record at {platform.record_path}: only the app's own packages "
                "are removed; dependencies an earlier install added cannot be told apart"
            )
            record = InstallRecord(
                platform.app_name,
                platform.manifest_id,
                [canonical(d) for d in platform.distributions],
            )
        else:
            ui.ok(
                f"record found: added {', '.join(record.added) or 'nothing'}; "
                f"replaced {', '.join(record.replaced) or 'nothing'}"
            )
            if record.pending_before is not None:
                ui.warn("the last install was interrupted; its changes are included")
                record.merge(record.pending_before, installed)

        ui.step("App")
        if platform.running_app() == platform.app_name:
            if not opts.dry_run:
                platform.stop_app()
            ui.ok(f"{would}stop {platform.app_name}")
        else:
            ui.ok(f"{platform.app_name} is not running")
        current = platform.autostart()
        if current == platform.app_name:
            if not opts.dry_run:
                platform.set_autostart(record.autostart_before)
            ui.ok(f"{would}set the autostart app back to {record.autostart_before or 'none'}")
        else:
            ui.ok(f"autostart app is {current or 'none'}: left as it is")

        ui.step("Packages")
        plan = plan_uninstall(record, installed, py.requires())
        for name, users in plan.kept.items():
            ui.warn(f"keeping {name}: needed by {', '.join(users)}")
        for name, version in plan.drifted.items():
            ui.warn(f"{name} changed since the install (now {version}): left as it is")
        if plan.remove:
            ui.ok(f"{would}remove {', '.join(f'{n} {installed[n]}' for n in plan.remove)}")
        if plan.restore:
            ui.ok(
                f"{would}reinstall "
                f"{', '.join(f'{n} {v}' for n, v in plan.restore.items())} (needs internet)"
            )
        if plan.empty:
            ui.ok("no package to remove or restore")
        if not opts.dry_run:
            py.uninstall(plan.remove)
            py.install_exact(plan.restore)
        if not (opts.dry_run and plan.remove):  # before removal, the modules are expected
            for module, path in leftovers(py, platform):
                if not opts.dry_run:
                    remote.run(f"rm -rf {shlex.quote(path)}")
                ui.warn(f"{would}delete leftover {path} ({module})")

        ui.step("Clean up and verify")
        platform.cleanup(ui, py, opts.dry_run)
        listed = remote.run(f"ls -d {REMOTE_TMP_PREFIX}.* 2>/dev/null || true").split()
        tmp_dirs = [d for d in listed if TMP_DIR_PATTERN.fullmatch(d)]
        for d in tmp_dirs:
            if not opts.dry_run:
                remote.run(f"rm -rf {shlex.quote(d)}")
            ui.ok(f"{would}delete {d}")
        if opts.dry_run:
            ui.info(f"{would}delete {platform.record_path} once verified")
            ui.title("dry run passed")
            return
        verify_uninstall(ui, platform, py, record, plan.restore)
        remote.run(f"rm -f {shlex.quote(platform.record_path)}")
        ui.ok(f"deleted {platform.record_path}")
        ui.title("uninstalled")

    _with_remote(opts, ui, platform, host, body)


def verify_uninstall(
    ui: Ui, platform: Platform, py: RemotePython, record: InstallRecord, restored: dict[str, str]
) -> None:
    """The app is gone and every restored version is back; else keep the record and stop."""
    now = py.installed()
    problems = [f"{n} still installed" for n in map(canonical, record.packages) if n in now]
    problems += [
        f"{n} is {now.get(n, 'missing')}, expected {v}"
        for n, v in restored.items()
        if now.get(n) != v
    ]
    problems += [f"{p} still present ({m})" for m, p in leftovers(py, platform)]
    if problems:
        raise WizardError("uninstall incomplete (record kept, run again): " + "; ".join(problems))
    ui.ok("app packages gone, original versions back")
    platform.verify_removed(ui)


def leftovers(py: RemotePython, platform: Platform) -> list[tuple[str, str]]:
    """The app's modules still found inside the target environment's site-packages."""
    site = posixpath.normpath(py.site_packages())
    found = py.find_modules(list(platform.modules))
    return [(m, p) for m, paths in found.items() for p in paths if _inside(p, site)]


def _inside(path: str, parent: str) -> bool:
    """``path`` is strictly below ``parent`` once normalised (no ``..`` escape): it is about
    to be deleted, and it comes from the robot."""
    norm = posixpath.normpath(path)
    return posixpath.isabs(norm) and norm != parent and norm.startswith(parent + "/")


def _json(output: str) -> dict:
    """pip's ``--report -`` JSON, skipping any warning lines before it."""
    start = output.find("{")
    if start < 0:
        raise WizardError(f"no report from pip:\n{output}")
    return json.JSONDecoder().raw_decode(output[start:])[0]


# ---------------------------------------------------------------- command line


def parse_args(argv: list[str] | None = None) -> Options:
    p = argparse.ArgumentParser(description="Install or uninstall RoboEmot on a robot (SPEC 4A).")
    p.add_argument("--host", help="robot hostname or IP (default: pick from robots found by mDNS)")
    p.add_argument("--user", help="SSH user (default: ask, the platform's usual user)")
    p.add_argument(
        "--robot",
        help=f"manifest robot.id (installed: {', '.join(installed_platforms()) or 'none'})",
    )
    p.add_argument("--dev", action=argparse.BooleanOptionalAction, help="dev config (SPEC 12)")
    p.add_argument(
        "--start", action=argparse.BooleanOptionalAction, help="start the app after install"
    )
    p.add_argument(
        "--startup-app",
        dest="startup",
        action=argparse.BooleanOptionalAction,
        help="start the app automatically when the robot wakes up",
    )
    p.add_argument(
        "--uninstall", action="store_true", help="remove everything the installer changed"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="read-only checks; nothing on the robot changes"
    )
    p.add_argument("-y", "--yes", action="store_true", help="answer yes to every confirmation")
    a = p.parse_args(argv)
    return Options(
        a.host, a.user, a.robot, a.dev, a.start, a.startup, a.dry_run, a.yes, a.uninstall
    )


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    opts = parse_args(argv)
    ui = Ui(interactive=sys.stdin.isatty() and not os.environ.get("CI"), assume_yes=opts.yes)
    root = root or Path.cwd()
    try:
        (run_uninstall if opts.uninstall else run_install)(opts, ui, root)
    except WizardError as e:
        ui.fail(str(e))
        return 1
    except KeyboardInterrupt:
        ui.fail("interrupted")
        return 130
    return 0


__all__ = [
    "ENTRY_POINT_GROUP",
    "Options",
    "Platform",
    "Ui",
    "WizardError",
    "load_platform",
    "main",
    "paint",
    "run_install",
    "run_uninstall",
]
