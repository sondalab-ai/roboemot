"""Robot installer: install, verify, start and cleanly uninstall the app (SPEC 4A, Packaging).

Runs on a PC on the robot's LAN, never on the robot. The flow is robot-agnostic; everything
that depends on the robot model sits behind a :class:`Platform`, registered by the platform
package in the ``roboemot.installers`` entry-point group under the manifest's ``robot.id``
(Reachy Mini: ``roboemot_reachymini.install``).

Install: find the robot (mDNS, every manifest that declares discovery), check SSH, the target
interpreter, architecture, Python and free space, build the wheels locally, install them with
the versions already on the robot pinned, record exactly what changed, verify, optionally
start the app and make it the autostart app. It also creates the private app data folder
(``~/.local/share/roboemot``: registry, models, trait cache) when it is not there yet and
records it. Unless declined (`--no-piper`) it also creates a Piper virtualenv in the data folder
(`piper-tts` is GPL-3.0 and runs as its own process, SPEC 8), recorded for uninstall.
Optionally (``--secret NAME``, or asked on a terminal) it stores an API key in the robot's secret
store (SPEC 8, 12 Data), recorded by name; uninstall removes only the names it added.

Uninstall: stop the app, give back the previous autostart app (only if the setting still names
this app), uninstall what the record says this installer added, reinstall the original versions
it replaced, keep anything another distribution still needs, remove leftovers and the recorded
data folder (registry, models and trait cache included), verify, and only then delete the
record. Running it twice is harmless. Both flows have a read-only ``--dry-run``.
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
import traceback
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

import yaml

from .. import paths
from ..secret_store import NAME as SECRET_NAME
from .build import build_wheels, distribution_names, min_python, third_party_requirements
from .piper_lock import lock_path
from .pyenv import RemotePython, report_changes
from .record import InstallRecord, canonical, plan_uninstall
from .remote import Remote
from .ui import (
    COMMAND,
    HeaderParser,
    TermError,
    Ui,
    WizardError,
    app_version,
    last_line,
    print_header,
)

try:
    from sondalab_palette import term
except ImportError:  # pragma: no cover - the robot imports this package and has no palette
    term = None

ENTRY_POINT_GROUP = "roboemot.installers"
MIN_FREE_MB = 300  # wheels plus numpy/scipy/openai if the target environment lacks them
REMOTE_TMP_PREFIX = "/tmp/roboemot-install"
TMP_DIR_PATTERN = re.compile(re.escape(REMOTE_TMP_PREFIX) + r"\.[A-Za-z0-9]+")
PIN_FILE = "webui_pin.json"  # roboemot.webui.auth.PIN_FILE, inside the app data folder
PIN_RETRIES = 3
DEFAULT_SECRET_NAME = "cloud_key"  # the name configs/companion.yaml gives the cloud route key
PIPER_TTS_VERSION = "1.8.0"  # exact pin: the worker protocol is tested against this API
PLATFORM_PROBE = (
    "import sys, platform; print(platform.machine()); print('%d.%d' % sys.version_info[:2])"
)
PROBE_PATTERN = re.compile(r"[A-Za-z0-9_]+\n\d+\.\d+")
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
    piper: bool | None = None
    secrets: list[str] | None = None
    debug: bool = False
    version: bool = False


# ---------------------------------------------------------------- platforms


def installed_platforms() -> list[str]:
    return sorted({ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)})


def load_platform(manifest_id: str) -> Platform:
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == manifest_id:
            try:
                return ep.load()()
            except Exception as e:
                raise WizardError(f"Could not load the {manifest_id} installer", str(e)) from e
    have = ", ".join(installed_platforms()) or "none"
    raise WizardError(f"No installer for {manifest_id}", f"installed: {have}")


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


def _robot_model(ui: Ui, robot: str | None) -> str:
    """``--robot``, else the only installed platform, else ask which one."""
    if robot is not None:
        return robot
    options = installed_platforms()
    if not options:
        raise WizardError(
            "No robot installer is installed",
            "no package registers one in the roboemot.installers entry-point group",
            'pip install -e "packages/roboemot-reachymini"',
        )
    if len(options) == 1:
        return options[0]
    return options[ui.choose("Which robot model?", options, flag="--robot")]


def select_robot(ui: Ui, opts: Options, root: Path) -> tuple[str, Platform]:
    """Host and platform from the flags, else from mDNS, else by asking.

    A given ``--host`` skips discovery. Without one and without a terminal nothing is looked
    up or picked: the run stops naming ``--host`` (never the first robot found).
    """
    if opts.host:
        return opts.host, load_platform(_robot_model(ui, opts.robot))
    ui.require("--host", False)
    robot = opts.robot
    try:
        with ui.running("Looking for robots (mDNS)…"):
            found = [f for f in find_robots(root) if robot in (None, f[2])]
        ui.ok("Looked for robots (mDNS)", f"{len(found)} found")
    except (ImportError, RuntimeError, OSError) as e:
        ui.warn("Discovery unavailable", str(e))
        found = []
    if found:
        labels = [label for label, _, _ in found] + ["another host or IP"]
        i = ui.choose("Which robot?", labels, flag="--host")
        if i < len(found):
            return found[i][1], load_platform(found[i][2])
    platform = load_platform(_robot_model(ui, robot))
    prompt = "No robot found. Robot hostname or IP" if not found else "Robot hostname or IP"
    return ui.ask(prompt, platform.default_host, flag="--host"), platform


# ---------------------------------------------------------------- shared steps


def probe_platform(remote: Remote, python: str) -> tuple[str, str]:
    """The interpreter's CPU architecture and ``"major.minor"``; anything else is an error."""
    out = RemotePython(remote, python).py(PLATFORM_PROBE).strip()
    if not PROBE_PATTERN.fullmatch(out):
        raise WizardError(
            "Could not probe the target Python",
            f"unexpected answer from {python} for architecture and version: {out!r}",
        )
    arch, version = out.split("\n")
    return arch, version


def check_ssh(
    ui: Ui, platform: Platform, remote: Remote, need: tuple[int, int] | None
) -> RemotePython:
    """Log in once; check the target interpreter, architecture, Python and free space."""
    ui.active("Connecting over SSH…", f"{remote.target}; ssh may ask for the password")
    remote.connect()
    ui.ok("Connected over SSH", remote.target)
    py = RemotePython(remote, platform.python)
    if not py.exists():
        raise WizardError(
            "Target interpreter not found",
            f"{platform.python} does not exist: is this a {platform.display_name}?",
        )
    arch, version = probe_platform(remote, platform.python)
    if arch != platform.arch:
        raise WizardError(
            "Wrong robot architecture", f"the robot is {arch}, the app needs {platform.arch}"
        )
    if need and tuple(map(int, version.split("."))) < need:
        raise WizardError(
            "Target Python too old", f"Python {version}; the app needs {need[0]}.{need[1]}+"
        )
    ui.ok("Checked the interpreter", f"{arch}, Python {version}, {platform.python}")
    if need:
        for path in platform.space_paths:
            free_kb = int(remote.run(f"df -Pk {shlex.quote(path)} | tail -1").split()[3])
            if free_kb < MIN_FREE_MB * 1024:
                raise WizardError(
                    f"Not enough free space on {path}",
                    f"{free_kb // 1024} MB free, {MIN_FREE_MB} MB needed",
                )
            ui.ok("Checked free space", f"{free_kb // 1024} MB on {path}")
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
    user = opts.user or ui.ask("SSH user", platform.default_user, flag="--user")
    # The ssh control socket path must stay under 104 bytes: macOS $TMPDIR is too long.
    with tempfile.TemporaryDirectory(prefix="re-ssh-", dir="/tmp") as sock_dir:
        remote = Remote(user, host, Path(sock_dir))
        try:
            body(remote)
        finally:
            remote.close()


# ---------------------------------------------------------------- data folder


def remote_data_dir(remote: Remote) -> tuple[str, str]:
    """The robot user's home and the app data folder below it (``roboemot.paths`` default)."""
    home = posixpath.normpath(remote.run('printf %s "$HOME"').strip())
    if not posixpath.isabs(home) or home == "/":
        raise WizardError(
            "Could not find the robot user's home folder", f"$HOME is {home!r} on the robot"
        )
    return home, posixpath.join(home, paths.DEFAULT_REL.as_posix())


def _exists(remote: Remote, path: str) -> bool:
    return remote.run(f"test -e {shlex.quote(path)} && echo yes || echo no").strip() == "yes"


def claim_data_dir(
    ui: Ui, platform: Platform, remote: Remote, record: InstallRecord, dry_run: bool
) -> None:
    """Record and create the private app data folder, unless it is already there.

    A folder recorded by an earlier install stays recorded. One that exists but is not in the
    record predates the installer (or an earlier run without a record) and is left alone. The
    record is written before the folder is created, so an interrupted install can be undone.
    The dry run only reports.
    """
    home, path = remote_data_dir(remote)
    if record.data_dir:
        ui.ok("Kept the data folder", f"{record.data_dir}, already recorded")
        return
    if _exists(remote, path):
        ui.warn("Data folder not recorded", f"{path} was already there; uninstall leaves it")
        return
    parents = []
    parent = posixpath.dirname(path)
    while parent != home and not _exists(remote, parent):
        parents.append(parent)
        parent = posixpath.dirname(parent)
    parents.reverse()
    if dry_run:
        ui.would("Create the data folder", f"{path} (0700), recorded for uninstall")
        return
    record.data_dir, record.data_dir_parents = path, parents
    write_record(remote, platform.record_path, record)
    remote.run(f"mkdir -p -m 700 {shlex.quote(path)}")
    ui.ok("Created the data folder", f"{path} (registry, models, trait cache)")


# ---------------------------------------------------------------- Piper virtualenv


def piper_paths(remote: Remote) -> tuple[str, str]:
    """The Piper virtualenv and its interpreter (``roboemot.paths.piper_python`` on the robot)."""
    _, data = remote_data_dir(remote)
    venv = posixpath.join(data, paths.PIPER_VENV_SUBDIR)
    return venv, posixpath.join(venv, "bin", "python")


def install_piper(
    ui: Ui, platform: Platform, remote: Remote, record: InstallRecord, dry_run: bool
) -> None:
    """Create the Piper virtualenv in the data folder and install the pinned ``piper-tts``.

    piper-tts is GPL-3.0, so it lives in its own environment and runs as its own process (SPEC
    8). The venv is recorded before it is created, so an interrupted install can be undone; a
    folder that exists but is not in the record is not ours and is left alone. Every pip call
    passes ``--no-cache-dir``; the install reads the hash-pinned lock for the robot's architecture
    and Python (``piper_lock``) with ``--require-hashes``; the dry run only reports.
    """
    venv, python = piper_paths(remote)
    recreate = True
    if record.piper_venv == venv and _exists(remote, python):
        probe = remote.run(f"{shlex.quote(python)} -I -c 'import piper' && echo yes || echo no")
        if probe.strip().endswith("yes"):
            ui.ok("Piper already installed", venv)
            return
        ui.warn("Piper venv is broken", f"{venv} cannot import piper; installing it again")
        recreate = False
    if record.piper_venv != venv and _exists(remote, venv):
        ui.warn("Piper folder not recorded", f"{venv} was already there; left alone")
        return
    lock = lock_path(*probe_platform(remote, platform.python))
    if dry_run:
        ui.would(
            "Create the Piper venv",
            f"{venv} with piper-tts=={PIPER_TTS_VERSION}, hashes from {lock.name}",
        )
        return
    record.piper_venv = venv
    write_record(remote, platform.record_path, record)
    tmp = remote.run(f"mktemp -d {REMOTE_TMP_PREFIX}.XXXXXX")
    try:
        remote.copy([lock], tmp)
        if recreate:
            remote.run(f"mkdir -p -m 700 {shlex.quote(posixpath.dirname(venv))}")
            remote.run(f"{shlex.quote(platform.python)} -m venv {shlex.quote(venv)}")
            ui.ok("Created the Piper venv", venv)
        with ui.running("Installing piper-tts…"):
            result = RemotePython(remote, python).install_locked(f"{tmp}/{lock.name}")
    finally:
        remote.run(f"rm -rf {shlex.quote(tmp)}", check=False)
    if not result.ok:
        raise WizardError(
            "Could not install piper-tts on the robot",
            f"pip install piper-tts failed: {last_line(result.output)}",
            output=result.output,
        )
    ui.ok("Installed piper-tts", PIPER_TTS_VERSION)
    ui.hint("Voice files, on the robot", "roboemot models fetch --voice")


def remove_piper_venv(ui: Ui, remote: Remote, record: InstallRecord, dry_run: bool) -> None:
    """Delete the recorded Piper virtualenv, only at exactly the path this installer would pick."""
    if record.piper_venv is None:
        return
    venv, _ = piper_paths(remote)
    if record.piper_venv != venv:
        ui.warn("Piper venv left", f"recorded {record.piper_venv!r} is not {venv}")
        return
    if not _exists(remote, venv):
        ui.ok("Piper venv gone", f"{venv} is not there")
        return
    real = remote.run(f"realpath {shlex.quote(venv)} 2>/dev/null || true").strip()
    if real != venv:
        ui.warn("Piper venv left", f"{venv} resolves to {real or 'nothing'}: symlink?")
        return
    if not dry_run:
        remote.run(f"rm -rf {shlex.quote(venv)}")
    ui.change(dry_run, "Delete the Piper venv", "Deleted the Piper venv", venv)


def ask_new_pin(ui: Ui) -> str | None:
    """Ask twice for the operator PIN on this PC (hidden); None when skipped or not interactive."""
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.webui.auth import PinError, check_pin_format

    rules = RoboEmotConfig().webui
    for _ in range(PIN_RETRIES):
        pin = ui.ask_secret(f"Operator PIN for the web UI ({rules.pin_min_length}+ digits)")
        if pin is None:
            return None
        try:
            check_pin_format(pin, rules)
        except PinError as e:
            ui.warn("PIN rejected", str(e))
            continue
        if ui.ask_secret("Repeat the PIN") == pin:
            return pin
        ui.warn("PIN not confirmed", "the two entries differ")
    return None


def set_operator_pin(ui: Ui, py: RemotePython, remote: Remote, dry_run: bool) -> None:
    """Set the web UI operator PIN in the app data folder, piping it to the CLI over SSH stdin.

    Runs after the data folder exists. The PIN is typed on the PC, travels only on the SSH
    channel's standard input and appears in no command line, log or message; uninstall removes
    it with the data folder. A dry run only says it would.
    """
    _, folder = remote_data_dir(remote)
    path = posixpath.join(folder, PIN_FILE)
    exists = _exists(remote, path)
    if dry_run:
        ui.would("Replace the operator PIN" if exists else "Ask for the operator PIN", path)
        return
    if exists and not ui.interactive:
        ui.ok("Kept the operator PIN", f"{path}; no terminal to type a new one")
        return
    if exists and not ui.confirm("An operator PIN is already set. Replace it?", False):
        ui.ok("Kept the operator PIN", path)
        return
    pin = ask_new_pin(ui)
    command = f"{shlex.quote(py.python)} -m roboemot.webui set-pin --data-dir {shlex.quote(folder)}"
    if pin is None:
        ui.warn("Operator PIN not set", "the web UI lets nobody in until it is")
        ui.hint("Set it on the robot", command)
        return
    remote.run(command, stdin=pin + "\n")
    ui.ok("Set the operator PIN", path)


def robot_cli(py: RemotePython, folder: str, *args: str) -> str:
    """The ``roboemot secrets`` command line on the robot, for the folder ``folder``."""
    action, names = args[0], args[1:]
    tail = " ".join(["--", *map(shlex.quote, names)]) if names else ""
    return (
        f"{shlex.quote(py.python)} -m roboemot.cli secrets {action} "
        f"--data-dir {shlex.quote(folder)} {tail}"
    ).rstrip()


def ask_secret_value(ui: Ui, name: str) -> str | None:
    """Ask twice for the value of secret ``name`` on this PC (hidden); None when skipped, not
    interactive or the entries differ."""
    value = ui.ask_secret(f"Value of the secret {name!r}")
    if not value:
        return None
    if ui.ask_secret("Repeat the value") == value:
        return value
    ui.warn("Secret not confirmed", "the two entries differ")
    return None


def check_secret_names(names: list[str]) -> None:
    """Reject a name that is not a valid secret name, before anything runs on the robot."""
    for name in names:
        if not SECRET_NAME.match(name):
            raise WizardError(
                f"Invalid secret name {name!r}", "use letters, numbers, '_', '.' or '-'"
            )


def wanted_secrets(ui: Ui, opts: Options) -> list[str]:
    """Names to store: from ``--secret``, else (interactive, not ``-y``) ask once for a cloud
    route key. Names that are not valid are rejected before anything runs."""
    names = opts.secrets
    if names is None and ui.interactive and not ui.assume_yes:
        if not ui.confirm("Store an API key for a cloud LLM route on the robot?", False):
            return []
        names = [
            ui.ask(
                "Secret name (the ${secret:name} in the route)",
                DEFAULT_SECRET_NAME,
                flag="--secret",
            )
        ]
    check_secret_names(names or [])
    return names or []


def set_secrets(
    ui: Ui,
    platform: Platform,
    py: RemotePython,
    remote: Remote,
    record: InstallRecord,
    names: list[str],
    dry_run: bool,
) -> None:
    """Store the secrets ``names`` in the robot's secret store (SPEC 8, 12 Data).

    Each value is typed on this PC and travels only on the SSH channel's standard input, so it
    is in no command line, log or message. A name that the store did not have is recorded (by
    name) before it is written, so uninstall removes exactly it; one that already existed is
    replaced only on request and stays the robot owner's. The dry run only says what it would do
    and runs nothing on the robot.
    """
    _, folder = remote_data_dir(remote)
    for name in names:
        if dry_run:
            ui.would("Ask for a secret", f"{name!r}, stored in {folder}/secrets.json")
            continue
        have = remote.run(robot_cli(py, folder, "list")).split()
        if name in have and not ui.interactive:
            ui.ok("Kept the secret", f"{name}; no terminal to type a new value")
            continue
        if name in have and not ui.confirm(f"Secret {name!r} is already set. Replace it?", False):
            ui.ok("Kept the secret", name)
            continue
        value = ask_secret_value(ui, name)
        if value is None:
            ui.warn("Secret not set", name)
            ui.hint("Set it on the robot", robot_cli(py, folder, "set", name))
            continue
        if name not in have and name not in record.secrets:
            record.secrets.append(name)
            write_record(remote, platform.record_path, record)
        remote.run(robot_cli(py, folder, "set", name), stdin=value + "\n")
        ui.ok("Stored the secret", name)


def remove_secrets(
    ui: Ui,
    platform: Platform,
    py: RemotePython,
    remote: Remote,
    record: InstallRecord,
    dry_run: bool,
) -> None:
    """Remove the secrets this installer added, by recorded name; others stay. Runs while the
    app's Python is still installed. A name that is gone already is fine."""
    if not record.secrets:
        return
    _, folder = remote_data_dir(remote)
    names = [n for n in record.secrets if SECRET_NAME.match(n)]  # the record came from the robot
    if dry_run:
        ui.would("Remove the secrets", ", ".join(names))
        return
    remote.run(robot_cli(py, folder, "remove", *names))
    record.secrets = []
    write_record(remote, platform.record_path, record)
    ui.ok("Removed the secrets", ", ".join(names))


def checked_data_dir(remote: Remote, record: InstallRecord) -> tuple[str | None, str | None]:
    """The recorded data folder if it is safe to delete, else a reason.

    The record is read from the robot and the folder is deleted recursively, so only exactly
    the path this installer would choose now is accepted (``$HOME`` of the SSH user plus the
    default location), and when it exists its ``realpath`` must be the same path, so no
    symlinked parent or foreign home redirects the delete. Returns ``(path, None)``, or
    ``(None, None)`` when nothing is recorded, or ``(None, reason)``.
    """
    if record.data_dir is None:
        return None, None
    _, expected = remote_data_dir(remote)
    if record.data_dir != expected:
        return None, f"recorded data folder {record.data_dir!r} is not {expected}"
    if _exists(remote, expected):
        real = remote.run(f"realpath {shlex.quote(expected)} 2>/dev/null || true").strip()
        if real != expected:
            return None, f"data folder {expected} resolves to {real or 'nothing'}: symlink?"
    return expected, None


def _created_parents(record: InstallRecord, home: str, path: str) -> list[str]:
    """The recorded parents that really lie between ``home`` and ``path``, innermost first."""
    between = set()
    parent = posixpath.dirname(path)
    while parent != home and parent.startswith(home + "/"):
        between.add(parent)
        parent = posixpath.dirname(parent)
    return [p for p in reversed(record.data_dir_parents) if p in between]


def remove_data_dir(ui: Ui, remote: Remote, record: InstallRecord, dry_run: bool) -> None:
    """Delete the recorded data folder (registry, models and trait cache included), then the
    parents the installer created if they are empty. Without a record nothing is deleted; a
    record that fails ``checked_data_dir`` is reported and left (``verify_uninstall`` then
    fails)."""
    if record.data_dir is None:
        _, path = remote_data_dir(remote)
        if _exists(remote, path):
            ui.warn("Data folder not recorded", f"{path} left as it is; delete it by hand")
        return
    path, problem = checked_data_dir(remote, record)
    if path is None:
        ui.warn("Data folder left as it is", problem or "")
        return
    if _exists(remote, path):
        if not dry_run:
            remote.run(f"rm -rf {shlex.quote(path)}")
        detail = f"{path} (registry, models, trait cache)"
        ui.change(dry_run, "Delete the data folder", "Deleted the data folder", detail)
    else:
        ui.ok("Data folder gone", f"{path} is not there")
    home, _ = remote_data_dir(remote)
    for parent in _created_parents(record, home, path):
        if not dry_run:
            remote.run(f"rmdir {shlex.quote(parent)} 2>/dev/null || true", check=False)


def warn_data_dir_overrides(ui: Ui, remote: Remote, dirs: list[Path]) -> None:
    """Warn when the app would keep its data outside the folder the installer records.

    Looks at ``ROBOEMOT_DATA_DIR`` in the SSH user's environment and at a ``data_dir`` in the
    staged ``app.yaml``. The app's own environment may differ, so this is a hint, not a check.
    """
    env = remote.run('printf %s "${ROBOEMOT_DATA_DIR:-}"').strip()
    if env:
        ui.warn("Data may land elsewhere", f"ROBOEMOT_DATA_DIR={env} is set for the SSH user")
    for root in dirs:
        for cfg in root.rglob("app.yaml"):
            try:
                value = (yaml.safe_load(cfg.read_text()) or {}).get("data_dir")
            except (OSError, yaml.YAMLError, AttributeError):
                continue
            if value:
                ui.warn(
                    "Data may land elsewhere",
                    f"{cfg.name} sets data_dir={value}; uninstall leaves that folder",
                )


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
    warn_data_dir_overrides(ui, remote, dirs)
    before = py.installed()
    record = read_record(remote, platform.record_path) or InstallRecord(
        platform.app_name, platform.manifest_id, own
    )
    if record.pending_before is not None:
        ui.warn("Earlier install interrupted", "its changes are folded into the record")
        record.merge(record.pending_before, before)
    record.packages = sorted(set(record.packages) | set(own))
    pins = py.constraints(before, set(own))
    if not dry_run and not ui.interactive and not ui.assume_yes:
        # Read-only, before anything is copied or recorded: a shared-package change needs a
        # yes that nobody can type, so stop now rather than after the wheels are on the robot.
        with ui.running("Resolving dependencies…"):
            plan = py.plan_install(third_party_requirements(dirs), pins)
        if not plan.ok:
            ui.require("--yes", False, "Shared packages must change, and only --yes allows it")
    if dry_run:
        reqs = third_party_requirements(dirs)
        with ui.running("Resolving dependencies…"):
            plan = py.plan_install(reqs, pins)
            free = None if plan.ok else py.plan_install(reqs, None)
        if free is None:
            added, _ = report_changes(_json(plan.output), before)
            if added:
                ui.would("Add dependencies", _versions(added))
            else:
                ui.ok("Checked dependencies", "already satisfied")
        else:
            if not free.ok:
                raise _resolve_error(free.output)
            added, changed = report_changes(_json(free.output), before)
            ui.warn("Shared packages must change", f"{_versions(changed)} (asks first)")
            if added:
                ui.would("Add dependencies", _versions(added))
        claim_data_dir(ui, platform, remote, record, dry_run=True)
        ui.would("Record the changes", f"{platform.record_path}; nothing copied or installed")
        return record
    tmp = remote.run(f"mktemp -d {REMOTE_TMP_PREFIX}.XXXXXX")
    try:
        remote.copy(wheels, tmp)
        ui.ok("Copied the wheels", ", ".join(w.name for w in wheels))
        files = [f"{tmp}/{w.name}" for w in wheels]
        record.pending_before = before
        write_record(remote, platform.record_path, record)
        ui.info("The first install builds numpy and scipy and may take minutes.")
        with ui.running("Installing the app…"):
            result = py.install(files, pins)
        if not result.ok:
            ui.warn("Shared packages conflict", "the app does not fit the versions on the robot")
            with ui.running("Resolving dependencies…"):
                free = py.plan_install(files, None)
            if not free.ok:
                raise _resolve_error(free.output)
            _, changed = report_changes(_json(free.output), before)
            for n, c in changed.items():
                ui.kv(n, c)
            question = "Change these shared packages? Uninstall restores them (needs internet)"
            if not ui.confirm(question, False):
                raise WizardError("Install cancelled", "shared packages left as they were")
            with ui.running("Installing the app…"):
                result = py.install(files, None)
            if not result.ok:
                raise WizardError(
                    "Could not install the app on the robot",
                    f"pip install failed: {last_line(result.output)}",
                    output=result.output,
                )
        with ui.running("Reinstalling own packages…"):
            py.reinstall_own(files)
        after = py.installed()
        record.merge(before, after)
        claim_data_dir(ui, platform, remote, record, dry_run=False)
        write_record(remote, platform.record_path, record)
        ui.ok("Installed the app", f"dependencies added: {', '.join(record.added) or 'none'}")
        ui.ok("Recorded the changes", platform.record_path)
    finally:
        remote.run(f"rm -rf {shlex.quote(tmp)}", check=False)
    return record


def preflight(opts: Options, ui: Ui) -> None:
    """Without a terminal, stop before anything runs unless every question has its flag.

    Host and user are never defaulted; the confirmations are answered by their own flag or
    by ``--yes``. Questions that depend on what the robot says (shared packages, a running
    app, an existing PIN or secret) are checked when they come up and name ``--yes``. An empty
    value (``--user ""``) counts as missing.
    """
    ui.require("--host", bool(opts.host))
    if len(installed_platforms()) > 1:
        ui.require("--robot", bool(opts.robot))
    ui.require("--user", bool(opts.user))
    if opts.uninstall:
        return
    answered = [("--dev", opts.dev), ("--piper", opts.piper)]
    if not opts.dry_run:
        answered += [("--start", opts.start), ("--startup-app", opts.startup)]
    for flag, value in answered:
        given = value is not None or opts.yes or ui.assume_yes
        ui.require(flag, given, f"Missing {flag} or --no-{flag[2:]}")


def _versions(items: dict[str, str]) -> str:
    return ", ".join(f"{n} {v}" for n, v in sorted(items.items()))


def _resolve_error(output: str) -> WizardError:
    return WizardError(
        "Could not resolve the dependencies on the robot", last_line(output), output=output
    )


def _rerun(opts: Options) -> str:
    """The installer command line for ``opts`` without ``--dry-run``: every flag that was set,
    so the dry run's ``Next:`` line runs the same choices for real."""
    args = []
    for flag, value in (("--host", opts.host), ("--user", opts.user), ("--robot", opts.robot)):
        if value:
            args += [flag, value]
    for flag, value in (
        ("dev", opts.dev),
        ("piper", opts.piper),
        ("start", opts.start),
        ("startup-app", opts.startup),
    ):
        if value is not None:
            args.append(f"--{flag}" if value else f"--no-{flag}")
    for name in opts.secrets or []:
        args += ["--secret", name]
    args += ["--uninstall"] * opts.uninstall + ["--yes"] * opts.yes
    return shlex.join([*COMMAND.split(), *args])


def _check_running_app(opts: Options, ui: Ui, platform: Platform) -> None:
    """Off a terminal, ask the read-only question "is another app running?" before anything
    is installed: starting would have to stop it, and only ``--yes`` can say so."""
    if opts.dry_run or not opts.start or ui.assume_yes or ui.interactive:
        return
    other = platform.running_app()
    if other and other != platform.app_name:
        ui.require("--yes", False, f"{other} is running and --start would stop it")


def run_install(opts: Options, ui: Ui, root: Path) -> None:
    ui.steps = INSTALL_STEPS
    check_secret_names(opts.secrets or [])
    preflight(opts, ui)
    ui.title("installer", "dry run, the robot is not modified" if opts.dry_run else "")
    ui.step("Robot")
    host, platform = select_robot(ui, opts, root)
    ui.ok("Selected the robot", f"{platform.display_name} at {host}")
    platform.connect(ui, host)
    _check_running_app(opts, ui, platform)

    def body(remote: Remote) -> None:
        with tempfile.TemporaryDirectory(prefix="roboemot-") as tmp:
            ui.step("Configuration")
            dev = opts.dev
            if dev is None:
                ui.info(
                    "The manifest dynamics are not measured yet. Dev mode lets the app run with "
                    "conservative fallback limits (SPEC 12); without it, the app refuses to start."
                )
                dev = ui.confirm("Install in dev mode?", False, flag="--dev")
                opts.dev = dev  # the dry run's Next: line repeats the answer
            if dev:
                ui.ok("Chose dev mode", "configs/dev-unmeasured.yaml")
            else:
                ui.ok("Chose the production config")
            dirs = platform.stage(root, Path(tmp) / "src", dev)

            ui.step("SSH access")
            py = check_ssh(ui, platform, remote, min_python(dirs))

            ui.step("Build")
            with ui.running("Building the wheels…"):
                wheels = build_wheels(dirs, Path(tmp) / "wheels")
            ui.ok("Built the wheels", ", ".join(w.name for w in wheels))

            ui.step("Install (dependency check only)" if opts.dry_run else "Install")
            record = install_packages(ui, platform, py, remote, dirs, wheels, opts.dry_run)
            if not opts.dry_run:
                platform.verify(ui, py, dev)
            if opts.piper is None:
                opts.piper = ui.confirm(
                    "Install Piper for on-board speech (own virtualenv)?", False, flag="--piper"
                )
            if opts.piper:
                install_piper(ui, platform, remote, record, opts.dry_run)
            set_operator_pin(ui, py, remote, opts.dry_run)
            names = wanted_secrets(ui, opts)
            set_secrets(ui, platform, py, remote, record, names, opts.dry_run)

            ui.step("Start")
            if opts.dry_run:
                running = platform.running_app()
                if running:
                    ui.would(f"Stop {running}", "only one app runs at a time")
                ui.would(f"Start {platform.app_name}", "after waking the robot")
                ui.kv("Autostart app", platform.autostart() or "none")
                ui.ok("Dry run passed", "the robot was not modified")
                ui.next(_rerun(opts))
                ui.summary()
                return
            if (
                opts.start
                if opts.start is not None
                else ui.confirm(f"Start {platform.app_name} now?", False, flag="--start")
            ):
                platform.start(ui)
            startup = opts.startup
            if startup is None:
                startup = ui.confirm(
                    f"Start {platform.app_name} automatically when the robot wakes up?",
                    False,
                    flag="--startup-app",
                )
            if startup:
                if not record.autostart_recorded:
                    record.autostart_before, record.autostart_recorded = platform.autostart(), True
                    write_record(remote, platform.record_path, record)
                platform.set_autostart(platform.app_name)
                ui.ok("Set the autostart app", f"{platform.app_name} starts on wake-up")
            ui.ok("Installed", f"{platform.app_name} on {host}")
            ui.next(shlex.join(["python", "scripts/robot_ssh.py", "--host", host, "--debug"]))
            ui.hint("Undo", _rerun(Options(host=host, user=opts.user, uninstall=True)))
            ui.summary()

    _with_remote(opts, ui, platform, host, body)


# ---------------------------------------------------------------- uninstall


def run_uninstall(opts: Options, ui: Ui, root: Path) -> None:
    ui.steps = UNINSTALL_STEPS
    preflight(opts, ui)
    ui.title("uninstaller", "dry run, the robot is not modified" if opts.dry_run else "")
    ui.step("Robot")
    host, platform = select_robot(ui, opts, root)
    ui.ok("Selected the robot", f"{platform.display_name} at {host}")
    platform.connect(ui, host)
    dry = opts.dry_run

    def body(remote: Remote) -> None:
        ui.step("SSH access")
        py = check_ssh(ui, platform, remote, None)

        ui.step("Install record")
        installed = py.installed()
        record = read_record(remote, platform.record_path)
        if record is None:
            ui.warn("No install record", f"{platform.record_path} is not there")
            ui.info(
                "Only the app's own packages are removed: dependencies an earlier install "
                "added cannot be told apart."
            )
            record = InstallRecord(
                platform.app_name,
                platform.manifest_id,
                [canonical(d) for d in platform.distributions],
            )
        else:
            added = ", ".join(record.added) or "nothing"
            replaced = ", ".join(record.replaced) or "nothing"
            ui.ok("Read the install record", f"added {added}; replaced {replaced}")
            if record.pending_before is not None:
                ui.warn("Last install interrupted", "its changes are included")
                record.merge(record.pending_before, installed)

        ui.step("App")
        app = platform.app_name
        if platform.running_app() == app:
            if not dry:
                platform.stop_app()
            ui.change(dry, f"Stop {app}", f"Stopped {app}")
        else:
            ui.ok("App not running", app)
        current = platform.autostart()
        if current == app:
            if not dry:
                platform.set_autostart(record.autostart_before)
            before = record.autostart_before or "none"
            ui.change(dry, "Restore the autostart app", "Restored the autostart app", before)
        else:
            ui.ok("Kept the autostart app", f"{current or 'none'}, not {app}")

        remove_secrets(ui, platform, py, remote, record, dry)

        ui.step("Packages")
        plan = plan_uninstall(record, installed, py.requires())
        for name, users in plan.kept.items():
            ui.warn("Kept a shared package", f"{name}: needed by {', '.join(users)}")
        for name, version in plan.drifted.items():
            ui.warn("Kept a changed package", f"{name} is now {version}, changed since the install")
        if plan.remove:
            removed = ", ".join(f"{n} {installed[n]}" for n in plan.remove)
            ui.change(dry, "Remove packages", "Removed packages", removed)
        if plan.restore:
            restored = f"{_versions(plan.restore)} (needs internet)"
            ui.change(dry, "Reinstall packages", "Reinstalled packages", restored)
        if plan.empty:
            ui.ok("Packages unchanged", "nothing to remove or restore")
        if not dry:
            with ui.running("Removing packages…"):
                py.uninstall(plan.remove)
                py.install_exact(plan.restore)
        if not (dry and plan.remove):  # before removal, the modules are expected
            for module, path in leftovers(py, platform):
                if dry:
                    ui.would("Delete a leftover module", f"{path} ({module})")
                else:
                    remote.run(f"rm -rf {shlex.quote(path)}")
                    ui.warn("Deleted a leftover module", f"{path} ({module})")

        ui.step("Clean up and verify")
        remove_piper_venv(ui, remote, record, dry)
        remove_data_dir(ui, remote, record, dry)
        platform.cleanup(ui, py, dry)
        listed = remote.run(f"ls -d {REMOTE_TMP_PREFIX}.* 2>/dev/null || true").split()
        tmp_dirs = [d for d in listed if TMP_DIR_PATTERN.fullmatch(d)]
        for d in tmp_dirs:
            if not dry:
                remote.run(f"rm -rf {shlex.quote(d)}")
            ui.change(dry, "Delete a temporary folder", "Deleted a temporary folder", d)
        if dry:
            ui.would("Delete the install record", f"{platform.record_path}, once verified")
            ui.ok("Dry run passed", "the robot was not modified")
            ui.next(_rerun(opts))
            ui.summary()
            return
        verify_uninstall(ui, platform, py, record, plan.restore)
        remote.run(f"rm -f {shlex.quote(platform.record_path)}")
        ui.ok("Deleted the install record", platform.record_path)
        ui.ok("Uninstalled", f"{app} from {host}")
        ui.summary()

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
    venv, _ = piper_paths(py.remote)
    if record.piper_venv == venv and _exists(py.remote, venv):
        problems.append(f"Piper virtualenv {venv} still present")
    data_dir, bad = checked_data_dir(py.remote, record)
    if bad:
        problems.append(bad)
    elif data_dir and _exists(py.remote, data_dir):
        problems.append(f"data folder {data_dir} still present")
    if problems:
        raise WizardError(
            "Uninstall incomplete, record kept",
            "; ".join(problems),
            f"{COMMAND} --uninstall",
        )
    ui.ok(
        "Verified the removal",
        "app packages gone, original versions back" + (", data folder gone" if data_dir else ""),
    )
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
        raise WizardError("No report from pip", last_line(output), output=output)
    return json.JSONDecoder().raw_decode(output[start:])[0]


# ---------------------------------------------------------------- command line


def parse_args(argv: list[str] | None = None, plain: bool = False) -> Options:
    p = HeaderParser(description="Install or uninstall RoboEmot on a robot (SPEC 4A).")
    p.plain = plain
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
    p.add_argument(
        "--piper",
        action=argparse.BooleanOptionalAction,
        help="install Piper (on-board speech) in its own virtualenv in the data folder",
    )
    p.add_argument(
        "--secret",
        action="append",
        metavar="NAME",
        help="store the secret NAME (an API key for ${secret:NAME}) on the robot; the value is "
        "asked on this PC without echo and never appears on a command line",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="answer yes to every confirmation: dev mode, Piper, start, autostart, stopping "
        "another app, changing shared packages, replacing a PIN or secret",
    )
    p.add_argument(
        "--debug", action="store_true", help="show raw tool output and tracebacks on errors"
    )
    p.add_argument("--version", action="store_true", help="print the version and exit")
    a = p.parse_args(argv)
    return Options(
        a.host,
        a.user,
        a.robot,
        a.dev,
        a.start,
        a.startup,
        a.dry_run,
        a.yes,
        a.uninstall,
        a.piper,
        a.secret,
        a.debug,
        a.version,
    )


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    """Run the installer; 0 on success, 1 on an error block, 130 on Ctrl-C.

    Errors are the Term three-line block on stderr. An unexpected exception is one too, whose
    ``Try:`` line names ``--debug``; with ``--debug`` the traceback and raw tool output print.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    vt = term is not None and all([term.enable_vt(sys.stdout), term.enable_vt(sys.stderr)])
    opts = parse_args(args, plain=not vt)
    if term is None:
        print(
            "The installer needs sondalab-palette: "
            'pip install -e "packages/roboemot-core[install]"',
            file=sys.stderr,
        )
        return 1
    if opts.version:
        mode = term.color_mode(isatty=sys.stdout.isatty()) if vt else "none"
        print(term.lockup("roboemot", app_version(), mode))
        return 0
    print_header(plain=not vt)
    ui = Ui(
        interactive=sys.stdin.isatty() and not os.environ.get("CI"),
        assume_yes=opts.yes,
        command=shlex.join(["python", "scripts/install_on_robot.py", *args]),
        vt=vt,
        debug=opts.debug,
    )
    root = root or Path.cwd()
    try:
        (run_uninstall if opts.uninstall else run_install)(opts, ui, root)
    except TermError as e:
        ui.error(e)
        return 1
    except KeyboardInterrupt:
        ui.error(WizardError("Interrupted", "stopped with Ctrl-C"))
        return 130
    except Exception as e:
        if opts.debug:
            traceback.print_exc()
        fix = "" if opts.debug else f"{ui.command} --debug"
        ui.error(WizardError(f"Unexpected error: {type(e).__name__}", str(e), fix))
        return 1
    return 0


__all__ = [
    "ENTRY_POINT_GROUP",
    "Options",
    "Platform",
    "Ui",
    "WizardError",
    "load_platform",
    "main",
    "run_install",
    "run_uninstall",
]
