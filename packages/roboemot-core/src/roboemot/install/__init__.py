"""Robot installer: install, verify, start and cleanly uninstall the app (SPEC 4A, Packaging).

Runs on a PC on the robot's LAN, never on the robot. The flow is robot-agnostic; everything
that depends on the robot model sits behind a :class:`Platform`, registered by the platform
package in the ``roboemot.installers`` entry-point group under the manifest's ``robot.id``
(for example ``roboemot_reachymini.install``).

Install: find the robot (mDNS, every manifest that declares discovery), check SSH, the target
interpreter, architecture, Python and free space, build the wheels locally, install them with
the versions already on the robot pinned, record exactly what changed, verify, optionally
start the app and make it the autostart app. If the target environment does not exist yet and
the platform can create it (``Platform.base_python``), the checks run against the creating
interpreter, the record is written, and then the platform creates the environment; the dry run
only reports it and skips the dependency check. It also creates the private app data folder
(``~/.local/share/roboemot``: registry, models, trait cache) when it is not there yet and
records it. It writes the app config (``data/app.yaml``) with ``voice.enabled: true`` (plus the
dev content with ``--dev``) unless ``--no-voice``, which also makes Piper and the models default
to off. Unless declined (`--no-piper`) it also creates a Piper virtualenv in the data folder
(`piper-tts` is GPL-3.0 and runs as its own process, SPEC 8), recorded for uninstall.
The platform's ``extras`` (``voice`` for a robot that listens, so vosk) are installed with
``roboemot-core``; the ``vision`` extra and the vision weights come too unless declined
(``--no-vision``, SPEC 4A Installer). The voice models (``roboemot models fetch --voice``) are
downloaded on the robot after Piper unless declined (``--no-models``), when the manifests'
sizes fit the free space. Before installing it offers to clear the platform's
``clearable_caches`` (``--clear-caches``), which uninstall cannot restore.
Optionally (``--secret NAME``, or asked on a terminal) it stores an API key in the robot's secret
store (SPEC 8, 12 Data), recorded by name; uninstall removes only the names it added.

Later, ``scripts/run_on_robot.py`` (``roboemot.install.run``) starts the installed app and
prints its web UI address.

Uninstall: stop the app, give back the previous autostart app (only if the setting still names
this app), uninstall what the record says this installer added, reinstall the original versions
it replaced, keep anything another distribution still needs, remove leftovers and the recorded
data folder (registry, models and trait cache included), verify, and only then delete the
record. An environment the record says this installer created is removed too, unless another
app lives in it (``Platform.env_users``); with the environment already gone, the rest of the
cleanup still runs. Running it twice is harmless. Both flows have a read-only ``--dry-run``.
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
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from .. import paths
from ..secret_store import NAME as SECRET_NAME
from .build import (
    build_wheels,
    distribution_names,
    min_python,
    third_party_requirements,
    with_extras,
)
from .caches import clear_caches
from .piper_lock import lock_path
from .pyenv import RemotePython, report_changes
from .record import InstallRecord, UninstallPlan, canonical, plan_uninstall, unknown_keys
from .remote import Remote
from .ui import Ui, WizardError, paint

ENTRY_POINT_GROUP = "roboemot.installers"
# Disk at the install's peak, in MB as (downloads, unpacked), measured from the aarch64 cp312
# wheels pip picks for each set (2026-10-22), unpacked sizes rounded up about 25% for the .pyc
# files pip compiles. The app with its dependencies: scipy, numpy, openai, pydantic (52 MiB
# downloaded, 184 MiB unpacked). Extras: voice is onnxruntime and vosk (24, 66); vision is
# opencv-python-headless and onnx (43, 105) plus onnxruntime (20, 53), counted again so that
# vision alone also fits.
APP_DISK_MB = (60, 230)
EXTRA_DISK_MB = {"voice": (30, 85), "vision": (70, 200)}
ENV_DOWNLOAD_SHARE = 0.5  # downloads next to a created environment's tree (measured 0.3-0.4)
ENV_FALLBACK_MB = 1500  # that tree when `du` on the creating environment answers nothing
REMOTE_TMP_PREFIX = "/tmp/roboemot-install"
TMP_DIR_PATTERN = re.compile(re.escape(REMOTE_TMP_PREFIX) + r"\.[A-Za-z0-9]+")
PIN_FILE = "webui_pin.json"  # roboemot.webui.auth.PIN_FILE, inside the app data folder
PIN_RETRIES = 3
DEFAULT_SECRET_NAME = "cloud_key"  # the name configs/companion.yaml gives the cloud route key
MIB = 1024 * 1024
VISION_DERIVED_BYTES = 170_000_000  # INT8 files made from the FP32 weights ("0.17 GB", models CLI)
PIPER_TTS_VERSION = "1.8.0"  # exact pin: the worker protocol is tested against this API
PLATFORM_PROBE = (
    "import sys, platform; print(platform.machine()); print('%d.%d' % sys.version_info[:2])"
)
PROBE_PATTERN = re.compile(r"[A-Za-z0-9_]+\n\d+\.\d+")
INSTALL_STEPS = 6
UNINSTALL_STEPS = 6


@runtime_checkable
class Platform(Protocol):
    """What the installer needs to know about one robot model."""

    manifest_id: str
    display_name: str
    app_name: str
    default_host: str
    default_user: str
    arch: str  # expected `platform.machine()` of the robot
    python: str  # interpreter of the environment the app is installed into
    env_dir: str  # that environment's folder (created and removed by the installer if missing)
    base_python: str | None  # interpreter that creates the environment; None: cannot create it
    record_path: str  # change record: outside that environment and outside /tmp
    space_paths: tuple[str, ...]  # filesystems the install fills (``install_need_mb``)
    distributions: tuple[str, ...]  # this app's own distributions
    modules: tuple[str, ...]  # their top-level import names (leftover check)
    extras: tuple[str, ...]  # roboemot-core extras the robot needs (installed by default)
    clearable_caches: tuple[str, ...]  # ``~/...`` caches safe to delete; not restored on uninstall

    def connect(self, ui: Ui, host: str) -> None: ...
    def plan_env(self, remote: Remote) -> str: ...
    def create_env(self, ui: Ui, remote: Remote, plan: str) -> None: ...
    def env_users(self, py: RemotePython) -> list[str]: ...
    def env_ready(self, py: RemotePython) -> bool: ...
    def probe(self, ui: Ui, py: RemotePython) -> None: ...
    def stage(
        self, root: Path, dest: Path, dev: bool, app_config: dict[str, object]
    ) -> list[Path]: ...
    def verify(
        self, ui: Ui, py: RemotePython, dev: bool, app_config: dict[str, object]
    ) -> None: ...
    def running_app(self) -> str | None: ...
    def stop_app(self) -> None: ...
    def app_installed(self) -> bool: ...
    def app_url(self, host: str) -> str | None: ...
    def start_blocker(self) -> str | None: ...
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
    clear_caches: bool | None = None
    vision: bool = True
    models: bool | None = None
    voice: bool = True


# ---------------------------------------------------------------- app config


def app_config_overrides(voice: bool) -> dict[str, object]:
    """What the installer sets in the robot's app config besides the platform's dev content:
    ``voice.enabled: true`` with voice (the library default is off, so a robot without a
    microphone or speaker is unchanged; the installer enables it for the robot it sets up).
    The platform writes it into ``app.yaml`` and reads it back to verify (SPEC 4A, Installer)."""
    return {"voice": {"enabled": True}} if voice else {}


def describe_app_config(dev: bool, app_config: dict[str, object]) -> str:
    """One line naming what the app config on the robot will contain (empty: no app.yaml)."""
    parts = [
        f"{key}.{sub}={str(value).lower()}"
        for key, group in app_config.items()
        for sub, value in (group.items() if isinstance(group, dict) else [("", group)])
    ]
    if dev:
        parts.append("safety.allow_unmeasured_dynamics=true (dev)")
    return ", ".join(parts)


def warn_voice_gaps(ui: Ui, opts: Options) -> None:
    """With voice on, say what a declined Piper or voice-models step costs: no speech, no
    listening. Called once both answers are known."""
    if not opts.voice:
        return
    if not opts.piper:
        ui.warn("voice is on but Piper is not installed: the robot will not speak")
    if not opts.models:
        ui.warn("voice is on but the voice models are not downloaded: the robot will not listen")


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


def probe_platform(remote: Remote, python: str) -> tuple[str, str]:
    """The interpreter's CPU architecture and ``"major.minor"``; anything else is an error."""
    out = RemotePython(remote, python).py(PLATFORM_PROBE).strip()
    if not PROBE_PATTERN.fullmatch(out):
        raise WizardError(
            f"unexpected answer probing {python} for architecture and version: {out!r}"
        )
    arch, version = out.split("\n")
    return arch, version


def check_ssh(
    ui: Ui,
    platform: Platform,
    remote: Remote,
    need: tuple[int, int] | None,
    free_space: Callable[[], None] | None = None,
    extras: tuple[str, ...] = (),
) -> RemotePython:
    """Log in once; check the target interpreter, architecture, Python and free space.

    ``free_space`` (stop the running app, clear caches) runs once the interpreter, architecture
    and Python checks passed and before the free-space check, so a refused install has stopped
    nothing and what it frees counts. The space needed comes from what will be installed
    (``install_need_mb``: the app, ``extras`` and, when the environment is created or finished,
    the tree of the creating one).
    """
    ui.info(f"connecting to {remote.target} (ssh may ask for the password)")
    remote.connect()
    ui.ok(f"ssh {remote.target}")
    py = RemotePython(remote, platform.python, ui)
    present = py.exists()
    if not present and need is None:
        ui.ok(f"{platform.env_dir} is not there")
        return py
    if not present and _exists(remote, platform.env_dir):
        raise WizardError(
            f"{platform.env_dir} exists without an interpreter ({platform.python}): "
            "remove it by hand or finish creating it, then run the installer again"
        )
    target = platform.python if present else creating_python(platform, remote)
    arch, version = probe_platform(remote, target)
    if arch != platform.arch:
        raise WizardError(f"robot architecture is {arch}, expected {platform.arch}")
    if need and tuple(map(int, version.split("."))) < need:
        raise WizardError(f"target Python is {version}; the app needs {need[0]}.{need[1]}+")
    ui.ok(f"{arch}, Python {version} at {target}")
    if not present:
        ui.info(f"{platform.env_dir} does not exist yet: it is created with {target}")
    ready = present and platform.env_ready(py)
    if free_space is not None:
        free_space()
    if need:
        downloads, unpacked = install_need_mb(platform, remote, extras, creating=not ready)
        tmp = posixpath.dirname(REMOTE_TMP_PREFIX)
        for space in platform.space_paths:
            path = posixpath.dirname(space) if not present and space == platform.env_dir else space
            want = downloads if space == tmp else downloads + unpacked
            free_kb = int(remote.run(f"df -Pk {shlex.quote(path)} | tail -1").split()[3])
            if free_kb < want * 1024:
                raise WizardError(f"only {free_kb // 1024} MB free on {path}, need {want} MB")
            ui.ok(f"{free_kb // 1024} MB free on {path} (need {want} MB)")
    if ready:
        platform.probe(ui, py)
    elif present:
        ui.warn(f"{platform.env_dir} is incomplete (an earlier install stopped while creating it?)")
    return py


def install_need_mb(
    platform: Platform, remote: Remote, extras: tuple[str, ...], creating: bool
) -> tuple[int, int]:
    """Disk the install needs at its peak, in MB, as (downloads, unpacked): the app and its
    dependencies (``APP_DISK_MB``), each extra (``EXTRA_DISK_MB``) and, when the environment is
    created or finished, the size of the creating interpreter's own environment (``du``) as a
    stand-in for the tree the platform installs into it, plus ``ENV_DOWNLOAD_SHARE`` of it for
    the downloads. Every pip call runs with ``--no-cache-dir``, so the wheels sit in a temporary
    folder until pip is done: downloads and unpacked files both count where pip unpacks."""
    downloads, unpacked = APP_DISK_MB
    for extra in sorted(set(extras)):
        d, u = EXTRA_DISK_MB.get(extra, (0, 0))
        downloads, unpacked = downloads + d, unpacked + u
    if creating and platform.base_python is not None:
        base_env = posixpath.dirname(posixpath.dirname(platform.base_python))
        out = remote.run(f"du -sk {shlex.quote(base_env)} 2>/dev/null | cut -f1 || true").strip()
        tree = int(out) // 1024 if out.isdigit() else ENV_FALLBACK_MB
        downloads += int(tree * ENV_DOWNLOAD_SHARE)
        unpacked += tree
    return downloads, unpacked


def current_python(platform: Platform, remote: Remote) -> str:
    """The target interpreter, or the one that will create it while the dry run has none yet."""
    if RemotePython(remote, platform.python).exists():
        return platform.python
    return creating_python(platform, remote)


def creating_python(platform: Platform, remote: Remote) -> str:
    """The interpreter that would create the missing environment, or a WizardError."""
    base = platform.base_python
    if base is None:
        raise WizardError(
            f"{platform.python} not found and {platform.display_name} cannot create it"
        )
    if not RemotePython(remote, base).exists():
        raise WizardError(
            f"{platform.python} not found, and neither is {base}, the interpreter that creates it"
        )
    return base


def read_record(remote: Remote, path: str, ui: Ui | None = None) -> InstallRecord | None:
    """The change record on the robot, or None. A record from a newer installer raises
    ``WizardError``; keys this installer does not know are dropped with a warning."""
    raw = remote.run(f"cat {shlex.quote(path)} 2>/dev/null || true")
    if not raw:
        return None
    data = json.loads(raw)
    record = InstallRecord.from_json(data)
    ignored = unknown_keys(data)
    if ignored and ui is not None:
        ui.warn(f"install record has fields this installer does not know: {', '.join(ignored)}")
    return record


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


# ---------------------------------------------------------------- data folder


def remote_data_dir(remote: Remote) -> tuple[str, str]:
    """The robot user's home and the app data folder below it (``roboemot.paths`` default)."""
    home = posixpath.normpath(remote.run('printf %s "$HOME"').strip())
    if not posixpath.isabs(home) or home == "/":
        raise WizardError(f"cannot work out the robot user's home folder (got {home!r})")
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
        ui.ok(f"data folder {record.data_dir} already recorded")
        return
    if _exists(remote, path):
        ui.warn(f"data folder {path} already exists: not recorded, uninstall will leave it")
        return
    parents = []
    parent = posixpath.dirname(path)
    while parent != home and not _exists(remote, parent):
        parents.append(parent)
        parent = posixpath.dirname(parent)
    parents.reverse()
    if dry_run:
        ui.info(f"would create the data folder {path} (0700) and record it for uninstall")
        return
    record.data_dir, record.data_dir_parents = path, parents
    write_record(remote, platform.record_path, record)
    remote.run(f"mkdir -p -m 700 {shlex.quote(path)}")
    ui.ok(f"created the data folder {path} (registry, models and trait cache live here)")


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
            ui.ok(f"Piper virtualenv {venv} already installed")
            return
        ui.warn(f"Piper virtualenv {venv} cannot import piper: installing piper-tts again")
        recreate = False
    if record.piper_venv != venv and _exists(remote, venv):
        ui.warn(f"{venv} exists but is not in the install record: left alone")
        return
    lock = lock_path(*probe_platform(remote, current_python(platform, remote)))
    if dry_run:
        ui.info(
            f"would create the Piper virtualenv {venv} and install piper-tts=={PIPER_TTS_VERSION} "
            f"with hashes from {lock.name} (recorded for uninstall)"
        )
        return
    record.piper_venv = venv
    write_record(remote, platform.record_path, record)
    tmp = remote.run(f"mktemp -d {REMOTE_TMP_PREFIX}.XXXXXX")
    try:
        remote.copy([lock], tmp)
        if recreate:
            remote.run(f"mkdir -p -m 700 {shlex.quote(posixpath.dirname(venv))}")
            with ui.progress(f"creating the Piper virtualenv {venv}", report=False):
                remote.run(f"{shlex.quote(platform.python)} -m venv {shlex.quote(venv)}")
        result = RemotePython(remote, python, ui).install_locked(f"{tmp}/{lock.name}", report=False)
    finally:
        remote.run(f"rm -rf {shlex.quote(tmp)}", check=False)
    if not result.ok:
        raise WizardError(f"pip install piper-tts failed on the robot:\n{result.output}")
    ui.ok(f"installed piper-tts {PIPER_TTS_VERSION} (voice files: roboemot models fetch --voice)")


def remove_piper_venv(ui: Ui, remote: Remote, record: InstallRecord, dry_run: bool) -> None:
    """Delete the recorded Piper virtualenv, only at exactly the path this installer would pick."""
    if record.piper_venv is None:
        return
    venv, _ = piper_paths(remote)
    if record.piper_venv != venv:
        ui.warn(f"recorded Piper virtualenv {record.piper_venv!r} is not {venv}: left as it is")
        return
    if not _exists(remote, venv):
        ui.ok(f"Piper virtualenv {venv} is not there")
        return
    real = remote.run(f"realpath {shlex.quote(venv)} 2>/dev/null || true").strip()
    if real != venv:
        ui.warn(f"{venv} resolves to {real or 'nothing'}: symlink? left as it is")
        return
    if not dry_run:
        remote.run(f"rm -rf {shlex.quote(venv)}")
    ui.ok(f"{'would delete' if dry_run else 'deleted'} the Piper virtualenv {venv}")


# ---------------------------------------------------------------- models on the robot


def model_dirs(remote: Remote, vision: bool) -> list[str]:
    """The folders ``roboemot models fetch`` fills on the robot: voice models, and the vision
    weights with ``vision`` (``roboemot.paths`` defaults inside the data folder)."""
    _, data = remote_data_dir(remote)
    subdirs = [paths.VOICE_MODELS_SUBDIR, *([paths.MODELS_SUBDIR] if vision else [])]
    return [posixpath.join(data, d) for d in subdirs]


def models_label(vision: bool) -> str:
    return "voice and vision models" if vision else "voice models"


def models_bytes(vision: bool, peak: bool = True) -> int:
    """Disk the fetch needs, from the pinned manifests: each file's download size plus, for a
    zip archive, its unpacked size (both exist while it unpacks; ``peak=False`` counts only what
    stays). Vision adds the INT8 files derived from the FP32 weights (``VISION_DERIVED_BYTES``)."""
    from ..vision import models as vision_models
    from ..voice import models as voice_models

    specs = list(voice_models.load_manifest().values())
    if vision:
        specs += list(vision_models.load_manifest().values())
    total = 0
    for f in (f for m in specs for f in m.files):
        stays = f.unpacked_bytes if f.archive else f.size_bytes
        total += (f.size_bytes or 0) + (f.unpacked_bytes or 0) if peak else (stays or 0)
    return total + (VISION_DERIVED_BYTES if vision else 0)


def free_mib_at(remote: Remote, path: str) -> int | None:
    """Free space in MiB on the filesystem holding ``path`` (its nearest existing parent), or
    None when ``df`` fails or answers something unreadable."""
    while not _exists(remote, path) and path != "/":
        path = posixpath.dirname(path)
    try:
        free_kib = int(remote.run(f"df -Pk {shlex.quote(path)} | tail -1").split()[3])
    except (WizardError, ValueError, IndexError):
        return None
    return free_kib // 1024


def fetch_models(
    ui: Ui,
    platform: Platform,
    py: RemotePython,
    remote: Remote,
    record: InstallRecord,
    vision: bool,
    dry_run: bool,
) -> None:
    """Download the voice models (and the vision weights with ``vision``) on the robot by running
    the installed CLI, so they land where the app reads them (``roboemot.paths``).

    Skipped with a warning, not an error, when the filesystem lacks the space the pinned
    manifests state (same unit on both sides: MiB), when the free space cannot be read, or when
    the download fails: ``roboemot models fetch`` can be run later. Model
    folders the installer creates inside a data folder that is not recorded as its own are
    recorded before the download, so uninstall removes them; inside a recorded data folder they
    go with it. The dry run only reports sizes and free space.
    """
    need_mib, label = -(-models_bytes(vision) // MIB), models_label(vision)
    folders = model_dirs(remote, vision)
    free_mib = free_mib_at(remote, folders[0])
    if free_mib is None:
        ui.warn(
            f"cannot read the free space for the {label} in {folders[0]}: skipped "
            "(run `roboemot models fetch` on the robot)"
        )
        return
    if free_mib < need_mib:
        ui.warn(
            f"only {free_mib} MiB free for the {label} in {folders[0]}, they need {need_mib} MiB: "
            "skipped (free space, then run `roboemot models fetch` on the robot"
            + (", or install with --no-vision" if vision else "")
            + ")"
        )
        return
    if dry_run:
        ui.info(f"would download the {label} ({need_mib} MiB needed, {free_mib} MiB free)")
        return
    ui.ok(f"{free_mib} MiB free, the {label} need {need_mib} MiB")
    if record.data_dir is None:
        new = [d for d in folders if d not in record.models_dirs and not _exists(remote, d)]
        if new:
            record.models_dirs += new
            write_record(remote, platform.record_path, record)
    result = py.cli(f"models fetch {'--all' if vision else '--voice'}", f"downloading the {label}")
    if not result.ok:
        ui.warn(
            f"the {label} were not downloaded:\n{result.output}\n"
            "run `roboemot models fetch` on the robot later"
        )
        return
    ui.ok(f"{label} are in {', '.join(folders)}")


def remove_models_dirs(ui: Ui, remote: Remote, record: InstallRecord, dry_run: bool) -> None:
    """Delete the recorded model folders, only at the exact paths the fetch would choose and
    only when their ``realpath`` is the same (the data folder's own removal covers the rest)."""
    expected = set(model_dirs(remote, vision=True))
    for folder in record.models_dirs:
        if folder not in expected:
            ui.warn(
                f"recorded model folder {folder!r} is not one of {sorted(expected)}: left as is"
            )
        elif not _exists(remote, folder):
            ui.ok(f"model folder {folder} is not there")
        elif remote.run(f"realpath {shlex.quote(folder)} 2>/dev/null || true").strip() != folder:
            ui.warn(f"{folder} resolves elsewhere: symlink? left as it is")
        else:
            if not dry_run:
                remote.run(f"rm -rf {shlex.quote(folder)}")
            ui.ok(f"{'would delete' if dry_run else 'deleted'} the model folder {folder}")


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
            ui.warn(f"PIN rejected: {e}")
            continue
        if ui.ask_secret("Repeat the PIN") == pin:
            return pin
        ui.warn("the two entries differ")
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
        ui.info(f"would {'offer to replace' if exists else 'ask for'} the operator PIN ({path})")
        return
    if exists and not ui.confirm("An operator PIN is already set. Replace it?", False):
        ui.ok("operator PIN kept")
        return
    pin = ask_new_pin(ui)
    command = f"{shlex.quote(py.python)} -m roboemot.webui set-pin --data-dir {shlex.quote(folder)}"
    if pin is None:
        ui.warn(
            "operator PIN not set; the web UI will not let anyone in. Set it later by running "
            f"on the robot: {command}"
        )
        return
    remote.run(command, stdin=pin + "\n")
    ui.ok("operator PIN set")


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
    ui.warn("the two entries differ")
    return None


def check_secret_names(names: list[str]) -> None:
    """Reject a name that is not a valid secret name, before anything runs on the robot."""
    for name in names:
        if not SECRET_NAME.match(name):
            raise WizardError(f"secret name {name!r} must use letters, numbers, '_', '.' or '-'")


def wanted_secrets(ui: Ui, opts: Options) -> list[str]:
    """Names to store: from ``--secret``, else (interactive, not ``-y``) ask once for a cloud
    route key. Names that are not valid are rejected before anything runs."""
    names = opts.secrets
    if names is None and ui.interactive and not ui.assume_yes:
        if not ui.confirm("Store an API key for a cloud LLM route on the robot?", False):
            return []
        names = [ui.ask("Secret name (the ${secret:name} in the route)", DEFAULT_SECRET_NAME)]
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
            ui.info(f"would ask for the secret {name!r} and store it in {folder}/secrets.json")
            continue
        have = remote.run(robot_cli(py, folder, "list")).split()
        if name in have and not ui.confirm(f"Secret {name!r} is already set. Replace it?", False):
            ui.ok(f"secret {name} kept")
            continue
        value = ask_secret_value(ui, name)
        if value is None:
            ui.warn(
                f"secret {name} not set; set it later on the robot with: "
                f"{robot_cli(py, folder, 'set', name)}"
            )
            continue
        if name not in have and name not in record.secrets:
            record.secrets.append(name)
            write_record(remote, platform.record_path, record)
        remote.run(robot_cli(py, folder, "set", name), stdin=value + "\n")
        ui.ok(f"secret {name} stored")


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
        ui.ok(f"would remove the secrets {', '.join(names)}")
        return
    remote.run(robot_cli(py, folder, "remove", *names))
    record.secrets = []
    write_record(remote, platform.record_path, record)
    ui.ok(f"removed the secrets {', '.join(names)}")


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
    would = "would " if dry_run else ""
    if record.data_dir is None:
        _, path = remote_data_dir(remote)
        if _exists(remote, path):
            ui.warn(f"no record of the data folder: {path} left as it is (delete it by hand)")
        return
    path, problem = checked_data_dir(remote, record)
    if path is None:
        ui.warn(f"{problem}: left as it is")
        return
    if _exists(remote, path):
        if not dry_run:
            remote.run(f"rm -rf {shlex.quote(path)}")
        ui.ok(f"{would}delete the data folder {path} (registry, models and trait cache included)")
    else:
        ui.ok(f"data folder {path} is not there")
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
        ui.warn(
            f"ROBOEMOT_DATA_DIR={env} is set for the SSH user: data may land outside the record"
        )
    for root in dirs:
        for cfg in root.rglob("app.yaml"):
            try:
                value = (yaml.safe_load(cfg.read_text()) or {}).get("data_dir")
            except (OSError, yaml.YAMLError, AttributeError):
                continue
            if value:
                ui.warn(f"{cfg.name} sets data_dir={value}: uninstall will not remove that folder")


# ---------------------------------------------------------------- install


@dataclass
class EnvFate:
    """What uninstall does with the environment the record says this installer created:
    ``action`` is ``none`` (not created by it), ``gone``, ``keep`` or ``delete``; ``note`` says
    why, for the report."""

    action: str
    note: str = ""


def created_env_fate(
    platform: Platform, py: RemotePython, remote: Remote, record: InstallRecord
) -> EnvFate:
    """Decide, read-only, whether the created environment goes: only the exact path the
    platform names, only with an interpreter, no other app or foreign package in it, and only
    if its ``realpath`` is the same."""
    env = record.env_created
    if env is None:
        return EnvFate("none")
    if env != platform.env_dir:
        return EnvFate(
            "keep", f"recorded environment {env!r} is not {platform.env_dir}: left as it is"
        )
    if not _exists(remote, env):
        return EnvFate("gone", f"environment {env} is not there")
    if not py.exists():
        return EnvFate(
            "keep", f"{env} has no interpreter: cannot rule out other apps in it, left as it is"
        )
    users = platform.env_users(py)
    if users:
        return EnvFate("keep", f"keeping {env}: also used by {', '.join(users)}")
    accounted = {canonical(p) for p in record.packages} | set(record.added)
    foreign = sorted(set(py.installed()) - accounted - set(record.env_baseline))
    if foreign:
        return EnvFate(
            "keep",
            f"keeping {env}: it holds packages this installer did not put there: "
            f"{', '.join(foreign)}",
        )
    real = remote.run(f"realpath {shlex.quote(env)} 2>/dev/null || true").strip()
    if real != env:
        return EnvFate("keep", f"{env} resolves to {real or 'nothing'}: symlink? left as it is")
    return EnvFate("delete")


def remove_created_env(
    ui: Ui, platform: Platform, remote: Remote, fate: EnvFate, dry_run: bool
) -> bool:
    """Carry out ``fate`` (from ``created_env_fate``). Returns True when the environment is
    deliberately left in place."""
    if fate.action == "gone":
        ui.ok(fate.note)
    elif fate.action == "keep":
        ui.warn(fate.note)
    elif fate.action == "delete":
        env = platform.env_dir
        if not dry_run:
            remote.run(f"rm -rf {shlex.quote(env)}")
        ui.ok(f"{'would delete' if dry_run else 'deleted'} the environment {env}")
    return fate.action == "keep"


def finish_env(
    ui: Ui,
    platform: Platform,
    py: RemotePython,
    remote: Remote,
    record: InstallRecord,
    dry_run: bool,
) -> bool:
    """Finish an environment an earlier install created but could not complete (the venv is
    there, the platform's base packages are not), by running the platform's create step again,
    then take the baseline anew. An incomplete environment this installer did not create is not
    its to change: a WizardError. Returns False after the dry run's report (nothing more to
    check without the environment)."""
    env = platform.env_dir
    if record.env_created != env:
        raise WizardError(
            f"{env} lacks what {platform.display_name} apps need and was not created by this "
            "installer: repair it with the robot's own software, then run the installer again"
        )
    plan = platform.plan_env(remote)
    if dry_run:
        ui.ok(f"would finish the environment {env} an earlier install created ({plan})")
        ui.info("dependency check skipped: the environment is not complete yet")
        claim_data_dir(ui, platform, remote, record, dry_run=True)
        return False
    ui.warn(f"{env} was created by an earlier install that stopped: finishing it")
    platform.create_env(ui, remote, plan)
    record.env_baseline = sorted(py.installed())
    write_record(remote, platform.record_path, record)
    return True


def install_packages(
    ui: Ui,
    platform: Platform,
    py: RemotePython,
    remote: Remote,
    dirs: list[Path],
    wheels: list[Path],
    dry_run: bool,
    extras: tuple[str, ...] = (),
) -> InstallRecord:
    own = distribution_names(dirs)
    warn_data_dir_overrides(ui, remote, dirs)
    record = read_record(remote, platform.record_path, ui) or InstallRecord(
        platform.app_name, platform.manifest_id, own
    )
    record.packages = sorted(set(record.packages) | set(own))
    if py.exists() and not platform.env_ready(py):
        if not finish_env(ui, platform, py, remote, record, dry_run):
            return record
    elif not py.exists():
        plan = platform.plan_env(remote)
        if dry_run:
            ui.ok(f"would create the environment {platform.env_dir} ({plan})")
            ui.info("dependency check skipped: the environment does not exist yet")
            claim_data_dir(ui, platform, remote, record, dry_run=True)
            ui.info(f"would record the changes in {platform.record_path}; nothing installed")
            return record
        record.env_created = platform.env_dir
        write_record(remote, platform.record_path, record)
        try:
            platform.create_env(ui, remote, plan)
        finally:
            if py.exists():
                record.env_baseline = sorted(py.installed())
                write_record(remote, platform.record_path, record)
    before = py.installed()
    if record.pending_before is not None:
        ui.warn("an earlier install was interrupted; folding its changes into the record")
        record.merge(record.pending_before, before)
    pins = py.constraints(before, set(own))
    if dry_run:
        reqs = third_party_requirements(dirs, extras)
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
        claim_data_dir(ui, platform, remote, record, dry_run=True)
        ui.info(f"would record the changes in {platform.record_path}; nothing copied or installed")
        return record
    tmp = remote.run(f"mktemp -d {REMOTE_TMP_PREFIX}.XXXXXX")
    try:
        remote.copy(wheels, tmp)
        ui.ok(f"copied {', '.join(w.name for w in wheels)}")
        files = [f"{tmp}/{w.name}" for w in wheels]
        targets = with_extras(files, extras)
        record.pending_before = before
        write_record(remote, platform.record_path, record)
        result = py.install(targets, pins)
        if not result.ok:
            ui.warn("the app does not fit the package versions already on the robot")
            free = py.plan_install(targets, None)
            if not free.ok:
                raise WizardError(f"pip cannot resolve the dependencies:\n{free.output}")
            _, changed = report_changes(_json(free.output), before)
            for n, c in changed.items():
                ui.info(f"{n} {c}")
            question = "Change these shared packages? Uninstall restores them (needs internet)"
            if not ui.confirm(question, False):
                raise WizardError("install cancelled: shared packages left as they were")
            result = py.install(targets, None)
            if not result.ok:
                raise WizardError(f"pip install failed on the robot:\n{result.output}")
        py.reinstall_own(files)
        after = py.installed()
        record.merge(before, after)
        claim_data_dir(ui, platform, remote, record, dry_run=False)
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
    check_secret_names(opts.secrets or [])
    ui.title("installer", "dry run: the robot is not modified" if opts.dry_run else "")
    ui.step("Robot")
    host, platform = select_robot(ui, opts, root)
    ui.ok(platform.display_name)
    platform.connect(ui, host)

    stopped: list[bool] = []  # whether this run stopped the running app

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
            app_config = app_config_overrides(opts.voice)
            content = describe_app_config(dev, app_config)
            ui.ok(f"app config (data/app.yaml): {content}" if content else "no app config file")
            dirs = platform.stage(root, Path(tmp) / "src", dev, app_config)

            ui.step("SSH access")

            def free_space() -> None:
                stopped.append(stop_own_app(ui, platform, opts.dry_run) and not opts.dry_run)
                home, _ = remote_data_dir(remote)
                clear_caches(
                    ui, remote, home, platform.clearable_caches, opts.clear_caches, opts.dry_run
                )

            extras = (*platform.extras, *(("vision",) if opts.vision else ()))
            py = check_ssh(ui, platform, remote, min_python(dirs), free_space, extras)

            ui.step("Build")
            wheels = build_wheels(dirs, Path(tmp) / "wheels")
            ui.ok(f"built {', '.join(w.name for w in wheels)}")

            ui.step("Install (dependency check only)" if opts.dry_run else "Install")
            if opts.vision and opts.dry_run:
                ui.info("would install the vision extra (OpenCV, ONNX); --no-vision skips it")
            record = install_packages(ui, platform, py, remote, dirs, wheels, opts.dry_run, extras)
            if not opts.dry_run:
                platform.verify(ui, py, dev, app_config)
            if opts.piper is None:
                opts.piper = opts.voice and ui.confirm(
                    "Install Piper for on-board speech (own virtualenv)?", True
                )
            if opts.piper:
                install_piper(ui, platform, remote, record, opts.dry_run)
            if opts.models is None:
                opts.models = (opts.voice or opts.vision) and ui.confirm(
                    f"Download the {models_label(opts.vision)} "
                    f"(~{models_bytes(opts.vision) / 1e9:.1f} GB at peak while unpacking, "
                    f"~{models_bytes(opts.vision, peak=False) / 1e9:.1f} GB once installed) "
                    "on the robot?",
                    True,
                )
            warn_voice_gaps(ui, opts)
            if opts.models:
                fetch_models(ui, platform, py, remote, record, opts.vision, opts.dry_run)
            set_operator_pin(ui, py, remote, opts.dry_run)
            names = wanted_secrets(ui, opts)
            set_secrets(ui, platform, py, remote, record, names, opts.dry_run)

            ui.step("Start")
            if opts.dry_run:
                running = platform.running_app()
                if running == platform.app_name:
                    action = "the stop is counted above; would "
                else:
                    action = f"would stop {running}, " if running else "no app running; would "
                ui.info(f"{action}wake the robot and start {platform.app_name}")
                blocker = platform.start_blocker()
                if blocker:
                    ui.warn(f"would refuse: {blocker}")
                ui.info(f"autostart app now: {platform.autostart() or 'none'}")
                ui.title("dry run passed")
                return
            blocker = platform.start_blocker()
            if blocker:
                ui.warn(f"{platform.app_name} not started: {blocker}")
                ui.info("start it once that is done: python scripts/run_on_robot.py")
            elif (
                opts.start
                if opts.start is not None
                else ui.confirm(f"Start {platform.app_name} now?", True)
            ):
                platform.start(ui)
                report_app_url(ui, platform, host)
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
            ui.info("start it later: python scripts/run_on_robot.py")
            ui.info("remove everything it changed: python scripts/install_on_robot.py --uninstall")

    try:
        _with_remote(opts, ui, platform, host, body)
    except Exception:
        if any(stopped):
            ui.warn(
                f"{platform.app_name} was stopped before installing; start it again with "
                "python scripts/run_on_robot.py once the problem is fixed"
            )
        raise


def stop_own_app(ui: Ui, platform: Platform, dry_run: bool) -> bool:
    """Stop the platform's app when it is the one running, before any package changes: a running
    process keeps the old code. Another app is left alone. True when the app was running (and,
    outside a dry run, is now stopped)."""
    if platform.running_app() != platform.app_name:
        return False
    if dry_run:
        ui.info(f"would stop {platform.app_name} before installing")
        return True
    platform.stop_app()
    ui.ok(f"stopped {platform.app_name} before installing: a running process keeps the old code")
    return True


def report_app_url(ui: Ui, platform: Platform, host: str) -> None:
    """Tell where the running app's web UI is, if the platform has one."""
    url = platform.app_url(host)
    if url:
        ui.ok(f"web UI: {url}")


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
        env_there = py.exists()
        installed = py.installed() if env_there else {}
        record = read_record(remote, platform.record_path, ui)
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

        if env_there:
            remove_secrets(ui, platform, py, remote, record, opts.dry_run)
        elif record.secrets and record.data_dir is None:
            ui.warn(f"cannot remove the secrets {', '.join(record.secrets)}: no environment to run")

        ui.step("Packages")
        fate = created_env_fate(platform, py, remote, record)
        if fate.action == "delete":
            # The whole environment goes below: restoring versions inside it first would only
            # need the package index (internet) for nothing.
            ui.ok(f"the app's packages are {would}removed with the environment {platform.env_dir}")
            plan = UninstallPlan([], {}, {}, {})
        else:
            plan = plan_uninstall(record, installed, py.requires() if env_there else {})
        if not env_there:
            plan.drifted = {}  # nothing to compare against: the environment is gone
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
        # before removal (dry run), the app's modules are expected
        if fate.action != "delete" and env_there and not (opts.dry_run and plan.remove):
            for module, path in leftovers(py, platform):
                if not opts.dry_run:
                    remote.run(f"rm -rf {shlex.quote(path)}")
                ui.warn(f"{would}delete leftover {path} ({module})")

        ui.step("Clean up and verify")
        keep_env = remove_created_env(ui, platform, remote, fate, opts.dry_run)
        remove_piper_venv(ui, remote, record, opts.dry_run)
        remove_models_dirs(ui, remote, record, opts.dry_run)
        remove_data_dir(ui, remote, record, opts.dry_run)
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
        verify_uninstall(ui, platform, py, record, plan.restore, keep_env)
        remote.run(f"rm -f {shlex.quote(platform.record_path)}")
        ui.ok(f"deleted {platform.record_path}")
        ui.title("uninstalled")

    _with_remote(opts, ui, platform, host, body)


def verify_uninstall(
    ui: Ui,
    platform: Platform,
    py: RemotePython,
    record: InstallRecord,
    restored: dict[str, str],
    keep_env: bool,
) -> None:
    """The app is gone and every restored version is back; else keep the record and stop."""
    env_there = py.exists()
    now = py.installed() if env_there else {}
    problems = [f"{n} still installed" for n in map(canonical, record.packages) if n in now]
    problems += [
        f"{n} is {now.get(n, 'missing')}, expected {v}"
        for n, v in restored.items()
        if now.get(n) != v
    ]
    if env_there:
        problems += [f"{p} still present ({m})" for m, p in leftovers(py, platform)]
    if (
        record.env_created == platform.env_dir
        and not keep_env
        and _exists(py.remote, platform.env_dir)
    ):
        problems.append(f"environment {platform.env_dir} still present")
    venv, _ = piper_paths(py.remote)
    if record.piper_venv == venv and _exists(py.remote, venv):
        problems.append(f"Piper virtualenv {venv} still present")
    expected_models = set(model_dirs(py.remote, vision=True))
    problems += [
        f"model folder {d} still present"
        for d in record.models_dirs
        if d in expected_models and _exists(py.remote, d)
    ]
    data_dir, bad = checked_data_dir(py.remote, record)
    if bad:
        problems.append(bad)
    elif data_dir and _exists(py.remote, data_dir):
        problems.append(f"data folder {data_dir} still present")
    if problems:
        raise WizardError("uninstall incomplete (record kept, run again): " + "; ".join(problems))
    ui.ok("app packages gone, original versions back" + (", data folder gone" if data_dir else ""))
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
    p.add_argument(
        "--voice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="turn the voice pipeline on in the app config (voice.enabled; default: on). "
        "--no-voice writes no voice setting and makes --piper and --models default to off "
        "(--models still defaults to on with vision, whose weights it fetches)",
    )
    p.add_argument(
        "--piper",
        action=argparse.BooleanOptionalAction,
        help="install Piper (on-board speech) in its own virtualenv in the data folder "
        "(default: ask, off with --no-voice)",
    )
    p.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="install the vision extra (OpenCV, ONNX) and, with the models step, the vision "
        "weights (default: on; --no-vision skips both, e.g. when the robot's disk is small)",
    )
    p.add_argument(
        "--models",
        action=argparse.BooleanOptionalAction,
        help="download the voice models (and the vision weights with --vision) on the robot "
        "(default: ask)",
    )
    p.add_argument(
        "--secret",
        action="append",
        metavar="NAME",
        help="store the secret NAME (an API key for ${secret:NAME}) on the robot; the value is "
        "asked on this PC without echo and never appears on a command line",
    )
    p.add_argument(
        "--clear-caches",
        action=argparse.BooleanOptionalAction,
        help="delete the robot's download caches the platform names (e.g. a uv cache) to "
        "free disk space; uninstall cannot restore them",
    )
    p.add_argument("-y", "--yes", action="store_true", help="answer yes to every confirmation")
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
        a.clear_caches,
        a.vision,
        a.models,
        a.voice,
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
    "report_app_url",
    "select_robot",
    "paint",
    "run_install",
    "run_uninstall",
]
