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
records it. It sets its own keys in the app config (``app.yaml`` in the data folder, kept across
reinstalls): ``voice.enabled: true`` unless ``--no-voice`` (which also makes Piper and the models
default to off) and the dev content with ``--dev``; every other key stays. Unless declined
(`--no-piper`) it also creates a Piper virtualenv in the data folder (`piper-tts` is GPL-3.0 and
runs as its own process, SPEC 8), recorded for uninstall.
The platform's ``extras`` (``voice`` for a robot that listens, so vosk; ``realtime``, websockets,
for the realtime speech route) are installed with
``roboemot-core``; the ``vision`` extra and the vision weights come too unless declined
(``--no-vision``, SPEC 4A Installer). The voice models (``roboemot models fetch --voice``) are
downloaded on the robot after Piper unless declined (``--no-models``), when the manifests'
sizes of the files not yet verified fit the free space. Before the free-space check it offers to
clear the platform's ``clearable_caches`` (``--clear-caches``; ``-y`` alone keeps them), which
uninstall cannot restore.
Optionally (``--secret NAME``, or asked on a terminal) it stores an API key in the robot's secret
store (SPEC 8, 12 Data), recorded by name; uninstall removes only the names it added.
The cloud-voice profile (``--cloud-voice``, an admin opt-in: SPEC 12) answers by voice through
the platform's realtime speech route (``Platform.cloud_voice``, SPEC 8): it writes that route and
the settings that use it into ``app.yaml`` (keys recorded, so ``--no-cloud-voice`` on a reinstall
removes exactly them), asks for the route's key when the store has none, skips Piper and its
voices and removes a recorded Piper virtualenv. ``-y`` does not turn it on; an unattended run
names ``--cloud-voice`` or ``--no-cloud-voice``.

Later, ``scripts/run_on_robot.py`` (``roboemot.install.run``) starts the installed app and
prints its web UI address.

Uninstall: stop the app, give back the previous autostart app (only if the setting still names
this app), uninstall what the record says this installer added, reinstall the original versions
it replaced, keep anything another distribution still needs, remove leftovers and the recorded
data folder (registry, models and trait cache included), verify, and only then delete the
record. An environment the record says this installer created is removed too, unless another
app lives in it (``Platform.env_users``); when it goes whole, its packages are not restored
first. With the environment already gone, the rest of the cleanup still runs. A rerun of install
finishes an environment an earlier run created but could not complete. Running it twice is
harmless. Both flows have a read-only ``--dry-run``.
"""

from __future__ import annotations

import argparse
import copy
import json
import operator
import os
import posixpath
import re
import shlex
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
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
# Disk at the install's peak, in MB as (downloads, unpacked), measured from the aarch64 cp312
# wheels pip picks for each set (2026-10-22), unpacked sizes rounded up about 25% for the .pyc
# files pip compiles. The app with its dependencies: scipy, numpy, openai, pydantic (52 MiB
# downloaded, 184 MiB unpacked). Extras: voice is onnxruntime and vosk (24, 66); vision is
# opencv-python-headless and onnx (43, 105) plus onnxruntime (20, 53), counted again so that
# vision alone also fits. Realtime is websockets 15.0.1 (0.18 MB downloaded, 0.83 MB unpacked,
# measured 2026-10-27), rounded up to whole MB; a robot that already has it keeps its version.
APP_DISK_MB = (60, 230)
EXTRA_DISK_MB = {"voice": (30, 85), "vision": (70, 200), "realtime": (1, 2)}
REALTIME_EXTRA = "realtime"
REALTIME_PACKAGE = "websockets"  # what the realtime extra brings (roboemot-core's pyproject)
VERSION_OPS = {  # two-character operators first: ">=" must not read as ">"
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
}
ENV_DOWNLOAD_SHARE = 0.5  # downloads next to a created environment's tree (measured 0.3-0.4)
ENV_FALLBACK_MB = 1500  # that tree when `du` on the creating environment answers nothing
REMOTE_TMP_PREFIX = "/tmp/roboemot-install"
TMP_DIR_PATTERN = re.compile(re.escape(REMOTE_TMP_PREFIX) + r"\.[A-Za-z0-9]+")
PIN_FILE = "webui_pin.json"  # roboemot.webui.auth.PIN_FILE, inside the app data folder
PIN_RETRIES = 3
DEFAULT_SECRET_NAME = "cloud_key"  # the name configs/companion.yaml gives the cloud route key
MIB = 1024 * 1024
PIPER_TTS_VERSION = "1.8.0"  # exact pin: the worker protocol is tested against this API
FETCH_ATTEMPTS = 2  # `roboemot models fetch` runs once more after a failure (it resumes)
# The model files already verified on the robot (marker check only, read-only: no folder made),
# as "voice:<path>" / "vision:<path>", and for the others the bytes already on disk that the
# fetch frees or keeps: a stale ``.part`` (deleted before the new download) and a complete-size
# file (kept when its digest matches), plus the bytes of retired vision files still in the folder
# (``roboemot models prune`` deletes them; the installer does not). Run with the app's interpreter.
VERIFIED_MODELS = r"""
import json
from roboemot.vision import models as m
from roboemot.voice import models as v
done, on_disk = [], {}
for kind, specs, root in (("vision", m.load_manifest(), m.models_dir()),
                          ("voice", v.load_manifest(), v.models_dir())):
    for spec in specs.values():
        for f in spec.files:
            key = kind + ":" + f.path
            try:
                m.check_file(root, f)
            except Exception:
                t = root / f.path
                part = t.with_name(t.name + (".zip" if f.archive else "") + m.PART_SUFFIX)
                n = part.stat().st_size if part.is_file() else 0
                if not f.archive and f.size_bytes and t.is_file():
                    n += f.size_bytes if t.stat().st_size == f.size_bytes else 0
                if n:
                    on_disk[key] = n
                continue
            done.append(key)
retired = sum(n for _, n in m.retired_on_disk(m.models_dir()))
print(json.dumps({"done": done, "on_disk": on_disk, "retired": retired}))
"""
PLATFORM_PROBE = (
    "import sys, platform; print(platform.machine()); print('%d.%d' % sys.version_info[:2])"
)
PROBE_PATTERN = re.compile(r"[A-Za-z0-9_]+\n\d+\.\d+")
INSTALL_STEPS = 6
DEV_CONFIG = "configs/dev-unmeasured.yaml"
APP_CONFIG_SCHEMA = 1
APP_CONFIG_HEADER = (
    "# RoboEmot app config (SPEC 4A, Installer). The installer sets only voice.enabled, in dev\n"
    "# mode safety.allow_unmeasured_dynamics and, with --cloud-voice, the cloud-voice route and\n"
    "# settings; every other key is kept on reinstall (comments are not). Uninstall removes this\n"
    "# whole file, settings saved from the web UI included.\n"
)
ROUTES_KEY = "llm_routes"  # a list keyed by route name: merged per entry, never whole
CLOUD_VOICE_FLAG = "--cloud-voice"
UNINSTALL_STEPS = 6
SHARED_NEEDS_YES = "Shared packages must change, and only --yes allows it"


@dataclass(frozen=True)
class CloudVoiceRoute:
    """The realtime speech route a platform offers for the cloud-voice profile (SPEC 4A,
    Installer; SPEC 8, Realtime speech route): the ``llm_routes`` entry (``name``, ``kind``,
    ``base_url``, ``model``; its key is ``${secret:<secret>}`` from the secret store), the
    ``voice.realtime_route.mode`` it needs, and ``provider``, who receives the audio, as the
    opt-in question names it."""

    name: str
    kind: str
    base_url: str
    secret: str
    model: str
    mode: str
    provider: str

    def entry(self) -> dict[str, str]:
        """The ``llm_routes`` entry as the installer writes it into ``app.yaml``."""
        return {
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "api_key": f"${{secret:{self.secret}}}",
            "model": self.model,
        }


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
    # Restart the running app, asking first, and wait until it runs (``run --restart``); None: the
    # platform cannot restart it.
    restart: Callable[[Ui], None] | None
    # The route of the cloud-voice profile (``--cloud-voice``); None: the platform has none.
    cloud_voice: CloudVoiceRoute | None

    def connect(self, ui: Ui, host: str) -> None: ...
    def plan_env(self, remote: Remote) -> str: ...
    def create_env(self, ui: Ui, remote: Remote, plan: str) -> None: ...
    def env_users(self, py: RemotePython) -> list[str]: ...
    def env_ready(self, py: RemotePython) -> bool: ...
    def probe(self, ui: Ui, py: RemotePython) -> None: ...
    def stage(self, root: Path, dest: Path) -> list[Path]: ...
    def verify(self, ui: Ui, py: RemotePython) -> None: ...
    def running_app(self) -> str | None: ...
    def stop_app(self) -> None: ...
    def app_installed(self) -> bool: ...
    def app_url(self, host: str, port: int | None = None) -> str | None: ...
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
    restart: bool = False
    debug: bool = False
    version: bool = False
    cloud_voice: bool | None = None  # the cloud-voice profile; None: not answered yet


# ---------------------------------------------------------------- app config


def app_config_overrides(voice: bool) -> dict[str, object]:
    """What the installer sets in the robot's app config besides the dev content:
    ``voice.enabled: true`` with voice (the library default is off, so a robot without a
    microphone or speaker is unchanged; the installer enables it for the robot it sets up).
    ``apply_app_config`` writes it into ``app.yaml`` in the data folder and reads it back to
    verify (SPEC 4A, Installer)."""
    return {"voice": {"enabled": True}} if voice else {}


def dev_config(root: Path) -> dict[str, object]:
    """The dev content (``configs/dev-unmeasured.yaml`` without its ``schema_version``; empty
    when the checkout has no such file)."""
    path = root / DEV_CONFIG
    data = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    return {k: v for k, v in data.items() if k != "schema_version"}


def leaves(tree: dict, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    """Every non-mapping value of a nested mapping, keyed by its path."""
    out: dict[tuple[str, ...], object] = {}
    for key, value in tree.items():
        if isinstance(value, dict) and value:
            out.update(leaves(value, (*prefix, key)))
        else:
            out[(*prefix, key)] = value
    return out


def owned_app_config(
    root: Path, dev: bool, voice: bool
) -> tuple[dict[tuple[str, ...], object], set[tuple[str, ...]]]:
    """The installer's keys in the app config: (the values asked for this run, every key it
    owns). It owns the voice switch and the dev content's keys; a key it owns but was not asked
    for this run (``--no-voice``, ``--no-dev``) is removed from the file."""
    dev_leaves = leaves(dev_config(root))
    if dev and not dev_leaves:
        raise WizardError("Dev mode needs its config", f"{DEV_CONFIG} is missing in {root}")
    voice_leaves = leaves(app_config_overrides(True))
    wanted = {**(dev_leaves if dev else {}), **(voice_leaves if voice else {})}
    return wanted, set(dev_leaves) | set(voice_leaves)


def cloud_voice_config(route: CloudVoiceRoute) -> dict[tuple[str, ...], object]:
    """The keys the cloud-voice profile sets in ``app.yaml`` (SPEC 4A, Installer; SPEC 8, 12):
    the platform's route entry; the realtime route through it, with the utterance's audio as its
    input (``privacy.realtime_audio``, the admin opt-in) and the skip tool; no local voice; wake
    policy ``never`` with the voice-at-face and facing checks (facing needs direction). The
    transcript timeout is the schema default, written so the profile states it."""
    from roboemot.config.schema import RealtimeRouteConfig

    rt = ("voice", "realtime_route")
    return {
        (ROUTES_KEY, route.name): route.entry(),
        (*rt, "enabled"): True,
        (*rt, "route"): route.name,
        (*rt, "mode"): route.mode,
        (*rt, "input"): "audio",
        (*rt, "skip_tool"): True,
        (*rt, "transcript_timeout_s"): RealtimeRouteConfig().transcript_timeout_s,
        ("privacy", "realtime_audio"): True,
        ("voice", "backend"): "none",
        ("voice", "fallback_backend"): "none",
        ("voice", "wake_word_policy"): "never",
        ("voice", "direction", "enabled"): True,
        ("voice", "facing", "enabled"): True,
    }


def key_text(path: tuple[str, ...], value: object) -> str:
    """One app config key as the installer prints it: ``a.b=value``, or ``llm_routes[name]``
    for a route entry."""
    if path[0] == ROUTES_KEY:
        return f"{ROUTES_KEY}[{path[1]}]"
    return ".".join(path) + f"={str(value).lower()}"


def config_value(config: dict, path: tuple[str, ...]) -> object:
    """The value at ``path`` in ``config`` (a route entry by name for ``llm_routes``), or None."""
    if path[0] == ROUTES_KEY:
        routes = config.get(ROUTES_KEY)
        entries = routes if isinstance(routes, list) else []
        return next((r for r in entries if isinstance(r, dict) and r.get("name") == path[1]), None)
    node: object = config
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _merge_route(config: dict, name: str, entry: object, keep: bool) -> None:
    """Set (``keep``) or remove the ``llm_routes`` entry called ``name`` in place, at its own
    position; the owner's other routes stay, and a list left empty goes."""
    routes = config.get(ROUTES_KEY)
    if not isinstance(routes, list):
        if not keep:
            return
        routes = []
    named = [isinstance(r, dict) and r.get("name") == name for r in routes]
    others = [r for r, same in zip(routes, named, strict=True) if not same]
    if keep:
        others.insert(named.index(True) if True in named else len(others), copy.deepcopy(entry))
    if others:
        config[ROUTES_KEY] = others
    else:
        config.pop(ROUTES_KEY, None)


def merge_app_config(
    existing: dict, wanted: dict[tuple[str, ...], object], owned: set[tuple[str, ...]]
) -> dict:
    """``existing`` with every owned key set to its wanted value or removed (and a mapping left
    empty by that removed too); every other key is kept as it is. A ``llm_routes`` key names
    one route entry, merged by name."""
    merged = copy.deepcopy(existing)
    for path in sorted(owned):
        if path[0] == ROUTES_KEY:
            _merge_route(merged, path[1], wanted.get(path), path in wanted)
            continue
        parents, node = [], merged
        for key in path[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                if path not in wanted:
                    break
                child = node[key] = {}
            parents.append((node, key))
            node = child
        else:
            if path in wanted:
                node[path[-1]] = wanted[path]
            else:
                node.pop(path[-1], None)
                for parent, key in reversed(parents):
                    if parent[key] == {}:
                        del parent[key]
    if merged and "schema_version" not in merged:
        merged = {"schema_version": APP_CONFIG_SCHEMA, **merged}
    return merged


def app_config_text(config: dict) -> str:
    """The app config as the installer writes it: a header saying what it owns, then the YAML."""
    return APP_CONFIG_HEADER + yaml.safe_dump(config, sort_keys=False)


def describe_app_config(dev: bool, app_config: dict[str, object]) -> str:
    """One line naming the installer's keys in the app config on the robot (empty: none)."""
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
    listening (with cloud voice the server speaks: no Piper warning). Called once both answers
    are known."""
    if not opts.voice:
        return
    if not opts.piper and not opts.cloud_voice:
        ui.warn("Voice without Piper", "Piper is not installed: the robot will not speak")
    if not opts.models:
        ui.warn("Voice without models", "they are not downloaded: the robot will not listen")


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
            "pip install -e <the robot's package>",
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
    ui: Ui,
    platform: Platform,
    remote: Remote,
    need: tuple[int, int] | None,
    free_space: Callable[[], str | None] | None = None,
    extras: tuple[str, ...] = (),
) -> RemotePython:
    """Log in once; check the target interpreter, architecture, Python and free space.

    ``free_space`` (validate the app config, stop the running app, clear caches) runs once the
    interpreter, architecture, Python and environment checks passed (an incomplete environment
    the installer may not finish is refused here) and before the free-space check, so a refused
    install has stopped nothing and what it frees counts; what it returns (caches it kept) is
    added to a free-space error. The space needed comes from what will be installed
    (``install_need_mb``: the app, ``extras`` and, when the environment is created or finished,
    the tree of the creating one).
    """
    ui.active("Connecting over SSH…", f"{remote.target}; ssh may ask for the password")
    remote.connect()
    ui.ok("Connected over SSH", remote.target)
    py = RemotePython(remote, platform.python, ui)
    present = py.exists()
    if not present and need is None:
        ui.ok("Environment not there", platform.env_dir)
        return py
    if not present and _exists(remote, platform.env_dir):
        raise WizardError(
            "Environment without an interpreter",
            f"{platform.env_dir} exists but {platform.python} does not; remove the folder by "
            "hand or finish creating it, then run the installer again",
        )
    target = platform.python if present else creating_python(platform, remote)
    arch, version = probe_platform(remote, target)
    if arch != platform.arch:
        raise WizardError(
            "Wrong robot architecture", f"the robot is {arch}, the app needs {platform.arch}"
        )
    if need and tuple(map(int, version.split("."))) < need:
        raise WizardError(
            "Target Python too old", f"Python {version}; the app needs {need[0]}.{need[1]}+"
        )
    ui.ok("Checked the interpreter", f"{arch}, Python {version}, {target}")
    if not present:
        ui.info(f"{platform.env_dir} does not exist yet: it is created with {target}.")
    ready = present and platform.env_ready(py)
    if ready:
        platform.probe(ui, py)
    elif present:
        ui.warn(
            "Environment incomplete",
            f"{platform.env_dir}: an earlier install stopped while creating it?",
        )
        if need:
            refusal = finish_refusal(platform, py, read_record(remote, platform.record_path))
            if refusal:
                raise WizardError("Cannot finish the environment", refusal)
    hint = free_space() if free_space is not None else None
    if need:
        downloads, unpacked = install_need_mb(platform, remote, extras, creating=not ready)
        tmp = posixpath.dirname(REMOTE_TMP_PREFIX)
        for space in platform.space_paths:
            path = posixpath.dirname(space) if not present and space == platform.env_dir else space
            want = downloads if space == tmp else downloads + unpacked
            free_kb = int(remote.run(f"df -Pk {shlex.quote(path)} | tail -1").split()[3])
            if free_kb < want * 1024:
                raise WizardError(
                    f"Not enough free space on {path}",
                    f"{free_kb // 1024} MB free, {want} MB needed (worst case: every package "
                    "counted as new)" + (f"; {hint}" if hint else ""),
                )
            ui.ok("Checked free space", f"{free_kb // 1024} MB on {path}, {want} MB needed")
    return py


def realtime_extra_line(dirs: list[Path], have: str | None) -> str:
    """The detail of the dry run's "Install the realtime extra" line: ``have`` is the robot's
    websockets version. "already present" only when it meets the extra's requirement
    (``reachy-mini`` allows an older one than the extra needs); otherwise what the extra needs."""
    spec = next(
        (
            r[len(REALTIME_PACKAGE) :].strip()
            for r in third_party_requirements(dirs, (REALTIME_EXTRA,))
            if canonical(r).startswith(REALTIME_PACKAGE)
            and not r[len(REALTIME_PACKAGE) :][:1].isalnum()
        ),
        "",
    )
    if not have:
        return f"{REALTIME_PACKAGE}, not present yet"
    if version_meets(have, spec):
        return f"{REALTIME_PACKAGE} {have}, already present"
    return f"{REALTIME_PACKAGE} {have} present, needs {spec}"


def version_meets(version: str, spec: str) -> bool:
    """Whether a plain dotted ``version`` meets ``spec`` (comma-separated ``>=``, ``>``, ``<=``,
    ``<``, ``==``, ``!=`` clauses on dotted numbers). Anything it cannot read counts as no."""

    def parts(text: str) -> tuple[int, ...] | None:
        nums = text.strip().split(".")
        return tuple(int(n) for n in nums) if all(n.isdigit() for n in nums) else None

    have = parts(version)
    if have is None:
        return False
    for clause in filter(None, (c.strip() for c in spec.split(","))):
        op = next((o for o in VERSION_OPS if clause.startswith(o)), None)
        want = parts(clause[len(op) :]) if op else None
        if op is None or want is None:
            return False
        width = max(len(have), len(want))
        a, b = have + (0,) * (width - len(have)), want + (0,) * (width - len(want))
        if not VERSION_OPS[op](a, b):
            return False
    return True


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
            "Target interpreter not found",
            f"{platform.python} does not exist and {platform.display_name} cannot create it",
        )
    if not RemotePython(remote, base).exists():
        raise WizardError(
            "Target interpreter not found",
            f"{platform.python} does not exist, and neither does {base}, which creates it",
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
        ui.warn("Unknown record fields", f"this installer drops {', '.join(ignored)}")
    return record


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
    lock = lock_path(*probe_platform(remote, current_python(platform, remote)))
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
            with ui.progress("Creating the Piper venv", report=False):
                remote.run(f"{shlex.quote(platform.python)} -m venv {shlex.quote(venv)}")
            ui.ok("Created the Piper venv", venv)
        result = RemotePython(remote, python, ui).install_locked(f"{tmp}/{lock.name}", report=False)
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


def remove_piper_venv(
    ui: Ui, remote: Remote, record: InstallRecord, dry_run: bool, why: str = ""
) -> bool:
    """Delete the recorded Piper virtualenv, only at exactly the path this installer would pick.
    True when it is gone (or would be, in the dry run); ``why`` ends the done line."""
    if record.piper_venv is None:
        return False
    venv, _ = piper_paths(remote)
    if record.piper_venv != venv:
        ui.warn("Piper venv left", f"recorded {record.piper_venv!r} is not {venv}")
        return False
    if not _exists(remote, venv):
        ui.ok("Piper venv gone", f"{venv} is not there")
        return True
    real = remote.run(f"realpath {shlex.quote(venv)} 2>/dev/null || true").strip()
    if real != venv:
        ui.warn("Piper venv left", f"{venv} resolves to {real or 'nothing'}: symlink?")
        return False
    if not dry_run:
        remote.run(f"rm -rf {shlex.quote(venv)}")
    detail = f"{venv}{'; ' + why if why else ''}"
    ui.change(dry_run, "Delete the Piper venv", "Deleted the Piper venv", detail)
    return True


def drop_piper_venv(
    ui: Ui, platform: Platform, remote: Remote, record: InstallRecord, dry_run: bool
) -> None:
    """Cloud voice (owner decision 7): delete the recorded Piper virtualenv and take it off the
    record, so uninstall and a later reinstall agree with the robot. A folder that is not in the
    record is not this installer's and stays. Switching back to local voice is a reinstall
    without ``--cloud-voice``."""
    why = "the server speaks; a reinstall with --no-cloud-voice brings Piper back"
    if remove_piper_venv(ui, remote, record, dry_run, why) and not dry_run:
        record.piper_venv = None
        write_record(remote, platform.record_path, record)


# ---------------------------------------------------------------- models on the robot


def model_dirs(remote: Remote, vision: bool) -> list[str]:
    """The folders ``roboemot models fetch`` fills on the robot: voice models, and the vision
    weights with ``vision`` (``roboemot.paths`` defaults inside the data folder)."""
    _, data = remote_data_dir(remote)
    subdirs = [paths.VOICE_MODELS_SUBDIR, *([paths.MODELS_SUBDIR] if vision else [])]
    return [posixpath.join(data, d) for d in subdirs]


def models_label(vision: bool, tts: bool = True) -> str:
    label = "voice and vision models" if vision else "voice models"
    return label if tts else f"{label} (no TTS voices: cloud voice)"


def voice_specs(tts: bool) -> list:
    """The voice manifest's models; without ``tts``, not the TTS voices (``tts_`` roles: Piper),
    which the cloud-voice profile does not need (the listener still needs VAD and ASR)."""
    from ..voice import models as voice_models

    specs = voice_models.load_manifest().values()
    return [s for s in specs if tts or not s.role.startswith(voice_models.TTS_ROLE)]


def fetch_args(vision: bool, tts: bool) -> str:
    """The ``roboemot models fetch`` arguments: ``--all`` / ``--voice``, or without ``tts`` the
    names of the voice models it keeps (and every vision model with ``vision``)."""
    if tts:
        return f"models fetch {'--all' if vision else '--voice'}"
    from ..vision import models as vision_models

    names = [s.name for s in voice_specs(tts)]
    if vision:
        names += list(vision_models.load_manifest())
    return "models fetch " + " ".join(map(shlex.quote, names))


def models_bytes(
    vision: bool,
    peak: bool = True,
    done: frozenset[str] = frozenset(),
    on_disk: dict[str, int] | None = None,
    tts: bool = True,
) -> int:
    """Disk the fetch needs, from the pinned manifests: each file's download size plus, for a
    zip archive, its unpacked size (both exist while it unpacks; ``peak=False`` counts only what
    stays). Without ``tts`` the TTS voices are left out. Files in ``done`` (``"voice:<path>"``,
    ``"vision:<path>"``: already verified on the robot) need nothing. ``on_disk`` (same keys) are
    bytes a file already holds on the robot (a stale partial download, a complete file not yet
    marked), taken off that file's download size."""
    from ..vision import models as vision_models

    groups = [("voice", voice_specs(tts))]
    if vision:
        groups.append(("vision", vision_models.load_manifest().values()))
    total = 0
    for kind, specs in groups:
        for f in (f for m in specs for f in m.files):
            if f"{kind}:{f.path}" in done:
                continue
            stays = f.unpacked_bytes if f.archive else f.size_bytes
            have = min((on_disk or {}).get(f"{kind}:{f.path}", 0), f.size_bytes or 0)
            download = (f.size_bytes or 0) - have
            total += download + (f.unpacked_bytes or 0) if peak else (stays or 0)
    return total


def verified_models(py: RemotePython) -> tuple[frozenset[str], dict[str, int], int]:
    """The model files already verified on the robot, the bytes others already hold and the
    bytes of retired model files still there (``VERIFIED_MODELS``); empty when the app is not
    installed yet or the query fails, so the full size counts."""
    if not py.exists():
        return frozenset(), {}, 0
    try:
        data = json.loads(py.py(VERIFIED_MODELS))
        on_disk = {str(k): int(n) for k, n in data["on_disk"].items()}
        return frozenset(map(str, data["done"])), on_disk, int(data.get("retired", 0))
    except (WizardError, ValueError, TypeError, KeyError, AttributeError):
        return frozenset(), {}, 0


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
    tts: bool = True,
) -> None:
    """Download the voice models (and the vision weights with ``vision``) on the robot by running
    the installed CLI, so they land where the app reads them (``roboemot.paths``). Without
    ``tts`` (cloud voice) the TTS voices are neither counted nor fetched.

    The space needed is what the pinned manifests state for the files not yet verified on the
    robot (a rerun after a partial fetch needs only the rest); with every file verified there is
    nothing to do. Skipped with a warning, not an error, when the filesystem lacks that space
    (same unit on both sides: MiB), when the free space cannot be read, or when the download
    fails twice (``FETCH_ATTEMPTS``): ``roboemot models fetch`` can be run later. Model
    folders the installer creates inside a data folder that is not recorded as its own are
    recorded before the download, so uninstall removes them; inside a recorded data folder they
    go with it. The dry run only reports sizes and free space. Retired model files still on the
    robot are reported with a hint to run ``roboemot models prune``; the installer deletes none.
    """
    label = models_label(vision, tts)
    done, on_disk, retired = verified_models(py)
    if retired:
        ui.warn("Retired model files", f"{-(-retired // MIB)} MiB still on the robot")
        ui.hint("Delete them, on the robot", "roboemot models prune")
    need_mib = -(-models_bytes(vision, done=done, on_disk=on_disk, tts=tts) // MIB)
    folders = model_dirs(remote, vision)
    if need_mib == 0:
        ui.ok("Checked the models", f"the {label} are there and verified: nothing to download")
        return
    free_mib = free_mib_at(remote, folders[0])
    if free_mib is None:
        ui.warn("Skipped the models", f"{label}: cannot read the free space in {folders[0]}")
        ui.hint("Fetch them, on the robot", "roboemot models fetch")
        return
    if free_mib < need_mib:
        ui.warn(
            "Skipped the models",
            f"{free_mib} MiB free in {folders[0]}, the {label} need {need_mib} MiB",
        )
        ui.hint("Free space, then on the robot", "roboemot models fetch")
        if vision:
            ui.hint("Or install without vision", f"{COMMAND} --no-vision")
        return
    if dry_run:
        ui.would("Download the models", f"{label}: {need_mib} MiB needed, {free_mib} MiB free")
        return
    ui.ok("Checked free space", f"{free_mib} MiB free, the {label} need {need_mib} MiB")
    if record.data_dir is None:
        new = [d for d in folders if d not in record.models_dirs and not _exists(remote, d)]
        if new:
            record.models_dirs += new
            write_record(remote, platform.record_path, record)
    args = fetch_args(vision, tts)
    result = py.cli(args, "Downloading the models")
    for _ in range(FETCH_ATTEMPTS - 1):
        if result.ok:
            break
        ui.warn("Download stopped", f"trying once more (it resumes): {last_line(result.output)}")
        result = py.cli(args, "Downloading the models again")
    if not result.ok:
        ui.warn("Skipped the models", f"{label}: the download failed: {last_line(result.output)}")
        ui.hint("Fetch them later, on the robot", "roboemot models fetch")
        return
    ui.ok("Downloaded the models", f"{label} in {', '.join(folders)}")


def remove_models_dirs(ui: Ui, remote: Remote, record: InstallRecord, dry_run: bool) -> None:
    """Delete the recorded model folders, only at the exact paths the fetch would choose and
    only when their ``realpath`` is the same (the data folder's own removal covers the rest)."""
    expected = set(model_dirs(remote, vision=True))
    for folder in record.models_dirs:
        if folder not in expected:
            ui.warn("Model folder left", f"recorded {folder!r} is not one of {sorted(expected)}")
        elif not _exists(remote, folder):
            ui.ok("Model folder gone", f"{folder} is not there")
        elif remote.run(f"realpath {shlex.quote(folder)} 2>/dev/null || true").strip() != folder:
            ui.warn("Model folder left", f"{folder} resolves elsewhere: symlink?")
        else:
            if not dry_run:
                remote.run(f"rm -rf {shlex.quote(folder)}")
            ui.change(dry_run, "Delete a model folder", "Deleted a model folder", folder)


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
) -> list[str]:
    """Store the secrets ``names`` in the robot's secret store (SPEC 8, 12 Data); the names
    stored by this call.

    Each value is typed on this PC and travels only on the SSH channel's standard input, so it
    is in no command line, log or message. A name that the store did not have is recorded (by
    name) before it is written, so uninstall removes exactly it; one that already existed is
    replaced only on request and stays the robot owner's. The dry run only says what it would do
    and runs nothing on the robot.
    """
    _, folder = remote_data_dir(remote)
    stored: list[str] = []
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
        stored.append(name)
    return stored


def set_cloud_voice_secret(
    ui: Ui,
    platform: Platform,
    py: RemotePython,
    remote: Remote,
    record: InstallRecord,
    route: CloudVoiceRoute,
    dry_run: bool,
) -> None:
    """Ask for the cloud-voice route's key when the robot's secret store has none (the value is
    typed on this PC, never an argument: ``set_secrets``); one already there is kept. Without
    it the route is unavailable, and with cloud voice the robot then does not answer by voice:
    said in a warning. The dry run runs nothing on the robot."""
    name = route.secret
    if dry_run:
        _, folder = remote_data_dir(remote)
        ui.would("Ask for a secret", f"{name!r} if the store has none, in {folder}/secrets.json")
        return
    _, folder = remote_data_dir(remote)
    if name in remote.run(robot_cli(py, folder, "list")).split():
        ui.ok("Kept the secret", f"{name}: already in the store")
        return
    if not set_secrets(ui, platform, py, remote, record, [name], dry_run):
        ui.warn(
            "Cloud voice without its key",
            f"the {route.name} route is unavailable and the robot does not answer by voice "
            f"until {name!r} is set",
        )


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


def warn_data_dir_overrides(ui: Ui, remote: Remote) -> None:
    """Warn when the app would keep its data outside the folder the installer records.

    Looks at ``ROBOEMOT_DATA_DIR`` in the SSH user's environment (a ``data_dir`` in the app
    config is checked by ``apply_app_config``). The app's own environment may differ, so this is
    a hint, not a check.
    """
    env = remote.run('printf %s "${ROBOEMOT_DATA_DIR:-}"').strip()
    if env:
        ui.warn("Data may land elsewhere", f"ROBOEMOT_DATA_DIR={env} is set for the SSH user")


# ---------------------------------------------------------------- app config


def app_config_remote_path(remote: Remote) -> str:
    """``app.yaml`` in the robot's data folder (``roboemot.paths.app_config_path`` there)."""
    _, data = remote_data_dir(remote)
    return posixpath.join(data, paths.APP_CONFIG_FILE)


def read_app_config(remote: Remote, path: str) -> dict | None:
    """The app config on the robot, None when there is no file; not YAML or not a mapping is a
    WizardError (the app would refuse it too)."""
    if not _exists(remote, path):
        return None
    raw = remote.run(f"cat {shlex.quote(path)}")
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise WizardError("App config is not valid YAML", f"{path}: {e}; fix or remove it") from e
    if not isinstance(data, dict):
        raise WizardError("App config is not a mapping", f"{path}; fix or remove it")
    return data


def apply_app_config(
    ui: Ui,
    platform: Platform,
    remote: Remote,
    record: InstallRecord,
    wanted: dict[tuple[str, ...], object],
    owned: set[tuple[str, ...]],
    dry_run: bool,
    profile: frozenset[tuple[str, ...]] = frozenset(),
) -> None:
    """Write the installer's keys into ``app.yaml`` in the data folder and keep every other key
    (SPEC 4A, Installer): owner edits survive a reinstall. Comments are not kept. Every file the
    installer writes, created or edited, is recorded (before the write), and uninstall removes it
    whole, including the settings saved from the web UI (owner decision 2026-10-22; no per-key
    restore); editing a file the owner already had says so. A ``data_dir`` in the file is warned
    about. The dry run only says what it would write. The content travels on the SSH channel's
    standard input, never on a command line. ``profile`` are the keys of a profile this run
    sets (``--cloud-voice``): recorded (``app_config_keys``) together with the earlier ones
    before the write and alone after it, so a reinstall without the profile removes them."""
    path = app_config_remote_path(remote)
    existing = read_app_config(remote, path)
    merged = merge_app_config(existing or {}, wanted, owned)
    if (existing or {}).get("data_dir"):
        ui.warn(
            "Data may land elsewhere",
            f"{path} sets data_dir={existing['data_dir']}; uninstall leaves that folder",
        )
    keys = ", ".join(key_text(k, v) for k, v in sorted(wanted.items()))
    removed = sorted(k for k in owned - set(wanted) if config_value(existing or {}, k) is not None)
    if removed and dry_run:
        ui.would("Remove app config keys", ", ".join(key_text(k, None) for k in removed))
    if merged == (existing or {}) and (existing is not None or not merged):
        ui.ok("App config already as asked", f"{path} ({keys or 'no installer keys'})")
        if not dry_run:
            _record_profile_keys(remote, platform, record, profile)
        return
    adopted = existing is not None and record.app_config != path
    if adopted:
        ui.warn(
            "App config was already there",
            f"{path} {'would be' if dry_run else 'is'} recorded; uninstall removes it whole, other "
            "settings and those saved from the web UI included",
        )
    if dry_run:
        ui.would(
            "Write the app config",
            f"{path} ({keys or 'no installer keys'}; other keys kept; uninstall removes the file)",
        )
        return
    if record.app_config != path:
        record.app_config = path
        write_record(remote, platform.record_path, record)
    earlier = {tuple(k) for k in record.app_config_keys}
    _record_profile_keys(remote, platform, record, profile | earlier)
    tmp = f"{path}.tmp"
    remote.run(
        f"umask 077 && mkdir -p {shlex.quote(posixpath.dirname(path))} && "
        f"cat > {shlex.quote(tmp)} && mv {shlex.quote(tmp)} {shlex.quote(path)}",
        stdin=app_config_text(merged),
    )
    _record_profile_keys(remote, platform, record, profile)
    ui.ok("Wrote the app config", f"{path}: {keys or 'installer keys removed'}; other keys kept")
    if removed:
        ui.ok("Removed app config keys", ", ".join(key_text(k, None) for k in removed))


def _record_profile_keys(
    remote: Remote, platform: Platform, record: InstallRecord, keys: frozenset[tuple[str, ...]]
) -> None:
    """Record ``keys`` as the profile keys the installer set (written only when they change)."""
    listed = sorted(list(k) for k in keys)
    if record.app_config_keys != listed:
        record.app_config_keys = listed
        write_record(remote, platform.record_path, record)


def configured_webui_port(remote: Remote) -> int | None:
    """``webui.port`` in the robot's app config when it sets a positive one, else None (the
    file was validated before the install changed anything)."""
    config = read_app_config(remote, app_config_remote_path(remote)) or {}
    webui = config.get("webui")
    port = webui.get("port") if isinstance(webui, dict) else None
    if isinstance(port, int) and not isinstance(port, bool) and port > 0:
        return port
    return None


def verify_app_config(
    remote: Remote, wanted: dict[tuple[str, ...], object], owned: set[tuple[str, ...]]
) -> None:
    """The app config on the robot has every owned key as asked (set or absent)."""
    path = app_config_remote_path(remote)
    found = read_app_config(remote, path) or {}
    for key in sorted(owned):
        value = config_value(found, key)
        if value != wanted.get(key):
            name = ".".join(key)
            raise WizardError(
                "App config not as asked",
                f"{path} has {name}={value}, expected {name}={wanted.get(key)}",
            )


def remove_app_config(ui: Ui, remote: Remote, record: InstallRecord, dry_run: bool) -> None:
    """Delete the app config the installer wrote (created or edited), whole: settings saved from
    the web UI go with it. Only at the exact path it would pick and only if its ``realpath`` is
    the same."""
    if record.app_config is None:
        return
    path = app_config_remote_path(remote)
    if record.app_config != path:
        ui.warn("App config left", f"recorded {record.app_config!r} is not {path}")
    elif not _exists(remote, path):
        ui.ok("App config gone", f"{path} is not there")
    elif remote.run(f"realpath {shlex.quote(path)} 2>/dev/null || true").strip() != path:
        ui.warn("App config left", f"{path} resolves elsewhere: symlink?")
    else:
        if not dry_run:
            remote.run(f"rm -f {shlex.quote(path)}")
        detail = f"{path}, settings saved from the web UI included"
        ui.change(dry_run, "Remove the app config", "Removed the app config", detail)


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
        ui.ok("Environment gone", fate.note)
    elif fate.action == "keep":
        ui.warn("Kept the environment", fate.note)
    elif fate.action == "delete":
        env = platform.env_dir
        if not dry_run:
            remote.run(f"rm -rf {shlex.quote(env)}")
        ui.change(dry_run, "Delete the environment", "Deleted the environment", env)
    return fate.action == "keep"


def finish_refusal(
    platform: Platform, py: RemotePython, record: InstallRecord | None
) -> str | None:
    """Why an incomplete environment may not be finished, or None: it must be the one the
    record says this installer created, and hold nothing beyond the baseline taken when it was
    created (the finish step installs unpinned, which must not touch another app's packages)."""
    env = platform.env_dir
    if record is None or record.env_created != env:
        return (
            f"{env} lacks what {platform.display_name} apps need and was not created by this "
            "installer: repair it with the robot's own software, then run the installer again"
        )
    extra = sorted(set(py.installed()) - set(record.env_baseline))
    if extra:
        return (
            f"{env} is incomplete but holds packages added since it was created "
            f"({', '.join(extra)}): repair it by hand, or uninstall, then run the installer again"
        )
    return None


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
    refusal = finish_refusal(platform, py, record)
    if refusal:
        raise WizardError("Cannot finish the environment", refusal)
    plan = platform.plan_env(remote)
    if dry_run:
        ui.would("Finish the environment", f"{env}, created by an earlier install ({plan})")
        ui.info("Dependency check skipped: the environment is not complete yet.")
        claim_data_dir(ui, platform, remote, record, dry_run=True)
        return False
    ui.warn("Finishing the environment", f"{env}: an earlier install created it and stopped")
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
    warn_data_dir_overrides(ui, remote)
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
            ui.would("Create the environment", f"{platform.env_dir} ({plan})")
            ui.info("Dependency check skipped: the environment does not exist yet.")
            claim_data_dir(ui, platform, remote, record, dry_run=True)
            ui.would("Record the changes", f"{platform.record_path}; nothing installed")
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
        ui.warn("Earlier install interrupted", "its changes are folded into the record")
        record.merge(record.pending_before, before)
    pins = py.constraints(before, set(own))
    if not dry_run and not ui.interactive and not ui.assume_yes:
        # Read-only, before anything is copied or recorded: a shared-package change needs a
        # yes that nobody can type, so stop now rather than after the wheels are on the robot.
        with ui.running("Resolving dependencies…"):
            plan = py.plan_install(third_party_requirements(dirs, extras), pins)
        if not plan.ok:
            ui.require("--yes", False, SHARED_NEEDS_YES)
    if dry_run:
        if REALTIME_EXTRA in extras:
            have = before.get(REALTIME_PACKAGE)
            ui.would("Install the realtime extra", realtime_extra_line(dirs, have))
        reqs = third_party_requirements(dirs, extras)
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
        targets = with_extras(files, extras)
        record.pending_before = before
        write_record(remote, platform.record_path, record)
        result = py.install(targets, pins)
        if not result.ok:
            ui.warn("Shared packages conflict", "the app does not fit the versions on the robot")
            with ui.running("Resolving dependencies…"):
                free = py.plan_install(targets, None)
            if not free.ok:
                raise _resolve_error(free.output)
            _, changed = report_changes(_json(free.output), before)
            for n, c in changed.items():
                ui.kv(n, c)
            question = "Change these shared packages? Uninstall restores them (needs internet)"
            if not ui.confirm(question, False):
                raise WizardError("Install cancelled", "shared packages left as they were")
            result = py.install(targets, None)
            if not result.ok:
                raise WizardError(
                    "Could not install the app on the robot",
                    f"pip install failed: {last_line(result.output)}",
                    output=result.output,
                )
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
    answered = [("--dev", opts.dev)]
    if opts.voice and not opts.cloud_voice:  # without voice or with cloud voice, no Piper
        answered.append(("--piper", opts.piper))
    if opts.voice or opts.vision:  # without both, the models are off without asking
        answered.append(("--models", opts.models))
    if not opts.dry_run:
        answered += [("--start", opts.start), ("--startup-app", opts.startup)]
    for flag, value in answered:
        given = value is not None or opts.yes or ui.assume_yes
        ui.require(flag, given, f"Missing {flag} or --no-{flag[2:]}")


def check_shared_packages(
    ui: Ui,
    platform: Platform,
    remote: Remote,
    dirs: list[Path],
    extras: tuple[str, ...],
    dry_run: bool,
) -> None:
    """Off a terminal without ``--yes``, stop before the first write (stopping the app, clearing
    caches) when the install would have to change shared packages: only ``--yes`` allows that.
    Read-only (a pip resolve); skipped while the environment is missing or incomplete, where
    ``install_packages`` checks again once it is there."""
    if dry_run or ui.interactive or ui.assume_yes:
        return
    py = RemotePython(remote, platform.python, ui)
    if not py.exists() or not platform.env_ready(py):
        return
    pins = py.constraints(py.installed(), set(distribution_names(dirs)))
    with ui.running("Resolving dependencies…"):
        plan = py.plan_install(third_party_requirements(dirs, extras), pins)
    if not plan.ok:
        ui.require("--yes", False, SHARED_NEEDS_YES)


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
        ("piper", opts.piper if opts.voice and not opts.cloud_voice else None),
        ("cloud-voice", opts.cloud_voice),
        ("start", opts.start),
        ("startup-app", opts.startup),
    ):
        if value is not None:
            args.append(f"--{flag}" if value else f"--no-{flag}")
    for flag, value in (("voice", opts.voice), ("vision", opts.vision)):
        if not value:
            args.append(f"--no-{flag}")
    for flag, value in (("models", opts.models), ("clear-caches", opts.clear_caches)):
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


def check_cloud_voice(ui: Ui, opts: Options, platform: Platform) -> None:
    """Refuse a cloud-voice choice that cannot work, before the robot is touched: the flag on a
    platform without a cloud-voice route, or with ``--no-voice`` or ``--piper``. A platform
    without the route needs no answer; off a terminal one with it needs ``--cloud-voice`` or
    ``--no-cloud-voice`` (``-y`` is not an answer: the opt-in sends audio off the robot)."""
    if platform.cloud_voice is None:
        if opts.cloud_voice:
            raise WizardError(
                f"No cloud voice for {platform.display_name}",
                "its installer offers no cloud-voice route",
                _rerun(replace(opts, cloud_voice=None)),
            )
        opts.cloud_voice = False
        return
    if opts.cloud_voice and not opts.voice:
        raise WizardError(f"{CLOUD_VOICE_FLAG} needs voice", "--no-voice turns voice off")
    if opts.cloud_voice and opts.piper:
        raise WizardError(
            f"{CLOUD_VOICE_FLAG} and --piper together",
            "with cloud voice the server speaks and Piper is removed",
        )
    ui.require(
        "--no-cloud-voice",
        opts.cloud_voice is not None,
        f"Missing {CLOUD_VOICE_FLAG} or --no-cloud-voice (-y does not answer it)",
    )


def cloud_voice_text(route: CloudVoiceRoute) -> str:
    """The opt-in text of the cloud-voice profile (SPEC 12: audio leaving the robot is an admin
    opt-in), in plain words."""
    return (
        "Cloud voice: the robot listens without a wake word, and the audio of every utterance "
        f"spoken while a person faces the robot and looks at it leaves the robot for "
        f"{route.provider}, phone calls included. The skip tool only decides whether the robot "
        "answers: the audio is already at the server. The robot has no local voice (Piper is "
        "removed); going back needs a reinstall with --no-cloud-voice."
    )


def choose_cloud_voice(ui: Ui, opts: Options, platform: Platform) -> None:
    """Settle the cloud-voice opt-in (after ``check_cloud_voice``): the flag, else off with
    ``-y`` (it does not imply the opt-in), else the question on the terminal after the privacy
    text, default no."""
    route = platform.cloud_voice
    if route is None or opts.cloud_voice is False:
        return
    if opts.cloud_voice is None and ui.assume_yes:
        opts.cloud_voice = False
        ui.ok("Cloud voice off", f"-y does not turn it on, {CLOUD_VOICE_FLAG} does")
        return
    if opts.cloud_voice is None:
        ui.info(cloud_voice_text(route))
        opts.cloud_voice = ui.confirm(
            "Turn on cloud voice (audio leaves the robot)?", False, flag=CLOUD_VOICE_FLAG
        )
    if opts.cloud_voice:
        opts.piper = False
        ui.ok("Chose cloud voice", f"{route.name} route to {route.provider}; no Piper")


def run_install(opts: Options, ui: Ui, root: Path) -> None:
    ui.steps = INSTALL_STEPS
    check_secret_names(opts.secrets or [])
    preflight(opts, ui)
    ui.title("installer", "dry run, the robot is not modified" if opts.dry_run else "")
    ui.step("Robot")
    host, platform = select_robot(ui, opts, root)
    check_cloud_voice(ui, opts, platform)
    ui.ok("Selected the robot", f"{platform.display_name} at {host}")
    platform.connect(ui, host)
    _check_running_app(opts, ui, platform)

    stopped: list[bool] = []  # whether this run stopped the running app

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
            choose_cloud_voice(ui, opts, platform)
            cloud = cloud_voice_config(platform.cloud_voice) if opts.cloud_voice else {}
            content = ", ".join(
                filter(
                    None,
                    [
                        describe_app_config(dev, app_config_overrides(opts.voice)),
                        ", ".join(key_text(k, v) for k, v in cloud.items()),
                    ],
                )
            )
            ui.ok(
                "Chose the app config",
                f"app.yaml in the data folder: {content or 'no installer keys'}",
            )
            wanted, owned = owned_app_config(root, dev, opts.voice)
            dirs = platform.stage(root, Path(tmp) / "src")

            ui.step("SSH access")

            def free_space() -> str | None:
                read_app_config(remote, app_config_remote_path(remote))  # refuse a broken one now
                check_shared_packages(ui, platform, remote, dirs, extras, opts.dry_run)
                stopped.append(stop_own_app(ui, platform, opts.dry_run) and not opts.dry_run)
                home, _ = remote_data_dir(remote)
                kept = clear_caches(
                    ui,
                    remote,
                    home,
                    platform.clearable_caches,
                    opts.clear_caches,
                    opts.dry_run,
                    (posixpath.dirname(platform.env_dir),),
                )
                if not kept:
                    return None
                return f"kept {', '.join(kept)}: --clear-caches frees it"

            extras = (*platform.extras, *(("vision",) if opts.vision else ()))
            py = check_ssh(ui, platform, remote, min_python(dirs), free_space, extras)

            ui.step("Build")
            with ui.running("Building the wheels…"):
                wheels = build_wheels(dirs, Path(tmp) / "wheels")
            ui.ok("Built the wheels", ", ".join(w.name for w in wheels))

            ui.step("Install (dependency check only)" if opts.dry_run else "Install")
            if opts.vision and opts.dry_run:
                ui.would("Install the vision extra", "OpenCV, ONNX; --no-vision skips it")
            record = install_packages(ui, platform, py, remote, dirs, wheels, opts.dry_run, extras)
            wanted = {**wanted, **cloud}
            owned = owned | set(cloud) | {tuple(k) for k in record.app_config_keys}
            apply_app_config(
                ui, platform, remote, record, wanted, owned, opts.dry_run, frozenset(cloud)
            )
            if not opts.dry_run:
                platform.verify(ui, py)
                verify_app_config(remote, wanted, owned)
            if opts.piper is None:
                opts.piper = opts.voice and ui.confirm(
                    "Install Piper for on-board speech (own virtualenv)?", False, flag="--piper"
                )
            if opts.piper:
                install_piper(ui, platform, remote, record, opts.dry_run)
            if opts.cloud_voice:
                drop_piper_venv(ui, platform, remote, record, opts.dry_run)
            tts = not opts.cloud_voice
            if opts.models is None:
                opts.models = (opts.voice or opts.vision) and ui.confirm(
                    f"Download the {models_label(opts.vision, tts)} "
                    f"(~{models_bytes(opts.vision, tts=tts) / 1e9:.1f} GB at peak while "
                    f"unpacking, ~{models_bytes(opts.vision, peak=False, tts=tts) / 1e9:.1f} GB "
                    "once installed) on the robot?",
                    False,
                    flag="--models",
                )
            warn_voice_gaps(ui, opts)
            if opts.models:
                fetch_models(ui, platform, py, remote, record, opts.vision, opts.dry_run, tts)
            set_operator_pin(ui, py, remote, opts.dry_run)
            names = wanted_secrets(ui, opts)
            route = platform.cloud_voice
            if opts.cloud_voice and route is not None:
                names = [n for n in names if n != route.secret]
                set_cloud_voice_secret(ui, platform, py, remote, record, route, opts.dry_run)
            set_secrets(ui, platform, py, remote, record, names, opts.dry_run)

            ui.step("Start")
            if opts.dry_run:
                running = platform.running_app()
                if running and running != platform.app_name:  # our own stop is counted above
                    ui.would(f"Stop {running}", "only one app runs at a time")
                ui.would(f"Start {platform.app_name}", "after waking the robot")
                blocker = platform.start_blocker()
                if blocker:
                    ui.warn("Start would be refused", blocker)
                ui.kv("Autostart app", platform.autostart() or "none")
                ui.ok("Dry run passed", "the robot was not modified")
                ui.next(_rerun(opts))
                ui.summary()
                return
            blocker = platform.start_blocker()
            if blocker:
                ui.warn(f"Did not start {platform.app_name}", blocker)
                ui.hint("Start it once that is done", "python scripts/run_on_robot.py")
            elif (
                opts.start
                if opts.start is not None
                else ui.confirm(f"Start {platform.app_name} now?", False, flag="--start")
            ):
                platform.start(ui)
                report_app_url(ui, platform, host, configured_webui_port(remote))
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
            ui.hint("Start it later", "python scripts/run_on_robot.py")
            ui.hint("Undo", _rerun(Options(host=host, user=opts.user, uninstall=True)))
            ui.summary()

    try:
        _with_remote(opts, ui, platform, host, body)
    except Exception:
        if any(stopped):
            ui.warn(f"{platform.app_name} is stopped", "it was stopped before installing")
            ui.hint("Start it once the problem is fixed", "python scripts/run_on_robot.py")
        raise


def stop_own_app(ui: Ui, platform: Platform, dry_run: bool) -> bool:
    """Stop the platform's app when it is the one running, before any package changes: a running
    process keeps the old code. Another app is left alone. True when the app was running (and,
    outside a dry run, is now stopped)."""
    if platform.running_app() != platform.app_name:
        return False
    if dry_run:
        ui.would(f"Stop {platform.app_name}", "before installing: it keeps the old code")
        return True
    platform.stop_app()
    ui.ok(f"Stopped {platform.app_name}", "before installing: a running process keeps the old code")
    return True


def report_app_url(ui: Ui, platform: Platform, host: str, port: int | None = None) -> None:
    """Tell where the running app's web UI is, if the platform has one. ``port`` is the
    ``webui.port`` the app config sets, when the caller could read it (the installer, over
    SSH); without it the platform's default is used (``run_on_robot.py`` has no SSH)."""
    url = platform.app_url(host) if port is None else platform.app_url(host, port)
    if url:
        ui.ok("Web UI", url)


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
        env_there = py.exists()
        installed = py.installed() if env_there else {}
        record = read_record(remote, platform.record_path, ui)
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

        if env_there:
            remove_secrets(ui, platform, py, remote, record, dry)
        elif record.secrets and record.data_dir is None:
            ui.warn("Secrets left", f"{', '.join(record.secrets)}: no environment to run")

        ui.step("Packages")
        fate = created_env_fate(platform, py, remote, record)
        if fate.action == "delete":
            # The whole environment goes below: restoring versions inside it first would only
            # need the package index (internet) for nothing.
            env = platform.env_dir
            ui.change(dry, "Remove the app's packages", "Removed the app's packages", f"with {env}")
            plan = UninstallPlan([], {}, {}, {})
        else:
            plan = plan_uninstall(record, installed, py.requires() if env_there else {})
        if not env_there:
            plan.drifted = {}  # nothing to compare against: the environment is gone
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
            py.uninstall(plan.remove)
            py.install_exact(plan.restore)
        # before removal (dry run), the app's modules are expected
        if fate.action != "delete" and env_there and not (dry and plan.remove):
            for module, path in leftovers(py, platform):
                if dry:
                    ui.would("Delete a leftover module", f"{path} ({module})")
                else:
                    remote.run(f"rm -rf {shlex.quote(path)}")
                    ui.warn("Deleted a leftover module", f"{path} ({module})")

        ui.step("Clean up and verify")
        keep_env = remove_created_env(ui, platform, remote, fate, dry)
        remove_piper_venv(ui, remote, record, dry)
        remove_models_dirs(ui, remote, record, dry)
        remove_app_config(ui, remote, record, dry)
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
        verify_uninstall(ui, platform, py, record, plan.restore, keep_env)
        remote.run(f"rm -f {shlex.quote(platform.record_path)}")
        ui.ok("Deleted the install record", platform.record_path)
        ui.ok("Uninstalled", f"{app} from {host}")
        ui.summary()

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
    config = app_config_remote_path(py.remote)
    if record.app_config == config and _exists(py.remote, config):
        problems.append(f"app config {config} still present")
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
        CLOUD_VOICE_FLAG,
        dest="cloud_voice",
        action=argparse.BooleanOptionalAction,
        help="cloud-voice profile: the robot answers by voice through the platform's realtime "
        "route; the audio of every utterance spoken while a person faces the robot and looks at "
        "it leaves the robot (SPEC 12 opt-in). Writes the route and its settings into app.yaml, "
        "asks for the route's key when the store has none, skips Piper and its voices and "
        "removes a recorded Piper venv (default: ask on a terminal; -y does not turn it on)",
    )
    p.add_argument(
        "--clear-caches",
        action=argparse.BooleanOptionalAction,
        help="delete the robot's download caches the platform names (e.g. a uv cache) to "
        "free disk space; uninstall cannot restore them (default: ask on a terminal; -y alone "
        "keeps them)",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="answer yes to every confirmation: dev mode, Piper, the models, start, autostart, "
        "stopping another app, changing shared packages, replacing a PIN or secret (not "
        "clearing caches: only --clear-caches does)",
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
        a.clear_caches,
        a.vision,
        a.models,
        a.voice,
        debug=a.debug,
        version=a.version,
        cloud_voice=a.cloud_voice,
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
    "CloudVoiceRoute",
    "Options",
    "Platform",
    "Ui",
    "WizardError",
    "load_platform",
    "main",
    "report_app_url",
    "select_robot",
    "run_install",
    "run_uninstall",
]
