"""Robot data folders frozen from real installs still load (SPEC 11A, 12; SPEC 4A, Installer).

Each folder under ``tests/compat`` is a robot's data folder as it stands after an install and,
for ``gemini_switch`` and ``dead_fields``, web UI edits: ``app.yaml``, ``webui_settings.json``
and the names (never the values) of the secrets it needs. ``cloud_voice`` and ``local_voice``
were generated with the installer code (``--dev --cloud-voice --timezone Europe/Rome``, and the
same without ``--cloud-voice``); the web files with a ``SettingsEditor`` driven the way the page
drives it (``gemini_switch``: the Gemini steps of ``docs/realtime-voice-setup.md``).

The fixtures are frozen: never edit one. When the installer or the web UI writes something else,
add a new fixture next to the old one (``cloud_voice_v2``): robots set up with the old files
still exist. The drift checks compare key paths only, never values.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import types
import typing
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel
from roboemot import install, paths
from roboemot.affect.clips import load_clip_library
from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import DROPPED_SETTINGS, MIGRATED_SETTINGS
from roboemot.manifest import Manifest, load_manifest
from roboemot.realtime.route import select_provider
from roboemot.realtime.session import DirectSession
from roboemot.robot_platform import load_robot_platform
from roboemot.runtime.host import RuntimeHost
from roboemot.runtime.webui import WebAdmin, restore_settings
from roboemot.secret_store import SecretStore
from roboemot.webui.settings import parse_okey

REPO = Path(__file__).resolve().parents[3]
COMPAT = Path(__file__).resolve().parent / "compat"
MANIFEST = REPO / "configs" / "manifests" / "reachy-mini-wireless.yaml"
PRESET_DIRS = [REPO / "configs" / "presets"]
FIXTURES = ("cloud_voice", "gemini_switch", "local_voice", "dead_fields")
INSTALLED = {"cloud_voice": True, "local_voice": False}  # fixture -> --cloud-voice
CLOUD_FIXTURES = ("cloud_voice", "gemini_switch")
DEAD = "dead_fields"
DEAD_PRESET_DIR = COMPAT / DEAD / "presets"
TIMEZONE = "Europe/Rome"
DUMMY_SECRET = "dummy-not-a-key"
OWNER_ONLY = 0o600
NOT_CONFIG = ("schema_version", "extends")  # file keys that are not settings
DROPPED_LINE = re.compile(r"setting (\S+) ignored: removed from the configuration$")
FROZEN = (
    "the current installer writes other keys than the frozen fixture {name}: add a new fixture "
    "({name}_v2) from the current installer and keep {name}; never edit a frozen fixture"
)


def robot_data(name: str) -> Path:
    """Copy fixture ``name`` into the test's data folder as an install leaves it: ``app.yaml``,
    the web UI overrides file (0600, else the editor refuses it) and every listed secret set to
    a dummy value. Returns the data folder."""
    src, data = COMPAT / name, paths.data_dir()
    shutil.copyfile(src / "app.yaml", paths.app_config_path())
    overrides = paths.settings_overrides_path()
    shutil.copyfile(src / "webui_settings.json", overrides)
    os.chmod(overrides, OWNER_ONLY)
    store = SecretStore.for_data_dir()
    for secret in json.loads((src / "secrets.json").read_text()):
        store.set(secret, DUMMY_SECRET)
    return data


def app_stack(preset: str = "default", preset_dirs: list[Path] = PRESET_DIRS) -> ConfigStack:
    """The config stack the robot app builds at start: the preset plus ``app.yaml`` from the
    data folder (``roboemot_reachymini.app``; ``configs/presets`` is the copy synced into its
    package data)."""
    return ConfigStack.from_files(
        app=paths.app_config_path(), preset=preset, preset_dirs=preset_dirs
    )


def start(stack: ConfigStack) -> tuple[WebAdmin | None, RoboEmotConfig, Manifest]:
    """What the runtime does with ``stack`` before it builds the parts: load the manifest, replay
    the web UI overrides, then add the manifest behaviour with the robot platform's body yaw.
    Returns the editor (None when the replay failed as a whole), the effective config and the
    manifest."""
    cfg = stack.effective()
    manifest = load_manifest(
        MANIFEST, allow_unmeasured_dynamics=cfg.safety.allow_unmeasured_dynamics, safety=cfg.safety
    )
    admin = restore_settings(stack, manifest, lambda: [])
    body_yaw = load_robot_platform(manifest.robot.id).gaze_map().body_yaw
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest), body_yaw=body_yaw
    )
    return admin, stack.effective(), manifest


def platform_route() -> str:
    """The name of the route the installer's cloud-voice profile writes for this robot."""
    robot_id = load_manifest(MANIFEST, allow_unmeasured_dynamics=True).robot.id
    return install.load_platform(robot_id).cloud_voice.name


def installed_app_config(cloud: bool, existing: dict | None = None) -> dict:
    """``app.yaml`` as the current installer writes it over ``existing`` (a fresh robot when
    None): dev mode, voice on, ``TIMEZONE`` and, with ``cloud``, the cloud-voice profile."""
    robot_id = load_manifest(MANIFEST, allow_unmeasured_dynamics=True).robot.id
    platform = install.load_platform(robot_id)
    wanted, owned = install.owned_app_config(REPO, True, True, TIMEZONE)
    extra = install.cloud_voice_config(platform.cloud_voice) if cloud else {}
    return install.merge_app_config(existing or {}, {**wanted, **extra}, owned | set(extra))


def key_paths(config: dict) -> set[tuple[str, ...]]:
    """Every leaf path of an app config, a route entry's fields under ``llm_routes.<name>``."""
    routes = config.get(install.ROUTES_KEY) or []
    found = set(install.leaves({k: v for k, v in config.items() if k != install.ROUTES_KEY}))
    for route in routes:
        found |= {(install.ROUTES_KEY, route["name"], field) for field in route}
    return found


def setting_paths(tree: dict[str, Any]) -> set[str]:
    """Every setting a config file sets, dotted, a keyed-list entry as ``*``: a mapping is walked,
    a list of mappings is walked per entry, anything else is a leaf."""
    found: set[str] = set()

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict) and node:
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, list) and node and all(isinstance(e, dict) for e in node):
            for entry in node:
                walk(entry, f"{prefix}.*")
        else:
            found.add(prefix)

    walk({k: v for k, v in tree.items() if k not in NOT_CONFIG}, "")
    return found


def override_paths(overrides: dict[str, Any]) -> set[str]:
    """Every setting a web UI overrides file sets, in ``setting_paths`` form: a merged-list file
    key's identity becomes ``*``, an added entry is walked field by field."""
    found: set[str] = set()
    for key, value in overrides.items():
        parsed = parse_okey(key)
        if parsed is None:
            found |= setting_paths({key: value}) if isinstance(value, dict) else {key}
            continue
        list_path, _, rest = parsed
        stem = f"{list_path}.*" + (f".{rest}" if rest else "")
        found |= {f"{stem}.{p}" for p in setting_paths(value)} if not rest else {stem}
    return found


def schema_has(model: type[BaseModel], parts: list[str]) -> bool:
    """Whether ``parts`` names a field of ``model``: a model is entered by field name, a list,
    tuple or mapping by any one segment (an entry or key), a union by any of its members. A
    path that ends on a field is found, whatever the field's type."""
    field = model.model_fields.get(parts[0])
    return field is not None and _type_has(field.annotation, parts[1:])


def _type_has(tp: Any, rest: list[str]) -> bool:
    """``schema_has`` below one field of type ``tp``: ``rest`` is what the path still names."""
    if not rest:
        return True
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return schema_has(tp, rest)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType, typing.Annotated):
        return any(_type_has(a, rest) for a in args)
    if origin in (list, tuple, dict) and args:
        return _type_has(args[-1], rest[1:])
    return False


def moved(path: str) -> bool:
    """Whether ``path`` (or a section holding it) is listed as dropped or migrated in
    ``moved_settings.yaml``; ``*`` in either matches any one segment."""
    parts = path.split(".")
    for pattern in (*DROPPED_SETTINGS, *MIGRATED_SETTINGS):
        want = pattern.split(".")
        if len(parts) >= len(want) and all(
            w == "*" or p == "*" or w == p for w, p in zip(want, parts, strict=False)
        ):
            return True
    return False


def fixture_settings(name: str) -> set[str]:
    """Every setting fixture ``name`` sets in ``app.yaml``, its presets and its web UI file."""
    src = COMPAT / name
    found = setting_paths(yaml.safe_load((src / "app.yaml").read_text()))
    for preset in sorted((src / "presets").glob("*.yaml")) if (src / "presets").exists() else []:
        found |= setting_paths(yaml.safe_load(preset.read_text()))
    stored = json.loads((src / "webui_settings.json").read_text())["overrides"]
    return found | override_paths(stored)


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The text of every WARNING or worse record captured so far."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("name", [f for f in FIXTURES if f != DEAD])
def test_fixture_loads_and_replays_every_override(name, caplog):
    data = robot_data(name)
    overrides = paths.settings_overrides_path(str(data))
    before = overrides.read_bytes()
    with caplog.at_level(logging.WARNING, logger="roboemot"):
        admin, _, _ = start(app_stack())
    assert admin is not None
    assert warnings_of(caplog) == []
    assert overrides.read_bytes() == before  # the replay rewrites the file only to drop entries


def test_dead_fields_load(caplog):
    robot_data(DEAD)
    with caplog.at_level(logging.WARNING, logger="roboemot"):
        admin, cfg, _ = start(app_stack())
    assert admin is not None
    for text in warnings_of(caplog):
        found = DROPPED_LINE.search(text)  # a field the schema no longer has is ignored, named
        assert found and moved(found.group(1)), text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert cfg.voice.enabled
    assert [r.name for r in cfg.llm_routes] == ["lan", "vision_cloud"]  # the web route stays


def test_dead_fields_preset_loads(caplog):
    robot_data(DEAD)
    with caplog.at_level(logging.WARNING, logger="roboemot"):
        admin, cfg, _ = start(app_stack(DEAD, [DEAD_PRESET_DIR, *PRESET_DIRS]))
    assert admin is not None
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert cfg.personality == DEAD


@pytest.mark.parametrize("name", CLOUD_FIXTURES)
def test_cloud_route_resolves_and_has_a_provider(name):
    robot_data(name)
    _, cfg, manifest = start(app_stack())
    router = RuntimeHost().make_router(cfg, None)
    route = router.realtime_route(cfg.voice.realtime_route.route)
    assert route is not None
    assert router.key_problem(route) is None
    assert router.auth_headers(route) == {"Authorization": f"Bearer {DUMMY_SECRET}"}
    provider = select_provider(cfg, manifest.robot.id, manifest)
    assert provider is not None
    direct = cfg.voice.realtime_route.mode == "direct"
    assert isinstance(provider, DirectSession) == direct


def test_gemini_switch_takes_effect():
    robot_data("gemini_switch")
    _, cfg, _ = start(app_stack())
    rt = cfg.voice.realtime_route
    route = next(r for r in cfg.llm_routes if r.name == "gemini")
    assert (rt.enabled, rt.mode, rt.route, rt.turns) == (True, "direct", "gemini", "robot")
    assert (route.kind, route.protocol, route.locality) == ("realtime", "gemini_live", "cloud")
    assert (route.api_key, route.voice, route.connect_timeout_s) == (
        "${secret:gemini_key}",
        "Kore",
        15.0,
    )
    assert [r.name for r in cfg.llm_routes] == [platform_route(), "gemini"]


def test_local_voice_is_local():
    robot_data("local_voice")
    _, cfg, _ = start(app_stack())
    assert cfg.voice.enabled and not cfg.voice.realtime_route.enabled
    assert (cfg.voice.backend, cfg.voice.fallback_backend) == ("piper", "piper")
    assert cfg.voice.asr.models  # Vosk, the local recogniser
    assert cfg.llm_routes == []


@pytest.mark.parametrize("name", sorted(INSTALLED))
def test_installer_still_writes_the_fixture_keys(name):
    frozen = yaml.safe_load((COMPAT / name / "app.yaml").read_text())
    assert key_paths(installed_app_config(INSTALLED[name])) == key_paths(frozen), FROZEN.format(
        name=name
    )


@pytest.mark.parametrize("name", sorted(INSTALLED))
def test_reinstall_leaves_the_fixture_as_it_is(name):
    frozen = yaml.safe_load((COMPAT / name / "app.yaml").read_text())
    assert installed_app_config(INSTALLED[name], frozen) == frozen, FROZEN.format(name=name)


@pytest.mark.parametrize("name", FIXTURES)
def test_every_fixture_key_is_a_setting_or_moved(name):
    unknown = sorted(
        p
        for p in fixture_settings(name)
        if not schema_has(RoboEmotConfig, p.split(".")) and not moved(p)
    )
    assert unknown == [], "not in the schema and not in moved_settings.yaml dropped or migrated"
