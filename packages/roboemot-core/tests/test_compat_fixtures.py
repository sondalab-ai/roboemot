"""Robot data folders frozen from real installs still load (SPEC 11A, 12; SPEC 4A, Installer).

Each folder under ``tests/compat`` is a robot's data folder as it stands after an install and,
for ``gemini_switch``, web UI edits: ``app.yaml``, ``webui_settings.json`` and the names (never
the values) of the secrets it needs. ``cloud_voice`` and ``local_voice`` were generated with the
installer code (``--dev --cloud-voice --timezone Europe/Rome``, and the same without
``--cloud-voice``); ``gemini_switch`` with a ``SettingsEditor`` driven through the Gemini steps
of ``docs/realtime-voice-setup.md``. Later slices change the config: these files must not change
with them, so an existing robot keeps working.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import pytest
import yaml
from roboemot import install, paths
from roboemot.affect.clips import load_clip_library
from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.manifest import load_manifest
from roboemot.runtime.webui import WebAdmin, restore_settings
from roboemot.secret_store import SecretStore

REPO = Path(__file__).resolve().parents[3]
COMPAT = Path(__file__).resolve().parent / "compat"
MANIFEST = REPO / "configs" / "manifests" / "reachy-mini-wireless.yaml"
PRESET_DIRS = [REPO / "configs" / "presets"]
FIXTURES = ("cloud_voice", "gemini_switch", "local_voice", "dead_fields")
INSTALLED = {"cloud_voice": True, "local_voice": False}  # fixture -> --cloud-voice
TIMEZONE = "Europe/Rome"
DUMMY_SECRET = "dummy-not-a-key"
OWNER_ONLY = 0o600


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


def start(stack: ConfigStack) -> tuple[WebAdmin | None, RoboEmotConfig]:
    """What the runtime does with ``stack`` before it builds the parts: load the manifest, replay
    the web UI overrides, then add the manifest behaviour. Returns the editor (None when the
    replay failed as a whole) and the effective config."""
    cfg = stack.effective()
    manifest = load_manifest(
        MANIFEST, allow_unmeasured_dynamics=cfg.safety.allow_unmeasured_dynamics, safety=cfg.safety
    )
    admin = restore_settings(stack, manifest, lambda: [])
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest), body_yaw=None
    )
    return admin, stack.effective()


def key_paths(config: dict) -> set[tuple[str, ...]]:
    """Every leaf path of an app config, a route entry's fields under ``llm_routes.<name>``."""
    routes = config.get(install.ROUTES_KEY) or []
    found = set(install.leaves({k: v for k, v in config.items() if k != install.ROUTES_KEY}))
    for route in routes:
        found |= {(install.ROUTES_KEY, route["name"], field) for field in route}
    return found


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_loads_and_replays_every_override(name, caplog):
    data = robot_data(name)
    overrides = paths.settings_overrides_path(str(data))
    before = overrides.read_bytes()
    with caplog.at_level(logging.WARNING, logger="roboemot"):
        admin, cfg = start(app_stack())
    assert admin is not None
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert overrides.read_bytes() == before  # the replay rewrites the file only to drop entries
    assert paths.data_dir(cfg.data_dir) == data


def test_gemini_switch_takes_effect():
    robot_data("gemini_switch")
    _, cfg = start(app_stack())
    rt = cfg.voice.realtime_route
    route = next(r for r in cfg.llm_routes if r.name == "gemini")
    assert (rt.enabled, rt.mode, rt.route, rt.turns) == (True, "direct", "gemini", "robot")
    assert (route.kind, route.protocol, route.locality) == ("realtime", "gemini_live", "cloud")
    assert (route.api_key, route.voice, route.connect_timeout_s) == (
        "${secret:gemini_key}",
        "Kore",
        15.0,
    )
    assert [r.name for r in cfg.llm_routes] == ["pollen", "gemini"]


def test_local_voice_is_local():
    robot_data("local_voice")
    _, cfg = start(app_stack())
    assert cfg.voice.enabled and not cfg.voice.realtime_route.enabled
    assert (cfg.voice.backend, cfg.voice.fallback_backend) == ("piper", "piper")
    assert cfg.voice.asr.models  # Vosk, the local recogniser
    assert cfg.llm_routes == []


def test_dead_fields_preset_loads():
    robot_data("dead_fields")
    _, cfg = start(app_stack("dead_fields", [COMPAT / "dead_fields" / "presets", *PRESET_DIRS]))
    assert cfg.personality == "dead_fields"


@pytest.mark.parametrize("name", sorted(INSTALLED))
def test_installer_still_writes_the_fixture_keys(name):
    platform = install.load_platform(
        load_manifest(MANIFEST, allow_unmeasured_dynamics=True).robot.id
    )
    wanted, owned = install.owned_app_config(REPO, True, True, TIMEZONE)
    cloud = install.cloud_voice_config(platform.cloud_voice) if INSTALLED[name] else {}
    written = install.merge_app_config({}, {**wanted, **cloud}, owned | set(cloud))
    frozen = yaml.safe_load((COMPAT / name / "app.yaml").read_text())
    assert key_paths(written) == key_paths(frozen)
