"""The voice provider a robot speaks through, as the installer sees it (SPEC 4A, Installer; 8, 12).

The installer runs on the PC. It reads the robot's ``app.yaml`` and the web UI's saved settings
(``webui_settings.json``) over SSH and builds the effective configuration here the way the app
does at start: the default preset, ``app.yaml``, then the saved settings replayed by the web
UI's own editor. The Settings voice switch (``webui.voice_switch``) then names the choice, with
the private keys compared (the installer runs as the owner). A choice the web settings make,
other than the one ``app.yaml`` alone gives, is the web UI's: a reinstall keeps it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..cloud_voice import ROUTES_KEY
from ..config import ConfigStack
from ..config.schema import LLMRoute
from ..llm.pseudonyms import route_is_local
from ..redact import origin
from ..webui import voice_switch as vs
from ..webui.robot_section import VoiceProfile, load_settings_plugin, platform_voice
from ..webui.settings import FILE_MODE, SettingsEditor, parse_okey
from ..webui.strings import DEFAULT_LANGUAGE

PRESET = "default"  # the preset the robot app starts with
PRESET_DIRS = Path("configs/presets")  # in the checkout; the robot's package holds a synced copy
LOCAL_LABEL = "local voice"
LOCAL = vs.LOCAL


@dataclass(frozen=True)
class RobotVoice:
    """The robot's voice choice: ``current`` with the web settings (a choice id or ``custom``),
    ``app_only`` from ``app.yaml`` alone, the ``label`` of ``current``, whether it is local,
    the platform route's label (``platform_label``, None without one), ``switch_back``: the
    saved setting keys to drop so the platform's choice comes back, and ``audio_to``: who
    receives the audio of ``current`` (the plugin's label or the route's host; None for local
    or custom)."""

    current: str
    app_only: str
    label: str
    local: bool
    platform_label: str | None
    switch_back: tuple[str, ...]
    audio_to: str | None = None

    @property
    def from_web(self) -> bool:
        """Whether the web settings choose a provider of their own: a known choice that differs
        from what ``app.yaml`` alone gives (``custom`` is a hand mix, never a provider)."""
        return self.current not in (vs.CUSTOM, self.app_only)


def robot_voice(
    root: Path, robot_id: str, app_config: dict, settings: dict[str, Any]
) -> RobotVoice | None:
    """The voice choice of a robot whose ``app.yaml`` is ``app_config`` and whose saved web
    settings file is ``settings`` (the whole file: overrides, acks, keys). Built in a private
    temporary folder that is removed at the end; the robot's files are never written. None when
    the configuration cannot be built (an older or broken file): the caller falls back."""
    try:
        with tempfile.TemporaryDirectory(prefix="roboemot-voice-") as tmp, _quiet():
            app_path, store = Path(tmp) / "app.yaml", Path(tmp) / "webui_settings.json"
            app_path.write_text(yaml.safe_dump(app_config))
            fd = os.open(store, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
            with os.fdopen(fd, "w") as handle:
                handle.write(json.dumps(settings))
            stack = ConfigStack.from_files(
                app=app_path, preset=PRESET, preset_dirs=[root / PRESET_DIRS]
            )
            plugin = load_settings_plugin(robot_id)
            alone = stack.effective()
            SettingsEditor(stack, store=store, plugin=plugin).restore()
            cfg = stack.effective()
            return _judged(plugin, alone, cfg, settings)
    except Exception:
        return None


def _judged(plugin: Any, alone: Any, cfg: Any, settings: dict[str, Any]) -> RobotVoice:
    """``robot_voice`` once both configurations are built: ``alone`` from ``app.yaml``, ``cfg``
    with the saved settings replayed. Runs inside ``robot_voice``'s guard, so a plugin or data
    error gives None there (the installer then keeps its older checks), never a crash of the
    install."""
    profile = platform_voice(plugin, cfg, DEFAULT_LANGUAGE)
    options = vs.choices(cfg, profile)
    current = vs.current(cfg, options, revealed=True)
    alone_profile = platform_voice(plugin, alone, DEFAULT_LANGUAGE)
    app_only = vs.current(alone, vs.choices(alone, alone_profile), revealed=True)
    choice = next((c for c in options if c.id == current), None)
    return RobotVoice(
        current,
        app_only,
        _label(choice, profile.label if profile else None),
        choice is not None and choice.kind == vs.KIND_LOCAL,
        profile.label if profile else None,
        _switch_back(settings.get("overrides") or {}, profile),
        choice.audio_to if choice is not None else None,
    )


def platform_choice(route: str) -> str:
    """The choice id of the platform's route called ``route``, as the Settings voice switch
    names it (``route:<name>``): the choice a cloud-voice run sets up. A run without cloud
    voice sets up ``LOCAL``. Only the id: whether the route exists on the robot is not
    judged here."""
    return vs.ROUTE_CHOICE + route


def without(settings: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """The web settings file ``settings`` without the saved settings ``keys``, their
    acknowledged endpoints (an ack under a removed key or a field of it) and, for a removed
    whole route entry, the key bound to that route from the web; every other entry and the
    format version stay as they are."""
    out = dict(settings)
    out["overrides"] = {k: v for k, v in (settings.get("overrides") or {}).items() if k not in keys}
    acks = {
        k: v
        for k, v in (settings.get("acks") or {}).items()
        if not any(k == key or k.startswith(key + ".") for key in keys)
    }
    names = {_route_name(k) for k in keys if _route_entry(k)}
    bound = {k: v for k, v in (settings.get("keys") or {}).items() if k not in names}
    for field, value in (("acks", acks), ("keys", bound)):
        out.pop(field, None)
        if value:
            out[field] = value
    return out


@dataclass(frozen=True)
class RouteCopies:
    """The saved settings that hold one route's name (a route added from the web UI, or a field
    of it), against the entry the installer writes: ``same`` (whole entries equal to it once
    both are read as a route, so a field at its default counts as equal), ``moved`` (the audio
    would go to another origin) and ``changed`` (same origin, other fields), with the names of
    the fields that differ (``fields``, never a value)."""

    same: tuple[str, ...] = ()
    moved: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()


def route_copies(settings: dict[str, Any], entry: dict[str, Any]) -> RouteCopies:
    """The saved settings of ``settings`` that hold the route ``entry``'s name, sorted into
    ``RouteCopies``. A whole entry's key bound from the web (``keys``) counts as its
    ``api_key``; a copy that is not a valid route differs in every field. A route of another
    name is never listed."""
    overrides, bound = settings.get("overrides") or {}, settings.get("keys") or {}
    want, home = _route_dump(entry), origin(entry["base_url"])
    same, moved, changed, fields = [], [], [], set()
    for key, value in overrides.items():
        if _route_name(key) != entry["name"]:
            continue
        rest = (parse_okey(key) or ("", {}, ""))[2]
        if rest:
            target = origin(value) if rest == "base_url" and isinstance(value, str) else home
            (moved if target != home else changed).append(key)
            fields.add(rest)
            continue
        whole = value if isinstance(value, dict) else {}
        got = _route_dump({**whole, "api_key": bound.get(entry["name"], whole.get("api_key"))})
        if got == want:
            same.append(key)
            continue
        url = whole.get("base_url")
        (moved if not isinstance(url, str) or origin(url) != home else changed).append(key)
        fields |= {f for f in want if got is None or got.get(f) != want[f]}
    return RouteCopies(*(tuple(sorted(x)) for x in (same, moved, changed, fields)))


def _route_dump(entry: dict[str, Any]) -> dict[str, Any] | None:
    """``entry`` read as a configured route (``LLMRoute``, defaults filled in, ``locality``
    resolved to local or cloud as the router reads it), or None when it is not a valid one:
    two entries that say the same route compare equal this way."""
    try:
        route = LLMRoute.model_validate(entry)
    except ValueError:
        return None
    resolved = "local" if route_is_local(route) else "cloud"
    return {**route.model_dump(mode="json"), "locality": resolved}


def _route_name(key: str) -> str | None:
    """The route name a saved setting addresses (``llm_routes[{"name": ...}]`` or a field of
    it), else None: a key of any other setting or list. Read from the file key alone, the way
    the web UI's replay reads it."""
    found = parse_okey(key)
    if found is None or found[0] != ROUTES_KEY:
        return None
    name = found[1].get("name")
    return name if isinstance(name, str) else None


def _route_entry(key: str) -> bool:
    """Whether a saved setting is a whole route entry (one added from the web UI), not the
    edit of one field of a route. Read from the file key alone, like ``_route_name``; a key of
    any other list or setting is not one."""
    found = parse_okey(key)
    return found is not None and found[0] == ROUTES_KEY and not found[2]


def settings_text(settings: dict[str, Any]) -> str:
    """The web settings file as the web UI writes it (``OverridesFile.save``: two-space JSON,
    keys in the order they were saved, so entries added to a merged list keep their order), so a
    file the installer rewrote reads the same to the app and to a person.
    The text holds the saved values: it travels only on the SSH channel's standard input,
    never on a command line or in a report."""
    return json.dumps(settings, indent=2)


def _switch_back(overrides: dict[str, Any], profile: VoiceProfile | None) -> tuple[str, ...]:
    """Every saved switch key (``voice_profiles.yaml`` ``keys``, whatever its value): dropping
    them all gives the voice keys back to ``app.yaml``, where the cloud-voice profile writes and
    records them, so a later ``--no-cloud-voice`` removes them too. A key left in the web file
    would sit outside the record. Empty without a platform route."""
    if profile is None:
        return ()
    keys = set(vs.profiles().keys)
    return tuple(sorted(k for k in overrides if k in keys))


def _label(choice: vs.Choice | None, platform_label: str | None) -> str:
    """A choice as the installer names it: the plugin's label for the platform route, the
    route's name for any other, ``local voice`` for local."""
    if choice is None:
        return vs.CUSTOM
    if choice.kind == vs.KIND_LOCAL:
        return LOCAL_LABEL
    if choice.kind == vs.KIND_PLATFORM and platform_label:
        return platform_label
    return f"the {choice.route} route"


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence the app's own log lines while the configuration is built on the PC: a dropped
    entry is the app's to report at its next start, not the installer's."""
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)
