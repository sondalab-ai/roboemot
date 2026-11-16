"""The voice-provider switch of the Settings page (SPEC 8, 11A, 12).

One choice per way the robot can speak: "local" (Piper and Vosk on the robot) and one per
``kind: realtime`` route of the configuration. A choice is a full set of voice keys
(``voice_profiles.yaml``): every choice writes the same keys, so a switch never depends on the
state before it, and the page saves it as one batch (``SettingsEditor.change_many``). The robot
platform's own route takes its keys from the robot's settings plugin (``VoiceProfile``); any
other route gets the core direct-mode keys, with the extra ones of its protocol.

No config key holds the choice: the current one is computed from the effective configuration,
so a robot set up before the switch existed shows its choice as it is, or ``custom``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import get_path
from roboemot.redact import endpoint, is_private, origin
from roboemot.secret_store import SECRET_REF

from .robot_section import VoiceProfile

log = logging.getLogger(__name__)

PROFILES_FILE = Path(__file__).with_name("voice_profiles.yaml")
LOCAL = "local"
CUSTOM = "custom"
ROUTE_CHOICE = "route:"  # a route's choice id is this plus the route name
REALTIME_KIND = "realtime"
KIND_LOCAL, KIND_PLATFORM, KIND_DIRECT = "local", "platform", "direct"
REALTIME_ON = "voice.realtime_route.enabled"
ROUTE_KEY = "voice.realtime_route.route"
SECRET_MISSING = "secret_missing"
MODELS_MISSING = "models_missing"
PIPER_MISSING = "piper_missing"


@dataclass(frozen=True)
class Profiles:
    """``voice_profiles.yaml``: the keys every choice writes, the keys that decide the current
    choice (``match``, plus ``match_realtime`` for a choice with the realtime route on, less
    ``unmatched_direct`` for a direct route), and the values of the core choices."""

    keys: tuple[str, ...]
    match: tuple[str, ...]
    match_realtime: tuple[str, ...]
    unmatched_direct: tuple[str, ...]
    common: dict[str, Any]
    local: dict[str, Any]
    direct: dict[str, Any]
    protocols: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Choice:
    """One choice of the switch: its ``id`` (``local`` or ``route:<name>``), its ``kind``
    (local, platform or direct), the route and its protocol, the plugin's label for the platform
    route, the full ``values`` it writes, who receives the audio (``audio_to``: the plugin's
    label or the route's host), and whether it can be picked now (``reason`` when not, with the
    missing ``secret`` name for ``secret_missing``)."""

    id: str
    kind: str
    values: dict[str, Any]
    route: str | None = None
    protocol: str | None = None
    label: str | None = None
    audio_to: str | None = None
    available: bool = True
    reason: str | None = None
    secret: str | None = None


@cache
def profiles() -> Profiles:
    """The switch's data file, read once. Every key a section names must be one of ``keys``,
    and so must every ``match`` key: a data error fails loud at the first use."""
    raw = yaml.safe_load(PROFILES_FILE.read_text())
    keys = tuple(raw["keys"])
    data = Profiles(
        keys,
        tuple(raw["match"]),
        tuple(raw["match_realtime"]),
        tuple(raw["unmatched_direct"]),
        dict(raw["common"]),
        dict(raw["local"]),
        dict(raw["direct"]),
        {name: dict(values) for name, values in raw["protocols"].items()},
    )
    named = [*data.match, *data.match_realtime, *data.unmatched_direct]
    named += [*data.common, *data.local, *data.direct]
    named += [k for values in data.protocols.values() for k in values]
    unknown = sorted(set(named) - set(keys))
    if unknown:
        raise ValueError(f"{PROFILES_FILE.name}: keys outside the key list: {unknown}")
    return data


@cache
def schema_defaults() -> dict[str, Any]:
    """The schema default of every switch key: the value a choice writes for a key it does not
    name (the schema, not the library defaults or a preset, so a choice is the same on every
    robot)."""
    defaults = RoboEmotConfig()
    return {key: get_path(defaults, key) for key in profiles().keys}


def full(*layers: dict[str, Any]) -> dict[str, Any]:
    """Every switch key with its value: the schema default, then each of ``layers`` in turn
    (a later one wins). A layer key outside the key list is not written (logged, by name)."""
    data = profiles()
    values = dict(schema_defaults())
    for layer in layers:
        for key, value in layer.items():
            if key in values:
                values[key] = value
            else:
                log.warning("voice switch: %s is not a switch key, ignored", key)
    return {key: values[key] for key in data.keys}


def choices(cfg: RoboEmotConfig, platform: VoiceProfile | None) -> list[Choice]:
    """Local first, then one choice per ``kind: realtime`` route in configuration order: the
    platform's route (``platform.route`` at the platform's own origin) with the plugin's keys,
    any other route in direct mode with its protocol's extra keys. Availability is not judged
    here (``judged``)."""
    data = profiles()
    out = [Choice(LOCAL, KIND_LOCAL, full(data.common, data.local))]
    for route in cfg.llm_routes:
        if route.kind != REALTIME_KIND:
            continue
        named = {ROUTE_KEY: route.name}
        if platform is not None and is_platform(route.name, route.base_url, platform):
            kind, label, audio_to = KIND_PLATFORM, platform.label, platform.label
            values = full(data.common, platform.keys, named)
        else:
            kind, label, audio_to = KIND_DIRECT, None, endpoint(route.base_url) or route.name
            extra = data.protocols.get(route.protocol, {})
            values = full(data.common, data.direct, extra, named)
        choice_id = ROUTE_CHOICE + route.name
        out.append(Choice(choice_id, kind, values, route.name, route.protocol, label, audio_to))
    return out


def is_platform(name: str, base_url: str, platform: VoiceProfile) -> bool:
    """Whether the route ``name`` at ``base_url`` is the platform's own: the same name and the
    same origin. A route that only borrows the name points elsewhere and runs in direct mode,
    never with the platform's mode or label."""
    if name != platform.route:
        return False
    mine = origin(platform.base_url)
    return bool(mine) and origin(base_url) == mine


def match_keys(choice: Choice, *, revealed: bool) -> list[str]:
    """The keys that decide whether the configuration is ``choice`` (``matches``): ``match``,
    plus ``match_realtime`` with the realtime route on, less ``unmatched_direct`` for a direct
    route (an opt-in that cannot act there), less the private keys without ``revealed``."""
    data = profiles()
    keys = list(data.match)
    if choice.values[REALTIME_ON]:
        keys += data.match_realtime
    if choice.kind == KIND_DIRECT:
        keys = [k for k in keys if k not in data.unmatched_direct]
    if not revealed:
        keys = [k for k in keys if not is_private(k)]
    return keys


def matches(cfg: RoboEmotConfig, choice: Choice, *, revealed: bool) -> bool:
    """Whether the effective ``cfg`` is ``choice``: every ``match`` key equal, and with the
    realtime route on in the choice every ``match_realtime`` key too. The tuning keys (timeouts,
    tools, wake policy, sound direction, turns) do not decide it. Without ``revealed`` (no PIN)
    the private keys (the privacy opt-ins) are not compared, so the answer never shows them."""
    keys = match_keys(choice, revealed=revealed)
    return all(get_path(cfg, key) == choice.values[key] for key in keys)


def current(cfg: RoboEmotConfig, options: list[Choice], *, revealed: bool) -> str:
    """The id of the first choice ``cfg`` is (``matches``), else ``custom``. Two choices never
    both match: they differ in the realtime switch or the route name."""
    return next((c.id for c in options if matches(cfg, c, revealed=revealed)), CUSTOM)


def judged(
    cfg: RoboEmotConfig, choice: Choice, known: set[str] | None, local_problem: str | None
) -> Choice:
    """``choice`` with its availability: local cannot speak with ``local_problem`` (the voice
    models or the Piper interpreter missing); a route whose ``${secret:name}`` key is not in the
    secret store (``known``, ``None`` when it cannot be read: then not judged) cannot answer."""
    if choice.kind == KIND_LOCAL:
        return choice if local_problem is None else _unavailable(choice, local_problem)
    route = next(r for r in cfg.llm_routes if r.name == choice.route)
    ref = SECRET_REF.match(route.api_key)
    if ref is None or known is None or ref.group(1) in known:
        return choice
    return _unavailable(choice, SECRET_MISSING, ref.group(1))


def _unavailable(choice: Choice, reason: str, secret: str | None = None) -> Choice:
    """A copy of ``choice`` that cannot be picked now, for ``reason``."""
    return replace(choice, available=False, reason=reason, secret=secret)


def voice_switch(
    cfg: RoboEmotConfig,
    platform: VoiceProfile | None,
    known: set[str] | None,
    local_problem: str | None,
    *,
    revealed: bool,
) -> dict[str, Any]:
    """The switch as the API sends it: ``value`` (a choice id or ``custom``), ``revealed`` and
    ``choices``, each with ``id``, ``kind``, ``route``, ``protocol``, ``label``, ``audio_to``,
    ``available``, ``reason``, ``secret``, ``private_match`` (the private keys that decide it
    once revealed) and ``changes`` (every key as ``{path, value}``, the body of the batch preview
    and save: the choice's own values, the same on every robot, never the current ones). Without
    ``revealed`` (no PIN) ``value`` is matched without the private keys (``matches``)."""
    options = choices(cfg, platform)
    return {
        "value": current(cfg, options, revealed=revealed),
        "revealed": revealed,
        "choices": [
            {
                "id": c.id,
                "kind": c.kind,
                "route": c.route,
                "protocol": c.protocol,
                "label": c.label,
                "audio_to": c.audio_to,
                "available": c.available,
                "reason": c.reason,
                "secret": c.secret,
                "private_match": [
                    k
                    for k in match_keys(c, revealed=True)
                    if k not in match_keys(c, revealed=False)
                ],
                "changes": [{"path": k, "value": v} for k, v in c.values.items()],
            }
            for c in (judged(cfg, c, known, local_problem) for c in options)
        ],
    }
