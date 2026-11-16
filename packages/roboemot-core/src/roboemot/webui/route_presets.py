"""Route presets of the Models tab (SPEC 8, 11A, 12): ready-made fields for the add card.

Core's presets are data (``route_presets.yaml``); the robot platform's own route comes from its
settings plugin (``RoutePresetted``). A preset holds the route fields and the name of the secret
its key is stored under, never a key. One whose route name is already configured is not offered,
so a route the installer wrote is never shadowed by a copy from the web UI.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from roboemot.config.schema import LLMRoute
from roboemot.secret_store import NAME as SECRET_NAME

from .robot_section import RoutePreset
from .strings import DEFAULT_LANGUAGE, language_for

log = logging.getLogger(__name__)

PRESETS_FILE = Path(__file__).with_name("route_presets.yaml")
# what a preset may fill in: the add card's fields, never the key
PRESET_FIELDS = frozenset(
    {"name", "kind", "protocol", "base_url", "model", "locality", "connect_timeout_s"}
)
SCALARS = (str, int, float, bool)  # with None: the JSON scalars a preset field may hold
REALTIME = "realtime"


@cache
def core_presets() -> tuple[dict[str, Any], ...]:
    """The presets of ``route_presets.yaml`` as written: ``id``, ``label`` (one text per page
    language), ``secret`` and ``fields``. Read once; a broken file fails the tests, not the
    page, and core never builds one by hand."""
    raw = yaml.safe_load(PRESETS_FILE.read_text())
    return tuple(raw["presets"])


def preset_ok(preset: RoutePreset) -> bool:
    """Whether ``preset`` can be offered: its fields are add-card fields only (no key) with
    JSON scalar values and a text ``name``, ``connect_timeout_s`` only on a realtime route, its
    secret is a valid secret name and the fields with that key make a valid ``LLMRoute``. A
    preset that fails is dropped with a log line that names its id only."""
    fields = preset.fields
    if (
        not set(fields) <= PRESET_FIELDS
        or not all(v is None or isinstance(v, SCALARS) for v in fields.values())
        or not isinstance(fields.get("name"), str)
        or ("connect_timeout_s" in fields and fields.get("kind") != REALTIME)
        or SECRET_NAME.fullmatch(preset.secret) is None
    ):
        log.warning("route preset %s dropped: not add-card fields", preset.id)
        return False
    try:
        LLMRoute.model_validate({**fields, "api_key": f"${{secret:{preset.secret}}}"})
    except ValidationError:
        log.warning("route preset %s dropped: not a valid route", preset.id)
        return False
    return True


def labelled(raw: dict[str, Any], language: str) -> RoutePreset:
    """One core preset of the file as a ``RoutePreset`` in the page ``language`` (English when
    the file has no label in it). The fields are a copy: the cached file data stays as read."""
    labels = raw["label"]
    label = labels.get(language_for(language), labels[DEFAULT_LANGUAGE])
    return RoutePreset(raw["id"], label, dict(raw["fields"]), raw["secret"])


def route_presets(
    configured: set[str], platform: list[RoutePreset], language: str
) -> list[dict[str, Any]]:
    """The presets the add card offers, the platform's first: ``{id, label, fields, secret}``.
    A preset whose route name is in ``configured`` (the effective ``llm_routes`` names), whose
    id or name an earlier preset already has, or that fails ``preset_ok`` is left out. Core's
    ids are its own: a platform preset that reuses one is dropped, never core's."""
    out: list[dict[str, Any]] = []
    core = [labelled(raw, language) for raw in core_presets()]
    reserved = {p.id for p in core}
    ids: set[str] = set()
    names = set(configured)
    candidates = [(p, False) for p in platform] + [(p, True) for p in core]
    for preset, own in candidates:
        try:
            offered = (
                preset.id not in ids
                and (own or preset.id not in reserved)
                and preset_ok(preset)
                and preset.fields["name"] not in names
            )
        except Exception as exc:  # one malformed plugin preset must never fail the list
            log.warning("route preset dropped: %s", type(exc).__name__)
            continue
        if not offered:
            continue
        ids.add(preset.id)
        names.add(preset.fields["name"])
        out.append(
            {
                "id": preset.id,
                "label": preset.label,
                "fields": dict(preset.fields),
                "secret": preset.secret,
            }
        )
    return out
