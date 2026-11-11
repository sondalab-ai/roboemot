"""SPEC 11A: ``field_meta`` memoizes its schema walk per (model, field) without changing a result.

The guardrails (``guarded_leaves``, ``violations``) and the settings catalogue call it for every
leaf, so it is memoized; the memo is keyed by the schema only, so a path with any key or index
(route names, moods, list indices) never grows it.
"""

from __future__ import annotations

from typing import Any, get_origin

from roboemot.config import stack as stack_module
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import (
    INHERITED_META,
    _inner_model,
    field_meta,
    flatten,
    strip_optional,
)


def reference(dotted: str) -> dict[str, Any]:
    """The schema walk without a memo, as ``field_meta`` was written before it had one."""
    parts = dotted.split(".")
    model: Any = RoboEmotConfig
    meta: dict[str, Any] = {}
    inherited: dict[str, Any] = {}
    i = 0
    while i < len(parts):
        fields = getattr(model, "model_fields", None)
        if not fields or parts[i] not in fields:
            return {}
        info = fields[parts[i]]
        extra = info.json_schema_extra
        meta = {**inherited, **(extra if isinstance(extra, dict) else {})}
        inherited = {k: v for k, v in meta.items() if k in INHERITED_META and v}
        annotation = strip_optional(info.annotation)
        i += 1
        if get_origin(annotation) in (dict, list):
            i += 1
        model = _inner_model(annotation)
        if model is None:
            return meta if i >= len(parts) else {}
    return meta


def every_path() -> list[str]:
    tree = RoboEmotConfig().model_dump(mode="python")
    paths = set(flatten(tree))
    for path in list(paths):  # every prefix too: containers carry metadata of their own
        parts = path.split(".")
        paths.update(".".join(parts[:n]) for n in range(1, len(parts)))
    return sorted(paths)


def test_every_path_of_the_schema_gives_what_the_plain_walk_gives():
    paths = every_path() + [
        "nope",
        "safety.nope",
        "voice.moods.joy.rate",
        "llm_routes.0.api_key",
        "llm_routes.cloud.base_url",
        "voice.moods.joy.rate.deeper",
        "",
    ]
    assert len(paths) > 200
    for path in paths:
        assert field_meta(path) == reference(path), path


def test_distinct_keys_and_indices_never_grow_the_memo():
    field_meta("llm_routes.warm.api_key")
    field_meta("voice.moods.warm.rate")
    size = stack_module._field_step.cache_info().currsize
    for n in range(500):
        field_meta(f"llm_routes.route{n}.api_key")
        field_meta(f"voice.moods.mood{n}.rate")
        field_meta(f"llm_routes.{n}.base_url")
        field_meta(f"unknown{n}.x")
        field_meta(f"safety.unknown{n}")
    assert stack_module._field_step.cache_info().currsize == size


def test_a_changed_result_does_not_change_the_next_one():
    path = next(p for p in every_path() if field_meta(p).get("conservative"))
    first = field_meta(path)
    first["conservative"] = "changed"
    first["injected"] = True
    assert field_meta(path) == reference(path)
