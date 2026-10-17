"""Builds the catalogue service for the web UI (SPEC 6, Catalogue import).

Kept apart from ``roboemot.registry.catalogue`` so the registry stays free of web code: the
name validator is the web UI's own (pattern and ``webui.name_max_chars``), and the embedder is
looked up on each call because the vision pipeline may start after the page is served.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from roboemot.config.schema import RoboEmotConfig
from roboemot.registry import Registry
from roboemot.registry.catalogue import AppearanceSource, CatalogueService, LiveWorld

from . import validation as v

BUDGET_SHARE = 0.5  # of the request deadline: the rest is for parsing, writing and replying


def open_catalogue(
    registry: Registry | None,
    cfg: Callable[[], RoboEmotConfig],
    *,
    embedder: Callable[[], AppearanceSource | None] = lambda: None,
    world: LiveWorld | None = None,
    purge_candidates: Callable[[], int] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> CatalogueService | None:
    """A ``CatalogueService`` over ``registry`` (None when the registry is unavailable).

    ``cfg`` returns the effective configuration, so limits follow a hot reload.
    """
    if registry is None:
        return None
    current = cfg()

    def validate(text: str) -> str:
        return v.name(text, "name", cfg().webui)

    service = CatalogueService(
        registry,
        current.catalogue,
        embedder=embedder,
        world=world,
        validate_name=validate,
        now=monotonic,
        purge_candidates=purge_candidates,
        budget_cap=lambda: cfg().webui.request_deadline_s * BUDGET_SHARE,
    )
    return service
