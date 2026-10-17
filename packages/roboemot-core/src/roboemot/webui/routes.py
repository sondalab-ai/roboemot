"""LLM routes for the web UI: list them and run the conformance test (SPEC 8, 12).

The check is advisory in v1: the last report per route is kept in memory and in a small JSON
file in the data folder, and shown next to the route; it does not gate the route. Nothing
secret leaves this module: a route is described by name, model, the host of its base URL and
the minors flag, and a report holds statuses and short reasons only.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .. import paths
from ..config.schema import RoboEmotConfig
from ..llm.conformance import check_route
from ..llm.router import Router

log = logging.getLogger(__name__)


class RouteBusy(RuntimeError):
    """Another check is running: they are serialised so they cannot spend the budget in parallel."""


class RouteChecker:
    """What the web UI needs for routes: ``list_routes()`` and ``check(name)``.

    ``cfg`` returns the effective config (routes and ``llm_check`` follow a hot reload).
    ``router`` returns the app's router so the check is charged to the shared daily budget;
    None (or a missing router) falls back to a private one.
    """

    def __init__(
        self,
        cfg: Callable[[], RoboEmotConfig],
        store: Path | None = None,
        *,
        router: Callable[[], Router | None] = lambda: None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.cfg, self.store, self._router = cfg, store, router
        self._factory = http_client_factory
        self._lock = threading.Lock()
        self._reports: dict[str, dict[str, Any]] = self._load()

    def list_routes(self) -> list[dict[str, Any]]:
        """Routes with their last report; ``api_key`` and the full URL are never included."""
        return [
            {
                "name": r.name,
                "model": r.model,
                "host": urlsplit(r.base_url).hostname or "",
                "allowed_for_minors": r.allowed_for_minors,
                "last_check": self._reports.get(r.name),
            }
            for r in self.cfg().llm_routes
        ]

    def check(self, name: str) -> dict[str, Any]:
        """Run the test on the route ``name`` within ``llm_check.deadline_s`` and keep the report.

        Raises ``KeyError`` (unknown route), ``RouteBusy``, ``RouteBudgetExhausted`` and
        ``TimeoutError``.
        """
        cfg = self.cfg()
        route = next((r for r in cfg.llm_routes if r.name == name), None)
        if route is None:
            raise KeyError(name)
        if not self._lock.acquire(blocking=False):
            raise RouteBusy(name)
        try:
            report = asyncio.run(
                asyncio.wait_for(
                    check_route(
                        route,
                        cfg.safety,
                        settings=cfg.llm_check,
                        router=self._router(),
                        http_client_factory=self._factory,
                    ),
                    cfg.llm_check.deadline_s,
                )
            )
        finally:
            self._lock.release()
        data = report.to_dict()
        self._reports[name] = data
        self._save()
        return data

    def _load(self) -> dict[str, dict[str, Any]]:
        if self.store is None:
            return {}
        try:
            data = json.loads(self.store.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        if self.store is None:
            return
        tmp = self.store.with_name(self.store.name + ".tmp")
        try:
            self.store.parent.mkdir(mode=paths.DIR_MODE, parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self._reports, indent=2))
            os.replace(tmp, self.store)
        except OSError as exc:
            log.warning("route checks: not written (%s)", type(exc).__name__)
            with contextlib.suppress(OSError):
                tmp.unlink()
