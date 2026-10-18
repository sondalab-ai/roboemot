"""LLM routes for the web UI: list them and run the conformance test (SPEC 8, 12).

The check is advisory in v1: the last report per route is kept in memory and in a small JSON
file in the data folder, and shown next to the route; it does not gate the route. It runs on
its own thread, bounded by ``llm_check.deadline_s``, because it outlives the per-request
deadline of the web server (``webui.request_deadline_s``): the POST returns at once and the
page polls the list. Nothing secret leaves this module: a route is described by name, model,
``host[:port]`` of its base URL and the minors flag, and a report holds statuses and short
reasons only.
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

import httpx

from .. import paths
from ..config.schema import LLMRoute, RoboEmotConfig
from ..llm.conformance import (
    RouteBudgetExhausted,
    RouteKeyUnavailable,
    RouteReport,
    check_route,
    route_endpoint,
)
from ..llm.pseudonyms import NameSource
from ..llm.router import SECRET, Router
from ..secret_store import SecretStore

log = logging.getLogger(__name__)
FILE_MODE = 0o600


class RouteBusy(RuntimeError):
    """Another check is running; they are serialised so they cannot spend a budget in parallel."""


class RouteChecker:
    """What the web UI needs for routes: ``list_routes()`` and ``start(name)``.

    ``cfg`` returns the effective config (routes and ``llm_check`` follow a hot reload).
    ``router`` returns the app's router so the check is charged to the shared daily budget;
    without one, a private router is kept for the life of this object, so budgets still add up
    across checks. ``names`` (a :class:`NameSource`) gives a route that is not local the same
    aliases as the conversation does (SPEC 8, Pseudonyms) when ``privacy.pseudonyms_to_cloud``
    is on.
    """

    def __init__(
        self,
        cfg: Callable[[], RoboEmotConfig],
        store: Path | None = None,
        *,
        router: Callable[[], Router | None] = lambda: None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        names: NameSource | None = None,
    ) -> None:
        self.cfg, self.store, self._router = cfg, store, router
        self._names = names
        self._factory = http_client_factory
        self._private: Router | None = None
        self._lock = threading.Lock()
        self._running: str | None = None
        self.thread: threading.Thread | None = None
        self._reports: dict[str, dict[str, Any]] = self._load()

    def list_routes(self) -> list[dict[str, Any]]:
        """Routes with their running flag and last report; a report made for another model or
        endpoint than the route has now is not shown. A route whose key is a ``${secret:name}``
        reference shows ``secret`` (the name and whether it is set), never the value. ``api_key``,
        credentials, path and query of the URL are never included."""
        with self._lock:
            running = self._running
            cfg = self.cfg()
            store = SecretStore.for_data_dir(cfg.data_dir)
            return [
                {
                    "name": r.name,
                    "model": r.model,
                    "endpoint": route_endpoint(r),
                    "allowed_for_minors": r.allowed_for_minors,
                    "running": r.name == running,
                    "secret": _secret_state(r, store),
                    "last_check": self._current_report(r),
                }
                for r in cfg.llm_routes
            ]

    def start(self, name: str) -> None:
        """Start the test of route ``name`` on a thread and return at once.

        Raises ``KeyError`` (unknown route), ``RouteKeyUnavailable`` (its secret cannot be read)
        and ``RouteBudgetExhausted`` (nothing is sent in either case) and ``RouteBusy`` (a check
        is running).
        """
        cfg = self.cfg()
        route = next((r for r in cfg.llm_routes if r.name == name), None)
        if route is None:
            raise KeyError(name)
        with self._lock:
            if self._running is not None:
                raise RouteBusy(name)
            router = self._router() or self._private_router(cfg)
            problem = router.key_problem(route)
            if problem is not None:
                raise RouteKeyUnavailable(f"route {name}: {problem}")
            if router.used(name) >= route.daily_token_budget:
                raise RouteBudgetExhausted(f"route {name}: daily token budget is spent")
            self._running = name
        self.thread = threading.Thread(
            target=self._run, args=(route, cfg, router), name="roboemot-route-check", daemon=True
        )
        self.thread.start()

    def _private_router(self, cfg: RoboEmotConfig) -> Router:
        if self._private is None or self._private.secrets_changed():
            previous = self._private
            self._private = Router(
                cfg.llm_routes,
                cfg.safety,
                secrets=SecretStore.for_data_dir(cfg.data_dir).get,
                http_client_factory=self._factory,
            )
            if previous is not None:  # a secret changed: keep the day's token accounting
                self._private.adopt_budgets(previous)
        return self._private

    def _run(self, route: LLMRoute, cfg: RoboEmotConfig, router: Router) -> None:
        """Thread body: run the check under the deadline and keep the outcome, whatever it is."""
        error = ""
        try:
            if self._names is None and cfg.privacy.pseudonyms_to_cloud:
                log.warning("route check: pseudonyms are on but no name source is wired")
            pseudonyms = self._names.for_config(cfg) if self._names else None
            report = asyncio.run(
                asyncio.wait_for(
                    check_route(
                        route,
                        cfg.safety,
                        settings=cfg.llm_check,
                        router=router,
                        pseudonyms=pseudonyms,
                        http_client_factory=self._factory,
                    ),
                    cfg.llm_check.deadline_s,
                )
            )
        except TimeoutError:
            error = "timeout"
        except RouteBudgetExhausted:
            error = "budget"
        except RouteKeyUnavailable:
            error = "secret"
        except Exception as exc:
            log.error("route check %s failed: %s", route.name, type(exc).__name__)
            error = "failed"
        if error:
            report = RouteReport(
                route.name, route.model, route_endpoint(route), _now_iso(), error=error
            )
        with self._lock:
            self._reports[route.name] = report.to_dict()
            self._save()
            self._running = None

    def _current_report(self, route: LLMRoute) -> dict[str, Any] | None:
        report = self._reports.get(route.name)
        same = report and report.get("model") == route.model
        return report if same and report.get("endpoint") == route_endpoint(route) else None

    def _load(self) -> dict[str, dict[str, Any]]:
        """Stored reports; anything that is not a report-shaped entry is dropped."""
        if self.store is None:
            return {}
        try:
            data = json.loads(self.store.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            name: rep
            for name, rep in data.items()
            if isinstance(name, str)
            and isinstance(rep, dict)
            and isinstance(rep.get("checks"), list)
        }

    def _save(self) -> None:
        """Write the reports atomically (caller holds the lock); the data folder is created
        private when missing."""
        if self.store is None:
            return
        tmp = self.store.with_name(self.store.name + ".tmp")
        try:
            folder = self.store.parent
            if not folder.exists():
                folder.mkdir(mode=paths.DIR_MODE, parents=True)
                folder.chmod(paths.DIR_MODE)
            tmp.write_text(json.dumps(self._reports, indent=2))
            tmp.chmod(FILE_MODE)
            os.replace(tmp, self.store)
        except OSError as exc:
            log.warning("route checks: not written (%s)", type(exc).__name__)
            with contextlib.suppress(OSError):
                tmp.unlink()


def _secret_state(route: LLMRoute, store: SecretStore) -> dict[str, Any] | None:
    """``{"name", "set"}`` for a ``${secret:name}`` key, else None. The value is never read out."""
    ref = SECRET.match(route.api_key)
    return {"name": ref.group(1), "set": store.has(ref.group(1))} if ref else None


def _now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
