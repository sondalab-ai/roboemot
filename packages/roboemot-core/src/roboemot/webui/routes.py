"""LLM routes for the web UI: list them and run the conformance test of a chat route or the
realtime check of a ``kind: realtime`` route (``realtime.check``) (SPEC 8, 12).

The check is advisory in v1: the last report per route is kept in memory and in a small JSON
file in the data folder, and shown next to the route; it does not gate the route. It runs on
its own thread, bounded by ``llm_check.deadline_s``, because it outlives the per-request
deadline of the web server (``webui.request_deadline_s``): the POST returns at once and the
page polls the list. Nothing secret leaves this module: a route is described by name, model,
``host[:port]`` of its base URL, its locality, its kind and the kind of its key, plus the
full base URL only when it holds no credentials (the settings list shows the same value), and a
report holds statuses and short reasons only.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
from ..config.schema import LLMRoute, RoboEmotConfig, active_realtime_route
from ..config.stack import NO_KEY
from ..llm.conformance import (
    RouteBudgetExhausted,
    RouteKeyUnavailable,
    RouteKindUnsupported,
    RouteReport,
    check_route,
    route_endpoint,
)
from ..llm.pseudonyms import NameSource
from ..llm.router import SECRET, Router
from ..realtime.check import check_realtime_route
from ..realtime.preview import PreviewFailed, preview_voice
from ..realtime.session import DirectSession, SessionProvider
from ..redact import secret_shaped
from ..secret_store import SecretStore

log = logging.getLogger(__name__)
FILE_MODE = 0o600
# what the Routes tab shows of the running realtime session (``RealtimeRoute.snapshot``)
SESSION_FIELDS = (
    "state",
    "reason",
    "idle_closes",
    "go_aways",
    "usage_reports",
    "items_delete_emulated",
)


class RouteBusy(RuntimeError):
    """Another check or voice preview is running; they are serialised so they cannot spend a
    budget in parallel."""


class RouteChecker:
    """What the web UI needs for routes: ``list_routes()`` and ``start(name)``.

    ``cfg`` returns the effective config (routes and ``llm_check`` follow a hot reload).
    ``router`` returns the app's router so the check is charged to the shared daily budget;
    without one, a private router is kept for the life of this object, so budgets still add up
    across checks. ``names`` (a :class:`NameSource`) gives a route that is not local the same
    aliases as the conversation does (SPEC 8, Pseudonyms) when ``privacy.pseudonyms_to_cloud``
    is on. ``realtime_provider`` returns the robot's session provider (the one the runtime
    built); without it, direct mode uses :class:`DirectSession` and platform mode has none.
    ``realtime_tools()`` builds a ``gemini_live`` check's setup tools (``offline_tools`` with
    the robot's gestures); without it, ``offline_tools`` of the config. ``realtime_status``
    returns the running realtime
    route's debug snapshot: the active realtime route's entry shows a few of its fields
    (``session``).
    """

    def __init__(
        self,
        cfg: Callable[[], RoboEmotConfig],
        store: Path | None = None,
        *,
        router: Callable[[], Router | None] = lambda: None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        names: NameSource | None = None,
        realtime_provider: Callable[[], SessionProvider | None] | None = None,
        realtime_tools: Callable[[], list[dict[str, Any]] | None] | None = None,
        realtime_status: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self.cfg, self.store, self._router = cfg, store, router
        self._realtime_provider = realtime_provider
        self._realtime_tools = realtime_tools
        self._realtime_status = realtime_status
        self._names = names
        self._factory = http_client_factory
        self._private: Router | None = None
        self._lock = threading.Lock()
        self._running: str | None = None
        self._previewing = False
        self.thread: threading.Thread | None = None
        self._reports: dict[str, dict[str, Any]] = self._load()

    def list_routes(self) -> list[dict[str, Any]]:
        """Routes with their running flag and last report; a report made for another model or
        endpoint than the route has now is not shown. A route whose key is a ``${secret:name}``
        reference shows ``secret`` (the name and whether it is set), never the value; ``key`` says
        which kind of key the route has (``none``, ``secret`` or ``literal``). ``api_key`` and the
        credentials of the URL are never included, and ``base_url`` is null when it holds any.
        A realtime route also has ``session``: the running session when it is the active
        realtime route (``_session``), else null."""
        cfg = self.cfg()
        session = self._session(cfg)  # outside the lock: the route takes its own
        active = active_realtime_route(cfg)
        with self._lock:
            running = self._running
            store = SecretStore.for_data_dir(cfg.data_dir)
            return [
                {
                    "name": r.name,
                    "model": r.model,
                    "endpoint": route_endpoint(r),
                    "base_url": None if secret_shaped(r.base_url) else r.base_url,
                    "locality": r.locality,
                    "kind": r.kind,
                    **_set_fields(r),
                    "key": _key_kind(r),
                    "running": r.name == running,
                    "secret": _secret_state(r, store),
                    "last_check": self._current_report(r),
                    **(
                        {"session": session if r is active else None}
                        if r.kind == "realtime"
                        else {}
                    ),
                }
                for r in cfg.llm_routes
            ]

    def start(self, name: str) -> None:
        """Start the test of route ``name`` on a thread and return at once.

        A chat route gets the conformance test, a realtime route the realtime check. Raises
        ``KeyError`` (unknown route), ``RouteKindUnsupported`` (another kind),
        ``RouteKeyUnavailable`` (its secret cannot be read) and
        ``RouteBudgetExhausted`` (nothing is sent in any of these cases) and ``RouteBusy`` (a check
        is running).
        """
        cfg = self.cfg()
        route = next((r for r in cfg.llm_routes if r.name == name), None)
        if route is None:
            raise KeyError(name)
        if route.kind not in ("chat", "realtime"):
            raise RouteKindUnsupported(f"route {name} is a {route.kind} route")
        with self._lock:
            if self._running is not None or self._previewing:
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

    def preview(self, name: str, voice: str, wait_s: float) -> bytes:
        """Say the voice preview of realtime route ``name`` in ``voice`` (SPEC 8, Voice
        preview) and return its WAV; one at a time, never beside a check (``RouteBusy``). It
        runs on its own thread, bounded by ``wait_s``; this call waits that long for the audio
        (``PreviewFailed("timeout")`` past it), the session's close comes after. ``KeyError``
        for an unknown route; the check's refusals and :class:`PreviewFailed` otherwise."""
        cfg = self.cfg()
        route = next((r for r in cfg.llm_routes if r.name == name), None)
        if route is None:
            raise KeyError(name)
        if route.kind != "realtime":
            raise RouteKindUnsupported(f"route {name} is a {route.kind} route")
        with self._lock:
            if self._running is not None or self._previewing:
                raise RouteBusy(name)
            router = self._router() or self._private_router(cfg)
            self._previewing = True
        result: concurrent.futures.Future[bytes] = concurrent.futures.Future()
        thread = threading.Thread(
            target=self._preview,
            args=(route, voice, cfg, router, wait_s, result),
            name="roboemot-voice-preview",
            daemon=True,
        )
        thread.start()
        try:
            return result.result(wait_s)
        except concurrent.futures.TimeoutError as exc:
            raise PreviewFailed("timeout") from exc

    def _preview(
        self,
        route: LLMRoute,
        voice: str,
        cfg: RoboEmotConfig,
        router: Router,
        wait_s: float,
        result: concurrent.futures.Future[bytes],
    ) -> None:
        """Thread body of :meth:`preview`: the audio or the failure goes to ``result`` as soon
        as it is known (before the session's close); the busy flag is cleared only once the
        session is closed."""

        def ready(out: bytes | PreviewFailed) -> None:
            if isinstance(out, PreviewFailed):
                result.set_exception(out)
            else:
                result.set_result(out)

        try:
            preview_voice(
                route,
                voice,
                cfg,
                router=router,
                provider=self._provider(cfg),
                tools=self._realtime_tools,
                deadline_s=wait_s,
                config=self.cfg,
                ready=ready,
            )
        except (PreviewFailed, RouteKeyUnavailable, RouteBudgetExhausted) as exc:
            if not result.done():
                result.set_exception(exc)
        except Exception as exc:
            log.error("voice preview %s failed: %s", route.name, type(exc).__name__)
            if not result.done():
                result.set_exception(PreviewFailed("failed"))
        finally:
            with self._lock:
                self._previewing = False

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
            if route.kind == "realtime":
                realtime = check_realtime_route(
                    route,
                    cfg,
                    router=router,
                    provider=self._provider(cfg),
                    tools=self._realtime_tools,
                    config=self.cfg,
                )
                self._keep(route, realtime.to_dict())
                return
            if self._names is None and cfg.privacy.pseudonyms_to_cloud:
                log.warning("route check: pseudonyms are on but no name source is wired")
            pseudonyms = self._names.for_config(cfg) if self._names else None
            report = asyncio.run(
                asyncio.wait_for(
                    check_route(
                        route,
                        cfg.safety,
                        settings=cfg.llm_check,
                        config=cfg,
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
        self._keep(route, report.to_dict())

    def _keep(self, route: LLMRoute, report: dict[str, Any]) -> None:
        """Store ``report`` as the route's last one; the check is over."""
        with self._lock:
            self._reports[route.name] = report
            self._save()
            self._running = None

    def _session(self, cfg: RoboEmotConfig) -> dict[str, Any] | None:
        """The running realtime session as the Routes tab shows it (SPEC 8, 12A): its state,
        reason code and the counts that matter for a Gemini route (``SESSION_FIELDS``), with
        the codec's non-zero counts (``wire``: this session and all). None while realtime is
        off, no realtime route is active, no route runs or its snapshot fails (logged by type
        only)."""
        rt = cfg.voice.realtime_route
        if self._realtime_status is None or not rt.enabled or active_realtime_route(cfg) is None:
            return None
        try:
            snap = self._realtime_status()
        except Exception as exc:
            log.warning("routes: realtime session status: %s", type(exc).__name__)
            return None
        if snap is None:
            return None
        wire = snap.get("wire") or {}
        return {
            **{k: snap.get(k) for k in SESSION_FIELDS},
            "wire_session": _nonzero(wire.get("session")),
            "wire_total": _nonzero(wire.get("total")),
        }

    def _provider(self, cfg: RoboEmotConfig) -> SessionProvider | None:
        """The session provider of the realtime check (see the class docstring)."""
        if self._realtime_provider is not None:
            return self._realtime_provider()
        return DirectSession() if cfg.voice.realtime_route.mode == "direct" else None

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


def _set_fields(route: LLMRoute) -> dict[str, Any]:
    """The realtime fields ``protocol`` and ``voice`` of a route row, only where the route sets
    them: a row of a route that sets neither stays as it was, and an explicit default also
    stays out. The page reads a missing one as the default (``openai_realtime``, no voice of
    its own)."""
    return route.model_dump(include={"protocol", "voice"}, exclude_defaults=True)


def _key_kind(route: LLMRoute) -> str:
    """``none``, ``secret`` (a ``${secret:name}`` reference) or ``literal`` (a key in the config)"""
    if route.api_key in NO_KEY:
        return "none"
    return "secret" if SECRET.match(route.api_key) else "literal"


def _secret_state(route: LLMRoute, store: SecretStore) -> dict[str, Any] | None:
    """``{"name", "set"}`` for a ``${secret:name}`` key, else None. The value is never read out."""
    ref = SECRET.match(route.api_key)
    return {"name": ref.group(1), "set": store.has(ref.group(1))} if ref else None


def _now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _nonzero(counts: dict[str, int] | None) -> dict[str, int]:
    """The non-zero entries of a codec's counts (none for the plain subset)."""
    return {k: v for k, v in (counts or {}).items() if v}
