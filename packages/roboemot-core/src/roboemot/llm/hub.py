"""The app's one LLM router (SPEC 8): every consumer reads it from here.

The traits worker, the conversation worker, the speech backend and the web UI route check all
charge the same daily budgets. A configuration change that touches the routes or the safety
config rebuilds the router; the new one adopts the old one's budgets (and lock), so a stream
still running on the old router and a request on the new one count into the same accounting.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from ..config.schema import RoboEmotConfig
from .router import Router

log = logging.getLogger(__name__)


class RouterHub:
    """Holds the current :class:`Router`; ``sync`` swaps it when the configuration changed."""

    def __init__(self, router: Router, make_router: Callable[[RoboEmotConfig], Router]) -> None:
        self._router = router
        self.make_router = make_router
        self._lock = threading.Lock()

    @property
    def router(self) -> Router:
        return self._router

    def sync(self, cfg: RoboEmotConfig) -> Router:
        """The router for ``cfg``: rebuilt (budgets kept) when routes or safety changed, the old
        one if building fails. Safe from any thread."""
        with self._lock:
            current = self._router
            if current.routes == cfg.llm_routes and current.safety == cfg.safety:
                return current
            try:
                fresh = self.make_router(cfg)
            except Exception as exc:
                log.warning("llm router not rebuilt (%s)", type(exc).__name__)
                return current
            fresh.adopt_budgets(current)
            self._router = fresh
            return fresh
