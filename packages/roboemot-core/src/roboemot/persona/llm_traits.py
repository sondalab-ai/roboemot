"""Persona traits rated by an LLM (SPEC 11A, Persona: trait source ``llm``).

One request per change of prompt or model, cached on disk. The reply is plain text parsed
line by line (no ``response_format``, no forced tool choice, SPEC 8) and validated against
the trait schema. Any failure keeps the manual traits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from ..config.schema import TRAITS, PersonaConfig, PersonaTraits, RoboEmotConfig
from ..config.stack import ConfigChange, ConfigStack
from ..llm.router import Done, NoRouteAvailable, Router, TextDelta

log = logging.getLogger(__name__)

SOURCE = "persona.llm"
INSTRUCTIONS = (
    "Rate the robot persona described by the user on each trait, from 0.0 to 1.0. "
    "warmth: warm and affectionate (1) or cool and detached (0). "
    "energy: lively and fast (1) or slow and quiet (0). "
    "playfulness: playful and reactive (1) or composed (0). "
    "formality: formal (1) or casual (0) wording. "
    "Reply with exactly four lines in the form trait=value and nothing else."
)
LINE = re.compile(r"^\s*([a-z]+)\s*[=:]\s*([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)


def cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}\0{prompt}".encode()).hexdigest()


def parse_traits(text: str) -> PersonaTraits | None:
    """Traits from ``trait=value`` lines; None unless all four are present and in range."""
    found = {k: float(v) for k, v in LINE.findall(text) if k in TRAITS}
    if set(found) != set(TRAITS):
        return None
    try:
        return PersonaTraits(**found)
    except ValidationError:
        return None


class TraitCache:
    """JSON file mapping ``cache_key`` to rated traits."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def get(self, key: str) -> PersonaTraits | None:
        data = self._load().get(key)
        try:
            return PersonaTraits(**data) if isinstance(data, dict) else None
        except ValidationError:
            return None

    def put(self, key: str, traits: PersonaTraits) -> None:
        data = self._load()
        data[key] = traits.model_dump()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))


async def rate_traits(
    router: Router, persona: PersonaConfig, cache: TraitCache
) -> PersonaTraits | None:
    """Traits for ``persona.prompt`` from the cache or the first route that answers; None when
    no route is configured, eligible or reachable, or the reply does not validate."""
    routes = router.eligible(minor_present=False)
    if not routes:
        return None
    cached = cache.get(cache_key(persona.prompt, routes[0].model))
    if cached:
        return cached
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": persona.prompt},
    ]
    text, route_name = "", None
    try:
        async for ev in router.stream(messages):
            if isinstance(ev, TextDelta):
                text += ev.text
            elif isinstance(ev, Done):
                route_name = ev.route
    except NoRouteAvailable:
        return None
    except Exception as exc:  # a mid-answer failure: same outcome as no answer
        log.warning("persona traits: route failed (%s)", type(exc).__name__)
        return None
    traits = parse_traits(text)
    model = next((r.model for r in routes if r.name == route_name), None)
    if traits and model:
        cache.put(cache_key(persona.prompt, model), traits)
    return traits


async def apply_llm_traits(stack: ConfigStack, router: Router, cache: TraitCache) -> bool:
    """With ``persona.traits_source: llm``, replace the manual traits with the rated ones in
    the stack's derived layer (audited as ``persona.llm``, published as ``config.changed``).
    Run it at startup and again on every ``config.changed``: when the source is manual or the
    current prompt gets no valid rating, earlier rated traits are dropped and the manual
    traits apply. Returns True only when rated traits are in effect."""
    persona = stack.effective().persona
    traits = await rate_traits(router, persona, cache) if persona.traits_source == "llm" else None
    if traits is None:
        if persona.traits_source == "llm":
            log.warning("persona traits: no valid LLM rating, keeping the manual traits")
        stack.clear("persona.traits", source=SOURCE, layer="derived")
        return False
    if stack.derived.get("persona", {}).get("traits") != traits.model_dump():
        stack.set("persona.traits", traits.model_dump(), source=SOURCE, layer="derived")
    return True


WATCHED = ("persona.", "llm_routes")  # config keys the rating depends on
JOIN_TIMEOUT_S = 2.0  # how long close() waits for the worker thread after cancelling it


class TraitsWorker:
    """Runs ``apply_llm_traits`` on its own thread and event loop (SPEC 11A, Persona), so a slow
    route never blocks the app tick or the motion loop.

    ``on_change`` is the ``config.changed`` listener: changes to ``persona.*`` or ``llm_routes``
    wake the worker; wake-ups coalesce into one run. Changes published by the worker itself
    (the derived layer ``persona.llm``) are ignored, so it cannot trigger itself. The router is
    rebuilt (keeping its token budgets) when the routes or the safety config change. ``close``
    cancels a request in flight and never waits longer than ``JOIN_TIMEOUT_S``.
    """

    def __init__(
        self,
        stack: ConfigStack,
        router: Router,
        cache: TraitCache,
        make_router: Callable[[RoboEmotConfig], Router],
    ) -> None:
        self.stack, self.router, self.cache, self.make_router = stack, router, cache, make_router
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._ready = threading.Event()
        self._closed = False

    def start(self) -> None:
        """Start the worker thread and request the first run (the startup rating)."""
        self._thread = threading.Thread(target=self._run, name="llm-traits", daemon=True)
        self._thread.start()
        self._ready.wait(JOIN_TIMEOUT_S)
        self.request()

    def request(self) -> None:
        """Ask for a run; safe from any thread, a no-op from the worker's own."""
        loop, wake = self._loop, self._wake
        if self._closed or loop is None or wake is None:
            return
        if threading.current_thread() is self._thread:
            return
        loop.call_soon_threadsafe(wake.set)

    def on_change(self, change: ConfigChange) -> None:
        """``config.changed`` listener: wake the worker when a key the rating depends on moved."""
        if any(k.startswith(WATCHED) for k in change.changed_keys):
            self.request()

    def close(self) -> None:
        """Cancel any request in flight and stop the thread."""
        self._closed = True
        loop, task = self._loop, self._task
        if loop is not None and task is not None and not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)
        if self._thread is not None:
            self._thread.join(JOIN_TIMEOUT_S)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except asyncio.CancelledError:
            pass
        finally:
            self._ready.set()

    async def _main(self) -> None:
        self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        self._wake = asyncio.Event()
        self._ready.set()
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                cfg = self.stack.effective()
                if self.router.routes != cfg.llm_routes or self.router.safety != cfg.safety:
                    fresh = self.make_router(cfg)
                    fresh.adopt_budgets(self.router)
                    self.router = fresh
                await apply_llm_traits(self.stack, self.router, self.cache)
            except Exception as exc:  # the manual traits stay in effect
                log.warning("persona traits: %s", type(exc).__name__)
