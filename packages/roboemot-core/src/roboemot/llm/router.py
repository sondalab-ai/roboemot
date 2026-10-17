"""OpenAI-compatible LLM router (SPEC 8, LLM router and OpenAI-compatible routes).

Uses only the common subset of Chat Completions: messages with system/user/assistant/tool
roles, ``stream=True``, ``tools`` with ``tool_choice="auto"``, ``max_tokens``. Routes are tried
in order; a route is skipped when it is not cleared for minors (while a minor is present) or
its daily token budget is spent, and abandoned for the next one on error or when the first
token arrives later than ``first_token_timeout_s``, as long as nothing was yielded yet.

With a :class:`~roboemot.health.network.NetworkMonitor` (SPEC 12A), routes known to be
unreachable are skipped without waiting for their timeouts, every request reports whether its
route was reachable (a timeout only asks for a prompt reachability check), and when no route
is reachable :class:`Offline` is raised at once, so the caller can answer nonverbally instead
of leaving the robot waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import re
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from ..config.schema import LLMRoute, SafetyConfig
from ..health.network import NetworkMonitor
from .prompt import CHARS_PER_TOKEN, trim_history

log = logging.getLogger(__name__)
SECRET = re.compile(r"^\$\{secret:([A-Za-z0-9_.-]+)\}$")


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Done:
    route: str
    tokens: int
    estimated: bool


Event = TextDelta | ToolCall | Done


class NoRouteAvailable(RuntimeError):
    """Every route was skipped or failed; the caller plays the register fallback (SPEC 8)."""


class Offline(NoRouteAvailable):
    """No route is reachable (SPEC 12A): reply nonverbally, do not wait for timeouts."""


# A timeout may be a slow model on a fine network: it triggers a reachability check instead.
SLOW = (TimeoutError, APITimeoutError)
UNREACHABLE = (APIConnectionError, OSError)  # checked after SLOW (both contain timeouts)


@dataclass
class FirstToken:
    """Seconds from the request to the first real content, filled in by ``stream_route``."""

    seconds: float | None = None


@dataclass
class _Budget:
    day: dt.date
    used: int = 0


@dataclass
class Router:
    routes: list[LLMRoute]
    safety: SafetyConfig
    secrets: Callable[[str], str] = lambda name: ""
    http_client_factory: Callable[[], httpx.AsyncClient] | None = None
    today: Callable[[], dt.date] = dt.date.today
    monitor: NetworkMonitor | None = None
    _budgets: dict[str, _Budget] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def adopt_budgets(self, other: Router) -> None:
        """Keep ``other``'s daily token accounting (a rebuilt router after a config reload must
        not hand every route a fresh budget)."""
        self._budgets = other._budgets

    @property
    def offline(self) -> bool:
        return self.monitor is not None and not self.monitor.online

    def eligible(self, minor_present: bool) -> list[LLMRoute]:
        out = []
        for r in self.routes:
            if self.monitor and not self.monitor.is_online(r.name):
                continue
            if minor_present and not (r.allowed_for_minors and self.safety.minors_allowed_llm):
                continue
            if self.used(r.name) >= r.daily_token_budget:
                continue
            out.append(r)
        return out

    def used(self, name: str) -> int:
        b = self._budgets.get(name)
        return b.used if b and b.day == self.today() else 0

    def _charge(self, name: str, tokens: int) -> None:
        with self._lock:  # the web UI's check thread charges next to the event loop
            b = self._budgets.get(name)
            if not b or b.day != self.today():
                b = self._budgets[name] = _Budget(self.today())
            b.used += tokens

    def charge(self, name: str, tokens: int) -> None:
        """Count ``tokens`` against a route's daily budget (the speech backend charges the
        characters it sends, estimated at ``CHARS_PER_TOKEN`` per token)."""
        self._charge(name, tokens)

    def auth_headers(self, route: LLMRoute) -> dict[str, str]:
        """``Authorization`` header for ``route`` (empty for the ``none`` key), the
        ``${secret:name}`` reference resolved. The key itself is not handed out."""
        key = self._resolved_key(route)
        return {"Authorization": f"Bearer {key}"} if key and key != "none" else {}

    def _resolved_key(self, route: LLMRoute) -> str:
        m = SECRET.match(route.api_key)
        return self.secrets(m.group(1)) if m else route.api_key

    def _client(self, route: LLMRoute) -> AsyncOpenAI:
        key = self._resolved_key(route)
        http = self.http_client_factory() if self.http_client_factory else None
        return AsyncOpenAI(
            base_url=route.base_url, api_key=key or "none", http_client=http, max_retries=0
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        minor_present: bool = False,
    ) -> AsyncIterator[Event]:
        if self.offline:
            raise Offline("no route reachable")
        routes = self.eligible(minor_present)
        if not routes:
            raise NoRouteAvailable(
                "no eligible route (minor present, budgets spent or unreachable)"
            )
        for route in routes:
            yielded = False
            gen = self._stream_route(route, messages, tools)
            try:
                async for ev in gen:
                    yielded = True
                    yield ev
                self._reachable(route, True)
                return
            except Exception as exc:  # CancelledError/GeneratorExit are not caught
                self.failures.append((route.name, type(exc).__name__))
                if isinstance(exc, SLOW):
                    if self.monitor:
                        self.monitor.recheck(route.name)
                elif isinstance(exc, UNREACHABLE):
                    self._reachable(route, False)
                elif isinstance(exc, APIStatusError):
                    self._reachable(route, True)  # the server answered: the network is fine
                if yielded:
                    raise  # mid-answer failure: no silent switch to another model
            finally:
                await gen.aclose()  # barge-in: closes the HTTP stream and charges the budget
        failed = self.failures[-len(routes) :]
        if self.offline:
            raise Offline(f"no route reachable: {failed}")
        raise NoRouteAvailable(f"all routes failed: {failed}")

    def _reachable(self, route: LLMRoute, ok: bool) -> None:
        if self.monitor:
            self.monitor.report(route.name, ok)

    def stream_route(
        self,
        route: LLMRoute,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timing: FirstToken | None = None,
    ) -> AsyncIterator[Event]:
        """Stream ONE route, bypassing eligibility and fallback (the conformance test). Errors
        propagate; the budget is charged as for any request."""
        return self._stream_route(route, messages, tools, timing)

    async def _stream_route(
        self,
        route: LLMRoute,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        timing: FirstToken | None = None,
    ) -> AsyncIterator[Event]:
        """Stream one route. ``first_token_timeout_s`` bounds the time from the request to the
        first real content (text or tool call; role-only chunks do not count);
        ``stall_timeout_s`` bounds every later gap. Tokens are charged even when the stream
        ends early (barge-in, timeout, error)."""
        messages, dropped = trim_history(messages, route.context_budget_tokens)
        if dropped:
            log.debug(
                "llm route %s: dropped %d oldest turns to fit the budget", route.name, dropped
            )
        loop = asyncio.get_running_loop()
        client = self._client(route)  # not timed: the first one pays openai's lazy imports
        started = loop.time()
        deadline = started + route.first_token_timeout_s
        kwargs: dict[str, Any] = {
            "model": route.model,
            "messages": messages,
            "stream": True,
            "max_tokens": route.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        stream = None
        out_chars, usage_tokens = 0, None
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(**kwargs), max(0.0, deadline - loop.time())
            )
            it = stream.__aiter__()
            calls: dict[int, dict[str, Any]] = {}
            got_content = False
            while True:
                timeout = route.stall_timeout_s if got_content else deadline - loop.time()
                if timeout <= 0:
                    raise TimeoutError("first token timeout")
                try:
                    chunk = await asyncio.wait_for(it.__anext__(), timeout)
                except StopAsyncIteration:
                    break
                if getattr(chunk, "usage", None):
                    usage_tokens = chunk.usage.total_tokens
                for choice in chunk.choices or []:
                    delta = choice.delta
                    first = delta.content or delta.tool_calls
                    if first and timing is not None and timing.seconds is None:
                        timing.seconds = loop.time() - started
                    if delta.content:
                        got_content = True
                        out_chars += len(delta.content)
                        yield TextDelta(delta.content)
                    for tc in delta.tool_calls or []:
                        got_content = True
                        slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
                            out_chars += len(tc.function.arguments)
            for slot in calls.values():
                try:
                    args = json.loads(slot["args"] or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": slot["args"]}
                yield ToolCall(slot["id"], slot["name"], args)
            tokens = self._tokens(usage_tokens, prompt_chars, out_chars)
            yield Done(route.name, tokens, usage_tokens is None)
        finally:
            # Charged once, also on barge-in / timeout / error (review finding 9).
            self._charge(route.name, self._tokens(usage_tokens, prompt_chars, out_chars))
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.close()
            with contextlib.suppress(Exception):
                await client.close()

    @staticmethod
    def _tokens(usage: int | None, prompt_chars: int, out_chars: int) -> int:
        return usage if usage is not None else (prompt_chars + out_chars) // CHARS_PER_TOKEN


def build_messages(
    *,
    system: str,
    profile_summary: str | None,
    scene_summary: str | None,
    history: list[dict[str, str]],
    user_text: str,
    affect: str | None = None,
    names: dict[str, str] | None = None,
    summary_max_chars: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Text-only request (SPEC 8, 12). ``system`` is parts 1 to 4 from
    ``roboemot.llm.prompt.system_prompt``; the speaker, scene and affect parts follow, the
    summaries cut to ``summary_max_chars`` (profile, scene). ``names`` maps real names to
    pseudonyms when the route option is on; embeddings, audio and images never enter here."""

    def scrub(text: str | None) -> str | None:
        if text is None or not names:
            return text
        for real, alias in names.items():
            text = text.replace(real, alias)
        return text

    profile_cap, scene_cap = summary_max_chars or (None, None)
    sys_parts = [system]
    if profile_summary:
        sys_parts.append(f"Speaker: {scrub(profile_summary)[:profile_cap]}")
    if scene_summary:
        sys_parts.append(f"Scene: {scrub(scene_summary)[:scene_cap]}")
    if affect:
        sys_parts.append(f"Your mood: {affect}")
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "\n".join(sys_parts)}]
    msgs += [{"role": m["role"], "content": scrub(m["content"])} for m in history]
    msgs.append({"role": "user", "content": scrub(user_text)})
    return msgs
