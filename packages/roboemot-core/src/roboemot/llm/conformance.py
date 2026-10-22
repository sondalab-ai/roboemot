"""Route conformance test (SPEC 8, OpenAI-compatible routes).

``check_route`` runs three requests against ONE route, bypassing fallback and eligibility:
an Italian prompt without tools (streaming, first-token latency, Italian output) and a tool
round trip (a call to ``get_time``, then its result sent back). The result is advisory in v1
(SPEC 8): it is shown and stored, it does not gate the route. The requests are charged to the
route's daily token budget like any other; a route whose budget is spent is refused.

The report holds statuses and short reasons only: never prompts, replies or keys, and errors
are reduced to a type or an HTTP status.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from ..config.schema import LLMCheckConfig, LLMRoute, SafetyConfig
from ..secret_store import no_secrets
from .pseudonyms import Pseudonyms
from .router import Done, FirstToken, Router, TextDelta, ToolCall

log = logging.getLogger(__name__)

PASS, FAIL, SKIP = "pass", "fail", "skip"
STREAMING, TOOL_ROUND_TRIP, FIRST_TOKEN, ITALIAN = (
    "streaming",
    "tool_round_trip",
    "first_token_latency",
    "italian",
)
TOOL_NAME = "get_time"
TOOL_RESULT = "10:30"
ITALIAN_PROMPT = "Reply in Italian, in two short sentences: greet me and describe a sunny day."
TOOL_PROMPT = "What time is it? Use the get_time tool."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Current local time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]
WORD = re.compile(r"[^\W\d_]+")


class RouteBudgetExhausted(RuntimeError):
    """The route's daily token budget is spent: the check would exceed it, so it is refused."""


class RouteKeyUnavailable(RuntimeError):
    """The route's ``${secret:name}`` key cannot be read: nothing is sent. The message names
    the secret and the reason, never a value."""


@dataclass
class CheckResult:
    name: str
    status: str
    reason: str


@dataclass
class RouteReport:
    route: str
    model: str
    endpoint: str
    checked_at: str
    checks: list[CheckResult] = field(default_factory=list)
    first_token_s: float | None = None
    tokens: int = 0
    error: str = ""  # "timeout", "budget", "secret" or "failed": the check did not complete

    @property
    def ok(self) -> bool:
        """The check completed and no check failed (skips do not count against the route)."""
        return not self.error and all(c.status != FAIL for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ok": self.ok}


@dataclass
class _Answer:
    text: str = ""
    chunks: int = 0
    calls: list[ToolCall] = field(default_factory=list)
    timing: FirstToken = field(default_factory=FirstToken)
    error: Exception | None = None

    @property
    def timed_out(self) -> bool:
        return isinstance(self.error, (TimeoutError, APITimeoutError))

    @property
    def unreachable(self) -> bool:
        return isinstance(self.error, (APIConnectionError, OSError)) and not self.timed_out


def route_endpoint(route: LLMRoute) -> str:
    """``host`` or ``host:port`` of the route's base URL: no credentials, path or query."""
    parts = urlsplit(route.base_url)
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{host}:{port}" if port else host


def _why(exc: Exception) -> str:
    """A reason that carries no server text: it may echo the request or a key. An HTTP status
    names the path called (no host, credentials or query), so a wrong base URL shows at once."""
    if isinstance(exc, APIStatusError):
        return f"HTTP {exc.status_code} at {exc.request.url.path}"
    if isinstance(exc, (TimeoutError, APITimeoutError)):
        return "timed out"
    if isinstance(exc, (APIConnectionError, OSError)):
        return "route unreachable"
    return type(exc).__name__


def italian_ratio(text: str, stopwords: list[str]) -> tuple[int, float]:
    """(word count, share of words that are Italian stopwords); elisions split at the apostrophe."""
    words = WORD.findall(text.lower().replace("'", " ").replace("’", " "))
    known = set(stopwords)
    hits = sum(1 for w in words if w in known)
    return len(words), (hits / len(words) if words else 0.0)


async def _ask(
    router: Router,
    route: LLMRoute,
    messages: list[dict[str, Any]],
    tools: list | None,
    pseudonyms: Pseudonyms | None = None,
) -> _Answer:
    ans = _Answer()
    try:
        async for ev in router.stream_route(route, messages, tools, ans.timing, pseudonyms):
            if isinstance(ev, TextDelta):
                ans.text += ev.text
                ans.chunks += 1
            elif isinstance(ev, ToolCall):
                ans.calls.append(ev)
            elif isinstance(ev, Done):
                pass
    except Exception as exc:  # CancelledError (the caller's deadline) is not caught
        ans.error = exc
    return ans


def _streaming(ans: _Answer) -> CheckResult:
    if ans.error is not None and not ans.chunks:
        return CheckResult(STREAMING, FAIL, f"no streamed text: {_why(ans.error)}")
    if ans.error is not None:
        return CheckResult(
            STREAMING, FAIL, f"stream broke after {ans.chunks} chunks: {_why(ans.error)}"
        )
    if ans.chunks > 1:
        return CheckResult(STREAMING, PASS, f"{ans.chunks} chunks")
    return CheckResult(
        STREAMING, FAIL, "text came as a single chunk" if ans.chunks else "no text in the stream"
    )


def _latency(ans: _Answer, route: LLMRoute) -> CheckResult:
    limit = route.first_token_timeout_s
    seconds = ans.timing.seconds
    if seconds is None:
        if ans.timed_out:
            return CheckResult(FIRST_TOKEN, FAIL, f"no content within {limit:g} s")
        return CheckResult(FIRST_TOKEN, SKIP, "no answer to measure")
    verdict = PASS if seconds <= limit else FAIL
    return CheckResult(
        FIRST_TOKEN,
        verdict,
        f"{seconds:.2f} s of {limit:g} s allowed (one cold sample, TLS setup included)",
    )


def _italian(ans: _Answer, settings: LLMCheckConfig) -> CheckResult:
    if not ans.text:
        return CheckResult(ITALIAN, SKIP, "no reply to judge")
    count, ratio = italian_ratio(ans.text, settings.italian_stopwords)
    if count < settings.italian_min_words:
        return CheckResult(ITALIAN, FAIL, f"reply too short to judge ({count} words)")
    need = settings.italian_min_stopword_ratio
    verdict = PASS if ratio >= need else FAIL
    return CheckResult(ITALIAN, verdict, f"{ratio:.0%} Italian stopwords, {need:.0%} needed")


async def check_route(
    route: LLMRoute,
    safety: SafetyConfig,
    *,
    settings: LLMCheckConfig | None = None,
    router: Router | None = None,
    secrets: Callable[[str], str] = no_secrets,
    pseudonyms: Pseudonyms | None = None,
    http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> RouteReport:
    """Run the four checks against ``route`` only. Pass the app's ``router`` to share its daily
    budgets; otherwise a private one is built from ``safety``, ``secrets`` and the client
    factory. Raises :class:`RouteKeyUnavailable` when the route's secret cannot be read and
    :class:`RouteBudgetExhausted` when the budget is spent, both before anything is sent. With
    ``pseudonyms`` a route that is not local gets the requests aliased like any cloud request.
    Failures of the route become ``fail`` results, never exceptions."""
    settings = settings or LLMCheckConfig()
    router = router or Router(
        [route], safety, secrets=secrets, http_client_factory=http_client_factory
    )
    problem = router.key_problem(route)
    if problem is not None:
        raise RouteKeyUnavailable(f"route {route.name}: {problem}")
    spent = router.used(route.name)
    if spent >= route.daily_token_budget:
        raise RouteBudgetExhausted(f"route {route.name}: daily token budget is spent")
    report = RouteReport(
        route.name, route.model, route_endpoint(route), now().isoformat(timespec="seconds")
    )

    ask = await _ask(router, route, [{"role": "user", "content": ITALIAN_PROMPT}], None, pseudonyms)
    report.first_token_s = ask.timing.seconds
    tool = await _tool_round_trip(router, route, ask.unreachable, pseudonyms)
    report.checks += [_streaming(ask), tool, _latency(ask, route), _italian(ask, settings)]
    report.tokens = router.used(route.name) - spent
    log.info(
        "llm route %s check: %s",
        route.name,
        ", ".join(f"{c.name}={c.status}" for c in report.checks),
    )
    return report


async def _tool_round_trip(
    router: Router, route: LLMRoute, unreachable: bool, pseudonyms: Pseudonyms | None = None
) -> CheckResult:
    """One tool call with parsable arguments, then a text answer to its result."""
    if unreachable:
        return CheckResult(TOOL_ROUND_TRIP, SKIP, "route unreachable")
    if router.used(route.name) >= route.daily_token_budget:
        return CheckResult(TOOL_ROUND_TRIP, SKIP, "daily token budget spent by the earlier request")
    messages: list[dict[str, Any]] = [{"role": "user", "content": TOOL_PROMPT}]
    first = await _ask(router, route, messages, TOOLS, pseudonyms)
    if first.error is not None:
        return CheckResult(TOOL_ROUND_TRIP, FAIL, f"tool request failed: {_why(first.error)}")
    if not first.calls:
        return CheckResult(TOOL_ROUND_TRIP, FAIL, "no tool call in the reply")
    call = first.calls[0]
    if call.name != TOOL_NAME:
        return CheckResult(TOOL_ROUND_TRIP, FAIL, "tool call names an unknown tool")
    if "_raw" in call.arguments:
        return CheckResult(
            TOOL_ROUND_TRIP, FAIL, "tool call arguments are not a JSON object within limits"
        )
    if not call.id:
        return CheckResult(TOOL_ROUND_TRIP, FAIL, "tool call has no id")
    if router.used(route.name) >= route.daily_token_budget:
        return CheckResult(TOOL_ROUND_TRIP, SKIP, "daily token budget spent before the answer")
    messages += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call.id, "content": TOOL_RESULT},
    ]
    second = await _ask(router, route, messages, TOOLS, pseudonyms)
    if second.error is not None:
        return CheckResult(TOOL_ROUND_TRIP, FAIL, f"tool result rejected: {_why(second.error)}")
    if not second.text.strip():
        return CheckResult(TOOL_ROUND_TRIP, FAIL, "no text answer after the tool result")
    return CheckResult(TOOL_ROUND_TRIP, PASS, "call, result and answer")
