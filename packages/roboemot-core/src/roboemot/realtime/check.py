"""Realtime route check (SPEC 8, Realtime speech route; SPEC 12; SPEC 13).

``check_realtime_route`` opens one session on ONE ``kind: realtime`` route through the same
client, session provider and URL rules as the runtime (``RealtimeClient``, ``connect_target``)
and runs, within ``llm_check.deadline_s``:

- connect: the provider's allocation, the socket, ``session.update`` and its
  ``session.updated`` (the same ``session.update`` the route sends, with the chat check's system
  prompt as instructions);
- one ``input_text`` turn with a client-chosen item id, then ``response.create``: the time from
  that request to the first audio delta (``first_audio_ms``);
- ``conversation.item.delete`` of that item id: ``accepted`` (``conversation.item.deleted``),
  ``refused`` (an ``error`` echoing the delete or the item's create), ``unconfirmed`` (no answer
  within the route's ``stall_timeout_s``) or ``not_run``.

It sends no microphone audio. The response is charged to the route's daily token budget (its
usage, else the chars/4 estimate, as the route charges); a spent budget or an unreadable key
refuses the check before anything is sent. Like the chat check it is advisory and stored next
to the route.

The report names hosts only: the allocator's (platform mode) and the connect host the provider
allocated, also when the key was refused for it (``url_key_host``), so the owner can declare
that host exactly. It never holds a key, a URL's path or query, the platform identifier's value
(only whether it was sent), server error text, audio or the reply's words. Field names avoid
the words the debug report redacts (``redact.SECRET_WORDS``).
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import logging
import queue
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ..config.schema import LLMRoute, RoboEmotConfig
from ..llm.conformance import (
    FAIL,
    PASS,
    SKIP,
    CheckResult,
    RouteBudgetExhausted,
    RouteKeyUnavailable,
    RouteKindUnsupported,
    realistic_turn,
    route_endpoint,
)
from ..llm.prompt import CHARS_PER_TOKEN
from ..llm.pseudonyms import route_is_local
from . import protocol
from .route import EVENT_PREFIX, ITEM_PREFIX, MS_PER_S, realtime_extra, session_update
from .session import NO_KEY, DirectSession, SessionProvider

log = logging.getLogger(__name__)

CONNECT = "realtime_connect"
FIRST_AUDIO = "realtime_first_audio"
ITEM_DELETE = "realtime_item_delete"
ITEM_ID = f"{ITEM_PREFIX}check_1"  # the client-chosen id of the check's input_text item
CREATE_EVENT = "ev_check_item"
RESPONSE_EVENT = f"{EVENT_PREFIX}check_1"
DELETE_EVENT = "ev_check_delete"
STALL = "stall"  # as the client's own watchdog names a response that stopped
SEND_FAILED = "send_failed"
# client failures that stop the connect before the handshake: no key left the robot then
BEFORE_HANDSHAKE = ("pseudonyms", "reconfigured")
SEND_WAITS = 2  # a send's write waits at most this many send_timeout_s (the client's own bound)


@dataclass
class _Closed:
    reason: str


@dataclass
class RealtimeReport:
    """One check of a realtime route; see the module docstring for what each field holds.
    ``header_to_connect_host``: ``sent``, ``refused`` (``url_key_host``), ``not_sent`` (the
    connect failed before) or ``none`` (the route has no key). ``failure``: the client's reason
    code when the session failed. ``server_error_code`` / ``server_error_about``: the last server
    error's code and the type of our event it names (never its message). ``error``: the check
    did not complete (``timeout``, ``budget``, ``secret``, ``provider``, ``no_extra``,
    ``failed``)."""

    route: str
    model: str
    endpoint: str
    checked_at: str
    kind: str = "realtime"
    mode: str = "direct"
    allocator_host: str | None = None
    connect_host: str | None = None
    header_to_connect_host: str = "not_sent"
    platform_id_sent: bool = False
    update_seen: bool = False
    first_audio_ms: float | None = None
    item_delete: str = "not_run"
    charged: int = 0
    failure: str | None = None
    server_error_code: str | None = None
    server_error_about: str | None = None
    checks: list[CheckResult] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        """The check completed and no check failed (skips do not count against the route)."""
        return not self.error and all(c.status != FAIL for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ok": self.ok}


def _host(url: str | None) -> str | None:
    """The host of ``url`` (lower case, no port, path or query), or None."""
    if not url:
        return None
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def _turn(cfg: RoboEmotConfig, route: LLMRoute) -> tuple[str, str]:
    """The instructions and the user text: the chat check's realistic turn. On a route that is
    not local with pseudonyms on, built from the library's defaults (no owner text)."""
    if cfg.privacy.pseudonyms_to_cloud and not route_is_local(route):
        cfg = RoboEmotConfig()
    messages = realistic_turn(cfg)
    return messages[0]["content"], messages[-1]["content"]


def check_realtime_route(
    route: LLMRoute,
    cfg: RoboEmotConfig,
    *,
    router: Any,
    provider: SessionProvider | None,
    client_factory: Callable[..., Any] | None = None,
    extra: Callable[[], bool] = realtime_extra,
    deadline_s: float | None = None,
    config: Callable[[], RoboEmotConfig] | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> RealtimeReport:
    """Run the check (module docstring) and return its report; blocking, never on an event
    loop. ``router`` gives the key (``auth_headers``) and the daily budget (``used``,
    ``charge``); ``provider`` is the session provider ``voice.realtime_route.mode`` selects
    (None in platform mode without one: ``error: provider``). Raises
    :class:`RouteKindUnsupported`, :class:`RouteKeyUnavailable` and
    :class:`RouteBudgetExhausted` before anything is sent. ``deadline_s`` defaults to
    ``llm_check.deadline_s`` and bounds the waits after the connect; the connect is bounded
    by the client's own ``connect_timeout_s + send_timeout_s``, and its close by about three
    ``send_timeout_s`` after that. ``config`` returns the committed configuration: the
    privacy gates (pseudonyms, the platform identifier) are read from it at each use, as the
    route reads them (``cfg`` when omitted). ``mode`` is the provider's (direct when it is a
    :class:`DirectSession`), not the configuration's: ``mode`` takes effect at a restart."""
    if route.kind != "realtime":
        raise RouteKindUnsupported(f"route {route.name} is a {route.kind} route")
    problem = router.key_problem(route)
    if problem is not None:
        raise RouteKeyUnavailable(f"route {route.name}: {problem}")
    if router.used(route.name) >= route.daily_token_budget:
        raise RouteBudgetExhausted(f"route {route.name}: daily token budget is spent")
    rt = cfg.voice.realtime_route
    report = RealtimeReport(
        route.name, route.model, route_endpoint(route), now().isoformat(timespec="seconds")
    )
    if route.api_key in NO_KEY:
        report.header_to_connect_host = "none"
    report.mode = "direct" if isinstance(provider, DirectSession) else "platform"
    report.allocator_host = _host(route.base_url) if report.mode == "platform" else None
    if provider is None:
        report.error = "provider"
        return report
    live = config or (lambda: cfg)
    if client_factory is None:
        if not extra():
            report.error = "no_extra"
            return report
        from .client import RealtimeClient  # needs the realtime extra

        client_factory = RealtimeClient
    end = clock() + (cfg.llm_check.deadline_s if deadline_s is None else deadline_s)
    run = _Run(route, cfg, router, report, clock, end)
    client = client_factory(
        route,
        rt,
        provider,
        auth_headers=lambda: router.auth_headers(route),
        on_event=run.events.put,
        on_close=lambda reason: run.events.put(_Closed(reason)),
        local_only=lambda: live().privacy.pseudonyms_to_cloud,
        platform_id=lambda: live().privacy.realtime_platform_id,
    )
    try:
        run.go(client)
    finally:
        client.close()
    log.info(
        "realtime route %s check: %s",
        route.name,
        report.error or ", ".join(f"{c.name}={c.status}" for c in report.checks),
    )
    return report


class _Run:
    """The check's steps over one client; the client thread feeds ``events``."""

    def __init__(
        self,
        route: LLMRoute,
        cfg: RoboEmotConfig,
        router: Any,
        report: RealtimeReport,
        clock: Callable[[], float],
        end: float,
    ) -> None:
        self.route, self.cfg, self.router, self.report = route, cfg, router, report
        self.clock, self.end = clock, end
        self.events: queue.Queue[Any] = queue.Queue()
        self.create_refused = False
        self.reply_chars = 0

    def go(self, client: Any) -> None:
        instructions, text = _turn(self.cfg, self.route)
        ok = client.connect(session_update(self.cfg, instructions))
        self._hosts(client.snapshot(), client.reason)
        r = self.report
        r.update_seen = ok
        if not ok:
            r.failure = client.reason
            r.checks.append(CheckResult(CONNECT, FAIL, client.reason or "closed"))
            return
        r.checks.append(CheckResult(CONNECT, PASS, ""))
        if not self._sent(client, {**protocol.text_item(text, ITEM_ID), "event_id": CREATE_EVENT}):
            return
        sent_at = self.clock()
        if not self._sent(client, protocol.response_create(event_id=RESPONSE_EVENT)):
            return
        done = self._response(sent_at)
        self._charge(done, len(instructions) + len(text))
        if r.error or r.failure is not None:
            return  # the response did not complete: the session is not trusted further
        self._delete(client)

    def _sent(self, client: Any, event: dict[str, Any]) -> bool:
        """Send ``event`` and wait for the write (bounded by the client's ``send_timeout_s``);
        a failed write ends the check at once (``send_failed``), not at a stall."""
        wait_s = self.cfg.voice.realtime_route.send_timeout_s * SEND_WAITS
        try:
            ok = bool(client.send(event).result(wait_s))
        except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
            ok = False
        if not ok:
            self.report.failure = SEND_FAILED
            self.report.checks.append(CheckResult(FIRST_AUDIO, FAIL, SEND_FAILED))
        return ok

    def _hosts(self, snap: dict[str, Any], reason: str | None) -> None:
        r = self.report
        r.connect_host = _host(snap.get("endpoint")) or snap.get("refused_host")
        r.platform_id_sent = bool(snap.get("platform_id_sent"))
        r.server_error_code = snap.get("server_error_code")
        r.server_error_about = snap.get("server_error_about")
        if r.header_to_connect_host == "none":
            return
        if snap.get("key_sent") and reason not in BEFORE_HANDSHAKE:
            r.header_to_connect_host = "sent"
        elif reason == "url_key_host":
            r.header_to_connect_host = "refused"

    def _next(self, wait_s: float, *, deadline_is_error: bool = True) -> Any:
        """The next event within ``wait_s`` and the deadline; None past either (``error:
        timeout`` when it was the deadline, unless ``deadline_is_error`` is False)."""
        left = self.end - self.clock()
        if left <= 0:
            if deadline_is_error:
                self.report.error = "timeout"
            return None
        try:
            return self.events.get(timeout=min(wait_s, left))
        except queue.Empty:
            if deadline_is_error and self.clock() >= self.end:
                self.report.error = "timeout"
            return None

    def _response(self, sent_at: float) -> protocol.ResponseDone | None:
        """Follow the response to its ``response.done`` (the first audio delta is timed);
        None when it ended otherwise (``failure`` says why)."""
        r, route = self.report, self.route
        created = False
        while True:
            event = self._next(route.stall_timeout_s if created else route.first_token_timeout_s)
            if event is None:
                if not r.error:
                    r.failure = STALL
                break
            if isinstance(event, _Closed):
                r.failure = event.reason
                break
            if isinstance(event, protocol.ServerError):
                if event.event_id == CREATE_EVENT:
                    self.create_refused = True
                if event.event_id == RESPONSE_EVENT:
                    r.failure = "refused"
                    break
                continue
            if isinstance(event, protocol.ResponseCreated):
                created = True
            elif isinstance(event, protocol.AudioDelta) and r.first_audio_ms is None:
                r.first_audio_ms = round((self.clock() - sent_at) * MS_PER_S, 1)
            elif isinstance(event, protocol.ReplyTranscript):
                self.reply_chars += len(event.text)  # counted for the estimate, never kept
            elif isinstance(event, protocol.ResponseDone):
                self._first_audio()
                return event
        self._first_audio()
        return None

    def _first_audio(self) -> None:
        r = self.report
        if r.first_audio_ms is not None:
            r.checks.append(CheckResult(FIRST_AUDIO, PASS, f"{r.first_audio_ms:g} ms"))
        else:
            r.checks.append(CheckResult(FIRST_AUDIO, FAIL, r.failure or r.error or "no audio"))

    def _charge(self, done: protocol.ResponseDone | None, sent_chars: int) -> None:
        """The response's usage, else the chars/4 estimate of what was sent and heard."""
        tokens = done.total_tokens if done is not None else None
        if tokens is None:
            tokens = (sent_chars + self.reply_chars) // CHARS_PER_TOKEN
        if tokens > 0:
            self.router.charge(self.route.name, tokens)
        self.report.charged = tokens

    def _delete(self, client: Any) -> None:
        """``conversation.item.delete`` of the check's item, and the server's answer."""
        r = self.report
        if self.create_refused:
            r.item_delete = "refused"
            r.checks.append(CheckResult(ITEM_DELETE, FAIL, "item create refused"))
            return
        client.send({**protocol.item_delete(ITEM_ID), "event_id": DELETE_EVENT})
        verdict = "unconfirmed"
        while True:
            # the deadline here is like no answer: the route answered, the delete is unconfirmed
            event = self._next(self.route.stall_timeout_s, deadline_is_error=False)
            if event is None or isinstance(event, _Closed):
                break
            if isinstance(event, protocol.ItemDeleted) and event.item_id == ITEM_ID:
                verdict = "accepted"
                break
            if isinstance(event, protocol.ServerError) and event.event_id in (
                DELETE_EVENT,
                CREATE_EVENT,
            ):
                verdict = "refused"
                break
        r.item_delete = verdict
        status = {"accepted": PASS, "refused": FAIL}.get(verdict, SKIP)
        r.checks.append(CheckResult(ITEM_DELETE, status, verdict))
