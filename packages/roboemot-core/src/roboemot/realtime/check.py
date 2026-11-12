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

A ``gemini_live`` route (SPEC 8, Gemini Live wire) is checked in direct mode only, whatever
provider is passed: its key goes to its own origin, never to a session allocator. Its setup
carries the tools a turn offers (:func:`offline_tools`: the default skills, the skip and tone
tools, with no name in view, SPEC 12); a tool call is answered (never run) and the
continuation is followed, up to ``conversation.max_tool_rounds`` (calls still open past it:
``tool_rounds_exceeded``; a setup with no tool: ``no_tools``, both advisory). Its item delete
is the codec's own ack: ``emulated``, not a server confirmation. Its tokens come after the
response ended (``UsageReport``): the estimate is charged at each ``response.done`` and the
late total charges only what it adds; the check waits for it at most
``voice.realtime_route.check_usage_wait_s``, and never past the deadline
(``usage_reported``).

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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from ..config.schema import LLMRoute, RoboEmotConfig, realtime_connect_timeout_s
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
from ..skills import SkillRegistry
from ..skills.defaults import SkillEnv, default_skills
from . import protocol
from .route import EVENT_PREFIX, ITEM_PREFIX, MS_PER_S, realtime_extra, session_update
from .session import GEMINI_LIVE, NO_KEY, DirectSession, SessionProvider
from .turn import TONE_NOTED, session_tools

log = logging.getLogger(__name__)

CONNECT = "realtime_connect"
FIRST_AUDIO = "realtime_first_audio"
ITEM_DELETE = "realtime_item_delete"
TOOLS = "realtime_tools"  # advisory: a gemini_live setup without tools, a round cap reached
ITEM_ID = f"{ITEM_PREFIX}check_1"  # the client-chosen id of the check's input_text item
CREATE_EVENT = "ev_check_item"
RESPONSE_EVENT = f"{EVENT_PREFIX}check_1"
DELETE_EVENT = "ev_check_delete"
STALL = "stall"  # as the client's own watchdog names a response that stopped
SEND_FAILED = "send_failed"
SETUP_UNREADY = "setup_unready"  # as the route names a Gemini setup without the turn's tools
# the delete's verdict when it ran; ``emulated``: the wire codec acked it (Gemini has no delete)
ITEM_VERDICTS = ("accepted", "refused", "unconfirmed", "emulated")
CALL_OUTPUT = TONE_NOTED  # the answer to a call: never run, acknowledged as the tone tool is
# Every ``CheckResult.code`` this module gives (the page words each; a test walks the module's
# syntax tree and holds the calls to this list).
REASON_CODES = (
    "connected",
    "session_failed",
    "first_audio",
    "no_audio",
    "item_create_refused",
    *(f"item_delete_{v}" for v in ITEM_VERDICTS),
    "no_tools",
    "tool_rounds_exceeded",
)
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
    ``failed``). ``setup_ms``: the connect, from the tools' build to ``session.updated``.
    ``close_code``: the server's close code when it closed the connect (never its reason
    text). ``tools_sent``: the tools of a ``gemini_live`` setup. ``usage_reported``: the late
    token totals a ``gemini_live`` server reported (None: none came in time)."""

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
    protocol: str = "openai_realtime"
    setup_ms: float | None = None
    close_code: int | None = None
    tools_sent: int = 0
    usage_reported: int | None = None

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


def offline_tools(
    cfg: RoboEmotConfig, gestures: Callable[[], Sequence[str]] | None = None
) -> list[dict[str, Any]]:
    """The tools a ``gemini_live`` check's setup carries: the default skills listed in
    ``conversation.llm_skills``, built over an environment that acts on nothing, and the skip
    and tone tools, as the worker offers them (``session_tools``). ``gestures``: the robot's
    playable names (the affect labels when None). No name in view ever goes in a choice list
    (``look_at`` offers its categories only, SPEC 12)."""
    labels = list(cfg.affect.labels)
    env = SkillEnv(
        gestures=gestures or (lambda: labels),
        play=lambda _name: False,
        look_names=list,
        look_at=lambda _target: False,
        affect=SimpleNamespace(cfg=cfg.affect),
    )
    conv = cfg.conversation
    offered = default_skills(env).offered(conv.llm_skills)
    chat = SkillRegistry.tools(
        offered, enum_max=conv.tool_enum_max_items, max_chars=conv.tool_arg_max_chars
    )
    functions, toned = session_tools(chat, cfg)
    return functions + toned


def refuse_unavailable(route: LLMRoute, router: Any) -> None:
    """Refuse ``route`` before anything is sent: not a ``kind: realtime`` route
    (:class:`RouteKindUnsupported`), its key unreadable (:class:`RouteKeyUnavailable`) or
    its daily budget spent (:class:`RouteBudgetExhausted`). The check and the voice preview
    (``realtime.preview``) both start here."""
    if route.kind != "realtime":
        raise RouteKindUnsupported(f"route {route.name} is a {route.kind} route")
    problem = router.key_problem(route)
    if problem is not None:
        raise RouteKeyUnavailable(f"route {route.name}: {problem}")
    if router.used(route.name) >= route.daily_token_budget:
        raise RouteBudgetExhausted(f"route {route.name}: daily token budget is spent")


def check_realtime_route(
    route: LLMRoute,
    cfg: RoboEmotConfig,
    *,
    router: Any,
    provider: SessionProvider | None,
    tools: Callable[[], list[dict[str, Any]] | None] | None = None,
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
    (None in platform mode without one: ``error: provider``); a ``gemini_live`` route always
    uses :class:`DirectSession` instead. ``tools()`` builds a ``gemini_live`` setup's tools
    (None from it, or an exception, fails the connect: ``setup_unready``); without it they are
    :func:`offline_tools` of ``cfg``. Raises
    :class:`RouteKindUnsupported`, :class:`RouteKeyUnavailable` and
    :class:`RouteBudgetExhausted` before anything is sent. ``deadline_s`` defaults to
    ``llm_check.deadline_s``; no wait after the connect goes past it; the connect is bounded
    by the client's own ``connect_timeout_s`` (the route's, else the shared one) plus
    ``send_timeout_s``, and its close by about three
    ``send_timeout_s`` after that. ``config`` returns the committed configuration: the
    privacy gates (pseudonyms, the platform identifier) are read from it at each use, as the
    route reads them (``cfg`` when omitted). ``mode`` is the provider's (direct when it is a
    :class:`DirectSession`), not the configuration's: ``mode`` takes effect at a restart."""
    refuse_unavailable(route, router)
    rt = cfg.voice.realtime_route
    report = RealtimeReport(
        route.name, route.model, route_endpoint(route), now().isoformat(timespec="seconds")
    )
    report.protocol = route.protocol
    if route.protocol == GEMINI_LIVE:
        provider = DirectSession()  # its key goes to its own origin only, never an allocator
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
    run.tools = tools or (lambda: offline_tools(cfg))
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
        # a close during the turn has its code too
        report.close_code = client.snapshot().get("close_code") or report.close_code
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
        self.gemini = route.protocol == GEMINI_LIVE
        self.tools: Callable[[], list[dict[str, Any]] | None] = lambda: None
        self.create_refused = False
        self.reply_chars = 0
        self.response_event = RESPONSE_EVENT
        self.calls: list[str] = []  # the call ids of the last response (never their arguments)
        self.estimates: dict[str, int] = {}  # response id -> estimate a late total corrects
        self.reported: set[str] = set()
        self.usage_seen = False  # a usage report came after the last response.done

    def go(self, client: Any) -> None:
        instructions, text = _turn(self.cfg, self.route)
        r = self.report
        started = self.clock()
        budget = realtime_connect_timeout_s(self.route, self.cfg.voice.realtime_route)
        tools: list[dict[str, Any]] | None = None
        timeout: dict[str, float] = {}
        if self.gemini:  # its setup is all the session gets: the tools a turn offers
            tools = self._tools()
            if tools is None:
                r.failure = SETUP_UNREADY
                r.checks.append(
                    CheckResult(
                        CONNECT, FAIL, SETUP_UNREADY, "session_failed", {"failure": SETUP_UNREADY}
                    )
                )
                return
            r.tools_sent = len(tools)
            timeout = {"timeout_s": max(0.0, budget - (self.clock() - started))}
        update = session_update(
            self.cfg,
            instructions,
            tools,
            create_response=False,
            interrupt_response=False,
            route=self.route,
        )
        ok = client.connect(update, **timeout)
        snap = client.snapshot()
        self._hosts(snap, client.reason)
        r.close_code = snap.get("close_code")
        r.update_seen = ok
        if not ok:
            r.failure = client.reason
            why = client.reason or "closed"
            r.checks.append(CheckResult(CONNECT, FAIL, why, "session_failed", {"failure": why}))
            return
        r.setup_ms = round((self.clock() - started) * MS_PER_S, 1)
        r.checks.append(CheckResult(CONNECT, PASS, "", "connected"))
        if self.gemini and not tools:
            r.checks.append(CheckResult(TOOLS, SKIP, "no tools", "no_tools"))
        if not self._sent(client, {**protocol.text_item(text, ITEM_ID), "event_id": CREATE_EVENT}):
            return
        sent_at = self.clock()
        if not self._sent(client, protocol.response_create(event_id=RESPONSE_EVENT)):
            return
        done = self._response(sent_at)
        self._charge(done, len(instructions) + len(text))
        rounds, cap = 0, self.cfg.conversation.max_tool_rounds
        while self.gemini and done is not None and self.calls and rounds < cap:
            rounds += 1
            sent = self._answer_calls(client, rounds)
            if sent is None:
                return
            done = self._response(sent_at)
            self._charge(done, sent)
        capped = self.gemini and done is not None and bool(self.calls)
        if capped:  # Gemini still waits on calls: no turn total is coming
            why = f"{len(self.calls)} calls open after {rounds} rounds"
            r.checks.append(
                CheckResult(TOOLS, SKIP, why, "tool_rounds_exceeded", {"rounds": rounds})
            )
        self._first_audio()
        if r.error or r.failure is not None:
            return  # the response did not complete: the session is not trusted further
        self._delete(client)
        if not capped:
            self._await_usage()

    def _tools(self) -> list[dict[str, Any]] | None:
        """The setup's tools, or None when their build failed (logged by type only)."""
        try:
            return self.tools()
        except Exception as exc:
            log.warning(
                "realtime route %s check: setup tools: %s", self.route.name, type(exc).__name__
            )
            return None

    def _answer_calls(self, client: Any, round_no: int) -> int | None:
        """Answer the last response's calls (``CALL_OUTPUT``: no skill runs) and ask for the
        continuation; the characters sent, or None when a write failed."""
        calls, self.calls = self.calls, []
        for call_id in calls:
            if not self._sent(client, protocol.function_output(call_id, CALL_OUTPUT)):
                return None
        self.response_event = f"{RESPONSE_EVENT}_{round_no}"
        if not self._sent(client, protocol.response_create(event_id=self.response_event)):
            return None
        return len(CALL_OUTPUT) * len(calls)

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
            self.report.checks.append(
                CheckResult(
                    FIRST_AUDIO, FAIL, SEND_FAILED, "session_failed", {"failure": SEND_FAILED}
                )
            )
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
            event = self.events.get(timeout=min(wait_s, left))
        except queue.Empty:
            if deadline_is_error and self.clock() >= self.end:
                self.report.error = "timeout"
            return None
        if isinstance(event, protocol.UsageReport):
            self._usage(event)
        return event

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
                if event.event_id == self.response_event:
                    r.failure = "refused"
                    break
                continue
            if isinstance(event, protocol.ResponseCreated):
                created = True
            elif isinstance(event, protocol.AudioDelta):
                if r.first_audio_ms is None:
                    r.first_audio_ms = round((self.clock() - sent_at) * MS_PER_S, 1)
                if not self._audio(event.pcm):
                    break
            elif isinstance(event, protocol.SessionEvent):
                self._session(event)
            elif isinstance(event, protocol.ReplyTranscript):
                self.reply_chars += len(event.text)  # counted for the estimate, never kept
            elif isinstance(event, protocol.FunctionCall):
                self.calls.append(event.call_id)
            elif isinstance(event, protocol.ResponseDone):
                self.usage_seen = False
                return event
        return None

    def _audio(self, pcm: bytes) -> bool:
        """A reply audio chunk: the check keeps nothing (never audio, SPEC 12); False ends
        the response here (a subclass's cap, with ``failure`` set)."""
        return True

    def _session(self, event: protocol.SessionEvent) -> None:
        """A ``session.created`` / ``session.updated`` the connect left queued: unused here."""

    def _first_audio(self) -> None:
        r = self.report
        if r.first_audio_ms is not None:
            ms = r.first_audio_ms
            r.checks.append(CheckResult(FIRST_AUDIO, PASS, f"{ms:g} ms", "first_audio", {"ms": ms}))
        elif r.failure or r.error:
            why = r.failure or r.error
            r.checks.append(CheckResult(FIRST_AUDIO, FAIL, why, "session_failed", {"failure": why}))
        else:
            r.checks.append(CheckResult(FIRST_AUDIO, FAIL, "no audio", "no_audio"))

    def _charge(self, done: protocol.ResponseDone | None, sent_chars: int) -> None:
        """The response's usage, else the chars/4 estimate of what was sent and heard (kept
        by response id on a ``gemini_live`` route, for its late total to correct)."""
        tokens = done.total_tokens if done is not None else None
        rid = done.response_id if done is not None else None
        if tokens is None:
            tokens = (sent_chars + self.reply_chars) // CHARS_PER_TOKEN
            if self.gemini and rid in self.reported:
                tokens = 0  # its late total came first and was charged in full
            elif self.gemini and rid is not None:
                self.estimates[rid] = tokens
        self.reply_chars = 0
        if tokens > 0:
            self.router.charge(self.route.name, tokens)
        self.report.charged += tokens

    def _usage(self, report: protocol.UsageReport) -> None:
        """A late token total (Gemini's ``turnComplete``), as the route takes it: the
        estimates already charged for its responses come off it, the rest is charged; a
        report naming only responses already reported charges nothing. One naming none covers
        every response still at its estimate (all of them come off: never charged twice)."""
        self.usage_seen = True
        fresh = [i for i in report.response_ids if i not in self.reported]
        if report.response_ids and not fresh:
            return
        if not report.response_ids:
            fresh = list(self.estimates)
        self.reported.update(fresh)
        extra = max(0, report.total_tokens - sum(self.estimates.pop(i, 0) for i in fresh))
        if extra > 0:
            self.router.charge(self.route.name, extra)
        r = self.report
        r.charged += extra
        r.usage_reported = (r.usage_reported or 0) + report.total_tokens

    def _await_usage(self) -> None:
        """Wait for the late total of the responses still at their estimate, until a usage
        report came after the last ``response.done``, at most ``check_usage_wait_s`` and never
        past the deadline (no error either way); a close ends the wait."""
        until = self.clock() + self.cfg.voice.realtime_route.check_usage_wait_s
        while self.estimates and not self.usage_seen:
            left = until - self.clock()
            if left <= 0:
                return
            event = self._next(left, deadline_is_error=False)
            if event is None or isinstance(event, _Closed):
                return

    def _delete(self, client: Any) -> None:
        """``conversation.item.delete`` of the check's item, and the server's answer."""
        r = self.report
        if self.create_refused:
            r.item_delete = "refused"
            why = "item create refused"
            r.checks.append(CheckResult(ITEM_DELETE, FAIL, why, "item_create_refused"))
            return
        client.send({**protocol.item_delete(ITEM_ID), "event_id": DELETE_EVENT})
        verdict = "unconfirmed"
        while True:
            # the deadline here is like no answer: the route answered, the delete is unconfirmed
            event = self._next(self.route.stall_timeout_s, deadline_is_error=False)
            if event is None or isinstance(event, _Closed):
                break
            if isinstance(event, protocol.ItemDeleted) and event.item_id == ITEM_ID:
                verdict = "emulated" if event.emulated else "accepted"
                break
            if isinstance(event, protocol.ServerError) and event.event_id in (
                DELETE_EVENT,
                CREATE_EVENT,
            ):
                verdict = "refused"
                break
        r.item_delete = verdict
        status = {"accepted": PASS, "refused": FAIL}.get(verdict, SKIP)
        r.checks.append(CheckResult(ITEM_DELETE, status, verdict, "item_delete_" + verdict))
