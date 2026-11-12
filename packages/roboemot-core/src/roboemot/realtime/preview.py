"""Voice preview of a realtime route (SPEC 8, Voice preview; SPEC 12).

``preview_voice`` opens one short session on a ``kind: realtime`` route through the realtime
check's path (``realtime.check``: the same client, session provider, URL and key rules, the
direct-mode swap of a ``gemini_live`` route, the up-front refusals and the charge), with the
route's ``voice`` replaced on a copy, and asks the server's voice to say one fixed sentence
(``prompts/voice_preview.<lang>.txt`` in ``conversation.language``, else English) word for
word. It returns the reply's audio as a WAV file for the web UI to play.

Only text leaves the robot: the sentence and the line rule, plus on a ``gemini_live`` route
the check's offline tools (no name in view). A tool call is answered with the check's
``{"ok": true}`` (no skill runs), up to ``conversation.max_tool_rounds``. Each response is
charged its estimate to the route's daily budget at its ``response.done``; the preview does
not wait for a late token total: one that comes while the preview still reads is charged
once (``_Run._usage``), one after the close is never read and the estimate stays.

The audio is held in memory up to ``voice.realtime_route.preview_max_bytes`` and handed to the
caller only; it is never logged, stored or kept. The reply's words are counted, never kept. A
failure is a code (:class:`PreviewFailed`), never a server's text.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from collections.abc import Callable
from typing import Any

from ..config.schema import LLMRoute, RoboEmotConfig, realtime_connect_timeout_s
from ..llm.conformance import route_endpoint
from ..llm.prompt import LANGUAGE_NAMES, preview_line
from ..llm.pseudonyms import route_is_local
from . import protocol
from .check import (
    SEND_FAILED,
    SETUP_UNREADY,
    RealtimeReport,
    _Closed,
    _Run,
    offline_tools,
    refuse_unavailable,
)
from .route import EVENT_PREFIX, PCM_SAMPLE_BYTES, realtime_extra, session_update
from .session import GEMINI_LIVE, DirectSession, SessionProvider
from .turn import line_instructions

log = logging.getLogger(__name__)

# Gemini Live's 30 prebuilt voices (the owner's list, 2026-11-11).
GEMINI_VOICES = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)
# Qwen3-TTS's voices, which the platform's openai_realtime server speaks (the owner's list,
# 2026-11-11).
QWEN3_VOICES = (
    "Aiden",
    "Ryan",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Serena",
    "Sohee",
    "Uncle_Fu",
    "Vivian",
)
# The voices the web UI offers per route protocol; any other name may still be typed.
KNOWN_VOICES = {GEMINI_LIVE: GEMINI_VOICES, "openai_realtime": QWEN3_VOICES}
PREVIEW_EVENT = f"{EVENT_PREFIX}preview_1"
WAV_CHANNELS = 1
OFFLINE = "offline"  # the router knows the network is down
PSEUDONYMS = "pseudonyms"  # as the client names it: real names may not reach a cloud host
AUDIO_CAP = "audio_cap"
NO_AUDIO = "no_audio"
TOOL_ROUNDS = "tool_rounds_exceeded"


class PreviewFailed(RuntimeError):
    """The preview gave no audio: ``code`` says why (a reason code, never a server's text);
    ``close_code`` is the server's close code when it closed the connect (1008: refused,
    perhaps a second session on the free tier)."""

    def __init__(self, code: str, close_code: int | None = None) -> None:
        super().__init__(code)
        self.code, self.close_code = code, close_code


def preview_voice(
    route: LLMRoute,
    voice: str,
    cfg: RoboEmotConfig,
    *,
    router: Any,
    provider: SessionProvider | None,
    tools: Callable[[], list[dict[str, Any]] | None] | None = None,
    client_factory: Callable[..., Any] | None = None,
    extra: Callable[[], bool] = realtime_extra,
    deadline_s: float | None = None,
    config: Callable[[], RoboEmotConfig] | None = None,
    ready: Callable[[bytes | PreviewFailed], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """Say the preview sentence in ``voice`` on ``route`` and return it as a WAV file
    (module docstring); blocking, never on an event loop. Refuses before anything is sent as
    the check does (``refuse_unavailable``), and with :class:`PreviewFailed` when the router
    is offline or pseudonyms keep a cloud route closed; raises :class:`PreviewFailed` on any
    other failure. ``deadline_s`` (default ``llm_check.deadline_s``) bounds the whole
    preview, its connect included; ``ready`` gets the audio, or the session's failure, before
    the session's close. The stored route and ``cfg`` are never changed."""
    refuse_unavailable(route, router)
    live = config or (lambda: cfg)
    if router.offline:
        raise PreviewFailed(OFFLINE)
    if live().privacy.pseudonyms_to_cloud and not route_is_local(route):
        raise PreviewFailed(PSEUDONYMS)
    route = route.model_copy(update={"voice": voice})
    if route.protocol == GEMINI_LIVE:
        provider = DirectSession()  # its key goes to its own origin only, never an allocator
    if provider is None:
        raise PreviewFailed("provider")
    if client_factory is None:
        if not extra():
            raise PreviewFailed("no_extra")
        from .client import RealtimeClient  # needs the realtime extra

        client_factory = RealtimeClient
    end = clock() + (cfg.llm_check.deadline_s if deadline_s is None else deadline_s)
    report = RealtimeReport(route.name, route.model, route_endpoint(route), "")
    run = _Preview(route, cfg, router, report, clock, end)
    run.tools = tools or (lambda: offline_tools(cfg))
    client = client_factory(
        route,
        cfg.voice.realtime_route,
        provider,
        auth_headers=lambda: router.auth_headers(route),
        on_event=run.events.put,
        on_close=lambda reason: run.events.put(_Closed(reason)),
        local_only=lambda: live().privacy.pseudonyms_to_cloud,
        platform_id=lambda: live().privacy.realtime_platform_id,
    )
    try:
        wav = run.go(client)
    except PreviewFailed as exc:
        log.info("realtime route %s: voice preview failed: %s", route.name, exc.code)
        if ready is not None:
            ready(exc)
        raise
    else:
        log.info("realtime route %s: voice preview, %d bytes", route.name, len(wav))
        if ready is not None:
            ready(wav)
        return wav
    finally:
        client.close()


class _Preview(_Run):
    """The check's steps, keeping the reply's audio (capped) and the session's output rate."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.pcm = bytearray()
        self.rate_hz: int | None = None
        self.response_event = PREVIEW_EVENT

    def go(self, client: Any) -> bytes:  # type: ignore[override]
        """Connect with the line rule as instructions, ask for the line, answer any calls,
        and return the audio as WAV; :class:`PreviewFailed` when no audio came."""
        lang, sentence = preview_line(self.cfg)
        instructions = line_instructions("", sentence, LANGUAGE_NAMES.get(lang, lang))
        started = self.clock()
        rt = self.cfg.voice.realtime_route
        tools: list[dict[str, Any]] | None = None
        if self.gemini:  # its setup is all the session gets: without tools it is refused
            tools = self._tools()
            if tools is None:
                raise PreviewFailed(SETUP_UNREADY)
        budget = realtime_connect_timeout_s(self.route, rt) - (self.clock() - started)
        left = self.end - self.clock() - rt.send_timeout_s  # the connect's own close bound
        update = session_update(
            self.cfg,
            instructions,
            tools,
            create_response=False,
            interrupt_response=False,
            route=self.route,
        )
        if not client.connect(update, timeout_s=max(0.0, min(budget, left))):
            raise PreviewFailed(client.reason or "closed", client.snapshot().get("close_code"))
        # no text item: with none, Gemini's codec sends the instructions as the turn's line
        if not self._sent(client, protocol.response_create(instructions, event_id=PREVIEW_EVENT)):
            raise PreviewFailed(SEND_FAILED)
        sent_at = self.clock()
        done = self._response(sent_at)
        self._charge(done, len(instructions) * 2)  # the setup's and the request's
        rounds, cap = 0, self.cfg.conversation.max_tool_rounds
        while done is not None and self.calls and rounds < cap:
            rounds += 1
            sent = self._answer_calls(client, rounds)
            if sent is None:
                raise PreviewFailed(SEND_FAILED)
            done = self._response(sent_at)
            self._charge(done, sent)
        r = self.report
        if r.failure or r.error:
            close_code = client.snapshot().get("close_code")
            raise PreviewFailed(r.failure or r.error, close_code)
        if not self.pcm:
            raise PreviewFailed(TOOL_ROUNDS if self.calls else NO_AUDIO)
        return _wav(bytes(self.pcm), self.rate_hz or protocol.NATIVE_OUTPUT_RATE_HZ)

    def _audio(self, pcm: bytes) -> bool:
        """Keep the chunk unless it would pass ``preview_max_bytes`` (``audio_cap``)."""
        if len(self.pcm) + len(pcm) > self.cfg.voice.realtime_route.preview_max_bytes:
            self.report.failure = AUDIO_CAP
            return False
        self.pcm += pcm
        return True

    def _session(self, event: protocol.SessionEvent) -> None:
        """The output rate the server says (Gemini's codec: 24 kHz); else the native one."""
        if event.output_rate_hz:
            self.rate_hz = event.output_rate_hz


def _wav(pcm: bytes, rate_hz: int) -> bytes:
    """``pcm`` (PCM16 mono at ``rate_hz``) as a WAV file in memory, never on disk."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(WAV_CHANNELS)
        w.setsampwidth(PCM_SAMPLE_BYTES)
        w.setframerate(rate_hz)
        w.writeframes(pcm)
    return out.getvalue()
