"""OpenAI-compatible speech backend, ``POST {base_url}/audio/speech`` (SPEC 8, TTS backends).

Follows the LLM routes' rules: it uses the route named by ``voice.openai_speech.route`` (URL,
key, ``first_token_timeout_s`` for the first audio chunk, ``stall_timeout_s`` for later gaps),
is eligible only when the router would allow that route (reachable, daily budget left, and
``allowed_for_minors`` while a minor is present), and charges the characters it sends against the
route's budget at ``CHARS_PER_TOKEN`` per token. Failures before the first chunk raise
:class:`~roboemot.voice.tts.TTSError`, so :class:`~roboemot.voice.tts.FallbackTTS` answers with
Piper. Only the sentence text leaves the robot (SPEC 12); it and the response body are never
logged.

``response_format: pcm`` is documented as 24 kHz, 16-bit, mono little-endian.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import numpy as np

from ..config.schema import LLMRoute, OpenAISpeechConfig, VoiceConfig
from ..llm.prompt import CHARS_PER_TOKEN
from ..llm.router import Router
from .prosody import Delivery, to_openai
from .tts import AudioChunk, TTSError, TTSUnavailable

log = logging.getLogger(__name__)

OPENAI_PCM_RATE_HZ = 24000
INT16_SCALE = 32768.0
BYTES_PER_SAMPLE = 2
SPEECH_PATH = "/audio/speech"
NO_KEY = "none"


class OpenAISpeechBackend:
    """:class:`~roboemot.voice.tts.TTSBackend` for a remote ``/audio/speech`` route."""

    name = "openai_speech"

    def __init__(
        self,
        config: OpenAISpeechConfig,
        router: Router,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.config, self.router = config, router
        self.http_client_factory = http_client_factory
        self.controls = frozenset({"instructions"} if config.instructable else ())

    def _route(self) -> LLMRoute | None:
        return next((r for r in self.router.routes if r.name == self.config.route), None)

    def available(self, *, minor_present: bool = False) -> bool:
        """The route exists and the router would use it now (online, budget, minors rule)."""
        route = self._route()
        return route is not None and any(
            r.name == route.name for r in self.router.eligible(minor_present)
        )

    async def synthesize(
        self,
        text: str,
        delivery: Delivery,
        *,
        language: str,
        voice: VoiceConfig,
        minor_present: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        route = self._route()
        if route is None or not self.available(minor_present=minor_present):
            raise TTSUnavailable("speech route not eligible")
        body: dict[str, Any] = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": text,
            "response_format": "pcm",
        }
        if self.config.instructable:
            body.update(to_openai(delivery))
        key = self.router.api_key(route)
        headers = {"Authorization": f"Bearer {key}"} if key and key != NO_KEY else {}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + route.first_token_timeout_s
        client = self.http_client_factory() if self.http_client_factory else httpx.AsyncClient()
        url = route.base_url.rstrip("/") + SPEECH_PATH
        response_cm = client.stream("POST", url, json=body, headers=headers)
        entered = False
        got_audio = False
        try:
            try:
                response = await asyncio.wait_for(
                    response_cm.__aenter__(), max(0.0, deadline - loop.time())
                )
                entered = True
                if response.status_code >= httpx.codes.BAD_REQUEST:
                    self._report(route, True)
                    raise TTSError(f"speech route answered HTTP {response.status_code}")
                chunks = response.aiter_bytes().__aiter__()
                carry = b""
                while True:
                    timeout = route.stall_timeout_s if got_audio else deadline - loop.time()
                    try:
                        data = await asyncio.wait_for(chunks.__anext__(), max(0.0, timeout))
                    except StopAsyncIteration:
                        break
                    data = carry + data
                    usable = len(data) - len(data) % BYTES_PER_SAMPLE
                    carry = data[usable:]
                    if not usable:
                        continue
                    got_audio = True
                    pcm = np.frombuffer(data[:usable], dtype="<i2")
                    yield AudioChunk(pcm.astype(np.float32) / INT16_SCALE, OPENAI_PCM_RATE_HZ)
                if not got_audio:
                    raise TTSError("speech route sent no audio")
                self._report(route, True)
            except TimeoutError as exc:
                if self.router.monitor:
                    self.router.monitor.recheck(route.name)
                raise TTSError("speech route timed out") from exc
            except httpx.TimeoutException as exc:
                if self.router.monitor:
                    self.router.monitor.recheck(route.name)
                raise TTSError("speech route timed out") from exc
            except (httpx.TransportError, OSError) as exc:
                self._report(route, False)
                raise TTSError(f"speech route unreachable: {type(exc).__name__}") from exc
        finally:
            self.router.charge(route.name, -(-len(text) // CHARS_PER_TOKEN))
            if entered:
                with contextlib.suppress(Exception):
                    await response_cm.__aexit__(None, None, None)
            with contextlib.suppress(Exception):
                await client.aclose()

    def _report(self, route: LLMRoute, ok: bool) -> None:
        if self.router.monitor:
            self.router.monitor.report(route.name, ok)

    def cancel(self) -> None:
        """Nothing to drain: cancelling the task that iterates :meth:`synthesize` closes the
        HTTP stream."""

    async def warm(self, language: str, voice: VoiceConfig) -> None:
        return None

    async def aclose(self) -> None:
        return None
