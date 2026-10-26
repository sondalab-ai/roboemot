"""OpenAI-compatible speech backend, ``POST {base_url}/audio/speech`` (SPEC 8, TTS backends).

Follows the LLM routes' rules: it uses the route named by ``voice.openai_speech.route`` (URL,
key, ``first_token_timeout_s`` for the first audio chunk, ``stall_timeout_s`` for later gaps),
is eligible when the router would allow that route (reachable, budget, key), and
charges the characters it sends against the route's budget at ``CHARS_PER_TOKEN`` per token.
Failures before the first chunk raise :class:`~roboemot.voice.tts.TTSError`, so
:class:`~roboemot.voice.tts.FallbackTTS` answers with Piper. Only the sentence text leaves the
robot (SPEC 12); it and the response body are never logged. The text is what is spoken, so it
holds the real names even with ``privacy.pseudonyms_to_cloud`` (owner decision, 2026-10-18).

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


class OpenAISpeechBackend:
    """:class:`~roboemot.voice.tts.TTSBackend` for a remote ``/audio/speech`` route."""

    name = "openai_speech"

    def __init__(self, config: OpenAISpeechConfig, router: Callable[[], Router]) -> None:
        """``router`` is a getter: the app rebuilds its router when the routes change."""
        self.config, self._router = config, router
        self.controls = frozenset({"instructions"} if config.instructable else ())

    @property
    def router(self) -> Router:
        return self._router()

    def _route(self) -> LLMRoute | None:
        return next((r for r in self.router.routes if r.name == self.config.route), None)

    def available(self) -> bool:
        """The route exists and the router would use it now (online, budget, key)."""
        route = self._route()
        return route is not None and any(r.name == route.name for r in self.router.eligible())

    async def synthesize(
        self,
        text: str,
        delivery: Delivery,
        *,
        language: str,
        voice: VoiceConfig,
    ) -> AsyncIterator[AudioChunk]:
        router = self.router  # one router for the whole request, even if the app swaps it
        route = next((r for r in router.routes if r.name == self.config.route), None)
        if route is None or route not in router.eligible():
            raise TTSUnavailable("speech route not eligible")
        body: dict[str, Any] = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": text,
            "response_format": "pcm",
        }
        if self.config.instructable:
            body.update(to_openai(delivery))
        try:
            headers = router.auth_headers(route)
        except Exception as exc:  # e.g. a missing secret: treat the route as unusable
            raise TTSUnavailable(f"speech route key unavailable: {type(exc).__name__}") from exc
        loop = asyncio.get_running_loop()
        deadline = loop.time() + route.first_token_timeout_s
        client = (
            router.http_client_factory()
            if router.http_client_factory
            else httpx.AsyncClient(follow_redirects=False)
        )
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
                if not response.is_success:  # an error, or a redirect (never followed)
                    self._report(router, route, True)
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
                self._report(router, route, True)
            except TimeoutError as exc:
                if router.monitor:
                    router.monitor.recheck(route.name)
                raise TTSError("speech route timed out") from exc
            except httpx.TimeoutException as exc:
                if router.monitor:
                    router.monitor.recheck(route.name)
                raise TTSError("speech route timed out") from exc
            except (httpx.TransportError, OSError) as exc:
                self._report(router, route, False)
                raise TTSError(f"speech route unreachable: {type(exc).__name__}") from exc
        finally:
            if entered:  # only a request the server accepted is charged
                router.charge(route.name, -(-len(text) // CHARS_PER_TOKEN))
                with contextlib.suppress(Exception):
                    await response_cm.__aexit__(None, None, None)
            with contextlib.suppress(Exception):
                await client.aclose()

    @staticmethod
    def _report(router: Router, route: LLMRoute, ok: bool) -> None:
        if router.monitor:
            router.monitor.report(route.name, ok)

    def cancel(self) -> None:
        """Nothing to drain: cancelling the task that iterates :meth:`synthesize` closes the
        HTTP stream."""

    async def warm(self, language: str, voice: VoiceConfig) -> None:
        return None

    def apply_config(self, voice: VoiceConfig) -> None:
        """Nothing kept here: the speech settings it reads are restart fields."""

    async def aclose(self) -> None:
        return None
