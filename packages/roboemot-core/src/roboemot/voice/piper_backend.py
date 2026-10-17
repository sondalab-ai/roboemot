"""Piper TTS backend: a persistent subprocess per voice, a request per sentence (SPEC 8).

``piper-tts`` is GPL-3.0, so it lives in its own virtualenv and is only ever executed
(``<voice.piper.python> -I piper_worker.py``), never imported here. One worker serves every
sentence of its language; the protocol is documented in ``piper_worker.py``. The sample rate comes
from the worker's hello (read from the voice's ``.onnx.json``). A crashed worker is started again
after an exponential backoff; while it waits, :meth:`PiperBackend.available` is false so a
fallback (or silence) takes over instead of stalling the queue. A cancel stops forwarding the
request's frames and tells the worker to skip the rest; late frames are dropped.

Pitch has no native control in Piper. With ``voice.pitch_shift_backend: dsp`` the sentence is
synthesised ``f`` times slower (``length_scale * f``) and its audio declared at ``rate * f``, so
playing it resamples it back to the right duration at a pitch ``12 * log2(f)`` semitones higher
(varispeed: formants move with the pitch, acceptable within ``voice.limits.pitch_st``).

Privacy (SPEC 12): the worker's stderr is counted and dropped (it could quote text), except
before its hello, when it is the only clue to a failed start and no text was sent yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config.schema import PiperConfig, VoiceConfig
from .prosody import Delivery, to_piper
from .tts import AudioChunk, TTSError, TTSUnavailable

log = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).with_name("piper_worker.py")
HEADER = struct.Struct(">cII")  # same frame header as piper_worker.py (kind, id, length)
INT16_SCALE = 32768.0
SEMITONES_PER_OCTAVE = 12.0
STARTUP_TAIL_BYTES = 2000
STDERR_READ_BYTES = 4096
STARTUP_LOG_CHARS = 400
QUIT_GRACE_S = 1.0
_DEAD, _CANCELLED = b"-", b"~"  # queue markers, never sent by a worker


@dataclass
class _Worker:
    """One running worker process and the requests waiting on it."""

    proc: asyncio.subprocess.Process
    rate_hz: int = 0
    hello: asyncio.Future = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    pending: dict[int, asyncio.Queue] = field(default_factory=dict)
    next_id: int = 1
    dead: bool = False
    stderr_bytes: int = 0
    startup_tail: bytearray = field(default_factory=bytearray)
    tasks: list[asyncio.Task] = field(default_factory=list)

    def send(self, message: dict[str, Any]) -> None:
        if self.dead or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(json.dumps(message).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            self.dead = True

    def fail_pending(self) -> None:
        self.dead = True
        for q in self.pending.values():
            q.put_nowait((_DEAD, b""))
        if not self.hello.done():
            self.hello.set_exception(TTSError("worker exited before its hello"))
            self.hello.exception()  # nobody may be waiting: mark it retrieved

    async def read_frames(self) -> None:
        stdout = self.proc.stdout
        assert stdout is not None
        try:
            while True:
                kind, rid, size = HEADER.unpack(await stdout.readexactly(HEADER.size))
                payload = await stdout.readexactly(size) if size else b""
                if kind == b"H":
                    if not self.hello.done():
                        self.hello.set_result(json.loads(payload)["rate_hz"])
                elif (q := self.pending.get(rid)) is not None:  # unknown ids: a cancelled request
                    q.put_nowait((kind, payload))
        except (asyncio.IncompleteReadError, ConnectionResetError, ValueError, KeyError):
            pass
        finally:
            self.fail_pending()

    async def drain_stderr(self) -> None:
        stderr = self.proc.stderr
        assert stderr is not None
        while data := await stderr.read(STDERR_READ_BYTES):
            self.stderr_bytes += len(data)
            if not self.hello.done() and len(self.startup_tail) < STARTUP_TAIL_BYTES:
                self.startup_tail += data[: STARTUP_TAIL_BYTES - len(self.startup_tail)]

    async def stop(self) -> None:
        self.send({"quit": True})
        if self.proc.stdin is not None:
            with contextlib.suppress(Exception):
                self.proc.stdin.close()
        try:
            await asyncio.wait_for(self.proc.wait(), QUIT_GRACE_S)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
            await self.proc.wait()
        for task in self.tasks:
            task.cancel()
        self.fail_pending()


class PiperBackend:
    """:class:`~roboemot.voice.tts.TTSBackend` over persistent Piper workers (one per language)."""

    name = "piper"
    controls = frozenset({"rate", "pitch", "volume", "variability"})

    def __init__(
        self,
        config: PiperConfig,
        python: str | Path,
        voices_dir: str | Path,
        *,
        script: str | Path = WORKER_SCRIPT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config, self.python, self.voices_dir = config, str(python), Path(voices_dir)
        self.script, self.clock = str(script), clock
        self._workers: dict[str, _Worker] = {}
        self._starting: dict[str, asyncio.Task] = {}
        self._failures: dict[str, int] = {}
        self._retry_at: dict[str, float] = {}
        self.restarts = 0

    # ----------------------------------------------------------------- availability

    def available(self, *, minor_present: bool = False) -> bool:
        """Local and offline: always eligible; a worker in restart backoff makes the request
        fail fast with :class:`TTSUnavailable`, which a fallback backend answers."""
        return True

    def _backoff_s(self, failures: int) -> float:
        first, longest = self.config.restart_backoff_s
        return min(longest, first * 2 ** max(0, failures - 1))

    def _failed(self, language: str) -> None:
        n = self._failures[language] = self._failures.get(language, 0) + 1
        self._retry_at[language] = self.clock() + self._backoff_s(n)

    # ----------------------------------------------------------------- workers

    def voice_path(self, language: str, voice: VoiceConfig) -> Path:
        name = voice.voices.get(language)
        if not name:
            raise TTSError(f"no voice configured for language {language!r}")
        path = self.voices_dir / f"{name}.onnx"
        if not path.is_file() or not Path(f"{path}.json").is_file():
            raise TTSError(f"voice files for {name} not found in {self.voices_dir}")
        return path

    async def warm(self, language: str, voice: VoiceConfig) -> None:
        """Start the language's worker now, so the first sentence does not pay the load."""
        try:
            await self._worker(language, voice)
        except TTSError as exc:
            log.warning("piper warm-up failed: %s", type(exc).__name__)

    async def _worker(self, language: str, voice: VoiceConfig) -> _Worker:
        worker = self._workers.get(language)
        if worker is not None and not worker.dead:
            return worker
        if worker is not None:
            self._workers.pop(language, None)
            self._failed(language)
            await worker.stop()
        wait = self._retry_at.get(language, 0.0) - self.clock()
        if wait > 0:
            raise TTSUnavailable(f"piper worker restarting in {wait:.1f} s")
        task = self._starting.get(language)
        if task is None:
            task = self._starting[language] = asyncio.ensure_future(self._start(language, voice))
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._starting.pop(language, None)

    async def _start(self, language: str, voice: VoiceConfig) -> _Worker:
        path = self.voice_path(language, voice)
        if self._failures.get(language):
            self.restarts += 1
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python, "-I", self.script, "--voice", str(path),
                "--chunk-ms", str(self.config.chunk_ms),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )  # fmt: skip
        except OSError as exc:
            self._failed(language)
            raise TTSError(f"cannot start the Piper worker: {type(exc).__name__}") from exc
        worker = _Worker(proc)
        worker.tasks = [
            asyncio.ensure_future(worker.read_frames()),
            asyncio.ensure_future(worker.drain_stderr()),
        ]
        try:
            hello = asyncio.wait_for(worker.hello, self.config.startup_timeout_s)
            worker.rate_hz = int(await hello)
        except (TimeoutError, TTSError) as exc:
            tail = bytes(worker.startup_tail).decode(errors="replace").strip()
            log.warning(
                "piper worker did not start (%s): %s", type(exc).__name__, tail[-STARTUP_LOG_CHARS:]
            )
            self._failed(language)
            await worker.stop()
            raise TTSError("the Piper worker did not start") from exc
        self._workers[language] = worker
        return worker

    # ----------------------------------------------------------------- synthesis

    async def synthesize(
        self,
        text: str,
        delivery: Delivery,
        *,
        language: str,
        voice: VoiceConfig,
        minor_present: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        params = to_piper(delivery, voice)
        factor = 2.0 ** (params["pitch_shift_st"] / SEMITONES_PER_OCTAVE)
        for attempt in (0, 1):
            worker = await self._worker(language, voice)
            rid = worker.next_id
            worker.next_id += 1
            queue: asyncio.Queue = asyncio.Queue()
            worker.pending[rid] = queue
            worker.send(
                {
                    "id": rid,
                    "text": text,
                    "length_scale": params["length_scale"] * factor,
                    "noise_scale": params["noise_scale"],
                    "noise_w_scale": params["noise_w_scale"],
                    "volume": params["volume"],
                }
            )
            finished = yielded = False
            try:
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(
                            queue.get(), self.config.request_timeout_s
                        )
                    except TimeoutError as exc:
                        worker.fail_pending()
                        self._kill(worker)
                        raise TTSError("piper worker timed out") from exc
                    if kind == b"A":
                        yielded = True
                        pcm = np.frombuffer(payload, dtype="<i2")
                        yield AudioChunk(
                            pcm.astype(np.float32) / INT16_SCALE, round(worker.rate_hz * factor)
                        )
                    elif kind == b"E":
                        finished = True
                        self._failures.pop(language, None)
                        self._retry_at.pop(language, None)
                        return
                    elif kind == b"X":
                        finished = True
                        raise TTSError(f"piper worker error: {json.loads(payload)['error']}")
                    elif kind == _CANCELLED:
                        finished = True
                        return
                    elif kind == _DEAD:
                        break
            finally:
                worker.pending.pop(rid, None)
                if not finished:
                    worker.send({"cancel": rid})
            self._workers.pop(language, None)
            self._failed(language)
            if yielded or attempt:
                raise TTSError("piper worker died")
            await asyncio.sleep(max(0.0, self._retry_at[language] - self.clock()))
        raise TTSError("piper worker died")  # pragma: no cover

    @staticmethod
    def _kill(worker: _Worker) -> None:
        with contextlib.suppress(ProcessLookupError):
            worker.proc.kill()

    def cancel(self) -> None:
        """Stop forwarding every request in flight and tell the workers to skip them."""
        for worker in self._workers.values():
            for rid, q in list(worker.pending.items()):
                worker.send({"cancel": rid})
                q.put_nowait((_CANCELLED, b""))

    async def aclose(self) -> None:
        workers, self._workers = list(self._workers.values()), {}
        for task in self._starting.values():
            task.cancel()
        for worker in workers:
            await worker.stop()

    def snapshot(self) -> dict[str, Any]:
        """Counts only: languages loaded, restarts, stderr bytes seen."""
        return {
            "languages": sorted(self._workers),
            "restarts": self.restarts,
            "stderr_bytes": sum(w.stderr_bytes for w in self._workers.values()),
        }
