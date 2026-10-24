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

from ..config.schema import VOICE_NAME, PiperConfig, VoiceConfig
from .models import ModelError, load_manifest, model_file
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
HELLO_MAX_BYTES = 4096  # H, X and E payloads are tiny JSON
FRAME_SLACK_BYTES = 4096  # audio frames may exceed chunk_ms by a little
RATE_RANGE_HZ = (8000, 192000)  # sane voice sample rates
BYTES_PER_SAMPLE = 2
MS_PER_S = 1000.0
_DEAD, _CANCELLED, _PROTOCOL = b"-", b"~", b"!"  # queue markers, never sent by a worker
RESTART_FIELDS = frozenset(
    name
    for name, info in PiperConfig.model_fields.items()
    if (info.json_schema_extra or {}).get("restart")
)


class _ProtocolError(Exception):
    """The worker sent a frame outside the protocol."""


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
    protocol_error: bool = False
    chunk_ms: float = 0.0
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
        marker = _PROTOCOL if self.protocol_error else _DEAD
        for q in self.pending.values():
            q.put_nowait((marker, b""))
        if not self.hello.done():
            reason = "violated the protocol" if self.protocol_error else "exited before its hello"
            self.hello.set_exception(TTSError(f"worker {reason}"))
            self.hello.exception()  # nobody may be waiting: mark it retrieved

    def _check_frame(self, kind: bytes, size: int) -> None:
        """Reject a frame no well-behaved worker sends (the worker is a separate program)."""
        if kind == b"A":
            limit = int(self.rate_hz * self.chunk_ms / MS_PER_S) * BYTES_PER_SAMPLE
            ok = self.hello.done() and size <= limit + FRAME_SLACK_BYTES
            ok = ok and size % BYTES_PER_SAMPLE == 0
        else:
            ok = kind in (b"H", b"X", b"E") and size <= HELLO_MAX_BYTES
            ok = ok and (kind != b"H" or not self.hello.done())
        if not ok:
            raise _ProtocolError

    def _set_hello(self, payload: bytes) -> None:
        rate = json.loads(payload)["rate_hz"]
        if not isinstance(rate, int) or isinstance(rate, bool):
            raise _ProtocolError
        if not RATE_RANGE_HZ[0] <= rate <= RATE_RANGE_HZ[1]:
            raise _ProtocolError
        self.rate_hz = rate
        self.hello.set_result(rate)

    async def read_frames(self) -> None:
        stdout = self.proc.stdout
        assert stdout is not None
        try:
            while True:
                kind, rid, size = HEADER.unpack(await stdout.readexactly(HEADER.size))
                self._check_frame(kind, size)
                payload = await stdout.readexactly(size) if size else b""
                if kind == b"H":
                    self._set_hello(payload)
                elif (q := self.pending.get(rid)) is not None:  # unknown ids: a cancelled request
                    q.put_nowait((kind, payload))
        except (_ProtocolError, ValueError, KeyError, TypeError):
            self.protocol_error = True
            self.kill()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            self.fail_pending()
            await self.reap()

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self.proc.kill()

    async def reap(self) -> None:
        """Wait (bounded) for the process to exit so no zombie is left."""
        with contextlib.suppress(TimeoutError, ProcessLookupError):
            await asyncio.wait_for(self.proc.wait(), QUIT_GRACE_S)

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
            self.kill()
            await self.reap()
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
        voice_files: Callable[[str], tuple[Path, Path]],
        *,
        script: str | Path = WORKER_SCRIPT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config, self.python, self.voice_files = config, str(python), voice_files
        self.script, self.clock = str(script), clock
        self._workers: dict[str, _Worker] = {}
        self._starting: dict[str, asyncio.Task] = {}
        self._failures: dict[str, int] = {}
        self._retry_at: dict[str, float] = {}
        self.restarts = 0

    def apply_config(self, voice: VoiceConfig) -> None:
        """Hot reload (SPEC 11A): ``voice.piper``'s timeouts and restart backoff apply from the
        next request or worker start; its restart fields (``python``, ``chunk_ms``, the thread
        counts) stay."""
        keep = {name: getattr(self.config, name) for name in RESTART_FIELDS}
        self.config = voice.piper.model_copy(update=keep)

    # ----------------------------------------------------------------- availability

    def available(self) -> bool:
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
        """Verified ``.onnx`` of the language's voice; unverified or missing files are
        :class:`TTSUnavailable` (``voice_files`` enforces the model registry's sha256 marker)."""
        name = voice.voices.get(language)
        if not name:
            raise TTSError(f"no voice configured for language {language!r}")
        if not VOICE_NAME.fullmatch(name) or ".." in name:
            raise TTSUnavailable("invalid voice name")
        try:
            model, config = self.voice_files(name)
        except (ModelError, OSError) as exc:
            raise TTSUnavailable(f"voice {name} is not available: {type(exc).__name__}") from exc
        if not model.is_file() or not config.is_file():
            raise TTSUnavailable(f"voice files for {name} are missing")
        return model

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
                "--intra-op-threads", str(self.config.intra_op_threads),
                "--inter-op-threads", str(self.config.inter_op_threads),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )  # fmt: skip
        except OSError as exc:
            self._failed(language)
            raise TTSError(f"cannot start the Piper worker: {type(exc).__name__}") from exc
        worker = _Worker(proc, chunk_ms=self.config.chunk_ms)
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
                        self._workers.pop(language, None)
                        self._failed(language)
                        worker.fail_pending()
                        worker.kill()
                        await worker.reap()
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
                    elif kind == _PROTOCOL:
                        finished = True
                        self._workers.pop(language, None)
                        self._failed(language)
                        raise TTSError("piper worker violated the protocol")
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


def manifest_voice_files(root: Path) -> Callable[[str], tuple[Path, Path]]:
    """Resolver over the voice model registry: ``<root>/piper/<voice>.onnx`` and ``.onnx.json``,
    both with a valid ``.verified`` marker (``roboemot models fetch --voice``)."""

    def resolve(voice: str) -> tuple[Path, Path]:
        suffix = f"piper/{voice}.onnx"
        for spec in load_manifest().values():
            if any(f.path == suffix for f in spec.files):
                return (
                    model_file(root, spec.name, f"{voice}.onnx"),
                    model_file(root, spec.name, f"{voice}.onnx.json"),
                )
        raise ModelError(f"voice {voice} is not in the voice model registry")

    return resolve
