"""Piper worker: one persistent process per loaded voice (SPEC 8, TTS backends).

Runs INSIDE the Piper virtualenv, as ``python -I piper_worker.py --voice <voice.onnx> --chunk-ms
<ms> --intra-op-threads <n> --inter-op-threads <n>``, never imported by RoboEmot: ``piper-tts``
(piper1-gpl) is GPL-3.0, so it stays behind a process boundary. Only the standard library is
needed to speak the protocol; ``piper`` and ``onnxruntime`` are imported when the engine loads.

Protocol (little glue, no text ever leaves the process except as audio):

- stdin, JSON lines: ``{"id": n, "text": s, "length_scale": f, "noise_scale": f,
  "noise_w_scale": f, "volume": f}`` to synthesise one sentence, ``{"cancel": n}`` to stop
  forwarding request ``n`` (checked between pieces; a request still queued is skipped), and
  ``{"quit": true}``.
- stdout, binary frames ``>cII`` (kind, request id, payload length) plus payload: ``H`` hello
  (JSON ``{"rate_hz": r, "voice": name}``, once, rate read from ``<voice>.onnx.json``), ``A``
  little-endian int16 mono PCM of at most ``--chunk-ms``, ``E`` end of a request (also after a
  cancel), ``X`` failure (JSON ``{"error": exception class name}``, followed by ``E``).
- the thread counts (``voice.piper.intra_op_threads`` / ``inter_op_threads``, 0 = the ONNX
  Runtime default) are applied by rebuilding the voice's session: ``PiperVoice.load`` takes no
  session options. A rebuild that fails keeps the loaded session (one line on stderr).
- volume never clips: Piper is asked for ``PIPER_PRE_GAIN`` (so its own clip at full scale never
  acts on the model output) and the worker applies the rest of the requested volume, capped so
  the sentence's peak stays at ``PEAK_LIMIT`` of full scale.
- stderr is free text for humans; the parent drains it. File descriptor 1 is pointed at stderr
  right after start-up, so a stray ``print`` cannot corrupt the frames.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any, BinaryIO

HEADER = struct.Struct(">cII")
BYTES_PER_SAMPLE = 2
MS_PER_S = 1000.0
PIPER_PRE_GAIN = 0.5  # 6 dB of headroom: Piper clips at +-1 after its own volume
PEAK_LIMIT = 0.98  # highest peak, as a share of full scale, the requested volume may reach
INT16_FULL_SCALE = 32767


def read_sample_rate(voice_path: str) -> int:
    """Sample rate of a Piper voice, from ``<voice>.onnx.json`` (``audio.sample_rate``)."""
    with open(f"{voice_path}.json", encoding="utf-8") as f:
        return int(json.load(f)["audio"]["sample_rate"])


def session_options(onnxruntime: Any, intra_op_threads: int, inter_op_threads: int) -> Any:
    """ONNX Runtime session options with the given thread counts (0 leaves a count at the
    runtime's default)."""
    options = onnxruntime.SessionOptions()
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        options.inter_op_num_threads = inter_op_threads
    return options


class PiperEngine:
    """The real engine: ``piper.PiperVoice`` (piper-tts 1.x), one int16 piece per sentence."""

    def __init__(
        self, voice_path: str, intra_op_threads: int = 0, inter_op_threads: int = 0
    ) -> None:
        import numpy
        import onnxruntime
        from piper import PiperVoice, SynthesisConfig

        self.sample_rate = read_sample_rate(voice_path)
        # load() builds a default session that is replaced below: building the voice ourselves
        # would tie the worker to more of Piper's internals than one public field.
        self._voice = PiperVoice.load(voice_path)
        if intra_op_threads > 0 or inter_op_threads > 0:
            try:
                self._voice.session = onnxruntime.InferenceSession(
                    voice_path,
                    sess_options=session_options(onnxruntime, intra_op_threads, inter_op_threads),
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:  # a bad option must not silence the robot
                print(f"thread options not applied: {type(exc).__name__}", file=sys.stderr)
        self._config = SynthesisConfig
        self._np = numpy

    def synthesize(self, request: dict[str, Any]) -> Iterator[bytes]:
        # normalize_audio off: the volume the caller computed must mean something absolute.
        config = self._config(
            length_scale=request.get("length_scale"),
            noise_scale=request.get("noise_scale"),
            noise_w_scale=request.get("noise_w_scale"),
            volume=PIPER_PRE_GAIN,
            normalize_audio=False,
        )
        volume = float(request.get("volume", 1.0))
        for chunk in self._voice.synthesize(request["text"], config):
            yield limited_pcm(self._np, chunk.audio_float_array, volume / PIPER_PRE_GAIN)


def limited_pcm(np: Any, audio: Any, gain: float) -> bytes:
    """``audio * gain`` as little-endian int16, the gain lowered when it would push the peak
    above ``PEAK_LIMIT`` of full scale (a volume that would clip plays as loud as it can)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        gain = min(gain, PEAK_LIMIT / peak)
    out = np.clip(audio * gain, -1.0, 1.0) * INT16_FULL_SCALE
    return out.astype("<i2").tobytes()


def _send(out: BinaryIO, kind: bytes, request_id: int, payload: bytes = b"") -> None:
    out.write(HEADER.pack(kind, request_id, len(payload)) + payload)
    out.flush()


def serve(load_engine: Callable[[str, int, int], Any], argv: list[str] | None = None) -> int:
    """Run the request loop with ``load_engine(voice_path, intra_op_threads, inter_op_threads)``
    (an object with ``sample_rate`` and ``synthesize(request) -> Iterator[bytes]``); tests plug a
    fake engine in here."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    parser.add_argument("--chunk-ms", type=float, required=True)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    args = parser.parse_args(argv)
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    engine = load_engine(args.voice, args.intra_op_threads, args.inter_op_threads)
    piece_bytes = max(
        BYTES_PER_SAMPLE,
        int(engine.sample_rate * args.chunk_ms / MS_PER_S) * BYTES_PER_SAMPLE,
    )
    hello = {"rate_hz": engine.sample_rate, "voice": os.path.basename(args.voice)}
    _send(out, b"H", 0, json.dumps(hello).encode())

    requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
    cancelled: set[int] = set()
    lock = threading.Lock()

    def read_stdin() -> None:
        for line in sys.stdin.buffer:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "cancel" in msg:
                with lock:
                    cancelled.add(int(msg["cancel"]))
            elif msg.get("quit"):
                break
            else:
                requests.put(msg)
        requests.put(None)

    threading.Thread(target=read_stdin, name="stdin", daemon=True).start()

    def is_cancelled(request_id: int) -> bool:
        with lock:
            return request_id in cancelled

    while (request := requests.get()) is not None:
        rid = int(request["id"])
        try:
            if not is_cancelled(rid):
                for pcm in engine.synthesize(request):
                    for start in range(0, len(pcm), piece_bytes):
                        if is_cancelled(rid):
                            break
                        _send(out, b"A", rid, pcm[start : start + piece_bytes])
                    if is_cancelled(rid):
                        break
        except BrokenPipeError:
            return 1
        except Exception as exc:  # reported by class name only: messages may quote the text
            _send(out, b"X", rid, json.dumps({"error": type(exc).__name__}).encode())
        _send(out, b"E", rid)
        with lock:
            cancelled.intersection_update({c for c in cancelled if c > rid})
    return 0


if __name__ == "__main__":
    sys.exit(serve(PiperEngine))
