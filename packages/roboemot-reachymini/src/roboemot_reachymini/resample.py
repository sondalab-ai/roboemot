"""Sample-rate conversion for the microphone hub and the speaker (SPEC 4A, 8).

``resample`` converts one whole array (a spoken sentence). ``StreamResampler`` converts a
stream chunk by chunk with the same filter: it keeps filter history and a few samples of
lookahead, so chunk boundaries leave no click, at the price of that small lookahead in latency.
``flush`` ends a stream: it returns the held-back tail, so the whole stream equals ``resample`` of
the concatenated input, and starts a fresh one.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly

FILTER_HALF_LEN = 10  # resample_poly's default FIR half-length, in taps per max(up, down)


def _ratio(in_hz: int, out_hz: int) -> tuple[int, int]:
    g = math.gcd(in_hz, out_hz)
    return out_hz // g, in_hz // g


def resample(x: np.ndarray, in_hz: int, out_hz: int) -> np.ndarray:
    """Whole-array conversion of mono float32 audio; returns the input itself at equal rates."""
    if in_hz <= 0 or out_hz <= 0:
        raise ValueError("sample rates must be positive")
    if in_hz == out_hz or x.size == 0:
        return x
    up, down = _ratio(in_hz, out_hz)
    return resample_poly(x, up, down).astype(np.float32, copy=False)


class StreamResampler:
    """Chunked mono conversion from ``in_hz`` to ``out_hz`` that matches ``resample`` on the whole.

    Input is consumed in blocks of ``down`` samples (``up`` outputs each), with ``history``
    samples before and ``lookahead`` after the block as filter support. Samples that do not fill
    a block, or whose lookahead has not arrived, wait for the next call; ``flush`` releases them.
    """

    def __init__(self, in_hz: int, out_hz: int) -> None:
        if in_hz <= 0 or out_hz <= 0:
            raise ValueError("sample rates must be positive")
        self.in_hz = in_hz
        self.up, self.down = _ratio(in_hz, out_hz)
        half_in = math.ceil(FILTER_HALF_LEN * max(self.up, self.down) / self.up)
        self._hist = math.ceil(half_in / self.down) * self.down
        self._look = half_in + self.down
        self._reset()

    def _reset(self) -> None:
        self._buf = np.zeros(self._hist, np.float32)
        self._in_count = 0
        self._out_count = 0

    @property
    def lookahead_s(self) -> float:
        """Input time the converter holds back: its output trails the input by this much."""
        return self._look / self.in_hz

    def process(self, x: np.ndarray) -> np.ndarray:
        """Convert the next chunk; may return fewer (or no) samples while it fills a block."""
        self._in_count += x.size
        return self._convert(x)

    def flush(self) -> np.ndarray:
        """End the stream: the output still held back (the input followed by silence, as
        ``resample`` pads the end of a whole array), then reset for a new stream."""
        total = math.ceil(self._in_count * self.up / self.down)
        tail = self._convert(np.zeros(self._look + self.down, np.float32))
        out = tail[: max(0, total - (self._out_count - tail.size))]
        self._reset()
        return out

    def _convert(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)])
        blocks = (buf.size - self._hist - self._look) // self.down
        if blocks <= 0:
            self._buf = buf
            return np.empty(0, np.float32)
        end = self._hist + blocks * self.down
        y = resample_poly(buf[: end + self._look], self.up, self.down)
        start_out = self._hist // self.down * self.up
        out = y[start_out : start_out + blocks * self.up].astype(np.float32)
        self._buf = buf[end - self._hist :]
        self._out_count += out.size
        return out
