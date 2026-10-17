"""Sample-rate conversion: chunked equals whole (SPEC 4A, 8)."""

from __future__ import annotations

import numpy as np
import pytest
from roboemot_reachymini.resample import StreamResampler, resample

PAIRS = [(48000, 16000), (16000, 48000), (44100, 16000), (16000, 24000), (22050, 16000)]


def tone(rate: int, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1234 * t)).astype(
        np.float32
    )


@pytest.mark.parametrize("chunk", [1, 7, 480])
@pytest.mark.parametrize(("a", "b"), PAIRS)
def test_chunked_conversion_equals_the_whole_within_tolerance(a, b, chunk):
    x = tone(a, 0.1 if chunk == 1 else 0.5)
    s = StreamResampler(a, b)
    out = np.concatenate([s.process(x[i : i + chunk]) for i in range(0, x.size, chunk)])
    ref = resample(x, a, b)
    held_back = ref.size - out.size
    assert 0 <= held_back <= s.lookahead_s * b + b / a * 2 + s.up + 2
    assert held_back / b <= s.lookahead_s + 2 * s.down / a
    assert np.abs(out - ref[: out.size]).max() < 1e-6


def test_equal_rates_pass_through_and_bad_rates_are_refused():
    x = tone(16000, 0.01)
    assert resample(x, 16000, 16000) is x
    for bad in (0, -1):
        with pytest.raises(ValueError):
            resample(x, bad, 16000)
        with pytest.raises(ValueError):
            StreamResampler(16000, bad)
        with pytest.raises(ValueError):
            StreamResampler(bad, 16000)
