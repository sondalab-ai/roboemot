"""The view picker: still frames only, the sharpest of a window (designs 2026-11-24, 5.3;
2026-11-25)."""

from __future__ import annotations

import numpy as np
from roboemot.config.schema import SurroundingsConfig
from roboemot.surroundings.picker import Picker, sharpness
from roboemot.surroundings.types import Candidate


def board(cell: int = 8, size: int = 64) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    gray = (((ys // cell) + (xs // cell)) % 2 * 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def blurred(image: np.ndarray, k: int = 5) -> np.ndarray:
    pad = k // 2
    f = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros(image.shape, np.float32)
    for dy in range(k):
        for dx in range(k):
            out += f[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return (out / (k * k)).astype(np.uint8)


def flat() -> np.ndarray:
    return np.full((64, 64, 3), 128, np.uint8)


def picker(**kw) -> Picker:
    cfg = SurroundingsConfig(**kw)
    return Picker(lambda: cfg)


def test_a_sharp_frame_scores_higher_than_its_blurred_copy():
    assert sharpness(board()) > 10 * sharpness(blurred(board()))


def test_the_sharpest_of_the_window_is_chosen():
    p = picker(candidates=3, min_sharpness=0)
    p.offer(blurred(board()), 0.1, head_moving=False)
    p.offer(board(), 0.2, head_moving=False)
    p.offer(blurred(board(), 3), 0.3, head_moving=False)
    got = p.result(0.3)
    assert isinstance(got, Candidate)
    assert got.stamp == 0.2 and np.array_equal(got.image, board())


def test_the_window_closes_after_window_s():
    p = picker(candidates=5, window_s=1.0, min_sharpness=0)
    p.offer(board(), 0.1, head_moving=False)
    assert p.result(0.5) is None
    assert isinstance(p.result(1.2), Candidate)


def test_head_moving_frames_never_become_candidates():
    p = picker(candidates=1, min_sharpness=0)
    p.offer(board(), 0.1, head_moving=True)
    assert p.result(0.2) is None and p.best() is None


def test_a_blurred_window_is_cleared_and_never_chosen():
    p = picker(candidates=1, min_sharpness=1000)
    p.offer(flat(), 0.1, head_moving=False)
    assert p.result(0.2) is None  # a blurred window clears and the view goes on
    p.offer(board(), 0.5, head_moving=False)
    assert isinstance(p.result(0.5), Candidate)


def test_best_takes_the_sharpest_of_an_open_window_above_the_floor():
    p = picker(candidates=5, window_s=1.0, min_sharpness=0)
    p.offer(blurred(board()), 0.1, head_moving=False)
    p.offer(board(), 0.2, head_moving=False)
    got = p.best()  # the dwell ended before the window closed
    assert got is not None and got.stamp == 0.2
    q = picker(candidates=5, window_s=1.0, min_sharpness=1000)
    q.offer(flat(), 0.1, head_moving=False)
    assert q.best() is None
