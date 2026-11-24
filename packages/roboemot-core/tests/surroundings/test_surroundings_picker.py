"""The snapshot picker: still frames only, the sharpest of a window (design 2026-11-24, 5.3)."""

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


def picker(now: float = 0.0, wait_s: float = 4.0, **kw) -> Picker:
    cfg = SurroundingsConfig(**kw)
    return Picker(lambda: cfg, wait_s, now)


def test_a_sharp_frame_scores_higher_than_its_blurred_copy():
    assert sharpness(board()) > 10 * sharpness(blurred(board()))


def test_the_sharpest_of_the_window_is_chosen():
    p = picker(candidates=3, min_sharpness=0)
    p.offer(blurred(board()), 0.1, head_moving=False, object_moving=False)
    p.offer(board(), 0.2, head_moving=False, object_moving=False)
    p.offer(blurred(board(), 3), 0.3, head_moving=False, object_moving=False)
    got = p.result(0.3)
    assert isinstance(got, Candidate)
    assert got.stamp == 0.2 and np.array_equal(got.image, board())


def test_the_window_closes_after_window_s():
    p = picker(candidates=5, window_s=1.0, min_sharpness=0)
    p.offer(board(), 0.1, head_moving=False, object_moving=False)
    assert p.result(0.5) is None
    assert isinstance(p.result(1.2), Candidate)


def test_head_moving_frames_never_become_candidates():
    p = picker(candidates=1, min_sharpness=0, wait_s=1.0)
    p.offer(board(), 0.1, head_moving=True, object_moving=False)
    assert p.result(0.2) is None
    assert p.last_skip == "head_moving"
    assert p.result(1.1) == "head_moving"


def test_object_moving_frames_never_become_candidates():
    p = picker(candidates=1, min_sharpness=0, wait_s=1.0)
    p.offer(board(), 0.1, head_moving=False, object_moving=True)
    assert p.result(1.1) == "object_moving"


def test_all_blurred_ends_too_blurry_at_wait_s():
    p = picker(candidates=1, min_sharpness=1000, wait_s=1.0)
    p.offer(flat(), 0.1, head_moving=False, object_moving=False)
    assert p.result(0.2) is None  # a blurred window clears and the attempt goes on
    p.offer(board(), 0.5, head_moving=False, object_moving=False)
    assert isinstance(p.result(0.5), Candidate)
    q = picker(candidates=1, min_sharpness=1000, wait_s=1.0)
    q.offer(flat(), 0.1, head_moving=False, object_moving=False)
    q.result(0.2)
    assert q.result(1.1) == "too_blurry"


def test_no_frames_ends_no_frame():
    p = picker(wait_s=1.0)
    p.offer(None, 0.1, head_moving=False, object_moving=False)
    assert p.result(0.5) is None
    assert p.result(1.1) == "no_frame"
