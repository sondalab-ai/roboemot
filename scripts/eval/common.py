"""Shared helpers of the M1 perception evaluation (SPEC 13, 14): dev only, nothing here ships.

The evaluation scripts run the shipped code (``roboemot.vision`` stages and pipeline, the
registry matcher) on datasets and weights kept outside the repository, in
``~/.cache/roboemot-bench/`` (override with ``ROBOEMOT_BENCH_CACHE``; see ``scripts/bench``).
Run them one at a time: ``OMP_NUM_THREADS=2 nice -n 10 python -I scripts/eval/<name>.py``.
Result files hold aggregates only: no image names, embeddings or home-directory paths.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

CACHE = Path(os.environ.get("ROBOEMOT_BENCH_CACHE", Path.home() / ".cache" / "roboemot-bench"))
DATASETS = CACHE / "datasets"
MODELS = CACHE / "models"
RESULTS = Path(__file__).resolve().parent / "results"
SCRATCH = Path(os.environ.get("ROBOEMOT_EVAL_SCRATCH", "/tmp/roboemot-eval"))
WILSON_Z = 1.96


def config(overrides: dict[str, Any] | None = None):
    """The shipped default configuration with optional nested overrides (a plain dict)."""
    from roboemot.config.schema import RoboEmotConfig

    return RoboEmotConfig.model_validate(overrides or {})


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval of k successes in n trials (0, 0 when n is 0)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = WILSON_Z**2
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> dict[str, Any]:
    """``{k, n, rate, ci95}`` for a count; ``rate`` is None when n is 0."""
    lo, hi = wilson(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci95": [lo, hi]}


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in 0..1) of a non-empty list."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def cpu_label() -> str:
    """Machine description for labelling timings (never a substitute for the CM4)."""
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        brand = platform.processor() or platform.machine()
    return f"{brand}, {os.cpu_count()} logical cores, {platform.system()} {platform.release()}"


def versions() -> dict[str, str]:
    """Library versions that affect the numbers."""
    import cv2
    import onnxruntime

    return {
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def write_result(name: str, data: dict[str, Any]) -> Path:
    """Write ``results/<name>.json``; refuses text that looks like a home path or image name."""
    text = json.dumps(data, indent=1, sort_keys=True)
    for needle in ("/Users", str(Path.home()), ".jpg", ".png"):
        if needle in text:
            raise ValueError(f"result {name} contains {needle!r}: aggregates only")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(text + "\n")
    return path


def load_stages(cfg, *, profile: str | None = None, **kwargs: Any):
    """The shipped ONNX backends from the bench models folder (markers verified)."""
    from roboemot.vision.onnx import load_stages as load

    return load(MODELS, cfg.vision, profile=profile, **kwargs)


def log(*parts: object) -> None:
    """Progress line on stderr, flushed (results go to files, not stdout)."""
    print(*parts, file=sys.stderr, flush=True)
