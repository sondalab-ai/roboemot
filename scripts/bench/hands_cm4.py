"""CPU cost of a hand pipeline on the robot: MediaPipe palm detector + 21-point hand landmark.

Measurement only (SPEC 5, 13): nothing here ships in the library or touches a running RoboEmot.
The models are the opencv_zoo ONNX conversions (Apache-2.0), fp32 and the zoo's int8 variant,
pinned to opencv_zoo commit f88e9b2 (tag 4.10.0, as ``vittrack`` in ``vision/models.yaml``).
They are downloaded into a temporary folder (deleted on exit; ``--cache-dir`` keeps them),
checked against the pinned size and sha256 (the Git LFS oid), and only ever loaded as ONNX
weights. The zoo's block-quantised ``int8bq`` files are left out: they target OpenCV DNN, and
ONNX Runtime 1.32 refuses to run them ("block_size must be 0 for per-tensor quantization").

Each run feeds synthetic camera frames (random pixels, ``--frame`` 1280x720), so no camera is
needed; the network cost does not depend on the pixel values. One pass is:

- palm: letterbox the frame, centred, into the detector input (192x192), BGR to RGB, /255, run,
  then the sigmoid and best score of the 2016 anchors;
- landmark: crop a square of ``--hand-px`` around the frame centre (as if one palm was found),
  pad it for rotation, rotate it, resize it to the landmark input (224x224), RGB, /255, run.

Times include that preprocessing (the debug report's ``face_detect`` includes the face
detector's letterbox too); ``net`` is the ``session.run`` alone. CPU is ``time.process_time``,
which counts the ONNX Runtime pool threads; ``cores`` is CPU time over wall time. ONNX Runtime
runs with ``intra_op_num_threads = --threads`` (default 2, ``vision.threads``),
``inter_op_num_threads = 1`` and, unless ``--spinning``, no spin-wait between operators (the
default of the object detector, ``vision.object_spinning``), like ``roboemot.vision.onnx``.

On the robot (needs the apps venv's onnxruntime and OpenCV, which come with the vision extra);
the script is read from stdin, so nothing is left on the robot:

    python scripts/robot_ssh.py --host reachy-mini.local --user pollen -- \\
        /venvs/apps_venv/bin/python -I - --runs 200 < scripts/bench/hands_cm4.py

With RoboEmot running the numbers include its load (the load average is printed before and
after); stop it from the dashboard first to measure the models alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ZOO_COMMIT = "f88e9b2bafd21f1cad242fb5af6d78f2bcba16a3"
ZOO = f"https://github.com/opencv/opencv_zoo/raw/{ZOO_COMMIT}/models"
PALM_DIR = "palm_detection_mediapipe"
HAND_DIR = "handpose_estimation_mediapipe"


@dataclass(frozen=True)
class ModelFile:
    """One pinned ONNX file: name on disk, download URL, size and sha256 (the Git LFS oid)."""

    name: str
    url: str
    size_bytes: int
    sha256: str


def _zoo(folder: str, name: str, size_bytes: int, sha256: str) -> ModelFile:
    return ModelFile(name, f"{ZOO}/{folder}/{name}", size_bytes, sha256)


# variant -> (palm detector, hand landmark)
VARIANTS: dict[str, tuple[ModelFile, ModelFile]] = {
    "fp32": (
        _zoo(
            PALM_DIR,
            "palm_detection_mediapipe_2023feb.onnx",
            3905734,
            "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
        ),
        _zoo(
            HAND_DIR,
            "handpose_estimation_mediapipe_2023feb.onnx",
            4099621,
            "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
        ),
    ),
    "int8": (
        _zoo(
            PALM_DIR,
            "palm_detection_mediapipe_2023feb_int8.onnx",
            1157004,
            "9f014de96ef5b6816b3eb9a5fed21a7371ef0f104ea440aa19ce9129fe2af5f6",
        ),
        _zoo(
            HAND_DIR,
            "handpose_estimation_mediapipe_2023feb_int8.onnx",
            1167628,
            "e97bc1fb83b641954d33424c82b6ade719d0f73250bdb91710ecfd5f7b47e321",
        ),
    ),
}

DEFAULT_THREADS = 2  # vision.threads (checked by the unit test)
DEFAULT_SPINNING = False  # vision.object_spinning (checked by the unit test)
DEFAULT_FRAME = (1280, 720)
DEFAULT_HAND_PX = 360
HAND_ROTATION_DEG = 20.0  # any non-zero angle: the zoo rotates every crop upright
SYNTHETIC_FRAMES = 8
CHANNELS = 3
PIXEL_MAX = 255.0
DOWNLOAD_TIMEOUT_S = 60
CHUNK = 1 << 16
MS = 1000.0
P50, P90 = 0.5, 0.9
LOG_ERROR = 3
STAGES = ("palm", "landmark", "pass")


class BenchError(RuntimeError):
    """A failure the command reports in one line (download, digest, missing dependency)."""


def parse_size(text: str) -> tuple[int, int]:
    """``WxH`` (for example ``1280x720``) as two positive integers."""
    parts = text.lower().split("x")
    try:
        w, h = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not WIDTHxHEIGHT") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be positive")
    return w, h


def parse_variants(text: str) -> list[str]:
    """Comma-separated variant names, in the given order, each known and listed once."""
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(VARIANTS)} (got {text!r})")
    return list(dict.fromkeys(names))


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be at least 1")
    return value


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be 0 or more")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options; see the module docstring."""
    p = argparse.ArgumentParser(
        prog="hands_cm4.py",
        description="Time the MediaPipe palm detector + hand landmark ONNX models on this CPU "
        "with synthetic frames (SPEC 5, 13). Touches no running app.",
    )
    p.add_argument("--runs", type=_positive, default=200, help="timed passes per variant")
    p.add_argument("--warmup", type=_non_negative, default=10, help="untimed passes first")
    p.add_argument(
        "--threads",
        type=_positive,
        default=DEFAULT_THREADS,
        help="ONNX Runtime intra-op threads and OpenCV threads (default: vision.threads, 2)",
    )
    p.add_argument(
        "--spinning",
        action="store_true",
        default=DEFAULT_SPINNING,
        help="let ONNX Runtime threads spin-wait between operators (default off, like "
        "vision.object_spinning)",
    )
    p.add_argument(
        "--variants",
        type=parse_variants,
        default=list(VARIANTS),
        help=f"comma-separated, from {', '.join(VARIANTS)} (default: all)",
    )
    p.add_argument(
        "--frame",
        type=parse_size,
        default=DEFAULT_FRAME,
        help="synthetic camera frame WxH (default 1280x720)",
    )
    p.add_argument(
        "--hand-px",
        type=_positive,
        default=DEFAULT_HAND_PX,
        help="side of the square hand crop in frame pixels (default 360)",
    )
    p.add_argument("--seed", type=int, default=0, help="seed of the synthetic frames")
    p.add_argument(
        "--cache-dir",
        type=Path,
        help="keep the downloaded models here and reuse verified ones (default: a temporary "
        "folder, deleted on exit)",
    )
    p.add_argument("--json", type=Path, help="also write the results as JSON to this file")
    a = p.parse_args(argv)
    if a.hand_px > min(a.frame):
        p.error(f"--hand-px {a.hand_px} does not fit in a {a.frame[0]}x{a.frame[1]} frame")
    return a


def nearest_rank(values: list[float], q: float) -> float:
    """The ``q`` quantile (0 to 1) of ``values`` by nearest rank; ``values`` must not be empty."""
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summarise(wall_s: list[float], cpu_s: list[float], net_s: list[float] | None) -> dict:
    """Wall ms (p50, p90, max), network-only ms p50, CPU ms per run p50, and CPU over wall."""
    wall_ms = [x * MS for x in wall_s]
    cpu_ms = [x * MS for x in cpu_s]
    total_wall = sum(wall_s)
    return {
        "runs": len(wall_s),
        "wall_p50_ms": round(nearest_rank(wall_ms, P50), 2),
        "wall_p90_ms": round(nearest_rank(wall_ms, P90), 2),
        "wall_max_ms": round(max(wall_ms), 2),
        "net_p50_ms": round(nearest_rank([x * MS for x in net_s], P50), 2) if net_s else None,
        "cpu_p50_ms": round(nearest_rank(cpu_ms, P50), 2),
        "cores": round(sum(cpu_s) / total_wall, 2) if total_wall > 0 else None,
    }


def letterbox(width: int, height: int, side: int) -> tuple[int, int, int, int]:
    """Centred letterbox of a ``width`` x ``height`` frame into a ``side`` square, aspect kept
    (the opencv_zoo palm detector's preprocessing): resized width, height, left and top pad."""
    scale = min(side / width, side / height)
    nw, nh = int(width * scale), int(height * scale)
    return nw, nh, (side - nw) // 2, (side - nh) // 2


def rotation_side(crop_px: int) -> int:
    """Side of the padded square that keeps every corner of a ``crop_px`` crop when rotated."""
    return math.ceil(math.hypot(crop_px, crop_px))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path: Path, spec: ModelFile) -> bool:
    """True when ``path`` exists with the pinned size and sha256."""
    return (
        path.is_file() and path.stat().st_size == spec.size_bytes and sha256_of(path) == spec.sha256
    )


def fetch(spec: ModelFile, folder: Path, urlopen: Any = urllib.request.urlopen) -> Path:
    """The verified file of ``spec`` in ``folder``: reused when it matches, else downloaded to a
    ``.part`` file that becomes the file only when its size and sha256 match the pin."""
    path = folder / spec.name
    if verified(path, spec):
        return path
    part = path.with_name(path.name + ".part")
    try:
        with urlopen(spec.url, timeout=DOWNLOAD_TIMEOUT_S) as r, part.open("wb") as f:
            shutil.copyfileobj(r, f, CHUNK)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise BenchError(f"download of {spec.name} failed: {e}") from e
    if not verified(part, spec):
        size = part.stat().st_size
        part.unlink()
        raise BenchError(
            f"{spec.name}: {size} bytes, digest does not match the pin "
            f"({spec.size_bytes} bytes, sha256 {spec.sha256}); file deleted"
        )
    part.replace(path)
    return path


def host_info() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("Model", "model name", "Hardware")):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "python": platform.python_version(),
        "cores_visible": os.cpu_count(),
    }


def load_average() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def _imports() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        raise BenchError(
            f"{e.name} is missing: run this with the RoboEmot environment's Python and the "
            "vision extra (onnxruntime, OpenCV, numpy)"
        ) from e
    return cv2, np, ort


def _session(ort: Any, path: Path, threads: int, spinning: bool) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = LOG_ERROR
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if not spinning:
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _square_nhwc(session: Any) -> tuple[str, int]:
    """Input name and side of a model with one NHWC float input of shape (1, S, S, 3)."""
    inp = session.get_inputs()[0]
    _, h, w, c = inp.shape
    if h != w or c != CHANNELS:
        raise BenchError(f"unexpected input shape {inp.shape} of {inp.name}")
    return inp.name, int(h)


def bench_variant(
    variant: str, palm_path: Path, hand_path: Path, frames: list[Any], args: argparse.Namespace
) -> dict:
    """Time ``args.runs`` passes (after ``args.warmup``) of one variant on ``frames`` in turn."""
    cv2, np, ort = _imports()
    palm = _session(ort, palm_path, args.threads, args.spinning)
    hand = _session(ort, hand_path, args.threads, args.spinning)
    palm_in, palm_side = _square_nhwc(palm)
    hand_in, hand_side = _square_nhwc(hand)
    width, height = args.frame
    nw, nh, left, top = letterbox(width, height, palm_side)
    crop = args.hand_px
    side = rotation_side(crop)
    x0, y0 = (width - crop) // 2, (height - crop) // 2
    pad = (side - crop) // 2
    rotate = cv2.getRotationMatrix2D((side / 2, side / 2), HAND_ROTATION_DEG, 1.0)

    def palm_pass(frame: Any) -> float:
        small = cv2.resize(frame, (nw, nh))
        boxed = cv2.copyMakeBorder(
            small, top, palm_side - nh - top, left, palm_side - nw - left, cv2.BORDER_CONSTANT
        )
        x = (cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).astype(np.float32) / PIXEL_MAX)[None]
        t0 = time.perf_counter()
        _, scores = palm.run(None, {palm_in: x})
        net = time.perf_counter() - t0
        int(np.argmax(1.0 / (1.0 + np.exp(-scores[0, :, 0]))))
        return net

    def hand_pass(frame: Any) -> float:
        roi = frame[y0 : y0 + crop, x0 : x0 + crop]
        padded = cv2.copyMakeBorder(
            roi, pad, side - crop - pad, pad, side - crop - pad, cv2.BORDER_CONSTANT
        )
        upright = cv2.warpAffine(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), rotate, (side, side))
        x = (cv2.resize(upright, (hand_side, hand_side)).astype(np.float32) / PIXEL_MAX)[None]
        t0 = time.perf_counter()
        hand.run(None, {hand_in: x})
        return time.perf_counter() - t0

    for i in range(args.warmup):
        palm_pass(frames[i % len(frames)])
        hand_pass(frames[i % len(frames)])
    wall: dict[str, list[float]] = {s: [] for s in STAGES}
    cpu: dict[str, list[float]] = {s: [] for s in STAGES}
    net: dict[str, list[float]] = {"palm": [], "landmark": []}
    for i in range(args.runs):
        frame = frames[i % len(frames)]
        w0, c0 = time.perf_counter(), time.process_time()
        net["palm"].append(palm_pass(frame))
        w1, c1 = time.perf_counter(), time.process_time()
        net["landmark"].append(hand_pass(frame))
        w2, c2 = time.perf_counter(), time.process_time()
        for stage, dw, dc in (
            ("palm", w1 - w0, c1 - c0),
            ("landmark", w2 - w1, c2 - c1),
            ("pass", w2 - w0, c2 - c0),
        ):
            wall[stage].append(dw)
            cpu[stage].append(dc)
    return {
        "variant": variant,
        "inputs": {"palm": palm_side, "landmark": hand_side},
        "files": {"palm": palm_path.name, "landmark": hand_path.name},
        "stages": {s: summarise(wall[s], cpu[s], net.get(s)) for s in STAGES},
    }


def synthetic_frames(width: int, height: int, seed: int) -> list[Any]:
    _, np, _ = _imports()
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, (height, width, CHANNELS), dtype=np.uint8)
        for _ in range(SYNTHETIC_FRAMES)
    ]


def render(report: dict) -> str:
    """The results as a markdown table with a one-line header of the run's settings."""
    h, s = report["host"], report["settings"]
    lines = [
        f"host: {h['cpu']} ({h['machine']}, {h['cores_visible']} cores), python {h['python']}, "
        f"onnxruntime {report['onnxruntime']}, opencv {report['opencv']}",
        f"threads={s['threads']} spinning={s['spinning']} runs={s['runs']} "
        f"frame={s['frame'][0]}x{s['frame'][1]} hand_px={s['hand_px']} "
        f"loadavg before={report['loadavg_before']} after={report['loadavg_after']}",
        "",
        "| variant | stage | input | wall p50 ms | wall p90 ms | wall max ms | net p50 ms "
        "| CPU p50 ms | cores |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report["results"]:
        if "error" in r:
            lines.append(f"| {r['variant']} | not run: {r['error']} | | | | | | | |")
            continue
        for stage, st in r["stages"].items():
            size = r["inputs"].get(stage)
            lines.append(
                f"| {r['variant']} | {stage} | {f'{size}x{size}' if size else ''} | "
                f"{st['wall_p50_ms']} | {st['wall_p90_ms']} | {st['wall_max_ms']} | "
                f"{'' if st['net_p50_ms'] is None else st['net_p50_ms']} | "
                f"{st['cpu_p50_ms']} | {st['cores']} |"
            )
    return "\n".join(lines)


def run(args: argparse.Namespace, folder: Path) -> dict:
    cv2, _, ort = _imports()
    ort.set_default_logger_severity(LOG_ERROR)
    cv2.setNumThreads(args.threads)
    frames = synthetic_frames(*args.frame, args.seed)
    before = load_average()
    results = []
    for variant in args.variants:
        palm_spec, hand_spec = VARIANTS[variant]
        palm_path, hand_path = fetch(palm_spec, folder), fetch(hand_spec, folder)
        try:
            results.append(bench_variant(variant, palm_path, hand_path, frames, args))
        except Exception as e:  # one variant the runtime cannot load must not stop the others
            if isinstance(e, BenchError):
                raise
            first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            results.append({"variant": variant, "error": first})
    return {
        "host": host_info(),
        "onnxruntime": ort.__version__,
        "opencv": cv2.__version__,
        "settings": {
            "threads": args.threads,
            "spinning": args.spinning,
            "runs": args.runs,
            "warmup": args.warmup,
            "frame": list(args.frame),
            "hand_px": args.hand_px,
            "zoo_commit": ZOO_COMMIT,
        },
        "loadavg_before": before,
        "loadavg_after": load_average(),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    keep = args.cache_dir is not None
    folder = args.cache_dir if keep else Path(tempfile.mkdtemp(prefix="roboemot-hands-"))
    try:
        folder.mkdir(parents=True, exist_ok=True)
        report = run(args, folder)
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        if not keep:
            shutil.rmtree(folder, ignore_errors=True)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
