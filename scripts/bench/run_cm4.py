"""Self-contained CPU latency bench for the M1 candidate models (SPEC 13, cm4 profile).

Copy this file and a models dir to the robot (Raspberry Pi CM4, aarch64, Python 3.9+) and run

    pip install -r requirements-cm4.txt
    python run_cm4.py --models-dir ./models --json cm4.json

Each model runs in its own subprocess so peak RSS is per model. Inputs are random tensors of the
shape the pipeline would feed (detection 320x240, face crops 112x112, objects 416x416 or
320x320, ...); latency of these
networks does not depend on pixel values. ONNX Runtime CPU, intra_op_num_threads=2 by default
(the cm4 budget gives vision two shared cores). YuNet runs through OpenCV (cv2.FaceDetectorYN),
its reference runtime, because the zoo ONNX has a fixed 640x640 input.

Missing model files are skipped. Expected file names are listed in MODELS below; the repo helper
`scripts/bench/fetch.py --export-flat DIR` produces them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

# name: (file, (N, C, H, W), note)
MODELS: dict[str, tuple[str, tuple[int, int, int, int], str]] = {
    "yunet_320x240": (
        "face_detection_yunet_2023mar.onnx",
        (1, 3, 240, 320),
        "face detection (cv2)",
    ),
    "scrfd_500m_320x240": ("det_500m.onnx", (1, 3, 240, 320), "face detection"),
    "scrfd_2.5g_320x240": ("scrfd_2.5g_bnkps.onnx", (1, 3, 240, 320), "face detection"),
    "scrfd_500m_640x480": ("det_500m.onnx", (1, 3, 480, 640), "face detection (reference size)"),
    "mbf_112": ("w600k_mbf.onnx", (1, 3, 112, 112), "face embedding"),
    "r50_112": ("w600k_r50.onnx", (1, 3, 112, 112), "face embedding"),
    "sface_112": ("face_recognition_sface_2021dec.onnx", (1, 3, 112, 112), "face embedding"),
    "genderage_96": ("genderage.onnx", (1, 3, 96, 96), "age/gender"),
    "fairface_age_vit_224": ("fairface_age_vit.onnx", (1, 3, 224, 224), "age (ViT-B)"),
    "fairface_gender_vit_224": ("fairface_gender_vit.onnx", (1, 3, 224, 224), "gender (ViT-B)"),
    "fairface_age_vit_int8_224": (
        "fairface_age_vit_int8.onnx",
        (1, 3, 224, 224),
        "age (ViT-B, INT8 dynamic)",
    ),
    "fairface_gender_vit_int8_224": (
        "fairface_gender_vit_int8.onnx",
        (1, 3, 224, 224),
        "gender (ViT-B, INT8 dynamic)",
    ),
    # The age model shipped 2026-10-22 to 2026-10-23 (removed with all age estimation, SPEC 5).
    "yolo11n_face_age_128": ("yolo11n_face_age.onnx", (1, 3, 128, 128), "age (retired, 128 px)"),
    "yolo11n_face_age_160": ("yolo11n_face_age.onnx", (1, 3, 160, 160), "age (retired, 160 px)"),
    "yolo11n_face_age_224": ("yolo11n_face_age.onnx", (1, 3, 224, 224), "age (retired, 224 px)"),
    "dinov2_s_224": ("dinov2_s.onnx", (1, 3, 224, 224), "individual embedding"),
    "clip_b32_224": ("clip_b32_vision.onnx", (1, 3, 224, 224), "individual embedding"),
    "mobileclip_s0_256": ("mobileclip_s0_vision.onnx", (1, 3, 256, 256), "individual embedding"),
    "yolox_nano_416": ("yolox_nano.onnx", (1, 3, 416, 416), "object detection (network only)"),
    "nanodet_plus_m_320": (
        "nanodet_plus_m_320.onnx",
        (1, 3, 320, 320),
        "object detection (network only)",
    ),
}

# name: (model names run back to back per call, both loaded in one process: RSS and ms per crop)
PAIRS: dict[str, tuple[str, str]] = {
    "fairface_pair_fp32": ("fairface_age_vit_224", "fairface_gender_vit_224"),
    "fairface_pair_int8": ("fairface_age_vit_int8_224", "fairface_gender_vit_int8_224"),
}


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def run_single(
    name: str, models_dir: Path, threads: int, runs: int, warmup: int, inter: int = 1
) -> dict:
    """Time one model (or a PAIRS entry, both models per call) in this process.

    Returns latency stats (ms) and peak RSS.
    """
    import numpy as np

    parts = PAIRS.get(name, (name,))
    fname, shape, note = MODELS[parts[0]]
    if name in PAIRS:
        note = "age + gender per crop, both loaded"
    paths = [models_dir / MODELS[p][0] for p in parts]
    path = paths[0]
    rng = np.random.default_rng(0)
    times: list[float] = []
    if name.startswith("yunet"):
        import cv2

        cv2.setNumThreads(threads)
        h, w = shape[2], shape[3]
        det = cv2.FaceDetectorYN.create(str(path), "", (w, h), 0.5, 0.3, 5000)
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        runner = lambda: det.detect(img)  # noqa: E731
        engine = f"opencv {cv2.__version__}"
    else:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = inter
        so.log_severity_level = 3
        sessions = [
            ort.InferenceSession(str(q), so, providers=["CPUExecutionProvider"]) for q in paths
        ]
        inp = sessions[0].get_inputs()[0].name
        x = rng.standard_normal(shape).astype(np.float32)

        def runner() -> None:
            for s in sessions:
                s.run(None, {inp: x})

        engine = f"onnxruntime {ort.__version__}"
    for _ in range(warmup):
        runner()
    for _ in range(runs):
        t0 = time.perf_counter()
        runner()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "model": name,
        "note": note,
        "file": fname,
        "size_mb": round(sum(q.stat().st_size for q in paths) / 1e6, 1),
        "input": list(shape),
        "engine": engine,
        "threads": threads,
        "runs": runs,
        "p50_ms": round(pct(times, 0.5), 2),
        "p95_ms": round(pct(times, 0.95), 2),
        "mean_ms": round(sum(times) / len(times), 2),
        "peak_rss_mb": round(peak_rss_mb(), 0),
    }


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models-dir", type=Path, required=False, default=Path("models"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--inter-threads", type=int, default=1, help="ORT inter_op_num_threads")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--only", default="", help="comma-separated model names")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--single", help="internal: run one model and print JSON")
    args = ap.parse_args()

    if args.single:
        print(
            json.dumps(
                run_single(
                    args.single,
                    args.models_dir,
                    args.threads,
                    args.runs,
                    args.warmup,
                    args.inter_threads,
                )
            )
        )
        return 0

    names = [n for n in [*MODELS, *PAIRS] if not args.only or n in args.only.split(",")]
    results = []
    for n in names:
        files = [MODELS[p][0] for p in PAIRS.get(n, (n,))]
        missing = [f for f in files if not (args.models_dir / f).exists()]
        if missing:
            print(f"skip {n}: {missing[0]} not in {args.models_dir}", file=sys.stderr)
            continue
        cmd = [sys.executable, __file__, "--single", n, "--models-dir", str(args.models_dir)]
        cmd += [
            "--threads",
            str(args.threads),
            "--runs",
            str(args.runs),
            "--warmup",
            str(args.warmup),
            "--inter-threads",
            str(args.inter_threads),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"FAILED {n}: {p.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
            continue
        results.append(json.loads(p.stdout.strip().splitlines()[-1]))

    info = host_info()
    print(
        f"\nhost: {info['cpu']} ({info['machine']}), python {info['python']}, "
        f"threads={args.threads}, runs={args.runs}\n"
    )
    print("| model | what | size MB | input | p50 ms | p95 ms | peak RSS MB |")
    print("| --- | --- | ---: | --- | ---: | ---: | ---: |")
    for r in results:
        i = "x".join(str(v) for v in r["input"][2:][::-1])
        print(
            f"| {r['model']} | {r['note']} | {r['size_mb']} | {i} | {r['p50_ms']} | "
            f"{r['p95_ms']} | {int(r['peak_rss_mb'])} |"
        )
    if args.json:
        args.json.write_text(json.dumps({"host": info, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
