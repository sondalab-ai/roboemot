"""Latency of the shipped vision code on this machine, labelled not-CM4 (SPEC 13, 14).

``track``: wall time from the first camera frame that shows a new person to ``person.recognized``,
through the real ``LatestFrameRunner``, ``pump`` and ``VisionPipeline`` (governor rates, burst
of ``vision.embedding_burst`` evaluations at least ``burst_min_gap_s`` apart, 3 of 5 voting).
The camera is a stand-in that reads at 30 Hz and shows a new LFW photo of the person every
0.2 s (independent photos, not video, so each evaluation sees a different one). Scenarios:
one new person at 5 and at 8 detections per second and three people appearing together. The
results in ``results/`` predate the removal of the age model (2026-10-23) and still hold a
scenario with the age job running.

``stages``: p50 and p95 of one call of each stage (YuNet, SFace, NanoDet, DINOv2-S) at 1 and
2 threads.

The Mac is not the Raspberry Pi CM4 (4x Cortex-A72); no number here is a CM4 number.

    OMP_NUM_THREADS=2 nice -n 10 python -I scripts/eval/latency.py stages
    OMP_NUM_THREADS=2 nice -n 10 python -I scripts/eval/latency.py track --trials 20
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

LFW = common.DATASETS / "lfw" / "x" / "lfw"
PHOTO_PERIOD_S = 0.2
CAMERA_HZ = 30.0
TRACK_WINDOW_S = 6.0
GAP_S = 2.0
ENROLLED = 10
CANVAS = (480, 640)
APPROX_BELOW = 30  # a p95 from fewer samples is only approximate
STAGE_RUNS = 60
WARMUP = 5


def stats(values: list[float]) -> dict:
    """p50, p95, max and n of a list of seconds or milliseconds."""
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": common.percentile(values, 0.5),
        "p95": common.percentile(values, 0.95),
        "max": max(values),
        "p95_approximate": len(values) < APPROX_BELOW,
    }


def bench(fn, runs: int = STAGE_RUNS) -> dict:
    """Milliseconds per call: p50, p95 over ``runs`` calls after a short warm-up."""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return stats(times)


def stage_row(threads: int, photo: np.ndarray, street: np.ndarray) -> dict:
    """p50 and p95 of one call of each stage with ``threads`` CPU threads."""
    from roboemot.vision import models as vmodels
    from roboemot.vision.onnx import DinoV2Embedder

    cfg = common.config({"vision": {"threads": threads}})
    row: dict = {}
    st = common.load_stages(cfg, profile="cm4", appearance=False)
    face = st.face_detector.detect(photo)[0]
    row["yunet_detect_320x240_incl_resize"] = bench(lambda: st.face_detector.detect(photo))
    row["sface_embed_incl_align"] = bench(lambda: st.face_embedder.embed(photo, face))
    row["nanodet_detect_640x480"] = bench(lambda: st.object_detector.detect(street))
    path = vmodels.check_file(common.MODELS, vmodels.load_manifest()["dinov2-small"].files[0])
    dino = DinoV2Embedder(path, threads)
    box = (0.0, 0.0, 250.0, 250.0)
    row["dinov2_small_embed"] = bench(lambda: dino.embed(photo, box), runs=30)
    return row


def stages(args: argparse.Namespace) -> None:
    import cv2

    photo = cv2.imread(str(next((LFW / "Colin_Powell").glob("*.jpg"))))
    street = cv2.resize(photo, (640, 480))
    out: dict = {}
    for threads in (1, 2):
        out[f"threads_{threads}"] = stage_row(threads, photo, street)
        common.log(f"threads {threads} done")
    result = {
        "unit": "milliseconds per call",
        "note": "Mac, not the CM4; random-free real inputs; includes pre- and post-processing",
        "stages": out,
        "machine": common.cpu_label(),
        "versions": common.versions(),
    }
    print(common.write_result("latency_stages", result))


class Scene:
    """Camera stand-in: blank until ``start``, then a new photo of each person every 0.2 s."""

    def __init__(self, tracks: list[list[np.ndarray]]):
        self.tracks = tracks
        self.t0: float | None = None
        self.n = 0
        self.canvas = np.zeros((*CANVAS, 3), np.uint8)

    def start(self) -> float:
        self.t0 = time.monotonic()
        return self.t0

    def read(self):
        from roboemot.vision import Frame

        now = time.monotonic()
        self.n += 1
        img = self.canvas.copy()
        if self.t0 is not None and now >= self.t0 and now - self.t0 < TRACK_WINDOW_S:
            idx = int((now - self.t0) / PHOTO_PERIOD_S)
            width = CANVAS[1] // len(self.tracks)
            for k, photos in enumerate(self.tracks):
                photo = photos[idx % len(photos)]
                side = min(width, CANVAS[0], 250 * 2 if len(self.tracks) == 1 else width)
                tile = cv2_resize(photo, side)
                img[0 : tile.shape[0], k * width : k * width + tile.shape[1]] = tile
        img[0, 0, 1], img[0, 0, 2] = self.n % 250, (self.n // 250) % 250
        return Frame(img, now)


def cv2_resize(photo: np.ndarray, side: int) -> np.ndarray:
    import cv2

    return cv2.resize(photo, (side, side))


def run_scenario(name, args, stages_, cfg, registry, people, tracks_per_trial):
    """Trials of one scenario; returns the per-person decision times in seconds."""
    import cv2
    from roboemot.bus import Bus
    from roboemot.registry import Matcher
    from roboemot.vision import LatestFrameRunner, VisionPipeline, pump

    rng = random.Random(1)
    decisions, firsts, extra_tracks, late = [], [], 0, []
    for trial in range(args.trials):
        chosen = rng.sample(people, tracks_per_trial)
        scene = Scene(
            [
                [cv2.imread(str(p)) for p in rng.sample(files, min(len(files), 25))]
                for _, files in chosen
            ]
        )
        bus = Bus()
        seen: dict[str, float] = {}
        first: dict[int, float] = {}

        def on_recognized(msg, seen=seen):
            seen.setdefault(msg.payload.person_id, time.monotonic())

        def on_face(msg, first=first):
            first.setdefault(msg.payload.track_id, time.monotonic())

        bus.subscribe("person.recognized", on_recognized)
        bus.subscribe("percept.face", on_face)
        pipeline = VisionPipeline(
            cfg.vision,
            cfg.identity,
            bus,
            Matcher(registry, cfg.identity),
            face_detector=stages_.face_detector,
            face_embedder=stages_.face_embedder,
            refresh=False,
        )
        runner = LatestFrameRunner(pipeline)
        stop = threading.Event()
        runner.start()
        thread = threading.Thread(target=pump, args=(scene, runner, stop, CAMERA_HZ), daemon=True)
        thread.start()
        time.sleep(0.5)
        t0 = scene.start()
        deadline = t0 + TRACK_WINDOW_S
        while time.monotonic() < deadline and len(seen) < tracks_per_trial:
            time.sleep(0.01)
        for ids, _ in chosen:
            decisions.append(seen[ids] - t0 if ids in seen else None)
        firsts += [min(first.values()) - t0] if first else []
        extra_tracks += max(0, len(first) - tracks_per_trial)
        late += [t - t0 for t in sorted(first.values())[tracks_per_trial:]]
        stop.set()
        thread.join()
        runner.stop()
        time.sleep(GAP_S)
        common.log(
            f"{name} trial {trial}: "
            f"{[None if d is None else round(d, 2) for d in decisions[-tracks_per_trial:]]}"
        )
    got = [d for d in decisions if d is not None]
    return {
        "scenario": name,
        "people": tracks_per_trial,
        "decisions": stats(got),
        "no_decision_within_window": len(decisions) - len(got),
        "first_percept_earliest_track_per_trial": stats(firsts),
        "tracks_beyond_the_planted_ones": extra_tracks,
        "extra_track_first_percept": stats(late),
    }


def track(args: argparse.Namespace) -> None:
    import cv2
    from roboemot.registry import Consent, FileStore, Registry

    base = common.config()
    results = []
    paths = {p.name: sorted(p.glob("*.jpg")) for p in LFW.iterdir() if p.is_dir()}
    pool = sorted(n for n, f in paths.items() if len(f) >= 40)
    rng = random.Random(0)
    names = rng.sample(pool, ENROLLED)
    scenarios = [
        ("one_person_5hz", 5.0, 1, 2),
        ("one_person_8hz", 8.0, 1, 2),
        ("three_people_8hz_burst", 8.0, 3, 2),
    ]
    for label, hz, people_n, threads in scenarios:
        if args.only and args.only not in label:
            continue
        cfg = common.config({"vision": {"rates": {"detection_hz": hz}, "threads": threads}})
        st = common.load_stages(cfg, profile="cm4", appearance=False, objects=False)
        folder = common.SCRATCH / "latency-registry"
        shutil.rmtree(folder, ignore_errors=True)
        registry = Registry(FileStore(folder, allow_plaintext=True), cfg.identity)
        consent = Consent("eval", datetime(2026, 1, 1, tzinfo=UTC), ("face",))
        people = []
        for n in names:
            embs = []
            for p in paths[n][:5]:
                image = cv2.imread(str(p))
                embs.append(st.face_embedder.embed(image, st.face_detector.detect(image)[0]))
            rec = registry.enroll(n[:2], embs, st.face_embedder.model_id, consent=consent)
            people.append((rec.person_id, paths[n][5:]))
        row = run_scenario(label, args, st, cfg, registry, people, people_n)
        row["detection_hz"] = hz
        row["threads"] = {"frame_path": threads}
        results.append(row)
        shutil.rmtree(folder, ignore_errors=True)
    out = {
        "protocol": {
            "trials_per_scenario": args.trials,
            "photo_period_s": PHOTO_PERIOD_S,
            "camera_hz": CAMERA_HZ,
            "enrolled": ENROLLED,
            "vision": base.vision.model_dump(
                include={"embedding_burst", "burst_min_gap_s", "max_embeds_per_frame"}
            ),
            "identity": base.identity.model_dump(include={"votes_required", "votes_window"}),
            "definition": "first camera frame showing the person to person.recognized, seconds",
        },
        "scenarios": results,
        "machine": common.cpu_label(),
        "versions": common.versions(),
    }
    print(common.write_result(f"latency_track{('_' + args.only) if args.only else ''}", out))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stages")
    t = sub.add_parser("track")
    t.add_argument("--trials", type=int, default=20)
    t.add_argument("--only", default="")
    args = ap.parse_args()
    stages(args) if args.cmd == "stages" else track(args)


if __name__ == "__main__":
    main()
