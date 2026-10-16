"""Age group and gender error per demographic group on FairFace (SPEC 5, 14 Perception eval).

Shipped path: frames go through ``VisionPipeline`` (YuNet, tracker, face crop with the shipped
margin, ``vision.attribute_crops`` crops at least ``attribute_crop_interval_s`` apart) into
``AttributeWorker`` (FairFace ViT-B pair, per-crop values, the median age group and the
majority gender with the 0.8 confidence rule of ``roboemot.registry.estimates``) and the result
comes back on ``percept.face``. ``load_stages`` picks the INT8 files for profile ``cm4`` and
FP32 otherwise.

FairFace has one image per face, so the three crops of a track are simulated: the original and
two small random shifts, scales, rotations and brightness changes (a stand-in for head motion
between frames, not real video). Images where YuNet finds no face start no track, no job runs
and the result is ``unknown`` (counted). FairFace ages are bins; 10-19 cannot be split at 13
and is scored as ``teen``. The minor-safety metric uses the shipped ``possible_minor``.

    python -I scripts/eval/attributes.py run --precision int8 --start 0 --count 700
    python -I scripts/eval/attributes.py report --out attributes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PARQUET_DIR = common.DATASETS / "fairface" / "0.25"
RACES = [
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Southeast Asian",
]
GENDERS = ["male", "female"]  # FairFace label order: 0 male, 1 female
BIN_NAMES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
BIN_GROUP = ["child", "child", "teen", "adult", "adult", "adult", "adult", "senior", "senior"]
GROUPS = ["child", "teen", "adult", "senior"]
MINOR_BINS = (0, 1, 2)  # under 20
CHILD_BINS = (0, 1)  # under 10
SUBSET = 1400
SEED = 0
FRAME_GAP_S = 0.35
SHIFT = 0.06
SCALE = (0.95, 1.06)
ROTATE_DEG = 4.0
BRIGHTNESS = (0.9, 1.1)
SCRATCH_PREFIX = "attrs"


def subset(total: int, n: int, seed: int) -> np.ndarray:
    """Sorted indices of a seeded random subset (the same call as the M1 spike)."""
    if not n or n >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, n, replace=False))


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A small random shift, scale, rotation and brightness change of a frame."""
    import cv2

    height, width = image.shape[:2]
    angle = rng.uniform(-ROTATE_DEG, ROTATE_DEG)
    scale = rng.uniform(*SCALE)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += rng.uniform(-SHIFT, SHIFT) * width
    matrix[1, 2] += rng.uniform(-SHIFT, SHIFT) * height
    out = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out.astype(np.float32) * rng.uniform(*BRIGHTNESS), 0, 255).astype(np.uint8)


class Recorder:
    """Wraps the estimator and keeps the per-crop samples of the current job."""

    def __init__(self, inner):
        self.inner, self.samples = inner, []

    def estimate(self, crop):
        sample = self.inner.estimate(crop)
        self.samples.append(sample)
        return sample


def run(args: argparse.Namespace) -> None:
    import cv2
    import pyarrow.parquet as pq
    from roboemot.bus import Bus
    from roboemot.registry import FileStore, Matcher, Registry
    from roboemot.vision import Frame, InlineExecutor, VisionPipeline

    profile = "cm4" if args.precision == "int8" else None
    cfg = common.config({"vision": {"attribute_threads": args.threads}})
    stages = common.load_stages(cfg, profile=profile, appearance=False, objects=False)
    parquet = next(PARQUET_DIR.glob("validation-*.parquet"))
    table = pq.ParquetFile(parquet).read().to_pydict()
    chosen = subset(len(table["age"]), SUBSET, SEED)[args.start : args.start + args.count]
    registry = Registry(
        FileStore(common.SCRATCH / "attrs-registry", allow_plaintext=True), cfg.identity
    )
    bus = Bus()
    percepts: list = []
    bus.subscribe("percept.face", lambda m: percepts.append(m.payload))
    recorder = Recorder(stages.attribute_estimator)
    pipeline = VisionPipeline(
        cfg.vision,
        cfg.identity,
        bus,
        Matcher(registry, cfg.identity),
        face_detector=stages.face_detector,
        face_embedder=stages.face_embedder,
        attribute_estimator=recorder,
        refresh=False,
        executor=InlineExecutor(),
    )
    rows = []
    started = time.perf_counter()
    for n, i in enumerate(chosen):
        original = cv2.imdecode(
            np.frombuffer(table["image"][i]["bytes"], np.uint8), cv2.IMREAD_COLOR
        )
        rng = np.random.default_rng(int(i))
        a, b = augment(original, rng), augment(original, rng)
        frames = [original, original, a, b, b]
        pipeline.reset()
        recorder.samples.clear()
        last = None
        for j, image in enumerate(frames):
            percepts.clear()
            pipeline.process(Frame(image, 100.0 + j * FRAME_GAP_S))
            if percepts:
                last = percepts[0]
        samples = [
            (s.age_group, s.age_confidence, s.gender, s.gender_confidence) for s in recorder.samples
        ]
        rows.append(
            {
                "idx": int(i),
                "face": last is not None,
                "job": len(samples) == cfg.vision.attribute_crops,
                "age": last.age_group if last else "unknown",
                "gender": last.gender if last else "unknown",
                "crops": samples,
            }
        )
        if n % 50 == 0:
            common.log(f"{args.precision} {n}/{len(chosen)} {time.perf_counter() - started:.0f}s")
    elapsed = time.perf_counter() - started
    pipeline.close()
    out = common.SCRATCH / f"{SCRATCH_PREFIX}_{args.precision}_{args.start}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "precision": args.precision,
        "start": args.start,
        "count": len(chosen),
        "seconds": elapsed,
        "threads": args.threads,
    }
    out.write_text(json.dumps({"meta": meta, "rows": rows}))
    common.log(f"wrote {len(rows)} rows, {elapsed:.0f}s, {elapsed / max(1, len(rows)):.2f}s/face")


def load_rows(precision: str) -> tuple[dict[int, dict], list[dict]]:
    """Merged per-face rows of all parts of one precision, keyed by dataset index."""
    rows: dict[int, dict] = {}
    metas = []
    for path in sorted(common.SCRATCH.glob(f"{SCRATCH_PREFIX}_{precision}_*.json")):
        data = json.loads(path.read_text())
        metas.append(data["meta"])
        rows.update({r["idx"]: r for r in data["rows"]})
    return rows, metas


def bucket(rows: list[dict], truth: dict[int, dict]) -> dict:
    """Error rates of a set of faces: aggregated per track, with the unknown rate and safety."""
    r = common.rate
    n = len(rows)
    age_known = [x for x in rows if x["age"] != "unknown"]
    gen_known = [x for x in rows if x["gender"] != "unknown"]
    age_wrong = sum(x["age"] != truth[x["idx"]]["group"] for x in age_known)
    gen_wrong = sum(x["gender"] != truth[x["idx"]]["gender"] for x in gen_known)
    minors = [x for x in rows if truth[x["idx"]]["bin"] in MINOR_BINS]
    children = [x for x in rows if truth[x["idx"]]["bin"] in CHILD_BINS]
    adults = [x for x in rows if truth[x["idx"]]["group"] in ("adult", "senior")]
    adult_like = ("adult", "senior")
    return {
        "n": n,
        "no_face_found": r(sum(not x["face"] for x in rows), n),
        "age_unknown": r(n - len(age_known), n),
        "age_group_error_of_known": r(age_wrong, len(age_known)),
        "gender_unknown": r(n - len(gen_known), n),
        "gender_error_of_known": r(gen_wrong, len(gen_known)),
        "minors_n": len(minors),
        "minor_labelled_adult_or_senior": r(
            sum(x["age"] in adult_like for x in minors), len(minors)
        ),
        "children_under_10_n": len(children),
        "child_labelled_adult_or_senior": r(
            sum(x["age"] in adult_like for x in children), len(children)
        ),
        "adults_n": len(adults),
        "adult_or_senior_treated_as_possible_minor": r(
            sum(x["age"] not in adult_like for x in adults), len(adults)
        ),
    }


def single_crop(rows: list[dict], truth: dict[int, dict], conf: float) -> dict:
    """The same tables from the first crop alone with the 0.8 rule (no aggregation)."""
    out = []
    for x in rows:
        if not x["crops"]:
            out.append({**x, "age": "unknown", "gender": "unknown"})
            continue
        age, age_conf, gender, gender_conf = x["crops"][0]
        out.append(
            {
                **x,
                "age": age if age_conf >= conf else "unknown",
                "gender": gender if gender_conf >= conf else "unknown",
            }
        )
    return bucket(out, truth)


def report(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    cfg = common.config()
    parquet = next(PARQUET_DIR.glob("validation-*.parquet"))
    table = pq.ParquetFile(parquet).read(columns=["age", "gender", "race"]).to_pydict()
    truth = {
        i: {
            "bin": table["age"][i],
            "group": BIN_GROUP[table["age"][i]],
            "gender": GENDERS[table["gender"][i]],
            "race": RACES[table["race"][i]],
        }
        for i in range(len(table["age"]))
    }
    result: dict = {
        "protocol": {
            "dataset": "FairFace validation (10,954 faces; one image per face)",
            "subset": f"seeded random {SUBSET} (seed {SEED}, the M1 spike's indices); each "
            "precision ran on the faces listed in its part files",
            "crops_per_track": cfg.vision.attribute_crops,
            "estimate_frames_min": cfg.identity.estimate_frames_min,
            "estimate_confidence_min": cfg.identity.estimate_confidence_min,
            "crop_margin": cfg.vision.attribute_crop_margin,
            "crops_simulated_by": f"original + 2 random augmentations: shift +-{SHIFT:.0%}, "
            f"scale {SCALE}, rotation +-{ROTATE_DEG} deg, brightness {BRIGHTNESS}",
        },
        "machine": common.cpu_label(),
        "versions": common.versions(),
    }
    runs = {p: load_rows(p) for p in ("int8", "fp32")}
    int8_rows = runs["int8"][0]
    for precision, (rows, metas) in runs.items():
        if not rows:
            continue
        data = [rows[i] for i in sorted(rows)]
        by: dict[str, dict] = {"all": bucket(data, truth)}
        for race in RACES:
            by[f"race:{race}"] = bucket([x for x in data if truth[x["idx"]]["race"] == race], truth)
        for gender in GENDERS:
            by[f"gender:{gender}"] = bucket(
                [x for x in data if truth[x["idx"]]["gender"] == gender], truth
            )
        for race in RACES:
            for gender in GENDERS:
                key = f"race_x_gender:{race}|{gender}"
                by[key] = bucket(
                    [
                        x
                        for x in data
                        if truth[x["idx"]]["race"] == race and truth[x["idx"]]["gender"] == gender
                    ],
                    truth,
                )
        for name in BIN_NAMES:
            b = BIN_NAMES.index(name)
            by[f"age_bin:{name}"] = bucket([x for x in data if truth[x["idx"]]["bin"] == b], truth)
        for group in GROUPS:
            by[f"age_group:{group}"] = bucket(
                [x for x in data if truth[x["idx"]]["group"] == group], truth
            )
        confusion = {g: {h: 0 for h in (*GROUPS, "unknown")} for g in GROUPS}
        for x in data:
            confusion[truth[x["idx"]]["group"]][x["age"]] += 1
        result[precision] = {
            "n": len(data),
            "parts": metas,
            "aggregated": by,
            "first_crop_only_all": single_crop(data, truth, cfg.identity.estimate_confidence_min),
            "age_group_confusion_true_by_estimate": confusion,
        }
    fp32 = runs["fp32"][0]
    if fp32 and int8_rows:
        common_idx = sorted(set(fp32) & set(int8_rows))
        result["int8_vs_fp32_same_faces"] = {
            "n": len(common_idx),
            "int8": bucket([int8_rows[i] for i in common_idx], truth),
            "fp32": bucket([fp32[i] for i in common_idx], truth),
            "age_label_differs": common.rate(
                sum(int8_rows[i]["age"] != fp32[i]["age"] for i in common_idx), len(common_idx)
            ),
            "gender_label_differs": common.rate(
                sum(int8_rows[i]["gender"] != fp32[i]["gender"] for i in common_idx),
                len(common_idx),
            ),
        }
    print(common.write_result(args.out, result))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--precision", choices=["int8", "fp32"], required=True)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--count", type=int, required=True)
    r.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
