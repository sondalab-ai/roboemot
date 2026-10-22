"""Age group and gender error per demographic group on FairFace (SPEC 5, 14 Perception eval).

Written for the FairFace ViT-B pair, retired on 2026-10-22 (SPEC 5, 13); ``results/`` holds its
numbers. ``load_stages`` now loads yolo11n-face-age whatever ``--precision`` says, so the int8 and
fp32 runs are the same model, and the 0.8-rule tables (``single_crop``, ``crop_stats``) do not
describe it: its floor is ``vision.attribute_score_min``. Rework before using it on the new model.

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
    python -I scripts/eval/attributes.py run --precision fp32 --full --max-seconds 3000
    python -I scripts/eval/attributes.py report --out attributes

``--full`` runs all validation faces and appends one JSON line per face to
``attrs_<precision>_full.jsonl`` in the scratch folder, so a stopped run resumes where it ended
(``--max-seconds`` stops cleanly). The report then also prints the 1,400-face subset of the
same run, so subset and full-set numbers come from identical per-face results.
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
    total = len(table["age"])
    if args.full:
        cache = common.SCRATCH / f"{SCRATCH_PREFIX}_{args.precision}_full.jsonl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.touch()
        if cache.stat().st_size and not cache.read_text().endswith("\n"):
            cache.open("a").write("\n")
        done = {r["idx"] for r in read_jsonl(cache)}
        chosen = np.array([i for i in range(total) if i not in done])
        common.log(f"{args.precision} full: {len(done)} cached, {len(chosen)} to run")
    else:
        chosen = subset(total, SUBSET, SEED)[args.start : args.start + args.count]
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
        if args.full and time.perf_counter() - started > args.max_seconds:
            common.log(f"stopping after {n} faces: --max-seconds reached")
            break
        face_started = time.perf_counter()
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
        row = {
            "idx": int(i),
            "face": last is not None,
            "job": len(samples) == cfg.vision.attribute_crops,
            "age": last.age_group if last else "unknown",
            "gender": last.gender if last else "unknown",
            "crops": samples,
            "t": time.perf_counter() - face_started,
        }
        rows.append(row)
        if args.full:
            with cache.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        if n % 50 == 0:
            common.log(f"{args.precision} {n}/{len(chosen)} {time.perf_counter() - started:.0f}s")
    elapsed = time.perf_counter() - started
    pipeline.close()
    if args.full:
        common.log(f"wrote {len(rows)} rows to the cache, {elapsed:.0f}s")
        return
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


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a resume cache; a last line cut short by a killed run is dropped."""
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def load_rows(precision: str) -> tuple[dict[int, dict], list[dict]]:
    """Merged per-face rows of all parts of one precision, keyed by dataset index."""
    rows: dict[int, dict] = {}
    metas = []
    for path in sorted(common.SCRATCH.glob(f"{SCRATCH_PREFIX}_{precision}_*.json")):
        data = json.loads(path.read_text())
        metas.append(data["meta"])
        rows.update({r["idx"]: r for r in data["rows"]})
    cache = common.SCRATCH / f"{SCRATCH_PREFIX}_{precision}_full.jsonl"
    if cache.exists():
        full = read_jsonl(cache)
        metas.append(
            {
                "precision": precision,
                "full_run": True,
                "count": len(full),
                "seconds": sum(r.get("t", 0.0) for r in full),
            }
        )
        rows.update({r["idx"]: r for r in full})
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


def crop_stats(rows: list[dict], cfg) -> dict:
    """Per-crop argmax age groups with their mean probability and share at the 0.8 rule.

    Shows which groups the model predicts on single crops and whether they pass the confidence
    rule. ``faces_with_two_jobs`` counts faces where the tracker split the track and a second
    job ran (more crops than ``vision.attribute_crops``).
    """
    need = cfg.vision.attribute_crops
    conf_min = cfg.identity.estimate_confidence_min
    by: dict[str, list[float]] = {g: [] for g in GROUPS}
    for x in rows:
        for group, probability, _, _ in x["crops"]:
            by[group].append(probability)
    return {
        "crops": sum(len(v) for v in by.values()),
        "by_group": {
            g: {
                "crops": len(v),
                "mean_probability": (sum(v) / len(v)) if v else None,
                "share_at_or_above_rule": (sum(p >= conf_min for p in v) / len(v)) if v else None,
            }
            for g, v in by.items()
        },
        "faces_with_two_jobs": common.rate(sum(len(x["crops"]) > need for x in rows), len(rows)),
    }


def paired(flags_a: list[bool], flags_b: list[bool]) -> dict:
    """INT8 (a) minus FP32 (b) on the same faces: rates, difference, 95% CI and exact McNemar p.

    ``only_int8`` / ``only_fp32`` count the faces where just one precision has the property. The
    difference interval is the normal one on the paired discordant counts; ``p_exact`` is the
    two-sided exact binomial test of the discordant counts (ties give 1).
    """
    from math import comb, sqrt

    n = len(flags_a)
    only_a = sum(x and not y for x, y in zip(flags_a, flags_b, strict=True))
    only_b = sum(y and not x for x, y in zip(flags_a, flags_b, strict=True))
    d = (only_a - only_b) / n if n else 0.0
    se = sqrt(max(0.0, only_a + only_b - (only_a - only_b) ** 2 / n)) / n if n else 0.0
    m = only_a + only_b
    tail = sum(comb(m, k) for k in range(min(only_a, only_b) + 1)) / 2**m if m else 1.0
    return {
        "n": n,
        "int8": common.rate(sum(flags_a), n),
        "fp32": common.rate(sum(flags_b), n),
        "only_int8": only_a,
        "only_fp32": only_b,
        "difference_int8_minus_fp32": d,
        "difference_ci95": [d - common.WILSON_Z * se, d + common.WILSON_Z * se],
        "p_exact": min(1.0, 2 * tail),
    }


def paired_table(a: dict[int, dict], b: dict[int, dict], ids: list[int], truth: dict) -> dict:
    """The key safety and accuracy properties as paired INT8-vs-FP32 comparisons."""
    adult_like = ("adult", "senior")
    props = {
        "minor_labelled_adult_or_senior": (
            lambda x, t: x["age"] in adult_like,
            lambda t: t["bin"] in MINOR_BINS,
        ),
        "child_labelled_adult_or_senior": (
            lambda x, t: x["age"] in adult_like,
            lambda t: t["bin"] in CHILD_BINS,
        ),
        "age_unknown": (lambda x, t: x["age"] == "unknown", lambda t: True),
        "age_group_wrong_of_all": (
            lambda x, t: x["age"] != "unknown" and x["age"] != t["group"],
            lambda t: True,
        ),
        "gender_unknown": (lambda x, t: x["gender"] == "unknown", lambda t: True),
        "gender_wrong_of_all": (
            lambda x, t: x["gender"] != "unknown" and x["gender"] != t["gender"],
            lambda t: True,
        ),
        "adult_or_senior_treated_as_possible_minor": (
            lambda x, t: x["age"] not in adult_like,
            lambda t: t["group"] in adult_like,
        ),
    }
    out = {}
    for name, (flag, select) in props.items():
        keep = [i for i in ids if select(truth[i])]
        out[name] = paired(
            [flag(a[i], truth[i]) for i in keep], [flag(b[i], truth[i]) for i in keep]
        )
    return out


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
            "dataset": f"FairFace validation ({len(truth):,} faces; one image per face)",
            "subset": f"seeded random {SUBSET} (seed {SEED}, the M1 spike's indices); "
            "`subset_of_this_run` and `paired_int8_vs_fp32_subset` restrict a full run to it",
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
            "per_crop_predictions": crop_stats(data, cfg),
        }
    fp32 = runs["fp32"][0]
    in_subset = set(subset(len(truth), SUBSET, SEED).tolist())
    for precision, (rows, _) in runs.items():
        sub = [rows[i] for i in sorted(rows) if i in in_subset]
        if rows and len(rows) > len(sub):
            result[precision]["subset_of_this_run"] = bucket(sub, truth)
    if fp32 and int8_rows:
        both = sorted(set(fp32) & set(int8_rows))
        result["paired_int8_vs_fp32_all_common"] = paired_table(int8_rows, fp32, both, truth)
        result["paired_int8_vs_fp32_subset"] = paired_table(
            int8_rows, fp32, [i for i in both if i in in_subset], truth
        )
        for key in ("race", "gender"):
            groups = RACES if key == "race" else GENDERS
            result[f"paired_by_{key}"] = {
                g: paired_table(int8_rows, fp32, [i for i in both if truth[i][key] == g], truth)
                for g in groups
            }
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
    r.add_argument("--count", type=int, default=0)
    r.add_argument("--full", action="store_true")
    r.add_argument("--max-seconds", type=float, default=float("inf"))
    r.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
