"""Animals and objects: detection and auto-enrollment of the shipped code (SPEC 5, 5A, 14).

``detect``: ``NanoDetDetector`` (``roboemot.vision.onnx.load_object_detector``) with the shipped
classes and ``vision.object_score_min``, on a seeded COCO val2017 subset; recall and precision
per shipped class and per kind (animal, object) at IoU 0.5, one frame at a time (the tracker's
2-hit confirmation, SPEC 13, needs video and is not applied).

``dogs`` and ``sop``: ``DinoV2Embedder`` appearance embeddings of whole photos of MPDD dogs and
Stanford Online Products household items go through the shipped ``AutoEnroller``
(``roboemot.world.autoenroll``: join radius, spread, days counted from retained members, known
match threshold) in a simulated multi-day sighting stream, and each enrolled record is scored
against the individuals that fed it.

    OMP_NUM_THREADS=2 nice -n 10 python -I scripts/eval/objects.py detect --n 1500
    OMP_NUM_THREADS=2 nice -n 10 python -I scripts/eval/objects.py dogs
    OMP_NUM_THREADS=2 nice -n 10 python -I scripts/eval/objects.py sop
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

COCO = common.DATASETS / "coco"
SEED = 0
IOU = 0.5
SMALL_AREA, MEDIUM_AREA = 32**2, 96**2
SOP_CATEGORIES = ["mug", "kettle", "stapler", "toaster", "coffee maker"]
SOP_ITEMS_PER_CATEGORY = 150
SOP_IMAGES_PER_ITEM = 6
STREAM_INDIVIDUALS = 20
STREAM_DAYS = 6
STREAM_PRESENCE = 0.8
STREAM_PER_DAY = (1, 2)
STREAM_REPEATS = 10
SIGHTING_GAP_S = 400.0  # above world.autoenroll.sighting_min_interval_s
DAY0 = datetime(2026, 1, 1, 12, 0).astimezone()


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of boxes a (N, 4) and b (M, 4), xyxy."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(dets: list[tuple[np.ndarray, float]], gts: np.ndarray, crowd: np.ndarray):
    """Greedy by score: (matched GT flags, false positives); crowd-only overlaps are ignored."""
    taken = np.zeros(len(gts), bool)
    fp = 0
    for box, _ in sorted(dets, key=lambda d: -d[1]):
        ious = iou(box[None], gts)[0] if len(gts) else np.zeros(0)
        hit = next(
            (int(j) for j in np.argsort(-ious) if ious[j] >= IOU and not taken[j] and not crowd[j]),
            None,
        )
        if hit is not None:
            taken[hit] = True
        elif len(gts) and (ious[crowd] >= IOU).any():
            continue
        else:
            fp += 1
    return taken, fp


def detect(args: argparse.Namespace) -> None:
    import cv2

    cfg = common.config()
    detector = common_detector(cfg)
    data = json.loads((COCO / "instances_val2017.json").read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    ids = sorted(i["id"] for i in data["images"])
    files = {i["id"]: i["file_name"] for i in data["images"]}
    chosen = sorted(
        int(i)
        for i in np.random.default_rng(SEED).choice(ids, min(args.n, len(ids)), replace=False)
    )
    kinds = {
        **dict.fromkeys(cfg.vision.object_categories, "object"),
        **dict.fromkeys(cfg.vision.animal_categories, "animal"),
    }
    by_name = {v.replace("_", " "): v for v in kinds}
    shipped = {c for c, n in names.items() if n in by_name}
    gts: dict[tuple[int, str], list] = defaultdict(list)
    for a in data["annotations"]:
        if a["category_id"] in shipped:
            x, y, w, h = a["bbox"]
            gts[(a["image_id"], names[a["category_id"]])].append(
                ([x, y, x + w, y + h], bool(a["iscrowd"]), a["area"])
            )
    tally = defaultdict(lambda: defaultdict(int))
    times, missing = [], 0
    score_min = cfg.vision.object_score_min
    for n, image_id in enumerate(chosen):
        path = COCO / "val2017" / files[image_id]
        image = cv2.imread(str(path)) if path.exists() else None
        if image is None:
            missing += 1
            continue
        import time

        t0 = time.perf_counter()
        found = [d for d in detector.detect(image) if d.score >= score_min and d.label in kinds]
        times.append((time.perf_counter() - t0) * 1000)
        per_class: dict[str, list] = defaultdict(list)
        for d in found:
            per_class[d.label.replace("_", " ")].append((np.array(d.box, np.float64), d.score))
        for name in by_name:
            truth = gts.get((image_id, name), [])
            boxes = np.array([t[0] for t in truth], np.float64).reshape(-1, 4)
            crowd = np.array([t[1] for t in truth], bool)
            taken, fp = match(per_class.get(name, []), boxes, crowd)
            row = tally[name]
            row["fp"] += fp
            row["tp"] += int(taken.sum())
            for hit, (_, is_crowd, area) in zip(taken, truth, strict=True):
                if is_crowd:
                    continue
                row["gt"] += 1
                size = "small" if area < SMALL_AREA else "medium" if area < MEDIUM_AREA else "large"
                row[f"gt_{size}"] += 1
                row[f"tp_{size}"] += int(hit)
        if n % 300 == 0:
            common.log(f"coco {n}/{len(chosen)}")

    def summary(rows: list[dict]) -> dict:
        t = {k: sum(r.get(k, 0) for r in rows) for k in ("tp", "fp", "gt")}
        out = {
            "recall": common.rate(t["tp"], t["gt"]),
            "precision": common.rate(t["tp"], t["tp"] + t["fp"]),
            "false_positives": t["fp"],
        }
        for size in ("small", "medium", "large"):
            out[f"recall_{size}"] = common.rate(
                sum(r.get(f"tp_{size}", 0) for r in rows), sum(r.get(f"gt_{size}", 0) for r in rows)
            )
        return out

    result = {
        "protocol": {
            "images": len(chosen) - missing,
            "seed": SEED,
            "score_min": score_min,
            "nms_iou": cfg.vision.object_nms_iou,
            "iou_match": IOU,
            "note": "single frames; the tracker's object_confirm_hits is not applied",
            "animal_classes": list(cfg.vision.animal_categories),
            "object_classes": list(cfg.vision.object_categories),
        },
        "per_class": {n: summary([tally[n]]) for n in by_name},
        "animal": summary([tally[n] for n in by_name if kinds[by_name[n]] == "animal"]),
        "object": summary([tally[n] for n in by_name if kinds[by_name[n]] == "object"]),
        "all_shipped_classes": summary(list(tally.values())),
        "detect_ms": {
            "p50": common.percentile(times, 0.5),
            "p95": common.percentile(times, 0.95),
            "note": "Mac, includes decode-free detect() only",
        },
        "machine": common.cpu_label(),
        "versions": common.versions(),
    }
    print(common.write_result("objects_detection", result))


def common_detector(cfg):
    """The shipped NanoDet detector from the bench models folder."""
    from roboemot.vision.onnx import load_object_detector

    return load_object_detector(common.MODELS, cfg.vision)


def embed_all(items, cfg, name: str):
    """DINOv2-S embeddings of (loader) images via the shipped embedder; cached in scratch."""
    from roboemot.vision import models as vmodels
    from roboemot.vision.onnx import DinoV2Embedder

    cache = common.SCRATCH / f"dino_{name}.npz"
    if cache.exists():
        data = np.load(cache)
        common.log(f"{name}: embeddings from the scratch cache")
        return data["emb"], list(data["ids"]), list(data["cats"])
    path = vmodels.check_file(common.MODELS, vmodels.load_manifest()["dinov2-small"].files[0])
    embedder = DinoV2Embedder(path, cfg.vision.threads)
    emb, ids, cats = [], [], []
    for n, (image, ident, cat) in enumerate(items()):
        height, width = image.shape[:2]
        emb.append(embedder.embed(image, (0.0, 0.0, float(width), float(height))))
        ids.append(ident)
        cats.append(cat)
        if n % 300 == 0:
            common.log(f"{name}: embedded {n}")
    common.SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=np.stack(emb), ids=np.array(ids), cats=np.array(cats))
    return np.stack(emb), ids, cats


def mpdd_items():
    import cv2

    root = common.DATASETS / "mpdd" / "x" / "MPDD" / "pytorch"
    for p in sorted(root.glob("*/*.jpg")):
        yield cv2.imread(str(p)), f"dog:{p.stem.split('_')[0]}", "dog"


def sop_items():
    import cv2
    import pyarrow.parquet as pq

    rng = random.Random(SEED)
    for pf in sorted((common.DATASETS / "sop").glob("test-*.parquet")):
        meta = pq.ParquetFile(pf).read(columns=["item_id", "category"]).to_pydict()
        rows: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, (item, cat) in enumerate(zip(meta["item_id"], meta["category"], strict=True)):
            if cat in SOP_CATEGORIES:
                rows[(cat, item)].append(i)
        per_cat: dict[str, list] = defaultdict(list)
        for (cat, item), idx in rows.items():
            if len(idx) >= SOP_IMAGES_PER_ITEM:
                per_cat[cat].append((item, sorted(idx)[:SOP_IMAGES_PER_ITEM]))
        picked = []
        for cat, items in sorted(per_cat.items()):
            rng.shuffle(items)
            picked += [(cat, item, i) for item, idx in items[:SOP_ITEMS_PER_CATEGORY] for i in idx]
        if not picked:
            continue
        column = pq.ParquetFile(pf).read(columns=["image"]).column("image")
        for cat, item, i in picked:
            raw = column[i].as_py()["bytes"]
            yield (
                cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
                f"sop:{cat}:{item}",
                cat,
            )


def stream(emb, ids, category: str, kind: str, rng: random.Random, cfg, tag: str) -> dict:
    """One simulated multi-day stream through the shipped AutoEnroller; scored per record."""
    from roboemot.registry import FileStore, Registry
    from roboemot.world import Observation
    from roboemot.world.autoenroll import AutoEnroller
    from roboemot.world.types import Entry, ObserveResult

    by_id: dict[str, list[int]] = defaultdict(list)
    for i, ident in enumerate(ids):
        by_id[ident].append(i)
    pool = sorted(k for k, v in by_id.items() if len(v) >= SOP_IMAGES_PER_ITEM)
    members = rng.sample(pool, min(STREAM_INDIVIDUALS, len(pool)))
    folder = common.SCRATCH / f"autoenroll-{tag}"
    shutil.rmtree(folder, ignore_errors=True)
    registry = Registry(FileStore(folder, allow_plaintext=True), cfg.identity)
    clock = {"day": 0, "t": 0.0}
    enroller = AutoEnroller(
        registry,
        registry.store,
        cfg.world.autoenroll,
        "dinov2-small",
        clock=lambda: DAY0 + timedelta(days=clock["day"]),
        monotonic=lambda: clock["t"],
    )
    remaining = {m: rng.sample(by_id[m], len(by_id[m])) for m in members}
    attached: list[tuple[str, str]] = []
    n_sightings = 0
    for day in range(STREAM_DAYS):
        clock["day"] = day
        today = []
        for m in members:
            if rng.random() <= STREAM_PRESENCE:
                for _ in range(rng.randint(*STREAM_PER_DAY)):
                    if remaining[m]:
                        today.append((m, remaining[m].pop()))
        rng.shuffle(today)
        for m, row in today:
            clock["t"] += SIGHTING_GAP_S
            n_sightings += 1
            obs = Observation(kind, clock["t"], "eval", 0.0, 0.0, category, embedding=emb[row])
            entry = Entry(
                f"e{n_sightings}", kind, category, "unknown", 0.0, 0.0, None, 0.0, 0.5, 0.0, 0.0
            )  # type: ignore[arg-type]
            owner = enroller.observe(obs, ObserveResult(entry=entry))
            if owner is not None:
                attached.append((owner, m))
    records = [r for r in registry.entities() if r.origin == "auto"]
    sight_label = {}
    for m in members:
        for row in by_id[m]:
            sight_label[row] = m
    record_label = {}
    pure = mixed = 0
    for r in records:
        sims = np.asarray(r.embeddings) @ emb.T
        labels = [sight_label[int(j)] for j in sims.argmax(1)]
        top, count = max(((x, labels.count(x)) for x in set(labels)), key=lambda t: t[1])
        record_label[r.entity_id] = top
        if len(set(labels)) == 1:
            pure += 1
        else:
            mixed += 1
    enrolled_individuals = list(record_label.values())
    seen_individuals = {m for m in members if len(by_id[m]) > len(remaining[m])}
    duplicates = sum(
        c - 1 for c in (enrolled_individuals.count(x) for x in set(enrolled_individuals))
    )
    wrong_attach = sum(record_label.get(owner) != m for owner, m in attached)
    return {
        "individuals_in_stream": len(members),
        "sightings": n_sightings,
        "records_created": len(records),
        "records_pure": pure,
        "records_mixing_individuals": mixed,
        "individuals_with_a_record": len(set(enrolled_individuals)),
        "duplicate_records_of_one_individual": duplicates,
        "individuals_seen": len(seen_individuals),
        "sightings_attached_to_a_record": len(attached),
        "attached_to_the_wrong_individual": wrong_attach,
        "candidate_clusters_left": enroller.stats()["clusters"],
    }


def retrieval(args: argparse.Namespace, name: str, items, categories: list[str], kind: str) -> None:
    cfg = common.config()
    emb, ids, cats = embed_all(items, cfg, name)
    ids_a, cats_a = np.array(ids), np.array(cats)
    runs, per_category = [], defaultdict(list)
    for category in categories:
        mask = cats_a == category
        if not mask.any():
            continue
        for r in range(STREAM_REPEATS):
            rng = random.Random(SEED * 100 + r)
            res = stream(
                emb[mask],
                list(ids_a[mask]),
                category,
                kind,
                rng,
                cfg,
                f"{name}-{category}-{r}".replace(" ", "_"),
            )
            per_category[category].append(res)
            runs.append(res)

    def total(rows: list[dict]) -> dict:
        out = {k: sum(r[k] for r in rows) for k in rows[0]}
        out["streams"] = len(rows)
        return out

    result = {
        "protocol": {
            "dataset": name,
            "embedder": "DINOv2-S (shipped DinoV2Embedder, whole photo as the box)",
            "enroller": "roboemot.world.autoenroll.AutoEnroller with the shipped defaults",
            "autoenroll_config": cfg.world.autoenroll.model_dump(),
            "streams_per_category": STREAM_REPEATS,
            "individuals_per_stream": STREAM_INDIVIDUALS,
            "days_per_stream": STREAM_DAYS,
            "presence_probability_per_day": STREAM_PRESENCE,
            "sightings_per_individual_per_day": list(STREAM_PER_DAY),
            "images_per_individual_cap": "all for dogs; 6 for products",
            "days_note": "dogs: days are simulated; the dataset's capture sequences are not used",
        },
        "total": total(runs),
        "per_category": {c: total(v) for c, v in per_category.items()},
        "machine": common.cpu_label(),
        "versions": common.versions(),
    }
    print(common.write_result(f"autoenroll_{name}", result))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--n", type=int, default=1500)
    sub.add_parser("dogs")
    sub.add_parser("sop")
    args = ap.parse_args()
    if args.cmd == "detect":
        detect(args)
    elif args.cmd == "dogs":
        retrieval(args, "mpdd_dogs", mpdd_items, ["dog"], "animal")
    else:
        retrieval(args, "sop_household", sop_items, SOP_CATEGORIES, "object")


if __name__ == "__main__":
    main()
