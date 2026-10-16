"""Object detector accuracy and speed on a seeded COCO val2017 subset (SPEC 5, 13, 15).

YOLOX-Nano (416) and NanoDet-Plus-m (320) on the same images, restricted to the household
classes in ``objects_lib.RELEVANT``. Reports AP50 per class (pycocotools, IoU 0.5, 101-point,
maxDets 100) over the 16 classes, and recall and precision at fixed score thresholds
(class-aware greedy matching at IoU 0.5, crowd regions ignored). Images are fetched one by one
from the COCO host into ``datasets/coco/val2017`` and only decoded, never executed.

    python scripts/bench/eval_objects.py --n 1500 --seed 0 --out scripts/bench/results/objects.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent))
from bench_lib import DATASETS  # noqa: E402
from objects_lib import COCO_ANNOTATIONS, RELEVANT, NanodetPlus, YoloxNano, coco_names  # noqa: E402

IMAGE_URL = "http://images.cocodataset.org/val2017/{name}"
THRESHOLDS = (0.3, 0.4, 0.5)
SCORE_FLOOR = 0.01
IOU = 0.5


def fetch_image(name: str) -> np.ndarray:
    """Decode one val2017 image, downloading it into the cache on first use."""
    path = DATASETS / "coco" / "val2017" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(IMAGE_URL.format(name=name), headers={"User-Agent": "rb"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            path.write_bytes(r.read())
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    return img


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4), both xyxy."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_counts(
    dets: list[tuple[np.ndarray, float]], gts: np.ndarray, crowds: np.ndarray, thr: float
) -> tuple[int, int]:
    """(true positives, false positives) of one class in one image at score ``thr``.

    Detections are matched greedily by score to the best free non-crowd box with IoU of at
    least 0.5; one that only overlaps a crowd region is ignored, as in the COCO protocol.
    """
    kept = sorted((d for d in dets if d[1] >= thr), key=lambda d: -d[1])
    taken = np.zeros(len(gts), dtype=bool)
    tp = fp = 0
    for box, _ in kept:
        best = -1
        ious = iou_matrix(box[None], gts)[0] if len(gts) else np.zeros(0)
        for j in np.argsort(-ious):
            if ious[j] < IOU:
                break
            if not taken[j] and not crowds[j]:
                best = int(j)
                break
        if best >= 0:
            taken[best] = True
            tp += 1
        elif len(gts) and (ious[crowds] >= IOU).any():
            continue
        else:
            fp += 1
    return tp, fp


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def ground_truth(coco: COCO, image_id: int, cat_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (xyxy) and crowd flags of one class in one image."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=cat_id))
    boxes = np.array(
        [
            [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
            for a in anns
        ],
        dtype=np.float64,
    ).reshape(-1, 4)
    return boxes, np.array([bool(a["iscrowd"]) for a in anns], dtype=bool)


def evaluate(model, coco: COCO, image_ids: list[int], cat_ids: dict[str, int], labels: list[str]):
    """Run one detector over the images; AP50 per class, TP/FP/GT at each threshold, timing."""
    id_of_label = [cat_ids.get(label) for label in labels]
    results: list[dict] = []
    per_image: dict[int, dict[int, list[tuple[np.ndarray, float]]]] = {}
    times: list[float] = []
    for n, image_id in enumerate(image_ids):
        img = fetch_image(coco.imgs[image_id]["file_name"])
        t0 = time.perf_counter()
        boxes, scores, classes = model.detect(img, SCORE_FLOOR)
        times.append((time.perf_counter() - t0) * 1000)
        for b, s, c in zip(boxes, scores, classes, strict=True):
            cid = id_of_label[c]
            if cid is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in b)
            results.append(
                {
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                }
            )
            per_image.setdefault(image_id, {}).setdefault(cid, []).append((b, float(s)))
        if (n + 1) % 250 == 0:
            print(f"  {model.name}: {n + 1}/{len(image_ids)}", file=sys.stderr, flush=True)

    ev = COCOeval(coco, coco.loadRes(results), "bbox")
    ev.params.imgIds = image_ids
    ev.params.catIds = list(cat_ids.values())
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    prec = ev.eval["precision"][0, :, :, 0, 2]  # IoU 0.5, recall points, class, all areas, 100
    ap50 = {}
    for k, name in enumerate(cat_ids):
        col = prec[:, k]
        ap50[name] = float(col[col > -1].mean()) if (col > -1).any() else float("nan")

    truth = {(i, c): ground_truth(coco, i, c) for i in image_ids for c in cat_ids.values()}
    at_thr: dict[str, dict[str, dict[str, int]]] = {}
    for thr in THRESHOLDS:
        per_class = {}
        for name, cid in cat_ids.items():
            tp = fp = ngt = 0
            for image_id in image_ids:
                gts, crowds = truth[(image_id, cid)]
                ngt += int((~crowds).sum())
                t, f = match_counts(per_image.get(image_id, {}).get(cid, []), gts, crowds, thr)
                tp, fp = tp + t, fp + f
            per_class[name] = {"tp": tp, "fp": fp, "gt": ngt}
        at_thr[str(thr)] = per_class
    return {
        "model": model.name,
        "ap50": ap50,
        "at_threshold": at_thr,
        "detect_ms": {"p50": pct(times, 0.5), "p95": pct(times, 0.95), "n": len(times)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    coco = COCO(str(COCO_ANNOTATIONS))
    ids = sorted(coco.getImgIds())
    rng = np.random.default_rng(args.seed)
    image_ids = sorted(int(i) for i in rng.choice(ids, size=min(args.n, len(ids)), replace=False))
    by_name = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    cat_ids = {name: by_name[name] for name in RELEVANT}
    report = {
        "n_images": len(image_ids),
        "seed": args.seed,
        "classes": list(RELEVANT),
        "models": [],
    }
    for model in (YoloxNano(threads=args.threads), NanodetPlus(threads=args.threads)):
        print(f"{model.name} ...", file=sys.stderr, flush=True)
        report["models"].append(evaluate(model, coco, image_ids, cat_ids, coco_names()))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    for m in report["models"]:
        non_person = [v for k, v in m["ap50"].items() if k != "person"]
        print(m["model"], "mean AP50 (15 non-person):", round(float(np.nanmean(non_person)), 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
