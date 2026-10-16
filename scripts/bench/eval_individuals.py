"""Can a general image embedding tell individual animals / objects apart? (SPEC 5A, 15 spike)

Datasets (see README for licences):
  - MPDD (dogs, whole-body photos, 192 identities, 1657 images, CC BY 4.0)
  - Stanford Online Products test split, household categories (mug, kettle, stapler, toaster,
    coffee maker), a seeded subset of items with >= 4 images (hard negatives: same category)

Reports leave-one-out rank-1 retrieval, same- vs different-individual cosine distributions
(mean, d', EER, thresholds at FAR) and the spread of simulated auto-enrollment clusters
(`world.autoenroll.max_spread`): clusters of k same-individual images vs clusters that mix two
individuals. Spread = max over members of (1 - cosine to the cluster centroid).

    python scripts/bench/eval_individuals.py --out scripts/bench/results/individuals.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from bench_lib import DATASETS, ImageEmbedder, dprime, eer, roc_threshold  # noqa: E402

MODELS = ["dinov2_s", "clip_b32", "mobileclip_s0"]
SOP_CATEGORIES = ["mug", "kettle", "stapler", "toaster", "coffee maker"]
SOP_ITEMS_PER_CATEGORY = 150
SOP_MAX_IMAGES_PER_ITEM = 6
SEED = 0


def load_mpdd() -> tuple[list[np.ndarray], list[str], list[str], list[str]]:
    root = DATASETS / "mpdd/x/MPDD/pytorch"
    imgs, ids, sessions = [], [], []
    for p in sorted(root.glob("*/*.jpg")):
        parts = p.stem.split("_")  # <id>_c<cam>s<seq>_<n>
        imgs.append(cv2.imread(str(p)))
        ids.append(f"mpdd:{parts[0]}")
        sessions.append(parts[1])
    return imgs, ids, ["dog"] * len(ids), sessions


def load_sop() -> tuple[list[np.ndarray], list[str], list[str], list[str]]:
    rng = random.Random(SEED)
    imgs, ids, cats = [], [], []
    for pf in sorted((DATASETS / "sop").glob("test-*.parquet")):
        meta = pq.ParquetFile(pf).read(columns=["item_id", "category"]).to_pydict()
        by_item: dict[str, list[int]] = defaultdict(list)
        for i, (it, c) in enumerate(zip(meta["item_id"], meta["category"], strict=True)):
            if c in SOP_CATEGORIES:
                by_item[(c, it)].append(i)
        want: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
        for (c, it), rows in by_item.items():
            if len(rows) >= 4:
                want[c].append((it, rows))
        picked: list[tuple[str, str, int]] = []
        for c, items in want.items():
            rng.shuffle(items)
            for it, rows in items[:SOP_ITEMS_PER_CATEGORY]:
                for r in sorted(rows)[:SOP_MAX_IMAGES_PER_ITEM]:
                    picked.append((c, it, r))
        if not picked:
            continue
        table = pq.ParquetFile(pf).read(columns=["image"])
        col = table.column("image")
        for c, it, r in picked:
            raw = col[r].as_py()["bytes"]
            imgs.append(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
            ids.append(f"sop:{c}:{it}")
            cats.append(c)
    return imgs, ids, cats, ["-"] * len(ids)


def spread(emb: np.ndarray) -> float:
    c = emb.mean(0)
    c /= np.linalg.norm(c)
    return float((1 - emb @ c).max())


def cluster_spreads(
    e: np.ndarray, ids: np.ndarray, cats: np.ndarray, k: int, n: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Spreads of n pure k-image clusters and n clusters mixing two individuals of one category."""
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, x in enumerate(ids):
        by_id[x].append(i)
    ok = [x for x, v in by_id.items() if len(v) >= k]
    by_cat: dict[str, list[str]] = defaultdict(list)
    for x in ok:
        by_cat[cats[by_id[x][0]]].append(x)
    pure, mixed = [], []
    for _ in range(n):
        x = rng.choice(ok)
        pure.append(spread(e[rng.sample(by_id[x], k)]))
        pool = by_cat[cats[by_id[x][0]]]
        y = rng.choice([p for p in pool if p != x])
        kx = max(1, k - k // 2)
        a = rng.sample(by_id[x], min(kx, len(by_id[x])))
        b = rng.sample(by_id[y], min(k - len(a), len(by_id[y])))
        mixed.append(spread(e[a + b]))
    return np.array(pure), np.array(mixed)


def evaluate(e: np.ndarray, ids: list[str], cats: list[str], sessions: list[str]) -> dict:
    ida, cata, ses = np.array(ids), np.array(cats), np.array(sessions)
    sim = e @ e.T
    n = len(ids)
    np.fill_diagonal(sim, -2)
    same = ida[:, None] == ida[None, :]
    np.fill_diagonal(same, False)
    tri = np.triu(np.ones((n, n), bool), 1)
    has_pos = same.any(1)
    out: dict = {"n_images": n, "n_identities": int(len(set(ids)))}

    def rank1(mask_gallery: np.ndarray) -> float:
        s = np.where(mask_gallery, sim, -2)
        top = s.argmax(1)
        ok = same[np.arange(n), top]
        return float(ok[has_pos].mean())

    out["rank1_all_gallery"] = rank1(np.ones((n, n), bool))
    if len(set(cats)) > 1:
        out["rank1_same_category_gallery"] = rank1(cata[:, None] == cata[None, :])
    if len(set(sessions)) > 1:
        # gallery excludes images from the query's own capture sequence (harder, cross-session)
        sess_ok = ses[:, None] != ses[None, :]
        s = np.where(same & ~sess_ok, -2, sim)  # drop same-sequence positives only
        top = s.argmax(1)
        has_cross = (same & sess_ok).any(1)
        out["rank1_cross_session"] = float(same[np.arange(n), top][has_cross].mean())
        out["cross_session_queries"] = int(has_cross.sum())
    pos = sim[tri & same]
    out["pos_pairs"] = int(len(pos))
    out["pos_mean"] = float(pos.mean())
    out["pos_p05"] = float(np.quantile(pos, 0.05))
    for label, mask in (
        ("any_negatives", ~same),
        ("same_category_negatives", (cata[:, None] == cata[None, :]) & ~same),
    ):
        if label == "same_category_negatives" and len(set(cats)) == 1:
            continue
        neg = sim[tri & mask]
        d: dict = {
            "neg_pairs": int(len(neg)),
            "neg_mean": float(neg.mean()),
            "neg_p99": float(np.quantile(neg, 0.99)),
            "dprime": dprime(pos, neg),
        }
        d["eer"], d["eer_thr"] = eer(pos, neg)
        for far in (1e-2, 1e-3):
            t = roc_threshold(neg, far)
            d[f"thr@far{far:g}"] = t
            d[f"tar@far{far:g}"] = float((pos > t).mean())
        out[label] = d

    rng = random.Random(SEED)
    out["clusters"] = {}
    for k in (3, 5):
        pure, mixed = cluster_spreads(e, ida, cata, k, 3000, rng)
        # best single max_spread: maximise accept(pure) - accept(mixed)
        cand = np.quantile(np.concatenate([pure, mixed]), np.linspace(0.01, 0.99, 197))
        best = max(cand, key=lambda t: (pure <= t).mean() - (mixed <= t).mean())
        out["clusters"][f"k={k}"] = {
            "pure_p50": float(np.quantile(pure, 0.5)),
            "pure_p95": float(np.quantile(pure, 0.95)),
            "pure_p99": float(np.quantile(pure, 0.99)),
            "mixed_p01": float(np.quantile(mixed, 0.01)),
            "mixed_p05": float(np.quantile(mixed, 0.05)),
            "mixed_p50": float(np.quantile(mixed, 0.5)),
            "best_max_spread": float(best),
            "pure_accepted_at_best": float((pure <= best).mean()),
            "mixed_accepted_at_best": float((mixed <= best).mean()),
            "n_clusters": 3000,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    result: dict = {}
    for dname, loader in (("mpdd_dogs", load_mpdd), ("sop_household_objects", load_sop)):
        imgs, ids, cats, sessions = loader()
        if args.limit:
            imgs, ids, cats, sessions = (x[: args.limit] for x in (imgs, ids, cats, sessions))
        print(f"{dname}: {len(imgs)} images, {len(set(ids))} identities", flush=True)
        result[dname] = {}
        for m in MODELS:
            emb = ImageEmbedder(m)
            e = np.stack([emb(im) for im in imgs])
            result[dname][m] = evaluate(e, ids, cats, sessions)
            r = result[dname][m]
            print(f"  {m}: rank1 {r['rank1_all_gallery']:.3f}  pos {r['pos_mean']:.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
