"""LFW face verification (6000 standard pairs, 10 folds) for every detector x embedder combo.

Detects on a 320x320 letterbox, aligns with the detector's own 5 landmarks to the ArcFace 112x112
template, embeds, and reports accuracy, TAR@FAR and the cosine threshold at that FAR.
Impostor thresholds are also taken from all different-identity image pairs (far more negatives
than the 3000 in the standard pairs), which is the seed for `identity.match_threshold`.

    python scripts/bench/eval_lfw.py --out scripts/bench/results/lfw.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_lib import (  # noqa: E402
    ARCFACE_TEMPLATE,
    DATASETS,
    MODEL_FILES,
    ArcEmbedder,
    Scrfd,
    SFace,
    YuNet,
    align,
    dprime,
    eer,
    largest,
    roc_threshold,
)

LFW = DATASETS / "lfw"
DET_SIZE = (320, 320)


def load_pairs() -> tuple[list[tuple[str, str, int]], list[int]]:
    """Return ((path_a, path_b, same), fold_index) for the standard 10x600 pairs."""
    lines = (LFW / "pairs.txt").read_text().splitlines()[1:]
    pairs, folds = [], []
    for i, line in enumerate(lines):
        p = line.split("\t")
        if len(p) == 3:
            a = f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg"
            b = f"{p[0]}/{p[0]}_{int(p[2]):04d}.jpg"
            same = 1
        else:
            a = f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg"
            b = f"{p[2]}/{p[2]}_{int(p[3]):04d}.jpg"
            same = 0
        pairs.append((a, b, same))
        folds.append(i // 600)
    return pairs, folds


def fallback_kps() -> np.ndarray:
    """Centre-crop landmarks used when a detector finds nothing (counted and reported)."""
    return ARCFACE_TEMPLATE * (125 / 112) + 62


def centermost(faces: list, img: np.ndarray):
    """Face whose box centre is nearest the image centre (None if no face)."""
    if not faces:
        return None
    c = np.array([img.shape[1] / 2, img.shape[0] / 2])
    return min(faces, key=lambda f: np.linalg.norm((f.box[:2] + f.box[2:]) / 2 - c))


def cv_accuracy(scores: np.ndarray, same: np.ndarray, folds: np.ndarray) -> tuple[float, float]:
    accs = []
    thrs = np.linspace(-0.2, 0.9, 551)
    for k in range(10):
        tr, te = folds != k, folds == k
        best = max(thrs, key=lambda t: ((scores[tr] > t) == same[tr]).mean())
        accs.append(((scores[te] > best) == same[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--select",
        choices=["largest", "center"],
        default="largest",
        help="which face to keep when several are detected (LFW has one centred subject)",
    )
    ap.add_argument(
        "--embedders", default="mbf,r50,sface", help="comma-separated subset of mbf,r50,sface"
    )
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N pairs per fold")
    args = ap.parse_args()

    pairs, folds_l = load_pairs()
    if args.limit:
        keep = [i for i in range(len(pairs)) if i % 300 < args.limit]
        pairs, folds_l = [pairs[i] for i in keep], [folds_l[i] for i in keep]
    same = np.array([p[2] for p in pairs])
    folds = np.array(folds_l)
    images = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    idx = {n: i for i, n in enumerate(images)}
    ids = np.array([n.split("/")[0] for n in images])
    print(f"{len(pairs)} pairs, {len(images)} unique images", flush=True)

    detectors = {
        "scrfd_500m": Scrfd(MODEL_FILES["scrfd_500m"], DET_SIZE),
        "scrfd_2.5g": Scrfd(MODEL_FILES["scrfd_2.5g"], DET_SIZE),
        "yunet": YuNet(MODEL_FILES["yunet"], DET_SIZE),
    }
    embedders = {
        "mbf": ArcEmbedder(MODEL_FILES["mbf"]),
        "r50": ArcEmbedder(MODEL_FILES["r50"]),
        "sface": SFace(MODEL_FILES["sface"]),
    }

    imgs = {n: cv2.imread(str(LFW / "x/lfw" / n)) for n in images}
    kps_by_det: dict[str, np.ndarray] = {}
    det_fail: dict[str, int] = {}
    multi: dict[str, int] = {}
    det_ms: dict[str, float] = {}
    for dn, det in detectors.items():
        t0 = time.time()
        kps = np.zeros((len(images), 5, 2), np.float32)
        fail = more = 0
        for i, n in enumerate(images):
            faces = det.detect(imgs[n])
            f = largest(faces) if args.select == "largest" else centermost(faces, imgs[n])
            more += len(faces) > 1
            if f is None:
                fail += 1
                kps[i] = fallback_kps()
            else:
                kps[i] = f.kps
        kps_by_det[dn], det_fail[dn], multi[dn] = kps, fail, more
        det_ms[dn] = (time.time() - t0) / len(images) * 1000
        print(f"detector {dn}: no face {fail}/{len(images)}, >1 faces {more}", flush=True)

    results = {}
    for dn in detectors:
        for en, emb in embedders.items():
            if en not in args.embedders.split(","):
                continue
            t0 = time.time()
            e = np.stack([emb(align(imgs[n], kps_by_det[dn][i])) for i, n in enumerate(images)])
            ia = np.array([idx[p[0]] for p in pairs])
            ib = np.array([idx[p[1]] for p in pairs])
            scores = (e[ia] * e[ib]).sum(1)
            pos, neg = scores[same == 1], scores[same == 0]
            acc, acc_std = cv_accuracy(scores, same, folds)
            sim = e @ e.T
            tri = np.triu(np.ones_like(sim, dtype=bool), 1)
            diff = ids[:, None] != ids[None, :]
            wide_neg = sim[tri & diff]
            wide_pos = sim[tri & ~diff]
            row = {
                "n_pairs": int(len(pairs)),
                "n_images": int(len(images)),
                "acc_10fold_mean": acc,
                "acc_10fold_std": acc_std,
                "pos_mean": float(pos.mean()),
                "neg_mean": float(neg.mean()),
                "dprime": dprime(pos, neg),
                "eer": eer(pos, neg)[0],
                "std_neg_n": int(len(neg)),
                "wide_neg_n": int(len(wide_neg)),
                "wide_pos_n": int(len(wide_pos)),
                "det_no_face": det_fail[dn],
            }
            for far in (1e-2, 1e-3):
                t = roc_threshold(neg, far)
                row[f"std_tar@far{far:g}"] = float((pos > t).mean())
                row[f"std_thr@far{far:g}"] = t
            for far in (1e-2, 1e-3, 1e-4):
                t = roc_threshold(wide_neg, far)
                row[f"wide_tar@far{far:g}"] = float((wide_pos > t).mean())
                row[f"wide_thr@far{far:g}"] = t
            results[f"{dn}+{en}"] = row
            print(
                f"{dn}+{en}: acc {acc:.4f} tar@1e-3(std) {row['std_tar@far0.001']:.3f} "
                f"thr(wide,1e-3) {row['wide_thr@far0.001']:.3f}  [{time.time() - t0:.0f}s]",
                flush=True,
            )
    out = {
        "face_selection": args.select,
        "detector_no_face": det_fail,
        "detector_multi_face": multi,
        "detector_ms_per_image_incl_resize": det_ms,
        "pipelines": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
