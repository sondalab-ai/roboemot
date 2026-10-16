"""FP32 vs INT8 FairFace ViT-B age and gender on the FairFace validation set (SPEC 5, 13).

Same preprocessing, group mapping and 0.8 confidence rule as `eval_fairface.py`, ViT pair only.
Runs one precision per invocation on a fixed random subset (`--n`, `--seed`; the same indices for
every precision) and reports per race group, with and without the 0.8 rule. Per-sample
probabilities are saved next to the JSON so two runs can be compared sample by sample.

    python scripts/bench/eval_fairface_int8.py --precision fp32 --n 1500 --out results/ff_fp32.json
    python scripts/bench/eval_fairface_int8.py --precision int8 --n 1500 --out results/ff_int8.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from bench_lib import MODEL_FILES, MODELS, session  # noqa: E402
from eval_fairface import (  # noqa: E402
    BIN_TO_GROUP,
    CONF,
    PARQUET,
    RACES,
    softmax,
    vit_input,
)

DIRS = {"fp32": None, "int8": "fairface_int8", "quint8": "fairface_quint8"}


def model_path(precision: str, key: str) -> Path:
    """FP32 file from the standard cache layout, quantized ones from `fairface_<type>/`."""
    d = DIRS[precision]
    return MODEL_FILES[key] if d is None else MODELS / d / MODEL_FILES[key].name


def pick(n_total: int, n: int, seed: int) -> np.ndarray:
    """Sorted indices of a seeded random subset (all rows when n is 0 or >= n_total)."""
    if not n or n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def metrics(a_true, a_bin, g_true, ga_pred, gg_pred, rule: bool) -> dict:
    """Error rates for one subset; with `rule`, predictions under CONF count as unknown."""
    a_pred, g_pred = ga_pred.copy(), gg_pred.copy()
    if rule:
        a_pred = np.where(ga_pred[1] >= CONF, ga_pred[0], -1)
        g_pred = np.where(gg_pred[1] >= CONF, gg_pred[0], -1)
    else:
        a_pred, g_pred = ga_pred[0], gg_pred[0]
    a_ok, g_ok = a_pred >= 0, g_pred >= 0
    minors = a_bin <= 2
    return {
        "n": int(len(a_true)),
        "minor_n": int(minors.sum()),
        "age_group_error": float((a_pred != a_true)[a_ok].mean()) if a_ok.any() else None,
        "age_unknown": float(1 - a_ok.mean()),
        "gender_error": float((g_pred != g_true)[g_ok].mean()) if g_ok.any() else None,
        "gender_unknown": float(1 - g_ok.mean()),
        "minor_as_adult_or_senior": float(np.isin(a_pred[minors], (2, 3)).mean())
        if minors.any()
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=sorted(DIRS), required=True)
    ap.add_argument("--n", type=int, default=0, help="subset size (0 = all rows)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tb = pq.ParquetFile(PARQUET).read().to_pydict()
    idx = pick(len(tb["age"]), args.n, args.seed)
    print(f"FairFace validation, {args.precision}: n={len(idx)} of {len(tb['age'])}", flush=True)
    vit_age = session(model_path(args.precision, "fairface_age_vit"))
    vit_gen = session(model_path(args.precision, "fairface_gender_vit"))
    ain, gin = vit_age.get_inputs()[0].name, vit_gen.get_inputs()[0].name

    pa_all = np.zeros((len(idx), 9), np.float32)
    pg_all = np.zeros((len(idx), 2), np.float32)
    t0 = time.time()
    for k, i in enumerate(idx):
        bgr = cv2.imdecode(np.frombuffer(tb["image"][i]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        x = vit_input(bgr)
        pa_all[k] = softmax(vit_age.run(None, {ain: x})[0][0])
        pg_all[k] = softmax(vit_gen.run(None, {gin: x})[0][0])
        if k % 250 == 0:
            print(k, f"{time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0

    a_bin = np.array([tb["age"][i] for i in idx])
    a_true = np.array([BIN_TO_GROUP[b] for b in a_bin])
    g_true = np.array([tb["gender"][i] for i in idx])  # FairFace: 0 Male, 1 Female
    race = np.array([tb["race"][i] for i in idx])
    gp = np.stack(
        [pa_all[:, 0] + pa_all[:, 1], pa_all[:, 2], pa_all[:, 3:7].sum(1), pa_all[:, 7:].sum(1)], 1
    )
    ga_pred = np.stack([gp.argmax(1), gp.max(1)])
    gg_pred = np.stack([np.where(pg_all.argmax(1) == 1, 0, 1), pg_all.max(1)])  # 0 F, 1 M -> FF

    result: dict = {
        "precision": args.precision,
        "n": len(idx),
        "seed": args.seed,
        "elapsed_s": round(elapsed, 1),
        "confidence_rule": CONF,
    }
    for rule in (True, False):
        key = "with_rule" if rule else "no_rule"
        result[key] = {"all": metrics(a_true, a_bin, g_true, ga_pred, gg_pred, rule)}
        for r, name in enumerate(RACES):
            m = race == r
            sub = (a_true[m], a_bin[m], g_true[m], ga_pred[:, m], gg_pred[:, m])
            result[key][name] = metrics(*sub, rule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    np.savez(args.out.with_suffix(".npz"), idx=idx, pa=pa_all, pg=pg_all)
    print(json.dumps(result["with_rule"]["all"], indent=1))
    print(json.dumps(result["no_rule"]["all"], indent=1))


if __name__ == "__main__":
    main()
