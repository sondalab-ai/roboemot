"""Age group / gender error per race group on the FairFace validation set (SPEC 5, 14).

Candidates: InsightFace genderage.onnx (on a SCRFD-500M detection) and the Apache-2.0 FairFace-
trained ViT-B age and gender classifiers (dima806, ONNX export by onnx-community; whole image).
Age is mapped to the SPEC groups child (<13), teen (13-19), adult (20-59), senior (60+); the
FairFace bin 10-19 cannot be split at 13, so it is scored as teen and `minor` (<20) is reported
separately because that is the safety-relevant direction (SPEC 5: unknown counts as a minor).

    python scripts/bench/eval_fairface.py --out scripts/bench/results/fairface.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from bench_lib import DATASETS, MODEL_FILES, Scrfd, largest, session  # noqa: E402

PARQUET = next((DATASETS / "fairface/0.25").glob("validation-*.parquet"))
RACES = [
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Southeast Asian",
]
GENDERS = ["Male", "Female"]  # FairFace label order
GROUPS = ["child", "teen", "adult", "senior"]
BIN_TO_GROUP = [0, 0, 1, 2, 2, 2, 2, 3, 3]  # 0-2,3-9,10-19,20-29,...,60-69,70+
CONF = 0.8


def years_to_group(y: float) -> int:
    return 0 if y < 13 else 1 if y < 20 else 2 if y < 60 else 3


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def genderage_crop(bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """InsightFace Attribute preprocessing: centre on the box, scale so the face spans ~64/96."""
    w, h = box[2] - box[0], box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = 96 / (max(w, h) * 1.5)
    m = np.array([[s, 0, 48 - cx * s], [0, s, 48 - cy * s]], np.float32)
    return cv2.warpAffine(bgr, m, (96, 96), borderValue=0)


def vit_input(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(
        cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB
    )
    return ((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]


def summarize(rows: list[dict], key_fn, keys) -> dict:
    out = {}
    for k in keys:
        sel = [r for r in rows if key_fn(r) == k]
        out[str(k)] = stats(sel)
    out["all"] = stats(rows)
    return out


def stats(sel: list[dict]) -> dict:
    n = len(sel)
    if not n:
        return {"n": 0}
    g_known = [r for r in sel if r["g_pred"] is not None]
    a_known = [r for r in sel if r["a_pred"] is not None]
    minors = [r for r in sel if r["a_true_bin"] <= 2]
    adults = [r for r in sel if r["a_true"] >= 2]
    return {
        "n": n,
        "no_face": sum(r["no_face"] for r in sel) / n,
        "gender_unknown_rate": 1 - len(g_known) / n,
        "gender_error_of_confident": float(np.mean([r["g_pred"] != r["g_true"] for r in g_known]))
        if g_known
        else None,
        "age_unknown_rate": 1 - len(a_known) / n,
        "age_group_error_of_confident": float(
            np.mean([r["a_pred"] != r["a_true"] for r in a_known])
        )
        if a_known
        else None,
        "minor_n": len(minors),
        "minor_as_adult_or_senior_of_all_minors": float(
            np.mean([r["a_pred"] in (2, 3) for r in minors])
        )
        if minors
        else None,
        "adult_n": len(adults),
        "adult_or_senior_as_child_or_teen": float(
            np.mean([r["a_pred"] in (0, 1) for r in adults if r["a_pred"] is not None])
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tb = pq.ParquetFile(PARQUET).read().to_pydict()
    n = len(tb["age"]) if not args.limit else args.limit
    print(f"FairFace validation (padding 0.25): n={n}", flush=True)
    det = Scrfd(MODEL_FILES["scrfd_500m"], (320, 320))
    ga = session(MODEL_FILES["genderage"])
    vit_age = session(MODEL_FILES["fairface_age_vit"])
    vit_gen = session(MODEL_FILES["fairface_gender_vit"])

    rows = {"genderage": [], "vit": []}
    for i in range(n):
        bgr = cv2.imdecode(np.frombuffer(tb["image"][i]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        age_bin, gender, race = tb["age"][i], tb["gender"][i], tb["race"][i]
        base = {
            "a_true": BIN_TO_GROUP[age_bin],
            "a_true_bin": age_bin,
            "g_true": gender,
            "race": race,
        }
        # genderage on a detection
        f = largest(det.detect(bgr))
        r = dict(base, no_face=f is None, g_pred=None, a_pred=None)
        if f is not None:
            blob = genderage_crop(bgr, f.box).astype(np.float32).transpose(2, 0, 1)[None][:, ::-1]
            out = ga.run(None, {ga.get_inputs()[0].name: np.ascontiguousarray(blob)})[0][0]
            p = softmax(out[:2])  # index 0 = female, 1 = male (InsightFace)
            g_male = int(p.argmax() == 1)
            if p.max() >= CONF:
                r["g_pred"] = 0 if g_male else 1  # FairFace: 0 Male, 1 Female
            r["a_pred"] = years_to_group(float(out[2]) * 100)  # regression: no confidence
        rows["genderage"].append(r)
        # ViT classifiers on the whole image
        x = vit_input(bgr)
        pa = softmax(vit_age.run(None, {vit_age.get_inputs()[0].name: x})[0][0])
        pg = softmax(vit_gen.run(None, {vit_gen.get_inputs()[0].name: x})[0][0])  # 0 F, 1 M
        gp = np.array([pa[0] + pa[1], pa[2], pa[3:7].sum(), pa[7] + pa[8]])
        v = dict(base, no_face=False, g_pred=None, a_pred=None)
        if pg.max() >= CONF:
            v["g_pred"] = 0 if pg.argmax() == 1 else 1
        if gp.max() >= CONF:
            v["a_pred"] = int(gp.argmax())
        v["a_pred_nothreshold"] = int(gp.argmax())
        rows["vit"].append(v)
        if i % 1000 == 0:
            print(i, flush=True)

    result: dict = {"n": n, "confidence_rule": CONF}
    for name, rs in rows.items():
        result[name] = {
            "by_race": summarize(rs, lambda r: RACES[r["race"]], RACES),
            "by_gender": summarize(rs, lambda r: GENDERS[r["g_true"]], GENDERS),
        }
    # ViT age without the 0.8 rule, for reference
    rs = rows["vit"]
    result["vit_age_argmax_no_threshold"] = {
        "age_group_error": float(np.mean([r["a_pred_nothreshold"] != r["a_true"] for r in rs])),
        "minor_as_adult_or_senior": float(
            np.mean([r["a_pred_nothreshold"] in (2, 3) for r in rs if r["a_true_bin"] <= 2])
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["genderage"]["by_race"]["all"], indent=1))
    print(json.dumps(result["vit"]["by_race"]["all"], indent=1))


if __name__ == "__main__":
    main()
