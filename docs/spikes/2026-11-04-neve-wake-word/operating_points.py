"""One table per model: recall, false fires and false accepts per hour at several thresholds.

Usage: python -I operating_points.py NAME=EVAL.json [NAME=EVAL.json ...]
Reads the per-clip maxima in EVAL.json and the stream scores in EVAL.streams.npz (evaluate.py).
Accepts on streams: a window >= threshold, then a 2 s refractory time.
"""

import json
import sys
from pathlib import Path

import numpy as np

THRS = [0.5, 0.9, 0.99, 0.995, 0.998, 0.999]
CLIPS = [
    ("pos_paola", "clean"),
    ("pos_paola", "snr10"),
    ("pos_serena_medium", "clean"),
    ("pos_libritts_heldout", "clean"),
    ("pos_libritts_heldout", "snr10"),
    ("neg_neve_sentences", "clean"),
    ("neg_neve_sentences", "snr10"),
    ("neg_near_words", "clean"),
]
STREAMS = ["mls_test", "ls_test", "oww_fp_validation"]


def accepts(s, t, refractory=25):
    n, last = 0, -(10**9)
    for i in np.flatnonzero(s >= t):
        if i - last >= refractory:
            n, last = n + 1, i
    return n


for arg in sys.argv[1:]:
    name, path = arg.split("=")
    r = json.loads(Path(path).read_text())
    sc = np.load(Path(path).with_suffix(".streams.npz"))
    hours = {k: r["streams"][k]["hours"] for k in STREAMS}
    head = [f"{s} {c}" for s, c in CLIPS] + [f"{k} FA/h" for k in STREAMS]
    print(f"\n{name}\n\n| threshold | " + " | ".join(head) + " |")
    print("| ---: | " + " | ".join("---:" for _ in head) + " |")
    for t in THRS:
        cells = [f"{100 * np.mean(np.array(r['clips'][s]['_max'][c]) >= t):.0f}%" for s, c in CLIPS]
        cells += [f"{accepts(sc[k], t) / hours[k]:.1f}" for k in STREAMS]
        print(f"| {t} | " + " | ".join(cells) + " |")
    print("\nmax stream score:", {k: float(sc[k].max()) for k in STREAMS})
