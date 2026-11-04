"""Markdown tables from evaluate.py results (one or more models).

Usage: python -I summarize.py TEXTMAP.json NAME=EVAL.json [NAME=EVAL.json ...]
Prints recall / false-fire tables per condition at a few thresholds, streams per hour, and for the
hard-negative sets the texts that fire most (clean and snr10) at threshold 0.5 for each model.
"""

import json
import sys
from collections import defaultdict

with open(sys.argv[1]) as fh:
    tm = json.load(fh)
runs = []
for a in sys.argv[2:]:
    with open(a.split("=")[1]) as fh:
        runs.append((a.split("=")[0], json.load(fh)))
CONDS = ["clean", "white10", "reverb", "snr10", "snr5"]


def pct(v):
    return f"{100 * v:.0f}%"


for name, r in runs:
    thrs = [t for t in r["clips"]["pos_paola"]["clean"] if t != "n"]
    print(f"\n### {name}\n")
    for t in thrs:
        print(f"\nthreshold {t}\n")
        print("| Set | n | " + " | ".join(CONDS) + " |")
        print("| --- | ---: | " + " | ".join("---:" for _ in CONDS) + " |")
        for s, v in r["clips"].items():
            print(f"| {s} | {v['clean']['n']} | " + " | ".join(pct(v[c][t]) for c in CONDS) + " |")
        print("\n| Stream | hours | accepts | per hour |")
        print("| --- | ---: | ---: | ---: |")
        for s, v in r["streams"].items():
            print(f"| {s} | {v['hours']} | {v[t]['accepts']} | {v[t]['per_hour']} |")

for name, r in runs:
    print(f"\n### {name}: texts that fire at 0.5\n")
    for s in ("neg_neve_sentences", "neg_near_words", "neg_en_near"):
        v = r["clips"][s]
        for c in ("clean", "snr10"):
            fires, total = defaultdict(int), defaultdict(int)
            for f, m in zip(v["_files"], v["_max"][c], strict=True):
                text = tm[f"{s}/{f}"]
                total[text] += 1
                fires[text] += m >= 0.5
            top = sorted(total, key=lambda k: -fires[k] / total[k])[:8]
            print(
                f"- {s} {c}: " + "; ".join(f'"{k}" {fires[k]}/{total[k]}' for k in top if fires[k])
            )
    v = r["clips"]["neg_neve_sentences"]
    for c in ("clean", "snr10"):
        fin = [
            m >= 0.5
            for f, m in zip(v["_files"], v["_max"][c], strict=True)
            if tm[f"neg_neve_sentences/{f}"].rstrip(".!?").lower().endswith("neve")
        ]
        mid = [
            m >= 0.5
            for f, m in zip(v["_files"], v["_max"][c], strict=True)
            if not tm[f"neg_neve_sentences/{f}"].rstrip(".!?").lower().endswith("neve")
        ]
        print(
            f"- neve sentences {c}: sentence-final 'neve' fires {sum(fin)}/{len(fin)}, "
            f"other {sum(mid)}/{len(mid)}"
        )
    for s in ("pos_paola", "pos_libritts_heldout"):
        v = r["clips"][s]
        by = defaultdict(list)
        for f, m in zip(v["_files"], v["_max"]["clean"], strict=True):
            by[tm[f"{s}/{f}"]].append(m >= 0.5)
        print(
            f"- {s} clean recall by text: "
            + "; ".join(f'"{k}" {sum(b)}/{len(b)}' for k, b in sorted(by.items()))
        )
