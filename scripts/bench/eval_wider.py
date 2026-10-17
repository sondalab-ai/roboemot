"""Quick face-detection recall check on a WIDER FACE validation subset at the cm4 input size.

Not the official WIDER protocol: every 11th validation image, letterboxed to 320x240 (the cm4
detection input). Eligible ground-truth faces are `invalid=0, blur=0, occlusion=0` and at least
`MIN_H` px tall after the resize ("easy-like"). Recall is IoU >= 0.5 at score >= 0.5. A detection
counts as a false positive when it overlaps no ground-truth face (any kind) at IoU >= 0.3.

    python scripts/bench/eval_wider.py --out scripts/bench/results/wider.json
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_lib import DATASETS, MODEL_FILES, Scrfd, YuNet  # noqa: E402

WIDER = DATASETS / "wider"
SIZE = (320, 240)
MIN_H = 24
STEP = 11


def parse_gt(text: str) -> dict[str, list[list[int]]]:
    lines = text.splitlines()
    out, i = {}, 0
    while i < len(lines):
        name, n = lines[i], int(lines[i + 1])
        rows = [list(map(int, lines[i + 2 + j].split())) for j in range(max(n, 1))]
        out[name] = rows if n else []
        i += 2 + max(n, 1)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with zipfile.ZipFile(WIDER / "wider_face_split.zip") as z:
        gt = parse_gt(z.read("wider_face_split/wider_face_val_bbx_gt.txt").decode())
    names = sorted(gt)[::STEP]
    dets = {
        "yunet": YuNet(MODEL_FILES["yunet"], SIZE),
        "scrfd_500m": Scrfd(MODEL_FILES["scrfd_500m"], SIZE),
        "scrfd_2.5g": Scrfd(MODEL_FILES["scrfd_2.5g"], SIZE),
    }
    hit = dict.fromkeys(dets, 0)
    fp = dict.fromkeys(dets, 0)
    eligible = 0
    used = 0
    with zipfile.ZipFile(WIDER / "WIDER_val.zip") as z:
        for name in names:
            img = cv2.imdecode(np.frombuffer(z.read(f"WIDER_val/images/{name}"), np.uint8), 1)
            h0, w0 = img.shape[:2]
            scale = min(SIZE[0] / w0, SIZE[1] / h0)
            boxes_all, boxes_ok = [], []
            for x, y, w, h, blur, _expr, _ill, invalid, occ, _pose in gt[name]:
                b = np.array([x, y, x + w, y + h], float) * scale
                boxes_all.append(b)
                if invalid == 0 and blur == 0 and occ == 0 and h * scale >= MIN_H:
                    boxes_ok.append(b)
            if not boxes_ok:
                continue
            used += 1
            eligible += len(boxes_ok)
            small = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
            for dn, det in dets.items():
                fs = det.detect(small)
                for b in boxes_ok:
                    hit[dn] += any(iou(f.box, b) >= 0.5 for f in fs)
                fp[dn] += sum(all(iou(f.box, g) < 0.3 for g in boxes_all) for f in fs)
    result = {
        "images_used": used,
        "images_sampled": len(names),
        "eligible_faces": eligible,
        "input": f"{SIZE[0]}x{SIZE[1]} letterbox",
        "recall": {d: hit[d] / eligible for d in dets},
        "false_positives_per_image": {d: fp[d] / used for d in dets},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
