"""openWakeWord embeddings for folders of 16 kHz WAV speech (negatives).

Usage: python -I neg_features.py WAKEWORD_DIR
- data/feat/neg_train.npy: (N, 16, 96) windows, stride 4 frames, from mls_9h + ls_dev (MLS
  utterances whose transcript has a 'nev*' word are skipped). Used as feature_data_files in
  train.py.
- data/feat/fpval_mls_dev.npy: (T, 96) continuous frames from mls_dev, for train.py's FP validation.
- data/feat/eval_<set>.npy: (T, 96) continuous frames per eval set (mls_test, ls_test), with
  data/feat/eval_<set>.json listing per-file frame offsets and the 'nev*' file ids.
"""

import json
import sys
from pathlib import Path

import numpy as np
from openwakeword.utils import AudioFeatures
from scipy.io import wavfile

W = Path(sys.argv[1])
WAV = W / "data/wav"
OUT = W / "data/feat"
OUT.mkdir(parents=True, exist_ok=True)
F = AudioFeatures(inference_framework="onnx", ncpu=4)


def nev_ids(name):
    p = WAV / f"{name}.neve.json"
    return {r["id"] for r in json.loads(p.read_text())} if p.exists() else set()


def embed_file(path):
    sr, x = wavfile.read(path)
    pad = np.zeros(8000, dtype=np.int16)
    return F._get_embeddings(np.concatenate([pad, x, pad]))


def frames(name, skip=()):
    out, index = [], []
    pos = 0
    for f in sorted((WAV / name).glob("*.wav")):
        if f.stem in skip:
            continue
        e = embed_file(f)
        out.append(e)
        index.append((f.stem, pos, pos + len(e)))
        pos += len(e)
    return np.concatenate(out).astype(np.float32), index


def windows(e, n=16, stride=4):
    return np.stack([e[i : i + n] for i in range(0, len(e) - n, stride)])


train = []
for name in ("mls_9h", "ls_dev"):
    e, _ = frames(name, skip=nev_ids(name))
    train.append(windows(e))
    print(name, e.shape, flush=True)
np.save(OUT / "neg_train.npy", np.concatenate(train))

e, _ = frames("mls_dev", skip=nev_ids("mls_dev"))
np.save(OUT / "fpval_mls_dev.npy", e)
print("fpval", e.shape, flush=True)

for name in ("mls_test", "ls_test"):
    e, idx = frames(name)
    np.save(OUT / f"eval_{name}.npy", e)
    (OUT / f"eval_{name}.json").write_text(json.dumps({"index": idx, "nev": sorted(nev_ids(name))}))
    print(name, e.shape, f"{len(e) * 0.08 / 3600:.2f} h", flush=True)
