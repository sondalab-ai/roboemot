"""Augmented FP-validation stream for model selection: mls_dev with RIR + noise, as (T, 96) frames.

Usage: python -I fpval_aug.py WAKEWORD_DIR
Same augmentation as aug_neg_features.py; half of the files augmented, half clean, concatenated.
Writes data/feat/fpval_mls_dev_aug.npy.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:2]
sys.path.insert(0, str(Path(__file__).parent))
import aug_neg_features as A  # noqa: E402


def job(args):
    path, seed = args
    rng = np.random.default_rng(seed)
    x = A.load(path)
    if seed % 2:
        x = A.augment(x, rng)
    pad = np.zeros(8000, np.float32)
    x = np.concatenate([pad, x, pad])
    return A.feats()._get_embeddings((x * 32767).astype(np.int16)).astype(np.float32)


if __name__ == "__main__":
    files = sorted((A.W / "data/wav/mls_dev").glob("*.wav"))
    with Pool(8) as p:
        parts = p.map(job, [(f, i) for i, f in enumerate(files)], chunksize=16)
    e = np.concatenate(parts)
    np.save(A.W / "data/feat/fpval_mls_dev_aug.npy", e)
    print(e.shape)
