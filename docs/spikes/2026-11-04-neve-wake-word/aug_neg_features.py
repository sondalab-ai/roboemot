"""Round 2 negatives: augmented speech (RIR + noise) and pure noise, as openWakeWord windows.

Usage: python -I aug_neg_features.py WAKEWORD_DIR SET [SET ...]
For each SET folder under data/wav/ (16 kHz WAV), every file gets one augmented copy: a simulated
RIR (p 0.6) and coloured noise or hum from data/bg_noise (p 0.8) at 0 to 20 dB SNR, gain 0.1 to 1.
Also 900 pure-noise clips (bg_noise with RIR and gain). Writes data/feat/neg_aug_<SETS>.npy with
(N, 16, 96) windows at stride 4 frames. The clean copies of the same sets are in neg_train.npy
(mls_9h, ls_dev) or written here as data/feat/neg_clean_<SET>.npy for new sets (--clean).
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

W = Path(sys.argv[1])
SETS = [a for a in sys.argv[2:] if not a.startswith("--")]
CLEAN = "--clean" in sys.argv
RIRS = sorted((W / "data/rir_sim").glob("*.wav"))
NOISE = sorted((W / "data/bg_noise").glob("*.wav"))
_F = None


def feats():
    global _F
    if _F is None:
        from openwakeword.utils import AudioFeatures

        _F = AudioFeatures(inference_framework="onnx", ncpu=1)
    return _F


def load(p):
    return wavfile.read(p)[1].astype(np.float32) / 32768


def augment(x, rng):
    if rng.random() < 0.6:
        x = fftconvolve(x, load(RIRS[rng.integers(len(RIRS))]))[: len(x)]
    x = x / (np.abs(x).max() + 1e-9)
    if rng.random() < 0.8:
        n = load(NOISE[rng.integers(len(NOISE))])
        n = np.tile(n, int(np.ceil(len(x) / len(n))))[: len(x)]
        snr = rng.uniform(0, 20)
        n = n * np.sqrt(np.mean(x**2) / (np.mean(n**2) * 10 ** (snr / 10) + 1e-12))
        x = x + n
    return x / (np.abs(x).max() + 1e-9) * rng.uniform(0.1, 1.0)


def windows(e, n=16, stride=4):
    if len(e) <= n:
        return np.zeros((0, n, 96), np.float32)
    return np.stack([e[i : i + n] for i in range(0, len(e) - n, stride)]).astype(np.float32)


def job(args):
    path, seed, mode = args
    rng = np.random.default_rng(seed)
    if mode == "noise":
        x = load(path)
        x = fftconvolve(x, load(RIRS[rng.integers(len(RIRS))]))[: len(x)]
        x = x / (np.abs(x).max() + 1e-9) * rng.uniform(0.02, 1.0)
    else:
        x = load(path)
        if mode == "aug":
            x = augment(x, rng)
    pad = np.zeros(8000, np.float32)
    x = np.concatenate([pad, x, pad])
    return windows(feats()._get_embeddings((x * 32767).astype(np.int16)))


def run(jobs):
    with Pool(8) as p:
        parts = p.map(job, jobs, chunksize=16)
    return np.concatenate([q for q in parts if len(q)])


def main():
    files = [f for s in SETS for f in sorted((W / "data/wav" / s).glob("*.wav"))]
    jobs = [(f, i, "aug") for i, f in enumerate(files)]
    jobs += [(NOISE[i % len(NOISE)], 10**6 + i, "noise") for i in range(900)]
    out = run(jobs)
    tag = "_".join(SETS)
    np.save(W / f"data/feat/neg_aug_{tag}.npy", out)
    print("aug", tag, out.shape, f"{out.shape[0] * 4 * 0.08 / 3600:.1f} h of windows", flush=True)
    if CLEAN:
        for s in SETS:
            c = run([(f, 0, "clean") for f in sorted((W / "data/wav" / s).glob("*.wav"))])
            np.save(W / f"data/feat/neg_clean_{s}.npy", c)
            print("clean", s, c.shape, flush=True)


if __name__ == "__main__":
    main()
