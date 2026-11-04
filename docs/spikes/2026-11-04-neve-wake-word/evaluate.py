"""Evaluate a trained openWakeWord classifier (ONNX) for the Neve spike.

Usage: python -I evaluate.py WAKEWORD_DIR MODEL.onnx OUT.json
- Clip sets (out/eval/*): each clip is padded (1 s before, 0.5 s after) and scored in four
  conditions: clean, reverb (simulated RIR, seed unseen in training), reverb + noise at 10 dB and
  at 5 dB SNR (coloured noise or LibriSpeech test-clean babble). A clip fires when any 80 ms window
  scores >= threshold.
- Streams (MLS Italian test, LibriSpeech test-clean, openWakeWord's FP validation features): false
  accepts per hour, an accept = a run of windows >= threshold, runs closer than 2 s merged.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyroomacoustics as pra
from openwakeword.utils import AudioFeatures
from scipy.io import wavfile
from scipy.signal import fftconvolve

W = Path(sys.argv[1])
MODEL = sys.argv[2]
OUT = Path(sys.argv[3])
THRS = [0.5, 0.9, 0.99, 0.995, 0.998, 0.999]
STREAM_SCORES = {}
SR = 16000
rng = np.random.default_rng(99)
F = AudioFeatures(inference_framework="onnx", ncpu=4)
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name


def score_frames(e):
    """Scores for every 16-frame window of an (T, 96) embedding sequence."""
    if len(e) < 16:
        return np.zeros(0)
    win = np.stack([e[i : i + 16] for i in range(len(e) - 15)]).astype(np.float32)
    out = []
    for i in range(0, len(win), 4096):
        out.append(sess.run(None, {INAME: win[i : i + 4096]})[0].reshape(-1))
    return np.concatenate(out)


def eval_rirs(n=40):
    rirs = []
    for _ in range(n):
        dims = rng.uniform([3, 3, 2.4], [8, 7, 3.2])
        e_abs, mo = pra.inverse_sabine(rng.uniform(0.2, 0.7), dims)
        room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(e_abs), max_order=min(mo, 25))
        mic = rng.uniform([0.5, 0.5, 0.7], dims - [0.5, 0.5, 1.2])
        src = mic + rng.uniform(-1.5, 1.5, 3)
        src = np.clip(src, [0.3, 0.3, 1.0], dims - [0.3, 0.3, 0.9])
        room.add_source(src)
        room.add_microphone(mic)
        room.compute_rir()
        h = np.asarray(room.rir[0][0], dtype=np.float32)
        rirs.append(h / np.abs(h).max())
    return rirs


RIRS = eval_rirs()
BABBLE = [
    wavfile.read(f)[1].astype(np.float32) / 32768
    for f in sorted((W / "data/wav/ls_test").glob("*.wav"))[::20]
]


def noise(n):
    if rng.random() < 0.5:
        b = BABBLE[rng.integers(len(BABBLE))]
        b = np.tile(b, int(np.ceil(n / len(b))) + 1)
        s = rng.integers(0, len(b) - n)
        return b[s : s + n]
    x = rng.normal(size=n)
    if rng.random() < 0.5:
        x = np.cumsum(x)
        x -= np.convolve(x, np.ones(400) / 400, mode="same")
    return x


def condition(x, cond):
    if (
        cond == "white10"
    ):  # same as vosk_baseline.py: white noise at 10 dB SNR over the raw clip, no reverb
        x = x + rng.normal(size=len(x)) * np.sqrt(np.mean(x**2) / 10)
        x = np.clip(x, -1, 1)
    x = np.concatenate([np.zeros(SR), x, np.zeros(SR // 2)])
    if cond == "white10":
        return (x * 32767).astype(np.int16)
    if cond != "clean":
        x = fftconvolve(x, RIRS[rng.integers(len(RIRS))])[: len(x)]
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    if cond.startswith("snr"):
        snr = float(cond[3:])
        nz = noise(len(x))
        p_s = np.mean(x[SR : -SR // 2] ** 2)
        nz = nz * np.sqrt(p_s / (np.mean(nz**2) * 10 ** (snr / 10)))
        x = x + nz
        x = x / max(1.0, np.abs(x).max() / 0.9)
    x = x + rng.normal(scale=1e-4, size=len(x))
    return (x * 32767).astype(np.int16)


def clip_set(d):
    files = sorted(d.glob("*.wav"))
    res = {}
    for cond in ("clean", "white10", "reverb", "snr10", "snr5"):
        maxes = []
        for f in files:
            x = wavfile.read(f)[1].astype(np.float32) / 32768
            s = score_frames(F._get_embeddings(condition(x, cond)))
            maxes.append(float(s.max()) if len(s) else 0.0)
        maxes = np.array(maxes)
        res[cond] = {str(t): float((maxes >= t).mean()) for t in THRS}
        res[cond]["n"] = len(files)
        res["_files"] = [f.name for f in files]
        res.setdefault("_max", {})[cond] = maxes.tolist()
    return res


def events(scores, thr, refractory=25):
    """Accept starts: a window >= thr, then nothing for `refractory` windows (25 x 80 ms = 2 s)."""
    starts, last = [], -(10**9)
    for i in np.flatnonzero(scores >= thr):
        if i - last >= refractory:
            starts.append(int(i))
            last = i
    return starts


def stream(name, e, index=None, nev=()):
    s = score_frames(e)
    STREAM_SCORES[name] = s.astype(np.float32)
    hours = len(e) * 0.08 / 3600
    r = {"hours": round(hours, 2)}
    for t in THRS:
        ev = events(s, t)
        r[str(t)] = {"accepts": len(ev), "per_hour": round(len(ev) / hours, 2)}
        if index:
            hit = [fid for fid, a, b in index for p in ev if a <= p < b]
            r[str(t)]["in_nev_utterances"] = sum(1 for h in hit if h in nev)
    print(name, r, flush=True)
    return r


results = {"clips": {}, "streams": {}}
for d in sorted((W / "out/eval").iterdir()):
    results["clips"][d.name] = clip_set(d)
    print(
        d.name,
        {k: v for k, v in results["clips"][d.name].items() if not k.startswith("_")},
        flush=True,
    )
for name in ("mls_test", "ls_test"):
    meta = json.loads((W / f"data/feat/eval_{name}.json").read_text())
    results["streams"][name] = stream(
        name, np.load(W / f"data/feat/eval_{name}.npy"), meta["index"], set(meta["nev"])
    )
results["streams"]["oww_fp_validation"] = stream(
    "oww_fp_validation", np.load(W / "data/oww_features/validation_set_features.npy")
)
OUT.write_text(json.dumps(results, indent=1))
np.savez_compressed(OUT.with_suffix(".streams.npz"), **STREAM_SCORES)
