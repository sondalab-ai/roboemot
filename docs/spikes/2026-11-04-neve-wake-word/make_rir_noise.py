"""Simulated room impulse responses (pyroomacoustics, MIT) and synthetic noise clips.

Usage: python -I make_rir_noise.py WAKEWORD_DIR
Writes data/rir_sim/*.wav (300 RIRs) and data/bg_noise/*.wav (300 x 5 s coloured noise and hum).
No third-party audio data, so no data licence applies.
"""

import sys
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

W = Path(sys.argv[1])
rng = np.random.default_rng(7)
SR = 16000

rd = W / "data/rir_sim"
rd.mkdir(parents=True, exist_ok=True)
for i in range(300):
    dims = rng.uniform([3, 3, 2.4], [8, 7, 3.2])
    rt60 = rng.uniform(0.15, 0.8)
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(e_abs), max_order=min(max_order, 25))
    mic = rng.uniform([0.5, 0.5, 0.7], dims - [0.5, 0.5, 1.2])
    for _ in range(50):
        src = rng.uniform([0.3, 0.3, 1.0], dims - [0.3, 0.3, 0.9])
        if 0.4 < np.linalg.norm(src - mic) < 3.5:
            break
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    h = np.asarray(room.rir[0][0], dtype=np.float32)
    h = h / (np.abs(h).max() + 1e-9)
    wavfile.write(rd / f"rir_{i:03d}.wav", SR, (h * 32767).astype(np.int16))

nd = W / "data/bg_noise"
nd.mkdir(parents=True, exist_ok=True)
n = SR * 5
f = np.fft.rfftfreq(n, 1 / SR)
for i in range(300):
    kind = i % 4
    spec = rng.normal(size=f.size) + 1j * rng.normal(size=f.size)
    if kind == 1:  # pink
        spec[1:] /= np.sqrt(f[1:])
    elif kind == 2:  # brown
        spec[1:] /= f[1:]
    x = np.fft.irfft(spec, n)
    if kind == 3:  # hum + harmonics (fan, motor) over light noise
        t = np.arange(n) / SR
        f0 = rng.uniform(50, 300)
        x = 0.2 * x / np.abs(x).max() + sum(
            rng.uniform(0.1, 1) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 5)
        )
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    wavfile.write(nd / f"noise_{i:03d}.wav", SR, (x * 32767).astype(np.int16))
print("ok")
