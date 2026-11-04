"""CPU cost of the openWakeWord front end (+ optional classifier) per second of audio, ONNX Runtime.

Usage: python -I bench_cpu.py [CLASSIFIER.onnx]
Streams 120 s of noise in 80 ms chunks (1280 samples) as the robot would, 1 intra-op thread.
Reports wall and process CPU time per audio second.
"""

import sys
import time

import numpy as np
import onnxruntime as ort
from openwakeword.utils import AudioFeatures

SECONDS = 120
CHUNK = 1280
rng = np.random.default_rng(0)
audio = (rng.normal(scale=3000, size=16000 * SECONDS)).astype(np.int16)

F = AudioFeatures(inference_framework="onnx", ncpu=1)
clf = None
if len(sys.argv) > 1:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    clf = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
    iname = clf.get_inputs()[0].name

for rep in range(3):
    F.reset()
    t0, c0 = time.perf_counter(), time.process_time()
    per_chunk = []
    for i in range(0, len(audio) - CHUNK, CHUNK):
        a = time.perf_counter()
        F(audio[i : i + CHUNK])
        if clf is not None:
            clf.run(None, {iname: F.get_features(16)})
        per_chunk.append(time.perf_counter() - a)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    pc = np.array(per_chunk) * 1000
    print(
        f"run {rep}: wall {wall / SECONDS * 1000:.2f} ms/s audio, "
        f"process CPU {cpu / SECONDS * 1000:.2f} ms/s, "
        f"chunk p50 {np.median(pc):.3f} ms p99 {np.percentile(pc, 99):.3f} ms"
    )
