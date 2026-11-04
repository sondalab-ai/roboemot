"""Turn the range-downloaded head of the ACAV100M feature file into a float32 .npy.

NC comparison only.

Usage: python -I acav_subset.py RAW_HEAD OUT.npy
The source file is a plain .npy (128-byte header, '<f2', (5625000, 16, 96), C order), so its first N
rows are the first N * 3072 bytes after the header.
"""

import sys

import numpy as np
from numpy.lib.format import open_memmap

raw = sys.argv[1]
n = (np.memmap(raw, dtype=np.uint8, mode="r").shape[0] - 128) // (16 * 96 * 2)
src = np.memmap(raw, dtype="<f2", mode="r", offset=128, shape=(n, 16, 96))
dst = open_memmap(sys.argv[2], mode="w+", dtype=np.float32, shape=(n, 16, 96))
for i in range(0, n, 50000):
    dst[i : i + 50000] = src[i : i + 50000].astype(np.float32)
dst.flush()
print(n, "rows,", f"{n * 1.28 / 3600:.0f} h")
