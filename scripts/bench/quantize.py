"""INT8 dynamic quantization of the FairFace ViT-B age and gender models (SPEC 5, 13).

Weights only (MatMul/Gemm, per ONNX Runtime defaults); activations are quantized at run time, so
no calibration data is needed. Output goes to `<cache>/models/fairface_int8/` (or
`fairface_quint8/`) next to the FP32 files, never into the repo.

    python scripts/bench/quantize.py                 # QInt8 (default)
    python scripts/bench/quantize.py --type quint8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).parent))
from bench_lib import MODEL_FILES, MODELS  # noqa: E402

TYPES = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}
SOURCES = {"age_model.onnx": "fairface_age_vit", "gender_model.onnx": "fairface_gender_vit"}


def out_dir(kind: str) -> Path:
    """Directory holding the quantized files for one weight type."""
    return MODELS / ("fairface_int8" if kind == "qint8" else f"fairface_{kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=sorted(TYPES), default="qint8")
    args = ap.parse_args()
    dest = out_dir(args.type)
    dest.mkdir(parents=True, exist_ok=True)
    for fname, key in SOURCES.items():
        src = MODEL_FILES[key]
        out = dest / fname
        quantize_dynamic(src, out, weight_type=TYPES[args.type])
        mb_in, mb_out = src.stat().st_size / 1e6, out.stat().st_size / 1e6
        print(f"{src.name}: {mb_in:.1f} MB -> {out} {mb_out:.1f} MB")


if __name__ == "__main__":
    main()
