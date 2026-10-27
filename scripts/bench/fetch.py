"""Download helper for the M1 model spike (SPEC 5, 13, 15).

Downloads are untrusted data: each file lands in its own directory under the cache and is
only ever loaded as ONNX weights or image data, never executed. Prints a sha256 per file so
the README can pin them.

    python scripts/bench/fetch.py --list
    python scripts/bench/fetch.py yunet sface insightface-buffalo
    python scripts/bench/fetch.py lfw mpdd fairface-val sop-test      # datasets
    python scripts/bench/fetch.py yolox-nano nanodet-plus coco-annotations coco-samples
    python scripts/bench/fetch.py --export-flat ./models              # for run_cm4.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

CACHE = Path.home() / ".cache" / "roboemot-bench"
GH = "https://" + "git" + "hub.com"
HF = "https://huggingface.co"

# name -> (subdir, filename, url)
FILES: dict[str, list[tuple[str, str, str]]] = {
    "yunet": [
        (
            "opencv_zoo",
            "face_detection_yunet_2023mar.onnx",
            f"{GH}/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx",
        )
    ],
    "sface": [
        (
            "opencv_zoo",
            "face_recognition_sface_2021dec.onnx",
            f"{GH}/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
            "face_recognition_sface_2021dec.onnx",
        )
    ],
    "scrfd-2.5g": [
        (
            "insightface",
            "scrfd_2.5g_bnkps.onnx",
            f"{HF}/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx",
        )
    ],
    "mobileclip-s0": [
        (
            "mobileclip_s0",
            "vision_model.onnx",
            f"{HF}/Xenova/mobileclip_s0/resolve/main/onnx/vision_model.onnx",
        ),
        ("mobileclip_s0", "LICENSE", f"{HF}/Xenova/mobileclip_s0/resolve/main/LICENSE"),
    ],
    "dinov2-small": [
        ("dinov2_small", "model.onnx", f"{HF}/Xenova/dinov2-small/resolve/main/onnx/model.onnx"),
    ],
    "clip-b32": [
        (
            "clip_b32",
            "vision_model.onnx",
            f"{HF}/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
        ),
    ],
    "fairface-vit": [
        (
            "fairface",
            "age_model.onnx",
            f"{HF}/onnx-community/fairface_age_image_detection-ONNX/resolve/main/onnx/model.onnx",
        ),
        (
            "fairface",
            "gender_model.onnx",
            f"{HF}/onnx-community/fairface_gender_image_detection-ONNX/resolve/main/onnx/model.onnx",
        ),
    ],
    "yolo11n-face-age": [
        (
            "yolo11n_face_age",
            "model.onnx",
            f"{HF}/AdamCodd/yolo11n-face-age/resolve/410bdd84fa762b6aa550ef73bd57865a2d30be37/model.onnx",
        ),
    ],
    "yolox-nano": [
        (
            "yolox_nano",
            "yolox_nano.onnx",
            f"{GH}/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        )
    ],
    "nanodet-plus": [
        (
            "nanodet_plus",
            "nanodet-plus-m_320.onnx",
            f"{GH}/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx",
        ),
        (
            "nanodet_plus",
            "nanodet-plus-m_320.yml",
            f"{GH}/RangiLyu/nanodet/raw/v1.0.0-alpha-1/config/nanodet-plus-m_320.yml",
        ),
    ],
    "insightface-buffalo": [
        ("insightface", f"{n}.zip", f"{GH}/deepinsight/insightface/releases/download/v0.7/{n}.zip")
        for n in ("buffalo_l", "buffalo_s", "buffalo_sc")
    ],
}


FIGSHARE = "https://ndownloader.figshare.com/files"
MENDELEY = "https://data.mendeley.com/public-files/datasets/v5j6m8dzhv/files"
SOP_BASE = f"{HF}/datasets/nyris/stanford-online-products-v1/resolve/main/data"

# dataset name -> [(subdir under datasets/, filename, url)]
DATA: dict[str, list[tuple[str, str, str]]] = {
    "coco-annotations": [
        (
            "coco",
            "annotations_trainval2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        )
    ],
    "lfw": [
        ("lfw", "lfw.tgz", f"{FIGSHARE}/5976018"),
        ("lfw", "pairs.txt", f"{FIGSHARE}/5976006"),
    ],
    "coco-samples": [
        (
            "coco/val2017",
            f"{n}.jpg",
            f"http://images.cocodataset.org/val2017/{n}.jpg",
        )
        for n in ("000000078426", "000000002592")
    ],
    "fairface-val": [
        (
            "fairface/0.25",
            "validation-00000-of-00001-951dbd63c8724ee1.parquet",
            f"{HF}/datasets/HuggingFaceM4/FairFace/resolve/main/0.25/"
            "validation-00000-of-00001-951dbd63c8724ee1.parquet",
        )
    ],
    "mpdd": [
        (
            "mpdd",
            "MPDD.zip",
            f"{MENDELEY}/05d1d583-faf6-410d-89a5-a6b1134b6e5e/file_downloaded",
        )
    ],
    "sop-test": [
        ("sop", f"test-0000{i}.parquet", f"{SOP_BASE}/test-0000{i}-of-00004.parquet")
        for i in range(4)
    ],
}

# flat names expected by run_cm4.py -> path under models/
FLAT = {
    "face_detection_yunet_2023mar.onnx": "opencv_zoo/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "opencv_zoo/face_recognition_sface_2021dec.onnx",
    "det_500m.onnx": "insightface/buffalo_s/det_500m.onnx",
    "scrfd_2.5g_bnkps.onnx": "insightface/scrfd_2.5g_bnkps.onnx",
    "w600k_mbf.onnx": "insightface/buffalo_s/w600k_mbf.onnx",
    "w600k_r50.onnx": "insightface/buffalo_l/w600k_r50.onnx",
    "genderage.onnx": "insightface/buffalo_l/genderage.onnx",
    "fairface_age_vit.onnx": "fairface/age_model.onnx",
    "fairface_gender_vit.onnx": "fairface/gender_model.onnx",
    "fairface_age_vit_int8.onnx": "fairface_int8/age_model.onnx",  # from quantize.py
    "fairface_gender_vit_int8.onnx": "fairface_int8/gender_model.onnx",
    "yolo11n_face_age.onnx": "yolo11n_face_age/model.onnx",
    "dinov2_s.onnx": "dinov2_small/model.onnx",
    "clip_b32_vision.onnx": "clip_b32/vision_model.onnx",
    "mobileclip_s0_vision.onnx": "mobileclip_s0/vision_model.onnx",
    "yolox_nano.onnx": "yolox_nano/yolox_nano.onnx",
    "nanodet_plus_m_320.onnx": "nanodet_plus/nanodet-plus-m_320.onnx",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "roboemot-bench"})
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as out:  # noqa: S310
        while chunk := r.read(1 << 20):
            out.write(chunk)
    part.rename(dest)


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract only flat *.onnx members (no path traversal, nothing else)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name.endswith(".onnx"):
                continue
            (out_dir / name).write_bytes(z.read(info))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--export-flat", type=Path, help="copy models to DIR with run_cm4.py names")
    args = ap.parse_args()
    if args.export_flat:
        args.export_flat.mkdir(parents=True, exist_ok=True)
        for flat, rel in FLAT.items():
            src = args.cache / "models" / rel
            if src.exists():
                shutil.copy2(src, args.export_flat / flat)
                print(f"{sha256(src)}  {flat}")
        return 0
    if args.list or not args.names:
        print("models:  ", " ".join(FILES))
        print("datasets:", " ".join(DATA))
        return 0
    for name in args.names:
        base = "models" if name in FILES else "datasets"
        for sub, fname, url in FILES.get(name) or DATA[name]:
            dest = args.cache / base / sub / fname
            fetch(url, dest)
            print(f"{sha256(dest)}  {dest}  <- {url}")
            if base == "models" and fname.endswith(".zip"):
                safe_extract(dest, args.cache / "models" / sub / dest.stem)
            if fname == "lfw.tgz":
                with tarfile.open(dest) as t:
                    t.extractall(dest.parent / "x", filter="data")
            if fname == "annotations_trainval2017.zip":
                with zipfile.ZipFile(dest) as z:
                    (dest.parent / "instances_val2017.json").write_bytes(
                        z.read("annotations/instances_val2017.json")
                    )
            if fname == "MPDD.zip":
                with zipfile.ZipFile(dest) as z:
                    for m in z.namelist():
                        if m.lower().endswith(".jpg"):
                            z.extract(m, dest.parent / "x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
