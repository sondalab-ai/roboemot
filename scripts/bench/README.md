# M1 model spike: accuracy and CPU latency bench

Measurement scripts behind `docs/spikes/2026-10-m1-models.md` (SPEC 5, 5A, 6, 13, 15).
Nothing here ships in the library. Datasets and weights live outside the repo, in
`~/.cache/roboemot-bench/` (override with `ROBOEMOT_BENCH_CACHE`); never commit them. Downloaded
files are untrusted: scripts only load ONNX weights and image data from them.

## Set up (Mac or Linux, Python 3.11 to 3.13)

```bash
python3 -m venv .venv-bench && .venv-bench/bin/pip install -r scripts/bench/requirements-eval.txt
PY=.venv-bench/bin/python
$PY -I scripts/bench/fetch.py yunet sface insightface-buffalo scrfd-2.5g fairface-vit \
    dinov2-small clip-b32 mobileclip-s0                    # model weights, prints sha256
$PY -I scripts/bench/fetch.py lfw fairface-val mpdd sop-test   # datasets
```

`fetch.py --list` shows the names. Weights come from the URLs in `fetch.py` (OpenCV Zoo, the
InsightFace v0.7 release zips, Hugging Face). WIDER FACE (optional, detection recall):
`WIDER_val.zip` and `wider_face_split.zip` from
`https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/main/data` into
`~/.cache/roboemot-bench/datasets/wider/`.

SHA-256 of the files used (as downloaded on 2026-10-16):

| File | sha256 |
| --- | --- |
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| `det_500m.onnx` (buffalo_s) | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` |
| `scrfd_2.5g_bnkps.onnx` (HF mirror) | `bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce` |
| `w600k_mbf.onnx` (buffalo_s) | `9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f` |
| `w600k_r50.onnx` (buffalo_l) | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| `genderage.onnx` (buffalo_l) | `4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb` |
| `fairface_age_vit.onnx` | `7f28fc7e890b4c21bf48f65d9985cefaea3430c28df1391e4f91293c0d89555a` |
| `fairface_gender_vit.onnx` | `f9a361ef12ea495caf10c7ca1aff24b5fad21a4bd77f0e363597129779be836d` |
| `dinov2_s.onnx` | `83141175ec78b4ff9a2bb58a4c7c264ba0054d1c2e122e5a8114b79a8d4179ea` |
| `clip_b32_vision.onnx` | `fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40` |
| `mobileclip_s0_vision.onnx` | `17d3c037b1d488c10c50e09f6009ea5a198caef4e0e8f4ea5617b7cb2d067ac0` |
| `buffalo_l.zip` | `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` |
| `buffalo_s.zip` | `d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e` |
| `yolox_nano.onnx` (YOLOX release 0.1.1rc0) | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |
| `nanodet-plus-m_320.onnx` (NanoDet v1.0.0-alpha-1) | `4f12723cce3d48e47ca92cb925ba74d97a965c069208edca660bbb9f7ce2c610` |
| `annotations_trainval2017.zip` | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` (md5 matches the S3 ETag) |
| `MPDD.zip` | `6c800c1b4aa67629544dec7444dee85bc57781abde1ae1d077ea9ef1804284cd` (matches Mendeley) |

No upstream checksums are published for most of these; they pin what was measured.

## Accuracy runs (one at a time; they are CPU heavy, ORT is limited to 2 threads)

```bash
nice -n 10 $PY -I scripts/bench/eval_lfw.py         --out scripts/bench/results/lfw.json
nice -n 10 $PY -I scripts/bench/eval_fairface.py    --out scripts/bench/results/fairface.json
nice -n 10 $PY -I scripts/bench/eval_individuals.py --out scripts/bench/results/individuals.json
nice -n 10 $PY -I scripts/bench/eval_wider.py       --out scripts/bench/results/wider.json
```

`results/*.json` hold the numbers quoted in the spike document. `bench_lib.py` has the
detectors, the 5-point ArcFace alignment and the embedders shared by the scripts.

## INT8 FairFace (SPEC 5, 13)

Dynamic INT8 quantization of the FairFace ViT-B pair, accuracy versus FP32 on a fixed seeded
subset, and speed. Needs `onnx` for the quantizer (`.venv-bench/bin/pip install onnx`). Run one
step at a time; `nice` and 2 threads keep a shared machine usable.

```bash
export OMP_NUM_THREADS=2
nice -n 10 $PY -I scripts/bench/quantize.py             # -> ~/.cache/roboemot-bench/models/fairface_int8/
for p in fp32 int8; do
  nice -n 10 $PY -I scripts/bench/eval_fairface_int8.py --precision $p --n 1400 --seed 0 \
      --out scripts/bench/results/fairface_int8_spike_$p.json
done                                                    # ~10 min fp32, ~3 min int8 on an M3 Pro
$PY -I scripts/bench/fetch.py --export-flat ./models    # also exports the INT8 files
nice -n 10 $PY -I scripts/bench/run_cm4.py --models-dir ./models --threads 2 --inter-threads 2 \
    --only fairface_age_vit_224,fairface_age_vit_int8_224,fairface_pair_fp32,fairface_pair_int8
```

`--n 0` evaluates all 10,954 images (about 70 min FP32). The `fairface_pair_*` entries load age
and gender in one process and time both per call. To bench INT8 on the robot, copy the two
`fairface_*_vit_int8.onnx` files into the `models` dir next to the FP32 ones; the commands in the
next section pick them up automatically, or use `--only` as above and `--threads 1` / `--threads 2`.

## Age model yolo11n-face-age (SPEC 5, 13)

The shipped age model since 2026-10-22 (CC-BY-NC-4.0, revision `410bdd84`, sha256
`e37945afcbc1b855f4a66bbf1e13db9d19d5ca2e8ddb2ffe2be2906d53f5ff93`). The library runs it with one
thread on a letterboxed crop of `vision.attribute_input_px` (160; 128 is the fallback). Time it on
the robot (network only; the letterbox adds well under 1 ms on a Mac) and keep 160 if it stays
under about 100 ms:

```bash
$PY -I scripts/bench/fetch.py yolo11n-face-age && $PY -I scripts/bench/fetch.py --export-flat ./models
python run_cm4.py --models-dir ./models --threads 1 --runs 100 \
    --only yolo11n_face_age_128,yolo11n_face_age_160,yolo11n_face_age_224 --json cm4_age.json
```

`roboemot models fetch yolo11n-face-age` on the robot puts the same file at
`~/.local/share/roboemot/models/yolo11n_face_age/model.onnx`; copy it as `models/yolo11n_face_age.onnx`.

## Object detector (SPEC 5, 13)

YOLOX-Nano and NanoDet-Plus-m 320 on a seeded COCO val2017 subset (Apache-2.0 code and official
ONNX; needs `pycocotools`, in `requirements-eval.txt`). Images are downloaded one by one into
`datasets/coco/val2017` on first use (about 250 MB for 1,500; the COCO host is slow, so the first
run can take 20 minutes or more; a rerun is fast). `objects_lib.py` holds the reference
preprocessing, decode and NMS that `tests/test_vision_objects.py` checks the shipped backend
against (`pytest -m models`).

```bash
export OMP_NUM_THREADS=2
$PY -I scripts/bench/fetch.py yolox-nano nanodet-plus coco-annotations coco-samples
nice -n 10 $PY -I scripts/bench/eval_objects.py --n 1500 --seed 0 --out scripts/bench/results/objects.json
$PY -I scripts/bench/fetch.py --export-flat ./models
nice -n 10 $PY -I scripts/bench/run_cm4.py --models-dir ./models --threads 1 --runs 100 \
    --only yolox_nano_416,nanodet_plus_m_320 --json scripts/bench/results/objects_mac_1thread.json
nice -n 10 $PY -I scripts/bench/run_cm4.py --models-dir ./models --threads 2 --runs 100 \
    --only yolox_nano_416,nanodet_plus_m_320 --json scripts/bench/results/objects_mac_2threads.json
```

Pass `--threads 2 --only yolox_nano_416,nanodet_plus_m_320` to the `run_cm4.py` commands below to
time them on the robot (network only; decode and NMS are not included).

## Latency on the robot (Raspberry Pi CM4, aarch64)

`run_cm4.py` is self-contained (onnxruntime, numpy; OpenCV only for YuNet). On the Mac:

```bash
$PY -I scripts/bench/fetch.py --export-flat ./models
scp -r scripts/bench/run_cm4.py scripts/bench/requirements-cm4.txt models pollen@<robot>:~/bench/
```

On the robot (stop other load first; the daemon and app compete for the same cores):

```bash
cd ~/bench && python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements-cm4.txt
.venv/bin/python run_cm4.py --models-dir ./models --threads 2 --runs 50 --json cm4.json
```

It prints a markdown table (p50 and p95 ms per inference, peak RSS per model, each model in its
own process). Run the same command on the Mac for the comparison column. Random input tensors
are used; detector post-processing (NMS, decoding) is not included in the timings. Remove the
bench dir and venv afterwards (`rm -rf ~/bench`) to leave the robot clean (CLAUDE.md, installer
rules).
