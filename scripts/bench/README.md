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
