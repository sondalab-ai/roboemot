"""Write the three train.py configs (v1, v2, NC reference) and the clip-generator stub.

Usage: python -I train_configs.py WAKEWORD_DIR
Writes WAKEWORD_DIR/{neve.yaml,neve_v2.yaml,neve_nc.yaml}, WAKEWORD_DIR/stub_psg/generate_samples.py
(train.py imports a Piper sample generator; clips come from gen_clips.py instead) and the output
folders out_v2/ and out_nc/ that share out/neve/ (the features from --augment_clips).
"""

import sys
from pathlib import Path

W = Path(sys.argv[1]).resolve()
BASE = {
    "model_name": "neve",
    "target_phrase": ["neve"],
    "custom_negative_phrases": [],
    "n_samples": 6400,
    "n_samples_val": 1000,
    "tts_batch_size": 50,
    "augmentation_batch_size": 16,
    "piper_sample_generator_path": str(W / "stub_psg"),
    "rir_paths": [str(W / "data/rir_sim")],
    "background_paths": [str(W / "data/bg_noise"), str(W / "data/wav/ls_dev")],
    "background_paths_duplication_rate": [1, 1],
    "augmentation_rounds": 1,
    "model_type": "dnn",
    "layer_size": 32,
    "steps": 50000,
    "max_negative_weight": 1500,
    "target_false_positives_per_hour": 0.2,
}
FEAT = W / "data/feat"
RUNS = {
    "neve.yaml": (
        "out",
        "fpval_mls_dev.npy",
        {"speech_neg": (FEAT / "neg_train.npy", 1024)},
    ),
    "neve_v2.yaml": (
        "out_v2",
        "fpval_mls_dev_aug.npy",
        {
            "speech_neg": (FEAT / "neg_train.npy", 512),
            "speech_aug": (FEAT / "neg_aug_mls_9h_ls_dev.npy", 1024),
        },
    ),
    "neve_nc.yaml": (
        "out_nc",
        "fpval_mls_dev.npy",
        {
            "speech_neg": (FEAT / "neg_train.npy", 256),
            "acav": (W / "data/oww_features/acav_700k_f32.npy", 1024),
        },
    ),
}


def dump(cfg):
    """Tiny YAML writer for this flat config (keeps the script free of a PyYAML dependency)."""
    out = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.append(f"{k}:")
            out += [f'  "{a}": {b if isinstance(b, int) else repr(str(b))}' for a, b in v.items()]
        elif isinstance(v, list):
            out.append(f"{k}: [{', '.join(repr(x) for x in v)}]")
        else:
            out.append(f"{k}: {repr(v) if isinstance(v, str) else v}")
    return "\n".join(out) + "\n"


(W / "stub_psg").mkdir(exist_ok=True)
(W / "stub_psg/generate_samples.py").write_text(
    "def generate_samples(*a, **k):\n    raise RuntimeError('clips come from gen_clips.py')\n"
)
for name, (out, fpval, feats) in RUNS.items():
    (W / out).mkdir(exist_ok=True)
    if out != "out" and not (W / out / "neve").exists():
        (W / out / "neve").symlink_to(W / "out/neve")
    cfg = dict(BASE, output_dir=str(W / out), false_positive_validation_data_path=str(FEAT / fpval))
    cfg["feature_data_files"] = {k: v[0] for k, v in feats.items()}
    cfg["batch_n_per_class"] = {k: v[1] for k, v in feats.items()}
    cfg["batch_n_per_class"].update({"adversarial_negative": 100, "positive": 100})
    (W / name).write_text(dump(cfg))
    print("wrote", W / name)
