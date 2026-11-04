"""Generate positive and hard-negative clips for the Neve spike with Piper (runs in the piper venv).

Usage: python -I gen_clips.py OUT_DIR
Layout: OUT/neve/{positive_train,positive_test,negative_train,negative_test} (openWakeWord train.py
layout) and OUT/eval/<set>/ for held-out evaluation. 16 kHz mono int16 WAV, one file per clip.
"""

import os
import random
import sys
from math import gcd
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import texts as T  # noqa: E402

# piper-tts 1.8.0 on macOS falls back to a build-time espeak path when the data path is long (an
# espeak-ng path buffer limit), so point ESPEAK_DATA at a short symlink to
# <venv>/.../piper/espeak-ng-data.
ED = os.environ["ESPEAK_DATA"]
VD = os.environ["WAKEWORD_DIR"] + "/data/piper/"
LIBRI = "en_US-libritts_r-medium"
N_LIBRI = 904

_voices = {}


def voice(name):
    if name not in _voices:
        _voices[name] = PiperVoice.load(VD + name + ".onnx", espeak_data_dir=ED)
    return _voices[name]


def synth(job):
    """job = (out_path, voice, speaker_id, text, seed); text '[[...]]' is raw IPA."""
    out, vname, sid, text, seed = job
    if Path(out).exists():
        return
    rnd = random.Random(seed)
    v = voice(vname)
    cfg = SynthesisConfig(
        speaker_id=sid,
        length_scale=rnd.uniform(0.8, 1.35),
        noise_scale=rnd.uniform(0.4, 0.9),
        noise_w_scale=rnd.uniform(0.5, 1.1),
    )
    audio = np.concatenate([c.audio_int16_array for c in v.synthesize(text, syn_config=cfg)])
    sr = v.config.sample_rate
    if sr != 16000:
        g = gcd(sr, 16000)
        audio = resample_poly(audio.astype(np.float32), 16000 // g, sr // g)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, 16000, audio)


def pos_texts(vname):
    if vname.startswith("it_"):
        return T.POS_IT_TEXT + [f"[[{p}]]" for p in T.POS_IPA_IT]
    if vname == LIBRI:
        return T.POS_EN_TEXT + [f"[[{p}]]" for p in T.POS_IPA_EN + T.POS_IPA_IT]
    return [f"[[{p}]]" for p in T.POS_IPA_IT]


def jobs(out):
    rnd = random.Random(1234)
    J = []
    libri_train = [s for s in range(N_LIBRI) if s % 10]
    libri_held = [s for s in range(N_LIBRI) if s % 10 == 0]

    def add(dirname, vname, sids, texts_, n, tag):
        for i in range(n):
            sid = rnd.choice(sids) if sids else None
            J.append(
                (
                    f"{out}/{dirname}/{tag}_{vname}_{sid}_{i:05d}.wav",
                    vname,
                    sid,
                    rnd.choice(texts_),
                    rnd.random(),
                )
            )

    tr_it = ["it_IT-riccardo-x_low", "it_IT-serena-high"]
    tr_other = [
        "es_ES-davefx-medium",
        "pt_PT-tugão-medium",
        "ro_RO-mihai-medium",
        "es_AR-daniela-high",
        "pt_BR-cadu-medium",
    ]
    # positives, training voices
    for v in tr_it:
        add("neve/positive_train", v, None, pos_texts(v), 1200, "pos")
        add("neve/positive_test", v, None, pos_texts(v), 150, "pos")
    for v in tr_other:
        add("neve/positive_train", v, None, pos_texts(v), 200, "pos")
        add("neve/positive_test", v, None, pos_texts(v), 30, "pos")
    add("neve/positive_train", LIBRI, libri_train, pos_texts(LIBRI), 3000, "pos")
    add("neve/positive_test", LIBRI, libri_train, pos_texts(LIBRI), 250, "pos")
    # positives, held-out voices
    add("eval/pos_paola", "it_IT-paola-medium", None, pos_texts("it_"), 400, "pos")
    add("eval/pos_serena_medium", "it_IT-serena-medium", None, pos_texts("it_"), 200, "pos")
    add("eval/pos_es_mx_claude", "es_MX-claude-high", None, pos_texts("x"), 150, "pos")
    add("eval/pos_pt_br_faber", "pt_BR-faber-medium", None, pos_texts("x"), 150, "pos")
    add("eval/pos_libritts_heldout", LIBRI, libri_held, pos_texts(LIBRI), 500, "pos")
    # hard negatives
    S = T.splits()
    for k, d, reps in (("train", "neve/negative_train", 6), ("val", "neve/negative_test", 3)):
        it_neg = S[k]["neve"] + S[k]["near"]
        for v in tr_it:
            add(d, v, None, it_neg, len(it_neg) * reps, "neg")
        add(d, LIBRI, libri_train, S[k]["en"], len(S[k]["en"]) * 40, "neg")
    for v in ("it_IT-paola-medium", "it_IT-serena-medium", *tr_it):
        add(
            "eval/neg_neve_sentences", v, None, S["test"]["neve"], len(S["test"]["neve"]) * 4, "neg"
        )
        add("eval/neg_near_words", v, None, S["test"]["near"], len(S["test"]["near"]) * 4, "neg")
    add("eval/neg_en_near", LIBRI, libri_held, S["test"]["en"], 200, "neg")
    return J


def main():
    out = sys.argv[1]
    J = jobs(out)
    if "--skip-slow" in sys.argv:  # serena-high is ~10x slower; keep only the clips already made
        J = [j for j in J if j[1] != "it_IT-serena-high" or Path(j[0]).exists()]
    J.sort(key=lambda j: j[1])  # group by voice so each worker loads few voices
    with Pool(8) as p:
        for i, _ in enumerate(p.imap_unordered(synth, J, chunksize=64)):
            if i % 2000 == 0:
                print(i, len(J), flush=True)
    print("done", len(J))


if __name__ == "__main__":
    main()
