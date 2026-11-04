"""Baseline: today's Vosk-text wake rule on the same eval clips (free text only, no grammar pass).

Usage: python -I vosk_baseline.py WAKEWORD_DIR OUT.json [SET,SET,...]
Wake = final text holds 'ehi neve' / 'ciao neve' / 'hey neve', or the word 'neve' in a final of at
most 4 words (voice.fastpath.alias_wake_max_words). Conditions: clean, and white noise at 10 dB SNR
("snr10" in the output keys). Per file: the final text and whether it woke.
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)
W = Path(sys.argv[1])
model = Model(str(W / "data/vosk/vosk-model-small-it-0.22"))
rng = np.random.default_rng(5)
PHRASES = ("ehi neve", "ciao neve", "hey neve")


def text_of(x):
    rec = KaldiRecognizer(model, 16000)
    pad = np.zeros(8000, dtype=np.int16)
    rec.AcceptWaveform(np.concatenate([pad, x, pad, pad]).tobytes())
    return json.loads(rec.FinalResult()).get("text", "")


def wake(t):
    w = t.split()
    return any(p in t for p in PHRASES) or ("neve" in w and len(w) <= 4), "neve" in w


SETS = sys.argv[3].split(",") if len(sys.argv) > 3 else None
res = {}
for name in SETS or ("pos_paola", "pos_serena_medium", "neg_neve_sentences", "neg_near_words"):
    files = sorted((W / "out/eval" / name).glob("*.wav"))
    for cond in ("clean", "snr10"):
        n_wake = n_word = 0
        per_file = []
        for f in files:
            with wave.open(str(f)) as wf:
                x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
            if cond == "snr10":
                x = x + rng.normal(size=len(x)) * np.sqrt(np.mean(x**2) / 10)
            x = np.clip(x, -32768, 32767).astype(np.int16)
            t = text_of(x)
            a, b = wake(t)
            per_file.append([f.name, t, bool(a)])
            n_wake += a
            n_word += b
        res[f"{name}/{cond}"] = {
            "n": len(files),
            "wake": n_wake / len(files),
            "word_neve": n_word / len(files),
            "_files": per_file,
        }
        print(name, cond, {k: v for k, v in res[f"{name}/{cond}"].items() if k != "_files"})
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
