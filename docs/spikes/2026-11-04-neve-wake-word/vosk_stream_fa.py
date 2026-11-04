"""Baseline false wakes per hour of today's Vosk-text rule on MLS Italian test (free text only).

Usage: python -I vosk_stream_fa.py WAKEWORD_DIR OUT.json
Each MLS utterance (10 to 20 s) is decoded as one stream; every final result counts as a final text
(as the listener would see it after its VAD split, roughly). Wake rule as in vosk_baseline.py.
"""

import json
import sys
import wave
from pathlib import Path

from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)
W = Path(sys.argv[1])
model = Model(str(W / "data/vosk/vosk-model-small-it-0.22"))
PHRASES = ("ehi neve", "ciao neve", "hey neve")
secs = wakes = word = 0
for f in sorted((W / "data/wav/mls_test").glob("*.wav")):
    with wave.open(str(f)) as wf:
        data = wf.readframes(wf.getnframes())
        secs += wf.getnframes() / 16000
    rec = KaldiRecognizer(model, 16000)
    finals = []
    for i in range(0, len(data), 8000):
        if rec.AcceptWaveform(data[i : i + 8000]):
            finals.append(json.loads(rec.Result()).get("text", ""))
    finals.append(json.loads(rec.FinalResult()).get("text", ""))
    for t in finals:
        w = t.split()
        word += w.count("neve")
        wakes += any(p in t for p in PHRASES) or ("neve" in w and len(w) <= 4)
h = secs / 3600
res = {
    "hours": round(h, 2),
    "wakes": wakes,
    "wakes_per_hour": round(wakes / h, 2),
    "word_neve": word,
}
print(res)
Path(sys.argv[2]).write_text(json.dumps(res))
