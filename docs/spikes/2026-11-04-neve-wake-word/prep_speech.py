"""Decode MLS Italian (parquet, opus) and LibriSpeech (flac) into 16 kHz int16 WAV folders.

Usage: python -I prep_speech.py WAKEWORD_DIR
Writes data/wav/{mls_9h,mls_test,mls_dev,ls_dev,ls_test}/ and data/wav/<set>.neve.txt (ids whose
transcript holds a word starting with 'nev', for annotation only).
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly

W = Path(sys.argv[1])
OUT = W / "data/wav"


def save(path, audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr)
    wavfile.write(path, 16000, np.clip(audio * 32767, -32768, 32767).astype(np.int16))
    return len(audio) / 16000


def mls(name, parquet):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    total, nev = 0.0, []
    pf = pq.ParquetFile(parquet)
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=["audio", "id", "transcript"]).to_pylist():
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            total += save(d / f"{r['id']}.wav", a, sr)
            if any(w.startswith("nev") for w in r["transcript"].split()):
                nev.append({"id": r["id"], "text": r["transcript"]})
    (OUT / f"{name}.neve.json").write_text(json.dumps(nev, ensure_ascii=False, indent=1))
    print(name, f"{total / 3600:.2f} h", "nev-utterances", len(nev), flush=True)


def libri(name, root):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for f in sorted(Path(root).rglob("*.flac")):
        a, sr = sf.read(f, dtype="float32")
        total += save(d / (f.stem + ".wav"), a, sr)
    print(name, f"{total / 3600:.2f} h", flush=True)


mls("mls_9h", W / "data/mls_it/9_hours-00000-of-00001.parquet")
mls("mls_test", W / "data/mls_it/test-00000-of-00001.parquet")
mls("mls_dev", W / "data/mls_it/dev-00000-of-00001.parquet")
libri("ls_dev", W / "data/librispeech/LibriSpeech/dev-clean")
libri("ls_test", W / "data/librispeech/LibriSpeech/test-clean")
