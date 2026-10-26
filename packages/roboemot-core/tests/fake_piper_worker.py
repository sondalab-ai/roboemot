"""Stand-in for the Piper engine: the real protocol loop (``roboemot.voice.piper_worker``) with a
synthetic tone, so the backend's framing, cancel and restart paths are tested without Piper.

Behaviour is picked by words in the text: CRASH exits mid-sentence (every time), CRASHONCE does
so only the first time for this voice, SLOW sends 6 pieces 0.15 s apart, FLOOD writes 3 MB to
stderr first, BOOM raises. The thread counts it was started with are written to
``<voice>.threads``. Otherwise a 0.1 s tone, scaled by ``length_scale`` and ``volume``.
"""

import importlib.util
import math
import os
import struct
import sys
import time
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "src/roboemot/voice/piper_worker.py"
spec = importlib.util.spec_from_file_location("piper_worker", WORKER)
piper_worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(piper_worker)


class ToneEngine:
    def __init__(self, voice_path: str, intra_op_threads: int, inter_op_threads: int) -> None:
        self.voice_path = voice_path
        Path(voice_path + ".threads").write_text(f"{intra_op_threads} {inter_op_threads}")
        self.sample_rate = piper_worker.read_sample_rate(voice_path)

    def tone(self, seconds: float, volume: float) -> bytes:
        n = int(self.sample_rate * seconds)
        amp = int(10000 * volume)
        return b"".join(
            struct.pack("<h", int(amp * math.sin(2 * math.pi * 440 * i / self.sample_rate)))
            for i in range(n)
        )

    def synthesize(self, request):
        text = request["text"]
        if "FLOOD" in text:
            sys.stderr.write("x" * 3_000_000)
            sys.stderr.flush()
        if "CRASH" in text and "CRASHONCE" not in text:
            os._exit(3)
        if "CRASHONCE" in text:
            marker = Path(self.voice_path + ".crashed")
            if not marker.exists():
                marker.write_text("1")
                os._exit(3)
        if "BOOM" in text:
            raise ValueError(f"cannot say {text}")
        seconds = 0.1 * request["length_scale"]
        if "SLOW" in text:
            for _ in range(6):
                time.sleep(0.15)
                yield self.tone(seconds / 6, request["volume"])
            return
        yield self.tone(seconds, request["volume"])


if __name__ == "__main__":
    sys.exit(piper_worker.serve(ToneEngine))
