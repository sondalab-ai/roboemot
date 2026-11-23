"""Saved clips of the robot's own cloud voice (SPEC 8 Voice clips cache, 12 Data).

The robot plays these when the cloud does not answer ("one moment, retrying", "sorry", "let me
think"). One WAV per kind lives in ``<data folder>/voice-clips/`` as ``<kind>-<digest>.wav``
(PCM16 mono, mode 0600). The digest covers provider, model, voice, language and line, so any
change is a miss, and storing the new clip deletes the old file of its kind. The files hold the
robot's synthetic voice only, never room audio; the line text never appears in a file name.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import wave
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from .. import paths
from ..config.schema import RoboEmotConfig
from ..fileio import replace_file
from .keywords import normalise
from .speech import PCM16_FULL_SCALE

log = logging.getLogger(__name__)

ClipKind = Literal["retrying", "sorry", "thinking"]
CLIP_KINDS: tuple[ClipKind, ...] = get_args(ClipKind)
# What a request to play a clip did: it played; the language has the line and its clip is not
# loaded; or it was refused (no line, no output, or the moment does not allow a sound).
ClipPlay = Literal["played", "missing", "refused"]
FOLDER_MODE = 0o700
FILE_MODE = 0o600
WAV_CHANNELS = 1  # mono
PCM_SAMPLE_BYTES = 2  # PCM16
PCM16_MIN, PCM16_MAX = -32768, 32767


@dataclass(frozen=True)
class ClipKey:
    """What a clip was recorded with; a clip is valid only for an equal key."""

    provider: str
    model: str
    voice: str
    language: str
    line: str

    def digest(self) -> str:
        """sha256 hex of every field, JSON-encoded so field boundaries cannot shift."""
        return hashlib.sha256(json.dumps(astuple(self)).encode()).hexdigest()


@dataclass(frozen=True, eq=False)
class Clip:
    """Mono audio: float32 samples in [-1, 1) at ``rate_hz`` (what speech output plays)."""

    samples: np.ndarray
    rate_hz: int


def same_words(heard: str, line: str) -> bool:
    """Whether ``heard`` (a reply's output transcript) says ``line``: case, accents,
    punctuation and spaces ignored. An empty transcript says nothing."""
    said = normalise(heard).replace(" ", "")
    return bool(said) and said == normalise(line).replace(" ", "")


def clip_folder(cfg: RoboEmotConfig) -> Path:
    """``<data folder>/voice-clips`` (the data folder is created private; this one is not)."""
    return paths.data_dir(cfg.data_dir) / paths.VOICE_CLIPS_SUBDIR


class ClipCache:
    """The clip files in ``folder``; reads never create anything, a write creates it 0700."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _path(self, kind: ClipKind, key: ClipKey) -> Path:
        return self.folder / f"{kind}-{key.digest()}.wav"

    def get(self, kind: ClipKind, key: ClipKey) -> Clip | None:
        """The clip stored for ``kind`` and ``key``; None when missing, unreadable or not
        PCM16 mono (logged by exception type only)."""
        path = self._path(kind, key)
        try:
            with wave.open(str(path), "rb") as w:
                if w.getnchannels() != WAV_CHANNELS or w.getsampwidth() != PCM_SAMPLE_BYTES:
                    log.warning("voice clips: %s clip is not PCM16 mono, ignored", kind)
                    return None
                rate_hz = w.getframerate()
                pcm = w.readframes(w.getnframes())
        except FileNotFoundError:
            return None
        except (wave.Error, EOFError, OSError, ValueError) as exc:
            log.warning("voice clips: %s clip unreadable, ignored: %s", kind, type(exc).__name__)
            return None
        samples = np.frombuffer(pcm, "<i2").astype(np.float32) / np.float32(PCM16_FULL_SCALE)
        return Clip(samples=samples, rate_hz=rate_hz)

    def put(self, kind: ClipKind, key: ClipKey, clip: Clip) -> None:
        """Store ``clip`` for ``kind`` and ``key`` (atomic, mode 0600), then delete every other
        file of ``kind`` (best effort: the new clip is stored, a file left behind is only
        disk)."""
        if not self.folder.exists():
            self.folder.mkdir(mode=FOLDER_MODE, parents=True, exist_ok=True)
            self.folder.chmod(FOLDER_MODE)  # mkdir's mode is masked by the umask
        path = self._path(kind, key)
        replace_file(path, _wav(clip), FILE_MODE)
        for old in self.folder.glob(f"{kind}-*.wav"):
            if old != path:
                with contextlib.suppress(OSError):
                    old.unlink(missing_ok=True)

    def snapshot(self) -> dict[str, str]:
        """Each kind ``cached`` when a file of that kind exists (for any key), else
        ``missing``."""
        return {
            kind: "cached" if any(self.folder.glob(f"{kind}-*.wav")) else "missing"
            for kind in CLIP_KINDS
        }


def _wav(clip: Clip) -> bytes:
    """``clip`` as a PCM16 mono WAV file in memory."""
    scaled = np.round(np.asarray(clip.samples, np.float64) * PCM16_FULL_SCALE)
    pcm = np.clip(scaled, PCM16_MIN, PCM16_MAX).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(WAV_CHANNELS)
        w.setsampwidth(PCM_SAMPLE_BYTES)
        w.setframerate(clip.rate_hz)
        w.writeframes(pcm)
    return out.getvalue()
