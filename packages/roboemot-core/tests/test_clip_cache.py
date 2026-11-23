"""SPEC 8, 12: the on-disk cache of the robot's own cloud-voice clips."""

from __future__ import annotations

import stat
import wave
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from roboemot.voice.clip_cache import Clip, ClipCache, ClipKey, clip_folder

KEY = ClipKey(
    provider="api.openai.com",
    model="gpt-realtime",
    voice="alloy",
    language="it",
    line="Un momento, riprovo.",
)
# multiples of 1/32768 survive the PCM16 round trip exactly
CLIP = Clip(samples=np.array([0, 1, -1, 16384, -32768, 32767], np.float32) / 32768, rate_hz=24000)


def test_a_stored_clip_reads_back_for_the_same_key(tmp_path):
    c = ClipCache(tmp_path)
    c.put("sorry", KEY, CLIP)
    got = c.get("sorry", KEY)
    assert got is not None
    assert got.rate_hz == CLIP.rate_hz
    assert got.samples.dtype == np.float32
    np.testing.assert_array_equal(got.samples, CLIP.samples)


def test_a_missing_clip_is_a_miss(tmp_path):
    assert ClipCache(tmp_path).get("sorry", KEY) is None
    assert ClipCache(tmp_path / "absent").get("sorry", KEY) is None


def test_a_changed_voice_or_language_misses_the_cache(tmp_path):
    c = ClipCache(tmp_path)
    c.put("sorry", KEY, CLIP)
    assert c.get("sorry", replace(KEY, voice="Puck")) is None
    assert c.get("sorry", replace(KEY, language="en")) is None
    assert c.get("sorry", replace(KEY, line="Scusa.")) is None
    assert c.get("retrying", KEY) is None


def test_fields_do_not_run_into_each_other_in_the_digest():
    a = replace(KEY, provider="ab", model="c")
    b = replace(KEY, provider="a", model="bc")
    assert a.digest() != b.digest()


def test_a_new_clip_replaces_the_old_file_of_its_kind(tmp_path):
    c = ClipCache(tmp_path)
    c.put("sorry", KEY, CLIP)
    c.put("retrying", KEY, CLIP)
    c.put("sorry", replace(KEY, voice="Puck"), CLIP)
    assert len(list(tmp_path.glob("sorry-*.wav"))) == 1
    assert len(list(tmp_path.glob("retrying-*.wav"))) == 1
    assert c.get("sorry", KEY) is None
    assert c.get("sorry", replace(KEY, voice="Puck")) is not None


def test_clip_files_are_private_and_hold_no_line_text(tmp_path):
    folder = tmp_path / "voice-clips"
    ClipCache(folder).put("thinking", KEY, CLIP)
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    (path,) = folder.iterdir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.name == f"thinking-{KEY.digest()}.wav"
    for word in KEY.line.lower().replace(",", "").replace(".", "").split():
        assert word not in path.name.lower()


def test_a_corrupt_or_foreign_wav_is_a_miss(tmp_path):
    c = ClipCache(tmp_path)
    path = tmp_path / f"sorry-{KEY.digest()}.wav"
    path.write_bytes(b"not a wav file")
    assert c.get("sorry", KEY) is None
    with wave.open(str(path), "wb") as w:  # stereo: not what the cache writes
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00\x00\x00")
    assert c.get("sorry", KEY) is None


def test_snapshot_tells_each_kind_cached_or_missing(tmp_path):
    c = ClipCache(tmp_path / "voice-clips")
    assert c.snapshot() == {"retrying": "missing", "sorry": "missing", "thinking": "missing"}
    assert not (tmp_path / "voice-clips").exists()  # reading never creates the folder
    c.put("sorry", KEY, CLIP)
    assert c.snapshot() == {"retrying": "missing", "sorry": "cached", "thinking": "missing"}


def test_the_clip_folder_is_in_the_app_data_folder(tmp_path):
    cfg = SimpleNamespace(data_dir=str(tmp_path / "data"))
    assert clip_folder(cfg) == tmp_path / "data" / "voice-clips"


def test_a_stale_file_that_cannot_be_deleted_keeps_the_new_clip(tmp_path, monkeypatch):
    c = ClipCache(tmp_path)
    c.put("sorry", KEY, CLIP)
    new = replace(KEY, voice="Puck")
    real_unlink = type(tmp_path).unlink

    def refuse(self, missing_ok=False):
        if self.name.startswith("sorry-") and KEY.digest() in self.name:
            raise PermissionError("read-only")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(type(tmp_path), "unlink", refuse)
    c.put("sorry", new, CLIP)  # best-effort cleanup: no error
    assert c.get("sorry", new) is not None
