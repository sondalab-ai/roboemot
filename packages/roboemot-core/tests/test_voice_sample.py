"""SPEC 8: `roboemot voice sample` renders the fixed reply in every mood, through the Piper worker
protocol (here the fake engine: a tone whose length follows length_scale)."""

from __future__ import annotations

import asyncio
import sys
import wave

import pytest
from roboemot import cli
from roboemot.config.schema import PiperConfig, RoboEmotConfig
from roboemot.voice import piper_backend
from roboemot.voice.piper_backend import PiperBackend
from roboemot.voice.sample import SAMPLE_RATE_HZ, SAMPLE_TEXT, render_samples
from test_tts import FAKE_WORKER, make_voice, resolver

CFG = RoboEmotConfig()


def _voices(tmp_path):
    voices = make_voice(tmp_path, name="it_IT-paola-medium", rate=16000)
    make_voice(tmp_path, name="en_US-amy-medium", rate=16000)
    return voices


def _frames(path):
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SAMPLE_RATE_HZ)
        return w.getnframes()


def test_one_file_per_mood_plus_neutral_with_the_mood_delivery(tmp_path):
    backend = PiperBackend(
        PiperConfig(chunk_ms=20.0), sys.executable, resolver(_voices(tmp_path)), script=FAKE_WORKER
    )

    async def run():
        try:
            return await render_samples(
                CFG, backend, tmp_path / "out", language="it", intensity=0.5
            )
        finally:
            await backend.aclose()

    samples = asyncio.run(run())
    assert [s.mood for s in samples] == ["neutral", *CFG.voice.moods]
    by_mood = {s.mood: s for s in samples}
    for s in samples:
        assert s.path == tmp_path / "out" / f"it-{s.mood}.wav" and _frames(s.path) > 0
        assert len(s.deliveries) == len(SAMPLE_TEXT["it"])
        assert s.deliveries[0].mood == (None if s.mood == "neutral" else s.mood)
    # a slower mood makes a longer file (the fake tone scales with length_scale)
    assert by_mood["sadness"].seconds > by_mood["neutral"].seconds > by_mood["surprise"].seconds


def test_the_same_run_writes_the_same_files(tmp_path):
    def render(out):
        backend = PiperBackend(
            PiperConfig(chunk_ms=20.0), sys.executable, resolver(_voices(tmp_path)),
            script=FAKE_WORKER,
        )  # fmt: skip

        async def run():
            try:
                return await render_samples(CFG, backend, out, language="it", intensity=0.5)
            finally:
                await backend.aclose()

        return asyncio.run(run())

    first, second = render(tmp_path / "a"), render(tmp_path / "b")
    for a, b in zip(first, second, strict=True):
        assert a.deliveries == b.deliveries and a.path.read_bytes() == b.path.read_bytes()


def test_cli_voice_sample_uses_piper_with_the_manifest_layer(tmp_path, monkeypatch, capsys):
    voices = _voices(tmp_path)
    monkeypatch.setattr(cli, "PIPER_SCRIPT", FAKE_WORKER)
    monkeypatch.setattr(piper_backend, "manifest_voice_files", lambda root: resolver(voices))
    manifest = (
        cli.Path(__file__).resolve().parents[3] / "configs/manifests/reachy-mini-wireless.yaml"
    )
    code = cli.main(
        ["voice", "sample", "--out", str(tmp_path / "wav"), "--lang", "en",
         "--python", sys.executable, "--manifest", str(manifest), "--dev"]
    )  # fmt: skip
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 + len(CFG.voice.moods)
    assert (tmp_path / "wav" / "en-neutral.wav").is_file()
    # the robot's base pitch (+5 st) applies, give or take the sentence variation
    first_pitch = float(lines[0].split("pitch_st ")[1].split()[0])
    bound = CFG.voice.variation.pitch_st * CFG.voice.expressivity
    assert abs(first_pitch - 5.0) <= bound + 0.05


@pytest.mark.parametrize("argv", [["--intensity", "1.5"], ["--lang", "fr"]])
def test_cli_voice_sample_rejects_bad_arguments(tmp_path, argv, capsys):
    try:
        code = cli.main(["voice", "sample", "--out", str(tmp_path), *argv])
    except SystemExit as exc:  # argparse choices
        code = exc.code
    assert code != 0


def test_cli_voice_sample_needs_dev_for_a_manifest_with_unmeasured_dynamics(tmp_path, capsys):
    manifest = (
        cli.Path(__file__).resolve().parents[3] / "configs/manifests/reachy-mini-wireless.yaml"
    )
    assert cli.main(["voice", "sample", "--out", str(tmp_path), "--manifest", str(manifest)]) == 1
    assert "allow_unmeasured_dynamics" in capsys.readouterr().err


def test_cli_voice_sample_errors_say_what_to_do_and_leave_no_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PIPER_SCRIPT", FAKE_WORKER)
    empty = tmp_path / "no-voices"
    empty.mkdir()
    monkeypatch.setattr(piper_backend, "manifest_voice_files", lambda root: resolver(empty))
    out = tmp_path / "wav"
    argv = ["voice", "sample", "--out", str(out), "--python", sys.executable]
    assert cli.main(argv) == 1
    assert "roboemot models fetch --voice" in capsys.readouterr().err
    monkeypatch.setattr(
        piper_backend, "manifest_voice_files", lambda root: resolver(_voices(tmp_path))
    )
    argv[-1] = str(tmp_path / "no-python")
    assert cli.main(argv) == 1
    assert "--python <piper-venv>/bin/python" in capsys.readouterr().err
    assert not out.exists()


def test_a_chunk_above_the_sample_rate_is_refused_not_aliased():
    import numpy as np
    from roboemot.voice.sample import _resample

    with pytest.raises(ValueError, match="above the sample rate"):
        _resample(np.ones(10, np.float32), SAMPLE_RATE_HZ * 2, SAMPLE_RATE_HZ)
