"""Vosk + Silero on synthesized speech clips (needs the voice extra and fetched voice models).

The clips are macOS `say` output (Alice it_IT, Samantha en_US), 16 kHz mono, under
tests/data/voice. Latencies use the audio clock plus the measured compute of each chunk, so they
are what a real-time stream would see, not the speed of a file read.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision.models import ModelError
from roboemot.voice import models as voice_models
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.asr import ASREvent, VoskASR
from roboemot.voice.fastpath import FastPath
from roboemot.voice.listener import VoiceListener
from roboemot.voice.vad import Endpointer, SileroVAD

pytestmark = pytest.mark.models

DATA = Path(__file__).parent / "data" / "voice"
RATE = 16000
CHUNK = 512
LEAD_S = 0.5
CFG = RoboEmotConfig()


def load(name: str) -> np.ndarray:
    with wave.open(str(DATA / f"{name}.wav")) as w:
        assert (w.getframerate(), w.getnchannels()) == (RATE, 1)
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return (pcm / 32768.0).astype(np.float32)


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("vosk")
    root = voice_models.models_dir()
    try:
        vad = voice_models.model_file(root, "silero-vad")
        asr = {
            lang: voice_models.model_file(root, name) for lang, name in CFG.voice.asr.models.items()
        }
    except ModelError:
        pytest.skip("voice models not fetched: run `roboemot models fetch --voice`")
    return vad, asr


@pytest.fixture(scope="module")
def vosk_asr(models):
    asr = VoskASR(models[1], RATE)
    asr.load(models[1])
    return asr


def run(models, vosk_asr, clip: str, language: str):
    """Play LEAD silence, the clip and 1.5 s of silence through the listener in real time."""
    vad_path, _ = models
    audio = np.concatenate([np.zeros(round(LEAD_S * RATE), np.float32), load(clip)])
    audio = np.concatenate([audio, np.zeros(round(1.5 * RATE), np.float32)])
    state = SimpleNamespace(end=0.0, began=time.perf_counter())
    order, utterances = [], []
    fast = FastPath(
        CFG.voice,
        lambda name, now, payload: order.append((name, now)),
        None,
        clock=lambda: state.end + (time.perf_counter() - state.began),
    )
    listener = VoiceListener(
        CFG,
        lambda t: None,
        Endpointer(SileroVAD(vad_path, 1), CFG.voice.vad),
        vosk_asr,
        SpeechActivity(),
        fast,
        utterances.append,
        speaker_language=lambda: language,
        clock=lambda: state.end + (time.perf_counter() - state.began),
    )
    for i in range(0, audio.size, CHUNK):
        part = audio[i : i + CHUNK]
        state.end, state.began = (i + part.size) / RATE, time.perf_counter()
        listener.process(SimpleNamespace(samples=part, rate_hz=RATE, stamp=state.end, gap=False))
    return utterances, order, listener.snapshot()["latency"]


CASES = [
    # clip, language, words in the transcript, events before the endpoint
    ("it_greeting_name", "it", {"ciao"}, {"speech.keyword.greeting", "speech.name_called"}),
    ("it_stop", "it", {"fermati"}, {"speech.keyword.stop"}),
    ("en_hello_name", "en", {"hello"}, {"speech.keyword.greeting", "speech.name_called"}),
    ("en_stop", "en", {"stop"}, {"speech.keyword.stop"}),
]


@pytest.mark.parametrize("clip,language,words,events", CASES)
def test_asr_transcript_and_fast_path_before_the_endpoint(
    models, vosk_asr, clip, language, words, events, capsys
):
    utterances, order, latency = run(models, vosk_asr, clip, language)
    assert len(utterances) == 1
    u = utterances[0]
    assert words <= set(u.text.split()), "transcript lost a key word"
    assert u.language == language and u.final
    fired = {name for name, _ in order}
    assert events <= fired
    end_silence_s = CFG.voice.vad.end_silence_ms / 1000
    for name, at in order:
        if name in events:
            assert at < u.last_speech_stamp + end_silence_s, f"{name} came after the end silence"
    kw = latency["keyword_vs_last_speech"]["last_ms"]
    assert kw < CFG.voice.vad.end_silence_ms
    # SPEC 13: end of speech to final transcript, within the end-silence plus ASR finish
    assert latency["final_vs_last_speech"]["last_ms"] < CFG.voice.vad.end_silence_ms + 300
    with capsys.disabled():
        print(
            f"\n[latency] {clip}: keyword {kw:+.0f} ms, "
            f"reaction {latency['reaction_vs_last_speech']['last_ms']:+.0f} ms, "
            f"final {latency['final_vs_last_speech']['last_ms']:+.0f} ms vs last speech"
        )


def test_asr_wake_phrases_are_in_the_vocabulary(vosk_asr, models):
    """`reachy` is out of vocabulary in both small models, so the wake phrases use common words."""
    vosk_asr.load(CFG.voice.asr.models)
    for language, ks in CFG.voice.keywords.items():
        model = vosk_asr._models[language]
        for phrase in ks.wake_phrases:
            for word in phrase.split():
                assert model.vosk_model_find_word(word) >= 0, f"{language}: {word!r} not in vocab"
        assert model.vosk_model_find_word("reachy") == -1
        assert vosk_asr.missing_words(language, ["reachy", "mini"]) == ["reachy"]
        assert vosk_asr.missing_words(language, ["neve"]) == []


def test_asr_wake_word_alias_addresses_a_stranger(models, vosk_asr):
    (u,), _, _ = run(models, vosk_asr, "it_greeting_name", "it")
    assert u.wake_word and u.addressed
    (u,), _, _ = run(models, vosk_asr, "it_stop", "it")
    assert not u.wake_word and not u.addressed


def test_asr_prosody_of_a_real_voice(models, vosk_asr):
    (u,), _, _ = run(models, vosk_asr, "it_stop", "it")
    assert -40 < u.prosody.rms_db < -5
    assert u.prosody.pitch_hz is None or 60 < u.prosody.pitch_hz < 450


def test_asr_backend_accumulates_across_internal_pauses(models, vosk_asr):
    audio = load("it_stop")
    vosk_asr.start("it")
    last = None
    for i in range(0, audio.size, CHUNK):
        ev = vosk_asr.accept(audio[i : i + CHUNK])
        last = ev or last
    assert isinstance(last, ASREvent) and "fermati" in last.text
    assert "fermati" in vosk_asr.finish()
    assert vosk_asr.finish() == ""


def test_asr_languages_are_separate_models(vosk_asr):
    vosk_asr.start("en")
    vosk_asr.accept(load("en_stop"))
    assert "stop" in vosk_asr.finish()
    vosk_asr.start("it")
    vosk_asr.accept(load("it_stop"))
    assert "fermati" in vosk_asr.finish()


# ---------------------------------------------------------------- barge-in on real models


def over_the_robot(
    models,
    vosk_asr,
    robot_clip: str,
    user_clip: str | np.ndarray | None,
    user_at=0.8,
    echo_gain=0.4,
    **cfg,
):
    """The robot plays ``robot_clip`` (heard back at gain 0.4, 80 ms late, by the microphone,
    and kept as the echo reference); the user says ``user_clip`` 0.8 s in. Real Silero (a second
    instance, as in the app) and real Vosk. Returns (stop stamps, the BargeIn, user onset)."""
    from roboemot.voice.bargein import BargeIn, PlaybackReference

    vad_path, _ = models
    config = RoboEmotConfig.model_validate({"voice": {"barge_in": cfg}}) if cfg else CFG
    robot = load(robot_clip)
    seconds = robot.size / RATE
    t0 = 1000.0
    mic = np.zeros(round((seconds + 0.5) * RATE), np.float32)
    lag = round(0.08 * RATE)
    mic[lag : lag + robot.size] += echo_gain * robot
    user = load(user_clip) if isinstance(user_clip, str) else user_clip
    if user is not None:
        lo = round(user_at * RATE)
        mic[lo : lo + user.size] += user[: mic.size - lo]
    ref = PlaybackReference(RATE, config.voice.barge_in.echo_reference_s)
    ref.add(t0, robot)
    state = SimpleNamespace(now=t0, stops=[], events=[])
    bi = BargeIn(
        config,
        SileroVAD(vad_path, 1),
        vosk_asr,
        ref,
        language=lambda: (
            "it" if isinstance(user_clip, str) and user_clip.startswith("it_") else "en"
        ),
        emit=lambda name, now, payload: state.events.append(name),
        clock=lambda: state.now,
    )
    bi.bind(lambda: state.stops.append(state.now), lambda: None, None)
    for i in range(0, mic.size, 1024):
        part = mic[i : i + 1024]
        state.now = t0 + (i + part.size) / RATE
        bi.on_chunk(SimpleNamespace(samples=part, rate_hz=RATE, stamp=state.now, gap=False))
        if state.stops:
            break
    return state, bi, t0 + user_at


def test_the_robots_own_speech_in_the_microphone_does_not_stop_it_on_real_models(models, vosk_asr):
    state, bi, _ = over_the_robot(models, vosk_asr, "en_hello_name", None)
    assert state.stops == [] and bi.snapshot()["echo_frames"] > 5


def speech_start_s(clip: str, floor_db: float = -40.0, frame: int = 512) -> float:
    """Where the clip's speech really starts: its first 32 ms frame above ``floor_db``."""
    x = load(clip)
    for i in range(0, x.size - frame, frame):
        if 20 * np.log10(float(np.sqrt(np.mean(x[i : i + frame] ** 2))) + 1e-12) > floor_db:
            return i / RATE
    raise AssertionError("no speech in the clip")


def test_a_spoken_stop_over_the_robot_cuts_it_off_on_real_models(models, vosk_asr):
    state, bi, clip_start = over_the_robot(models, vosk_asr, "en_hello_name", "it_stop")
    assert len(state.stops) == 1
    snap = bi.snapshot()
    kind = "barge_in" if snap["triggers"] else "keyword"
    true_onset = clip_start + speech_start_s("it_stop")
    ms = 1000 * (state.stops[0] - true_onset)
    print(f"\nreal-model {kind}: stop {ms:.0f} ms after the true speech start (audio clock)")
    if snap["triggers"]:
        assert ms < 200  # SPEC 13: barge-in to TTS stop, on the audio clock
    else:
        assert state.events == ["speech.keyword.stop"] and ms < 600


# The /s/ of "stop" is at -40 to -29 dBFS: an echo much louder than that hides it from the guard
# (the documented limit), so the onset tests use the residual echo of a robot with hardware AEC.
QUIET_ECHO = 0.02


def test_the_robots_own_speech_at_low_echo_does_not_trigger_the_onset_path(models, vosk_asr):
    state, bi, _ = over_the_robot(models, vosk_asr, "en_hello_name", None, echo_gain=QUIET_ECHO)
    assert state.stops == [] and bi.snapshot()["triggers"] == 0


def test_an_english_stop_cuts_the_robot_off_from_its_first_sound_on_real_models(models, vosk_asr):
    """SPEC 13 counts the 200 ms from the first sound. English "stop" starts with a ~130 ms /s/ that
    Silero scores under the barge-in threshold, so the onset path (fricative-like frames) has to
    catch it. Chunks here are 64 ms and the user's phase against the frames is varied."""
    first = speech_start_s("en_stop")
    worst = 0.0
    for shift_ms in (0, 9, 17, 25, 33, 41, 50, 58):
        state, bi, clip_start = over_the_robot(
            models,
            vosk_asr,
            "en_hello_name",
            "en_stop",
            user_at=0.8 + shift_ms / 1000,
            echo_gain=QUIET_ECHO,
        )
        assert len(state.stops) == 1 and bi.snapshot()["triggers"] == 1, shift_ms
        ms = 1000 * (state.stops[0] - (clip_start + first))
        assert 0 <= ms < 200, (shift_ms, ms)
        worst = max(worst, ms)
    print(f"\nreal-model english stop: worst {worst:.0f} ms after the first sound (64 ms chunks)")


def _band(x: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    freq = np.fft.rfftfreq(x.size, 1 / RATE)
    spec[(freq < lo_hz) | (freq > hi_hz)] = 0
    return np.fft.irfft(spec, x.size).astype(np.float32)


def _envelope(n: int, attack_s: float, decay_s: float) -> np.ndarray:
    t = np.arange(n) / RATE
    return np.minimum(t / attack_s, 1.0) * np.exp(-np.maximum(t - attack_s, 0.0) / decay_s)


def short_noises() -> dict[str, np.ndarray]:
    """Synthetic non-speech sounds, each about 0.6 s: a click, a cough (band-limited noise with a
    fast tremolo), a clap, a door slam (low-passed noise burst) and a steady white noise."""
    rng = np.random.default_rng(3)
    n = round(0.6 * RATE)
    white = lambda: rng.standard_normal(n).astype(np.float32)  # noqa: E731
    click = np.zeros(n, np.float32)
    click[3200:3203] = 0.8
    tremolo = 1 + 0.5 * np.sin(2 * np.pi * 14 * np.arange(n) / RATE)
    return {
        "click": click,
        "cough": (_band(white(), 300, 6000) * 0.1 * _envelope(n, 0.02, 0.12) * tremolo).astype(
            np.float32
        ),
        "clap": (white() * 0.2 * _envelope(n, 0.001, 0.03)).astype(np.float32),
        "door": (_band(white(), 20, 300) * 0.25 * _envelope(n, 0.005, 0.08)).astype(np.float32),
        "white_noise": (white() * 0.05).astype(np.float32),
    }


@pytest.mark.parametrize("echo_gain", [0.0, 0.4], ids=["no_echo", "loud_echo"])
@pytest.mark.parametrize("name", list(short_noises()))
def test_short_noises_over_the_robot_do_not_cut_it_off_on_real_models(
    models, vosk_asr, name, echo_gain
):
    state, bi, _ = over_the_robot(
        models,
        vosk_asr,
        "en_hello_name",
        short_noises()[name],
        echo_gain=echo_gain,
        keyword_enabled=False,
    )
    assert state.stops == [], f"{name} stopped the robot"
    assert bi.snapshot()["triggers"] == 0
