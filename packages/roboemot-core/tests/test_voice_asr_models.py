"""Vosk + Silero on synthesized speech clips (needs the voice extra and fetched voice models).

The clips are macOS `say` output (Alice it_IT, Samantha en_US), 16 kHz mono, under
tests/data/voice. Latencies use the audio clock plus the measured compute of each chunk, so they
are what a real-time stream would see, not the speed of a file read.
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from reachy_config import reachy_config
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
CHUNK_20MS = 320
LEAD_S = 0.5
CFG = reachy_config()


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


ROOM_NOISE_RMS = 0.001  # about -60 dBFS


def room_noise(n: int) -> np.ndarray:
    return (ROOM_NOISE_RMS * np.random.default_rng(11).standard_normal(n)).astype(np.float32)


def room_response(tail_gain: float) -> np.ndarray:
    """Direct sound plus a diffuse tail (decay constant 70 ms) at ``tail_gain``: a small room."""
    rng = np.random.default_rng(12)
    n = round(0.3 * RATE)
    tail = rng.standard_normal(n) * np.exp(-np.arange(n) / RATE / 0.07) * tail_gain
    tail[0] = 1.0
    return tail.astype(np.float32)


def over_the_robot(
    models,
    vosk_asr,
    robot_clip: str,
    user_clip: str | np.ndarray | None,
    user_at=0.8,
    echo_gain=0.4,
    room: float | None = None,
    chunk=1024,
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
    if room is not None:  # a smeared echo (direct sound plus a decaying tail) over a noise floor
        mic += room_noise(mic.size)
        echo = np.convolve(robot, room_response(room))[: robot.size]
    else:
        echo = robot
    mic[lag : lag + robot.size] += echo_gain * echo
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
    for i in range(0, mic.size, chunk):
        part = mic[i : i + chunk]
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


# The /s/ of "stop" is at -40 to -29 dBFS: an echo louder than that hides it from any guard (the
# documented limit). QUIET_ECHO is the residual echo of a robot with hardware AEC.
QUIET_ECHO = 0.02
ECHO_GAINS = (0.02, 0.05, 0.1, 0.2, 0.4)
ROOM_TAILS = (0.02, 0.03, 0.04)  # from a damped to a lively room (room_response)
FRICATIVE_LINES = ("robot_en_sunsets", "robot_it_stelle", "en_hello_name")
SHIFTS_MS = (
    0,
    9,
    17,
    25,
    33,
    41,
    50,
    58,
)  # the user's start against the 20 ms chunks and 32 ms frames


def test_the_robots_own_speech_at_low_echo_does_not_trigger_the_onset_path(models, vosk_asr):
    state, bi, _ = over_the_robot(models, vosk_asr, "en_hello_name", None, echo_gain=QUIET_ECHO)
    assert state.stops == [] and bi.snapshot()["triggers"] == 0


@pytest.mark.parametrize("tail", ROOM_TAILS)
@pytest.mark.parametrize("line", FRICATIVE_LINES)
def test_the_onset_path_adds_no_false_trigger_on_the_robots_own_fricatives(
    models, vosk_asr, line, tail
):
    """No user: the robot plays a line full of "s" and the microphone hears it back at 0.02 to 0.4
    through a smeared room echo over a -60 dBFS noise floor. Whatever cuts it off with the onset
    path on must cut it off with the path off too (the main path's own limit, not new)."""
    row = []
    for gain in ECHO_GAINS:
        runs = {}
        for on in (True, False):
            state, _, _ = over_the_robot(
                models,
                vosk_asr,
                line,
                None,
                echo_gain=gain,
                room=tail,
                chunk=CHUNK_20MS,
                keyword_enabled=False,
                onset_enabled=on,
            )
            runs[on] = state.stops
        row.append(f"{gain}:{'cut' if runs[False] else 'ok'}/{'CUT' if runs[True] else 'ok'}")
        assert runs[True] == runs[False] or not runs[True], (line, tail, gain)
    print(
        f"\nown fricatives {line} tail {tail} (gain: main path / with onset path): {' '.join(row)}"
    )


@pytest.mark.parametrize(
    "line,relax",
    [
        ("en_hello_name", {"onset_min_level_db": -80}),
        ("robot_it_stelle", {"onset_echo_corr_max": 0.6, "onset_min_level_db": -80}),
    ],
    ids=["no_level_floor", "no_floor_and_loose_guard"],
)
def test_the_level_floor_and_the_echo_guards_are_what_keep_the_robots_own_s_quiet(
    models, vosk_asr, line, relax
):
    """Regression for the guards: at echo gain 0.02 in the damped room (a robot with hardware
    AEC), the robot's line cuts itself off through the onset path alone when the level floor (or
    the floor and the stricter echo guard) is relaxed, and does not with the defaults."""
    cuts = {}
    for name, cfg in (("relaxed", relax), ("default", {})):
        state, _, _ = over_the_robot(
            models,
            vosk_asr,
            line,
            None,
            echo_gain=QUIET_ECHO,
            room=0.02,
            chunk=CHUNK_20MS,
            keyword_enabled=False,
            **cfg,
        )
        cuts[name] = bool(state.stops)
    assert cuts == {"relaxed": True, "default": False}


@pytest.mark.parametrize("gain", [QUIET_ECHO, 0.05, 0.1])
@pytest.mark.parametrize("robot", ["en_hello_name", "robot_en_sunsets"])
def test_an_english_stop_cuts_the_robot_off_from_its_first_sound_on_real_models(
    models, vosk_asr, robot, gain
):
    """SPEC 13 counts the 200 ms from the first sound. English "stop" starts with a ~130 ms /s/
    that Silero scores under the barge-in threshold, so the onset path has to catch it. Run over
    the robot's speech, heard back at ``gain`` through a room echo and a noise floor, with the
    user's start shifted over 8 phases of the 20 ms chunk and 32 ms frame grids. At 0.1 the echo
    (about -33 dBFS) is as loud as the /s/: the stop still comes but may be late (reported, not
    asserted); at 0.02 and 0.05 it must be inside the budget."""
    first = speech_start_s("en_stop")
    ms = []
    for shift_ms in SHIFTS_MS:
        state, bi, clip_start = over_the_robot(
            models,
            vosk_asr,
            robot,
            "en_stop",
            user_at=0.8 + shift_ms / 1000,
            echo_gain=gain,
            room=0.03,
            chunk=CHUNK_20MS,
        )
        assert len(state.stops) == 1 and bi.snapshot()["triggers"] == 1, shift_ms
        ms.append(1000 * (state.stops[0] - (clip_start + first)))
    print(f"\nreal-model english stop over {robot} at echo {gain}: {[round(x) for x in ms]} ms")
    assert min(ms) >= 0
    if gain <= 0.05:
        assert max(ms) < 200, ms


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

    def white() -> np.ndarray:
        return rng.standard_normal(n).astype(np.float32)

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


# Sustained loud noise (the cough and the white noise here, 0.6 s at -26 to -20 dBFS) over a loud
# echo is outside what the guards can tell from speech: the window fills with noise, the
# correlation falls and Silero, still carrying the robot's speech in its state, scores it 0.5 to
# 0.7. It is listed in docs/eval/m3.md as a limit, not tested as a pass.
NOISE_CASES = [
    (name, gain)
    for name in short_noises()
    for gain in (0.0, 0.4)
    if not (gain == 0.4 and name in ("cough", "white_noise"))
]


@pytest.mark.parametrize("name,echo_gain", NOISE_CASES)
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


RUNS = 5  # Vosk varies run to run on some clips, a new recognizer too (4 texts in 30 runs seen)


def test_reused_recognizer_decodes_like_a_new_one(models):
    """One recognizer per language, reused across utterances and after an abort (half a clip
    then ``reset``), gives the texts a new recognizer gives (SPEC 8 stage 3, 13). Vosk is not
    deterministic, so each clip's texts must overlap a new recognizer's: a leaked half clip
    gives a different text every time."""
    import vosk

    asr = VoskASR(models[1], RATE)
    asr.load(models[1])
    built = asr.stats()["recognizers_built"]
    for language, path in models[1].items():
        model = vosk.Model(str(path))
        clips = sorted(p.stem for p in DATA.glob(f"*{language}_*.wav"))
        assert clips
        for clip in clips:
            audio = load(clip)
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
            fresh, reused, aborted = set(), set(), set()
            for _ in range(RUNS):
                rec = vosk.KaldiRecognizer(model, RATE)
                rec.AcceptWaveform(pcm)
                fresh.add(json.loads(rec.FinalResult())["text"])
                asr.start(language)
                asr.accept(audio)
                reused.add(asr.finish())
                asr.start(language)
                asr.accept(audio[: audio.size // 2])
                asr.reset()  # an abort: the half must not leak into the next utterance
                asr.start(language)
                asr.accept(audio)
                aborted.add(asr.finish())
            assert reused & fresh, clip
            assert aborted & fresh, clip
    assert asr.stats() == {
        "recognizers_built": built,
        "built_on_start": 0,
        "reset_failures": 0,
        "decodes": 0,
    }
