"""M3 exit (SPEC 14): an Italian and an English conversation inside the SPEC 13 latency budgets,
through the real app on a scripted SDK client, with the real Silero VAD and Vosk models.

Opt in with ``-m models`` (needs the ``voice`` extra and ``roboemot models fetch --voice``). What is
real: the audio hub, VAD, ASR, fast path, listener, conversation worker, sentence splitter, speech
output, speaker, barge-in and the router (its HTTP client runs over an in-process mock transport).
What is synthetic: the microphone (macOS ``say`` clips fed at real time, no room, no echo, 20 ms
chunks), the speaker (a recorder), the LLM route (first-token latency ``FIRST_TOKEN_S``, a token
rate), and the TTS when no Piper interpreter is given (``ROBOEMOT_PIPER_PYTHON``): then a stand-in
with a first-chunk delay of ``FAKE_TTS_FIRST_CHUNK_S``. See ``docs/eval/m3.md``.

Set ``ROBOEMOT_EXIT_REPORT`` to a file path to also get the measured numbers as JSON.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import json
import os
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import roboemot_reachymini.app as app_module
from roboemot import paths
from roboemot.config.schema import RoboEmotConfig, VoiceConfig
from roboemot.llm import Router
from roboemot.vision.models import ModelError
from roboemot.voice import factory
from roboemot.voice import models as voice_models
from roboemot.voice.tts import AudioChunk
from test_app_perception import data, wait_for  # noqa: F401  (data: fixture)
from test_app_reflex import ScriptedMini, enable, run_app

pytestmark = pytest.mark.models

RATE = 16000
MIC_CHUNK = 320  # 20 ms; the real chunk size is a SPEC 15 robot item
FRAME = 512  # a Silero frame, 32 ms
FLOOR_DB = -40.0
LEAD_S = 0.5
TAIL_S = 2.0
DATA = Path(__file__).parents[2] / "roboemot-core" / "tests" / "data" / "voice"
# Resolved at import, before the autouse fixture points the data folder at a temporary one.
MODELS_DIR = Path(os.environ.get("ROBOEMOT_VOICE_MODELS_DIR") or voice_models.models_dir())
PIPER_PYTHON = Path(os.environ.get("ROBOEMOT_PIPER_PYTHON") or paths.piper_python(""))

# SPEC 13 budgets (cm4 column for first audio; the npu one is 1.5 s).
FAST_PATH_BUDGET_MS = 300.0
FIRST_AUDIO_BUDGET_MS = 2000.0
BARGE_IN_BUDGET_MS = 200.0
# A LAN model server: time to first token 400 to 700 ms. The middle of that range.
FIRST_TOKEN_S = 0.55
TOKEN_GAP_S = 0.03  # about 30 tokens per second after the first
# Stand-in for Piper when no interpreter is given: roughly what Piper took per first sentence
# in a Linux container on the Mac (65 to 110 ms), with margin. The cm4 value is unmeasured.
FAKE_TTS_FIRST_CHUNK_S = 0.25

REPLIES = {
    "Italian": "Ciao, che bello vederti! Sono un piccolo robot curioso. Come stai oggi?",
    "English": "Hello, how nice to see you! I am a small curious robot. How are you today?",
}
CASES = [
    # The designed wake phrases ("ciao robot", "hello robot"); a name call is not enough in English:
    # Vosk hears "hello reachy" as "hello yeah gee" (docs/eval/m3.md).
    # language, wake clip, stop clip, words in the transcript
    ("it", "it_ciao_robot", "it_stop", {"ciao", "robot"}),
    ("en", "en_hello_robot", "en_stop", {"hello", "robot"}),
]
RESULTS: dict[str, dict] = {}


def load(name: str) -> np.ndarray:
    with wave.open(str(DATA / f"{name}.wav")) as w:
        assert (w.getframerate(), w.getnchannels()) == (RATE, 1)
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return (pcm / 32768.0).astype(np.float32)


def speech_span_s(clip: np.ndarray) -> tuple[float, float]:
    """Start of the first and of the last 32 ms frame louder than ``FLOOR_DB`` in the clip."""
    loud = [
        i
        for i in range(0, clip.size - FRAME, FRAME)
        if 20 * np.log10(float(np.sqrt(np.mean(clip[i : i + FRAME] ** 2))) + 1e-12) > FLOOR_DB
    ]
    assert loud, "no speech in the clip"
    return loud[0] / RATE, loud[-1] / RATE


class LiveMic:
    """A microphone paced by the wall clock: ``get_audio_sample`` returns the next 20 ms chunk once
    it has been captured, None before. ``play`` lays a clip onto the stream from the next chunk and
    returns the wall time its first sample is captured at."""

    def __init__(self) -> None:
        self._t0: float | None = None
        self._served = 0
        self._overlays: list[tuple[int, np.ndarray]] = []
        self._lock = threading.Lock()

    def start(self) -> float:
        with self._lock:
            if self._t0 is None:
                self._t0 = time.monotonic()
            return self._t0

    def play(self, samples: np.ndarray) -> float:
        with self._lock:
            assert self._t0 is not None
            now_idx = int((time.monotonic() - self._t0) * RATE)
            start = max(self._served * MIC_CHUNK, now_idx) + MIC_CHUNK
            self._overlays.append((start, samples))
            return self._t0 + start / RATE

    def read(self):
        with self._lock:
            if self._t0 is None:
                self._t0 = time.monotonic()
            if (time.monotonic() - self._t0) * RATE < (self._served + 1) * MIC_CHUNK:
                return None
            lo = self._served * MIC_CHUNK
            out = np.zeros(MIC_CHUNK, np.float32)
            for start, x in self._overlays:
                a, b = max(lo, start), min(lo + MIC_CHUNK, start + x.size)
                if a < b:
                    out[a - lo : b - lo] += x[a - start : b - start]
            self._served += 1
            return out


class ExitMini(ScriptedMini):
    """The scripted client with a live microphone and a recording speaker."""

    def __init__(self) -> None:
        super().__init__()
        self.mic = LiveMic()
        self.pushes: list[tuple[float, int]] = []
        self.clears: list[float] = []
        media = self.mini.media
        media.get_audio_sample.side_effect = self.mic.read
        media.push_audio_sample.side_effect = lambda x: self.pushes.append(
            (time.monotonic(), len(x))
        )
        media.audio = SimpleNamespace(clear_player=lambda: self.clears.append(time.monotonic()))


class DelayedTTS:
    """TTS stand-in: a first-chunk delay, then a quiet tone of about 300 ms per word, 22.05 kHz."""

    name = "fake"
    controls = frozenset()

    def __init__(self, spoken: list[tuple[str, str]]) -> None:
        self.spoken = spoken

    def available(self, *, minor_present=False) -> bool:
        return True

    async def synthesize(self, text, delivery, *, language, voice, minor_present=False):
        self.spoken.append((language, text))
        await asyncio.sleep(FAKE_TTS_FIRST_CHUNK_S)
        n = round(22050 * 0.3 * max(1, len(text.split())))
        yield AudioChunk((0.05 * np.sin(np.arange(n) * 0.1)).astype(np.float32), 22050)

    def cancel(self) -> None:
        pass

    async def warm(self, language: str, voice: VoiceConfig) -> None:
        pass

    async def aclose(self) -> None:
        pass


def sse(text: str) -> bytes:
    chunk = {
        "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }  # fmt: skip
    return f"data: {json.dumps(chunk)}\n\n".encode()


class ExitApp(app_module.RoboEmotApp):
    """The real app with the LLM route mocked in-process and the utterances collected."""

    minor_flag = False
    requests: list
    utterances: list

    @property
    def minor_present(self) -> bool:
        return self.minor_flag

    def on_utterance(self, utterance) -> None:
        self.utterances.append((time.monotonic(), utterance))
        super().on_utterance(utterance)

    def make_router(self, cfg, monitor):
        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            system = body["messages"][0]["content"]
            reply = REPLIES["Italian" if "reply in Italian" in system else "English"]
            await asyncio.sleep(FIRST_TOKEN_S)

            async def stream():
                words = reply.split(" ")
                for i in range(0, len(words), 2):
                    yield sse(" ".join(words[i : i + 2]) + " ")
                    await asyncio.sleep(TOKEN_GAP_S)
                yield b"data: [DONE]\n\n"

            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=stream()
            )

        return Router(
            cfg.llm_routes,
            cfg.safety,
            monitor=monitor,
            http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )


@pytest.fixture(scope="module")
def models_ready():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("vosk")
    try:
        voice_models.model_file(MODELS_DIR, "silero-vad")
        for name in RoboEmotConfig().voice.asr.models.values():
            voice_models.model_file(MODELS_DIR, name)
    except ModelError:
        pytest.skip("voice models not fetched: run `roboemot models fetch --voice`")


@pytest.fixture()
def real_piper():
    if not PIPER_PYTHON.exists():
        return None
    return PIPER_PYTHON


def yaml_for(language: str, piper: Path | None) -> str:
    lines = [
        "voice:",
        "  enabled: true",
        f"  models_dir: {MODELS_DIR}",
        *([f"  piper:\n    python: {piper}"] if piper else []),
        f"conversation:\n  language: {language}",
        "llm_routes:",
        "  - {name: lan, base_url: 'http://lan.local/v1', model: m1}",
    ]
    return "\n".join(lines) + "\n"


def start_app(data, monkeypatch, language, piper, minor=False, minors_llm=False):  # noqa: F811
    spoken: list[tuple[str, str]] = []
    # The route is an in-process mock, but the app's network monitor probes "lan.local" over TCP.
    # Where that name fails fast (CI) the link goes offline after three probes and the robot
    # replies nonverbally; where it hangs (macOS mDNS) it does not, which hid the dependency.
    monkeypatch.setattr(
        app_module,
        "NetworkMonitor",
        functools.partial(app_module.NetworkMonitor, probe=lambda host, port, timeout_s: True),
    )
    if piper is None:
        monkeypatch.setattr(factory, "build_backend", lambda cfg, router: DelayedTTS(spoken))
    else:
        from roboemot.voice import piper_backend

        real = piper_backend.PiperBackend.synthesize

        def spy(self, text, delivery, *, language, **kw):
            spoken.append((language, text))
            return real(self, text, delivery, language=language, **kw)

        monkeypatch.setattr(piper_backend.PiperBackend, "synthesize", spy)
    if minors_llm:  # the master switch on, the route still not cleared for minors
        text = (data / "app.yaml").read_text()
        (data / "app.yaml").write_text(
            text.replace("safety:\n", "safety:\n  minors_allowed_llm: true\n")
        )
    enable(data, yaml_for(language, piper))
    app = ExitApp.__new__(ExitApp)
    app.requests, app.utterances, app.minor_flag = [], [], minor
    mini = ExitMini()
    stop, thread, errors = run_app(app, mini.mini)
    try:
        assert wait_for(lambda: app.conversation is not None and app.voice_listener is not None, 30)
        assert wait_for(lambda: app.voice_listener.snapshot()["state"] == "running", 60)
        if piper is not None:  # the worker of the default language is started by the warm-up
            backend = app.speech_output.backend
            assert wait_for(lambda: language in backend.snapshot()["languages"], 60)
    except BaseException:
        stop.set()
        thread.join(timeout=20)
        raise
    mini.mic.start()
    return app, mini, spoken, stop, thread, errors


class GcPauses:
    """Longest collector pause per generation while the app ran (a secondary cause of the flake)."""

    def __init__(self) -> None:
        self.max_ms = {0: 0.0, 1: 0.0, 2: 0.0}
        self._t0 = 0.0

    def __call__(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t0 = time.perf_counter()
        else:
            gen = info["generation"]
            self.max_ms[gen] = max(self.max_ms[gen], (time.perf_counter() - self._t0) * 1000)


def finish(stop, thread, errors) -> None:
    stop.set()
    thread.join(timeout=20)
    assert not errors and not thread.is_alive()


def report(name: str, result: dict, capsys) -> None:
    RESULTS[name] = result
    with capsys.disabled():
        print(f"\n[m3 exit] {name}: {json.dumps(result, indent=1, default=str)}")
    path = os.environ.get("ROBOEMOT_EXIT_REPORT")
    if path:
        Path(path).write_text(json.dumps(RESULTS, indent=1, default=str))


@pytest.mark.parametrize("language,wake,stop_clip,words", CASES)
def test_conversation_inside_the_latency_budgets_then_a_barge_in(
    models_ready,
    data,  # noqa: F811
    monkeypatch,
    real_piper,
    capsys,
    language,
    wake,
    stop_clip,
    words,
):
    pauses = GcPauses()
    gc.callbacks.append(pauses)
    app, mini, spoken, stop, thread, errors = start_app(data, monkeypatch, language, real_piper)
    try:
        clip = load(wake)
        _, last = speech_span_s(clip)
        t_clip = mini.mic.play(np.concatenate([np.zeros(round(LEAD_S * RATE), np.float32), clip]))
        speech_end_wall = t_clip + LEAD_S + last  # start of the last loud frame, mic clock
        assert wait_for(lambda: bool(app.utterances), 20), "no utterance"
        u0 = app.utterances[0][1]
        assert wait_for(lambda: mini.pushes, 20), (
            f"the robot never answered: language={u0.language} final={u0.final} "
            f"addressed={u0.addressed} keywords={sorted(u0.keywords)} "
            f"requests={len(app.requests)} spoken={len(spoken)} "
            f"conversation={app.conversation.snapshot()} "
            f"network={app.diagnostics.report()['sections'].get('roboemot.network')}"
        )
        first_push = mini.pushes[0][0]
        assert wait_for(lambda: "latency_ms" in app.conversation.snapshot(), 5)
        assert wait_for(lambda: app.speaker.queued_s() > 0.5, 20)

        _, u = app.utterances[0]
        assert words <= set(u.text.split()), "transcript lost a key word"
        assert u.language == language and u.final and u.addressed
        assert "greeting" in u.keywords, "the greeting keyword was not spotted"
        latency = app.voice_listener.snapshot()["latency"]
        assert "keyword_vs_last_speech" in latency, "the fast path never saw the keyword"
        reaction_ms = latency["reaction_vs_last_speech"]["last_ms"]
        keyword_ms = latency["keyword_vs_last_speech"]["last_ms"]
        conv = app.conversation.snapshot()["latency_ms"]
        first_audio_ms = conv["first_audio"]
        independent_ms = (first_push - speech_end_wall) * 1000
        pushes_before, queued_at_cut = len(mini.pushes), app.speaker.queued_s()
        stop_audio = load(stop_clip)
        assert not mini.clears, "clear_player was called before the stop clip"
        onset_wall = mini.mic.play(stop_audio)
        acoustic_onset = onset_wall + speech_span_s(stop_audio)[0]
        assert wait_for(lambda: bool(mini.clears), 5), "barge-in never stopped the speaker"
        stop_acoustic_ms = (mini.clears[0] - acoustic_onset) * 1000
        bargein_snap = app.bargein.snapshot()
        triggers, latency_bi = bargein_snap["triggers"], bargein_snap["latency"]
        assert stop_acoustic_ms >= 0, "the speaker was cleared before the user spoke"
        time.sleep(0.3)
        assert app.speaker.queued_s() == 0.0
        late = [t for t, _ in mini.pushes[pushes_before:] if t > mini.clears[0]]
        assert not late, "audio pushed after the stop"

        report(
            language,
            {
                "tts": "piper (real)"
                if real_piper
                else f"fake, {FAKE_TTS_FIRST_CHUNK_S} s first chunk",
                "llm_first_token_s": FIRST_TOKEN_S,
                "transcript_words_ok": sorted(words),
                "fast_path_reaction_vs_last_speech_ms": round(reaction_ms),
                "fast_path_keyword_vs_last_speech_ms": round(keyword_ms),
                "first_audio_vs_last_speech_ms": {k: round(v) for k, v in conv.items() if v},
                "first_audio_push_vs_mic_last_speech_ms": round(independent_ms),
                "max_first_token_s_inside_budget": round(
                    FIRST_TOKEN_S + (FIRST_AUDIO_BUDGET_MS - first_audio_ms) / 1000, 2
                ),
                "barge_in_stop_after_acoustic_onset_ms": round(stop_acoustic_ms),
                "barge_in_trigger": "vad" if triggers else "keyword",
                "barge_in_detect_to_stop_ms": round(latency_bi["detect_to_stop"]["last_ms"], 1),
                "barge_in_stop_call_ms": round(latency_bi["stop_call"]["last_ms"], 1),
                "gc_max_pause_ms": {f"gen{g}": round(v, 1) for g, v in pauses.max_ms.items()},
                "answer_audio_queued_when_cut_s": round(queued_at_cut, 1),
            },
            capsys,
        )
        assert reaction_ms <= FAST_PATH_BUDGET_MS, "fast-path reaction over budget"
        assert first_audio_ms <= FIRST_AUDIO_BUDGET_MS, "first audio over the cm4 budget"
        assert independent_ms <= FIRST_AUDIO_BUDGET_MS, "first audio over budget (mic clock)"
        assert len(app.requests) == 1
        assert (
            "reply in " + {"it": "Italian", "en": "English"}[language]
            in (app.requests[0]["messages"][0]["content"])
        )
        assert spoken and {lang for lang, _ in spoken} == {language}, "answer in the wrong language"

        assert stop_acoustic_ms <= BARGE_IN_BUDGET_MS, (
            f"barge-in stop {stop_acoustic_ms:.0f} ms after the first sound; "
            f"barge-in state {app.bargein.snapshot()}"
        )
    finally:
        gc.callbacks.remove(pauses)
        finish(stop, thread, errors)


@pytest.mark.parametrize("minors_llm", [False, True], ids=["master_switch_off", "master_switch_on"])
def test_a_minor_without_a_cleared_route_gets_a_built_in_line_and_no_request(
    models_ready,
    data,  # noqa: F811
    monkeypatch,
    capsys,
    minors_llm,
):
    """No route has ``allowed_for_minors``: with ``safety.minors_allowed_llm`` off (the default) or
    on, a minor hears the built-in line and no request goes out (SPEC 8, 12)."""
    app, mini, spoken, stop, thread, errors = start_app(
        data, monkeypatch, "it", None, minor=True, minors_llm=minors_llm
    )
    try:
        assert app.conversation is not None
        clip = load("it_greeting_name")
        _, last = speech_span_s(clip)
        t_clip = mini.mic.play(np.concatenate([np.zeros(round(LEAD_S * RATE), np.float32), clip]))
        end_wall = t_clip + LEAD_S + last
        assert wait_for(lambda: bool(mini.pushes), 20), "no built-in line was spoken"
        ms = (mini.pushes[0][0] - end_wall) * 1000
        assert app.requests == [], "a request went out while a minor was present"
        assert spoken == [("it", RoboEmotConfig().safety.minor_lines["it"].greeting)]
        assert ms <= FIRST_AUDIO_BUDGET_MS
        report(
            f"child_it_minors_llm_{minors_llm}",
            {"tts": f"fake, {FAKE_TTS_FIRST_CHUNK_S} s first chunk", "requests": 0,
             "first_audio_push_vs_mic_last_speech_ms": round(ms)},
            capsys,
        )  # fmt: skip
    finally:
        finish(stop, thread, errors)


@pytest.mark.parametrize("language,stop_clip", [("it", "it_stop"), ("en", "en_stop")])
def test_a_keyword_final_utterance_reacts_inside_300_ms_of_the_last_speech_frame(
    models_ready,
    data,  # noqa: F811
    monkeypatch,
    capsys,
    language,
    stop_clip,
):
    """The whole utterance is the keyword, so the reaction cannot come mid-sentence: it is measured
    from the last speech frame and must be at most the SPEC 13 budget (negative = before it)."""
    app, mini, spoken, stop, thread, errors = start_app(data, monkeypatch, language, None)
    try:
        clip = load(stop_clip)
        mini.mic.play(np.concatenate([np.zeros(round(LEAD_S * RATE), np.float32), clip]))
        assert wait_for(lambda: bool(app.utterances), 20), "no utterance"
        u = app.utterances[0][1]
        assert "stop" in u.keywords and u.language == language and u.final
        latency = app.voice_listener.snapshot()["latency"]
        reaction_ms = latency["reaction_vs_last_speech"]["last_ms"]
        keyword_ms = latency["keyword_vs_last_speech"]["last_ms"]
        report(
            f"keyword_final_{language}",
            {"reaction_vs_last_speech_ms": round(reaction_ms),
             "keyword_vs_last_speech_ms": round(keyword_ms)},
            capsys,
        )  # fmt: skip
        assert reaction_ms <= FAST_PATH_BUDGET_MS, "fast-path reaction over budget"
        assert not app.requests, "a bare stop phrase must not be sent to the route"
    finally:
        finish(stop, thread, errors)
