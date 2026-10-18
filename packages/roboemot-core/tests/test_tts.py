"""SPEC 8, stage 7: sentence splitter, Piper worker protocol, OpenAI speech backend, fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from roboemot.config.schema import (
    LLMRoute,
    OpenAISpeechConfig,
    PiperConfig,
    RoboEmotConfig,
    SafetyConfig,
)
from roboemot.llm import Router
from roboemot.voice.models import ModelError
from roboemot.voice.openai_speech import OPENAI_PCM_RATE_HZ, OpenAISpeechBackend
from roboemot.voice.piper_backend import PiperBackend
from roboemot.voice.prosody import Delivery
from roboemot.voice.tts import FallbackTTS, SentenceSplitter, TTSError, TTSUnavailable

FAKE_WORKER = Path(__file__).with_name("fake_piper_worker.py")
RATE = 22050
CFG = RoboEmotConfig()
VOICE = CFG.voice


def delivery(**kw) -> Delivery:
    base = {"rate": 1.0, "pitch_st": 0.0, "volume_db": 0.0, "variability": 0.5, "pause_scale": 1.0}
    return Delivery(**{**base, **kw})


# ---------------------------------------------------------------- splitter


def split(chunks, max_chars=200, **kw):
    s = SentenceSplitter(max_chars, **kw)
    out = []
    for c in chunks:
        out += s.feed(c)
    out += s.flush()
    return [(x.text, x.mood) for x in out]


def test_splitter_emits_each_sentence_as_soon_as_it_is_complete():
    s = SentenceSplitter(200)
    assert s.feed("Ciao!") == [] and s.feed(" ") == []  # needs the look-ahead after the stop
    assert [x.text for x in s.feed("Come stai? Be")] == ["Ciao!", "Come stai?"]
    got = s.feed("ne")
    assert got == [] and [x.text for x in s.flush()] == ["Bene"]


def test_splitter_boundaries_across_arbitrary_delta_cuts():
    text = "Prima frase. Seconda frase!\nTerza? Quarta…  Quinta."
    one = split([text])
    assert [t for t, _ in one] == ["Prima frase.", "Seconda frase!", "Terza?", "Quarta…", "Quinta."]
    assert split(list(text)) == one  # char by char gives the same sentences


def test_splitter_keeps_decimals_initials_and_abbreviations_whole():
    got = split(["Costa 3.5 euro. Il sig. Rossi e il dott. Bianchi, A. Verdi, arrivano. Ok"])
    assert [t for t, _ in got] == [
        "Costa 3.5 euro.",
        "Il sig. Rossi e il dott. Bianchi, A. Verdi, arrivano.",
        "Ok",
    ]
    assert [t for t, _ in split(["Mr. Smith left. he came back"])] == [
        "Mr. Smith left. he came back"  # a dot before a lower-case word is not an end
    ]


def test_splitter_flushes_long_text_at_the_last_comma_else_space():
    got = split(["uno due tre, quattro cinque sei sette otto"], max_chars=30)
    assert [t for t, _ in got] == ["uno due tre,", "quattro cinque sei sette otto"]
    got = split(["aaaa bbbb cccc dddd eeee ffff"], max_chars=12)
    assert all(len(t) <= 12 for t, _ in got)
    assert " ".join(t for t, _ in got) == "aaaa bbbb cccc dddd eeee ffff"
    assert [t for t, _ in split(["x" * 25], max_chars=10)] == ["x" * 10, "x" * 10, "x" * 5]


def test_splitter_strips_mood_tags_even_when_split_across_deltas():
    got = split(
        ["[mood=joy] Che bel", "lo! [mo", "od=curiosity] Cos'è? [mood=sad"], strip_tags=True
    )
    assert got == [("Che bello!", "joy"), ("Cos'è?", "curiosity")]  # dangling tag dropped
    assert split(["[x] [mood=joy]Ciao [mood=sadness]addio"], strip_tags=True) == [
        ("[x]", None),
        ("Ciao", "joy"),
        ("addio", "sadness"),
    ]
    assert split(["[mood=joy] Ciao"], strip_tags=False) == [("[mood=joy] Ciao", None)]


def test_splitter_drops_fragments_without_letters():
    assert split(["... ?! Sì."]) == [("Sì.", None)]


# ---------------------------------------------------------------- piper worker


def make_voice(tmp_path: Path, name="v", rate=RATE) -> Path:
    voices = tmp_path / "voices"
    voices.mkdir(exist_ok=True)
    (voices / f"{name}.onnx").write_bytes(b"")
    (voices / f"{name}.onnx.json").write_text(json.dumps({"audio": {"sample_rate": rate}}))
    return voices


def resolver(voices: Path):
    def resolve(name: str):
        model = voices / f"{name}.onnx"
        if not model.exists():
            raise ModelError(f"{name}: not verified")
        return model, Path(f"{model}.json")

    return resolve


def voice_cfg(**voices):
    return VOICE.model_copy(update={"voices": voices or {"it": "v"}})


def piper(tmp_path, *, rate=RATE, **cfg) -> PiperBackend:
    cfg = {"request_timeout_s": 5.0, "restart_backoff_s": (0.05, 0.2), "chunk_ms": 20.0} | cfg
    return PiperBackend(
        PiperConfig(**cfg), sys.executable, resolver(make_voice(tmp_path, rate=rate)),
        script=FAKE_WORKER,
    )  # fmt: skip


async def collect(backend, text, d=None, language="it", voice=None):
    out = []
    async for c in backend.synthesize(
        text, d or delivery(), language=language, voice=voice or voice_cfg()
    ):
        out.append(c)
    return out


async def test_piper_rate_is_read_from_the_voice_json_and_audio_is_float32(tmp_path):
    b = piper(tmp_path, rate=16000)
    try:
        chunks = await collect(b, "ciao")
        assert {c.rate_hz for c in chunks} == {16000}
        assert all(c.samples.dtype == np.float32 and c.samples.ndim == 1 for c in chunks)
        assert len(chunks) > 1  # chunk_ms 20 splits the 0.1 s tone into pieces
        assert sum(c.samples.size for c in chunks) == pytest.approx(1600, abs=2)
        assert 0.1 < np.abs(np.concatenate([c.samples for c in chunks])).max() < 0.5
    finally:
        await b.aclose()


async def test_piper_one_persistent_process_serves_many_sentences(tmp_path):
    b = piper(tmp_path)
    try:
        await collect(b, "uno")
        pid = b._workers["it"].proc.pid
        for text in ("due", "tre", "quattro"):
            await collect(b, text)
        assert b._workers["it"].proc.pid == pid
        assert b.restarts == 0
    finally:
        await b.aclose()


async def test_piper_request_carries_the_delivery_mapping(tmp_path):
    b = piper(tmp_path)
    try:
        slow = await collect(b, "ciao", delivery(rate=0.5))  # length_scale 2 -> 0.2 s
        loud = await collect(b, "ciao", delivery(volume_db=-20.0))  # volume 0.1
        assert sum(c.samples.size for c in slow) == pytest.approx(RATE * 0.2, abs=4)
        assert np.abs(np.concatenate([c.samples for c in loud])).max() < 0.05
    finally:
        await b.aclose()


async def test_piper_pitch_shift_declares_a_higher_rate_and_lengthens_the_sentence(tmp_path):
    b = piper(tmp_path)
    try:
        chunks = await collect(b, "ciao", delivery(pitch_st=12.0))
        assert {c.rate_hz for c in chunks} == {2 * RATE}
        # synthesised twice as long, so playing it at 2x rate keeps the duration
        assert sum(c.samples.size for c in chunks) == pytest.approx(RATE * 0.2, abs=4)
        none = voice_cfg().model_copy(update={"pitch_shift_backend": "none"})
        plain = await collect(b, "ciao", delivery(pitch_st=12.0), voice=none)
        assert {c.rate_hz for c in plain} == {RATE}
    finally:
        await b.aclose()


async def test_piper_restarts_after_a_crash_and_retries_the_sentence(tmp_path):
    b = piper(tmp_path)
    try:
        await collect(b, "ciao")
        first = b._workers["it"].proc.pid
        chunks = await collect(b, "CRASHONCE")  # dies before any audio: retried on a new worker
        assert chunks and b.restarts == 1
        assert b._workers["it"].proc.pid != first
    finally:
        await b.aclose()


async def test_piper_crash_mid_request_raises_then_backoff_then_recovers(tmp_path):
    b = piper(tmp_path)
    try:
        with pytest.raises(TTSError):
            await collect(b, "CRASH")
        with pytest.raises(TTSUnavailable):  # inside the backoff window: fail fast
            await collect(b, "ciao")
        await asyncio.sleep(0.25)
        assert await collect(b, "ciao")
    finally:
        await b.aclose()


async def test_piper_worker_error_is_reported_by_class_only(tmp_path, caplog):
    b = piper(tmp_path)
    try:
        with pytest.raises(TTSError, match="ValueError") as err:
            await collect(b, "BOOM segreto")
        assert "segreto" not in str(err.value)
        assert await collect(b, "ciao")  # the same worker keeps serving
    finally:
        await b.aclose()


async def test_piper_cancel_mid_sentence_stops_forwarding_and_the_next_sentence_flows(tmp_path):
    b = piper(tmp_path)
    try:
        gen = b.synthesize("SLOW", delivery(), language="it", voice=voice_cfg())
        first = await gen.__anext__()
        assert first.samples.size
        t0 = time.monotonic()
        await gen.aclose()  # the task cancellation path: finally sends the cancel
        pid = b._workers["it"].proc.pid
        chunks = await collect(b, "ciao")
        assert chunks and b._workers["it"].proc.pid == pid
        # the cancelled one stopped after at most the piece in flight: well under 6 * 0.15 s
        assert time.monotonic() - t0 < 0.8
    finally:
        await b.aclose()


async def test_piper_backend_cancel_ends_waiting_generators(tmp_path):
    b = piper(tmp_path)
    try:
        gen = b.synthesize("SLOW", delivery(), language="it", voice=voice_cfg())
        await gen.__anext__()
        b.cancel()
        rest = [c async for c in gen]
        assert len(rest) < 6
    finally:
        await b.aclose()


async def test_piper_stderr_flood_does_not_block(tmp_path):
    b = piper(tmp_path)
    try:
        chunks = await asyncio.wait_for(collect(b, "FLOOD"), 10)
        assert chunks
        await asyncio.sleep(0.3)
        assert b.snapshot()["stderr_bytes"] >= 3_000_000  # drained, not just not blocked
    finally:
        await b.aclose()


async def test_piper_missing_voice_and_language_are_tts_errors(tmp_path):
    b = piper(tmp_path)
    with pytest.raises(TTSError, match="no voice"):
        await collect(b, "ciao", language="fr")
    with pytest.raises(TTSUnavailable, match="not available"):  # unverified or half-downloaded
        await collect(b, "ciao", voice=voice_cfg(it="absent"))


async def test_piper_unstartable_interpreter_is_a_tts_error(tmp_path):
    b = PiperBackend(
        PiperConfig(), tmp_path / "no-python", resolver(make_voice(tmp_path)), script=FAKE_WORKER
    )
    with pytest.raises(TTSError, match="cannot start"):
        await collect(b, "ciao")


async def test_piper_logs_never_contain_the_text(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    b = piper(tmp_path)
    try:
        await collect(b, "frase riservata")
        with pytest.raises(TTSError):
            await collect(b, "BOOM frase riservata")
    finally:
        await b.aclose()
    assert "riservata" not in caplog.text


# ---------------------------------------------------------------- openai speech


def pcm(n=480, value=16384) -> bytes:
    return np.full(n, value, dtype="<i2").tobytes()


def route(**kw) -> LLMRoute:
    base = {"name": "tts", "base_url": "https://speech.example/v1", "model": "m", "api_key": "k"}
    return LLMRoute(**(base | kw))


class SpeechServer:
    def __init__(self, handler=None) -> None:
        self.requests: list[httpx.Request] = []
        self.handler = handler or (lambda req: httpx.Response(200, content=pcm()))

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        result = self.handler(request)
        return await result if asyncio.iscoroutine(result) else result


def openai_backend(server, *routes, safety=None, secrets=None, **cfg):
    router = Router(
        list(routes or [route()]),
        safety or SafetyConfig(),
        secrets=secrets or (lambda name: ""),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )
    config = OpenAISpeechConfig(**({"route": "tts"} | cfg))
    return OpenAISpeechBackend(config, lambda: router), router


async def test_openai_pcm_is_converted_and_the_request_is_shaped(tmp_path):
    server = SpeechServer(lambda r: httpx.Response(200, content=pcm(480, 16384) + b"\x00"))
    backend, router = openai_backend(server)
    chunks = await collect(backend, "Ciao!", delivery(rate=0.9, volume_db=-3.0))
    assert [c.rate_hz for c in chunks] == [OPENAI_PCM_RATE_HZ]
    assert chunks[0].samples.dtype == np.float32 and chunks[0].samples.size == 480
    assert chunks[0].samples[0] == pytest.approx(0.5)
    req = server.requests[0]
    assert str(req.url) == "https://speech.example/v1/audio/speech"
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["response_format"] == "pcm" and body["input"] == "Ciao!"
    assert body["model"] == "gpt-4o-mini-tts" and body["voice"] == "marin"
    assert "slower" in body["instructions"] and "soft" in body["instructions"]
    assert router.used("tts") == 2  # 5 chars at 4 per token, charged against the route budget


async def test_openai_instructions_only_on_instructable_models():
    server = SpeechServer()
    backend, _ = openai_backend(server, instructable=False)
    await collect(backend, "Ciao", delivery(rate=0.9))
    assert "instructions" not in json.loads(server.requests[0].content)
    assert backend.controls == frozenset()


async def test_openai_is_refused_while_a_minor_is_present_unless_the_route_is_cleared():
    server = SpeechServer()
    backend, _ = openai_backend(server)
    assert backend.available(minor_present=False)
    assert not backend.available(minor_present=True)
    with pytest.raises(TTSUnavailable):
        async for _ in backend.synthesize(
            "x", delivery(), language="it", voice=VOICE, minor_present=True
        ):
            pass
    assert server.requests == []
    allowed = SafetyConfig(minors_allowed_llm=True)
    cleared, _ = openai_backend(server, route(allowed_for_minors=True), safety=allowed)
    assert cleared.available(minor_present=True)
    uncleared, _ = openai_backend(server, safety=allowed)
    assert not uncleared.available(minor_present=True)


async def test_openai_unknown_route_and_spent_budget_are_unavailable():
    backend, _ = openai_backend(SpeechServer(), route(daily_token_budget=0))
    assert not backend.available()
    missing, _ = openai_backend(SpeechServer(), route(), route="other")
    assert not missing.available()


async def test_openai_errors_raise_tts_error_and_never_log_the_text(caplog):
    caplog.set_level(logging.DEBUG)
    for handler in (
        lambda r: httpx.Response(500, json={"error": "frase riservata"}),
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")),
        lambda r: httpx.Response(200, content=b""),
    ):
        backend, _ = openai_backend(SpeechServer(handler))
        with pytest.raises(TTSError) as err:
            await collect(backend, "frase riservata")
        assert "riservata" not in str(err.value)
    assert "riservata" not in caplog.text


async def test_openai_first_chunk_timeout_uses_the_route_rule():
    async def slow(request):
        await asyncio.sleep(2)
        return httpx.Response(200, content=pcm())

    backend, _ = openai_backend(SpeechServer(slow), route(first_token_timeout_s=0.1))
    t0 = time.monotonic()
    with pytest.raises(TTSError, match="timed out"):
        await collect(backend, "ciao")
    assert time.monotonic() - t0 < 1.0


# ---------------------------------------------------------------- fallback


async def test_fallback_to_piper_when_the_remote_route_fails_or_is_not_eligible(tmp_path):
    local = piper(tmp_path)
    try:
        failing, _ = openai_backend(SpeechServer(lambda r: httpx.Response(503)))
        tts = FallbackTTS(failing, local)
        chunks = await collect(tts, "ciao")
        assert chunks[0].rate_hz == RATE and tts.last_used == "piper"
        # minor present, route not cleared: the remote is never contacted
        server = SpeechServer()
        remote, _ = openai_backend(server)
        chunks = await collect(FallbackTTS(remote, local), "ciao", language="it")
        assert chunks and chunks[0].rate_hz == OPENAI_PCM_RATE_HZ  # adult: remote is used
        tts = FallbackTTS(remote, local)
        got = []
        async for c in tts.synthesize(
            "ciao", delivery(), language="it", voice=voice_cfg(), minor_present=True
        ):
            got.append(c)
        assert got[0].rate_hz == RATE and len(server.requests) == 1
    finally:
        await local.aclose()


async def test_fallback_does_not_switch_voice_after_audio_started(tmp_path):
    def broken_stream(request):
        async def body():
            yield pcm()
            raise httpx.ReadError("cut")

        return httpx.Response(200, content=body())

    local = piper(tmp_path)
    try:
        remote, _ = openai_backend(SpeechServer(broken_stream))
        got = []
        with pytest.raises(TTSError):
            async for c in FallbackTTS(remote, local).synthesize(
                "ciao", delivery(), language="it", voice=voice_cfg()
            ):
                got.append(c)
        assert got and got[0].rate_hz == OPENAI_PCM_RATE_HZ
    finally:
        await local.aclose()


async def test_fallback_with_nothing_available_raises_unavailable():
    remote, _ = openai_backend(SpeechServer())
    with pytest.raises(TTSUnavailable):
        async for _ in FallbackTTS(remote, None).synthesize(
            "x", delivery(), language="it", voice=VOICE, minor_present=True
        ):
            pass


# ---------------------------------------------------------------- protocol hardening

ROGUE = Path(__file__).with_name("rogue_piper_worker.py")


@pytest.mark.parametrize(
    "mode",
    ["rate_zero", "rate_text", "rate_huge", "unknown_kind", "oversize", "odd", "big_error"],
)
async def test_piper_protocol_violations_kill_and_reap_the_worker(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("ROGUE_MODE", mode)
    procs = []
    real = asyncio.create_subprocess_exec

    async def spy(*a, **k):
        procs.append(await real(*a, **k))
        return procs[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    b = PiperBackend(
        PiperConfig(request_timeout_s=5.0, startup_timeout_s=5.0, chunk_ms=20.0),
        sys.executable,
        resolver(make_voice(tmp_path)),
        script=ROGUE,
    )
    try:
        with pytest.raises(TTSError):  # FallbackTTS treats it as a failure before audio
            await asyncio.wait_for(collect(b, "ciao"), 10)
        assert "it" not in b._workers
        for _ in range(100):
            if procs[0].returncode is not None:
                break
            await asyncio.sleep(0.05)
        assert procs[0].returncode is not None  # killed and reaped, no zombie
    finally:
        await b.aclose()


async def test_piper_timeout_kills_reaps_and_drops_the_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("ROGUE_MODE", "silent")
    procs = []
    real = asyncio.create_subprocess_exec

    async def spy(*a, **k):
        procs.append(await real(*a, **k))
        return procs[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    b = PiperBackend(
        PiperConfig(request_timeout_s=0.5, startup_timeout_s=5.0),
        sys.executable,
        resolver(make_voice(tmp_path)),
        script=ROGUE,
    )
    try:
        with pytest.raises(TTSError, match="timed out"):
            await collect(b, "ciao")
        assert "it" not in b._workers and procs[0].returncode is not None
    finally:
        await b.aclose()


async def test_openai_charges_only_requests_the_server_accepted():
    backend, router = openai_backend(
        SpeechServer(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x")))
    )
    with pytest.raises(TTSError):
        await collect(backend, "frase")
    assert router.used("tts") == 0


async def test_openai_secret_errors_are_unavailable_not_crashes():
    def secrets(name):
        raise KeyError(name)

    backend, _ = openai_backend(SpeechServer(), route(api_key="${secret:k}"), secrets=secrets)
    with pytest.raises(TTSUnavailable):
        await collect(backend, "ciao")


async def test_openai_uses_the_router_the_app_has_now_and_its_http_client():
    server = SpeechServer()
    first, router_a = openai_backend(server)
    _, router_b = openai_backend(server, route(base_url="https://other.example/v1"))
    current = {"router": router_a}
    backend = OpenAISpeechBackend(OpenAISpeechConfig(route="tts"), lambda: current["router"])
    await collect(backend, "uno")
    current["router"] = router_b  # the app rebuilt its router after a config reload
    await collect(backend, "due")
    assert [r.url.host for r in server.requests] == ["speech.example", "other.example"]
    assert router_b.used("tts") == 1 and router_a.used("tts") == 1


def test_router_does_not_hand_out_the_key():
    router = Router([route(api_key="${secret:k}")], SafetyConfig(), secrets=lambda n: "sek")
    assert router.auth_headers(router.routes[0]) == {"Authorization": "Bearer sek"}
    assert not hasattr(router, "api_key")
    assert Router([route(api_key="none")], SafetyConfig()).auth_headers(route(api_key="none")) == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ma no. Poi vado a casa.", ["Ma no.", "Poi vado a casa."]),
        ("Ho detto di no. Poi vado.", ["Ho detto di no.", "Poi vado."]),
        ("Sig.ra Rossi è qui.", ["Sig.ra Rossi è qui."]),
        ("Il dott. Bianchi arriva. Ciao.", ["Il dott. Bianchi arriva.", "Ciao."]),
        ("I love art. Then we go.", ["I love art.", "Then we go."]),
    ],
)
def test_splitter_abbreviations_do_not_swallow_ordinary_words(text, expected):
    assert [t for t, _ in split([text])] == expected
