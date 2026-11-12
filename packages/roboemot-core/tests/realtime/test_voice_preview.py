"""The voice preview of a realtime route (SPEC 8, Voice preview; SPEC 12) against the TLS fake
Gemini server and the Pollen fake server.

The WAV's rate is the session's, a tool call before the audio is answered, no audio and a 1008
refusal fail by code, the audio cap holds, the voice override never changes the stored route,
the sentence follows ``conversation.language`` (English when it has no file), and no key, audio
or reply words reach the logs or the error.
"""

from __future__ import annotations

import io
import json
import logging
import time
import wave
from types import SimpleNamespace

import pytest
from fake_server import REPLY_TEXT, FakeServer
from gemini_server import OUTPUT_RATE_HZ, REFUSAL, REPLY
from realtime_helpers import KEY
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.conformance import RouteBudgetExhausted, RouteKeyUnavailable
from roboemot.llm.prompt import preview_line
from roboemot.llm.router import Router
from roboemot.realtime import preview, protocol
from roboemot.realtime.session import DirectSession
from roboemot.realtime.turn import TONE_NOTED

POLICY = 1008
NATIVE_RATE_HZ = protocol.NATIVE_OUTPUT_RATE_HZ
SAID_RATE_HZ = 22050  # a Pollen session that names its own output rate


def gemini_route(base_url: str, **kw: object) -> LLMRoute:
    return LLMRoute(
        name="gemini",
        base_url=base_url,
        model="gemini-3.8-live",
        kind="realtime",
        protocol="gemini_live",
        api_key="${secret:gemini_key}",
        **{"locality": "local", "voice": "Puck", **kw},
    )


def pollen_route(base_url: str, **kw: object) -> LLMRoute:
    kw.setdefault("locality", "local")
    return LLMRoute(
        name="rt", base_url=base_url, model="s2s", kind="realtime", api_key="none", **kw
    )


def make_cfg(route: LLMRoute, *, language: str = "it", **rt: object) -> RoboEmotConfig:
    return RoboEmotConfig(
        llm_routes=[route],
        conversation={"language": language},
        voice={"realtime_route": {"mode": "direct", **rt}},
    )


def make_router(cfg: RoboEmotConfig) -> Router:
    return Router(cfg.llm_routes, cfg.safety, secrets=lambda _name: KEY)


def run(cfg: RoboEmotConfig, voice: str = "Kore", router: Router | None = None, **kw) -> bytes:
    return preview.preview_voice(
        cfg.llm_routes[0],
        voice,
        cfg,
        router=router or make_router(cfg),
        provider=DirectSession(),
        **kw,
    )


def wav_of(data: bytes) -> tuple[int, int, int, bytes]:
    with wave.open(io.BytesIO(data)) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())


def line_sent(srv) -> str:
    """The text of the user turn the preview's ``clientContent`` carried."""
    turns = [m["clientContent"]["turns"] for m in srv.received if "clientContent" in m]
    assert len(turns) == 1, srv.received
    return turns[0][-1]["parts"][0]["text"]


def test_a_gemini_preview_is_a_24k_wav_in_the_chosen_voice_and_closes_its_session(gemini, caplog):
    caplog.set_level(logging.DEBUG)
    srv = gemini([["audio"]], chunks=4)
    cfg = make_cfg(gemini_route(srv.base_url()))
    router = make_router(cfg)
    stored = cfg.llm_routes[0].model_dump()
    ready: list[bytes] = []
    data = run(cfg, "Kore", router, ready=ready.append)
    rate, channels, width, pcm = wav_of(data)
    assert (rate, channels, width) == (OUTPUT_RATE_HZ, 1, 2) and len(pcm) > 0
    assert ready == [data]
    speech = srv.setups[0]["generationConfig"]["speechConfig"]
    assert speech["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
    assert cfg.llm_routes[0].model_dump() == stored  # the override is a copy
    assert srv.client_closes == [1000]
    assert router.used("gemini") > 0  # the estimate is charged; no wait for the late total
    assert "text" not in json.dumps([m for m in srv.received if "realtimeInput" in m])
    text = caplog.text
    assert KEY not in text and REPLY not in text and REFUSAL not in text


def test_the_line_rides_the_text_turn_word_for_word_in_the_configured_language(gemini):
    srv = gemini([["audio"]])
    run(make_cfg(gemini_route(srv.base_url()), language="it"))
    lang, sentence = preview_line(make_cfg(gemini_route(srv.base_url()), language="it"))
    said = line_sent(srv)
    assert lang == "it" and json.dumps(sentence, ensure_ascii=False) in said
    assert "Italian" in said and "word for word" in said


def test_a_language_without_a_file_falls_back_to_english_and_says_so():
    route = pollen_route("http://127.0.0.1:1")
    lang, text = preview_line(make_cfg(route, language="de"))
    assert lang == "en" and text == preview_line(make_cfg(route, language="en"))[1]
    assert text != preview_line(make_cfg(route, language="it"))[1]


def test_a_tool_call_before_the_audio_is_answered_and_the_audio_follows(gemini):
    srv = gemini([["call:note_tone", "audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()), tone_tool=True)
    rate, _, _, pcm = wav_of(run(cfg))
    assert rate == OUTPUT_RATE_HZ and pcm
    outputs = [r["response"] for rs in srv.tool_responses for r in rs]
    assert outputs and all(json.dumps(o) == TONE_NOTED for o in outputs)


def test_the_setup_carries_the_checks_offline_tools(gemini):
    srv = gemini([["audio"]])
    run(make_cfg(gemini_route(srv.base_url())))
    names = [d["name"] for d in srv.setups[0]["tools"][0]["functionDeclarations"]]
    assert {"look_at", "express_emotion"} <= set(names)


def test_tools_that_cannot_be_built_send_nothing(gemini):
    srv = gemini([["audio"]])
    with pytest.raises(preview.PreviewFailed) as exc:
        run(make_cfg(gemini_route(srv.base_url())), tools=lambda: None)
    assert exc.value.code == "setup_unready" and srv.connections == 0


def test_no_audio_fails_by_code_never_empty_audio(gemini):
    srv = gemini([[]])
    with pytest.raises(preview.PreviewFailed) as exc:
        run(make_cfg(gemini_route(srv.base_url())))
    assert exc.value.code == preview.NO_AUDIO


def test_calls_left_open_past_the_round_cap_fail_by_code(gemini):
    srv = gemini([["call:note_tone", "call:note_tone"]])
    cfg = make_cfg(gemini_route(srv.base_url()), tone_tool=True)
    cfg = cfg.model_copy(
        update={"conversation": cfg.conversation.model_copy(update={"max_tool_rounds": 1})}
    )
    with pytest.raises(preview.PreviewFailed) as exc:
        run(cfg)
    assert exc.value.code == preview.TOOL_ROUNDS


def test_a_1008_refusal_is_a_code_and_a_close_code_never_the_reason(gemini, caplog):
    caplog.set_level(logging.DEBUG)
    srv = gemini(refuse=1)
    with pytest.raises(preview.PreviewFailed) as exc:
        run(make_cfg(gemini_route(srv.base_url())))
    assert exc.value.close_code == POLICY and exc.value.code
    assert REFUSAL not in caplog.text and REFUSAL not in str(exc.value)


def test_audio_past_the_cap_fails_the_preview_and_says_so_before_the_close(gemini):
    srv = gemini([["audio"]], chunks=40, chunk_ms=100)  # 40 x 4800 bytes
    cfg = make_cfg(gemini_route(srv.base_url()), preview_max_bytes=65_536, send_timeout_s=1)
    told: list[object] = []
    with pytest.raises(preview.PreviewFailed) as exc:
        run(cfg, ready=told.append)
    assert exc.value.code == preview.AUDIO_CAP and told == [exc.value]


def test_a_pollen_preview_uses_the_rate_the_session_names_else_the_native_one(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer() as server:
        cfg = make_cfg(pollen_route(server.http_base()))
        rate, _, _, pcm = wav_of(run(cfg, "Serena"))
        update = next(e for e in server.received if e["type"] == "session.update")
        create = next(e for e in server.received if e["type"] == "response.create")
        server.output_rate = SAID_RATE_HZ
        said, _, _, _ = wav_of(run(cfg, "Serena"))
    assert rate == NATIVE_RATE_HZ and pcm and said == SAID_RATE_HZ
    assert update["session"]["audio"]["output"]["voice"] == "Serena"
    assert "word for word" in create["response"]["instructions"]
    assert not [e for e in server.received if e["type"] == "conversation.item.create"]
    assert REPLY_TEXT not in caplog.text


def test_refusals_come_before_anything_is_sent(gemini):
    srv = gemini([["audio"]])
    cfg = make_cfg(gemini_route(srv.base_url()))
    spent = make_router(cfg)
    spent.charge("gemini", cfg.llm_routes[0].daily_token_budget)
    with pytest.raises(RouteBudgetExhausted):
        run(cfg, router=spent)

    def missing(_name):
        raise KeyError(_name)

    with pytest.raises(RouteKeyUnavailable):
        run(cfg, router=Router(cfg.llm_routes, cfg.safety, secrets=missing))
    offline = make_router(cfg)
    offline.monitor = SimpleNamespace(online=False)
    with pytest.raises(preview.PreviewFailed) as exc:
        run(cfg, router=offline)
    assert exc.value.code == preview.OFFLINE
    cloud = make_cfg(gemini_route(srv.base_url(), locality="cloud"))
    cloud = cloud.model_copy(
        update={"privacy": cloud.privacy.model_copy(update={"pseudonyms_to_cloud": True})}
    )
    with pytest.raises(preview.PreviewFailed) as exc:
        run(cloud)
    assert exc.value.code == preview.PSEUDONYMS
    assert srv.connections == 0


def test_the_known_voices_cover_both_protocols():
    assert len(preview.GEMINI_VOICES) == 30 == len(set(preview.GEMINI_VOICES))
    assert len(preview.QWEN3_VOICES) == 9 == len(set(preview.QWEN3_VOICES))
    assert "Kore" in preview.KNOWN_VOICES["gemini_live"]
    assert "Serena" in preview.KNOWN_VOICES["openai_realtime"]


def test_a_slow_setup_fails_within_the_previews_deadline(gemini):
    srv = gemini([["audio"]], setup_delay_s=3.0)
    cfg = make_cfg(gemini_route(srv.base_url(), connect_timeout_s=15), send_timeout_s=1)
    started = time.monotonic()
    with pytest.raises(preview.PreviewFailed) as exc:
        run(cfg, deadline_s=2.0)
    assert exc.value.code == "connect_timeout"
    assert time.monotonic() - started < 2.0 + 1.5  # the deadline, the close and some slack
