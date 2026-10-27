"""The realtime route's audio settings (SPEC 8, Realtime speech route; SPEC 11A, 12)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from roboemot.config.schema import PrivacyConfig, RealtimeRouteConfig, RoboEmotConfig


def test_audio_defaults_and_ranges():
    rt = RealtimeRouteConfig()
    assert rt.audio_gate == "addressed"
    assert rt.transcript_timeout_s == 4.0
    assert rt.server_silence_ms == 600.0
    assert rt.close_silence_ms == 900.0
    assert rt.daily_upload_budget_s == 1800.0
    fields = RealtimeRouteConfig.model_fields
    for name, lo, hi in (
        ("transcript_timeout_s", 0.5, 15),
        ("server_silence_ms", 64, 3000),
        ("close_silence_ms", 100, 5000),
        ("daily_upload_budget_s", 0, 86400),
    ):
        meta = {type(m).__name__: m for m in fields[name].metadata}
        assert meta["Ge"].ge == lo and meta["Le"].le == hi, name
    with pytest.raises(ValidationError):
        RealtimeRouteConfig(audio_gate="live")


def test_realtime_audio_is_an_off_by_default_opt_in():
    assert PrivacyConfig().realtime_audio is False
    field = PrivacyConfig.model_fields["realtime_audio"]
    assert field.json_schema_extra["opt_in"] is True
    assert field.description == (
        "Allow addressed microphone audio to leave the robot to the realtime route"
    )


def test_close_silence_must_exceed_the_server_silence():
    with pytest.raises(ValidationError, match="close_silence_ms"):
        RealtimeRouteConfig(server_silence_ms=900, close_silence_ms=900)
    assert RealtimeRouteConfig(server_silence_ms=900, close_silence_ms=901).close_silence_ms == 901


def test_server_silence_must_cover_the_local_end_silence():
    with pytest.raises(ValidationError, match="server_silence_ms"):
        RoboEmotConfig.model_validate(
            {
                "voice": {
                    "vad": {"end_silence_ms": 700},
                    "realtime_route": {"server_silence_ms": 600},
                }
            }
        )
    cfg = RoboEmotConfig.model_validate(
        {"voice": {"vad": {"end_silence_ms": 600}, "realtime_route": {"server_silence_ms": 600}}}
    )
    assert cfg.voice.realtime_route.server_silence_ms == 600
