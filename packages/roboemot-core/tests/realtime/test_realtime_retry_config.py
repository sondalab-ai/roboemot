"""The realtime route's retry settings (SPEC 8, Realtime speech route; SPEC 11A).

Every retry setting's Italian description is checked by
``tests/test_webui_one_language.py::test_every_setting_description_is_translated``, which walks
the whole schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from roboemot.config.schema import RoboEmotConfig


def test_retry_defaults_match_the_design():
    rt = RoboEmotConfig().voice.realtime_route
    assert (rt.retry.enabled, rt.retry.max_retries, rt.retry.total_budget_s) == (True, 2, 15.0)
    assert rt.retry.filler_after_s == 2.5 and rt.transcript_timeout_s == 9.0
    assert rt.retry.sorry_lines["it"].startswith("Scusa")
    assert rt.retry.reconnect_wait_s == 5.0
    assert rt.retry.announce_first_retry is True
    assert rt.retry.clip_record_attempts == 3
    assert rt.retry.retrying_lines == {
        "it": "Un attimo, riprovo.",
        "en": "One moment, let me try again.",
    }
    assert rt.retry.thinking_lines == {"it": "Fammi pensare…", "en": "Let me think…"}


@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [
        ("max_retries", 0, 3),
        ("total_budget_s", 1, 60),
        ("reconnect_wait_s", 0.5, 30),
        ("filler_after_s", 0.5, 10),
        ("clip_record_attempts", 1, 20),
    ],
)
def test_retry_ranges(name, lo, hi):
    retry = type(RoboEmotConfig().voice.realtime_route.retry)
    meta = {type(m).__name__: m for m in retry.model_fields[name].metadata}
    assert (meta["Ge"].ge, meta["Le"].le) == (lo, hi)
    with pytest.raises(ValidationError):
        retry(**{name: hi + 1})
