"""The voice wiring reads the live config after a hot reload, not the start-up one (SPEC 11A)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from reachy_config import reachy_config
from roboemot.bus import Bus
from roboemot.voice import pipeline
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.vad import EnergyVAD

NOW = 1000.0


class FakeASR:
    def __init__(self, paths, rate_hz):
        self.languages = list(paths)

    def missing_words(self, language, words):
        return None

    def grammar_words(self, language, samples, phrases):
        return None


class Bearing:
    def __init__(self):
        self.max_ages: list[float] = []

    def get(self, now, max_age_s):
        self.max_ages.append(max_age_s)
        return None

    def close(self):
        pass


def changed(cfg, section, **values):
    """``cfg`` with the fields ``values`` of the top-level ``section`` replaced."""
    return cfg.model_copy(update={section: getattr(cfg, section).model_copy(update=values)})


@pytest.fixture()
def built(monkeypatch):
    monkeypatch.setattr(pipeline.voice_models, "model_file", lambda root, name: Path(name))
    monkeypatch.setattr(pipeline, "SileroVAD", lambda path, threads: EnergyVAD(16000, 512))
    monkeypatch.setattr(pipeline, "VoskASR", FakeASR)
    cfg = reachy_config()
    bearing = Bearing()
    person = SimpleNamespace(kind="person", identity="anna", last_seen=NOW - 10.0)
    world = SimpleNamespace(entries=[person], get=lambda entry_id: None)
    listener, close, bargein = pipeline.build_listener(
        cfg,
        SimpleNamespace(read=lambda timeout: None),
        SpeechActivity(cfg.voice.echo_tail_ms),
        Bus(),
        lambda name, now, payload: None,
        SimpleNamespace(target=None),
        world,
        None,
        bearing,
        lambda u: None,
    )
    return cfg, listener, bargein, bearing


def test_enrolled_in_view_reads_the_reloaded_stale_window(built):
    cfg, listener, _, _ = built
    assert cfg.attention.stale_s < 10.0
    assert not listener._enrolled_in_view(NOW)
    listener.apply_config(changed(cfg, "attention", stale_s=20.0))
    assert listener._enrolled_in_view(NOW)


def test_the_sound_bearing_age_limit_follows_a_reload(built):
    cfg, listener, _, bearing = built
    listener._fast._bearing(NOW)
    voice = cfg.voice.model_copy(
        update={"fastpath": cfg.voice.fastpath.model_copy(update={"doa_max_age_ms": 250.0})}
    )
    listener.apply_config(cfg.model_copy(update={"voice": voice}))
    listener._fast._bearing(NOW)
    assert bearing.max_ages == [cfg.voice.fastpath.doa_max_age_ms / 1000.0, 0.25]


def test_the_barge_in_language_follows_a_reloaded_default_language(built):
    cfg, listener, bargein, _ = built
    other = next(lang for lang in cfg.voice.asr.models if lang != cfg.conversation.language)
    assert bargein._language() == cfg.conversation.language
    listener.apply_config(changed(cfg, "conversation", language=other))
    assert bargein._language() == other
