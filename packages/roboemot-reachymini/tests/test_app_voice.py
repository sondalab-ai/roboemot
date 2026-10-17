"""The voice listener wired into the app (SPEC 8 stages 2-4) and its app-side lookups."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import roboemot_reachymini.app as app_module
import roboemot_reachymini.voice as voice_module
from roboemot.attention import AttentionSelector
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.asr import ASREvent
from roboemot.voice.vad import EnergyVAD
from roboemot.world import Observation, WorldModel
from roboemot_reachymini.gaze import FRAME
from roboemot_reachymini.world import SOUND_SOURCE
from test_app_perception import data, wait_for  # noqa: F401  (data: fixture)
from test_app_reflex import ScriptedMini, enable, run_app

VOICE_YAML = "voice:\n  enabled: true\n"


def test_app_runs_without_voice_models_and_says_so(data, caplog):  # noqa: F811
    enable(data, VOICE_YAML)
    app = app_module.RoboEmotApp.__new__(app_module.RoboEmotApp)
    with caplog.at_level(logging.WARNING):
        stop, thread, errors = run_app(app, ScriptedMini().mini)
        try:
            assert wait_for(lambda: app.diagnostics is not None and app.audio is not None)
            section = app.diagnostics.report()["sections"]["roboemot.voice"]
            assert section["ok"] and section["data"] == {"state": "off"}
            assert app.voice_listener is None
            assert wait_for(lambda: "voice: not listening" in caplog.text)
        finally:
            stop.set()
            thread.join(timeout=10)
    assert not errors and not thread.is_alive()


class FakeASR:
    rate_hz = 16000

    def __init__(self, paths, rate):
        self.paths = paths

    def load(self, languages):
        pass

    def start(self, language):
        pass

    def accept(self, samples):
        return ASREvent("ciao")

    def finish(self):
        return "ciao"

    def reset(self):
        pass


@pytest.fixture()
def fakes(monkeypatch):
    monkeypatch.setattr(voice_module.voice_models, "model_file", lambda root, name: Path(name))
    monkeypatch.setattr(voice_module, "SileroVAD", lambda path, threads: EnergyVAD(16000, 512))
    monkeypatch.setattr(voice_module, "VoskASR", FakeASR)


def test_app_builds_the_listener_and_stops_it_cleanly(data, fakes):  # noqa: F811
    enable(data, VOICE_YAML)
    app = app_module.RoboEmotApp.__new__(app_module.RoboEmotApp)
    got = []
    app.on_utterance = got.append
    stop, thread, errors = run_app(app, ScriptedMini().mini)
    try:
        assert wait_for(lambda: app.voice_listener is not None)
        assert wait_for(lambda: app.voice_listener.snapshot()["state"] == "running")
        report = app.diagnostics.report()["sections"]["roboemot.voice"]["data"]
        assert report["state"] == "running" and "latency" in report
        u = SimpleNamespace(text="x")
        app.voice_listener._on_utterance(u)  # slice 5 hooks `on_utterance`
        assert got == [u]
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()
    assert app.voice_listener.snapshot()["state"] in ("running", "unavailable")
    assert app.voice_listener._thread is None


class Registry:
    def __init__(self, prefs):
        self.prefs = prefs

    def get_person(self, person_id):
        if person_id not in self.prefs:
            raise KeyError(person_id)
        return SimpleNamespace(preferences=self.prefs[person_id])


def wiring(registry, fakes_ok=True):
    cfg = RoboEmotConfig()
    bus = Bus()
    world = WorldModel(cfg.world, FRAME, bus)
    selector = AttentionSelector(cfg.attention, bus)
    events = []
    built = voice_module.build_listener(
        cfg,
        SimpleNamespace(read=lambda timeout: None),
        SpeechActivity(),
        bus,
        lambda *a: events.append(a),
        selector,
        world,
        registry,
        FRAME,
        lambda u: None,
    )
    assert built is not None
    return built[0], bus, world, selector


def put_person(world, selector, identity, now=10.0):
    world.observe(Observation("person", now, "vision", 0.0, 0.0, identity=identity), {})
    selector.update(now, world.entries)


def test_speaker_language_is_the_attended_persons_preference(fakes):
    listener, bus, world, selector = wiring(Registry({"p-1": {"language": "en"}, "p-2": {}}))
    assert listener._speaker_language() is None  # nobody attended
    put_person(world, selector, "p-1")
    assert listener._speaker_language() == "en"


def test_speaker_language_ignores_unknown_people_missing_records_and_bad_values(fakes):
    listener, bus, world, selector = wiring(Registry({"p-2": {"language": 3}}))
    put_person(world, selector, "unknown")
    assert listener._speaker_language() is None
    world2 = wiring(Registry({}))
    put_person(world2[2], world2[3], "p-9")
    assert world2[0]._speaker_language() is None  # record gone: default language
    put_person(world, selector, "p-2")
    assert listener._speaker_language() is None  # not a string


def test_enrolled_in_view_needs_a_fresh_known_person(fakes):
    listener, bus, world, selector = wiring(Registry({"p-1": {}}))
    assert not listener._enrolled_in_view(10.0)
    put_person(world, selector, "unknown")
    assert not listener._enrolled_in_view(10.0)
    put_person(world, selector, "p-1")
    assert listener._enrolled_in_view(10.5)
    assert not listener._enrolled_in_view(10.0 + RoboEmotConfig().attention.stale_s + 1)


def test_sound_bearing_is_rotated_into_the_base_frame_and_expires(fakes):
    listener, bus, world, selector = wiring(Registry({}))
    fast = listener._fast
    yaw = FRAME.yaw_dofs[0]
    obs = Observation("unknown", 5.0, SOUND_SOURCE, 0.2, 0.0, confidence=0.5)
    bus.publish("percept.entity", obs, 5.0, joint_state={yaw: 0.3})
    other = Observation("unknown", 5.1, "other", 1.0, 0.0)
    bus.publish("percept.entity", other, 5.1, joint_state={})
    az = fast._bearing(5.5)
    assert az == pytest.approx(0.5, abs=0.05)
    assert fast._bearing(5.0 + RoboEmotConfig().voice.fastpath.doa_max_age_ms / 1000 + 1) is None
