"""The voice listener wired into the app (SPEC 8 stages 2-4) and its app-side lookups."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import roboemot.voice.pipeline as voice_module
import roboemot_reachymini.app as app_module
from roboemot.attention import AttentionSelector
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.runtime import assembly
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.asr import ASREvent
from roboemot.voice.vad import EnergyVAD
from roboemot.voice.wake import AT_FACE, NO_FACING_DATA, NOT_FACING
from roboemot.world import Observation, WorldModel
from roboemot_reachymini.gaze import FRAME
from roboemot_reachymini.platform import ReachyMiniSession
from roboemot_reachymini.voice import LatestBearing
from roboemot_reachymini.world import SOUND_SOURCE, DoASource
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

    languages = ["it", "en"]

    def load(self, languages):
        pass

    def missing_words(self, language, words):
        return None

    def grammar_words(self, language, samples, phrases):
        return None

    def start(self, language):
        pass

    def accept(self, samples):
        return ASREvent("ciao")

    def finish(self):
        return "ciao"

    def reset(self):
        pass

    def stats(self):
        return {}


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
        assert report["direction"]["enabled"] is True
        assert report["direction"]["doa_available"] is True  # the manifest declares doa
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


def wiring(registry, fakes_ok=True, doa=True, cfg=None):
    cfg = cfg or RoboEmotConfig()
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
        LatestBearing(bus, FRAME),
        lambda u: None,
        doa_available=doa,
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


# ---- voice-at-face check through the real bearing source and world model (SPEC 8, 5A, 4A)

HEAD_YAW = 0.8  # rad, about 46 deg to the left


def hear(bus, sdk_angle, head_yaw, now=10.0):
    """One DoA reading through the real source: the SDK angle with the head at ``head_yaw``,
    published on the bus with its joint state as the app does."""
    mini = SimpleNamespace(media=SimpleNamespace(get_DoA=lambda: (sdk_angle, True)))
    source = DoASource(
        mini, RoboEmotConfig().world.scan, lambda: {"head.yaw": head_yaw}, lambda: now
    )
    sink = lambda obs, js: bus.publish("percept.entity", obs, obs.stamp, joint_state=js)  # noqa: E731
    assert source.poll_once(sink)


def check(listener, now=10.0):
    d = listener._direction
    d.begin()
    d.sample(now)
    return d.check() == AT_FACE


def face_ahead_of_turned_head(world, selector, now=10.0):
    """A face in the middle of the picture with the head turned ``HEAD_YAW`` to the left."""
    world.observe(Observation("person", now, "vision", 0.0, 0.0), {"head.yaw": HEAD_YAW})
    selector.update(now, world.entries)


def test_voice_from_the_face_matches_with_the_head_turned(fakes):
    listener, bus, world, selector = wiring(Registry({}))
    face_ahead_of_turned_head(world, selector)
    hear(bus, math.pi / 2, HEAD_YAW)  # straight ahead of the turned head: at the face
    assert check(listener)
    hear(bus, math.pi / 2, 0.0)  # same SDK angle, head straight: 46 deg off the face
    assert not check(listener)
    snap = listener.snapshot()["direction"]
    assert snap["checks"]["passed"] == 1 and snap["checks"]["rejected"] == 1


def test_music_60_deg_off_the_face_is_rejected_through_the_real_frame(fakes):
    listener, bus, world, selector = wiring(Registry({}))
    face_ahead_of_turned_head(world, selector)
    hear(bus, math.pi / 2 + math.radians(60.0), HEAD_YAW)  # 60 deg to the right of the face
    assert not check(listener)
    assert listener.snapshot()["direction"]["face_offset_deg"]["last"] == pytest.approx(60.0)


def test_without_a_reading_reachy_needs_the_wake_word(fakes):
    listener, bus, world, selector = wiring(Registry({}))
    face_ahead_of_turned_head(world, selector)
    assert not check(listener)  # the array could hear and gave nothing (auto: silent -> reject)
    assert listener.snapshot()["direction"]["degraded"]["silent"] == 1


def test_without_a_sound_direction_the_check_falls_back_to_the_attended_face(fakes):
    listener, bus, world, selector = wiring(Registry({}), doa=False)
    assert not check(listener)  # nobody in view
    face_ahead_of_turned_head(world, selector)
    assert check(listener)
    stale = RoboEmotConfig().attention.stale_s  # face_max_age_s empty: follows attention.stale_s
    assert not check(listener, now=10.0 + stale + 1)
    assert listener.snapshot()["direction"]["degraded"]["no_source"] == 3


def test_the_face_at_the_voice_must_look_at_the_robot_through_the_real_wiring(fakes):
    cfg = RoboEmotConfig.model_validate({"voice": {"facing": {"enabled": True}}})
    listener, bus, world, selector = wiring(Registry({}), cfg=cfg)
    seen = Observation("person", 10.0, "vision", 0.0, 0.0, facing_yaw_rad=math.radians(40.0))
    world.observe(seen, {"head.yaw": HEAD_YAW})  # watching a screen beside the robot
    selector.update(10.0, world.entries)
    hear(bus, math.pi / 2, HEAD_YAW)  # the voice at that face
    d = listener._direction
    d.begin()
    d.sample(10.0)
    assert d.check() == NOT_FACING
    world.observe(
        Observation("person", 10.2, "vision", 0.0, 0.0, facing_yaw_rad=math.radians(3.0)),
        {"head.yaw": HEAD_YAW},
    )
    assert check(listener, now=10.2)
    later = 10.2 + cfg.voice.facing.yaw_max_age_ms / 1000 + 1  # the face still visible
    hear(bus, math.pi / 2, HEAD_YAW, now=later)
    world.observe(Observation("person", later, "vision", 0.0, 0.0), {"head.yaw": HEAD_YAW})
    d.begin()
    d.sample(later)
    assert d.check() == NO_FACING_DATA  # no yaw near the voice: not looking at the robot
    hear(bus, math.pi / 2, HEAD_YAW, now=10.0)  # a listener behind: the yaw at its time
    d.begin()
    d.sample(10.0)
    assert d.check() == NOT_FACING
    facing = listener.snapshot()["direction"]["facing"]["checks"]
    assert facing["not_facing"] == 2 and facing["passed"] == 1 and facing["no_data"] == 1


def test_a_voice_from_a_face_behind_is_mirrored_to_the_front_and_rejected(fakes):
    listener, bus, world, selector = wiring(Registry({}))
    behind = math.radians(150.0)
    world.observe(Observation("person", 10.0, "vision", 0.0, 0.0), {"head.yaw": behind})
    selector.update(10.0, world.entries)
    # The voice comes from 150 deg with the head straight; the half-plane array reports 30 deg.
    hear(bus, math.pi / 2 - math.radians(30.0), 0.0)
    assert not check(listener)
    assert listener.snapshot()["direction"]["face_offset_deg"]["last"] == pytest.approx(120.0)


def test_a_rear_sound_mirrored_onto_a_front_face_passes(fakes):
    listener, bus, world, selector = wiring(Registry({}))
    front = math.radians(30.0)
    world.observe(Observation("person", 10.0, "vision", 0.0, 0.0), {"head.yaw": front})
    selector.update(10.0, world.entries)
    # Music from 150 deg (behind) is reported mirrored at 30 deg: on the face (known limit).
    hear(bus, math.pi / 2 - math.radians(30.0), 0.0)
    assert check(listener)


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


def test_without_a_listener_the_hub_queues_no_voice_audio(data):  # noqa: F811
    enable(data, VOICE_YAML + "reflexes:\n  audio:\n    enabled: false\n")
    app = app_module.RoboEmotApp.__new__(app_module.RoboEmotApp)
    stop, thread, errors = run_app(app, ScriptedMini().mini)
    try:
        assert wait_for(lambda: app.audio is not None and app.voice_listener is None)
        time.sleep(0.3)
        assert app.audio.snapshot()["voice"] is None and app.audio.snapshot()["state"] == "off"
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors


def test_yes_no_handler_is_wired_to_the_fast_path(data, fakes):  # noqa: F811
    enable(data, VOICE_YAML)
    app = app_module.RoboEmotApp.__new__(app_module.RoboEmotApp)
    stop, thread, errors = run_app(app, ScriptedMini().mini)
    try:
        assert wait_for(lambda: app.voice_listener is not None)
        assert app.voice_listener._fast._on_answer is app.yes_no
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors


def yes_no(speech_output):
    asked = []
    app = SimpleNamespace(speech_output=speech_output)
    speech = SpeechActivity(0.0)
    expression = SimpleNamespace(request=lambda label, now=None: asked.append(label) or True)
    handler = assembly.make_yes_no(app, expression, speech, RoboEmotConfig().voice.fastpath)
    return app, handler, asked, speech


def test_yes_no_handler_works_before_the_speech_output_exists():
    _, handler, asked, _ = yes_no(None)
    handler("yes", 10.0)
    handler("no", 11.0)
    assert asked == ["nod", "shake"]


def test_yes_no_handler_reads_the_question_from_the_speech_output_once_it_exists():
    app, handler, asked, speech = yes_no(None)
    app.speech_output = SimpleNamespace(question_end=lambda: 100.0)
    handler("yes", 101.0)
    assert asked == []
    handler("yes", 101.0 + RoboEmotConfig().voice.fastpath.answer_window_s)
    assert asked == ["nod"]
    speech.extend(500.0)
    handler("no", 400.0)  # the robot is speaking
    assert asked == ["nod"]


def test_yes_no_handler_follows_a_reloaded_window():
    _, handler, asked, _ = yes_no(SimpleNamespace(question_end=lambda: 100.0))
    handler("yes", 103.0)
    assert asked == []
    cfg = RoboEmotConfig().voice.fastpath
    handler.apply_config(cfg.model_copy(update={"answer_window_s": 1.0}))
    handler("yes", 103.0)
    assert asked == ["nod"]


def test_yes_no_handler_survives_a_broken_speech_output():
    def boom():
        raise RuntimeError("x")

    _, handler, asked, _ = yes_no(SimpleNamespace(question_end=boom))
    handler("yes", 1.0)
    assert asked == []


def test_the_app_hands_its_bus_to_the_robot_bearing_hook_and_the_result_to_the_listener(
    data,  # noqa: F811
    fakes,
    monkeypatch,
):
    enable(data, VOICE_YAML)
    seen = {}
    real_hook, real_build = ReachyMiniSession.bearing_source, assembly.build_listener

    def hook(self, bus):
        seen["hook_bus"] = bus
        seen["bearing"] = real_hook(self, bus)
        return seen["bearing"]

    def build(cfg, audio, speech, bus, emit, selector, world, registry, bearing, *rest, **kw):
        seen["listener_args"] = (bus, selector, world, bearing, audio, speech)
        return real_build(
            cfg, audio, speech, bus, emit, selector, world, registry, bearing, *rest, **kw
        )

    monkeypatch.setattr(ReachyMiniSession, "bearing_source", hook)
    monkeypatch.setattr(assembly, "build_listener", build)
    app = app_module.RoboEmotApp.__new__(app_module.RoboEmotApp)
    stop, thread, errors = run_app(app, ScriptedMini().mini)
    try:
        assert wait_for(lambda: app.voice_listener is not None)
        bus, selector, world, bearing, audio, speech = seen["listener_args"]
        assert seen["hook_bus"] is app.bus is bus
        assert bearing is seen["bearing"] and isinstance(bearing, LatestBearing)
        assert (selector, world, audio, speech) == (app.selector, app.world, app.audio, app.speech)
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors


def test_no_bearing_source_is_built_without_voice(data, monkeypatch):  # noqa: F811
    built = []
    monkeypatch.setattr(ReachyMiniSession, "bearing_source", lambda self, bus: built.append(bus))
    app = app_module.RoboEmotApp.__new__(app_module.RoboEmotApp)
    stop, thread, errors = run_app(app, ScriptedMini().mini)
    try:
        assert wait_for(lambda: app.diagnostics is not None)
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and built == []


def test_bearing_is_the_reading_nearest_to_the_asked_time_on_either_side():
    bus = Bus()
    bearing = LatestBearing(bus, FRAME)
    assert bearing.get(10.0, 0.5) is None
    for stamp, sdk_angle in ((10.0, math.pi / 2), (11.0, 0.0)):  # front, then full left
        hear(bus, sdk_angle, 0.0, now=stamp)
    assert bearing.get(10.1, 0.5) == pytest.approx(0.0)  # a lagging listener gets its own sound
    assert bearing.get(10.9, 0.5) == pytest.approx(math.pi / 2)
    assert bearing.get(9.0, 0.5) is None  # the nearest reading is a later sound
    assert bearing.get(12.0, 0.5) is None  # stale
    bearing.close()
