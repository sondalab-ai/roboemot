"""The reflex engine wired into the app (SPEC 10, 12, 12A, 13), against a mocked SDK client."""

from __future__ import annotations

import threading
import time

import numpy as np
import roboemot_reachymini.app as app_module
from roboemot.world import Observation
from test_app_perception import data, fake_mini, new_app, wait_for  # noqa: F401  (data: fixture)

QUIET, LOUD = 0.02, 0.5


class ScriptedMini:
    """A mini whose microphone returns quiet chunks until ``bang`` is set, then one loud one."""

    def __init__(self) -> None:
        self.mini = fake_mini()
        self.bang = threading.Event()
        media = self.mini.media
        media.audio = object()
        media.get_input_audio_samplerate.return_value = 16000
        media.get_input_channels.return_value = 1
        media.get_output_audio_samplerate.return_value = 16000
        media.get_audio_sample.side_effect = self._chunk

    def _chunk(self):
        level = LOUD if self.bang.is_set() else QUIET
        self.bang.clear()
        return np.full(160, level, np.float32)


def run_app(app, mini):
    stop, errors = threading.Event(), []

    def target():
        try:
            app.run(mini, stop)
        except Exception as exc:  # surfaced by the test
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return stop, thread, errors


def test_the_debug_report_has_a_reflex_section_with_the_audio_state(data):  # noqa: F811
    app = new_app()
    stop, thread, errors = run_app(app, fake_mini())  # a MagicMock media: no usable chunks
    try:
        assert wait_for(lambda: app.diagnostics is not None and app.reflexes is not None)
        assert wait_for(lambda: app.audio.snapshot()["state"] == "no_data")
        section = app.diagnostics.report()["sections"]["roboemot.reflexes"]
        assert section["ok"]
        body = section["data"]
        assert set(body["rules"]) == {"startle-loud", "withdraw-close", "turn-to-voice"}
        assert body["audio"]["state"] == "no_data" and body["audio"]["reason"]
        assert body["latency"] == {}
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()
    assert not [t for t in threading.enumerate() if t.name == "roboemot-audio"]


def test_a_loud_sound_from_the_microphone_startles_through_the_motion_loop(data):  # noqa: F811
    app = new_app()
    scripted = ScriptedMini()
    stop, thread, errors = run_app(app, scripted.mini)
    try:
        assert wait_for(lambda: app.reflexes is not None and app.audio is not None)
        assert wait_for(lambda: app.audio.snapshot()["chunks"] > 20)  # the floor is learned
        scripted.bang.set()
        assert wait_for(lambda: app.reflexes.snapshot()["rules"]["startle-loud"]["fired"] == 1)
        lat = app.reflexes.snapshot()["latency"]["loud_onset"]["command_ms"]
        # SPEC 13 asks 100 ms at p95; the exact budget is asserted on a fake clock in core, here
        # only that the path through real threads is nowhere near a stall.
        assert lat["max"] < 250.0
        report = app.diagnostics.report()["sections"]["roboemot.reflexes"]["data"]
        assert report["audio"]["state"] == "running"
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()
    assert not [t for t in threading.enumerate() if t.name == "roboemot-audio"]
    assert scripted.mini.media.stop_recording.called


def test_the_robot_does_not_turn_toward_a_voice_while_the_minor_flag_is_on(data):  # noqa: F811
    app = new_app()
    stop, thread, errors = run_app(app, fake_mini())
    try:
        assert wait_for(lambda: app.minor_presence is not None and app.diagnostics is not None)
        assert app.minor_present  # starts on, fails safe (SPEC 12) ...

        def loop_flag() -> bool:
            motion = app.diagnostics.report()["sections"]["roboemot.motion"]["data"]
            return motion["minor_present"]

        assert loop_flag()  # ... and so is the motion loop's, from its first tick
        now = time.monotonic()
        obs = Observation("unknown", now, "doa", 0.4, 0.0, confidence=0.5)
        app.bus.publish("percept.entity", obs, now, joint_state={"head.yaw": 0.0})

        def turn() -> dict:
            return app.reflexes.snapshot()["rules"]["turn-to-voice"]

        assert wait_for(lambda: turn()["suppressed_minor"])
        assert turn()["fired"] == 0
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors


VOICE_YAML = "voice:\n  enabled: true\n  echo_tail_ms: 150\n"


def enable(folder, extra: str) -> None:
    (folder / "app.yaml").write_text((folder / "app.yaml").read_text() + extra)


def startles(app) -> int:
    return app.reflexes.snapshot()["rules"]["startle-loud"]["fired"]


def test_the_hub_starts_for_voice_alone_and_the_reflex_sink_stays_detached(
    data,  # noqa: F811
    monkeypatch,
):
    from test_app_voice import FakeASR, voice_module

    monkeypatch.setattr(voice_module.voice_models, "model_file", lambda root, name: name)
    monkeypatch.setattr(voice_module, "SileroVAD", lambda path, threads: _Quiet())
    monkeypatch.setattr(voice_module, "VoskASR", FakeASR)
    enable(data, VOICE_YAML + "reflexes:\n  audio:\n    enabled: false\n")
    app = new_app()
    scripted = ScriptedMini()
    stop, thread, errors = run_app(app, scripted.mini)
    try:
        assert wait_for(lambda: app.audio is not None and app.speaker is not None)
        assert wait_for(lambda: (app.audio.snapshot()["voice"] or {}).get("frames", 0) > 1600)
        assert app.audio.snapshot()["state"] == "running"
        assert app.audio.snapshot()["voice"]["rate_hz"] == 16000
        scripted.bang.set()
        assert wait_for(lambda: not scripted.bang.is_set())  # the loud chunk was read ...
        time.sleep(0.2)
        assert startles(app) == 0  # ... and never reached the reflex engine
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()
    assert scripted.mini.media.stop_recording.called and scripted.mini.media.stop_playing.called


def test_the_hub_is_off_without_reflex_audio_and_without_voice(data):  # noqa: F811
    enable(data, "reflexes:\n  audio:\n    enabled: false\n")
    app = new_app()
    scripted = ScriptedMini()
    stop, thread, errors = run_app(app, scripted.mini)
    try:
        assert wait_for(lambda: app.audio is not None)
        time.sleep(0.2)
        assert app.audio.snapshot()["state"] == "off" and app.speaker is None
        assert not scripted.mini.media.start_recording.called
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors


def test_no_startle_while_the_robot_speaks_and_during_the_tail_then_startle_again(data):  # noqa: F811
    enable(data, VOICE_YAML)
    app = new_app()
    scripted = ScriptedMini()
    stop, thread, errors = run_app(app, scripted.mini)
    loud = []
    try:
        assert wait_for(lambda: app.reflexes is not None and app.speaker is not None)
        assert wait_for(lambda: app.audio.snapshot()["chunks"] > 20)  # the floor is learned
        app.bus.subscribe("sound.loud", lambda m: loud.append(m))
        until = app.speaker.push_audio_sample(np.zeros(16000, np.float32), 16000)  # 1 s of speech
        scripted.bang.set()  # its own voice comes back through the microphone, loud
        assert wait_for(lambda: not scripted.bang.is_set())
        assert time.monotonic() < until
        scripted.bang.set()  # a second one inside the echo tail, after the playback ended
        time.sleep(max(0.0, until + 0.05 - time.monotonic()))
        assert wait_for(lambda: not scripted.bang.is_set())
        time.sleep(0.2)
        assert startles(app) == 0 and loud == []
        time.sleep(max(0.0, until + 0.15 + 0.1 - time.monotonic()))  # tail over, quiet chunks
        assert wait_for(lambda: app.audio.snapshot()["chunks"] > 0)
        time.sleep(0.1)
        scripted.bang.set()
        assert wait_for(lambda: startles(app) == 1)
        assert len(loud) == 1
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()


class GatedSource:
    """A world source with the DoA gate attribute, recording nothing itself."""

    name = "doa"
    speaking = None

    def start(self, publish) -> None:
        pass

    def stop(self) -> None:
        pass


def test_the_sound_direction_source_is_gated_by_the_speaking_window(data, monkeypatch):  # noqa: F811
    enable(data, VOICE_YAML)
    source = GatedSource()
    real = app_module.load_world_plugin

    def with_source(*args):
        plugin = real(*args)
        plugin.sources.append(source)
        return plugin

    monkeypatch.setattr(app_module, "load_world_plugin", with_source)
    app = new_app()
    scripted = ScriptedMini()
    stop, thread, errors = run_app(app, scripted.mini)
    try:
        assert wait_for(lambda: app.speaker is not None and source.speaking is not None)
        assert not source.speaking(time.monotonic())  # nothing spoken yet
        until = app.speaker.push_audio_sample(np.zeros(8000, np.float32), 16000)
        assert source.speaking(until - 0.1) and source.speaking(until + 0.1)
        assert not source.speaking(until + 0.2)  # 150 ms tail
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors


def test_voice_builds_a_speech_output_that_plays_through_the_speaker(data):  # noqa: F811
    from roboemot.config.schema import RegisterParams, StateEffects
    from roboemot.voice.tts import AudioChunk

    class Tone:
        name, controls = "tone", frozenset()

        def available(self, *, minor_present=False):
            return True

        async def synthesize(self, text, delivery, *, language, voice, minor_present=False):
            yield AudioChunk(np.full(1600, 0.1, np.float32), 16000)

        def cancel(self):
            pass

        async def warm(self, language, voice):
            pass

        async def aclose(self):
            pass

    enable(data, VOICE_YAML)
    app = new_app()
    scripted = ScriptedMini()
    stop, thread, errors = run_app(app, scripted.mini)
    try:
        assert wait_for(lambda: app.speech_output is not None)
        app.speech_output.backend = Tone()
        pushed = scripted.mini.media.push_audio_sample
        app.speech_output.say(
            "Ciao.",
            language="it",
            register=RegisterParams(),
            circadian=StateEffects(),
            mood=None,
            intensity=0.0,
            minor=True,  # no face frames: the minor signal is on, an adult sentence would drop
        )
        assert wait_for(lambda: pushed.called)
        assert wait_for(lambda: app.speech.speaking(time.monotonic()))
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()


class _Quiet:
    """A VAD that never hears speech (the listener runs, the hub counts frames)."""

    rate_hz, frame_samples = 16000, 512

    def speech_prob(self, frame):
        return 0.0

    def reset(self):
        return None
