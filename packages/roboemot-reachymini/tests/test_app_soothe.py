"""The self-soothing runner wired into the app (SPEC 7B, 8, 7A), against a mocked SDK client:
an idle room gets a sound on the speaker and the matching emotion; a person cuts it."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from roboemot.soothe import SootheRunner
from roboemot_reachymini import app as app_module
from test_app_perception import (  # noqa: F401  (data is a fixture)
    AppWithFakes,
    FakeDetector,
    data,
    new_app,
    wait_for,
)
from test_app_reflex import ScriptedMini, enable, run_app

NOON = 12.0
LABELS = {"calm", "joy", "curiosity", "sleepy"}
SHORT_S = 0.4
LONG_S = 8.0
OUTPUT_RATE_HZ = 16000
LONG_INTERVAL_S = 1000.0


class NoonClock:
    """Daylight, real monotonic time: no quiet hours, no night rest."""

    monotonic = staticmethod(time.monotonic)

    def hour(self) -> float:
        return NOON


class SootheApp(AppWithFakes):
    def make_circadian_clock(self):
        return NoonClock()


def fast(runner: SootheRunner, length_s: float) -> None:
    """Sounds every fraction of a second. ``soothe.interval_s`` has a floor of 30 s in the
    schema, so the test swaps in an unvalidated copy."""
    shape = {"length_s": (length_s, length_s)}
    variants = [{"label": label, "sound": "hum", "shape": shape} for label in sorted(LABELS)]
    runner.apply_config(
        runner.cfg.model_copy(
            update={
                "interval_s": 0.2,
                "jitter": 0.0,
                "idle_min_s": 0.0,
                "cooldown_s": 0.1,
                "variants": [type(runner.cfg.variants[0]).model_validate(v) for v in variants],
            }
        )
    )


def boot(data, monkeypatch, empty=True):  # noqa: F811
    monkeypatch.setattr(FakeDetector, "empty", empty)
    enable(data, "voice:\n  enabled: true\ncircadian:\n  light_source: clock_only\n")
    scripted = ScriptedMini()
    app = new_app(SootheApp)
    stop, thread, errors = run_app(app, scripted.mini)
    assert wait_for(lambda: app.soothe_runner is not None and app.speech_output is not None)
    return app, scripted.mini, stop, thread, errors


def test_an_idle_room_gets_a_sound_on_the_speaker_and_the_matching_emotion(data, monkeypatch):  # noqa: F811
    app, mini, stop, thread, errors = boot(data, monkeypatch)
    try:
        events = []
        runner, real = app.soothe_runner, app.soothe_runner.on_event

        def spy(name, now, payload):
            events.append((name, payload))
            real(name, now, payload)

        runner.on_event = spy
        fast(runner, SHORT_S)
        assert wait_for(lambda: mini.media.push_audio_sample.called)
        assert wait_for(lambda: events)
        name, payload = events[0]
        assert name == "self.soothe" and payload["label"] in LABELS and payload["sound"] == "hum"
        pushed = mini.media.push_audio_sample.call_args_list
        assert sum(c.args[0].size for c in pushed) >= SHORT_S * OUTPUT_RATE_HZ * 0.9
        assert wait_for(lambda: app.expression.last_label == payload["label"])
        assert app.speech_output.snapshot()["spoken"] == 0  # a raw sound is not speech
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()


def test_a_person_appearing_cuts_the_sound(data, monkeypatch):  # noqa: F811
    app, mini, stop, thread, errors = boot(data, monkeypatch)
    try:
        cancelled = []
        output = app.speech_output
        real = output.cancel
        output.cancel = lambda job_id: (cancelled.append(job_id), real(job_id))[1]
        fast(app.soothe_runner, LONG_S)
        assert wait_for(lambda: mini.media.push_audio_sample.called)
        assert not cancelled
        monkeypatch.setattr(FakeDetector, "empty", False)  # a face walks in
        assert wait_for(lambda: cancelled, timeout=30)
        assert mini.media.stop_playing.called  # the sink was flushed (no clear_player here)
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()


def test_the_sound_does_not_cancel_itself_through_the_speaking_window(data, monkeypatch):  # noqa: F811
    app, mini, stop, thread, errors = boot(data, monkeypatch)
    try:
        cancelled = []
        output = app.speech_output
        real = output.cancel
        output.cancel = lambda job_id: (cancelled.append(job_id), real(job_id))[1]
        fast(app.soothe_runner, 1.5)
        assert wait_for(lambda: mini.media.push_audio_sample.called)
        app.soothe_runner.apply_config(
            app.soothe_runner.cfg.model_copy(update={"interval_s": LONG_INTERVAL_S})
        )  # one sound only
        assert wait_for(lambda: app.speech.speaking(time.monotonic()))  # our own sound
        assert wait_for(lambda: not app.speech.speaking(time.monotonic()), timeout=10)
        assert not cancelled
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()


def test_without_voice_there_is_no_runner(data, monkeypatch):  # noqa: F811
    monkeypatch.setattr(FakeDetector, "empty", True)
    app = new_app(SootheApp)
    stop, thread, errors = run_app(app, ScriptedMini().mini)
    try:
        assert wait_for(lambda: app.diagnostics is not None)
        assert app.speech_output is None and app.soothe_runner is None
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()


def idle(**over):
    """Arguments of ``soothe_idle`` for an empty room, with ``over`` changed."""
    base = {
        "attending": False,
        "selector": SimpleNamespace(target=None),
        "present": False,
        "speech": SimpleNamespace(speaking=lambda now: False),
        "output": SimpleNamespace(waiting=lambda exclude=None: 0),
        "soothe": SimpleNamespace(job=None, sounding=lambda now, tail=0.0: False),
        "echo_tail_s": 0.3,
        "in_session": False,
        "scanning": False,
    }
    return app_module.soothe_idle(0.0, **{**base, **over})


@pytest.mark.parametrize(
    "over",
    [
        {"attending": True},
        {"selector": SimpleNamespace(target=object())},
        {"present": True},
        {"in_session": True},
        {"scanning": True},
        {"speech": SimpleNamespace(speaking=lambda now: True)},
        {"output": SimpleNamespace(waiting=lambda exclude=None: 1)},
    ],
)
def test_any_activity_ends_idle(over):
    assert idle() and not idle(**over)


def test_our_own_sound_is_not_activity_but_someone_elses_audio_is():
    own = SimpleNamespace(job=7, sounding=lambda now, tail=0.0: True)
    speaking = SimpleNamespace(speaking=lambda now: True)
    assert idle(soothe=own, speech=speaking)
    seen = []
    queue = SimpleNamespace(waiting=lambda exclude=None: seen.append(exclude) or 1)
    assert not idle(soothe=own, speech=speaking, output=queue) and seen == [7]
