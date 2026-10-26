"""The self-soothing runner wired into the app (SPEC 7B, 8, 7A), against a mocked SDK client:
a sound plays while a person sits in view, with its emotion, and a queued sentence cuts it."""

from __future__ import annotations

import random
import time
from types import SimpleNamespace

import pytest
from roboemot.config.schema import RegisterParams, SootheConfig, StateEffects
from roboemot.soothe import SootheRunner
from roboemot_reachymini import app as app_module
from test_app_perception import (  # noqa: F401  (data is a fixture)
    AppWithFakes,
    data,
    new_app,
    wait_for,
)
from test_app_reflex import ScriptedMini, enable, run_app

NOON = 12.0
LABEL = "calm"
LONG_S = 8
LOW_JITTER_DRAW = -1.0  # u = -1: the first sound comes interval * (1 - jitter) = 3 s after idle
YAML = f"""voice:
  enabled: true
circadian:
  light_source: clock_only
soothe:
  interval_s: 30
  jitter: 0.9
  idle_min_s: 0
  variants:
    - {{label: {LABEL}, sound: hum, shape: {{length_s: [{LONG_S}, {LONG_S}]}}}}
"""


class NoonClock:
    """Daylight, real monotonic time: no quiet hours, no night rest."""

    monotonic = staticmethod(time.monotonic)

    def hour(self) -> float:
        return NOON


class SootheApp(AppWithFakes):
    def make_circadian_clock(self):
        return NoonClock()


class EarlyRandom(random.Random):
    """The soothe runner's generator: the first draw (the schedule's jitter) is the shortest
    gap, every later one (the synth's) is random."""

    drawn = False

    def uniform(self, a, b):
        if not self.drawn:
            self.drawn = True
            return LOW_JITTER_DRAW
        return super().uniform(a, b)


def test_a_sound_plays_next_to_a_visible_person_and_a_queued_sentence_cuts_it(data, monkeypatch):  # noqa: F811
    monkeypatch.setattr(app_module, "random", SimpleNamespace(Random=EarlyRandom))
    enable(data, YAML)  # validated settings; the 3 s first gap is the schema's shortest
    mini = ScriptedMini().mini
    app = new_app(SootheApp)  # the fake camera keeps one face in view
    stop, thread, errors = run_app(app, mini)
    try:
        assert wait_for(lambda: app.soothe_runner is not None and app.speech_output is not None)
        events = []
        runner, real = app.soothe_runner, app.soothe_runner.on_event
        runner.on_event = lambda name, now, payload: (
            events.append((name, payload)),
            real(name, now, payload),
        )
        cancelled = []
        output, real_cancel = app.speech_output, app.speech_output.cancel
        output.cancel = lambda job: (cancelled.append(job), real_cancel(job))[1]
        assert wait_for(lambda: events and mini.media.push_audio_sample.called, timeout=30)
        assert wait_for(lambda: app.selector.target is not None, timeout=15)  # a person is in view
        assert events[0] == ("self.soothe", {"label": LABEL, "sound": "hum"})
        assert wait_for(lambda: app.expression.last_label == LABEL)
        assert not cancelled  # the person did not cut it, nor did its own speaking window
        output.say(
            "Ciao.",
            language="it",
            register=RegisterParams(),
            circadian=StateEffects(),
            mood=None,
            intensity=0.0,
        )
        assert wait_for(lambda: cancelled, timeout=15)
        assert mini.media.stop_playing.called  # the sink was flushed (no clear_player here)
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()


def context(**over):
    """A ``SootheContext`` for a quiet room, with ``over`` changed."""
    base = {"in_session": False, "voice_recent": False, "other_queued": 0}
    return app_module.SootheContext(**{"speaking": False, "ours": False, **base, **over})


@pytest.mark.parametrize(
    "over",
    [
        {"in_session": True},
        {"voice_recent": True},
        {"other_queued": 1},
        {"speaking": True},
    ],
)
def test_interaction_or_other_audio_ends_idle(over):
    assert app_module.soothe_idle(context()) and not app_module.soothe_idle(context(**over))


def test_our_own_sound_is_not_activity_but_a_session_or_queue_still_is():
    own = {"speaking": True, "ours": True}
    assert app_module.soothe_idle(context(**own))
    assert not app_module.soothe_idle(context(**own, other_queued=1))
    assert not app_module.soothe_idle(context(**own, in_session=True))


def soothe_args(**over):
    base = {
        "selector": SimpleNamespace(
            target=object(), engaged=True, speaker_active=lambda now: False
        ),
        "speech": SimpleNamespace(speaking=lambda now: False),
        "output": SimpleNamespace(waiting=lambda exclude=None: 0),
        "soothe": SimpleNamespace(
            job=None,
            sounding=lambda now, tail=0.0: False,
            cfg=SimpleNamespace(speech_quiet_s=10.0),
        ),
        "in_session": False,
        "last_voice": float("-inf"),
        "echo_tail_s": 0.3,
    }
    return app_module.soothe_context(100.0, **{**base, **over})


def test_a_visible_face_and_an_attention_target_with_no_speech_keep_idle():
    assert app_module.soothe_idle(soothe_args())


def test_a_speaker_hint_or_a_recent_utterance_ends_idle_until_the_quiet_window_passes():
    hinted = SimpleNamespace(target=None, speaker_active=lambda now: True)
    assert not app_module.soothe_idle(soothe_args(selector=hinted))
    assert not app_module.soothe_idle(soothe_args(last_voice=95.0))
    assert app_module.soothe_idle(soothe_args(last_voice=85.0))


class BrokenOutput:
    def __init__(self):
        self.cancelled = []

    def play(self, samples, rate_hz, volume_db=0.0):
        raise RuntimeError("sink gone")

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def live(self, job_id):
        return True


def test_a_failing_soothe_tick_is_logged_and_does_not_stop_the_app(caplog):
    cfg = SootheConfig(interval_s=30, jitter=0, idle_min_s=0)
    output = BrokenOutput()
    runner = SootheRunner(cfg, output, on_event=lambda *a: None, rng=random.Random(1))
    quiet = app_module.SootheContext(False, False, 0, False, False)
    with caplog.at_level("WARNING"):
        for t in (0.0, cfg.interval_s):  # the second tick is due: play raises
            app_module.soothe_tick(runner, t, lambda: quiet, "awake", False)
        app_module.soothe_tick(runner, 1.0, lambda: 1 / 0, "awake", False)  # the context fails
    assert "sink gone" in caplog.text and "ZeroDivisionError" in caplog.text
