"""The self-soothing runner wired into the app (SPEC 7B, 8, 7A), against a mocked SDK client:
a sound plays while a person sits in view, with its emotion, and a queued sentence cuts it."""

from __future__ import annotations

import random
import threading
import time
from types import SimpleNamespace

import pytest
from roboemot.affect.expression import ExpressionRenderer
from roboemot.config.schema import RegisterParams, SootheConfig, StateEffects
from roboemot.runtime import startup as startup_module
from roboemot.runtime import ticks as ticks_module
from roboemot.soothe import SootheRunner
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


class SoothedApp:
    """What the test sees of the app's soothe runner and expression renderer, from the first
    tick: every soothe event (``("event", name, payload)``) and every expression started
    (``("start", label)``), in order, and the labelled impulses the renderer has taken
    (``impulses``). The runner plays nothing until ``allow`` is set.

    The face in view brings its own labelled impulse (``person.unknown_appears``: curiosity).
    In the same affect update as the sound's, the latest label wins and the sound's never
    starts; and ``last_label`` holds only the latest start, so a reader that comes late finds
    the face's or the region's label. The sound waits until the renderer took the face's."""

    def __init__(self, monkeypatch) -> None:
        self.log: list[tuple] = []
        self.allow = threading.Event()
        self.impulses = 0
        log, allow = self.log, self.allow

        class GatedSoothe(SootheRunner):
            def __init__(self, *args, on_event, **kw):
                def record(name, now, payload):
                    log.append(("event", name, payload))
                    on_event(name, now, payload)

                super().__init__(*args, on_event=record, **kw)

            def _start(self, now, state, quiet):
                if allow.is_set():  # held: still due, it plays at the first tick after
                    super()._start(now, state, quiet)

        real_update, real_start = ExpressionRenderer.update, ExpressionRenderer._start

        def update(renderer, state, register, now):
            started = real_update(renderer, state, register, now)
            self.impulses = state.impulse
            return started

        def start(renderer, label, *args):
            started = real_start(renderer, label, *args)
            if started:
                log.append(("start", label))
            return started

        monkeypatch.setattr(startup_module, "SootheRunner", GatedSoothe)
        monkeypatch.setattr(ExpressionRenderer, "update", update)
        monkeypatch.setattr(ExpressionRenderer, "_start", start)

    def events(self) -> list[tuple]:
        return [e[1:] for e in self.log if e[0] == "event"]

    def started_after_the_sound(self) -> list[str]:
        """The labels of the expressions started after the first soothe event."""
        first = next((i for i, e in enumerate(self.log) if e[0] == "event"), len(self.log))
        return [e[1] for e in self.log[first:] if e[0] == "start"]


def test_a_sound_plays_next_to_a_visible_person_and_a_queued_sentence_cuts_it(data, monkeypatch):  # noqa: F811
    monkeypatch.setattr(startup_module, "random", SimpleNamespace(Random=EarlyRandom))
    seen = SoothedApp(monkeypatch)
    enable(data, YAML)  # validated settings; the 3 s first gap is the schema's shortest
    mini = ScriptedMini().mini
    flushes = []
    mini.media.audio = SimpleNamespace(clear_player=lambda: flushes.append(1))
    app = new_app(SootheApp)  # the fake camera keeps one face in view
    stop, thread, errors = run_app(app, mini)
    try:
        assert wait_for(lambda: app.soothe_runner is not None and app.speech_output is not None)
        cancelled = []
        output, real_cancel = app.speech_output, app.speech_output.cancel
        output.cancel = lambda job: (cancelled.append(job), real_cancel(job))[1]
        assert wait_for(lambda: app.selector.target is not None, timeout=15)  # a person is in view
        assert wait_for(lambda: seen.impulses >= 1, timeout=15)  # the face's, taken
        seen.allow.set()
        assert wait_for(lambda: seen.events() and mini.media.push_audio_sample.called, timeout=30)
        assert seen.events()[0] == ("self.soothe", {"label": LABEL, "sound": "hum"})
        assert wait_for(lambda: LABEL in seen.started_after_the_sound())
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
        assert flushes  # the sink was flushed through clear_player
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()


def context(**over):
    """A ``SootheContext`` for a quiet room, with ``over`` changed."""
    base = {"in_session": False, "voice_recent": False, "other_queued": 0}
    return ticks_module.SootheContext(**{"speaking": False, "ours": False, **base, **over})


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
    assert ticks_module.soothe_idle(context()) and not ticks_module.soothe_idle(context(**over))


def test_our_own_sound_is_not_activity_but_a_session_or_queue_still_is():
    own = {"speaking": True, "ours": True}
    assert ticks_module.soothe_idle(context(**own))
    assert not ticks_module.soothe_idle(context(**own, other_queued=1))
    assert not ticks_module.soothe_idle(context(**own, in_session=True))


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
    return ticks_module.soothe_context(100.0, **{**base, **over})


def test_a_visible_face_and_an_attention_target_with_no_speech_keep_idle():
    assert ticks_module.soothe_idle(soothe_args())


def test_a_speaker_hint_or_a_recent_utterance_ends_idle_until_the_quiet_window_passes():
    hinted = SimpleNamespace(target=None, speaker_active=lambda now: True)
    assert not ticks_module.soothe_idle(soothe_args(selector=hinted))
    assert not ticks_module.soothe_idle(soothe_args(last_voice=95.0))
    assert ticks_module.soothe_idle(soothe_args(last_voice=85.0))


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
    quiet = ticks_module.SootheContext(False, False, 0, False, False)
    with caplog.at_level("WARNING"):
        for t in (0.0, cfg.interval_s):  # the second tick is due: play raises
            ticks_module.soothe_tick(runner, t, lambda: quiet, "awake", False)
        ticks_module.soothe_tick(runner, 1.0, lambda: 1 / 0, "awake", False)  # the context fails
    assert "sink gone" in caplog.text and "ZeroDivisionError" in caplog.text
