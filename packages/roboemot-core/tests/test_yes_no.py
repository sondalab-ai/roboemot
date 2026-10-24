"""SPEC 7, 8 stage 3: a nod on a spoken yes, a shake on a spoken no, unless it answers the robot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from roboemot.affect import AffectState
from roboemot.affect.expression import ExpressionRenderer
from roboemot.affect.register import Person, resolve_register
from roboemot.config import ConfigStack
from roboemot.config.schema import FastpathConfig, RoboEmotConfig
from roboemot.manifest import load_manifest
from roboemot.voice.fastpath import FastPath
from roboemot.voice.yes_no import YesNoGestures

MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
)
FP = FastpathConfig()
WINDOW = FP.answer_window_s
ASKED_END = 100.0  # the stamp at which the robot's question finished playing
GESTURES = ("nod", "shake")


@pytest.fixture(scope="module")
def setup():
    manifest = load_manifest(MANIFEST, allow_unmeasured_dynamics=True)
    stack = ConfigStack()
    stack.set_manifest_behaviour(manifest.model.behaviour, manifest.channels)
    return manifest, stack.effective()


class Rig:
    """The fast path wired to the handler and a real renderer, as the app does it."""

    def __init__(self, manifest, cfg) -> None:
        self.cfg = cfg
        self.renderer = ExpressionRenderer(
            cfg.expression,
            cfg.expression_map,
            manifest.dofs,
            0.15,
            cfg.gesture_map,
            body_yaw="body_yaw",
        )
        self.now = 10.0
        self.question_end: float | None = None
        self.speaking = False
        self.events: list[str] = []
        self.handler = YesNoGestures(
            self.renderer.request,
            lambda: self.question_end,
            lambda now: self.speaking,
            FP,
        )
        self.fast = FastPath(
            cfg.voice,
            lambda event, now, payload: (
                self.events.append(event) if event.startswith("speech.keyword") else None
            ),
            clock=lambda: self.now,
            on_answer=self.handler,
        )
        base = RoboEmotConfig()
        self.register = resolve_register([Person({})], base.register, base.persona)
        self.state = AffectState(0.0, 0.0, "calm", 0.5, 0)
        self.renderer.update(self.state, self.register, 0.0)  # the first state only sets the label

    def say(self, text: str, *, partials=(), addressed=True) -> str | None:
        """Run an utterance through the fast path and one renderer update; the nod or shake
        that started, or None."""
        self.fast.begin(self.now, self.now, "en")
        for p in partials:
            self.fast.partial(p, self.now)
        result = self.fast.end(
            text, np.zeros(1600, np.float32), 16000, last_speech_stamp=self.now,
            endpoint_stamp=self.now,
        )  # fmt: skip
        self.fast.impulses(result, addressed)
        started = self.renderer.update(self.state, self.register, self.now)
        label = self.renderer.last_label
        return label if started and label in GESTURES else None


@pytest.fixture()
def rig(setup):
    return Rig(*setup)


def test_a_spoken_yes_nods_and_the_affect_event_goes_out_as_before(rig):
    assert rig.say("yes") == "nod"
    assert rig.events == ["speech.keyword.yes"]


def test_a_spoken_no_shakes(rig):
    assert rig.say("no") == "shake"
    assert rig.events == ["speech.keyword.no"]


def test_partials_never_gesture(rig):
    assert rig.say("hello there friend and more", partials=["yes", "yes please"]) is None
    assert "speech.keyword.yes" in rig.events  # the fast reaction (affect) stays


def test_yes_and_no_in_one_utterance_gesture_neither(rig):
    assert rig.say("yes no") is None
    assert rig.events == ["speech.keyword.no", "speech.keyword.yes"]


def test_a_long_utterance_is_a_sentence_not_an_answer(rig):
    words = FP.gesture_max_words
    assert rig.say(" ".join(["yes"] + ["well"] * words)) is None  # one word over
    rig.now += 5
    assert rig.say(" ".join(["yes"] + ["well"] * (words - 1))) == "nod"  # at the limit


def test_an_utterance_not_addressed_to_the_robot_does_not_gesture(rig):
    assert rig.say("yes", addressed=False) is None
    assert rig.events == ["speech.keyword.yes"]


def test_no_gesture_while_the_robot_speaks(rig):
    rig.speaking = True  # its own "Certo!" through the microphone, or talking over it
    assert rig.say("yes") is None
    assert rig.events == ["speech.keyword.yes"]


@pytest.mark.parametrize("word", ["yes", "no"])
def test_no_gesture_while_or_just_after_the_robot_asks_a_question(rig, word):
    rig.question_end = rig.now + 1.0  # still asking: the answer barges in
    assert rig.say(word) is None
    rig.now = rig.question_end + WINDOW - 0.5
    assert rig.say(word) is None
    assert len(rig.events) == 2  # both still reach the affect engine


@pytest.mark.parametrize("word,gesture", [("yes", "nod"), ("no", "shake")])
def test_the_gesture_comes_back_after_the_window(rig, word, gesture):
    rig.question_end = ASKED_END
    rig.now = ASKED_END + WINDOW + 0.5
    assert rig.say(word) == gesture


def test_a_statement_after_the_question_ends_the_window(rig):
    rig.question_end = None  # SpeechOutput.question_end is None once a non-question was spoken
    rig.now = ASKED_END + 1.0
    assert rig.say("yes") == "nod"


def test_a_reloaded_window_applies_at_once(rig):
    rig.question_end = ASKED_END
    rig.now = ASKED_END + 3.0
    assert rig.say("yes") is None  # default window 6 s
    rig.handler.apply_config(FP.model_copy(update={"answer_window_s": 1.0}))
    rig.now += 5.0
    assert rig.say("yes") == "nod"


def test_a_nod_inside_the_min_interval_is_dropped_not_played_late(rig):
    assert rig.say("yes") == "nod"
    rig.now += rig.cfg.expression.min_interval_s / 2
    assert rig.say("no") is None
    assert rig.renderer._requested is None  # nothing queued to fire at the next tick


def test_the_renderer_still_decides_at_rest(rig):
    rig.renderer.circadian_factor = lambda: 0.0
    assert rig.say("yes") is None
    assert rig.events == ["speech.keyword.yes"]


def test_a_failing_source_means_no_gesture_and_no_exception():
    calls = []

    def broken() -> float | None:
        raise RuntimeError("boom")

    YesNoGestures(lambda g, n: calls.append(g) or True, broken, lambda n: False, FP)("yes", 1.0)
    assert calls == []
