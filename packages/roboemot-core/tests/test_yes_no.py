"""SPEC 7, 8 stage 3: a nod on a spoken yes, a shake on a spoken no, unless it answers the robot."""

from __future__ import annotations

from pathlib import Path

import pytest
from roboemot.affect import AffectState
from roboemot.affect.expression import ExpressionRenderer
from roboemot.affect.register import Person, resolve_register
from roboemot.config import ConfigStack
from roboemot.config.schema import FastpathConfig, RoboEmotConfig
from roboemot.manifest import load_manifest
from roboemot.voice.yes_no import YesNoGestures

MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
)
WINDOW = FastpathConfig().answer_window_s
ASKED_END = 100.0  # the stamp at which the robot's question finished playing


@pytest.fixture(scope="module")
def setup():
    manifest = load_manifest(MANIFEST, allow_unmeasured_dynamics=True)
    stack = ConfigStack()
    stack.set_manifest_behaviour(manifest.model.behaviour)
    return manifest, stack.effective()


class Rig:
    """The handler wired to a real renderer, with the robot's question end settable."""

    def __init__(self, manifest, cfg) -> None:
        self.renderer = ExpressionRenderer(
            cfg.expression, cfg.expression_map, manifest.dofs, 0.15, cfg.gesture_map
        )
        self.question_end: float | None = None
        self.events: list[str] = []
        self.handler = YesNoGestures(
            lambda event, now, payload: self.events.append(event),
            self.renderer.request,
            lambda: self.question_end,
            lambda: WINDOW,
        )
        base = RoboEmotConfig()
        self.register = resolve_register(
            [Person("adult", "declared", {})], base.register, base.persona
        )
        self.state = AffectState(0.0, 0.0, "calm", 0.5, 0)
        self.renderer.update(self.state, self.register, 0.0)  # the first state only sets the label

    def hear(self, event: str, now: float) -> str | None:
        """Emit the event, run one renderer update; the nod or shake that started, or None."""
        self.handler(event, now, {"language": "en"})
        started = self.renderer.update(self.state, self.register, now)
        label = self.renderer.last_label
        return label if started and label in ("nod", "shake") else None


@pytest.fixture()
def rig(setup):
    return Rig(*setup)


def test_a_spoken_yes_nods(rig):
    assert rig.hear("speech.keyword.yes", 10.0) == "nod"
    assert rig.events == ["speech.keyword.yes"]  # the affect event goes on as before


def test_a_spoken_no_shakes(rig):
    assert rig.hear("speech.keyword.no", 10.0) == "shake"
    assert rig.events == ["speech.keyword.no"]


def test_other_events_are_forwarded_without_a_gesture(rig):
    assert rig.hear("speech.keyword.greeting", 10.0) is None
    assert rig.events == ["speech.keyword.greeting"]


@pytest.mark.parametrize("event", ["speech.keyword.yes", "speech.keyword.no"])
def test_no_gesture_while_or_just_after_the_robot_asks_a_question(rig, event):
    rig.question_end = ASKED_END
    assert rig.hear(event, ASKED_END - 1.0) is None  # still asking: the answer barges in
    assert rig.hear(event, ASKED_END + WINDOW - 0.5) is None
    assert rig.events == [event, event]  # still reaches the affect engine


@pytest.mark.parametrize(
    "event,gesture", [("speech.keyword.yes", "nod"), ("speech.keyword.no", "shake")]
)
def test_the_gesture_comes_back_after_the_window(rig, event, gesture):
    rig.question_end = ASKED_END
    assert rig.hear(event, ASKED_END + WINDOW + 0.5) == gesture


def test_a_statement_after_the_question_ends_the_window(rig):
    rig.question_end = None  # SpeechOutput.question_end is None once a non-question was spoken
    assert rig.hear("speech.keyword.yes", ASKED_END + 1.0) == "nod"


def test_the_renderer_still_decides(rig):
    rig.renderer.circadian_factor = lambda: 0.0  # resting: the request is refused
    assert rig.hear("speech.keyword.yes", 10.0) is None
    assert rig.events == ["speech.keyword.yes"]
