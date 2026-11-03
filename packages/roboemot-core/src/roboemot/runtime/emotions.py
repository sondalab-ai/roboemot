"""The web UI's Emotions tab: see and test the robot's affect (SPEC 7, 11, 11A, 12).

``EmotionsState.state()`` builds one small JSON-able dict from parts that keep their state in
memory: the affect engine (state, baseline and its history of impulses), the register tracker's
active register, the persona's effects, the circadian runner and the expression renderer. Like
``LiveState`` it never calls the heavy debug sources and never reads the configuration stack:
``apply_config`` (a stack subscriber) keeps a copy. Codes and numbers only: no frames, audio,
embeddings, registry ids or names, and the history holds no payload.

Two test controls, behind the operator PIN and CSRF (the API checks them):

- ``play(label)``: an ``expression_map`` label or a named gesture of the manifest, asked for
  through ``ExpressionRenderer.request`` exactly as the ``play`` skill does (SPEC 11), so the
  minimum interval, the register, the persona, circadian scaling, the body rule and then the
  motion loop's safety limiter all apply. The answer says whether it was queued.
- ``inject(rule)``: the appraisal rule with that index is fired as if its event happened (its
  ``when`` as the payload), through the engine's own ``on_event``, so the cooldown applies (and
  starts: a real event right after is held back like after any other).

Both refuse while the robot is asleep, while the motors are off and while the motion loop has
yielded to external control (``REFUSALS``): a test must not fight a hand or the platform.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from roboemot.config.schema import RoboEmotConfig
from roboemot.persona import effects as persona_effects

ROUND_DIGITS = 3  # decimals kept in the numbers sent to the page
REFUSALS = ("robot_asleep", "motors_off", "external_control")  # why the controls are off


class EmotionsError(Exception):
    """A control was refused; ``code`` is the page's error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class EmotionsState:
    """Builds the Emotions tab's payload and runs its two controls. ``host`` and ``p`` are the
    running ``RuntimeHost`` and ``Pipeline``; ``clock`` is the app loop's clock, the one the
    affect engine and the renderer run on."""

    def __init__(
        self, host: Any, p: Any, cfg: RoboEmotConfig, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._host, self._p, self._clock = host, p, clock
        self._cfg = cfg
        self._keep(cfg)

    def apply_config(self, change: Any) -> None:
        """``ConfigStack`` subscriber: keep the settings this view reads (hot reload)."""
        self._cfg = change.config
        self._keep(change.config)

    def close(self) -> None:
        """The web UI stopped (or never started): the engine keeps no history any more."""
        self._p.runner.engine.keep_history(0)

    def _keep(self, cfg: RoboEmotConfig) -> None:
        """The engine keeps a history only while the tab is on."""
        page = cfg.webui.emotions
        self._p.runner.engine.keep_history(page.events if page.enabled else 0)

    @property
    def enabled(self) -> bool:
        return self._cfg.webui.emotions.enabled

    def state(self) -> dict[str, Any] | None:
        """The payload, or None while ``webui.emotions.enabled`` is off."""
        cfg = self._cfg
        if not cfg.webui.emotions.enabled:
            return None
        now = self._clock()
        refused = self.refused()
        return {
            "poll_ms": cfg.webui.emotions.poll_ms,
            "affect": self._affect(now),
            "register": self._register(),
            "persona": {"name": cfg.personality, "effects": self._persona(cfg)},
            "circadian": self._circadian(),
            "expression": self._expression(),
            "moved": self._moved(now),
            "controls": {"available": refused is None, "refused": refused},
            "playable": self._playable(),
            "rules": [
                {"index": i, "event": r.event, "label": r.label}
                for i, r in enumerate(cfg.appraisal.rules)
            ],
        }

    def refused(self) -> str | None:
        """Why the controls are off now (one of ``REFUSALS``), or None."""
        p = self._p
        if p.sleep.asleep:
            return "robot_asleep"
        if p.loop.motors_off:
            return "motors_off"
        if p.loop.external:
            return "external_control"
        return None

    def play(self, label: str) -> bool:
        """Ask for ``label`` as the ``play`` skill does; whether it was queued (the renderer
        refuses while resting, and starts it at its next update only after the minimum
        interval). ``EmotionsError`` when refused or unknown."""
        self._check()
        if label not in self._p.expression.playable():
            raise EmotionsError("unknown_expression")
        return bool(self._p.expression.request(label))

    def inject(self, index: int) -> bool:
        """Fire appraisal rule ``index`` as if its event happened; whether a rule fired (a rule
        in its cooldown does not). ``EmotionsError`` when refused or unknown."""
        self._check()
        rules = self._cfg.appraisal.rules
        if not 0 <= index < len(rules):
            raise EmotionsError("unknown_rule")
        rule = rules[index]
        return bool(self._p.runner.engine.on_event(rule.event, self._clock(), dict(rule.when)))

    def _check(self) -> None:
        if not self._cfg.webui.emotions.enabled:
            raise EmotionsError("emotions_disabled")
        refused = self.refused()
        if refused is not None:
            raise EmotionsError(refused)

    def _affect(self, now: float) -> dict[str, Any]:
        engine = self._p.runner.engine
        s = engine.state(now)
        return {
            "label": s.label,
            "valence": round(s.valence, ROUND_DIGITS),
            "arousal": round(s.arousal, ROUND_DIGITS),
            "intensity": round(s.intensity, ROUND_DIGITS),
            "baseline_valence": round(engine.baseline_valence, ROUND_DIGITS),
            "baseline_arousal": round(engine.baseline_arousal, ROUND_DIGITS),
        }

    def _register(self) -> dict[str, Any] | None:
        """The register in effect, persona-shaped: parameters and the count of people it covers
        (no person id, no free text)."""
        active = self._p.tracker.active
        if active is None:
            return None
        return {
            k: round(v, ROUND_DIGITS) if isinstance(v, float) else v
            for k, v in active.summary().items()
        }

    @staticmethod
    def _persona(cfg: RoboEmotConfig) -> dict[str, Any]:
        """The persona's effects on the register and the voice (``persona.effects``)."""
        return {
            k: round(v, ROUND_DIGITS) if isinstance(v, float) else v
            for k, v in asdict(persona_effects(cfg.persona)).items()
        }

    def _circadian(self) -> dict[str, Any] | None:
        """The circadian state, sleepiness and the expression amplitude factor; None while the
        circadian rhythm is off."""
        if self._host.circadian is None:
            return None
        snap = self._p.circ.snapshot()
        sleepiness = snap.get("sleepiness")
        return {
            "state": snap["state"],
            "sleepiness": None if sleepiness is None else round(sleepiness, ROUND_DIGITS),
            "amplitude": round(self._p.circ.amplitude_factor(), ROUND_DIGITS),
        }

    def _expression(self) -> dict[str, Any]:
        """The last expression started: its label, its amplitude and how many have started."""
        e = self._p.expression
        return {
            "last_label": e.last_label,
            "last_amplitude": round(e.last_amplitude, ROUND_DIGITS),
            "started": e.started,
        }

    def _moved(self, now: float) -> list[dict[str, Any]]:
        """The impulses that moved the affect, newest first, with their age."""
        return [
            {
                "age_s": round(max(0.0, now - m.t), 1),
                "event": m.event,
                "rule": m.rule,
                "label": m.label,
                "dv": round(m.dv, ROUND_DIGITS),
                "da": round(m.da, ROUND_DIGITS),
            }
            for m in reversed(self._p.runner.engine.history())
        ]

    def _playable(self) -> dict[str, list[str]]:
        """What the play control offers: the affect labels, then the named gestures."""
        e = self._p.expression
        names = e.playable()
        return {
            "labels": [n for n in names if n in e.expression_map],
            "gestures": [n for n in names if n not in e.expression_map],
        }
