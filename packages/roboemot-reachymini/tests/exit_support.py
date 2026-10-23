"""Shared probe for the M2 exit tests (SPEC 14): one person walks in, record how the robot reacts.

Used by the sim test (``test_sim.py``, real robot state) and the app-level test
(``test_app_exit.py``, the driver's commanded targets). Nothing here sleeps for a fixed time: it
waits on published bus state and on the app's own counters. No LLM is involved on this path.

Two phases per person:

- ``arrival``: the person walks into view of a settled robot (nobody in view, no expression
  playing). The appraisal rules, the register and the expression map do the rest.
- ``provoked``: the same labelled curiosity impulse is fed to the affect engine. The impulse is
  the same for everyone but lands on top of each person's own state, so the affect term differs
  a little; ``check_exit`` divides it out of the expression factor to compare the register alone
  (amplitude, approach gate).

One behaviour for everyone (owner decision 2026-10-23): a known adult and a known child get the
same register and the same reaction; a stranger differs only by what the affect rules say.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from roboemot.config.schema import RoboEmotConfig

# A body_yaw channel on the labels the exit test provokes. The shipped expression map has no body
# channel; with it, the body moves wherever the register allows approach (SPEC 7, 12), which the
# one register does for everyone.
BODY_MAP_YAML = """\
expression_map:
  calm:
    channels:
      body_yaw: [[0.0, 0], [2.0, 12], [4.0, 0]]
  joy:
    channels:
      body_yaw: [[0.0, 0], [0.15, 12], [0.3, -12], [0.45, 0]]
  curiosity:
    channels:
      body_yaw: [[0.0, 0], [0.5, 12], [1.8, 12], [2.4, 0]]
"""

ARRIVAL_LABEL = {"adult": "calm", "child": "calm", "stranger": "curiosity"}
SETTLED_DOFS = ("head.yaw", "body_yaw", "antenna_right", "antenna_left")
SETTLED_RAD = math.radians(3.0)  # head, body and antennas within this of home: settled
QUIET_MARGIN_S = 1.5  # beyond the label hold: the follow-on expression has started
PROVOKE = (0.3, 0.3, "curiosity")  # valence, arousal, label of the provoked phase


@dataclass
class Phase:
    """What one reaction looked like. Angles are radians away from the manifest home."""

    label: str = ""  # affect label when the reaction was observed
    valence: float = 0.0
    arousal: float = 0.0
    affect_intensity: float = 0.0
    people: int = 0  # register.active
    motion_amplitude: float = 0.0
    reaction_intensity: str = ""
    approach_allowed: bool = True
    expression_label: str = ""
    expression_amplitude: float = 0.0  # the renderer's factor
    antenna_peak: float = 0.0
    head_peak: float = 0.0  # max of |yaw|, |pitch|, |roll| away from home
    head_yaw_peak: float = 0.0
    body_yaw_peak: float = 0.0
    played: tuple[str, ...] = ()  # every expression started during the phase, in order


@dataclass
class Run:
    case: str
    arrival: Phase
    provoked: Phase
    registers: list[int]  # people counts of the register.active published, in order
    idle_antenna: float = 0.0  # antenna deviation of the settled robot before the arrival


class Probe:
    """Samples ``pose()`` on a thread and notes every expression the renderer starts."""

    def __init__(self, app, pose: Callable[[], dict[str, float]], home: dict[str, float]) -> None:
        self.app, self.pose, self.home = app, pose, home
        self.starts: list[tuple[int, str, float]] = []  # n, label, amplitude
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="exit-probe", daemon=True)
        self.reset()

    def reset(self) -> None:
        self.antenna = self.head_yaw = self.head = self.body = 0.0
        self.spans: dict[
            int, list[float]
        ] = {}  # expression n -> [antenna, body] peaks while it ran

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)

    def _dev(self, state: dict[str, float], name: str) -> float:
        return abs(state.get(name, self.home[name]) - self.home[name])

    def _run(self) -> None:
        count = self.app.expression.started
        while not self._stop.is_set():
            ex = self.app.expression
            if ex.started != count:
                count = ex.started
                self.starts.append((count, ex.last_label, ex.last_amplitude))
            try:
                state = self.pose()
            except Exception:
                state = {}
            if state:
                yaw, pitch, roll = (
                    self._dev(state, n) for n in ("head.yaw", "head.pitch", "head.roll")
                )
                self.antenna = max(
                    self.antenna,
                    self._dev(state, "antenna_right"),
                    self._dev(state, "antenna_left"),
                )
                self.head_yaw = max(self.head_yaw, yaw)
                self.head = max(self.head, yaw, pitch, roll)
                self.body = max(self.body, self._dev(state, "body_yaw"))
                span = self.spans.setdefault(count, [0.0, 0.0])
                span[0] = max(
                    span[0], self._dev(state, "antenna_right"), self._dev(state, "antenna_left")
                )
                span[1] = max(span[1], self._dev(state, "body_yaw"))
            time.sleep(0.01)


def wait(predicate: Callable[[], bool], timeout: float, what: str) -> None:
    """Poll ``predicate`` until true; fail naming ``what`` after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")


def _quiet(app, states: list, window_s: float) -> None:
    """Wait until nothing has happened for ``window_s``: no expression playing, none started and
    no labelled impulse. The label held by the last impulse (``affect.event_label_hold_s``) has
    then ended and its follow-on expression, if any, has played, so the next expression the
    renderer starts is the one the next trigger causes."""
    last = [time.monotonic(), app.expression.started, states[-1].impulse if states else 0]

    def quiet() -> bool:
        now = time.monotonic()
        seen = [app.expression.started, states[-1].impulse if states else 0]
        if seen != last[1:] or app.expression.playing:
            last[:] = [now, *seen]
        return now - last[0] >= window_s

    wait(quiet, 30, "the robot to be quiet")


def _phase(
    app, probe: Probe, states: list, expect: str, trigger: Callable[[], None], quiet_s: float = 0.0
) -> Phase:
    """Run ``trigger`` and record the reaction: the affect state with ``expect`` as its label
    after a new labelled impulse, and the expression the renderer started for it. With
    ``quiet_s`` the robot is first left alone for that long (see ``_quiet``)."""
    if quiet_s:
        _quiet(app, states, quiet_s)
    probe.reset()
    n_start = app.expression.started
    n_imp = states[-1].impulse if states else 0
    mark = len(states)
    trigger()
    wait(
        lambda: any(s.impulse > n_imp and s.label == expect for s in states[mark:]),
        30,
        f"affect {expect!r} after a labelled impulse",
    )
    hit = next(s for s in states[mark:] if s.impulse > n_imp and s.label == expect)
    wait(
        lambda: any(s[0] > n_start and s[1] == expect for s in probe.starts),
        15,
        f"an expression for {expect!r}",
    )
    started = next(s for s in probe.starts if s[0] > n_start and s[1] == expect)
    seen = False

    def ended() -> bool:
        nonlocal seen
        seen = seen or app.expression.playing
        return seen and not app.expression.playing

    wait(ended, 15, "the expression to end")
    reg = app.register_tracker.active
    assert reg is not None
    return Phase(
        label=hit.label,
        valence=hit.valence,
        arousal=hit.arousal,
        affect_intensity=hit.intensity,
        people=reg.people,
        motion_amplitude=reg.params.motion_amplitude,
        reaction_intensity=reg.params.reaction_intensity,
        approach_allowed=reg.params.approach_allowed,
        expression_label=started[1],
        expression_amplitude=started[2],
        antenna_peak=probe.spans[started[0]][0],  # while this expression ran, not the next one
        head_peak=probe.head,
        head_yaw_peak=probe.head_yaw,
        body_yaw_peak=probe.spans[started[0]][1],
        played=tuple(s[1] for s in probe.starts if s[0] > n_start),
    )


def react(
    app,
    pose: Callable[[], dict[str, float]],
    appear: Callable[[], None],
    case: str,
    home: dict[str, float],
    *,
    ready_timeout: float = 60.0,
) -> Run:
    """Let the robot settle with nobody in view, call ``appear()``, record both phases.

    Settled: nobody in view and no expression playing, so the arrival is what the robot reacts
    to, not the startup.
    """
    states, registers = [], []
    app.bus.subscribe("affect.state", lambda m: states.append(m.payload))
    app.bus.subscribe("register.active", lambda m: registers.append(m.payload.people))
    wait(
        lambda: (
            app.register_tracker is not None
            and app.register_tracker.active is not None
            and app.register_tracker.active.people == 0
        ),
        ready_timeout,
        "the register with nobody in view",
    )
    wait(lambda: states and not app.expression.playing, 20, "the startup expression to end")
    wait(
        lambda: all(abs(pose().get(n, home[n]) - home[n]) < SETTLED_RAD for n in SETTLED_DOFS),
        30,
        "the robot to be back at home",
    )
    probe = Probe(app, pose, home)
    probe.start()
    try:
        probe.reset()
        time.sleep(1.5)  # the settled robot's own antenna motion (breathing), the noise floor
        idle = probe.antenna
        arrival = _phase(app, probe, states, ARRIVAL_LABEL[case], appear)
        valence, arousal, label = PROVOKE
        provoked = _phase(
            app,
            probe,
            states,
            label,
            lambda: app.affect_runner.engine.impulse(time.monotonic(), valence, arousal, label),
            quiet_s=app.affect_runner.engine.cfg.event_label_hold_s + QUIET_MARGIN_S,
        )
    finally:
        probe.stop()
    return Run(case, arrival, provoked, registers, idle)


def format_run(run: Run) -> str:
    """Two lines per run, for ``-s`` output and the M2 report."""
    d = math.degrees
    lines = []
    for name, p in (("arrival", run.arrival), ("provoked", run.provoked)):
        lines.append(
            f"{run.case:8s} {name:8s} label={p.label} val={p.valence:+.2f} aro={p.arousal:+.2f} "
            f"int={p.affect_intensity:.2f} people={p.people} amp={p.motion_amplitude} "
            f"react={p.reaction_intensity} approach={p.approach_allowed} "
            f"expr={p.expression_label}@{p.expression_amplitude:.3f} "
            f"antenna={d(p.antenna_peak):.2f} head={d(p.head_peak):.2f} "
            f"head_yaw={d(p.head_yaw_peak):.2f} body_yaw={d(p.body_yaw_peak):.2f} (deg) "
            f"idle_antenna={d(run.idle_antenna):.2f} played={'/'.join(p.played)}"
        )
    return "\n".join(lines)


def check_exit(adult: Run, child: Run, stranger: Run, adult_yaw_max: float) -> None:
    """The M2 exit criterion's first half (SPEC 14, owner decision 2026-10-23): the robot reacts
    distinctly to a known person and to a stranger, and the same to a known adult and a known
    child. Shared by the app-level and the simulated robot tests; ``adult_yaw_max`` bounds the
    known people's head yaw (radians) for the caller's setup."""
    deg = math.radians
    known = (adult, child)
    # affect: a known person is calm whatever the age, a stranger raises curiosity
    assert (adult.arrival.label, child.arrival.label, stranger.arrival.label) == (
        "calm",
        "calm",
        "curiosity",
    )
    for r in known:
        assert r.arrival.valence > stranger.arrival.valence, r
    assert abs(adult.arrival.valence - child.arrival.valence) < 0.05
    # one register for everyone: the same parameters, approach allowed, nobody damped
    for r in (adult, child, stranger):
        for p in (r.arrival, r.provoked):
            assert (p.motion_amplitude, p.reaction_intensity, p.approach_allowed) == (
                1.0,
                "medium",
                True,
            ), r
    # the same curiosity impulse for everyone: the register term alone (the factor with the affect
    # term, floor + (1 - floor) x intensity, divided out) is the same for all three
    floor = RoboEmotConfig().expression.intensity_floor
    terms = [
        r.provoked.expression_amplitude
        / (floor + (1 - floor) * min(1.0, r.provoked.affect_intensity))
        for r in (adult, child, stranger)
    ]
    assert max(terms) - min(terms) < 1e-6 and min(terms) > 0, terms
    # the approach gate is open for everyone: the body moved for each provoked expression
    for r in (adult, child, stranger):
        assert r.provoked.body_yaw_peak > deg(1.0), r
    for r in (adult, child, stranger):  # every arrival moved the antennas, well over the idle noise
        assert r.arrival.antenna_peak > max(10 * r.idle_antenna, deg(0.5)), r
    for r in known:
        assert r.provoked.head_yaw_peak <= adult_yaw_max, r.provoked.head_yaw_peak
