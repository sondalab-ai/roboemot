"""Shared probe for the M2 exit tests (SPEC 14): one person walks in, record how the robot reacts.

Used by the sim test (``test_sim.py``, real robot state) and the app-level test
(``test_app_exit.py``, the driver's commanded targets). Nothing here sleeps for a fixed time: it
waits on published bus state and on the app's own counters. No LLM is involved on this path.

Two phases per person:

- ``arrival``: the person walks into view of a settled robot (nobody in view, minor caps
  released). The appraisal rules, the register and the expression map do the rest.
- ``provoked``: the same labelled curiosity impulse is fed to the affect engine, once the caps
  are what they will stay while the person is there. The affect input is then identical for
  everyone, so what differs is the register alone (amplitude, approach gate, minor cap).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

# A body_yaw channel on the labels the exit test provokes. The shipped expression map has no body
# channel, so without it "never approached" would hold trivially; with it, the renderer's approach
# gate is what keeps a child or a stranger from being approached (SPEC 7, 12).
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

ARRIVAL_LABEL = {"adult": "calm", "child": "joy", "stranger": "curiosity"}
SETTLED_RAD = math.radians(3.0)  # head and body yaw within this of home: settled
PROVOKE = (0.3, 0.3, "curiosity")  # valence, arousal, label of the provoked phase


@dataclass
class Phase:
    """What one reaction looked like. Angles are radians away from the manifest home."""

    label: str = ""  # affect label when the reaction was observed
    valence: float = 0.0
    arousal: float = 0.0
    affect_intensity: float = 0.0
    group: str = ""  # register.active
    motion_amplitude: float = 0.0
    reaction_intensity: str = ""
    approach_allowed: bool = True
    minor_policy: bool = False
    minor_caps: bool = False  # the motion loop's minor caps were on when the expression began
    expression_label: str = ""
    expression_amplitude: float = 0.0  # the renderer's factor, before the minor cap
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
    registers: list[str]  # register.active groups published, in order
    idle_antenna: float = 0.0  # antenna deviation of the settled robot before the arrival


class Probe:
    """Samples ``pose()`` on a thread and notes every expression the renderer starts."""

    def __init__(self, app, pose: Callable[[], dict[str, float]], home: dict[str, float]) -> None:
        self.app, self.pose, self.home = app, pose, home
        self.starts: list[tuple[int, str, float, bool]] = []  # n, label, amplitude, caps on
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
                self.starts.append(
                    (count, ex.last_label, ex.last_amplitude, self.app.minor_present)
                )
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


def _phase(app, probe: Probe, states: list, expect: str, trigger: Callable[[], None]) -> Phase:
    """Run ``trigger`` and record the reaction: the affect state with ``expect`` as its label
    after a new labelled impulse, and the expression the renderer started for it."""
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
        group=reg.group,
        motion_amplitude=reg.params.motion_amplitude,
        reaction_intensity=reg.params.reaction_intensity,
        approach_allowed=reg.params.approach_allowed,
        minor_policy=reg.params.minor_content_policy,
        minor_caps=started[3],
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

    Settled: the minor caps released (face frames without a face for the hold) and no expression
    playing, so the arrival is what the robot reacts to, not the startup.
    """
    states, registers = [], []
    app.bus.subscribe("affect.state", lambda m: states.append(m.payload))
    app.bus.subscribe("register.active", lambda m: registers.append(m.payload.group))
    wait(lambda: not app.minor_present, ready_timeout, "the minor caps to release (nobody in view)")
    wait(lambda: states and not app.expression.playing, 20, "the startup expression to end")
    wait(
        lambda: all(
            abs(pose().get(n, home[n]) - home[n]) < SETTLED_RAD for n in ("head.yaw", "body_yaw")
        ),
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
        if case == "adult":  # the caps release once the recognised adult has been seen long enough
            wait(
                lambda: (
                    not app.minor_present and app.register_tracker.active.params.approach_allowed
                ),
                30,
                "the minor caps to release for the recognised adult",
            )
        valence, arousal, label = PROVOKE
        provoked = _phase(
            app,
            probe,
            states,
            label,
            lambda: app.affect_runner.engine.impulse(time.monotonic(), valence, arousal, label),
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
            f"int={p.affect_intensity:.2f} register={p.group} amp={p.motion_amplitude} "
            f"react={p.reaction_intensity} approach={p.approach_allowed} caps={p.minor_caps} "
            f"expr={p.expression_label}@{p.expression_amplitude:.3f} "
            f"antenna={d(p.antenna_peak):.2f} head={d(p.head_peak):.2f} "
            f"head_yaw={d(p.head_yaw_peak):.2f} body_yaw={d(p.body_yaw_peak):.2f} (deg) "
            f"idle_antenna={d(run.idle_antenna):.2f} played={'/'.join(p.played)}"
        )
    return "\n".join(lines)


def check_exit(adult: Run, child: Run, stranger: Run) -> None:
    """The M2 exit criterion's first half (SPEC 14): the robot reacts distinctly to a known adult,
    a known child and a stranger. Shared by the app-level and the simulated robot tests."""
    deg = math.radians
    # The first reaction to a recognised adult is still under the minor caps: until the face is
    # recognised it is an unknown age, a possible minor, and the caps release only after a
    # full hold of frames without one (SPEC 5, 12). The register has to wait for that too.
    assert adult.arrival.minor_caps
    # affect: each arrival names its own label; the stranger is the quietest
    assert (adult.arrival.label, child.arrival.label, stranger.arrival.label) == (
        "calm",
        "joy",
        "curiosity",
    )
    assert adult.arrival.valence > stranger.arrival.valence > 0
    assert child.arrival.valence > adult.arrival.valence
    assert stranger.arrival.affect_intensity < adult.arrival.affect_intensity
    assert adult.arrival.affect_intensity < child.arrival.affect_intensity
    # register per person
    assert (adult.arrival.group, child.arrival.group, stranger.arrival.group) == (
        "adult",
        "child",
        "unknown",
    )
    assert stranger.arrival.reaction_intensity == "low"
    assert child.arrival.reaction_intensity == "high"
    assert adult.arrival.reaction_intensity == "medium"
    assert child.arrival.minor_policy and stranger.arrival.minor_policy
    assert not adult.provoked.minor_policy
    # the same joy impulse for everyone: the renderer's factor follows the register alone
    amps = [r.provoked.expression_amplitude for r in (child, adult, stranger)]
    assert amps[0] > amps[1] > amps[2] > 0
    # the minor caps are on for the child and the stranger, off for the adult once recognised
    assert child.provoked.minor_caps and stranger.provoked.minor_caps
    assert not adult.provoked.minor_caps
    # nobody who must not be approached is: no body yaw, whatever the expression map says
    assert child.arrival.approach_allowed is False and stranger.arrival.approach_allowed is False
    assert child.provoked.approach_allowed is False and stranger.provoked.approach_allowed is False
    for r in (child, stranger):
        assert r.arrival.body_yaw_peak < deg(0.5) and r.provoked.body_yaw_peak < deg(0.5), r
    assert adult.provoked.approach_allowed
    assert adult.provoked.body_yaw_peak > deg(1.0)  # the gate is open for the adult: body moved
    # the stranger's reaction is low intensity: its own arrival factor is the smallest, and the
    # same gesture (the provoked curiosity) is a fraction of the size it has for the others
    arrivals = [r.arrival.expression_amplitude for r in (child, adult, stranger)]
    assert arrivals[0] > arrivals[1] > arrivals[2] > 0
    for r in (adult, child):
        assert stranger.provoked.antenna_peak < 0.5 * r.provoked.antenna_peak, (stranger, r)
    assert all(r.arrival.antenna_peak > 0 for r in (adult, child, stranger))
