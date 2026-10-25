"""SPEC 7 (Speaking motion), 10, 12: the speaking motion renderer on a canned speech timeline.

The timeline is the real ``SpeechTimeline`` fed with synthetic tones on a fake clock; the renderer
reads it as the app tick does, and the worst cases go through ``MotionCore`` with the shipped Reachy
Mini dofs under the dev fallback dynamics (0.5 rad/s, 2 rad/s^2).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
from roboemot.affect import AffectState
from roboemot.affect.register import Person, resolve_register
from roboemot.affect.speaking import SOURCE, SpeakingRenderer
from roboemot.config import ConfigStack
from roboemot.config.schema import SpeechMotionLevels, SpeechMotionMap
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionCore, Priority
from roboemot.motion.core import Arbiter
from roboemot.voice.envelope import SpeechTimeline

MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
)
RATE = 16000
TICK = 0.05  # the 20 Hz app tick
LOOP_DT = 0.02  # the 50 Hz motion loop
PRESETS = ("subtle", "natural", "lively")
VISIBLE_SHARE = 0.8  # the limiter output reaches at least this share of the commanded peak
LAG_SHARE = 0.35  # and never trails the command by more than this share of the peak
MAIN = {
    "beat": "head.pitch",
    "phrase_nod": "head.pitch",
    "question_tilt": "head.roll",
    "accent": "antenna_left",
}


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST, allow_unmeasured_dynamics=True)


@pytest.fixture(scope="module")
def cfg(manifest):
    stack = ConfigStack()
    stack.set_manifest_behaviour(manifest.model.behaviour, manifest.channels, body_yaw="body_yaw")
    return stack.effective()


def seg(db: float | None, s: float) -> np.ndarray:
    """``s`` seconds of a 220 Hz tone at RMS ``db`` dBFS, or silence for None."""
    n = int(round(RATE * s))
    if db is None:
        return np.zeros(n, np.float32)
    amp = 10 ** (db / 20) * math.sqrt(2)
    return (amp * np.sin(2 * math.pi * 220 * np.arange(n) / RATE)).astype(np.float32)


def speak(tl, sid, start, parts, *, utt=1, question=False, mood="calm", continued=False):
    """Push sentence ``sid`` whole at ``start`` and finish it; returns its end."""
    x = np.concatenate([seg(db, s) for db, s in parts])
    tl.push(sid, x, RATE, start, utterance=utt, question=question, mood=mood, continued=continued)
    end = start + x.size / RATE
    tl.finish(sid, end)
    return end


STEADY = [(-26, 1.5)]
LOUD_STRETCH = [(-26, 0.6), (-14, 0.4), (-26, 0.6)]
PHRASES = [(-26, 0.6), (None, 0.4), (-26, 0.6)]
EMPHASIS = [(-26, 0.6), (-14, 0.04), (-26, 0.6)]


def only(cfg, *roles) -> SpeechMotionMap:
    """The shipped map with only ``roles`` kept."""
    raw = cfg.speech_motion_map.model_dump()
    return SpeechMotionMap.model_validate(
        {k: v if k in roles else ({} if k in ("beat", "sway") else None) for k, v in raw.items()}
    )


def settings(cfg, preset="lively", level=None, **fields):
    levels = cfg.speech_motion.levels.model_dump()
    levels[preset].update(level or {})
    update = {"liveliness": preset, "levels": SpeechMotionLevels.model_validate(levels), **fields}
    return cfg.speech_motion.model_copy(update=update)


def register(**params):
    defaults = ConfigStack().effective()
    return resolve_register([Person(params)], defaults.register, defaults.persona)


def make(cfg, manifest, tl, smap=None, sm=None, mood="calm", arousal=0.0, **kw):
    r = SpeakingRenderer(
        sm or settings(cfg),
        smap or cfg.speech_motion_map,
        manifest.dofs,
        body_yaw="body_yaw",
        max_amplitude=cfg.expression.max_amplitude,
        timeline=lambda: tl,
        **kw,
    )
    r.set_register(register())
    r.set_affect(AffectState(0.0, arousal, mood, 1.0))
    return r


def run(r, clock, t0, t1, each=None):
    """Tick the renderer from ``t0`` to ``t1``; ``[(t, targets or None)]`` per tick."""
    out, t = [], t0
    while t < t1 - 1e-9:
        clock.t = t
        if each is not None:
            each(t)
        cmds = r.commands(t)
        assert len(cmds) <= 1
        out.append((t, dict(cmds[0].targets) if cmds else None))
        t = round(t + TICK, 6)
    return out


def peak(ticks, channel) -> float:
    return max((abs(o.get(channel, 0.0)) for _, o in ticks if o), default=0.0)


# ------------------------------------------------------------------ visibility (design ZT 4)


CASES = {
    "beat": (LOUD_STRETCH, False, "calm"),
    "phrase_nod": (PHRASES, False, "calm"),
    "question_tilt": ([(-26, 1.0)], True, "calm"),
    "accent": (EMPHASIS, False, "joy"),  # joy has the largest mood gain
}


def visibility_cases():
    for preset in PRESETS:
        for role in CASES:
            marks = ()
            if preset == "lively" and role == "accent":
                marks = pytest.mark.xfail(
                    strict=True,
                    reason="expected-clipped until M0: the dev fallback dynamics flatten the "
                    "lively accent (design ZT 4); measured dynamics should make this pass",
                )
            yield pytest.param(preset, role, marks=marks, id=f"{preset}-{role}")


@pytest.mark.parametrize(("preset", "role"), list(visibility_cases()))
def test_each_preset_stays_visible_under_the_fallback_dynamics(cfg, manifest, preset, role):
    """Worst case: the preset's degrees with the default register, neutral arousal, awake, and the
    accent at the largest mood gain, so a preset cannot go invisible silently. Visible: the
    limiter output reaches VISIBLE_SHARE of the commanded peak and never trails the command by
    more than LAG_SHARE of it. A higher arousal (up to 1 + arousal_gain) is clipped more."""
    parts, question, mood = CASES[role]
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    speak(tl, 1, 0.5, parts, question=question, mood=mood)
    sm = settings(cfg, preset, {"nod_share": 1.0})
    r = make(cfg, manifest, tl, only(cfg, role), sm, mood=mood)
    core = MotionCore(
        manifest.dofs,
        manifest.constraints,
        crossfade_s=cfg.motion.arbiter.crossfade_ms / 1000,
        watchdog_s=10.0,
        watchdog_return_factor=cfg.safety.watchdog_return_factor,
    )
    ch, home = MAIN[role], manifest.dofs[MAIN[role]].home
    commanded = reached = lag = last = 0.0
    t, n = 0.0, 0
    while t < 6.0:
        clock.t = t
        if n % round(TICK / LOOP_DT) == 0:
            for cmd in r.commands(t):
                core.feed(cmd)
                last = cmd.targets.get(ch, 0.0)
                commanded = max(commanded, abs(last))
        pos = core.tick(t, LOOP_DT)[ch] - home
        reached, lag = max(reached, abs(pos)), max(lag, abs(pos - last))
        t, n = round(t + LOOP_DT, 6), n + 1
    if getattr(sm.level(), f"{role}_deg") == 0:  # the subtle accent is off
        assert commanded == 0.0
        return
    assert commanded > 0
    assert reached >= VISIBLE_SHARE * commanded, (reached, commanded)
    assert lag <= LAG_SHARE * commanded, (lag, commanded)


# ------------------------------------------------------------------ safety and priority


def test_a_reflex_suppresses_the_layer(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    speak(tl, 1, 0.5, LOUD_STRETCH)
    r = make(cfg, manifest, tl, only(cfg, "beat"))
    arb = Arbiter(manifest.dofs, 0.0)
    reflex = manifest.dofs["head.pitch"].home + 0.1
    seen = []
    for t, out in run(r, clock, 0.0, 2.5):
        if out:
            arb.submit(Command(SOURCE, Priority.EXPRESSION, out, t, additive=True))
            seen.append(out["head.pitch"])
            arb.submit(Command("reflex", Priority.REFLEX, {"head.pitch": reflex}, t))
            assert arb.step(t)["head.pitch"] == pytest.approx(reflex)
            arb.clear("reflex")
            assert arb.step(t)["head.pitch"] == pytest.approx(
                manifest.dofs["head.pitch"].home + out["head.pitch"]
            )
    assert max(seen) > 0


def test_a_barge_in_fades_to_zero_within_fade_ms_plus_one_tick(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    speak(tl, 1, 0.5, [(-26, 1.0), (-14, 1.5), (-26, 1.0)])  # loud from 1.5 s to 3.0 s
    r = make(cfg, manifest, tl, only(cfg, "beat"))
    cut = 2.2
    ticks = run(r, clock, 0.0, 3.5, each=lambda t: tl.truncate(t) if t == cut else None)
    before = [o for t, o in ticks if t < cut and o]
    assert before and abs(before[-1]["head.pitch"]) > math.radians(0.5)
    fade = cfg.speech_motion.fade_ms / 1000
    after = [(t, o) for t, o in ticks if t >= cut]
    moving = [t for t, o in after if o and any(abs(v) > 0 for v in o.values())]
    assert moving and max(moving) <= cut + fade + TICK
    sizes = [abs(o["head.pitch"]) for t, o in after if o]
    assert sizes == sorted(sizes, reverse=True)  # ramps down, never back up
    assert all(o is None for t, o in after if t > cut + fade + TICK)
    assert r.snapshot()["cuts"] == 1


def test_a_barge_in_drops_a_rising_tilt_within_fade_ms_plus_one_tick(cfg, manifest):
    """The tilt leads the end of a question by hold_at_s; a barge-in before the end drops it.
    (After the end a person speaking is the listening case: the tilt plays its return.)"""
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    end = speak(tl, 1, 0.5, [(-26, 1.0)], question=True)
    r = make(cfg, manifest, tl, only(cfg, "question_tilt"))
    cut = round(end - 0.2, 6)
    ticks = run(r, clock, 0.0, cut + 1.0, each=lambda t: tl.truncate(t) if t == cut else None)
    assert [o for t, o in ticks if t == round(cut - TICK, 6)][0]["head.roll"] != 0
    fade = cfg.speech_motion.fade_ms / 1000
    assert all(o is None for t, o in ticks if t > cut + fade + TICK)


def test_circadian_rest_gives_no_commands(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    speak(tl, 1, 0.5, PHRASES, question=True)
    r = make(cfg, manifest, tl)
    r.circadian_factor = lambda: 0.0
    assert all(o is None for _, o in run(r, clock, 0.0, 6.0))


def test_a_disabled_layer_moves_nothing(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    speak(tl, 1, 0.5, PHRASES, question=True)
    r = make(cfg, manifest, tl, sm=settings(cfg, enabled=False))
    assert all(o is None for _, o in run(r, clock, 0.0, 6.0))


def test_no_timeline_moves_nothing(cfg, manifest):
    r = make(cfg, manifest, None)
    assert all(o is None for _, o in run(r, Clock(), 0.0, 1.0))


@pytest.mark.parametrize("preset", PRESETS)
def test_no_command_ever_names_body_yaw_or_a_translation(cfg, manifest, preset):
    """Property over random replies (seeded): every channel is an angle dof, never the body."""
    allowed = {n for n, d in manifest.dofs.items() if d.kind == "angle"} - {"body_yaw"}
    rng = random.Random(preset)
    for _ in range(5):
        clock = Clock()
        tl = SpeechTimeline(cfg.speech_motion, clock)
        start = 0.3
        for sid in range(1, 4):
            parts = [(rng.choice([None, -30, -26, -20, -12]), rng.uniform(0.05, 0.5))
                     for _ in range(rng.randint(2, 8))]  # fmt: skip
            start = speak(tl, sid, start, parts, question=rng.random() < 0.5, mood="joy") + 0.2
        r = make(cfg, manifest, tl, sm=settings(cfg, preset), arousal=1.0)
        for _, out in run(r, clock, 0.0, start + 4.0):
            assert out is None or set(out) <= allowed


def test_the_renderer_refuses_a_map_that_moves_the_body(cfg, manifest):
    bad = SpeechMotionMap.model_validate({"sway": {"body_yaw": 1}})
    with pytest.raises(ValueError, match="body_yaw"):
        make(cfg, manifest, None, bad)
    translate = SpeechMotionMap.model_validate({"beat": {"head.z": 1}})
    with pytest.raises(ValueError, match="head.z"):
        make(cfg, manifest, None, translate)


def test_commands_expire_after_the_ttl(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    speak(tl, 1, 0.5, LOUD_STRETCH)
    r = make(cfg, manifest, tl, only(cfg, "beat"))
    ticks = run(r, clock, 0.0, 1.3)
    clock.t = 1.3
    cmd = r.commands(1.3)[0]
    assert cmd.priority == Priority.EXPRESSION and cmd.additive
    assert cmd.expires == pytest.approx(1.3 + cfg.speech_motion.command_ttl_s)
    arb = Arbiter(manifest.dofs, 0.0)
    arb.submit(cmd)
    home = manifest.dofs["head.pitch"].home
    assert arb.step(1.3)["head.pitch"] != pytest.approx(home)
    assert arb.step(cmd.expires)["head.pitch"] == pytest.approx(home)  # the app stalled
    assert ticks


# ------------------------------------------------------------------ gestures


def test_the_question_tilt_holds_and_releases_when_listening_starts(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    end = speak(tl, 1, 0.5, [(-26, 1.0)], question=True)
    tilt = cfg.speech_motion_map.question_tilt
    back = max(f[-1][0] for f in tilt.channels.values()) - tilt.hold_at_s
    hearing = {"on": False}
    r = make(cfg, manifest, tl, only(cfg, "question_tilt"), listening=lambda: hearing["on"])
    hold = cfg.speech_motion.question_hold_s
    ticks = run(r, clock, 0.0, end + 1.0)
    held = [abs(o["head.roll"]) for t, o in ticks if o and t >= end]
    want = math.radians(cfg.speech_motion.levels.lively.question_tilt_deg)
    assert held and min(held) == pytest.approx(want, rel=0.02)  # held at its peak
    hearing["on"] = True
    rest = run(r, clock, end + 1.0, end + 1.0 + back + 2 * TICK + cfg.speech_motion.fade_ms / 1000)
    assert rest[-1][1] is None  # returned well before question_hold_s ran out
    assert end + 1.0 + back < end + hold


def test_the_question_tilt_returns_after_question_hold_s(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    end = speak(tl, 1, 0.5, [(-26, 1.0)], question=True)
    r = make(cfg, manifest, tl, only(cfg, "question_tilt"))
    hold = cfg.speech_motion.question_hold_s
    ticks = run(r, clock, 0.0, end + hold + 3.0)
    moving = [t for t, o in ticks if o]
    assert end + hold - TICK < max(moving) < end + hold + 2.0


def test_a_stricter_register_lowers_the_amplitude_at_once(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    end = speak(tl, 1, 0.5, [(-26, 1.0)], question=True)
    r = make(cfg, manifest, tl, only(cfg, "question_tilt"))
    ticks = run(r, clock, 0.0, end + 0.5)
    full = abs(ticks[-1][1]["head.roll"])
    r.set_register(register(motion_amplitude=0.5))
    clock.t = end + 0.5
    lowered = abs(r.commands(end + 0.5)[0].targets["head.roll"])
    assert lowered == pytest.approx(full * 0.5, rel=0.02)


def test_channels_are_ducked_while_an_expression_plays(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    end = speak(tl, 1, 0.5, [(-26, 1.0)], question=True)
    playing = {"channels": frozenset()}
    r = make(cfg, manifest, tl, only(cfg, "question_tilt"), ducked=lambda: playing["channels"])
    run(r, clock, 0.0, end + 0.3)
    playing["channels"] = frozenset({"head.roll"})
    ticks = run(r, clock, end + 0.3, end + 0.3 + cfg.speech_motion.fade_ms / 1000 + TICK)
    last = ticks[-1][1]
    assert last["head.roll"] == pytest.approx(0.0, abs=1e-9)
    assert abs(last["head.pitch"]) > 0  # the channel the expression leaves alone keeps moving
    playing["channels"] = frozenset()
    back = run(r, clock, ticks[-1][0] + TICK, ticks[-1][0] + 0.5)
    assert abs(back[-1][1]["head.roll"]) > 0


def test_phrase_nods_are_seeded_and_follow_nod_share(cfg, manifest):
    def nods(share):
        clock = Clock()
        tl = SpeechTimeline(cfg.speech_motion, clock)
        start = 0.5
        for sid in range(1, 4):
            start = speak(tl, sid, start, PHRASES * 2, utt=7) + 0.1
        sm = settings(cfg, level={"nod_share": share})
        r = make(cfg, manifest, tl, only(cfg, "phrase_nod"), sm)
        ticks = run(r, clock, 0.0, start + 2.0)
        return r.snapshot()["phrase_nod"], [round(o["head.pitch"], 9) if o else 0 for _, o in ticks]

    assert nods(0.0)[0] == 0
    assert nods(1.0)[0] >= 3
    assert nods(0.5) == nods(0.5)  # the same reply moves the same way


def test_a_late_phrase_gap_still_nods_while_its_pause_lasts(cfg, manifest):
    """A streamed voice writes the phrase gap when the voice resumes: in the layer's past."""

    def case(write_at):
        clock = Clock()
        tl = SpeechTimeline(cfg.speech_motion, clock)
        first = seg(-26, 0.6)
        tl.push(1, first, RATE, 0.5, utterance=1, question=False, mood="calm", continued=False)
        rest = np.concatenate([seg(None, 0.4), seg(-26, 0.6)])
        sm = settings(cfg, level={"nod_share": 1.0})
        r = make(cfg, manifest, tl, only(cfg, "phrase_nod"), sm)

        def each(t):
            if t == write_at:
                tl.push(1, rest, RATE, 1.1, utterance=1, question=False, mood="calm",
                        continued=False)  # fmt: skip
                tl.finish(1, 2.1)

        run(r, clock, 0.0, 1.1, each)
        run(r, clock, 1.1, 2.0, each)
        return r.snapshot()

    in_pause = case(1.3)  # the gap runs 1.1 to 1.5
    assert in_pause["phrase_nod"] == 1 and in_pause["late"] == 0
    resumed = case(1.6)
    assert resumed["phrase_nod"] == 0 and resumed["late"] == 1


def test_the_accent_follows_the_mood_gain(cfg, manifest):
    def accent(mood):
        clock = Clock()
        tl = SpeechTimeline(cfg.speech_motion, clock)
        speak(tl, 1, 0.5, EMPHASIS, mood=mood)
        r = make(cfg, manifest, tl, only(cfg, "accent"), settings(cfg, "natural"), mood=mood)
        return peak(run(r, clock, 0.0, 4.0), "antenna_left")

    gains = cfg.speech_motion.mood_gain
    assert accent("joy") / accent("sadness") == pytest.approx(gains["joy"] / gains["sadness"])
    assert accent("joy") == pytest.approx(math.radians(6.0) * gains["joy"])


def test_accents_are_thinned_by_the_preset_interval(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    peaks = [(-26, 0.6), (-14, 0.04), (-26, 1.46), (-14, 0.04), (-26, 0.9)]  # 1.5 s apart
    speak(tl, 1, 0.5, peaks, mood="joy")
    emphases = len(tl.events_between(0.0, 10.0)) - 2  # minus start and end
    r = make(cfg, manifest, tl, only(cfg, "accent"), settings(cfg, "subtle", {"accent_deg": 6.0}))
    run(r, clock, 0.0, 5.0)
    snap = r.snapshot()
    assert emphases == 2 and snap["accent"] == 1 and snap["skipped"] == 0  # subtle: one per 2 s


def test_no_breath_before_a_continued_piece(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    end = speak(tl, 1, 0.5, STEADY)
    speak(tl, 2, end + 0.8, STEADY, continued=True)
    r = make(cfg, manifest, tl, only(cfg, "breath"))
    run(r, clock, 0.0, end + 4.0)
    assert r.snapshot()["breath"] == 1


def test_a_hot_reload_of_liveliness_applies_at_once(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    speak(tl, 1, 0.5, [(-26, 1.5), (-14, 1.0), (-26, 1.5)])  # loud from 2.0 s to 3.0 s
    smap = only(cfg, "beat")
    r = make(cfg, manifest, tl, smap)
    ticks = run(r, clock, 0.0, 2.8)
    lively = abs(ticks[-1][1]["head.pitch"])
    assert lively > 0.8 * math.radians(cfg.speech_motion.levels.lively.beat_deg)
    r.apply_config(settings(cfg, "subtle"), smap, cfg.expression.max_amplitude)
    clock.t = 2.8
    subtle = abs(r.commands(2.8)[0].targets["head.pitch"])
    levels = cfg.speech_motion.levels
    want = levels.subtle.beat_deg / levels.lively.beat_deg
    assert subtle / lively == pytest.approx(want, rel=0.05)


def test_the_snapshot_holds_integers_only(cfg, manifest):
    clock = Clock()
    tl = SpeechTimeline(cfg.speech_motion, clock)
    speak(tl, 1, 0.5, PHRASES, question=True)
    r = make(cfg, manifest, tl)
    run(r, clock, 0.0, 4.0)
    snap = r.snapshot()
    assert snap and all(type(v) is int for v in snap.values())
