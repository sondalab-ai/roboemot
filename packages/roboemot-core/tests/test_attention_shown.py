"""SPEC 5A Joint attention: a thing a person shows holds the gaze while its interest exceeds the
face's social pull; the return to the face emerges from the model, never from a timer."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import pytest
from pydantic import ValidationError
from roboemot.affect.appraisal import AppraisalAdapter
from roboemot.attention import AttentionSelector, AttentionTarget, SpeakerHint, interest_mod
from roboemot.attention.interest import InterestModel, in_region
from roboemot.bus import Bus
from roboemot.config.schema import (
    ATTENTION_CATEGORIES,
    AppraisalConfig,
    AttentionConfig,
    CircadianConfig,
    PersonaConfig,
    RoboEmotConfig,
    ShownObjectConfig,
    StateEffects,
    VisionRates,
)
from roboemot.persona import effects
from roboemot.runtime.ticks import interest_modulation, scan_context
from roboemot.vision import CIRCADIAN, RateGovernor
from roboemot.world import Entry, WorldModel
from roboemot.world.scan import ScanContext, ScanPolicy
from roboemot.world.types import Observation

DEG = math.radians(1)
CFG = AttentionConfig()
TICK = 0.05  # 20 Hz selector
OBS = 1.0  # 1 Hz object detection
APPEAR = 0.5  # the object is first detected here
FACE_EL, FACE_SIZE = 10.0, 10.0  # face centre and box height, degrees
THING_EL = -5.0  # 1.5 face heights below the face centre: in front of the chest
STATES = CircadianConfig().states


def face(seen: float, *, eid: str = "p1", az: float = 0.0) -> Entry:
    return Entry(
        entry_id=eid,
        kind="person",
        category="face",
        identity="unknown",
        azimuth_rad=az * DEG,
        elevation_rad=FACE_EL * DEG,
        distance_m=1.0,
        uncertainty_rad=0.1,
        confidence=0.9,
        first_seen=0.0,
        last_seen=seen,
        extent_rad=FACE_SIZE * DEG,
    )


def thing(seen: float, az: float, *, el: float = THING_EL, identity: str = "unknown") -> Entry:
    return Entry(
        entry_id="o1",
        kind="object",
        category="cup",
        identity=identity,
        azimuth_rad=az * DEG,
        elevation_rad=el * DEG,
        distance_m=None,
        uncertainty_rad=0.1,
        confidence=0.8,
        first_seen=APPEAR,
        last_seen=seen,
        extent_rad=5 * DEG,
    )


def wiggle(until: float) -> Callable[[float], float]:
    """A held thing waved side to side (±10 deg) until ``until``, then held still."""
    return lambda stamp: 10.0 * math.sin(1.5 * (min(stamp, until) - APPEAR))


def simulate(
    sel: AttentionSelector,
    end: float,
    path: Callable[[float], float | None],
    *,
    start: float = 0.0,
    mod=None,
    face_seen: Callable[[float], float] = lambda t: t,
    identity: str = "unknown",
    identity_from: float | None = None,
    el: float = THING_EL,
    on_tick: Callable[[float, AttentionSelector], None] | None = None,
) -> list[tuple[float, AttentionTarget | None]]:
    """Tick the selector at 20 Hz; the face is seen every tick (``face_seen``), the thing once
    per ``OBS`` from ``APPEAR`` at ``path(stamp)`` (None: not in view). ``identity_from``: the
    thing is recognised as a registry object (``cup-1``) from that stamp on."""
    trace = []
    t = start
    while t < end - 1e-9:
        entries = [face(face_seen(t))]
        if t >= APPEAR:
            stamp = APPEAR + math.floor((t - APPEAR) / OBS + 1e-9) * OBS
            az = path(stamp)
            if az is not None:
                who = identity
                if identity_from is not None and stamp >= identity_from:
                    who = "cup-1"
                entries.append(thing(stamp, az, el=el, identity=who))
        if on_tick is not None:
            on_tick(t, sel)
        trace.append((t, sel.update(t, entries, mod=mod)))
        t = round(t + TICK, 6)
    return trace


def first(trace, pred, after: float = -1.0) -> float | None:
    return next((t for t, tg in trace if t > after and pred(tg)), None)


def on_thing(tg) -> bool:
    return tg is not None and tg.entry_id == "o1" and tg.priority == "shown_object"


def on_face(tg) -> bool:
    return tg is not None and tg.entry_id == "p1"


def return_time(cfg: AttentionConfig = CFG, mod=None, **kw) -> float:
    """When the gaze comes back to the face after the thing was waved for 2 s, then held still."""
    sel = AttentionSelector(cfg)
    trace = simulate(sel, 60.0, wiggle(2.5), mod=mod, **kw)
    start = first(trace, on_thing)
    assert start is not None
    back = first(trace, on_face, after=start)
    assert back is not None
    assert sel.snapshot()["shown_object"]["ended"]["face_pull"] == 1
    return back


def mod_for(*, arousal=0.0, label="calm", state="awake", **traits):
    persona = PersonaConfig(traits=traits) if traits else PersonaConfig()
    return interest_mod(
        CFG.shown, arousal=arousal, label=label, circadian=STATES[state], persona=effects(persona)
    )


# ---- the episode


def test_a_thing_shown_below_the_face_takes_the_gaze_after_the_hysteresis():
    sel = AttentionSelector(CFG)
    trace = simulate(sel, 3.0, wiggle(100.0))
    assert all(on_face(tg) for t, tg in trace if t < CFG.switch_hysteresis_s - 1e-9)
    switched = first(trace, on_thing)
    assert switched == pytest.approx(CFG.switch_hysteresis_s)
    assert sel.engaged and sel.episode_active  # a person is showing it: scanning yields
    assert sel.snapshot()["shown_object"]["episodes"] == 1


def test_the_gaze_stays_on_a_moving_thing_and_follows_its_bearing():
    sel = AttentionSelector(CFG)
    trace = simulate(sel, 8.0, wiggle(100.0))
    during = [(t, tg) for t, tg in trace if 2.0 <= t < 8.0]
    assert all(on_thing(tg) for _, tg in during)
    for t, tg in during:
        stamp = APPEAR + math.floor((t - APPEAR) / OBS + 1e-9) * OBS
        assert tg.azimuth_rad == pytest.approx(wiggle(100.0)(stamp) * DEG)
    assert len({round(tg.azimuth_rad, 4) for _, tg in during}) >= 5  # it moved, the gaze too


def test_a_still_thing_hands_the_gaze_back_to_the_face_without_a_timer():
    sel = AttentionSelector(CFG)
    trace = simulate(sel, 30.0, wiggle(2.5))
    start = first(trace, on_thing)
    back = first(trace, on_face, after=start)
    assert start is not None and back is not None
    # no look back at a still thing (until the perfectly still face habituates, SPEC 5)
    assert all(on_face(tg) for t, tg in trace if back <= t < back + CFG.habituation_s)
    snap = sel.snapshot()["shown_object"]
    assert snap["ended"]["face_pull"] == 1 and not snap["active"]
    assert snap["duration_s"]["p50"] == pytest.approx(back - start)


def test_the_return_time_follows_the_decay_rate():
    base = return_time()
    faster = AttentionConfig(shown=ShownObjectConfig(decay_per_s=2 * CFG.shown.decay_per_s))
    slower = AttentionConfig(shown=ShownObjectConfig(decay_per_s=CFG.shown.decay_per_s / 2))
    assert return_time(faster) < base < return_time(slower)


def test_the_return_time_follows_the_arousal_and_the_curiosity_label():
    base = return_time(mod=mod_for())
    assert return_time(mod=mod_for(arousal=0.8)) > base > return_time(mod=mod_for(arousal=-0.8))
    assert return_time(mod=mod_for(label="curiosity")) > base


def test_a_sleepy_robot_loses_interest_sooner():
    awake = return_time(mod=mod_for(state="awake"))
    assert return_time(mod=mod_for(state="drowsy")) < awake
    assert return_time(mod=mod_for(state="resting")) < return_time(mod=mod_for(state="drowsy"))


def test_the_return_time_follows_the_persona_traits():
    neutral = return_time(mod=mod_for(energy=0.5, playfulness=0.5))
    assert return_time(mod=mod_for(energy=0.0)) < neutral < return_time(mod=mod_for(energy=1.0))
    assert return_time(mod=mod_for(playfulness=1.0)) > neutral


def test_interest_modulation_reads_the_live_affect_circadian_and_persona():
    cfg = RoboEmotConfig()

    class Affect:
        arousal, label = 0.0, "calm"

    fx = effects(cfg.persona)
    assert interest_modulation(cfg, Affect(), None, fx) == mod_for()
    Affect.label = "curiosity"
    m = interest_modulation(cfg, Affect(), None, fx)
    assert m.rise_x == pytest.approx(CFG.shown.curiosity_rise_x)
    assert m.decay_x == pytest.approx(CFG.shown.curiosity_decay_x)


def test_a_speaker_takes_the_gaze_at_once_during_an_episode():
    sel = AttentionSelector(CFG)
    simulate(sel, 3.0, wiggle(100.0))
    assert on_thing(sel.target)
    sel.hint_speaker(SpeakerHint(3.0, entry_id="p1"))
    trace = simulate(sel, 3.1, wiggle(100.0), start=3.0)
    assert (trace[0][1].entry_id, trace[0][1].priority) == ("p1", "active_speaker")
    assert sel.snapshot()["shown_object"]["ended"]["speaker"] == 1


def test_a_familiar_thing_holds_the_gaze_for_less_time_than_a_new_one():
    new = return_time()
    familiar = return_time(identity="cup-1")  # a registry object: a known individual
    assert familiar < new


def test_the_category_seen_again_is_less_new():
    model = InterestModel(CFG.shown)
    e = thing(APPEAR, 0.0)
    assert model._novelty(e, 0.0) == 1.0
    model._expose("cup", 0.0)
    assert model._novelty(e, 0.0) == pytest.approx(0.5)
    later = CFG.shown.exposure_half_life_s
    assert model._novelty(e, later) == pytest.approx(1 / 1.5)


def test_a_never_still_thing_still_ends_on_the_face():
    sel = AttentionSelector(CFG)
    trace = simulate(sel, 300.0, wiggle(math.inf))
    last_look = max(t for t, tg in trace if on_thing(tg))
    assert last_look < 250.0
    assert all(on_face(tg) for t, tg in trace if t > last_look + CFG.switch_hysteresis_s)
    snap = sel.snapshot()["shown_object"]
    assert snap["episodes"] >= 2 and snap["ended"]["face_pull"] == snap["episodes"]
    durations = list(sel._durations)
    assert durations[0] == max(durations) and durations[-1] < durations[0] / 2  # it dampens
    assert durations[1] < durations[0]


def test_movement_has_diminishing_returns():
    model = InterestModel(CFG.shown)
    people = [face(0.0)]
    gains = []
    for i in range(6):
        e = thing(APPEAR + i, 0.0)
        before = model.shown["o1"].interest if "o1" in model.shown else 0.0
        model.step(
            APPEAR + i, 0.0, [e], people, {"o1": 1.0}, watched=None, looking=None, mod=mod_for()
        )
        gains.append(model.shown["o1"].interest - before)
    move_gains = gains[1:]
    assert all(b < a for a, b in zip(move_gains, move_gains[1:], strict=False))


def test_a_thing_far_from_any_face_is_not_shown_and_keeps_salient_motion():
    sel = AttentionSelector(CFG)
    trace = simulate(sel, 10.0, lambda s: 60.0 + 10.0 * math.sin(s))
    assert all(on_face(tg) for _, tg in trace)
    assert sel.snapshot()["shown_object"]["episodes"] == 0
    sel = AttentionSelector(CFG)
    t, prev = 0.0, None
    while t < 4.0:
        stamp = math.floor(t)
        tg = sel.update(t, [thing(stamp, 60.0 + 30.0 * stamp)])
        prev = tg or prev
        t = round(t + TICK, 6)
    assert prev is not None and prev.priority == "salient_motion"


def test_the_region_is_in_face_heights_beside_and_below_the_face():
    f = face(0.0)
    size = FACE_SIZE
    sh = CFG.shown
    assert in_region(sh, thing(0.0, 0.0), f)
    assert in_region(sh, thing(0.0, sh.region_width_faces * size - 0.1), f)
    assert not in_region(sh, thing(0.0, sh.region_width_faces * size + 0.1), f)
    assert not in_region(sh, thing(0.0, 0.0, el=FACE_EL - sh.region_below_faces * size - 0.1), f)
    assert not in_region(sh, thing(0.0, 0.0, el=FACE_EL + sh.region_above_faces * size + 0.1), f)
    no_box = Entry("p", "person", "face", "unknown", 0, FACE_EL * DEG, None, 0, 1, 0, 0)
    assert not in_region(sh, thing(0.0, 0.0), no_box)


def test_the_episode_ends_when_the_thing_leaves_the_region_or_the_view():
    sel = AttentionSelector(CFG)
    simulate(sel, 3.0, wiggle(100.0))
    simulate(sel, 6.0, lambda s: 60.0, start=3.0)  # put down far to the side
    assert sel.snapshot()["shown_object"]["ended"]["left_region"] == 1
    sel = AttentionSelector(CFG)
    simulate(sel, 3.0, wiggle(100.0))
    simulate(sel, 12.0, lambda s: None, start=3.0)  # out of view: stale after stale_s
    assert sel.snapshot()["shown_object"]["ended"]["lost"] == 1


def test_sleep_ends_the_episode():
    sel = AttentionSelector(CFG)
    simulate(sel, 3.0, wiggle(100.0))
    sel.sleep(3.0)
    snap = sel.snapshot()["shown_object"]
    assert snap["ended"]["asleep"] == 1 and not snap["active"] and not sel.episode_active
    assert sel._interest.shown == {}


def test_the_holder_does_not_habituate_during_a_long_episode():
    cfg = AttentionConfig(shown=ShownObjectConfig(decay_per_s=0.01, pull_rate_per_s=0.02))
    sel = AttentionSelector(cfg)
    trace = simulate(sel, 60.0, wiggle(2.5))
    start = first(trace, on_thing)
    back = first(trace, on_face, after=start)
    assert back is not None and back - start > CFG.habituation_s
    returned = dict(trace)[back]
    assert returned.priority == "nearest_face"  # the holder did not habituate meanwhile


def test_a_poster_with_a_still_cup_below_it_still_habituates():
    """SPEC 5 habituation (2026-10-28): a face and a cup both seen from the start and never
    moving. The cup's appearance gives one short look; afterwards nothing is shown, so the
    face is a poster again and stops drawing the gaze."""
    sel = AttentionSelector(CFG)
    trace = simulate(sel, CFG.habituation_s + 30.0, lambda s: 0.0)
    assert sel.snapshot()["shown_object"]["episodes"] <= 1
    assert trace[-1][1] is None or not on_face(trace[-1][1])


def test_a_thing_recognised_after_it_was_shown_becomes_familiar():
    """Entity identity comes from votes over a few object frames, after the thing appears."""
    late = return_time(identity_from=APPEAR + 2.0)
    assert late < return_time()


def test_a_face_out_of_the_camera_while_the_robot_looks_down_is_still_there_to_return_to():
    """Looking at a thing below the face may leave the face out of the picture: the holder
    counts as seen while the robot looks at what it holds."""
    frozen = {"at": None}

    def face_seen(t: float) -> float:
        if frozen["at"] is None and on_thing(sel.target):
            frozen["at"] = t
        if frozen["at"] is not None and not on_face(sel.target):
            return frozen["at"]  # not refreshed while the head looks down
        return t

    sel = AttentionSelector(CFG)
    trace = simulate(sel, 30.0, wiggle(2.5), face_seen=face_seen)
    start = first(trace, on_thing)
    back = first(trace, on_face, after=start)
    assert back is not None and back - start > CFG.stale_s
    assert sel.snapshot()["shown_object"]["ended"]["face_pull"] == 1


def test_a_long_episode_holds_off_the_scan_with_no_time_limit():
    """No maximum duration: with no decay and no pull growth a shown thing holds the gaze, and
    a due scan (a wake-up sweep) never cuts in, as it would after object_max_hold_s for a thing
    nobody shows."""
    cfg = RoboEmotConfig(
        attention={"shown": {"decay_per_s": 0.0, "pull_rate_per_s": 0.0}},
    )
    sel = AttentionSelector(cfg.attention)
    policy = ScanPolicy(cfg.world.scan)
    policy.update(ScanContext(now=0.0, engaged=False, circadian="resting"))
    decisions = []

    def scan(t, s):
        if s.target is not None:
            ctx = scan_context(t, cfg, s, hour=12.0)
            decisions.append(policy.update(ctx))

    trace = simulate(sel, 2 * cfg.attention.object_max_hold_s, wiggle(2.5), on_tick=scan)
    start = first(trace, on_thing)
    assert all(on_thing(tg) for t, tg in trace if t >= start)
    assert [d for d in decisions[int(start / TICK) + 1 :] if d is not None] == []


def test_replay_gives_the_same_targets():
    a = simulate(AttentionSelector(CFG), 40.0, wiggle(10.0), mod=mod_for(arousal=0.3))
    b = simulate(AttentionSelector(CFG), 40.0, wiggle(10.0), mod=mod_for(arousal=0.3))
    assert a == b
    assert any(on_thing(tg) for _, tg in a) and any(on_face(tg) for _, tg in a)


def test_the_debug_section_holds_numbers_only():
    sel = AttentionSelector(CFG)
    simulate(sel, 30.0, wiggle(2.5))
    snap = sel.snapshot()
    text = json.dumps(snap)
    assert "cup" not in text and "o1" not in text and "p1" not in text
    shown = snap["shown_object"]
    assert set(shown["ended"]) == {
        "face_pull",
        "speaker",
        "left_region",
        "lost",
        "asleep",
        "other_object",
    }
    assert set(shown["duration_s"]) == {"count", "p50", "p90", "max"}


# ---- motion speed from the entry's own observations


def test_one_hz_observations_read_as_moving_on_every_tick():
    sel = AttentionSelector(CFG)
    speeds, targets = [], []
    t = 0.0
    while t < 6.0:
        stamp = math.floor(t)
        tg = sel.update(t, [thing(stamp, 30.0 * stamp)])
        if t >= 1.0:
            speeds.append(sel._motion["o1"].speed_dps)
            targets.append(tg)
        t = round(t + TICK, 6)
    # never 0, never 60 (the elevation of -5 deg shortens the arc a little)
    assert speeds and all(s == pytest.approx(30.0, rel=0.01) for s in speeds)
    assert all(tg is not None and tg.priority == "salient_motion" for tg in targets)


# ---- the governor's episode rate


def test_the_governor_runs_objects_at_the_active_rate_under_the_ceiling():
    rates = VisionRates(objects_hz=1.0, objects_active_hz=20.0, objects_max_hz=5.0)
    gov = RateGovernor(rates)
    assert gov.period("objects") == pytest.approx(1.0)
    gov.set_active("objects", True)
    assert gov.active("objects")
    assert gov.period("objects") == pytest.approx(1 / 5.0)  # the ceiling
    runs = sum(gov.allow("objects", i / 30.0) for i in range(60))  # 2 s of 30 Hz frames
    assert runs <= 2 * 5.0 + 1
    gov.set_scale(CIRCADIAN, 0.1)
    assert gov.period("objects") == pytest.approx(1 / 2.0)  # still scaled
    gov.set_active("objects", False)
    assert gov.period("objects") == pytest.approx(1 / 0.1)
    gov.set_active("detection", True)  # no detection_active_hz: no effect
    assert not gov.active("detection")


def test_the_episode_rate_lasts_as_long_as_the_episode():
    gov = RateGovernor(VisionRates())
    seen = []

    def loop(t, s):  # as the runtime loop does after the attention tick
        gov.set_active("objects", s.episode_active)
        seen.append((s.episode_active, gov.period("objects")))

    sel = AttentionSelector(CFG)
    simulate(sel, 30.0, wiggle(2.5), on_tick=loop)
    active = [p for a, p in seen if a]
    idle = [p for a, p in seen if not a]
    assert active and all(p == pytest.approx(1 / VisionRates().objects_active_hz) for p in active)
    assert all(p == pytest.approx(1 / VisionRates().objects_hz) for p in idle)
    assert not seen[-1][0]


# ---- appraisal


def test_a_look_at_a_shown_thing_emits_object_shown_with_the_category_only():
    bus, events = Bus(), []
    AppraisalAdapter(bus, AppraisalConfig(), lambda n, t, p: events.append((n, t, p)))
    look = AttentionTarget("o1", "object", 0.0, 0.0, "shown_object", "switched", 1.0, "cup")
    bus.publish("attention.target", look, 1.0)
    bus.publish("attention.target", look, 1.2)  # the same look, published again
    shown = [e for e in events if e[0] == "object.shown"]
    assert shown == [("object.shown", 1.0, {"category": "cup"})]
    person = AttentionTarget("p1", "person", 0.0, 0.0, "nearest_face", "switched", 2.0, "face")
    bus.publish("attention.target", person, 2.0)
    bus.publish("attention.target", look, 3.0)  # a new look at it
    assert len([e for e in events if e[0] == "object.shown"]) == 2


def test_object_shown_has_a_default_curiosity_rule():
    rules = [r for r in AppraisalConfig().rules if r.event == "object.shown"]
    assert len(rules) == 1 and rules[0].label == "curiosity" and rules[0].arousal > 0


# ---- configuration


def test_shown_object_sits_below_the_speaker_and_above_the_faces():
    order = CFG.priority_order
    assert order[0] == "active_speaker" and order[1] == "shown_object"
    assert order.index("shown_object") < order.index("enrolled_person")
    assert order.index("shown_object") < order.index("nearest_face")
    assert "shown_object" in ATTENTION_CATEGORIES
    assert AttentionConfig(priority_order=["shown_object"]).priority_order == ["shown_object"]
    with pytest.raises(ValidationError):
        AttentionConfig(priority_order=["shown_object", "shown_object"])
    with pytest.raises(ValidationError):
        AttentionConfig(priority_order=["held_object"])


def test_without_shown_object_in_the_order_a_shown_thing_is_only_motion():
    cfg = AttentionConfig(priority_order=["nearest_face", "salient_motion"])
    sel = AttentionSelector(cfg)
    trace = simulate(sel, 10.0, wiggle(100.0))
    assert all(on_face(tg) for _, tg in trace)
    assert sel.snapshot()["shown_object"]["episodes"] == 0


@pytest.mark.parametrize(
    ("model", "field", "bad"),
    [
        (ShownObjectConfig, "region_width_faces", -0.1),
        (ShownObjectConfig, "region_above_faces", 11),
        (ShownObjectConfig, "region_below_faces", -1),
        (ShownObjectConfig, "recent_s", 0),
        (ShownObjectConfig, "show_gain", -1),
        (ShownObjectConfig, "move_gain_per_s", 11),
        (ShownObjectConfig, "move_habituation_s", 0),
        (ShownObjectConfig, "familiar_novelty", 1.5),
        (ShownObjectConfig, "exposure_half_life_s", 0),
        (ShownObjectConfig, "decay_per_s", -0.1),
        (ShownObjectConfig, "pull_base", -0.1),
        (ShownObjectConfig, "start_margin", 6),
        (ShownObjectConfig, "pull_rate_per_s", -0.1),
        (ShownObjectConfig, "pull_recover_per_s", 11),
        (ShownObjectConfig, "arousal_rise_gain", 1.5),
        (ShownObjectConfig, "arousal_decay_gain", -0.1),
        (ShownObjectConfig, "curiosity_rise_x", 6),
        (ShownObjectConfig, "curiosity_decay_x", -1),
        (ShownObjectConfig, "report_window", 0),
        (VisionRates, "objects_active_hz", 0),
        (VisionRates, "objects_max_hz", 31),
        (StateEffects, "interest_decay_scale", 0),
    ],
)
def test_new_fields_are_range_checked(model, field, bad):
    with pytest.raises(ValidationError):
        model(**{field: bad})
    assert model.model_fields[field].description


def test_no_shown_object_field_is_a_maximum_duration():
    """Owner decision 2026-11-03: the episode length emerges from the model."""
    for name, f in ShownObjectConfig.model_fields.items():
        assert "max" not in name
        assert "maximum" not in (f.description or "").lower()


def test_persona_curves_are_neutral_at_the_middle():
    fx = effects(PersonaConfig(traits={"energy": 0.5, "playfulness": 0.5}))
    assert fx.interest_gain_x == pytest.approx(1.0) and fx.interest_decay_x == pytest.approx(1.0)


def test_circadian_states_speed_up_the_interest_decay():
    assert STATES["awake"].interest_decay_scale == 1.0
    assert 1.0 < STATES["drowsy"].interest_decay_scale < STATES["resting"].interest_decay_scale


# ---- world model: the box size behind the region


def test_the_world_entry_keeps_a_blended_box_size():
    world = WorldModel(RoboEmotConfig().world)
    obs = Observation("person", 0.0, "vision", 0.0, 0.0, category="person", extent_rad=0.1)
    world.observe(obs)
    obs2 = Observation("person", 0.1, "vision", 0.0, 0.0, category="person", extent_rad=0.2)
    world.observe(obs2)
    (e,) = world.entries
    w = RoboEmotConfig().world.position_blend
    assert e.extent_rad == pytest.approx(0.1 + w * 0.1)
