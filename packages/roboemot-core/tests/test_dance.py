"""SPEC 7 (Dance motion), 11A: the dance renderer on a canned beat, the map validation and the
config rules. The robot is the generic ``tests/data/dance-head.yaml``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from roboemot.affect import AffectState
from roboemot.affect.dance import SOURCE, DanceRenderer, fold
from roboemot.affect.register import Person, resolve_register
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import DanceConfig, DanceMotionMap, RoboEmotConfig
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.motion import MotionLoop, Priority
from roboemot.motion.beat import Beat, BeatSource
from roboemot.motion.curves import min_jerk

MANIFEST = Path(__file__).resolve().parent / "data" / "dance-head.yaml"
TICK = 0.05  # the 20 Hz app tick
BOB, SWAY = "neck_tilt", "neck_pan"
PERIOD = 1.2  # one bob per beat: 2 x min_half_s (0.5) fits
BARS = 4
NO_LEAD = {"lead_ms": 0.0, "fade_in_s": 0.1, "fade_out_s": 0.1}
YIELD_S = 0.15  # the speaking motion's default fade_ms


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST, allow_unmeasured_dynamics=True)


@pytest.fixture(scope="module")
def mapping(manifest):
    return DanceMotionMap.model_validate(manifest.model.behaviour["dance_motion_map"])


class Beats:
    """A settable beat and accents, as the source's providers."""

    def __init__(self, beat: Beat | None = None) -> None:
        self.beat, self.stamps = beat, []

    def source(self) -> BeatSource:
        return BeatSource(
            lambda now: self.beat,
            lambda now: None,
            accents=lambda since, now: [s for s in self.stamps if since < s <= now],
        )


def beat(period=PERIOD, phase=0.0, strength=1.0) -> Beat:
    return Beat(period, phase, strength, "own")


def register(**params):
    defaults = ConfigStack().effective()
    return resolve_register([Person(params)], defaults.register, defaults.persona)


def make(manifest, mapping, beats, *, cfg=None, max_amplitude=1.5, beats_per_bar=BARS, **kw):
    r = DanceRenderer(
        cfg or DanceConfig(**NO_LEAD),
        mapping,
        manifest.dofs,
        body_yaw=None,
        max_amplitude=max_amplitude,
        beats_per_bar=beats_per_bar,
        beat=beats.source(),
        yield_fade_s=lambda: YIELD_S,
        **kw,
    )
    r.set_register(register())
    r.set_affect(AffectState(0.0, 0.0, "calm", 1.0))
    return r


def run(r, t0, t1):
    """Tick from ``t0`` to ``t1``: ``[(t, targets or None)]`` per tick."""
    out, n = [], round((t1 - t0) / TICK)
    for i in range(n):
        t = t0 + i * TICK
        cmds = r.commands(t)
        out.append((t, cmds[0].targets if cmds else None))
    return out


def channel(track, name):
    return [(t, m.get(name, 0.0)) for t, m in track if m is not None]


def test_the_command_is_additive_expression_of_the_dance_source(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()))
    run(r, 0.0, 0.5)
    cmd = r.commands(0.5)[0]
    assert cmd.source == SOURCE and cmd.priority == Priority.EXPRESSION and cmd.additive
    assert cmd.expires == 0.5 + r.cfg.command_ttl_s


def test_bob_dips_once_per_beat(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()))
    run(r, 0.0, 2.0)  # past the fade
    track = run(r, 2.0, 2.0 + 4 * PERIOD)
    values = [v for _, v in channel(track, BOB)]
    dips = sum(
        values[i] > values[i - 1] and values[i] >= values[i + 1] for i in range(1, len(values) - 1)
    )
    assert dips == 4
    assert min(values) >= 0  # a dip is one-sided: the sign that dips the head


def test_bob_half_lasts_at_least_min_half_s_on_a_fast_hum(manifest, mapping):
    """SPEC 4A: a 128 BPM hum (the shipped joy hum tops out there) moves on every second beat."""
    period = 60.0 / 128.0
    r = make(manifest, mapping, Beats(beat(period)))
    run(r, 0.0, 1.0)
    track = run(r, 1.0, 1.0 + 8 * period)
    values = [v for _, v in channel(track, BOB)]
    peaks = [
        i
        for i in range(1, len(values) - 1)
        if values[i] > values[i - 1] and values[i] >= values[i + 1]
    ]
    gaps = [(b - a) * TICK for a, b in zip(peaks, peaks[1:], strict=False)]
    assert gaps and all(g / 2 >= r.cfg.min_half_s - TICK for g in gaps)


def test_a_tempo_too_fast_even_folded_rests_the_role(manifest, mapping):
    assert fold(0.1, 0.5) is None
    r = make(manifest, mapping, Beats(beat(period=0.1)), beats_per_bar=1)
    assert all(v == 0.0 for _, v in channel(run(r, 0.0, 3.0), BOB))


def test_sway_one_side_per_bar(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()))
    run(r, 0.0, 1.0)
    bar = PERIOD * BARS
    track = run(r, bar, bar + 2 * bar)  # bars 2 and 3
    sides = channel(track, SWAY)
    first = [v for t, v in sides if t < 2 * bar - 2 * TICK and t > bar + 2 * TICK]
    second = [v for t, v in sides if t > 2 * bar + 2 * TICK and t < 3 * bar - 2 * TICK]
    assert all(v < 0 for v in first) and all(v > 0 for v in second)


def test_accent_rate_limited(manifest, mapping):
    beats = Beats()
    beats.stamps = [1.0, 1.1, 1.5, 3.0]  # 0.5 s apart is inside the 1 s minimum
    r = make(manifest, mapping, beats)
    moved = channel(run(r, 0.0, 4.0), BOB)
    assert r.counts["accent"] == 2  # 1.0 and 3.0
    assert any(abs(v) > 0 for t, v in moved if 1.0 <= t < 2.0)
    assert any(abs(v) > 0 for t, v in moved if 2.9 <= t < 3.5)


def test_accent_peaks_on_its_stamp(manifest, mapping):
    beats = Beats()
    beats.stamps = [1.0]
    r = make(manifest, mapping, beats)
    track = channel(run(r, 0.0, 2.0), BOB)
    peak_t = max(track, key=lambda p: abs(p[1]))[0]
    assert peak_t == pytest.approx(1.0, abs=TICK)


def test_a_flick_in_progress_finishes_when_the_layer_is_disabled(manifest, mapping):
    """No step: the flick returns to zero along its own shape, then one last zero is sent."""
    beats = Beats()
    beats.stamps = [1.0]
    r = make(manifest, mapping, beats)
    before = run(r, 0.0, 0.9)  # the flick (0.3 s, peak on 1.0) is rising
    r.apply_config(DanceConfig(enabled=False, **NO_LEAD), mapping, 1.5, BARS)
    track = channel(before + run(r, 0.9, 1.5), BOB)
    peak = max(abs(v) for _, v in track)
    assert peak > 0
    assert max(abs(b[1] - a[1]) for a, b in zip(track, track[1:], strict=False)) < 0.6 * peak
    assert track[-1][1] == 0.0
    beats.stamps = [2.0]  # disabled: no new flick
    assert run(r, 1.5, 2.5)[-1][1] is None


def test_a_blirp_plays_its_accent_with_no_beat_and_no_fade(manifest, mapping):
    beats = Beats()  # a blirp: accents, no period
    beats.stamps = [0.5]
    r = make(manifest, mapping, beats, cfg=DanceConfig(fade_in_s=5.0, lead_ms=0.0))
    peak = max(abs(v) for _, v in channel(run(r, 0.0, 1.0), BOB))
    assert peak > 0.8 * r.scale(1.0) * math.radians(5.0)  # the 5 s fade does not touch it


def test_fades_in_without_jump(manifest, mapping):
    cfg = DanceConfig(lead_ms=0.0, fade_in_s=2.0)
    r = make(manifest, mapping, Beats(beat()), cfg=cfg)
    track = channel(run(r, 0.0, 3.0), BOB)
    full = r.scale(1.0) * math.radians(3.0)
    assert track[0][1] <= full * (TICK / 2.0)  # first command under the fade's slope
    worst = max(abs(b[1] - a[1]) for a, b in zip(track, track[1:], strict=False))
    assert worst < full  # no step anywhere


def test_a_tick_gap_restarts_the_fade(manifest, mapping):
    """A long sleep with no tick must not bring the layer back at full amplitude."""
    r = make(manifest, mapping, Beats(beat()))
    run(r, 0.0, 2.0)
    assert r._s.gain == pytest.approx(1.0)
    cmds = r.commands(2.0 + 10 * r.cfg.command_ttl_s)
    assert all(v == pytest.approx(0.0, abs=1e-6) for c in cmds for v in c.targets.values())
    assert r._s.gain == 0.0


def test_fades_out_when_beat_goes(manifest, mapping):
    beats = Beats(beat())
    r = make(manifest, mapping, beats, cfg=DanceConfig(lead_ms=0.0, fade_in_s=0.1, fade_out_s=0.5))
    run(r, 0.0, 1.0)
    held = r.commands(1.0)[0].targets[BOB]
    beats.beat = None
    track = channel(run(r, 1.0, 1.6), BOB)
    sizes = [abs(v) for _, v in track]
    assert sizes[0] <= abs(held) and sizes[-1] == pytest.approx(0.0, abs=1e-6)
    assert sizes == sorted(sizes, reverse=True)


def test_last_zero_command(manifest, mapping):
    beats = Beats(beat())
    r = make(manifest, mapping, beats)
    run(r, 0.0, 1.0)
    beats.beat = None
    track = run(r, 1.0, 1.0 + r.cfg.fade_out_s + 2 * TICK)
    moved = [m for _, m in track if m is not None]
    assert moved[-1] == {BOB: 0.0, SWAY: 0.0}
    assert track[-1][1] is None  # nothing more after the zero


def test_no_dance_while_resting(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()))
    run(r, 0.0, 1.0)
    r.resting = lambda: True
    end = 1.0 + r.cfg.fade_out_s + 2 * TICK
    track = run(r, 1.0, end)
    assert [m for _, m in track if m is not None][-1] == {BOB: 0.0, SWAY: 0.0}
    assert all(m is None for _, m in run(r, end, end + 1.0))  # tick by tick: no gap reset
    assert r.beat.beat_at(end) is not None


def test_circadian_factor_zero_stops_it(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()))
    r.circadian_factor = lambda: 0.0
    assert all(m is None for _, m in run(r, 0.0, 1.0))


def test_yields_to_clip_channels(manifest, mapping):
    ducked: set[str] = set()
    r = make(manifest, mapping, Beats(beat()), ducked=lambda: frozenset(ducked))
    run(r, 0.0, 1.0)
    ducked.add(BOB)
    track = run(r, 1.0, 1.5)
    assert track[-1][1][BOB] == pytest.approx(0.0, abs=1e-9)  # the clip's channel faded out
    assert abs(track[-1][1][SWAY]) > 0  # the other channel kept dancing


def test_speaking_channels_are_yielded(manifest, mapping):
    speaking: set[str] = set()
    r = make(manifest, mapping, Beats(beat()), speaking_channels=lambda: frozenset(speaking))
    run(r, 0.0, 1.0)
    speaking.add(BOB)
    assert run(r, 1.0, 1.5)[-1][1][BOB] == pytest.approx(0.0, abs=1e-9)


def test_disabled_freezes(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()))
    run(r, 0.0, 1.0)
    frozen = dict(r._s.raw)
    r.apply_config(DanceConfig(enabled=False, **NO_LEAD), mapping, 1.5, BARS)
    end = 1.0 + r.cfg.fade_out_s + 2 * TICK
    track = run(r, 1.0, end)
    moved = [m for _, m in track if m is not None]
    assert moved[-1] == {BOB: 0.0, SWAY: 0.0}
    assert frozen  # the raw offsets were frozen, not recomputed: they only ramp down
    assert all(m is None for _, m in run(r, end, end + 1.0))  # tick by tick: no gap reset


def test_amplitude_product_capped(manifest, mapping):
    cfg = DanceConfig(lead_ms=0.0, fade_in_s=0.1, arousal_gain=2.0, bob_deg=15.0)
    r = make(manifest, mapping, Beats(beat()), cfg=cfg, max_amplitude=0.5)
    r.set_affect(AffectState(0.0, 1.0, "joy", 1.0))
    run(r, 0.0, 2.0)
    assert r.scale(1.0) == 0.5
    values = [abs(v) for _, v in channel(run(r, 2.0, 2.0 + 2 * PERIOD), BOB)]
    assert max(values) == pytest.approx(0.5 * math.radians(15.0), rel=1e-2)


def test_offset_clamped_to_the_manifest_range(manifest, mapping):
    cfg = DanceConfig(lead_ms=0.0, fade_in_s=0.1, bob_deg=15.0, sway_deg=15.0)
    r = make(manifest, mapping, Beats(beat()), cfg=cfg, max_amplitude=3.0)
    r.set_register(register())
    dof = manifest.dofs[BOB]
    run(r, 0.0, 2.0)
    for _, v in channel(run(r, 2.0, 2.0 + 2 * PERIOD), BOB):
        assert dof.min - dof.home <= v <= dof.max - dof.home


def test_strength_scales_the_amplitude(manifest, mapping):
    r = make(manifest, mapping, Beats(beat(strength=0.5)))
    r.set_affect(AffectState(0.0, 0.0, "calm", 1.0))
    assert r.scale(0.5) == pytest.approx(r.scale(1.0) * 0.5)


def test_snapshot_is_counts_and_a_label_only(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()))
    run(r, 0.0, 1.0)
    snap = r.snapshot()
    assert snap == {"started": 1, "accent": 0, "skipped": 0, "source": "own"}


def test_live_snapshot_shows_the_source_and_each_roles_share(manifest, mapping):
    beats = Beats(beat(strength=0.5))
    r = make(manifest, mapping, beats, speaking_channels=lambda: frozenset({SWAY}))
    assert r.live_snapshot(0.0).active is False  # nothing ticked yet
    run(r, 0.0, 1.0)
    snap = r.live_snapshot(0.95)
    assert snap.active and snap.source == "own"
    assert snap.bob == pytest.approx(min(1.0, r.scale(0.5)))
    assert snap.sway == 0.0  # yields from the start: the roles are not swapped


def test_live_snapshot_level_follows_the_duck(manifest, mapping):
    speaking: set[str] = set()
    r = make(manifest, mapping, Beats(beat()), speaking_channels=lambda: frozenset(speaking))
    run(r, 0.0, 1.0)
    share = min(1.0, r.scale(1.0))
    assert r.live_snapshot(0.95).bob == pytest.approx(share)
    speaking.add(BOB)
    run(r, 1.0, 1.05)  # one tick of the yield fade
    snap = r.live_snapshot(1.0)
    assert snap.bob == pytest.approx(share * (1 - TICK / YIELD_S))
    assert snap.sway == pytest.approx(share)


def test_live_snapshot_level_uses_the_fade_curve(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()), cfg=DanceConfig(lead_ms=0.0, fade_in_s=1.0))
    run(r, 0.0, 0.3)
    gain = r._s.gain
    assert 0.1 < gain < 0.5
    expected = min_jerk(gain) * min(1.0, r.scale(1.0))
    assert expected != pytest.approx(gain * min(1.0, r.scale(1.0)))  # the curve is not linear here
    assert r.live_snapshot(0.25).bob == pytest.approx(expected)


def test_live_snapshot_carries_the_beat_it_follows(manifest, mapping):
    r = make(manifest, mapping, Beats(beat(phase=0.1)))
    assert r.live_snapshot(0.0).tempo_bpm() is None
    run(r, 0.0, 1.0)
    snap = r.live_snapshot(0.95)
    assert snap.tempo_bpm() == pytest.approx(60.0 / PERIOD)
    assert snap.beat_age_s(0.95) == pytest.approx(0.85)
    assert snap.beat_age_s(0.1 + PERIOD + 0.25) == pytest.approx(0.25)  # wraps per period


def test_live_snapshot_passes_a_room_beat_through(manifest, mapping):
    room = Beat(PERIOD, 0.0, 1.0, "room")
    r = make(manifest, mapping, Beats(room))
    run(r, 0.0, 1.0)
    assert r.live_snapshot(0.95).source == "room"


def test_live_snapshot_fades_with_the_beat(manifest, mapping):
    beats = Beats(beat())
    r = make(manifest, mapping, beats, cfg=DanceConfig(lead_ms=0.0, fade_in_s=0.1, fade_out_s=1.0))
    run(r, 0.0, 1.0)
    full = r.live_snapshot(0.95).bob
    beats.beat = None
    run(r, 1.0, 1.5)
    fading = r.live_snapshot(1.45)
    assert fading.active and fading.source is None and 0.0 < fading.bob < full
    run(r, 1.5, 3.0)
    done = r.live_snapshot(2.95)
    assert not done.active and done.bob == 0.0


def test_live_snapshot_is_idle_once_the_tick_stops(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()))
    run(r, 0.0, 1.0)
    stale = r.live_snapshot(0.95 + r.cfg.command_ttl_s + 0.1)
    assert not stale.active and stale.source is None and stale.bob == 0.0 and stale.sway == 0.0


def test_live_snapshot_of_a_rested_role_is_zero(manifest, mapping):
    r = make(manifest, mapping, Beats(beat(period=0.1)), cfg=DanceConfig(**NO_LEAD))  # no fold fits
    run(r, 0.0, 1.0)
    snap = r.live_snapshot(0.95)
    assert snap.source == "own" and snap.bob == 0.0


def test_live_snapshot_accent_age(manifest, mapping):
    beats = Beats()
    r = make(manifest, mapping, beats)
    run(r, 0.0, 0.5)
    assert r.live_snapshot(0.45).last_accent_age_s(0.45) is None
    beats.stamps = [1.0]
    run(r, 0.5, 1.5)
    snap = r.live_snapshot(1.45)
    assert snap.last_accent_age_s(1.25) == pytest.approx(0.25)
    assert snap.last_accent_age_s(0.8) == 0.0  # found ahead of the clock: never negative


def test_a_yielded_channel_leaves_within_the_speaking_fade(manifest, mapping):
    speaking: set[str] = set()
    cfg = DanceConfig(lead_ms=0.0, fade_in_s=0.1, fade_out_s=5.0)
    r = make(
        manifest, mapping, Beats(beat()), cfg=cfg, speaking_channels=lambda: frozenset(speaking)
    )
    run(r, 0.0, 1.0)
    speaking.add(BOB)
    track = run(r, 1.0, 1.0 + YIELD_S + 2 * TICK)
    assert track[-1][1][BOB] == 0.0  # not the 5 s of the fade-out
    assert abs(track[-1][1][SWAY]) > 0


def test_a_beat_that_returns_during_a_fade_out_rearms_without_a_step(manifest, mapping):
    beats = Beats(beat())
    cfg = DanceConfig(lead_ms=0.0, fade_in_s=1.0, fade_out_s=1.0)
    r = make(manifest, mapping, beats, cfg=cfg)
    before = run(r, 0.0, 2.0)
    beats.beat = None
    gone = run(r, 2.0, 2.3)
    assert r._s.frozen is not None
    beats.beat = beat()
    back = run(r, 2.3, 4.0)
    assert r._s.frozen is None and r._s.gain == pytest.approx(1.0)  # no wait for the fade's end
    track = channel(before + gone + back, BOB)
    peak = max(abs(v) for _, v in track)
    assert max(abs(b[1] - a[1]) for a, b in zip(track, track[1:], strict=False)) < 0.3 * peak
    assert r.counts["started"] == 1


def test_a_failed_apply_config_keeps_the_old_settings(manifest, mapping):
    cfg = DanceConfig(**NO_LEAD)
    r = make(manifest, mapping, Beats(beat()), cfg=cfg)
    r.body_yaw = "neck_pan"
    settings = r._b
    with pytest.raises(ValueError, match="dance_motion_map"):
        r.apply_config(DanceConfig(enabled=False), DanceMotionMap(sway={"neck_pan": 1}), 1.5, BARS)
    assert r._b is settings and r.cfg is cfg


def test_an_accent_on_a_yielded_channel_stays_at_zero(manifest, mapping):
    beats = Beats()
    beats.stamps = [0.5]
    r = make(manifest, mapping, beats, ducked=lambda: frozenset({BOB}))
    moved = channel(run(r, 0.0, 1.0), BOB)
    assert r.counts["accent"] == 1 and moved and all(v == 0.0 for _, v in moved)


def test_an_accent_while_resting_finishes_and_no_new_one_starts(manifest, mapping):
    beats = Beats()
    beats.stamps = [1.0, 3.0]
    r = make(manifest, mapping, beats)
    before = channel(run(r, 0.0, 0.95), BOB)
    r.resting = lambda: True
    track = before + channel(run(r, 0.95, 1.6), BOB)
    peak = max(abs(v) for _, v in track)
    assert peak > 0 and track[-1][1] == 0.0
    assert max(abs(b[1] - a[1]) for a, b in zip(track, track[1:], strict=False)) < 0.6 * peak
    assert all(m is None for _, m in run(r, 1.6, 4.0)) and r.counts["accent"] == 1


def test_a_late_stamp_starts_the_flick_from_its_zero(manifest, mapping):
    """A stalled tick (shorter than the TTL, so no reset) must not enter the shape mid-way."""
    beats = Beats()
    beats.stamps = [1.0]
    r = make(manifest, mapping, beats)
    run(r, 0.0, 0.7)
    late = r.commands(0.9)  # horizon jumps past the stamp, 0.1 s before it
    assert r.counts["accent"] == 1
    assert late[0].targets[BOB] == pytest.approx(0.0, abs=1e-9)


def test_strength_below_one_scales_the_commands(manifest, mapping):
    def peak(strength):
        r = make(manifest, mapping, Beats(beat(strength=strength)))
        run(r, 0.0, 1.0)
        return max(abs(v) for _, v in channel(run(r, 1.0, 1.0 + 2 * PERIOD), BOB))

    assert peak(0.5) == pytest.approx(0.5 * peak(1.0), rel=0.02)


def test_roles_on_one_channel_are_capped_together(manifest, mapping):
    """``bob`` and the accent share a channel: the sum stays under the cap of the larger."""
    beats = Beats(beat())
    beats.stamps = [2.4]  # on a beat: the dip's deepest point
    cfg = DanceConfig(
        lead_ms=0.0, fade_in_s=0.1, arousal_gain=2.0, bob_deg=5.0, accent_deg=5.0, sway_deg=0.0
    )
    r = make(manifest, mapping, beats, cfg=cfg, max_amplitude=1.0)
    r.set_affect(AffectState(0.0, 1.0, "joy", 1.0))
    assert r.scale(1.0) == 1.0
    assert r.counts["accent"] == 0
    moved = channel(run(r, 0.0, 3.0), BOB)
    assert r.counts["accent"] == 1
    assert max(abs(v) for _, v in moved) == pytest.approx(math.radians(5.0), rel=1e-6)


def test_sleep_entry_moves_nothing_and_the_wake_fades_in(manifest, mapping):
    """The app tick pauses only while the motors are held off: the loop writes nothing as the
    last command expires, and the dance comes back from zero."""
    cfg = ConfigStack().effective()
    driver = MockPoseDriver(home={n: d.home for n, d in manifest.dofs.items()})
    loop = MotionLoop(manifest, driver, cfg, rate_hz=50, clock=lambda: 0.0)
    loop.set_mode("active")
    r = make(manifest, mapping, Beats(beat()))
    t = 0.0
    while t < 1.0:
        for cmd in r.commands(t):
            loop.feed(cmd)
        loop.cycle(t, TICK)
        t = round(t + TICK, 6)
    assert driver.writes
    driver.mode = "limp"  # the platform puts the robot to sleep
    while not loop.motors_off and t < 2.0:  # the loop sees it at its next status probe
        loop.cycle(t, TICK)
        t = round(t + TICK, 6)
    assert loop.motors_off
    n = len(driver.writes)
    while t < 3.0:  # asleep: no app tick, so the last command expires
        t = round(t + TICK, 6)
        loop.cycle(t, TICK)
    assert len(driver.writes) == n
    assert all(abs(v) < 1e-6 for c in r.commands(t) for v in c.targets.values())
    assert r._s.gain == 0.0


# ------------------------------------------------------------------ map and config


def test_map_with_body_yaw_refused(manifest):
    """Load and apply_config: the dance never turns the body."""
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        {"dance_motion_map": {"sway": {"neck_pan": 1}}},
        manifest.channels,
        frozenset(),
        body_yaw=None,
    )
    assert stack.effective().dance_motion_map.sway == {"neck_pan": 1}
    with pytest.raises(ConfigError, match="dance_motion_map.sway"):
        ConfigStack().set_manifest_behaviour(
            {"dance_motion_map": {"sway": {"neck_pan": 1}}},
            manifest.channels,
            frozenset(),
            body_yaw="neck_pan",
        )
    r = DanceRenderer(
        DanceConfig(),
        DanceMotionMap(),
        manifest.dofs,
        body_yaw="neck_pan",
        max_amplitude=1.5,
        beats_per_bar=BARS,
        beat=BeatSource(lambda n: None, lambda n: None),
        yield_fade_s=lambda: YIELD_S,
    )
    with pytest.raises(ValueError, match="dance_motion_map.sway"):
        r.apply_config(DanceConfig(), DanceMotionMap(sway={"neck_pan": 1}), 1.5, BARS)


def test_a_runtime_override_naming_body_yaw_is_refused(manifest):
    stack = ConfigStack()
    stack.set_manifest_behaviour({}, manifest.channels, frozenset(), body_yaw="neck_pan")
    with pytest.raises(ConfigError, match="turns the body"):
        stack.set("dance_motion_map.bob", {"neck_pan": 1})


def test_unknown_channel_refused_at_load(manifest):
    with pytest.raises(ConfigError, match="dance_motion_map"):
        ConfigStack().set_manifest_behaviour(
            {"dance_motion_map": {"bob": {"no.such.dof": 1}}},
            manifest.channels,
            frozenset(),
            body_yaw=None,
        )


def test_every_field_has_a_range_and_the_timed_ones_a_unit():
    for name, f in DanceConfig.model_fields.items():
        extra = f.json_schema_extra or {}
        if name.endswith(("_deg", "_ms", "_s")):
            assert extra.get("unit"), name
        if f.annotation in ("float", "int", float, int):
            bounds = {type(m).__name__ for m in f.metadata}
            assert {"Ge", "Le"} <= bounds, name


def test_command_ttl_must_outlast_one_app_tick():
    with pytest.raises(ValidationError, match="dance.command_ttl_s"):
        RoboEmotConfig.model_validate(
            {"dance": {"command_ttl_s": 0.05}, "motion": {"app_tick_hz": 20}}
        )
