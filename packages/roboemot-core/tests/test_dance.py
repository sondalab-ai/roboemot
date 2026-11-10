"""SPEC 7 (Dance motion), 11A: the dance renderer on a canned beat, the map validation and the
config rules. The robot is the generic ``tests/data/dance-head.yaml``."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from audio_signals import drum_loop
from pydantic import ValidationError
from roboemot.affect import AffectState
from roboemot.affect.dance import SOURCE, DanceRenderer, fold
from roboemot.affect.register import Person, resolve_register
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import (
    DanceConfig,
    DanceMotionMap,
    DanceMove,
    MusicConfig,
    RoboEmotConfig,
)
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.motion import MotionLoop, Priority
from roboemot.motion.beat import Beat, BeatSource, room_beat
from roboemot.motion.curves import min_jerk
from roboemot.runtime.ticks import dance_calm
from roboemot.voice.music import MusicDetector, MusicState
from roboemot.voice.speech import SpeechOutput, _Job
from test_speech import FakeBackend, TimelineSink

MANIFEST = Path(__file__).resolve().parent / "data" / "dance-head.yaml"
TICK = 0.05  # the 20 Hz app tick
BOB, SWAY = "neck_tilt", "neck_pan"
PERIOD = 1.2  # one bob per beat: 2 x min_half_s (0.5) fits
BARS = 4
NO_LEAD = {"lead_ms": 0.0, "fade_in_s": 0.1, "fade_out_s": 0.1}
YIELD_S = 0.15  # the speaking motion's default fade_ms
RATE = 16000
CHUNK = 320


def feed(det, signal, t0):
    """Feed ``signal`` from ``t0`` in chunks, as the listener does."""
    for i in range(0, signal.size, CHUNK):
        det.feed(signal[i : i + CHUNK], t0 + i / RATE)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST, allow_unmeasured_dynamics=True)


@pytest.fixture(scope="module")
def mapping(manifest):
    return DanceMotionMap.model_validate(
        {**manifest.model.behaviour["dance_motion_map"], "moves": {}}
    )


@pytest.fixture(scope="module")
def moves_mapping(manifest):
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


class Room:
    """A music detector's state as the room provider reads it: the time since the last beat of
    a grid anchored at ``anchor``."""

    def __init__(self, bpm: float, *, confidence: float = 1.0, anchor: float = 0.0) -> None:
        self.bpm, self.confidence, self.anchor = bpm, confidence, anchor
        self.playing = True

    @property
    def period(self) -> float:
        return 60.0 / self.bpm

    def state(self, now: float) -> MusicState:
        if not self.playing:
            return MusicState(False, None, None, self.confidence, now)
        since = (now - self.anchor) % self.period
        return MusicState(True, self.bpm, since, self.confidence, now)

    def source(self) -> BeatSource:
        return BeatSource(lambda now: None, lambda now: room_beat(self.state(now)))


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


def test_live_snapshot_level_is_the_ramped_level_the_renderer_uses(manifest, mapping):
    cfg = DanceConfig(lead_ms=0.0, fade_in_s=0.1, fade_out_s=0.1, speaking_gain=0.5)
    room = Beat(PERIOD, 0.0, 1.0, "room")
    r = make(manifest, mapping, Beats(room), cfg=cfg, speaking=lambda _t: True)
    run(r, 0.0, 1.0)
    assert r.live_snapshot(0.95).bob == pytest.approx(min(1.0, r.scale(0.5)))


def test_live_snapshot_pulse_follows_the_beat_at_a_folded_move(manifest, mapping):
    period = 0.6  # the bob moves every second beat: 0.6 x 2 clears 2 x min_half_s
    r = make(manifest, mapping, Beats(beat(period=period)))
    run(r, 0.0, 1.0)
    assert r._s.moves["bob"].factor == 2
    snap = r.live_snapshot(0.95)
    assert snap.tempo_bpm() == pytest.approx(60.0 / period)
    assert snap.beat_age_s(0.95) == pytest.approx(0.95 % period)


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


# ------------------------------------------------------------------ room music (SPEC 7, 8)

SETTLE_S = 3.0  # past every fade of NO_LEAD
MIN_JERK_SLOPE = 1.875  # the steepest rate of min_jerk over its unit interval
SLOPE_MARGIN = 1.15  # sampling and float room over the analytic bound


def peaks(track):
    """Stamps of the local maxima of ``|value|`` that reach most of the track's largest."""
    values = [abs(v) for _, v in track]
    top = max(values)
    return [
        track[i][0]
        for i in range(1, len(values) - 1)
        if values[i] >= 0.9 * top and values[i - 1] < values[i] >= values[i + 1]
    ]


def steady(r, seconds):
    """The track of ``r`` after it has settled, for ``seconds``."""
    run(r, 0.0, SETTLE_S)
    return run(r, SETTLE_S, SETTLE_S + seconds)


def amplitude(track, name):
    return max(abs(v) for _, v in channel(track, name))


@pytest.mark.parametrize(
    ("bpm", "factor"),
    [(170.0, 4), (100.0, 2), (50.0, 1)],
)
def test_tempo_folding(manifest, mapping, bpm, factor):
    cfg = DanceConfig(**NO_LEAD)
    room = Room(bpm)
    assert fold(room.period, cfg.min_half_s) == factor
    assert (factor * room.period) / 2 >= cfg.min_half_s
    track = channel(steady(make(manifest, mapping, room, cfg=cfg), 8 * factor * room.period), BOB)
    gaps = [b - a for a, b in zip(peaks(track), peaks(track)[1:], strict=False)]
    assert gaps
    assert all(abs(g - factor * room.period) <= 2 * TICK for g in gaps), gaps
    assert all(g / 2 >= cfg.min_half_s - TICK for g in gaps)


@pytest.mark.parametrize("bpm", [100.0, 170.0])
def test_room_grid_is_smooth_across_beats_and_re_estimates(manifest, mapping, bpm):
    """A grid that re-anchors on every beat (what the detector gives) must not restart a move."""
    room = Room(bpm)
    r = make(manifest, mapping, room)
    run(r, 0.0, SETTLE_S)
    room.anchor += 3 * room.period  # an estimate: another beat of the same grid
    track = run(r, SETTLE_S, SETTLE_S + 16 * room.period)
    min_half_s = DanceConfig().min_half_s
    bob_span = fold(room.period, min_half_s) * room.period
    sway_span = fold(room.period * BARS, min_half_s) * room.period * BARS
    for name, span in ((BOB, bob_span), (SWAY, sway_span)):
        values = [v for _, v in channel(track, name)]
        amp = max(abs(v) for v in values)
        bound = amp * MIN_JERK_SLOPE * 2 * TICK / span * SLOPE_MARGIN
        assert max(abs(b - a) for a, b in zip(values, values[1:], strict=False)) <= bound


def test_lead_ms_shifts_peak(manifest, mapping):
    def peak_time(lead_ms):
        cfg = DanceConfig(**{**NO_LEAD, "lead_ms": lead_ms})
        track = channel(steady(make(manifest, mapping, Beats(beat()), cfg=cfg), 2 * PERIOD), BOB)
        return peaks(track)[0]

    assert peak_time(0.0) - peak_time(200.0) == pytest.approx(0.2, abs=TICK)


def test_speaking_gain_scales_amplitude(manifest, mapping):
    speaking = [False]
    cfg = DanceConfig(**NO_LEAD, speaking_gain=0.3)
    r = make(manifest, mapping, Room(100.0), cfg=cfg, speaking=lambda t: speaking[0])
    window = 2 * Room(100.0).period * 2
    run(r, 0.0, SETTLE_S)
    full = amplitude(run(r, SETTLE_S, SETTLE_S + window), SWAY)
    speaking[0] = True
    run(r, SETTLE_S + window, 2 * SETTLE_S + window)
    start = 2 * SETTLE_S + window
    spoken = amplitude(run(r, start, start + window), SWAY)
    assert spoken == pytest.approx(0.3 * full, rel=0.1)


def test_speaking_gain_leaves_an_own_beat_alone(manifest, mapping):
    r = make(manifest, mapping, Beats(beat()), speaking=lambda t: True)
    quiet = make(manifest, mapping, Beats(beat()))
    assert amplitude(steady(r, 2 * PERIOD), BOB) == pytest.approx(
        amplitude(steady(quiet, 2 * PERIOD), BOB)
    )


def test_speaking_channel_handover(manifest, mapping):
    drives = {BOB}
    cfg = DanceConfig(**NO_LEAD, speaking_gain=0.3)
    silent = make(manifest, mapping, Room(100.0), cfg=cfg)
    r = make(
        manifest,
        mapping,
        Room(100.0),
        cfg=cfg,
        speaking=lambda t: True,
        speaking_channels=lambda: frozenset(drives),
    )
    window = 4 * Room(100.0).period
    track = steady(r, window)
    assert amplitude(track, BOB) == 0.0  # the speaking motion owns it
    assert amplitude(track, SWAY) == pytest.approx(
        0.3 * amplitude(steady(silent, window), SWAY), rel=0.1
    )
    drives.clear()
    back = run(r, SETTLE_S + window, SETTLE_S + 2 * window)
    assert amplitude(back, BOB) > 0.0


def room_size(manifest, mapping, confidence, min_strength):
    """The steady sway amplitude at a room confidence with ``dance.min_strength``."""
    cfg = DanceConfig(**NO_LEAD, min_strength=min_strength)
    r = make(manifest, mapping, Room(100.0, confidence=confidence), cfg=cfg)
    return amplitude(steady(r, 8 * Room(100.0).period), SWAY)


def test_room_beat_strength_scales_amplitude(manifest, mapping):
    full = room_size(manifest, mapping, 1.0, 0.0)
    assert room_size(manifest, mapping, 0.5, 0.0) == pytest.approx(0.5 * full, rel=0.05)


def test_min_strength_raises_a_weak_room_beat(manifest, mapping):
    full = room_size(manifest, mapping, 1.0, 0.7)
    assert room_size(manifest, mapping, 0.3, 0.7) == pytest.approx(0.7 * full, rel=0.05)


def test_min_strength_leaves_a_strong_room_beat_alone(manifest, mapping):
    full = room_size(manifest, mapping, 1.0, 0.7)
    assert room_size(manifest, mapping, 0.9, 0.7) == pytest.approx(0.9 * full, rel=0.05)


def test_min_strength_leaves_an_own_beat_alone(manifest, mapping):
    def size(min_strength):
        cfg = DanceConfig(**NO_LEAD, min_strength=min_strength)
        r = make(manifest, mapping, Beats(beat(strength=0.5)), cfg=cfg)
        return amplitude(steady(r, 8 * PERIOD), SWAY)

    assert size(0.9) == pytest.approx(size(0.0), rel=0.01)


def test_a_new_confidence_is_no_step(manifest, mapping):
    room = Room(100.0)
    cfg = DanceConfig(**{**NO_LEAD, "fade_out_s": 1.0})
    r = make(manifest, mapping, room, cfg=cfg)
    run(r, 0.0, SETTLE_S)
    room.confidence = 0.2
    r.commands(SETTLE_S)
    assert r._s.level == pytest.approx(1.0 - TICK / cfg.fade_out_s)


def test_no_beat_when_the_room_goes_quiet(manifest, mapping):
    room = Room(100.0)
    r = make(manifest, mapping, room)
    run(r, 0.0, SETTLE_S)
    room.playing = False
    track = run(r, SETTLE_S, SETTLE_S + 1.0)
    assert r.snapshot()["source"] is None
    assert track[-1][1] is None  # nothing left to send
    sent = [m for _, m in track if m is not None]
    assert all(v == 0.0 for v in sent[-1].values())  # the last command returned to zero


# ------------------------------------------------------------------ continuity (SPEC 7)

MIN_HALF_S = DanceConfig().min_half_s


def steps(track, name):
    values = [v for _, v in channel(track, name)]
    return [abs(b - a) for a, b in zip(values, values[1:], strict=False)]


def slope_bound(amp):
    """Largest per-tick change of a move whose halves last at least ``min_half_s``."""
    return amp * MIN_JERK_SLOPE * 2 * TICK / (2 * MIN_HALF_S) * SLOPE_MARGIN


def assert_smooth(track, amp_bob, amp_sway):
    assert max(steps(track, BOB)) <= slope_bound(amp_bob)
    assert max(steps(track, SWAY)) <= slope_bound(amp_sway)


def settled_amplitudes(r, seconds):
    run(r, 0.0, SETTLE_S)
    track = run(r, SETTLE_S, SETTLE_S + seconds)
    return track, amplitude(track, BOB), amplitude(track, SWAY)


def test_fold_keeps_its_factor_inside_the_margin():
    margin = DanceConfig().fold_margin
    assert fold(0.4996, 0.5, 2, margin) == 4  # no longer fits
    assert fold(0.5004, 0.5, 4, margin) == 4  # fits at 2, not by the margin
    assert fold(0.5004, 0.5, 2, margin) == 2  # kept while it fits
    assert fold(0.56, 0.5, 4, margin) == 2  # clears the margin
    assert fold(0.5004, 0.5) == 2  # no current fold: the shortest that fits


def test_tempo_crossing_a_fold_threshold_never_steps(manifest, mapping):
    room = Room(119.9)
    r = make(manifest, mapping, room)
    _, amp_bob, amp_sway = settled_amplitudes(r, 4.0)
    start = SETTLE_S + 4.0
    track = []
    for i in range(16):  # an estimate a second, around 120 BPM
        room.bpm = 120.1 if i % 2 == 0 else 119.9
        track += run(r, start + i, start + i + 1.0)
    assert r._s.moves["bob"].factor == 4  # flipped once, then held
    assert_smooth(track, amp_bob, amp_sway)
    dips = peaks(channel(track, BOB))
    assert all(b - a >= 2 * MIN_HALF_S - TICK for a, b in zip(dips, dips[1:], strict=False))


def test_a_fractional_grid_shift_never_steps_and_converges(manifest, mapping):
    room = Room(100.0)
    r = make(manifest, mapping, room)
    _, amp_bob, amp_sway = settled_amplitudes(r, 6 * room.period)
    start = SETTLE_S + 6 * room.period
    room.anchor += 0.3 * room.period
    track = run(r, start, start + 3.0)
    assert_smooth(track, amp_bob, amp_sway)
    later = run(r, start + 3.0, start + 3.0 + 6 * room.period)
    for dip in peaks(channel(later, BOB)):
        off = math.remainder(dip - room.anchor, room.period)
        assert abs(off) <= 2 * TICK, dip


@pytest.mark.parametrize("new_bpm", [100.0, 150.0])
def test_a_tick_stall_restarts_from_zero(manifest, mapping, new_bpm):
    room = Room(100.0)
    cfg = DanceConfig(**{**NO_LEAD, "fade_in_s": 1.0})
    r = make(manifest, mapping, room, cfg=cfg)
    run(r, 0.0, SETTLE_S)
    room.bpm = new_bpm
    resume = SETTLE_S + 5 * room.period  # far beyond command_ttl_s
    assert r.commands(resume) == []  # the layer restarted: nothing held, the gain is zero
    second = r.commands(resume + TICK)[0].targets
    assert max(abs(v) for v in second.values()) <= math.radians(cfg.bob_deg) * 1.5 * TICK


def test_a_short_stall_only_advances_the_move(manifest, mapping):
    """A hiccup below ``command_ttl_s``: the place moves by the elapsed time, no restart."""
    room = Room(100.0)
    r = make(manifest, mapping, room)
    _, amp_bob, _ = settled_amplitudes(r, 4 * room.period)
    now = SETTLE_S + 4 * room.period
    before = r.commands(now)[0].targets[BOB]
    stall = 0.2
    after = r.commands(now + stall)[0].targets[BOB]
    assert abs(after - before) <= slope_bound(amp_bob) * stall / TICK


def test_resting_then_resuming_fades_in_without_a_step(manifest, mapping):
    room = Room(100.0)
    cfg = DanceConfig(**{**NO_LEAD, "fade_in_s": 1.0})
    resting = [False]
    r = make(manifest, mapping, room, cfg=cfg)
    r.resting = lambda: resting[0]
    run(r, 0.0, SETTLE_S)
    resting[0] = True
    run(r, SETTLE_S, SETTLE_S + 4.0)
    assert r.snapshot()["source"] is None
    resting[0] = False
    first = run(r, SETTLE_S + 4.0, SETTLE_S + 4.2)
    moved = [abs(v) for _, m in first if m for v in m.values()]
    assert max(moved) <= math.radians(cfg.bob_deg) * 1.5 * 0.2


def test_own_to_room_and_back_never_steps(manifest, mapping):
    room = Room(100.0)
    own = [beat(period=PERIOD)]
    source = BeatSource(lambda now: own[0], lambda now: room_beat(room.state(now)))
    holder = type("Holder", (), {"source": lambda self: source})()
    r = make(manifest, mapping, holder)
    _, amp_bob, amp_sway = settled_amplitudes(r, 4 * PERIOD)
    start = SETTLE_S + 4 * PERIOD
    own[0] = None
    to_room = run(r, start, start + 4.0)
    assert r.snapshot()["source"] == "room"
    own[0] = beat(period=PERIOD, phase=start + 4.0 + 0.37)
    back = run(r, start + 4.0, start + 8.0)
    assert r.snapshot()["source"] == "own"
    assert_smooth(to_room, amp_bob, amp_sway)
    assert_smooth(back, amp_bob, amp_sway)


def test_a_rising_confidence_ramps_up(manifest, mapping):
    room = Room(100.0, confidence=0.2)
    cfg = DanceConfig(**{**NO_LEAD, "fade_in_s": 2.0, "min_strength": 0.0})
    r = make(manifest, mapping, room, cfg=cfg)
    run(r, 0.0, SETTLE_S)
    assert r._s.level == pytest.approx(0.2)
    room.confidence = 1.0
    r.commands(SETTLE_S)
    assert r._s.level == pytest.approx(0.2 + TICK / cfg.fade_in_s)


def test_blind_hold_expiry_fades_out_then_restarts_cleanly(manifest, mapping):
    """A real detector: nothing heard past ``music.blind_hold_s`` ends the beat; music heard
    again confirms anew and the dance fades in from zero."""
    music = MusicConfig(blind_hold_s=2.0)
    det = MusicDetector(music, RATE)
    feed(det, drum_loop(120, 8.0, RATE), 0.0)
    holder = type(
        "Holder",
        (),
        {"source": lambda self: BeatSource(lambda n: None, lambda n: room_beat(det.state(n)))},
    )()
    cfg = DanceConfig(**{**NO_LEAD, "fade_in_s": 1.0, "fade_out_s": 0.5})
    r = make(manifest, mapping, holder, cfg=cfg)
    run(r, 6.0, 9.5)  # heard until about 8.0 (a batch may wait), held for 2 s more
    assert r.snapshot()["source"] == "room"
    ended = run(r, 9.5, 11.0)
    assert r.snapshot()["source"] is None
    assert ended[-1][1] is None
    again = 12.0
    feed(det, drum_loop(120, 8.0, RATE, seed=5), again)
    assert det.state(again + 8.0).playing
    first = run(r, again + 8.0, again + 8.2)
    assert r.snapshot()["source"] == "room"
    moved = [abs(v) for _, m in first if m for v in m.values()]
    assert max(moved) <= math.radians(cfg.bob_deg) * 1.5 * 0.2 / cfg.fade_in_s + 1e-9


def test_speaking_gain_holds_across_a_gap_between_sentences(manifest, mapping):
    """The real busy signal: no sentence on the timeline, the next one queued. Then the answer
    is over and the gain comes back."""
    out = SpeechOutput(RoboEmotConfig(), TimelineSink(), FakeBackend())
    playing = [True]
    out.timeline._sentence_at = lambda t: object() if playing[0] else None
    cfg = DanceConfig(**NO_LEAD, speaking_gain=0.3)
    r = make(manifest, mapping, Room(100.0), cfg=cfg, speaking=out.speaking_at)
    run(r, 0.0, SETTLE_S)
    assert r._s.level == pytest.approx(0.3)
    playing[0] = False
    out._jobs.append(object.__new__(_Job))
    run(r, SETTLE_S, SETTLE_S + 1.0)
    assert r._s.level == pytest.approx(0.3)  # the gap does not pump
    out._jobs.clear()
    run(r, SETTLE_S + 1.0, SETTLE_S + 2.0)
    assert r._s.level == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("enabled", "calm", "playing", "want"),
    [
        pytest.param(True, True, True, True, id="music"),
        pytest.param(True, True, False, False, id="no-music"),
        pytest.param(True, False, True, False, id="calm-off"),
        pytest.param(False, True, True, False, id="dance-off"),
    ],
)
def test_other_moves_stay_calm_only_while_dancing_to_room_music(enabled, calm, playing, want):
    cfg = DanceConfig(enabled=enabled, calm_other_moves=calm)
    listener = SimpleNamespace(music_state=lambda _now: SimpleNamespace(playing=playing))
    host = SimpleNamespace(dance=SimpleNamespace(cfg=cfg), voice_listener=listener)
    assert dance_calm(host, 1.0) is want


def test_nothing_stays_calm_without_a_listener():
    host = SimpleNamespace(dance=SimpleNamespace(cfg=DanceConfig()), voice_listener=None)
    assert dance_calm(host, 1.0) is False


MOVES_CFG = {"lead_ms": 0.0, "fade_in_s": 0.1, "fade_out_s": 0.1, "min_strength": 1.0}
MOVES_MAP = DanceMotionMap.model_validate(
    {
        "bob": {BOB: 1},
        "moves": {
            "nod": {BOB: {"deg": 15}},
            "tilt": {SWAY: {"deg": 16, "cycles": 0.5}},
        },
    }
)


class Free:
    """Music with no readable beat at ``bpm`` (the ``free`` source)."""

    def __init__(self, bpm=60.0, strength=1.0):
        self.beat = Beat(60.0 / bpm, 0.0, strength, "free")

    def source(self) -> BeatSource:
        return BeatSource(lambda now: None, lambda now: self.beat)


def test_free_music_plays_only_the_free_move(manifest):
    cfg = DanceConfig(**MOVES_CFG, free_move="tilt", move_bars=1)
    r = make(manifest, MOVES_MAP, Free(), cfg=cfg, beats_per_bar=2)
    run(r, 0.0, 1.0)
    track = run(r, 1.0, 12.0)
    assert max(abs(v) for _, v in channel(track, SWAY)) > 0.1
    assert all(abs(v) < 1e-9 for _, v in channel(track, BOB))


def test_without_the_free_move_free_music_plays_bob_and_sway(manifest):
    cfg = DanceConfig(**MOVES_CFG, free_move="missing")
    r = make(manifest, MOVES_MAP, Free(), cfg=cfg)
    run(r, 0.0, 1.0)
    assert max(abs(v) for _, v in channel(run(r, 1.0, 6.0), BOB)) > 0


def test_a_tempo_continues_the_free_move_without_a_step(manifest):
    cfg = DanceConfig(**MOVES_CFG, free_move="tilt", move_bars=4)
    free, room = Free(), Beat(0.5, 0.0, 1.0, "room")
    current = [free.beat]
    source = BeatSource(lambda now: None, lambda now: current[0])
    holder = SimpleNamespace(source=lambda: source)
    r = make(manifest, MOVES_MAP, holder, cfg=cfg)
    before = channel(run(r, 0.0, 5.0), SWAY)[-1][1]
    current[0] = room
    after = channel(run(r, 5.0, 5.0 + TICK * 2), SWAY)[0][1]
    assert abs(after - before) < math.radians(3)


def test_room_music_plays_a_move_per_beat_at_full_size(manifest):
    r = make(manifest, MOVES_MAP, Room(60.0), cfg=DanceConfig(**MOVES_CFG))
    run(r, 0.0, 1.0)  # past the fade in
    tilt = [t[BOB] for _, t in run(r, 1.0, 3.0)]
    crossings = sum(1 for a, b in zip(tilt, tilt[1:], strict=False) if a < 0 <= b)
    assert crossings == 2  # two beats at 60 BPM, one full nod each
    assert max(tilt) >= math.radians(13.0)


def test_the_own_sound_keeps_the_gentle_bob(manifest):
    r = make(manifest, MOVES_MAP, Beats(beat()), cfg=DanceConfig(**MOVES_CFG))
    run(r, 0.0, 1.0)
    track = run(r, 1.0, 3.0)
    assert max(t[BOB] for _, t in track) < math.radians(4.0)
    assert all(SWAY not in t for _, t in track)


def test_moves_can_be_turned_off(manifest):
    cfg = DanceConfig(**MOVES_CFG, moves_enabled=False)
    r = make(manifest, MOVES_MAP, Room(60.0), cfg=cfg)
    run(r, 0.0, 1.0)
    assert max(t[BOB] for _, t in run(r, 1.0, 3.0)) < math.radians(4.0)


def test_the_next_move_starts_after_the_bars_and_blends(manifest):
    cfg = DanceConfig(**MOVES_CFG, move_bars=1, move_blend_beats=1.0)
    r = make(manifest, MOVES_MAP, Room(60.0), cfg=cfg, beats_per_bar=2)
    first = run(r, 0.0, 1.9)
    assert all(abs((t or {}).get(SWAY, 0.0)) < 1e-9 for _, t in first)  # only the nod so far
    later = run(r, 1.9, 6.0)
    assert max(abs((t or {}).get(SWAY, 0.0)) for _, t in later) >= math.radians(13.0)


def test_a_wave_is_reduced_to_what_the_caps_allow(manifest):
    r = make(manifest, MOVES_MAP, Room(60.0), cfg=DanceConfig(**MOVES_CFG))
    fast = {SWAY: DanceMove(deg=45.0, cycles=2.0)}
    dof = manifest.dofs[SWAY]
    out = r._move_offsets(fast, 0.0, 0.5, 1.0)
    w = 2 * math.pi * 2.0 / 0.5
    assert out[SWAY] == pytest.approx(min(dof.max_vel / w, dof.max_acc / w**2))


def test_a_wave_keeps_its_direction(manifest):
    r = make(manifest, MOVES_MAP, Room(60.0), cfg=DanceConfig(**MOVES_CFG))
    assert r._move_offsets({SWAY: DanceMove(deg=-10.0)}, 0.0, 1.0, 1.0)[SWAY] < 0


@pytest.mark.parametrize("cycles", [0.75, 1.25])
def test_a_wave_must_loop_without_a_step(cycles):
    with pytest.raises(ValidationError):
        DanceMove(deg=10.0, cycles=cycles)


def test_the_reachy_moves_stay_large_enough_to_show():
    raw = load_manifest(
        Path(__file__).resolve().parents[3] / "configs/manifests/reachy-mini-wireless.yaml",
        allow_unmeasured_dynamics=True,
    ).model.behaviour["dance_motion_map"]
    moves = DanceMotionMap.model_validate(raw).moves
    assert len(moves) >= 5
    for move in moves.values():
        sizes = [abs(w.deg) for name, w in move.items() if name.startswith("head.")]
        assert sizes and max(sizes) >= 12.0
