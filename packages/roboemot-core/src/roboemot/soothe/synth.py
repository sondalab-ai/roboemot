"""Procedural self-soothing sounds: a childish hum or machine blirps (SPEC 7B).

Pure numpy; the caller injects the ``random.Random``, so a seed reproduces a sound and the module
reads no clock and no global generator. Every sound is *planned* first (notes or events, drawn
from the :class:`~roboemot.config.schema.SoundShape` ranges) and *rendered* second, so the plan
can be inspected without analysing audio. All sounds start and end exactly at 0 and peak at the
shape's ``gain_db``.

The constants below are fixed facts of the sound design (scale intervals, envelope times, the
character of each machine sound), not behaviour a deployment tunes; the ranges that make one
sound differ from the next live in ``soothe.variants[].shape`` (SPEC 11A).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..config.schema import SoundShape

SoundKind = Literal["hum", "blirp"]

SECONDS_PER_MINUTE = 60.0
SEMITONES_PER_OCTAVE = 12.0
DB_BASE = 10.0
DB_PER_AMPLITUDE_DECADE = 20.0
PEAK_MARGIN = 0.999  # the float32 cast must never push the peak over the gain
MAX_FREQ_SHARE = 0.35  # of the sample rate: partials above stay clear of the Nyquist alias
FADE_S = 0.04  # raised-cosine fade at both ends of every sound: starts and ends at 0, no click

# --- hum: major pentatonic over the base pitch, nursery-rhyme rhythm cells
PENTATONIC_ST = (0, 2, 4, 7, 9)
DEGREES_PER_OCTAVE = len(PENTATONIC_ST)
HUM_DEGREE_RANGE = (-2, 7)  # scale degrees the walk stays in (about a ninth and a half)
HUM_START_DEGREES = (0, 2, 4)  # tonic, third, fifth
HUM_LEAP_PROB = 0.15  # the rest of the moves are one scale degree
HUM_LEAP_DEGREES = (2, 3)
HUM_REPEAT_PROB = 0.12  # a note sung again (nursery rhymes do it a lot)
HUM_CELLS = (  # note lengths in beats
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 0.5, 0.5, 1.0),
    (0.5, 0.5, 1.0, 1.0, 1.0),
    (1.0, 0.5, 0.5, 1.0),
    (0.5, 0.5, 0.5, 0.5, 1.0),
    (1.5, 0.5, 1.0, 1.0),
)
HUM_FINAL_BEATS = 2.0  # the closing tonic is held
HARMONIC_GAINS = (1.0, 0.25, 0.1)  # sine plus two weak harmonics: a soft voice
VIBRATO_DEPTH_ST = 0.2
VIBRATO_RATE_HZ = (4.5, 6.0)
VIBRATO_ONSET_S = 0.3  # vibrato eases in, as a voice does
GLIDE_S = 0.07
GLIDE_MAX_SHARE = 0.3  # of the note it leaves
NOTE_ATTACK_S = 0.04
NOTE_DECAY_S = 0.08
NOTE_RELEASE_S = 0.08
NOTE_SUSTAIN = 0.8
NOTE_DIP = 0.35  # level between two notes: sung, not played legato

# --- blirp: a small friendly droid
BLIRP_COUNT = (2, 8)
BLIRP_MIN_SLOT_S = 0.1  # least time per sound plus its gap
BLIRP_MAX_SLOT_S = 0.6  # longer sounds are made of more events, not of longer gaps
BLIRP_SOUND_BEATS = 0.5  # longest sound, in beats of the shape's tempo
BLIRP_MIN_SOUND_S = 0.05
BLIRP_DUR_SHARE = 0.7  # of the length that sounds fill, the rest is gaps
BLIRP_DUR_JITTER = (0.7, 1.3)
BLIRP_GAP_JITTER = (0.3, 1.5)
BLIRP_PITCH_ST = (-5.0, 12.0)  # an event's pitch against the sound's base pitch
BLIRP_KINDS = ("chirp_up", "chirp_down", "fm", "trill", "warble")
CHIRP_SWEEP_ST = (7.0, 24.0)
FM_RATIOS = (1.0, 1.5, 2.0, 3.0)
FM_INDEX = (1.0, 4.0)
TRILL_INTERVAL_ST = (2.0, 5.0)
TRILL_RATE_HZ = (12.0, 20.0)
TRILL_SMOOTH_S = 0.004  # the jump between the two trill pitches is smoothed over this
WARBLE_DEPTH_ST = (2.0, 5.0)
WARBLE_RATE_HZ = (6.0, 10.0)
BLIRP_EDGE_S = 0.01  # attack and release of one blirp


@dataclass(frozen=True)
class HumNote:
    """One sung note: scale degree (0 is the tonic), pitch, start and length in seconds."""

    degree: int
    freq_hz: float
    start_s: float
    dur_s: float


@dataclass(frozen=True)
class HumPlan:
    """The melody of a hum. ``freq_hz`` of each note is ``base_hz`` raised by the pentatonic
    interval of its degree; the last note is the tonic."""

    base_hz: float
    length_s: float
    gain: float
    vibrato_hz: float
    notes: tuple[HumNote, ...]


@dataclass(frozen=True)
class BlirpEvent:
    """One machine sound. ``f1_hz`` is where a chirp ends or a trill jumps to; ``mod_hz`` and
    ``depth`` drive the FM bleep, the trill rate and the warble."""

    kind: str
    start_s: float
    dur_s: float
    f0_hz: float
    f1_hz: float
    mod_hz: float
    depth: float


@dataclass(frozen=True)
class BlirpPlan:
    """The sequence of machine sounds of a blirp, 2 to 8 events."""

    length_s: float
    gain: float
    events: tuple[BlirpEvent, ...]


def degree_freq_hz(base_hz: float, degree: int) -> float:
    """Frequency of scale ``degree`` (0 is the tonic) of the major pentatonic over ``base_hz``."""
    octave, step = divmod(degree, DEGREES_PER_OCTAVE)
    semitones = SEMITONES_PER_OCTAVE * octave + PENTATONIC_ST[step]
    return base_hz * 2.0 ** (semitones / SEMITONES_PER_OCTAVE)


def _gain(shape: SoundShape) -> float:
    """Linear peak amplitude of the sound."""
    return DB_BASE ** (shape.gain_db / DB_PER_AMPLITUDE_DECADE)


def _draw(rng: random.Random, bounds: tuple[float, float]) -> float:
    return rng.uniform(bounds[0], bounds[1])


def _semitones(st: float) -> float:
    """Frequency ratio of ``st`` semitones."""
    return 2.0 ** (st / SEMITONES_PER_OCTAVE)


def _next_degree(rng: random.Random, degree: int) -> int:
    """One move of the melody: mostly a step of one degree, sometimes a repeat or a leap, and
    never out of the singing range."""
    low, high = HUM_DEGREE_RANGE
    roll = rng.random()
    if roll < HUM_REPEAT_PROB:
        return degree
    size = rng.choice(HUM_LEAP_DEGREES) if roll > 1.0 - HUM_LEAP_PROB else 1
    up = rng.random() < 0.5
    if degree + size > high:
        up = False
    elif degree - size < low:
        up = True
    return degree + size if up else degree - size


def _nearest_tonic(degree: int) -> int:
    return round(degree / DEGREES_PER_OCTAVE) * DEGREES_PER_OCTAVE


def plan_hum(shape: SoundShape, rng: random.Random) -> HumPlan:
    """Draw the melody of one hum: tempo, base pitch and length from the shape, rhythm cells
    until the length is nearly filled, a closing held tonic. The beat is then stretched a
    little so the notes fill the drawn length exactly."""
    tempo_bpm = _draw(rng, shape.tempo_bpm)
    base_hz = _draw(rng, shape.pitch_hz)
    length_s = _draw(rng, shape.length_s)
    want_beats = max(HUM_FINAL_BEATS, length_s * tempo_bpm / SECONDS_PER_MINUTE)
    lengths: list[float] = []
    total = HUM_FINAL_BEATS
    while True:
        cell = rng.choice(HUM_CELLS)
        if lengths and total + sum(cell) - want_beats > want_beats - total:
            break
        lengths.extend(cell)
        total += sum(cell)
    lengths.append(HUM_FINAL_BEATS)
    beat_s = length_s / total
    degree = rng.choice(HUM_START_DEGREES)
    degrees = [degree]
    for _ in range(len(lengths) - 2):
        degree = _next_degree(rng, degree)
        degrees.append(degree)
    degrees.append(_nearest_tonic(degrees[-1]))
    notes, start = [], 0.0
    for deg, beats in zip(degrees, lengths, strict=True):
        dur = beats * beat_s
        notes.append(HumNote(deg, degree_freq_hz(base_hz, deg), start, dur))
        start += dur
    return HumPlan(base_hz, length_s, _gain(shape), _draw(rng, VIBRATO_RATE_HZ), tuple(notes))


def plan_blirp(shape: SoundShape, rng: random.Random) -> BlirpPlan:
    """Draw 2 to 8 machine sounds with random gaps that together fill the drawn length. The
    shape's tempo caps how long one sound lasts; its pitch is the base of every event."""
    tempo_bpm = _draw(rng, shape.tempo_bpm)
    base_hz = _draw(rng, shape.pitch_hz)
    length_s = _draw(rng, shape.length_s)
    most = min(BLIRP_COUNT[1], max(BLIRP_COUNT[0], int(length_s / BLIRP_MIN_SLOT_S)))
    fewest = min(most, max(BLIRP_COUNT[0], math.ceil(length_s / BLIRP_MAX_SLOT_S)))
    count = rng.randint(fewest, most)
    cap_s = max(BLIRP_MIN_SOUND_S, SECONDS_PER_MINUTE / tempo_bpm * BLIRP_SOUND_BEATS)
    weights = [_draw(rng, BLIRP_DUR_JITTER) for _ in range(count)]
    fill = length_s * BLIRP_DUR_SHARE / sum(weights)
    durs = [max(BLIRP_MIN_SOUND_S, min(cap_s, w * fill)) for w in weights]
    gap_weights = [_draw(rng, BLIRP_GAP_JITTER) for _ in range(count - 1)]
    gap_unit = max(0.0, length_s - sum(durs)) / sum(gap_weights)
    events, start = [], 0.0
    for i, dur in enumerate(durs):
        f0 = base_hz * _semitones(_draw(rng, BLIRP_PITCH_ST))
        sweep = _semitones(_draw(rng, CHIRP_SWEEP_ST))
        kind = rng.choice(BLIRP_KINDS)
        f1, mod, depth = f0, 0.0, 0.0
        if kind == "chirp_up":
            f1 = f0 * sweep
        elif kind == "chirp_down":
            f0, f1 = f0 * sweep, f0
        elif kind == "fm":
            mod, depth = f0 * rng.choice(FM_RATIOS), _draw(rng, FM_INDEX)
        elif kind == "trill":
            f1, mod = f0 * _semitones(_draw(rng, TRILL_INTERVAL_ST)), _draw(rng, TRILL_RATE_HZ)
        else:
            mod, depth = _draw(rng, WARBLE_RATE_HZ), _draw(rng, WARBLE_DEPTH_ST)
        events.append(BlirpEvent(kind, start, dur, f0, f1, mod, depth))
        start += dur + (gap_weights[i] * gap_unit if i < count - 1 else 0.0)
    return BlirpPlan(start, _gain(shape), tuple(events))


def _fade(n: int, fade_n: int) -> np.ndarray:
    """Gain curve that is exactly 0 at both ends and 1 between the raised-cosine fades."""
    out = np.ones(n)
    fade_n = min(fade_n, n // 2)
    if fade_n > 1:
        ramp = np.sin(np.linspace(0.0, np.pi / 2, fade_n)) ** 2
        out[:fade_n] = ramp
        out[n - fade_n :] = ramp[::-1]
    out[0] = out[-1] = 0.0
    return out


def _finish(wave: np.ndarray, gain: float, rate_hz: int) -> np.ndarray:
    """Fade the ends to 0 and scale so the peak is the gain."""
    wave = wave * _fade(wave.size, round(FADE_S * rate_hz))
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave = wave * (gain * PEAK_MARGIN / peak)
    return wave.astype(np.float32)


def _phase(freq_hz: np.ndarray, rate_hz: int) -> np.ndarray:
    """Phase of an oscillator whose frequency follows ``freq_hz`` (no clicks at changes)."""
    return 2.0 * np.pi * np.cumsum(np.minimum(freq_hz, MAX_FREQ_SHARE * rate_hz)) / rate_hz


def render_hum(plan: HumPlan, rate_hz: int) -> np.ndarray:
    """Sing ``plan``: one phase-continuous voice (sine plus two weak harmonics, light vibrato)
    that glides between notes, with an attack/decay/release envelope per note."""
    n = max(1, round(plan.length_s * rate_hz))
    t = np.arange(n) / rate_hz
    xs: list[float] = []
    st: list[float] = []
    env_x: list[float] = []
    env_y: list[float] = []
    for i, note in enumerate(plan.notes):
        end = note.start_s + note.dur_s
        glide = min(GLIDE_S, GLIDE_MAX_SHARE * note.dur_s)
        pitch = math.log2(note.freq_hz) * SEMITONES_PER_OCTAVE
        xs += [note.start_s, end - glide]
        st += [pitch, pitch]
        edge_in = 0.0 if i == 0 else NOTE_DIP
        edge_out = 0.0 if i == len(plan.notes) - 1 else NOTE_DIP
        env_x += [
            note.start_s,
            note.start_s + NOTE_ATTACK_S,
            note.start_s + NOTE_ATTACK_S + NOTE_DECAY_S,
            end - NOTE_RELEASE_S,
            end,
        ]
        env_y += [edge_in, 1.0, NOTE_SUSTAIN, NOTE_SUSTAIN, edge_out]
    pitch_st = np.interp(t, xs, st)
    onset = np.minimum(1.0, t / VIBRATO_ONSET_S)
    pitch_st = pitch_st + VIBRATO_DEPTH_ST * onset * np.sin(2.0 * np.pi * plan.vibrato_hz * t)
    phase = _phase(2.0 ** (pitch_st / SEMITONES_PER_OCTAVE), rate_hz)
    voice = sum(g * np.sin((k + 1) * phase) for k, g in enumerate(HARMONIC_GAINS))
    return _finish(voice * np.interp(t, env_x, env_y), plan.gain, rate_hz)


def _blirp_wave(ev: BlirpEvent, rate_hz: int) -> np.ndarray:
    """One machine sound, with a fast fade at both ends."""
    n = max(2, round(ev.dur_s * rate_hz))
    u = np.arange(n) / n  # 0..1 across the sound
    t = u * ev.dur_s
    if ev.kind in ("chirp_up", "chirp_down"):
        wave = np.sin(_phase(ev.f0_hz * (ev.f1_hz / ev.f0_hz) ** u, rate_hz))
    elif ev.kind == "fm":
        carrier = _phase(np.full(n, ev.f0_hz), rate_hz)
        wave = np.sin(carrier + ev.depth * np.sin(2.0 * np.pi * ev.mod_hz * t))
    elif ev.kind == "trill":
        high = (np.sin(2.0 * np.pi * ev.mod_hz * t) > 0.0).astype(float)
        smooth_n = max(1, round(TRILL_SMOOTH_S * rate_hz))
        high = np.convolve(high, np.ones(smooth_n) / smooth_n, mode="same")
        wave = np.sin(_phase(ev.f0_hz * (ev.f1_hz / ev.f0_hz) ** high, rate_hz))
    else:
        bend = ev.depth * np.sin(2.0 * np.pi * ev.mod_hz * t)
        wave = np.sin(_phase(ev.f0_hz * _semitones(bend), rate_hz))
    return wave * _fade(n, round(BLIRP_EDGE_S * rate_hz))


def render_blirp(plan: BlirpPlan, rate_hz: int) -> np.ndarray:
    """Lay the machine sounds of ``plan`` on a silent timeline."""
    out = np.zeros(max(1, round(plan.length_s * rate_hz)))
    for ev in plan.events:
        wave = _blirp_wave(ev, rate_hz)
        at = round(ev.start_s * rate_hz)
        wave = wave[: out.size - at]
        out[at : at + wave.size] += wave
    return _finish(out, plan.gain, rate_hz)


def synthesize(sound: SoundKind, shape: SoundShape, rng: random.Random, rate_hz: int) -> np.ndarray:
    """One new sound of kind ``sound`` ("hum" or "blirp"): float32, mono, ``rate_hz``, starting
    and ending at exactly 0, peak at the shape's ``gain_db``. Everything that varies is drawn
    from ``rng`` inside the shape's ranges, so the same seed repeats a sound and no two seeds
    give the same one."""
    if sound == "hum":
        return render_hum(plan_hum(shape, rng), rate_hz)
    if sound == "blirp":
        return render_blirp(plan_blirp(shape, rng), rate_hz)
    raise ValueError(f"unknown sound kind {sound!r}")
