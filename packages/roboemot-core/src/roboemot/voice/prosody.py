"""Compute the final voice delivery and map it onto TTS backends (SPEC 8).

1. start from ``voice.base``;
2. apply the register and the circadian state (multipliers for rate, offsets for pitch and
   volume);
3. add the mood deltas scaled by affect intensity times ``expressivity``;
4. clamp to ``voice.limits``, then render per backend.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..config.schema import AffectConfig, RegisterParams, StateEffects, VoiceConfig

TAG = re.compile(r"\[mood=([a-z_]+)\]\s*")
DB_PER_DECADE = 20.0
DECADE = 10.0


@dataclass
class Delivery:
    rate: float
    pitch_st: float
    volume_db: float
    variability: float
    pause_scale: float
    instructions: str = ""
    sounds: list[str] = field(default_factory=list)
    mood: str | None = None


def _clamp(x: float, rng: tuple[float, float]) -> float:
    return max(rng[0], min(rng[1], x))


def compose_delivery(
    voice: VoiceConfig,
    *,
    register: RegisterParams,
    circadian: StateEffects,
    mood: str | None,
    intensity: float,
) -> Delivery:
    b = voice.base
    rate = b.rate * register.speech_rate * circadian.speech_rate
    pitch = b.pitch_st + register.pitch_st
    volume = b.volume_db + circadian.volume_db
    variability, pause, instructions, sounds = b.variability, 1.0, "", []
    if voice.mood_source == "fixed":
        mood = voice.fixed_mood
    m = voice.moods.get(mood) if mood else None
    if m:
        k = max(0.0, min(1.0, intensity)) * voice.expressivity
        rate += m.rate * k
        pitch += m.pitch_st * k
        volume += m.volume_db * k
        variability += m.variability * k
        pause = 1.0 + (m.pause_scale - 1.0) * k
        instructions = m.instructions if k > 0 else ""
        sounds = list(m.sounds) if k > 0 else []
    lim = voice.limits
    return Delivery(
        _clamp(rate, lim.rate),
        _clamp(pitch, lim.pitch_st),
        _clamp(volume, lim.volume_db),
        _clamp(variability, lim.variability),
        pause,
        instructions,
        sounds,
        mood if m else None,
    )


def to_piper(d: Delivery, voice: VoiceConfig) -> dict[str, float]:
    """Piper SynthesisConfig fields (length_scale, noise_scale, noise_w_scale, volume) plus two
    post-processing hints: pitch shift by DSP (Piper has no pitch control) and pause scale."""
    ns, nw = voice.piper_noise_scale, voice.piper_noise_w_scale
    return {
        "length_scale": 1.0 / d.rate,
        "noise_scale": ns[0] + (ns[1] - ns[0]) * d.variability,
        "noise_w_scale": nw[0] + (nw[1] - nw[0]) * d.variability,
        "volume": math.pow(DECADE, d.volume_db / DB_PER_DECADE),
        "pitch_shift_st": d.pitch_st if voice.pitch_shift_backend == "dsp" else 0.0,
        "sentence_silence_scale": d.pause_scale,
    }


def to_openai(d: Delivery) -> dict[str, str]:
    """``/v1/audio/speech`` extras for instructable models: delivery as natural language."""
    parts = [d.instructions] if d.instructions else []
    if d.rate < 1.0:
        parts.append("speak a little slower than usual")
    elif d.rate > 1.0:
        parts.append("speak a little faster than usual")
    if d.volume_db < 0:
        parts.append("keep the voice soft")
    return {"instructions": ", ".join(parts)} if parts else {}


def parse_mood_tags(text: str) -> list[tuple[str | None, str]]:
    """Split ``"[mood=joy] Che bello! [mood=curiosity] Cos'è?"`` into (mood, sentence) pairs;
    tags are stripped before synthesis (``mood_source: llm_tagged``)."""
    out: list[tuple[str | None, str]] = []
    pos, mood = 0, None
    for m in TAG.finditer(text):
        if m.start() > pos and text[pos : m.start()].strip():
            out.append((mood, text[pos : m.start()].strip()))
        mood, pos = m.group(1), m.end()
    if text[pos:].strip():
        out.append((mood, text[pos:].strip()))
    return out


def resolve_tag(
    tag: str | None,
    *,
    affect_label: str,
    valence: float,
    arousal: float,
    affect: AffectConfig,
    voice: VoiceConfig,
) -> str:
    """Keep the LLM's tag only if its prototype is within ``llm_tag_max_deviation`` of the
    affect state, so voice and body never contradict each other."""
    if not tag or tag not in affect.labels:
        return affect_label
    p = affect.labels[tag]
    if math.hypot(p.valence - valence, p.arousal - arousal) > voice.llm_tag_max_deviation:
        return affect_label
    return tag
