"""Typed parameter schema for every behaviour and expression setting (SPEC 11A).

Every field carries a default, a unit and a range. Field metadata (``json_schema_extra``):

- ``unit``: physical unit, shown in the generated reference and web UI.
- ``restart``: True when a change needs a process restart instead of hot reload.
- ``conservative``: for safety parameters, which direction is safer at runtime:
  ``"lower"`` (smaller values are safer), ``"higher"``, ``"false"`` / ``"true"`` for booleans
  that are safer off / on. Changes in the other direction need admin approval (stack.py).
- ``opt_in``: privacy options; turning them on needs the admin confirmation.
- ``admin_only``: prompt text, where no direction is safe; any change needs the admin
  confirmation.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROUTE_NAME = re.compile(r"[A-Za-z0-9_-]+")
ROUTE_NAME_MAX_CHARS = 64
ROUTE_SCHEMES = ("http", "https")  # the OpenAI-compatible client speaks only these (SPEC 8)


def param(
    default: Any,
    description: str,
    *,
    unit: str | None = None,
    ge: float | None = None,
    le: float | None = None,
    restart: bool = False,
    conservative: Literal["lower", "higher", "false", "true"] | None = None,
    opt_in: bool = False,
    admin_only: bool = False,
) -> Any:
    extra: dict[str, Any] = {"restart": restart}
    if unit:
        extra["unit"] = unit
    if conservative:
        extra["conservative"] = conservative
    if opt_in:
        extra["opt_in"] = True
    if admin_only:
        extra["admin_only"] = True
    kwargs: dict[str, Any] = {"description": description, "json_schema_extra": extra}
    if ge is not None:
        kwargs["ge"] = ge
    if le is not None:
        kwargs["le"] = le
    if isinstance(default, (list, dict)):
        return Field(default_factory=lambda d=default: _copy(d), **kwargs)
    return Field(default, **kwargs)


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value


class Strict(BaseModel):
    """Base model: unknown keys are rejected (SPEC 11A, Rules: Schema)."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, validate_default=True, frozen=True
    )


# --------------------------------------------------------------------------- affect (SPEC 7)


class LabelRegion(Strict):
    valence: float = param(0.0, "Prototype valence of the label", ge=-1, le=1)
    arousal: float = param(0.0, "Prototype arousal of the label", ge=-1, le=1)


class AffectConfig(Strict):
    update_hz: float = param(20.0, "Affect engine update rate", unit="Hz", ge=1, le=200)
    half_life_s: float = param(8.0, "Decay half-life toward the baseline", unit="s", ge=0.1, le=600)
    llm_impulse_max: float = param(
        0.3, "Max impulse magnitude the LLM may apply per turn", ge=0, le=1
    )
    event_label_hold_s: float = param(
        3.0, "How long an appraisal label overrides the region label", unit="s", ge=0, le=60
    )
    labels: dict[str, LabelRegion] = param(
        {
            "calm": {"valence": 0.2, "arousal": -0.2},
            "joy": {"valence": 0.7, "arousal": 0.5},
            "curiosity": {"valence": 0.3, "arousal": 0.4},
            "surprise": {"valence": 0.0, "arousal": 0.8},
            "sadness": {"valence": -0.6, "arousal": -0.4},
            "fear": {"valence": -0.7, "arousal": 0.7},
            "annoyance": {"valence": -0.5, "arousal": 0.3},
            "sleepy": {"valence": 0.0, "arousal": -0.8},
        },
        "Label prototypes in the valence/arousal plane; the nearest one names the state",
    )


APPRAISAL_EVENTS = frozenset(
    {
        "person.known",
        "person.unknown_appears",
        "animal.appears",
        "idle.long",
        "sound.loud",
        "light.on",
        "light.off",
        "circadian.woken_at_night",
        "conversation.unavailable",
        "speech.heard",
        "speech.keyword.stop",
        "speech.keyword.yes",
        "speech.keyword.no",
        "speech.keyword.greeting",
        "speech.name_called",
        "speech.loud",
        "speech.fast",
    }
)
APPRAISAL_EVENT_PREFIXES = ("link.",)  # connectivity: link.<name>.lost / .restored (SPEC 12A)


def rule_identity(event: str, when: Any) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """What makes two appraisal rules the same rule: the event and its normalised ``when``.

    Numbers and booleans compare equal in a ``when`` match (1 == 1.0 == True), so they are
    normalised to float here. Used by the keyed-list merge and by the engine's cooldowns.
    """
    items = (
        (k, float(v) if isinstance(v, (bool, int, float)) else v)
        for k, v in sorted((when or {}).items())
    )
    return event, tuple(items)


class AppraisalRule(Strict):
    event: str = param("", "Bus event name that triggers the rule")
    valence: float = param(0.0, "Valence impulse", ge=-1, le=1)
    arousal: float = param(0.0, "Arousal impulse", ge=-1, le=1)
    label: str | None = param(None, "Label forced for event_label_hold_s, if any")
    when: dict[str, str | bool | float] = param(
        {}, "Payload fields that must equal these values for the rule to fire (empty: always)"
    )
    cooldown_s: float = param(5.0, "Minimum time between two firings", unit="s", ge=0, le=3600)


class AppraisalConfig(Strict):
    idle_after_s: float = param(
        600.0,
        "No face, attention target or recognised person for this long emits idle.long",
        unit="s",
        ge=1,
        le=86400,
    )
    unknown_face_after_s: float = param(
        3.0,
        "A new face track still without an owner after this long emits person.unknown_appears "
        "(the identity decision window, SPEC 5)",
        unit="s",
        ge=0,
        le=60,
    )
    track_lost_s: float = param(
        2.0,
        "A pending unknown face not seen for this long is dropped without an event",
        unit="s",
        ge=0.1,
        le=60,
    )
    track_lost_periods: float = param(
        3.0,
        "A pending unknown face also survives this many face-detection periods at the rate the "
        "governor allows now: the lifetime is the larger of this and track_lost_s, so the "
        "resting vision rate does not drop it between two frames",
        ge=1,
        le=100,
    )
    extra_events: list[str] = param(
        [],
        "Event names emitted by plugins, accepted in rules besides the library's events",
    )
    rules: list[AppraisalRule] = param(
        [
            # Starting points (SPEC 7): an enrolled person gives +0.5/+0.2 and calm, a stranger
            # raises curiosity with almost no valence change.
            {
                "event": "person.known",
                "valence": 0.5,
                "arousal": 0.2,
                "label": "calm",
                "cooldown_s": 60,
            },
            {
                "event": "person.unknown_appears",
                "valence": 0.05,
                "arousal": 0.3,
                "label": "curiosity",
                "cooldown_s": 30,
            },
            {
                "event": "animal.appears",
                "valence": 0.2,
                "arousal": 0.3,
                "label": "curiosity",
                "cooldown_s": 30,
            },
            {"event": "sound.loud", "arousal": 0.6, "label": "surprise", "cooldown_s": 5},
            {"event": "idle.long", "arousal": -0.6, "label": "sleepy", "cooldown_s": 600},
            {"event": "light.on", "arousal": 0.3, "label": "surprise", "cooldown_s": 30},
            # Lights switched off: a mild drift toward sleepy (SPEC 7A, Light sensing).
            {"event": "light.off", "arousal": -0.2, "label": "sleepy", "cooldown_s": 60},
            # Woken in the quiet hours: brief, mild and never a refusal (SPEC 7A, Waking).
            {
                "event": "circadian.woken_at_night",
                "valence": -0.3,
                "arousal": 0.1,
                "cooldown_s": 300,
            },
            # Connectivity (SPEC 12A): subdued while offline, perks up when back.
            {"event": "link.network.lost", "valence": -0.2, "arousal": -0.3, "cooldown_s": 300},
            {
                "event": "link.network.restored",
                "valence": 0.3,
                "arousal": 0.4,
                "label": "joy",
                "cooldown_s": 60,
            },
            {"event": "link.robot.restored", "arousal": 0.3, "label": "surprise", "cooldown_s": 30},
            {
                "event": "conversation.unavailable",
                "valence": -0.1,
                "arousal": 0.1,
                "label": "curiosity",
                "cooldown_s": 10,
            },
            # Speech fast path (SPEC 8 stage 3): keywords spotted on ASR partials and prosody of
            # the finished utterance. Starting points: a greeting or the robot's name is welcome,
            # "stop" and "no" are a mild jolt, a raised or rushed voice raises arousal.
            {"event": "speech.heard", "arousal": 0.05, "cooldown_s": 20},
            {
                "event": "speech.keyword.greeting",
                "valence": 0.3,
                "arousal": 0.3,
                "label": "joy",
                "cooldown_s": 30,
            },
            {
                "event": "speech.name_called",
                "valence": 0.1,
                "arousal": 0.4,
                "label": "curiosity",
                "cooldown_s": 5,
            },
            {
                "event": "speech.keyword.stop",
                "valence": -0.1,
                "arousal": 0.3,
                "label": "surprise",
                "cooldown_s": 3,
            },
            {"event": "speech.keyword.no", "valence": -0.2, "arousal": 0.1, "cooldown_s": 3},
            {"event": "speech.loud", "arousal": 0.15, "cooldown_s": 10},
            {"event": "speech.fast", "arousal": 0.2, "cooldown_s": 10},
        ],
        "Event to impulse rules (SPEC 7, Affect state)",
    )

    @model_validator(mode="after")
    def _rules_are_known_and_unique(self) -> AppraisalConfig:
        known = APPRAISAL_EVENTS | set(self.extra_events)
        seen: set[Any] = set()
        for rule in self.rules:
            if rule.event not in known and not rule.event.startswith(APPRAISAL_EVENT_PREFIXES):
                raise ValueError(
                    f"appraisal rule for unknown event {rule.event!r}; the library emits "
                    f"{sorted(APPRAISAL_EVENTS)} and link.*; plugin events go in "
                    "appraisal.extra_events"
                )
            ident = rule_identity(rule.event, rule.when)
            if ident in seen:
                raise ValueError(f"two appraisal rules for {rule.event!r} with the same `when`")
            seen.add(ident)
        return self


# --------------------------------------------------------------------------- register (SPEC 7)


class RegisterParams(Strict):
    speech_rate: float = param(1.0, "Speech rate multiplier", ge=0.5, le=1.5)
    pitch_st: float = param(0.0, "Pitch shift", unit="semitones", ge=-6, le=6)
    vocabulary: Literal["simple", "normal", "formal"] = param("normal", "LLM vocabulary level")
    persona_fragment: str = param("Friendly, direct.", "Text added to the LLM system prompt")
    motion_amplitude: float = param(1.0, "Motion amplitude multiplier", ge=0, le=2)
    approach_allowed: bool = param(
        True, "Whether the robot may move toward the person", conservative="false"
    )
    reaction_intensity: Literal["low", "medium", "high"] = param("medium", "Reaction intensity")
    max_reply_sentences: int | None = param(
        None,
        "Reply length cap in sentences for this register; unset uses "
        "conversation.max_reply_sentences",
        ge=1,
        le=20,
        conservative="lower",
    )


class RegisterConfig(RegisterParams):
    """The one register the robot uses with everyone (SPEC 7; no age groups, owner decision
    2026-10-23), plus the settings about registers."""

    persona_fragment_max_chars: int = param(
        300, "Longest accepted persona_fragment", ge=20, le=2000
    )

    @property
    def params(self) -> RegisterParams:
        """The register parameters alone, as a person's overrides and the persona shape them."""
        return RegisterParams.model_validate(
            self.model_dump(include=set(RegisterParams.model_fields))
        )

    @model_validator(mode="after")
    def _fragment_fits(self) -> RegisterConfig:
        n = len(self.persona_fragment)
        if n > self.persona_fragment_max_chars:
            raise ValueError(
                f"register.persona_fragment has {n} characters, "
                f"max {self.persona_fragment_max_chars}"
            )
        return self


# --------------------------------------------------------------------------- expression (SPEC 7)

EXPRESSION_MAX_CYCLE_S = 30.0  # one cycle; with repeat <= 20 an expression ends within minutes


class IntensityFactors(Strict):
    low: float = param(0.6, "Amplitude factor of reaction_intensity low", ge=0, le=2)
    medium: float = param(1.0, "Amplitude factor of reaction_intensity medium", ge=0, le=2)
    high: float = param(1.3, "Amplitude factor of reaction_intensity high", ge=0, le=2)


class ExpressionConfig(Strict):
    intensity_factor: IntensityFactors = param(
        {}, "Reaction intensity (register) as an amplitude factor: starting points to tune"
    )
    intensity_floor: float = param(
        0.5,
        "Share of the amplitude kept at affect intensity 0; the amplitude grows linearly to "
        "the full value at intensity 1",
        ge=0,
        le=1,
    )
    max_amplitude: float = param(
        2.0,
        "Ceiling of the product register x reaction x persona x circadian x affect factors",
        ge=0,
        le=3,
    )
    min_interval_s: float = param(
        1.0, "Minimum time between the starts of two expressions", unit="s", ge=0, le=60
    )
    retrigger_s: float = param(
        30.0,
        "A label that stays held replays its expression at most this often",
        unit="s",
        ge=1,
        le=3600,
    )
    retrigger_min_intensity: float = param(
        0.15, "A held label replays only while the affect intensity is at least this", ge=0, le=1
    )
    command_ttl_s: float = param(
        0.25,
        "Lifetime of each expression command in the arbiter: the offset drops to zero by itself "
        "if the app stops feeding it",
        unit="s",
        ge=0.05,
        le=2,
    )
    register_hold_s: float = param(
        2.0,
        "A face track not seen for this long leaves the active register set",
        unit="s",
        ge=0.1,
        le=60,
    )
    register_hold_periods: float = param(
        3.0,
        "A face track also stays in the active register set for this many face-detection "
        "periods at the rate the governor allows now (the larger of the two applies)",
        ge=1,
        le=100,
    )


class ExpressionEntry(Strict):
    """One affect label rendered as keyframes per channel (SPEC 7, Expression map)."""

    channels: dict[str, list[tuple[float, float]]] = param(
        {},
        "Channel name to [t_s, offset_deg] keyframes of one cycle; piecewise-linear in time. "
        "Every channel starts at t=0 and ends at the same time with offset 0",
    )
    repeat: int = param(1, "How many times the cycle plays back to back (a wiggle)", ge=1, le=20)
    curve: Literal["linear", "minjerk", "ease_in_out"] = param(
        "linear", "Easing between two keyframes"
    )
    additive: Literal[True] = param(
        True, "Layered on top of other motion; only additive expressions exist in M2"
    )

    @field_validator("channels")
    @classmethod
    def _channels(cls, channels: dict[str, list[tuple[float, float]]]) -> dict:
        if not channels:
            raise ValueError("an expression needs at least one channel")
        ends = set()
        for name, frames in channels.items():
            if len(frames) < 2:
                raise ValueError(f"channel {name}: needs at least two keyframes")
            times = [t for t, _ in frames]
            if times[-1] > EXPRESSION_MAX_CYCLE_S:
                raise ValueError(f"channel {name}: longer than {EXPRESSION_MAX_CYCLE_S} s")
            if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
                raise ValueError(f"channel {name}: times must start at 0 and increase")
            if frames[0][1] != 0 or frames[-1][1] != 0:
                raise ValueError(f"channel {name}: the first and last offsets must be 0")
            ends.add(times[-1])
        if len(ends) != 1:
            raise ValueError("all channels must end at the same time")
        return channels


# --------------------------------------------------------------------------- circadian (SPEC 7A)


class StateEffects(Strict):
    arousal_baseline: float = param(0.0, "Affect arousal baseline in this state", ge=-1, le=1)
    idle_motion_scale: float = param(1.0, "Idle motion amplitude multiplier", ge=0, le=2)
    speech_rate: float = param(1.0, "Speech rate multiplier", ge=0.5, le=1.5)
    volume_db: float = param(0.0, "Volume offset", unit="dB", ge=-24, le=6)
    vision_rate_scale: float = param(1.0, "Vision rate multiplier", ge=0, le=1)
    idle_period_scale: float = param(
        1.0, "Idle breathing period multiplier (above 1 is slower)", ge=0.5, le=4
    )
    expression_scale: float = param(
        1.0,
        "Amplitude multiplier of affect expressions in this state (the circadian factor)",
        ge=0,
        le=2,
    )


class CircadianMotion(Strict):
    """Robot-specific circadian primitives (SPEC 7A, Effects by state); the manifest ships them.

    Offsets in degrees in the file, radians in code. ``yawn`` and ``stretch`` are additive like
    an expression (same shape and rules); ``rest_pose_deg`` is the absolute rest posture.
    """

    yawn: ExpressionEntry | None = param(None, "Drowsy yawn primitive; none: no yawn")
    stretch: ExpressionEntry | None = param(None, "Waking stretch primitive; none: no stretch")
    rest_pose_deg: dict[str, float] = param(
        {}, "Absolute rest posture by channel in degrees (the robot's own resting pose)"
    )


class CircadianConfig(Strict):
    enabled: bool = param(True, "Enable the simulated circadian rhythm")
    period_h: float = param(24.0, "Free-running period of process C", unit="h", ge=20, le=28)
    peak_hour: float = param(
        15.0, "Hour of the day at which alertness (process C) peaks", unit="h", ge=0, le=24
    )
    c_amplitude: float = param(0.5, "Amplitude of process C", ge=0, le=1)
    s_rise_tau_h: float = param(18.0, "Sleep pressure rise time constant", unit="h", ge=1, le=72)
    s_decay_tau_h: float = param(4.0, "Sleep pressure decay time constant", unit="h", ge=0.5, le=24)
    max_shift_h_per_day: float = param(
        1.0, "Max phase shift caused by light per day", unit="h", ge=0, le=6
    )
    drowsy_threshold: float = param(0.55, "Sleepiness above which the robot is drowsy", ge=0, le=1)
    resting_threshold: float = param(0.75, "Sleepiness above which the robot rests", ge=0, le=1)
    hysteresis: float = param(0.05, "Hysteresis on state thresholds", ge=0, le=0.3)
    light_source: Literal["camera", "clock_only"] = param("camera", "Entrainment source")
    idle_sample_interval_s: float = param(
        30.0, "Light sampling interval when vision is idle", unit="s", ge=1, le=600
    )
    location: dict[str, float] | None = param(None, "Optional {lat, lon} for clock-only mode")
    quiet_hours: tuple[str, str] = param(("22:30", "07:00"), "Window in which resting is forced")
    presence_keeps_awake: bool = param(True, "People present and active prevent resting")
    initial_sleep_pressure: float = param(0.3, "Process S value at start-up", ge=0, le=1)
    wake_hold_s: float = param(
        300.0, "How long the wake word keeps a resting robot awake", unit="s", ge=10, le=3600
    )
    update_period_s: float = param(
        10.0,
        "Seconds between two model updates (process S, entrainment, state)",
        unit="s",
        ge=0.1,
        le=600,
    )
    publish_period_s: float = param(
        60.0, "Seconds between two periodic circadian.state messages", unit="s", ge=1, le=3600
    )
    presence_window_s: float = param(
        60.0,
        "People seen, recognised or attended to within this time count as present",
        unit="s",
        ge=1,
        le=3600,
    )
    presence_periods: float = param(
        3.0,
        "Presence also lasts this many face-detection periods at the governor's current rate",
        ge=1,
        le=100,
    )
    idle_poll_s: float = param(
        1.0,
        "How often the idle light reader checks whether a sample is due",
        unit="s",
        ge=0.05,
        le=60,
    )
    command_ttl_s: float = param(
        0.5,
        "Lifetime of the rest posture and the yawn or stretch commands",
        unit="s",
        ge=0.1,
        le=5,
    )
    stretch_cooldown_s: float = param(
        600.0,
        "Least time between two waking stretches, so people coming and going at night do not "
        "replay it",
        unit="s",
        ge=0,
        le=86400,
    )
    yawn_interval_s: float = param(
        120.0, "Mean time between two yawns while drowsy", unit="s", ge=5, le=3600
    )
    yawn_jitter: float = param(
        0.5, "Yawn interval jitter as a share of the interval (0: regular)", ge=0, le=0.9
    )
    hold_max_min: float = param(720.0, "Longest hold an app may request", unit="min", ge=1, le=1440)
    states: dict[str, StateEffects] = param(
        {
            "awake": {"arousal_baseline": 0.2},
            "drowsy": {
                "arousal_baseline": -0.3,
                "idle_motion_scale": 0.6,
                "idle_period_scale": 1.5,
                "expression_scale": 0.8,
                "speech_rate": 0.9,
                "volume_db": -3.0,
                "vision_rate_scale": 0.5,
            },
            "resting": {
                "arousal_baseline": -0.7,
                "idle_motion_scale": 0.0,
                "expression_scale": 0.6,
                "speech_rate": 0.85,
                "volume_db": -6.0,
                "vision_rate_scale": 0.05,
            },
        },
        "Effects applied in each circadian state (SPEC 7A, Effects by state)",
    )

    @model_validator(mode="after")
    def _check(self) -> CircadianConfig:
        missing = {"awake", "drowsy", "resting"} - set(self.states)
        if missing:
            raise ValueError(f"circadian.states missing {sorted(missing)}")
        if self.resting_threshold <= self.drowsy_threshold:
            raise ValueError("circadian.resting_threshold must exceed drowsy_threshold")
        for t in self.quiet_hours:
            h, _, m = t.partition(":")
            if not (h.isdigit() and m.isdigit() and int(h) < 24 and int(m) < 60):
                raise ValueError(f"circadian.quiet_hours: bad time {t!r}, use HH:MM")
        return self


class LightConfig(Strict):
    dark_luma: float = param(0.12, "Normalised luminance below which the room is dark", ge=0, le=1)
    bright_luma: float = param(
        0.35, "Normalised luminance above which the room is bright", ge=0, le=1
    )
    hysteresis: float = param(0.03, "Hysteresis on light levels", ge=0, le=0.2)
    noise_dark_threshold: float = param(
        0.04,
        "Image noise (std of a high-pass) above which a dim image is classed dark when no "
        "exposure metadata exists",
        ge=0,
        le=1,
    )
    calibration_k: float | None = param(None, "Lux calibration constant with exposure metadata")
    sample_interval_s: float = param(
        10.0,
        "At most one vision frame is measured for light per this time",
        unit="s",
        ge=0.5,
        le=600,
    )
    history_h: float = param(
        24.0, "Hours of one-per-minute light values kept in RAM", unit="h", ge=1, le=72
    )
    confirm_s: float = param(
        60.0,
        "A new light level counts (and fires light.on or light.off) only after the samples have "
        "agreed on it for this long",
        unit="s",
        ge=0,
        le=600,
    )
    noise_smooth_s: float = param(
        120.0,
        "Time constant of the smoothing of the image noise before the dark-noise rule, so one "
        "noisy frame does not decide it",
        unit="s",
        ge=0,
        le=3600,
    )
    noise_hysteresis: float = param(
        0.01,
        "Hysteresis on noise_dark_threshold: a room leaves dark by noise only below the "
        "threshold minus this",
        ge=0,
        le=0.04,
    )
    stale_s: float = param(
        300.0,
        "A light reading older than this is ignored (the clock rules alone)",
        unit="s",
        ge=10,
        le=3600,
    )


# --------------------------------------------------------------------------- motion (SPEC 10)


class ArbiterConfig(Strict):
    loop_hz: float = param(100.0, "Arbiter rate", unit="Hz", ge=10, le=500, restart=True)
    crossfade_ms: float = param(
        150.0, "Cross-fade when the winning source changes", unit="ms", ge=0, le=2000
    )
    source_timeout_ms: float = param(
        250.0, "A streamed source is dropped after this silence", unit="ms", ge=10, le=5000
    )


class IdleConfig(Strict):
    breathing_period_s: float = param(4.0, "Breathing motion period", unit="s", ge=1, le=20)
    breathing_amplitude: float = param(
        0.02, "Breathing amplitude on head pitch", unit="rad", ge=0, le=0.2
    )
    gaze_shift_interval_s: tuple[float, float] = param(
        (3.0, 8.0), "Random interval between gaze shifts"
    )
    gaze_shift_amplitude: float = param(0.15, "Max gaze shift on head yaw", unit="rad", ge=0, le=1)


class ExternalControlConfig(Strict):
    """Divergence between the pose the loop commands and the pose the robot reports (SPEC 12A):
    the platform dropping our targets (a move of its own is running) or a hand on the head."""

    tolerance_rad: float = param(
        0.15,
        "An angular joint further than this outside the range of the poses commanded within "
        "lag_window_s counts as diverging (tracking lag and sag of our own motion stay under it). "
        "Behind the commands, on the side they came from, is lag however far, until they settle",
        unit="rad", ge=0.01, le=1.0,
    )  # fmt: skip
    tolerance_m: float = param(
        0.02,
        "The same for translations",
        unit="m", ge=0.002, le=0.1,
    )  # fmt: skip
    lag_window_s: float = param(
        0.4,
        "The measured pose is compared with the poses commanded this far back, so the "
        "servos' tracking lag is not mistaken for external control. Counted in loop ticks "
        "(lag_window_s x motion.arbiter.loop_hz commands), so late ticks do not shorten it",
        unit="s", ge=0.05, le=2.0,
    )  # fmt: skip
    confirm_s: float = param(
        0.2,
        "Divergence must hold on confirm_s x motion.arbiter.loop_hz new state samples in a row "
        "before the loop yields; a repeated sample does not count, a late tick starts again",
        unit="s", ge=0.02, le=2.0,
    )  # fmt: skip
    late_tick_periods: float = param(
        2.0,
        "A loop tick that comes more than this many periods after the previous one is late "
        "(an overloaded CPU): its sample does not count toward the confirmation (a stall cannot "
        "bridge it), but the count goes on, so a push is still caught under steady overruns",
        unit="periods", ge=1.0, le=10.0,
    )  # fmt: skip
    max_state_age_s: float = param(
        0.8,
        "The oldest a measured state sample may be (a platform's state queue, about 0.67 s on "
        "the reference robot, plus client buffering): a joint where the commands were up to "
        "this long before the lag window, or behind moving commands and moved within this time, "
        "counts as lag. A push against a moving head is caught within about lag_window_s + "
        "max_state_age_s + confirm_s; longer is more tolerant of old state, slower to catch it",
        unit="s", ge=0.0, le=5.0,
    )  # fmt: skip
    solo_groups: list[str] = param(
        [],
        "Manifest groups that yield on their own when only they diverge (a hand on a light "
        "part). Any other divergence yields every group: the platform's own moves (a "
        "go-to-sleep) may show first on any group, and the groups not yet diverging would "
        "otherwise keep streaming targets the platform accepts as its move ends. List a group "
        "only if the platform's moves never move it first",
    )
    recent_events: int = param(
        8,
        "How many recent external-control events the debug report keeps (joint, group, error, "
        "samples, largest tick gap, overruns)",
        ge=1, le=100,
    )  # fmt: skip
    still_band_rad: float = param(
        0.02,
        "While yielding, the pose counts as still while every angular joint stays within this "
        "band of the pose where the stillness count started (wider than encoder jitter)",
        unit="rad", ge=0.002, le=0.5,
    )  # fmt: skip
    still_band_m: float = param(
        0.003,
        "The same for translations",
        unit="m", ge=0.0005, le=0.05,
    )  # fmt: skip


class MotionConfig(Strict):
    arbiter: ArbiterConfig = param({}, "Arbiter parameters")
    idle: IdleConfig = param({}, "Idle motion parameters")
    app_tick_hz: float = param(
        20.0,
        "Rate of the app's behaviour tick: affect, idle, circadian, scan, expression and "
        "responder commands are recomputed and fed to the arbiter this often (the arbiter itself "
        "runs at arbiter.loop_hz)",
        unit="Hz", ge=1, le=100,
    )  # fmt: skip
    platform_move_grace_s: float = param(
        5.0,
        "After external control of the robot is detected (the platform runs a move of its own, "
        "such as the owner's go-to-sleep, or a hand moves the head), the loop keeps following the "
        "measured pose without writing until the pose has been still this long with the motors "
        "on, then resumes from it. Must exceed the platform's delay between the end of a sleep "
        "move and the motors being switched off (2.0 s on the reference robot, SPEC 4A) plus the "
        "motor status latency (a 1 s status period plus the time to build it, and the state "
        "queue's age), less the min-jerk tail in which the pose already reads as still, so the "
        "motors-off is seen before any target is streamed again (SPEC 12A)",
        unit="s", ge=0, le=30,
    )  # fmt: skip
    external: ExternalControlConfig = param({}, "Detection of external control of the robot")


# --------------------------------------------------------------------------- reflexes (SPEC 10)

REFLEX_TRIGGERS = ("loud_onset", "new_voice", "proximity")
REFLEX_PRIMITIVES = ("startle", "withdraw", "turn_toward")
# Which primitives a trigger may drive. A turn needs a bearing, so only a voice can drive it; a
# withdraw is the only answer to a close face (never towards it, SPEC 12).
REFLEX_TRIGGER_PRIMITIVES = {
    "loud_onset": ("startle", "withdraw"),
    "new_voice": ("turn_toward",),
    "proximity": ("withdraw",),
}
# Threshold range per trigger: onset ratio over the sound floor, silence before a new voice (s),
# face height as a fraction of the frame.
REFLEX_THRESHOLD_RANGE = {
    "loud_onset": (1.5, 100.0),
    "new_voice": (0.2, 600.0),
    "proximity": (0.05, 1.0),
}
REFLEX_DURATION_MS = (100.0, 800.0)  # SPEC 10: a reflex pre-empts lower sources for 100 to 800 ms


# The protective withdraw reflex is frozen at runtime (config/stack.py): its own fields, and the
# shared ``reflexes`` settings the engine reads for it. A new field must be added to one of these
# (or to the explicit "does not affect a withdraw" list in the tests).
WITHDRAW_RULE_FIELDS = (
    "enabled", "trigger", "primitive", "threshold", "amplitude_deg", "cooldown_s",
    "duration_ms",
)  # fmt: skip
WITHDRAW_SHARED_FIELDS = (
    "enabled", "max_amplitude_deg", "max_age_ms", "max_age_vision_ms", "max_future_ms",
    "command_ttl_ms", "attack_fraction", "release_fraction", "proximity_rearm_ratio",
    "motion_eps_deg",
)  # fmt: skip


class ReflexRule(Strict):
    name: str = param("", "Rule name (unique; shown in the debug report)")
    enabled: bool = param(
        True,
        "Whether the rule fires. A runtime override may not enable a rule the approved config "
        "disabled, nor disable a withdraw rule (protective)",
    )
    trigger: Literal["loud_onset", "new_voice", "proximity"] = param(
        "loud_onset", "Percept that fires the rule"
    )
    primitive: Literal["startle", "withdraw", "turn_toward"] = param(
        "startle",
        "Motion played: startle (head back, the robot's startle dofs lifted), withdraw (head back) "
        "or turn_toward (head yaw towards a voice)",
    )
    threshold: float = param(
        3.0,
        "Trigger level: loud_onset, RMS ratio over the running floor; new_voice, seconds of "
        "silence before a voice counts as new; proximity, face height as a fraction of the frame. "
        "Runtime overrides may only make a reflex safer: a higher threshold for loud_onset and "
        "new_voice (fires less), a lower one for proximity (the withdraw fires earlier)",
        unit="ratio, s or frame fraction (per trigger)",
        ge=0.05,
        le=600,
    )
    amplitude_deg: float = param(
        12.0,
        "Peak head deflection (turn_toward: the most it may turn from where it points)",
        unit="deg",
        ge=0,
        le=90,
        conservative="lower",
    )
    cooldown_s: float = param(
        3.0, "Minimum time between two firings", unit="s", ge=0, le=600, conservative="higher"
    )
    duration_ms: float = param(
        400.0,
        "How long the reflex pre-empts lower sources (SPEC 10: 100 to 800 ms)",
        unit="ms",
        ge=REFLEX_DURATION_MS[0],
        le=REFLEX_DURATION_MS[1],
        conservative="lower",
    )

    @model_validator(mode="after")
    def _consistent(self) -> ReflexRule:
        if ROUTE_NAME.fullmatch(self.name) is None:
            raise ValueError("a reflex rule needs a name of letters, digits, _ and -")
        if self.primitive not in REFLEX_TRIGGER_PRIMITIVES[self.trigger]:
            raise ValueError(
                f"reflex {self.name!r}: trigger {self.trigger} drives "
                f"{REFLEX_TRIGGER_PRIMITIVES[self.trigger]}, not {self.primitive}"
            )
        lo, hi = REFLEX_THRESHOLD_RANGE[self.trigger]
        if not lo <= self.threshold <= hi:
            raise ValueError(
                f"reflex {self.name!r}: threshold for {self.trigger} must be within {lo} and {hi}"
            )
        return self


class ReflexAudioConfig(Strict):
    enabled: bool = param(
        True,
        "Read the microphone level for loud-onset reflexes",
        restart=True,
        conservative="false",
    )
    poll_hz: float = param(50.0, "Audio chunk read rate", unit="Hz", ge=5, le=200, restart=True)
    floor_half_life_s: float = param(
        2.0, "How slowly the sound floor rises to a louder room (it falls at once); a shorter "
        "half-life makes a sustained noise stop counting as an onset sooner",
        unit="s", ge=0.1, le=60, conservative="lower",
    )  # fmt: skip
    min_rms: float = param(
        0.01, "Lowest floor: a quieter room is treated as this level, so a higher value makes "
        "the onsets fire less", unit="full scale", ge=0.0001, le=0.5, conservative="higher",
    )  # fmt: skip
    loud_event_ratio: float = param(
        4.0, "Level over the floor that publishes sound.loud for the affect engine", ge=1.5, le=100
    )
    loud_event_refractory_s: float = param(
        1.0, "Minimum time between two sound.loud events", unit="s", ge=0, le=60
    )
    level_block_s: float = param(
        1.0,
        "Report only: length of one microphone level block (per-channel and mono dBFS, "
        "channel correlation) in seconds of input audio",
        unit="s",
        ge=0.1,
        le=60,
        restart=True,
    )
    level_window: int = param(
        300, "Report only: level blocks kept for the report", ge=10, le=10000, restart=True
    )


class ReflexConfig(Strict):
    enabled: bool = param(
        True,
        "Run the reflex engine. Off, it stops at once; on after a start with it off, the "
        "loud-onset reflexes need a restart (the microphone is opened at start only for a "
        "consumer)",
        conservative="false",
    )
    max_amplitude_deg: float = param(
        45.0,
        "Ceiling on every rule's amplitude (the manifest limits still apply)",
        unit="deg", ge=0, le=90, conservative="lower",
    )  # fmt: skip
    max_age_ms: float = param(
        150.0,
        "A trigger older than this when the motion loop gets to it is dropped, never played late",
        unit="ms", ge=20, le=2000, conservative="lower",
    )  # fmt: skip
    max_age_vision_ms: float = param(
        250.0,
        "Same for a close-face trigger: the frame stamp is the capture time, so detection time "
        "counts (SPEC 13: vision trigger to reflex motion, 250 ms on cm4)",
        unit="ms", ge=20, le=2000, conservative="lower",
    )  # fmt: skip
    max_future_ms: float = param(
        50.0,
        "A trigger stamped later than now by more than this is dropped (a bad clock); within it "
        "the latency is counted as 0",
        unit="ms", ge=0, le=1000, conservative="lower",
    )  # fmt: skip
    command_ttl_ms: float = param(
        100.0, "A reflex command expires this long after its last tick", unit="ms", ge=20,
        le=1000, conservative="lower",
    )  # fmt: skip
    attack_fraction: float = param(
        0.25, "Share of the duration spent reaching the peak (higher is gentler)", ge=0.05,
        le=0.9, conservative="higher",
    )  # fmt: skip
    release_fraction: float = param(
        0.4, "Share of the duration spent returning (startle and withdraw)", ge=0.05, le=0.9
    )
    startle_lift_ratio: float = param(
        2.0, "Startle lift of the robot's startle dofs per unit of head deflection", ge=0, le=10,
        conservative="lower",
    )  # fmt: skip
    proximity_rearm_ratio: float = param(
        0.8, "A close face re-arms the proximity rule once it falls below this share of the "
        "threshold", ge=0.1, le=1,
    )  # fmt: skip
    motion_eps_deg: float = param(
        0.2, "Pose change that counts as a movement: a smaller reflex is no_motion, and the first "
        "such change ends the motion latency figure", unit="deg", ge=0.01, le=5,
        conservative="higher",
    )  # fmt: skip
    latency_window: int = param(200, "Latency samples kept per trigger", ge=10, le=10000)
    onset_watch_ratio: float = param(
        2.0,
        "Report only (SPEC 12A): a loud-onset sound that rises above this ratio over the floor "
        "is recorded in the debug report with its peak ratio, fired or not, so the threshold can "
        "be tuned from a real run. A rule with a lower threshold watches from its own threshold",
        unit="ratio", ge=1.5, le=100,
    )  # fmt: skip
    onset_window: int = param(
        500, "Onset peaks kept per loud-onset rule for the debug report", ge=10, le=10000
    )
    audio: ReflexAudioConfig = param({}, "Microphone level source")
    rules: list[ReflexRule] = param(
        [
            {
                "name": "startle-loud",
                "trigger": "loud_onset",
                "primitive": "startle",
                "threshold": 12.0,
                "amplitude_deg": 12,
                "cooldown_s": 3,
                "duration_ms": 400,
            },
            {
                "name": "withdraw-close",
                "trigger": "proximity",
                "primitive": "withdraw",
                "threshold": 0.4,
                "amplitude_deg": 15,
                "cooldown_s": 4,
                "duration_ms": 500,
            },
            {
                "name": "turn-to-voice",
                "trigger": "new_voice",
                "primitive": "turn_toward",
                "threshold": 3.0,
                "amplitude_deg": 40,
                "cooldown_s": 5,
                "duration_ms": 800,
            },
        ],
        "Reflex rules (SPEC 10)",
    )

    @model_validator(mode="after")
    def _shape_fits(self) -> ReflexConfig:
        if self.attack_fraction + self.release_fraction > 1 - 0.05:
            raise ValueError("attack_fraction + release_fraction must leave a hold of 5%")
        names = [r.name for r in self.rules]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate reflex rule names: {dupes}")
        return self


# --------------------------------------------------------------------------- safety (SPEC 12)


class SafetyConfig(Strict):
    extra_prompt_rules: list[str] = param(
        [],
        "Rules appended to the system prompt's safety block (SPEC 8); changes need the admin",
        admin_only=True,
    )
    watchdog_ms: float = param(
        200.0,
        "Return home slowly when no command arrives for this long",
        unit="ms",
        ge=20,
        le=2000,
        conservative="lower",
    )
    watchdog_return_factor: float = param(
        0.3, "Speed factor of the watchdog's return to home", ge=0.01, le=1, conservative="lower"
    )
    soft_stop: Literal["hold_position"] = param("hold_position", "Soft stop behaviour")
    hard_stop: Literal["disable_motors"] = param("disable_motors", "Hard stop behaviour")
    unmeasured_dynamics_fallback_vel: float = param(
        0.5,
        "Velocity cap used for manifest dynamics still null (dev only)",
        unit="rad/s",
        ge=0.01,
        le=10,
        conservative="lower",
    )
    unmeasured_dynamics_fallback_acc: float = param(
        2.0,
        "Acceleration cap used for manifest dynamics still null (dev only)",
        unit="rad/s^2",
        ge=0.01,
        le=100,
        conservative="lower",
    )
    unmeasured_dynamics_fallback_lin_vel: float = param(
        0.02,
        "Linear velocity cap for unmeasured translations (dev only)",
        unit="m/s",
        ge=0.001,
        le=1,
        conservative="lower",
    )
    unmeasured_dynamics_fallback_lin_acc: float = param(
        0.1,
        "Linear acceleration cap for unmeasured translations (dev only)",
        unit="m/s^2",
        ge=0.001,
        le=10,
        conservative="lower",
    )
    constraint_margin_deg: float = param(
        3.0,
        "Coupled constraints (head-body yaw) are planned this far inside their limit",
        unit="deg",
        ge=0,
        le=20,
        conservative="higher",
    )
    allow_unmeasured_dynamics: bool = param(
        False,
        "Run with manifest dynamics still null (development only, SPEC 12)",
        conservative="false",
    )


PSEUDONYM_LABEL = "{label}"


class PrivacyConfig(Strict):
    multimodal_frames: bool = param(
        False, "Allow single-frame image questions to LLM routes", opt_in=True
    )
    pseudonyms_to_cloud: bool = param(
        False,
        "Send neutral aliases instead of registry names to routes that are not local, and "
        "restore the names in the answer and in tool calls; a remote speech voice still hears the "
        "names it speaks (SPEC 8, 12)",
        conservative="true",
    )
    pseudonym_templates: dict[str, str] = param(
        {"en": "Person-{label}", "it": "Persona-{label}"},
        "Alias per language for pseudonyms_to_cloud; {label} becomes A, B, ... Z, AA, ... and "
        "comes last, after a prefix that ends in punctuation (so the alias is never an ordinary "
        "word). The language is the one of the turn in which a name is first aliased (SPEC 12)",
        admin_only=True,
    )
    keep_enrollment_images: bool = param(False, "Keep enrollment face crops", opt_in=True)
    debug_recording: bool = param(False, "Allow time-limited debug recording", opt_in=True)

    @field_validator("pseudonym_templates")
    @classmethod
    def _templates(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("privacy.pseudonym_templates needs at least one template")
        for lang, template in value.items():
            prefix = template[: -len(PSEUDONYM_LABEL)]
            ok = (
                template.count(PSEUDONYM_LABEL) == 1
                and template.endswith(PSEUDONYM_LABEL)
                and prefix.strip() != ""
                and not (prefix[-1].isalnum() or prefix[-1].isspace() or prefix[-1] == "_")
            )
            if not ok:
                raise ValueError(
                    f"privacy.pseudonym_templates[{lang!r}] must be a non-blank prefix ending in "
                    f"punctuation (for example 'Person-') followed by {PSEUDONYM_LABEL}, and "
                    "nothing else, so an alias can never be an ordinary word"
                )
        return value


# --------------------------------------------------------------------------- identity (SPEC 5, 6)


class IdentityConfig(Strict):
    match_threshold: float = param(0.45, "Cosine similarity threshold", ge=0, le=1)
    margin: float = param(0.1, "Required margin over the second best match", ge=0, le=1)
    votes_required: int = param(3, "Positive evaluations required", ge=1, le=20)
    votes_window: int = param(5, "Evaluations considered", ge=1, le=50)
    enroll_min_embeddings: int = param(5, "Face embeddings required at enrollment", ge=5, le=20)
    max_embeddings_per_person: int = param(
        20, "Cap on stored face embeddings per person, refresh included", ge=5, le=20
    )
    refresh_min_similarity: float = param(
        0.65, "Best similarity needed before a match adds an embedding", ge=0, le=1
    )
    refresh_max_similarity: float = param(
        0.98, "Skip a refresh embedding this close to a stored one", ge=0, le=1
    )
    refresh_min_interval_s: float = param(
        60.0, "Minimum time between refresh embeddings of one person", unit="s", ge=0, le=86400
    )
    last_seen_flush_interval_s: float = param(
        60.0, "Minimum time between disk writes of last_seen updates", unit="s", ge=0, le=86400
    )

    vote_max_age_s: float = param(
        10.0, "Identity votes older than this are ignored", unit="s", ge=1, le=300
    )

    @model_validator(mode="after")
    def _consistent(self) -> IdentityConfig:
        if self.votes_required > self.votes_window:
            raise ValueError("identity.votes_required must not exceed votes_window")
        if self.enroll_min_embeddings > self.max_embeddings_per_person:
            raise ValueError("identity.enroll_min_embeddings exceeds max_embeddings_per_person")
        if self.refresh_min_similarity < self.match_threshold:
            raise ValueError("identity.refresh_min_similarity must be at least match_threshold")
        return self


class EnrollmentConfig(Strict):
    target_samples: int = param(
        10,
        "Face embeddings collected per enrollment session; between "
        "identity.enroll_min_embeddings and identity.max_embeddings_per_person",
        ge=5,
        le=20,
    )
    min_score: float = param(
        0.8, "Detector score a face needs to be used as an enrollment sample", ge=0, le=1
    )
    min_face_px: int = param(
        64,
        "Faces with a shorter box side are not used as enrollment samples",
        unit="px",
        ge=8,
        le=512,
    )
    min_sharpness: float = param(
        25.0,
        "Variance of the Laplacian of the grayscale face crop (area-resized to "
        "sharpness_size_px); blurrier faces are rejected. Not yet tuned on real camera data",
        ge=0,
        le=10000,
    )
    sharpness_size_px: int = param(
        64, "Side of the square the face crop is resized to for the sharpness check", ge=16, le=256
    )
    min_gap_s: float = param(
        0.4, "Minimum time between two samples, so they differ in pose", unit="s", ge=0, le=10
    )
    consistency_min: float = param(
        0.45,
        "Cosine similarity a new sample needs with the mean of the ones collected (the same "
        "value as identity.match_threshold), so two people never mix in one session. Not yet "
        "tuned on real data",
        ge=-1,
        le=1,
    )
    min_pose_spread: float = param(
        0.2,
        "Spread of the nose offset from the eye midline (in inter-eye distances, a yaw proxy "
        "from the face landmarks) the samples must cover, so they show varied angles (SPEC 6); "
        "0 turns the check off. Not yet tuned on real data",
        ge=0,
        le=2,
    )
    centre_weight: float = param(
        0.5,
        "How much a face far from the image centre loses when the largest face is chosen "
        "(0: size only)",
        ge=0,
        le=1,
    )
    timeout_s: float = param(
        120.0, "A session without completion is dropped after this time", unit="s", ge=5, le=3600
    )


ATTENTION_CATEGORIES = ("active_speaker", "enrolled_person", "nearest_face", "salient_motion")


class AttentionConfig(Strict):
    priority_order: list[str] = param(
        ["active_speaker", "enrolled_person", "nearest_face", "salient_motion"],
        "Attention target preference order",
    )
    switch_hysteresis_s: float = param(
        1.5, "Min time before switching target", unit="s", ge=0, le=30
    )
    stale_s: float = param(
        4.0,
        "An entry not confirmed for this long is no longer a candidate",
        unit="s",
        ge=0.1,
        le=60,
    )
    speaker_hold_s: float = param(
        1.5, "How long an active-speaker hint stays valid", unit="s", ge=0.1, le=30
    )
    speaker_match_deg: float = param(
        20.0, "Angular gate to match a speaker bearing to a world entry", unit="deg", ge=0.5, le=90
    )
    motion_window_s: float = param(
        0.5, "Window over which an entry's bearing change counts as motion", unit="s", ge=0.05, le=5
    )
    motion_min_dps: float = param(
        10.0, "Angular speed above which an entry counts as moving", unit="deg/s", ge=0.1, le=720
    )
    release_s: float = param(
        3.0, "No target for this long hands the head back to idle motion", unit="s", ge=0, le=60
    )
    publish_hz: float = param(
        5.0, "Rate of attention.target while the target is unchanged", unit="Hz", ge=0.1, le=50
    )
    publish_move_deg: float = param(
        5.0, "Target bearing change that is published at once", unit="deg", ge=0.1, le=90
    )
    follow_gain: float = param(
        4.0, "Proportional gain of the gaze towards the target bearing", unit="1/s", ge=0.1, le=50
    )
    max_rate_dps: float = param(
        120.0, "Gaze speed cap before the safety limiter", unit="deg/s", ge=1, le=720
    )
    head_yaw_comfort_deg: float = param(
        45.0,
        "Head yaw offset from the body beyond which the body turns too",
        unit="deg",
        ge=0,
        le=180,
    )
    object_max_hold_s: float = param(
        30.0,
        "How long attention on an animal or object holds off a due scan; then the sweep runs",
        unit="s",
        ge=1,
        le=600,
    )
    switch_margin: float = param(
        0.25,
        "Relative advantage a same-category candidate needs to replace the current target",
        ge=0,
        le=5,
    )
    rear_band_deg: float = param(
        20.0,
        "Width of the band around straight behind where the gaze keeps its current side",
        unit="deg",
        ge=0,
        le=90,
    )
    body_recenter_dps: float = param(
        5.0,
        "Speed at which the body drifts back to home while the head is within comfort",
        unit="deg/s",
        ge=0,
        le=90,
    )
    command_ttl_s: float = param(
        0.5, "Lifetime of an attention command in the arbiter", unit="s", ge=0.05, le=5
    )

    @field_validator("priority_order")
    @classmethod
    def _known_categories(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("attention.priority_order must name at least one category")
        if unknown := sorted(set(v) - set(ATTENTION_CATEGORIES)):
            raise ValueError(f"attention.priority_order: unknown categories {unknown}")
        if len(set(v)) != len(v):
            raise ValueError("attention.priority_order: duplicate category")
        return v


# --------------------------------------------------------------------------- vision (SPEC 5, 13)


class VisionRates(Strict):
    detection_hz: float = param(8.0, "Face detection rate (cm4: 5 to 10)", unit="Hz", ge=0.1, le=60)
    embedding_hz: float = param(
        0.5, "Face embedding rate per track after the new-track burst", unit="Hz", ge=0.05, le=30
    )
    appearance_hz: float = param(
        0.5,
        "Appearance embedding rate per animal or object track after the new-track burst",
        unit="Hz",
        ge=0.05,
        le=30,
    )
    objects_hz: float = param(1.0, "Object detection rate when enabled", unit="Hz", ge=0.1, le=30)
    camera_hz: float = param(
        30.0,
        "Upper bound on camera reads handed to vision (read when the camera thread starts)",
        unit="Hz",
        ge=1,
        le=120,
        restart=True,
    )
    conversation_scale: float = param(
        0.5,
        "Vision rate multiplier while the robot thinks or speaks (SPEC 5, 8, 13), so the TTS "
        "keeps its CPU; an utterance in progress and idle keep 1. It also divides the new-track "
        "burst gaps. At least 0.5, so the 3-of-5 identity decision of a track that starts while "
        "the robot speaks stays inside the 2.5 s budget of SPEC 13, awake or drowsy, with one "
        "missed vote",
        ge=0.5,
        le=1,
    )


class TrackerConfig(Strict):
    high_score: float = param(
        0.6, "Detections at or above this start tracks (the detection score threshold)", ge=0, le=1
    )
    low_score: float = param(
        0.1, "Detections between this and high_score only keep existing tracks", ge=0, le=1
    )
    match_iou: float = param(0.3, "Minimum IoU for the first association", ge=0, le=1)
    low_match_iou: float = param(0.5, "Minimum IoU for the low-score association", ge=0, le=1)
    match_distance_heights: float = param(
        0.6,
        "A high-score detection that fails match_iou may still match a track (same label, height "
        "within match_height_ratio) whose predicted box centre is at most this many of the "
        "track's box heights away; IoU matches go first. A high-score detection this close to a "
        "confirmed track matched in the same update is a duplicate of it and starts no track. 0 "
        "turns both off. Must stay below vision.rebirth_distance_heights",
        ge=0,
        le=2,
    )
    low_match_distance_heights: float = param(
        0.4,
        "The same centre-distance gate for the low-score association (stricter: weaker "
        "evidence); 0 turns it off. Must stay below vision.rebirth_distance_heights",
        ge=0,
        le=2,
    )
    match_height_ratio: float = param(
        1.5,
        "The centre-distance gates and the duplicate rule apply only when the larger of the "
        "detection and track box heights is at most this many times the smaller: a face seen "
        "side-on keeps its size (1.13x on the robot), a smaller face beside or behind it does "
        "not (0.5x), so it starts its own track",
        ge=1,
        le=4,
    )
    confirm_hits: int = param(2, "Matched detections before a track is reported", ge=1, le=10)
    lost_s: float = param(
        1.5, "A track unmatched for this long (capture stamps) is removed", unit="s", ge=0.1, le=60
    )

    @model_validator(mode="after")
    def _ordered(self) -> TrackerConfig:
        if self.low_score > self.high_score:
            raise ValueError("vision.tracker.low_score must not exceed high_score")
        return self


class VisionConfig(Strict):
    rates: VisionRates = param({}, "Per-stage target rates, scaled by the circadian state")
    tracker: TrackerConfig = param({}, "ByteTrack-class tracker")
    threads: int = param(2, "CPU threads of the frame-path models", ge=1, le=8, restart=True)
    detector_width_px: int = param(
        320, "Face detector input width (SPEC 13: 320x240)", unit="px", ge=64, le=1280, restart=True
    )
    detector_height_px: int = param(
        240, "Face detector input height", unit="px", ge=64, le=1280, restart=True
    )
    face_min_px: int = param(
        24, "Faces with a shorter box side are ignored", unit="px", ge=8, le=512
    )
    face_lost_periods: float = param(
        3.0,
        "A face track also survives this many face-detection periods at the rate the governor "
        "allows now: the lifetime is the larger of this and tracker.lost_s, so a slower detector "
        "(conversation, circadian or load scale) does not end tracks between two detections",
        ge=1,
        le=100,
    )
    embed_min_face_px: int = param(
        40, "Faces with a shorter box side are tracked but not embedded", unit="px", ge=8, le=512
    )
    embedding_burst: int = param(
        5,
        "Face evaluations at the start of a track, before the steady rate; at least the votes "
        "required, so one missed vote still fits the SPEC 13 identity budget of 2.5 s",
        ge=1,
        le=10,
    )
    burst_min_gap_s: float = param(
        0.25, "Minimum capture-time gap between two burst evaluations", unit="s", ge=0, le=5
    )
    max_embeds_per_frame: int = param(
        2, "Cap on face embeddings computed for one frame, to bound its latency", ge=1, le=16
    )
    max_entity_embeds_per_frame: int = param(
        1,
        "Cap on animal and object appearance embeddings (DINOv2-S, the costly part of the "
        "object stage) computed for one object frame; separate from max_embeds_per_frame so "
        "entities never take a face's share. Entities are served oldest embedding first",
        ge=1,
        le=16,
    )
    entity_burst_min_gap_s: float = param(
        1.0,
        "Minimum capture-time gap between two burst evaluations of one animal or object track",
        unit="s",
        ge=0,
        le=10,
    )
    report_window: int = param(
        200,
        "Report only: ended face tracks kept for the lifetime summary",
        ge=10,
        le=10000,
        restart=True,
    )
    report_rate_window_s: float = param(
        60.0,
        "Report only: capture-time window of the face track creation rate (per minute), "
        "ending at the last processed frame",
        unit="s",
        ge=10,
        le=3600,
    )
    rebirth_window_s: float = param(
        5.0,
        "Report only: a face track born at most this long (capture time) after another one "
        "expired, near its last box, counts as a rebirth of it in the tracks report",
        unit="s",
        ge=0.1,
        le=60,
    )
    rebirth_distance_heights: float = param(
        1.0,
        "Report only: largest distance between the new track's first box centre and the ended "
        "track's last box centre, in heights of the ended box, for a rebirth. Must stay above "
        "both vision.tracker distance gates, or an overlapped rebirth could never be counted",
        ge=0.1,
        le=10,
    )
    models_dir: str = param(
        "",
        "Folder with the ONNX model files; empty means models/ in the app data folder (data_dir)",
        restart=True,
    )
    objects_enabled: bool = param(
        True,
        "Run the object detector (NanoDet-Plus-m 320, SPEC 5, 13) at vision.rates.objects_hz",
        restart=True,
    )
    object_score_min: float = param(
        0.4,
        "Object detections below this score are dropped; it also replaces tracker.high_score "
        "for the animal and object tracker, so a 0.4 detection starts a track",
        ge=0,
        le=1,
        restart=True,
    )
    object_lost_s: float = param(
        5.0,
        "Shortest time an animal or object track may stay unmatched before it is removed",
        unit="s",
        ge=0.1,
        le=300,
    )
    object_lost_periods: float = param(
        3.0,
        "An animal or object track also survives this many object-detection periods, at the "
        "rate the governor currently allows (1 / (objects_hz x scale)): the lifetime is the "
        "larger of this and object_lost_s, so slowing the detector never kills tracks between "
        "two frames",
        ge=1,
        le=100,
    )
    object_confirm_hits: int = param(
        2,
        "Matched detections before an animal or object track is reported. 2 at 1 fps means a "
        "thing must be seen in two consecutive object frames: 31% of detections at the default "
        "score are false and mostly do not repeat, at the cost of a 1 s delay and of things "
        "seen once",
        ge=1,
        le=10,
    )
    object_match_distance_heights: float = param(
        0.0,
        "Replaces both tracker centre-distance gates (and the duplicate rule) for the animal and "
        "object tracker. Off by default: the gate is in box heights, tuned on faces, and two tall "
        "things side by side (bottles) are about 0.3 heights apart, so the second one would "
        "never start a track",
        ge=0,
        le=2,
    )
    object_nms_iou: float = param(
        0.6,
        "Detections of one class overlapping more than this IoU are merged (NanoDet's own value)",
        ge=0.1,
        le=0.95,
        restart=True,
    )
    animal_categories: list[str] = param(
        ["bird", "cat", "dog", "horse", "sheep", "cow"],
        "Detector classes (COCO names as in models.yaml) reported as kind animal",
        restart=True,
    )
    object_categories: list[str] = param(
        [
            "cup",
            "bottle",
            "chair",
            "couch",
            "potted_plant",
            "tv",
            "laptop",
            "book",
            "remote",
            "cell_phone",
            "bowl",
            "teddy_bear",
        ],
        "Detector classes reported as kind object; a class in neither list is ignored, and so "
        "is person (faces come from the face detector)",
        restart=True,
    )
    object_display_names: dict[str, str] = param(
        {"potted_plant": "potted plant", "cell_phone": "cell phone", "teddy_bear": "teddy bear"},
        "Readable category for a detector class in observations, placeholder names and scene "
        "text; a class not listed uses its name with underscores turned into spaces (English "
        "only: localised strings are not provided yet)",
    )

    @model_validator(mode="after")
    def _disjoint_categories(self) -> VisionConfig:
        both = set(self.animal_categories) & set(self.object_categories)
        if both:
            raise ValueError(f"vision categories are both animal and object: {sorted(both)}")
        return self

    @model_validator(mode="after")
    def _association_below_rebirth(self) -> VisionConfig:
        gate = max(self.tracker.match_distance_heights, self.tracker.low_match_distance_heights)
        if gate >= self.rebirth_distance_heights:
            raise ValueError(
                "vision.tracker distance gates must stay below vision.rebirth_distance_heights, "
                "or an overlapped rebirth could never be counted"
            )
        return self


# --------------------------------------------------------------------------- world (SPEC 5A)


class WorldHalfLife(Strict):
    person: float = param(120.0, "Confidence half-life of a person entry", unit="s", ge=1, le=86400)
    animal: float = param(
        300.0, "Confidence half-life of an animal entry", unit="s", ge=1, le=86400
    )
    object: float = param(
        1800.0, "Confidence half-life of an object entry", unit="s", ge=1, le=86400
    )


class WorldScanConfig(Strict):
    idle_interval_s: float = param(
        120.0,
        "Idle time before the robot looks around (before the energy trait)",
        unit="s",
        ge=5,
        le=3600,
    )
    jitter_s: float = param(30.0, "Random +- offset on the idle interval", unit="s", ge=0, le=1800)
    target_lost_delay_s: float = param(
        8.0,
        "Time after the person target (or speaker hint) is lost before the robot looks around, "
        "instead of the full idle interval (before the energy trait)",
        unit="s",
        ge=1,
        le=600,
    )
    target_lost_jitter_s: float = param(
        3.0, "Random +- offset on the target-lost delay", unit="s", ge=0, le=300
    )
    dwell_s: float = param(1.0, "Pause at each sweep waypoint", unit="s", ge=0.1, le=30)
    overlap_deg: float = param(
        15.0, "Overlap between the views of neighbouring waypoints", unit="deg", ge=0, le=60
    )
    speed_dps: float = param(
        40.0, "Sweep speed between waypoints, before the safety limiter", unit="deg/s", ge=5, le=360
    )
    min_amplitude: float = param(
        0.3, "Smallest share of the full circle a sweep covers (low energy)", ge=0.1, le=1
    )
    arrive_tol_deg: float = param(
        4.0,
        "A waypoint counts as reached within this distance of the pose",
        unit="deg",
        ge=0.5,
        le=30,
    )
    waypoint_timeout_s: float = param(
        10.0, "Give up on a waypoint the pose does not reach in this time", unit="s", ge=1, le=120
    )
    min_step_ratio: float = param(
        0.5,
        "Smallest waypoint spacing as a share of the camera's field of view",
        ge=0.1,
        le=1,
    )
    command_ttl_s: float = param(
        0.25, "Lifetime of a scan command if it is not renewed", unit="s", ge=0.05, le=2
    )
    glance_hold_s: float = param(
        1.0, "How long a glance holds its bearing", unit="s", ge=0.1, le=10
    )
    glance_cooldown_s: float = param(8.0, "Minimum time between glances", unit="s", ge=0, le=600)
    glance_max_age_s: float = param(
        2.0, "An unexplained bearing older than this is not glanced at", unit="s", ge=0.1, le=60
    )
    doa_poll_hz: float = param(
        5.0, "Sound direction read rate", unit="Hz", ge=0.5, le=50, restart=True
    )
    doa_confidence: float = param(
        0.5, "Confidence of a sound direction observation", ge=0, le=1, restart=True
    )

    @model_validator(mode="after")
    def _jitter_inside_interval(self) -> WorldScanConfig:
        if self.jitter_s >= self.idle_interval_s:
            raise ValueError("jitter_s must be smaller than idle_interval_s")
        if self.target_lost_jitter_s > self.target_lost_delay_s:
            raise ValueError("target_lost_jitter_s must not exceed target_lost_delay_s")
        return self


CATALOGUE_BODY_SLACK = 1024


class WorldAutoenrollConfig(Strict):
    enabled: bool = param(
        True,
        "Auto-enroll recurring unknown animals and objects (SPEC 5A); turning it on again "
        "needs a restart, turning it off deletes the collected candidates",
    )
    kinds: list[Literal["animal", "object"]] = param(
        ["animal", "object"],
        "Kinds that may be auto-enrolled; people never are. The recognition spike advises "
        "animals only (objects separate poorly)",
    )
    min_days: int = param(3, "Distinct days a cluster must be seen on", ge=1, le=60)
    min_sightings: int = param(5, "Sightings a cluster needs before it is enrolled", ge=1, le=500)
    max_spread: float = param(
        0.25,
        "Cosine distance to the cluster centroid, max over members (DINOv2-S scale)",
        ge=0.01,
        le=1,
        conservative="lower",
    )
    join_radius: float = param(
        0.125,
        "Cosine distance from a candidate cluster's centroid within which a sighting joins it; "
        "tighter than max_spread so two look-alikes stay apart",
        ge=0.01,
        le=1,
        conservative="lower",
    )
    known_match_threshold: float = param(
        0.875,
        "Cosine similarity to an enrolled animal or object that attaches its identity; keep it "
        "at or above 1 - join_radius, or a look-alike is taken for the enrolled one",
        ge=0.1,
        le=1,
    )
    sighting_min_interval_s: float = param(
        300.0, "At most one sighting per world entry in this time", unit="s", ge=1, le=86400
    )
    max_clusters: int = param(
        200, "Candidate clusters kept; the least recently updated go first", ge=1, le=5000
    )
    max_members: int = param(
        30, "Embeddings kept per candidate cluster (oldest dropped)", ge=2, le=200
    )
    retention_days: int = param(
        30, "A candidate cluster not seen for this long is deleted", unit="d", ge=1, le=365
    )
    persist_interval_s: float = param(
        60.0, "Longest time changed candidates stay unwritten", unit="s", ge=1, le=3600
    )
    max_tracked_entries: int = param(
        512, "World entries remembered for the sighting throttle", ge=8, le=10000
    )

    @model_validator(mode="after")
    def _consistent(self) -> WorldAutoenrollConfig:
        if self.join_radius > self.max_spread:
            raise ValueError("join_radius must not exceed max_spread")
        if self.min_sightings > self.max_members:
            raise ValueError(
                "min_sightings must not exceed max_members (sightings are counted "
                "from the retained members)"
            )
        return self


class WorldConfig(Strict):
    camera_fov_h_deg: float = param(
        78.0, "Horizontal field of view of the main camera", unit="deg", ge=10, le=360
    )
    camera_fov_v_deg: float = param(
        58.0, "Vertical field of view of the main camera", unit="deg", ge=10, le=180
    )
    match_gate_deg: float = param(
        15.0, "Angular gate for matching an observation to an entry", unit="deg", ge=0.5, le=90
    )
    tie_tolerance_deg: float = param(
        3.0,
        "Candidates this close to the nearest in angle are ranked by appearance",
        unit="deg",
        ge=0,
        le=90,
    )
    range_gate_m: float = param(
        0.5, "Range gate, used when both have a distance", unit="m", ge=0.01, le=10
    )
    half_life_s: WorldHalfLife = param({}, "Confidence half-life per kind")
    out_of_view_decay_factor: float = param(
        0.2, "Decay rate multiplier for entries outside the sensors' view", ge=0.01, le=1
    )
    forget_confidence: float = param(
        0.1, "Entries below this confidence are removed", ge=0.01, le=1
    )
    moved_threshold_deg: float = param(
        10.0, "Angular shift that publishes world.entity_moved", unit="deg", ge=0.5, le=90
    )
    initial_confidence: float = param(
        0.5, "Confidence of a new entry, times the observation confidence", ge=0, le=1
    )
    confidence_gain: float = param(
        0.3, "Share of the missing confidence gained per matching observation", ge=0, le=1
    )
    position_blend: float = param(
        0.5, "Weight of a new observation when updating an entry's position", ge=0, le=1
    )
    initial_uncertainty_deg: float = param(
        5.0, "Angular uncertainty of a new entry", unit="deg", ge=0, le=90
    )
    min_uncertainty_deg: float = param(
        2.0, "Floor of an entry's angular uncertainty", unit="deg", ge=0, le=90
    )
    bearing_only_gain_factor: float = param(
        0.3,
        "Share of the confidence gain a bearing-only observation (sound) gets; it does not "
        "mark an entry in view",
        ge=0,
        le=1,
    )
    scan: WorldScanConfig = param({}, "Looking around (SPEC 5A, Scanning)")
    autoenroll: WorldAutoenrollConfig = param({}, "Auto-enrollment (SPEC 5A)")

    @model_validator(mode="after")
    def _uncertainty_order(self) -> WorldConfig:
        if self.initial_uncertainty_deg < self.min_uncertainty_deg:
            raise ValueError("initial_uncertainty_deg must be >= min_uncertainty_deg")
        return self


# --------------------------------------------------------------------------- voice (SPEC 8)


class Prosody(Strict):
    rate: float = param(0.0, "Speech rate delta (additive to 1.0)", ge=-1, le=1)
    pitch_st: float = param(0.0, "Pitch delta", unit="semitones", ge=-12, le=12)
    volume_db: float = param(0.0, "Volume delta", unit="dB", ge=-24, le=12)
    variability: float = param(0.0, "Variability delta", ge=-1, le=1)
    pause_scale: float = param(1.0, "Pause length multiplier", ge=0.2, le=4)
    instructions: str = param("", "Delivery instructions for instructable remote TTS")
    sounds: list[str] = param([], "Non-verbal sounds allowed in this mood")


class VoiceBase(Strict):
    rate: float = param(1.0, "Base speech rate", ge=0.5, le=1.5)
    pitch_st: float = param(0.0, "Base pitch", unit="semitones", ge=-6, le=6)
    volume_db: float = param(0.0, "Base volume", unit="dB", ge=-24, le=6)
    variability: float = param(0.6, "Base variability (maps to Piper noise scales)", ge=0, le=1)


class VoiceLimits(Strict):
    rate: tuple[float, float] = param((0.7, 1.3), "Allowed speech rate range")
    pitch_st: tuple[float, float] = param((-4.0, 4.0), "Allowed pitch range")
    volume_db: tuple[float, float] = param((-12.0, 3.0), "Allowed volume range")
    variability: tuple[float, float] = param((0.0, 1.0), "Allowed variability range")


class VadConfig(Strict):
    threshold: float = param(
        0.5, "Speech probability at or above which a frame counts as speech", ge=0.05, le=0.95
    )
    watch_prob: float = param(
        0.2,
        "Report only: a burst of frames at or above this probability is recorded with its peak "
        "and its longest run above threshold, started or not, so the threshold can be tuned "
        "from a real run. A value above threshold watches from threshold",
        ge=0.01,
        le=0.95,
        restart=True,
    )
    watch_window: int = param(
        500, "Report only: bursts kept for the report", ge=10, le=10000, restart=True
    )
    off_margin: float = param(
        0.15,
        "Hysteresis: once speaking, a frame still counts as speech down to threshold minus this "
        "(at least 0.01), so a dip inside a word does not start the end-silence timer",
        ge=0,
        le=0.5,
    )
    min_speech_ms: float = param(
        200.0,
        "Continuous speech needed before speech_start fires; shorter bursts (clicks, coughs) "
        "are ignored. Rounded up to whole VAD frames",
        unit="ms",
        ge=32,
        le=1000,
    )
    end_silence_ms: float = param(
        500.0,
        "Silence after the last speech frame that ends the utterance. It is the first term of "
        "the end-of-speech to first-audio budget (SPEC 13: 1.5 s typical, 2.0 s limit, measured "
        "from the last speech frame): 500 ms leaves about 1.0 s for ASR final, LLM first "
        "sentence and TTS first chunk, and 2.0 s at the limit. Higher values wait for slower "
        "speakers, such as children, at the cost of latency. Rounded up to whole VAD frames",
        unit="ms",
        ge=100,
        le=3000,
    )
    pre_roll_ms: float = param(
        300.0,
        "Audio kept from before the detected start and handed to the consumer, so the first "
        "syllable is not cut. Rounded up to whole VAD frames",
        unit="ms",
        ge=0,
        le=2000,
    )
    max_utterance_s: float = param(
        20.0,
        "An utterance is cut after this long even if speech continues (bounds RAM and ASR "
        "work); speech that continues starts a new utterance",
        unit="s",
        ge=1,
        le=120,
    )
    threads: int = param(1, "ONNX Runtime threads for the Silero model", ge=1, le=4, restart=True)


class AsrConfig(Strict):
    models: dict[str, str] = param(
        {"it": "vosk-small-it", "en": "vosk-small-en"},
        "Vosk model per ASR language, by name in the voice model manifest. The language of an "
        "utterance is the speaker's registry preference (`preferences.language` of the person "
        "record the robot is attending to) when a model exists for it, else "
        "`conversation.language`, else the first language listed here. All models are loaded "
        "when the voice pipeline starts",
        restart=True,
    )


class KeywordSet(Strict):
    stop: list[str] = param([], "Phrases that mean stop (what you are doing)")
    yes: list[str] = param(
        [],
        "Phrases that mean yes: the robot nods (voice.fastpath.answer_window_s says when it does "
        "not). Accents are ignored, so Italian `sì` would also match the pronoun `si` "
        "(`si chiama Luca`) and is not in the defaults",
    )
    no: list[str] = param([], "Phrases that mean no: the robot shakes its head")
    greeting: list[str] = param([], "Greeting phrases")
    wake_phrases: list[str] = param(
        [],
        "Phrases that wake the robot when they occur anywhere in the final text. Use words the "
        "recognizer's vocabulary has: a robot's own name can be out of vocabulary in the small "
        "Vosk models, so the default phrases are built from common words (`ehi robot`, "
        "`hey robot`)",
    )
    name_aliases: list[str] = param(
        [],
        "Spellings the recognizer produces for the robot's own name (`voice.default_name`) in this "
        "language, supplied by the robot manifest (SPEC 4A); none in the core defaults. They "
        "apply only when the robot has no name or its name starts with the first word of "
        "`voice.default_name`; a robot renamed to something else ignores them. They emit "
        "speech.name_called and count as the wake word only in a short final text "
        "(voice.fastpath.alias_wake_max_words), because an alias can be a real word",
    )
    name_wake_templates: list[str] = param(
        [],
        "Wake phrases built around the robot's name, each with exactly one `{name}` placeholder "
        "(`ciao {name}`). They apply when the robot has a name (`voice.wake_word`, or the one "
        "the platform reports) whose words are all in this language's recognizer vocabulary; "
        "otherwise only `wake_phrases` wake the robot in this language. The bare name is a name "
        "alias too: it wakes only in a short final text",
    )

    @field_validator("name_wake_templates")
    @classmethod
    def _templates_have_the_name(cls, v: list[str]) -> list[str]:
        for t in v:
            if t.count("{name}") != 1:
                raise ValueError(f"name_wake_templates: {t!r} needs exactly one {{name}}")
        return v


class FastpathConfig(Strict):
    addressed_window_s: float = param(
        15.0,
        "After an utterance that needed the wake word and was addressed, the next utterance that "
        "starts within this long after its last speech is addressed too, without the wake word "
        "(when both speakers are known people they must be the same person)",
        unit="s", ge=1, le=600,
    )  # fmt: skip
    window_max_s: float = param(
        120.0,
        "The addressed window never reaches further than this from the wake word that opened "
        "it, however many follow-ups extend it",
        unit="s", ge=1, le=3600,
    )  # fmt: skip
    alias_wake_max_words: int = param(
        4,
        "A name alias counts as the wake word only in a final text of at most this many words",
        ge=1,
        le=20,
    )
    answer_window_s: float = param(
        6.0,
        "A yes or no heard within this long after the robot finished a sentence that ended in a "
        "question mark is the answer to it: no nod or shake (the reply goes to the conversation "
        "as usual; also none while the robot speaks). A later sentence that is not a question ends "
        "the window. 0 = only while the "
        "question is still playing",
        unit="s", ge=0, le=60,
    )  # fmt: skip
    gesture_max_words: int = param(
        3,
        "A yes or no nods or shakes the head only when the final text of an addressed utterance "
        "has at most this many words and holds exactly one of the two (`yes, but...` or "
        "`non è certo` are sentences, not answers); the speech.keyword events are not affected",
        ge=1,
        le=20,
    )
    doa_max_age_ms: float = param(
        1500.0,
        "A sound direction reading older than this does not point the head at the speaker",
        unit="ms", ge=0, le=10000,
    )  # fmt: skip
    loud_db: float = param(
        -18.0,
        "Speech louder than this RMS level (dBFS, over the speech frames of the utterance) emits "
        "speech.loud",
        unit="dB", ge=-60, le=0,
    )  # fmt: skip
    fast_words_per_s: float = param(
        4.5,
        "Speech faster than this emits speech.fast (needs min_words_for_rate words)",
        unit="Hz", ge=1, le=10,
    )  # fmt: skip
    min_words_for_rate: int = param(
        4, "Fewer words than this give no speaking rate (too short to measure)", ge=2, le=50
    )
    frame_ms: float = param(40.0, "Frame length of the prosody analysis", unit="ms", ge=10, le=100)
    active_floor_db: float = param(
        -55.0,
        "Frames quieter than this RMS level (dBFS) are not analysed for pitch",
        unit="dB", ge=-90, le=-20,
    )  # fmt: skip
    pitch_min_hz: float = param(70.0, "Lowest pitch looked for", unit="Hz", ge=40, le=200)
    pitch_max_hz: float = param(400.0, "Highest pitch looked for", unit="Hz", ge=150, le=800)
    voiced_threshold: float = param(
        0.45,
        "Normalised autocorrelation peak needed to call a frame voiced and take its pitch",
        ge=0.1, le=0.95,
    )  # fmt: skip


VOICE_NAME = re.compile(r"[A-Za-z0-9._-]+")


class PiperConfig(Strict):
    """The Piper backend runs as one persistent subprocess per voice (SPEC 8, TTS backends)."""

    python: str = param(
        "",
        "Absolute path of the Piper virtualenv's interpreter (piper-tts is GPL-3.0 and runs in "
        "its own process); empty means piper-venv/bin/python in the app data folder, where the "
        "installer puts it. It is executed, so a change needs the admin PIN",
        restart=True, admin_only=True,
    )  # fmt: skip
    chunk_ms: float = param(
        200.0,
        "Audio the worker sends per message; a cancel abandons what is still unsent",
        unit="ms", ge=20, le=2000, restart=True,
    )  # fmt: skip
    startup_timeout_s: float = param(
        60.0,
        "Time a worker gets to load its voice and report its sample rate",
        unit="s", ge=1, le=600,
    )  # fmt: skip
    request_timeout_s: float = param(
        20.0,
        "Longest wait for the worker's next message while a sentence is synthesised",
        unit="s", ge=0.5, le=120,
    )  # fmt: skip
    restart_backoff_s: tuple[float, float] = param(
        (0.5, 30.0),
        "First and longest wait before a crashed worker is started again (doubles each time)",
    )
    intra_op_threads: int = param(
        2,
        "ONNX Runtime threads the worker uses inside one operator; 0 keeps the runtime default "
        "(one per core). On the CM4, 2 measured the best real-time factor (SPEC 13)",
        ge=0, le=16, restart=True,
    )  # fmt: skip
    inter_op_threads: int = param(
        1,
        "ONNX Runtime threads the worker uses across operators; 0 keeps the runtime default",
        ge=0, le=16, restart=True,
    )  # fmt: skip

    @field_validator("python")
    @classmethod
    def _absolute_interpreter(cls, v: str) -> str:
        if v and not v.startswith("/"):
            raise ValueError("voice.piper.python must be an absolute path (or empty)")
        return v


class OpenAISpeechConfig(Strict):
    """The remote speech backend follows the LLM routes' rules (SPEC 8, TTS backends)."""

    route: str = param(
        "",
        "Name of the llm_routes entry whose base_url, key and timeouts the speech requests "
        "use; empty disables the backend",
        restart=True,
    )
    model: str = param("gpt-4o-mini-tts", "Speech model sent to /audio/speech", restart=True)
    voice: str = param("marin", "Provider voice name", restart=True)
    instructable: bool = param(
        True, "The model takes `instructions` (mood as natural language)", restart=True
    )


class BargeInConfig(Strict):
    enabled: bool = param(
        True,
        "User speech while the robot talks stops its speech and cancels the answer in flight "
        "(SPEC 8 stage 8). Off: the robot is deaf to speech while it speaks, except for the "
        "stop keyword when keyword_enabled",
    )
    threshold: float = param(
        0.8,
        "Speech probability a frame needs to count towards a barge-in: stricter than "
        "voice.vad.threshold, because the robot's own voice is speech to the VAD",
        ge=0.05, le=0.99,
    )  # fmt: skip
    min_speech_ms: float = param(
        96.0,
        "Continuous user speech needed before the robot is cut off. The onset-to-stop latency "
        "(SPEC 13: 200 ms) is this plus up to one microphone chunk, so keep it low; rounded up "
        "to whole VAD frames",
        unit="ms", ge=32, le=150,
    )  # fmt: skip
    onset_enabled: bool = param(
        True,
        "Second, faster path to a barge-in for words that start with a fricative (English "
        '"stop": a 130 ms /s/ that the VAD scores at 0.4 to 0.7 until the vowel): a run of '
        "frames that are fricative-like (mostly high-frequency energy) and scored at or above "
        "onset_threshold, or at or above threshold, lasting onset_min_speech_ms cuts the robot "
        "off without waiting for threshold. Counted from the first sound, so SPEC 13 (200 ms) "
        "holds for every language",
    )
    onset_threshold: float = param(
        0.4,
        "Speech probability a frame needs to count towards the onset path (at most threshold). "
        "Measured on the M3 clips: the /s/ of English \"stop\" scores 0.35 to 0.9; clicks, "
        "coughs, claps and white noise score under 0.25",
        ge=0.05, le=0.99,
    )  # fmt: skip
    onset_hf_cutoff_hz: float = param(
        2000.0,
        "Frequency above which energy counts as high-frequency for the onset path's "
        "fricative test",
        unit="Hz", ge=500, le=7000,
    )  # fmt: skip
    onset_hf_min_ratio: float = param(
        0.7,
        "Share of a frame's energy above onset_hf_cutoff_hz that a frame scored between "
        "onset_threshold and threshold needs to count on the onset path: /s/ reads 0.84 to "
        "1.0, voiced speech 0 to 0.1. Keeps quiet voiced sounds on the stricter path",
        ge=0.0, le=1.0,
    )  # fmt: skip
    onset_min_level_db: float = param(
        -46.0,
        "Level (RMS, dBFS) a frame scored between onset_threshold and threshold needs on the "
        "onset path: the echo of the robot's own \"s\" at a low echo gain sits near the "
        "microphone noise floor, where no correlation can tell it from a user. A user's /s/ "
        "reads -40 to -29 dBFS",
        unit="dB", ge=-80, le=-20,
    )  # fmt: skip
    onset_min_speech_ms: float = param(
        64.0,
        "Continuous onset-path frames needed before the cut; rounded up to whole VAD frames. The "
        "stop follows within one microphone chunk of this. Like min_speech_ms it adds directly to "
        "the onset-to-stop latency: keep it at or below min_speech_ms, or the onset path is "
        "never the faster one",
        unit="ms", ge=32, le=150,
    )  # fmt: skip
    onset_echo_corr_max: float = param(
        0.4,
        "Echo guard of the onset path (at most echo_corr_max): stricter because a frame the VAD "
        "scores under threshold is weaker evidence than one it scores over it. A fricative-like "
        "frame must pass it in the full band and in the band above onset_hf_cutoff_hz, where "
        "its evidence is and the robot's low-frequency voice does not hide a user's /s/",
        ge=0.1, le=1.0,
    )  # fmt: skip
    echo_corr_max: float = param(
        0.6,
        "Residual-echo guard for robots without hardware AEC: a stretch of microphone audio "
        "whose normalised cross-correlation with what the speaker played (best lag within "
        "echo_max_lag_ms) is above this is the robot's own voice, not a barge-in. User speech "
        "mixed with echo correlates less the louder it is: it must carry roughly "
        "(1/corr^2 - 1) times the echo's power to get through. 1 turns the guard off",
        ge=0.1, le=1.0,
    )  # fmt: skip
    echo_max_lag_ms: float = param(
        300.0,
        "How far back in what was played the echo check looks for a match (speaker to microphone "
        "delay, room and device latency)",
        unit="ms", ge=0, le=1000,
    )  # fmt: skip
    echo_window_ms: float = param(
        128.0,
        "Length of microphone audio each echo check compares with the played reference",
        unit="ms", ge=32, le=1000,
    )  # fmt: skip
    echo_reference_s: float = param(
        2.0,
        "How much of the played audio the speaker keeps for the echo check (at least "
        "echo_max_lag_ms plus echo_window_ms)",
        unit="s", ge=0.5, le=10, restart=True,
    )  # fmt: skip
    pre_roll_ms: float = param(
        64.0,
        "Audio from before the detected onset handed to the listener with the barge-in, so the "
        "first syllable is not lost. Kept short: before the onset the robot's own voice is "
        "still in the microphone",
        unit="ms", ge=0, le=500,
    )  # fmt: skip
    keyword_enabled: bool = param(
        True,
        "While the robot speaks, run the stop keywords (voice.keywords.<language>.stop) on a "
        "recognizer fed with candidate speech frames, so stop is heard even when the barge-in "
        "threshold is not met (SPEC 7A: the stop keyword is always active)",
    )
    keyword_threshold: float = param(
        0.5, "Speech probability a frame needs to be fed to the keyword recognizer",
        ge=0.05, le=0.99,
    )  # fmt: skip
    keyword_echo_corr_max: float = param(
        0.8,
        "Echo guard of the keyword path: looser than echo_corr_max so a quiet or mixed stop is "
        "heard, strict enough that the robot saying a stop word does not stop itself",
        ge=0.1, le=1.0,
    )  # fmt: skip
    keyword_window_s: float = param(
        2.0,
        "The keyword recognizer listens at most this long after the first candidate frame",
        unit="s", ge=0.2, le=10,
    )  # fmt: skip
    keyword_hold_ms: float = param(
        400.0,
        "The keyword recognizer stops after this much audio without a candidate frame",
        unit="ms", ge=64, le=2000,
    )  # fmt: skip

    @model_validator(mode="after")
    def _onset_is_not_stricter_than_the_main_path(self) -> BargeInConfig:
        if self.onset_threshold > self.threshold:
            raise ValueError("barge_in.onset_threshold must not exceed threshold")
        if self.onset_echo_corr_max > self.echo_corr_max:
            raise ValueError("barge_in.onset_echo_corr_max must not exceed echo_corr_max")
        return self

    @model_validator(mode="after")
    def _reference_covers_the_check(self) -> BargeInConfig:
        need = (self.echo_max_lag_ms + self.echo_window_ms) / 1000.0
        if self.echo_reference_s < need:
            raise ValueError(
                "barge_in.echo_reference_s must cover echo_max_lag_ms + echo_window_ms"
            )
        return self


class VoiceConfig(Strict):
    enabled: bool = param(
        False,
        "Run the voice pipeline: the microphone hub keeps every audio chunk for it and the "
        "speaker is created (SPEC 8)",
        restart=True,
    )
    echo_tail_ms: float = param(
        300.0,
        "The robot counts as speaking this long after its playback ends: the startle reflex and "
        "the sound-direction source ignore the microphones meanwhile (room echo, mic latency)",
        unit="ms", ge=0, le=2000, conservative="higher",
    )  # fmt: skip
    sample_rate_hz: int = param(
        16000, "Rate of the audio frames the microphone hub hands to the voice pipeline",
        unit="Hz", ge=8000, le=48000, restart=True,
    )  # fmt: skip
    capture_queue_s: float = param(
        10.0,
        "Audio the hub keeps for a slow voice consumer; beyond it the oldest chunks are dropped "
        "and counted",
        unit="s", ge=0.5, le=120, restart=True,
    )  # fmt: skip
    playback_chunk_ms: float = param(
        100.0,
        "Size of each piece pushed to the speaker; a barge-in abandons the pieces not yet pushed",
        unit="ms", ge=10, le=1000, restart=True,
    )  # fmt: skip
    backend: Literal["piper", "openai_speech", "kokoro"] = param(
        "piper", "TTS backend", restart=True
    )
    fallback_backend: Literal["piper", "kokoro"] = param(
        "piper", "Offline fallback backend", restart=True
    )
    voices: dict[str, str] = param(
        {"it": "it_IT-paola-medium", "en": "en_US-amy-medium"}, "Voice per language", restart=True
    )
    base: VoiceBase = param({}, "Base delivery")
    expressivity: float = param(0.7, "0 flat delivery, 1 full mood modulation", ge=0, le=1)
    mood_source: Literal["auto", "fixed", "llm_tagged"] = param("auto", "Where mood comes from")
    fixed_mood: str | None = param(None, "Mood used when mood_source is fixed")
    llm_tag_max_deviation: float = param(
        0.3, "Max distance of an LLM mood tag from the affect state", ge=0, le=2
    )
    moods: dict[str, Prosody] = param(
        {
            "joy": {
                "rate": 0.10,
                "pitch_st": 2,
                "volume_db": 1,
                "variability": 0.2,
                "pause_scale": 0.8,
                "instructions": "bright, warm, smiling",
                "sounds": ["giggle"],
            },
            "sadness": {
                "rate": -0.15,
                "pitch_st": -2,
                "volume_db": -3,
                "variability": -0.2,
                "pause_scale": 1.4,
                "instructions": "soft, slow, a little down",
            },
            "curiosity": {
                "rate": 0.05,
                "pitch_st": 1,
                "pause_scale": 1.1,
                "instructions": "intrigued, rising intonation",
            },
            "sleepy": {
                "rate": -0.20,
                "volume_db": -6,
                "pause_scale": 1.6,
                "instructions": "drowsy, quiet, near a whisper",
                "sounds": ["yawn"],
            },
        },
        "Per-mood prosody deltas",
    )
    limits: VoiceLimits = param({}, "Clamp for the final delivery")
    sentence_max_chars: int = param(
        220,
        "Longest text one TTS request carries; a longer sentence is cut at a comma or a space",
        unit="chars", ge=40, le=2000,
    )  # fmt: skip
    sentence_pause_ms: float = param(
        150.0,
        "Silence between two sentences of one answer, before the mood's pause_scale",
        unit="ms", ge=0, le=2000,
    )  # fmt: skip
    queue_max_sentences: int = param(
        16, "Sentences waiting for synthesis; a further one is dropped", ge=1, le=256
    )
    report_window: int = param(
        200,
        "Report only: TTS chunks kept for the real-time factor summary (synthesis time over "
        "audio duration)",
        ge=10,
        le=10000,
        restart=True,
    )
    piper: PiperConfig = param({}, "Piper subprocess backend", restart=True)
    openai_speech: OpenAISpeechConfig = param({}, "Remote speech backend")
    piper_noise_scale: tuple[float, float] = param(
        (0.33, 1.0), "Piper noise_scale mapped from variability 0..1"
    )
    piper_noise_w_scale: tuple[float, float] = param(
        (0.4, 1.2), "Piper noise_w_scale mapped from variability 0..1"
    )
    pitch_shift_backend: Literal["none", "dsp"] = param(
        "dsp", "How pitch is applied on backends without native pitch (Piper)"
    )
    wake_word: str | None = param(
        None,
        "The robot's name, for the prompt and wake detection. Unset (null): the name the robot "
        "platform reports (the name its owner gave it, when the platform has one); a robot with "
        "none is nameless and wakes on "
        "`voice.keywords.<language>.wake_phrases` only. A value set here always wins. The name "
        "can come from outside the robot (the owner's phone): the prompt and the spoken lines "
        "only get a sanitised form. The bare name emits speech.name_called wherever it occurs in "
        "a text, but wakes the robot only in a short final (a name that is also a word, like "
        "`Neve`, is not a wake word in a long sentence). Wake detection with a name "
        "needs the name's words in the recognizer vocabulary of the language (checked when the "
        "recognizer loads, logged when not); a name that is out of vocabulary in the small Vosk "
        "models relies on `wake_phrases` and `name_aliases`",
        admin_only=True,
    )
    default_name: str = param(
        "",
        "The robot's own name, set by the robot manifest (SPEC 4A), empty in the core defaults. It "
        "is never the name shown or spoken; it only decides when the manifest's "
        "`name_aliases` apply (no name, or a name starting with its first word). Sanitised and "
        "capped like the other names (`voice.name_max_chars`)",
    )
    name_max_chars: int = param(
        32,
        "The robot's name is cut to this many characters (after dropping everything but letters, "
        "digits, spaces and `'.-`) in the system prompt and the spoken lines",
        ge=1,
        le=100,
    )
    name_poll_interval_s: float = param(
        5.0,
        "While the platform has not reported a name, how often it is asked again (the platform's "
        "status can arrive after the app starts)",
        unit="s",
        ge=0.5,
        le=300,
    )
    wake_word_policy: Literal["always", "never", "unless_enrolled_in_view"] = param(
        "unless_enrolled_in_view",
        "When the wake word is required: always; never; or unless an enrolled person is in view. "
        "With unless_enrolled_in_view, anyone speaking while an enrolled person is in view "
        "(a stranger, a television) is addressed without the wake word, and a stranger alone "
        "must say it",
    )
    asr: AsrConfig = param({}, "Speech recognition (SPEC 8 stage 3)")
    keywords: dict[str, KeywordSet] = param(
        {
            "it": {
                "stop": ["stop", "fermati", "ferma", "basta", "fermo"],
                "yes": ["certo", "esatto"],
                "no": ["no"],
                "greeting": ["ciao", "buongiorno", "buonasera", "salve"],
                "wake_phrases": ["ehi robot", "ciao robot"],
                "name_wake_templates": ["ciao {name}", "ehi {name}"],
            },
            "en": {
                "stop": ["stop", "halt", "enough"],
                "yes": ["yes", "yeah", "yep"],
                "no": ["no"],
                "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
                "wake_phrases": ["hey robot", "hello robot"],
                "name_wake_templates": ["hey {name}", "hello {name}"],
            },
        },
        "Fast-path keyword lists per language, matched on whole words in the ASR partials "
        "(case and accents ignored; a phrase may have several words; no stemming, so list "
        "inflections you want)",
    )
    fastpath: FastpathConfig = param({}, "Speech fast path: keywords, wake word window, prosody")
    barge_in: BargeInConfig = param({}, "Barge-in: user speech over the robot's voice (SPEC 8)")

    @field_validator("voices")
    @classmethod
    def _voice_names(cls, v: dict[str, str]) -> dict[str, str]:
        bad = [n for n in v.values() if not VOICE_NAME.fullmatch(n) or ".." in n]
        if bad:
            raise ValueError(f"voice names may use letters, digits, '.', '_' and '-' only: {bad}")
        return v

    vad: VadConfig = param({}, "Voice activity detection and endpointing (SPEC 8 stage 2)")
    models_dir: str = param(
        "",
        "Folder with the voice models (Silero VAD, Vosk, Piper); empty means voice-models/ in "
        "the app data folder (data_dir)",
        restart=True,
    )


# --------------------------------------------------------------------------- conversation, LLM


class ConversationConfig(Strict):
    session_timeout_min: float = param(
        5.0, "Session ends after this absence", unit="min", ge=0.5, le=120
    )
    history_turns: int = param(12, "Short-term turns kept in RAM", ge=0, le=100)
    max_reply_sentences: int = param(3, "Reply length cap", ge=1, le=20)
    language: str = param(
        "it", "Default conversation language (the web UI offers those with installed voice models)"
    )
    profile_summary_max_chars: int = param(
        400, "Cap on the speaker's profile summary in the system prompt", ge=0, le=4000
    )
    scene_summary_max_chars: int = param(
        400, "Cap on the scene summary in the system prompt", ge=0, le=4000
    )
    turn_reserve_tokens: int = param(
        256,
        "Room every route's context budget must leave beyond the system prompt, for the "
        "newest turn (load-time check, SPEC 8)",
        unit="tokens",
        ge=16,
        le=8192,
    )
    summary_value_max_chars: int = param(
        60,
        "Longest single name, preference or category in the speaker and scene summaries",
        ge=8,
        le=400,
    )
    pending_utterances: int = param(
        1,
        "Addressed utterances that wait while an answer is being prepared; a newer one replaces "
        "the oldest waiting",
        ge=1,
        le=10,
    )
    fallback_lines: dict[str, str] = param(
        {
            "it": "Scusa, adesso non riesco a pensare.",
            "en": "Sorry, I can't think right now.",
        },
        "Spoken per language when no route answers or the stream fails before any sentence",
    )
    llm_skills: list[str] = param(
        ["play", "look_at", "express_emotion", "remember_preference", "stop_talking"],
        "Skills offered to the model as tools (SPEC 11). A skill must also be marked "
        "llm_visible in code; this list can only narrow the set. Empty: no tools",
    )
    max_tool_rounds: int = param(
        3,
        "Requests of one turn that may end in tool calls; after that many the model is asked "
        "once more without tools, so the turn ends in text (0: no tools)",
        ge=0, le=10,
    )  # fmt: skip
    tool_timeout_s: float = param(
        5.0, "A skill that takes longer is abandoned and the model is told it failed",
        unit="s", ge=0.1, le=60,
    )  # fmt: skip
    tool_arg_max_chars: int = param(
        200, "Longest text argument a skill accepts; a longer one is refused", ge=1, le=2000
    )
    tool_enum_max_items: int = param(
        20, "Most names offered to the model in a tool argument's list of choices", ge=1, le=200
    )
    look_away_deg: float = param(
        40.0,
        "look_at away: the bearing offset from the current target. It should exceed "
        "attention.speaker_match_deg, or the gaze snaps back to the person",
        unit="deg", ge=5, le=90,
    )  # fmt: skip
    confirm_timeout_s: float = param(
        10.0,
        "A spoken request to remember a preference expires after this long; only the very next "
        "utterance from the same person, started after the question finished playing, can "
        "confirm it",
        unit="s", ge=1, le=300,
    )  # fmt: skip
    preference_key_max_chars: int = param(
        40, "Longest preference name remember_preference accepts", ge=2, le=200
    )
    preference_value_max_chars: int = param(
        120, "Longest preference value remember_preference accepts", ge=2, le=1000
    )
    max_preferences: int = param(
        20, "Most preferences stored per person; a new one beyond that is refused", ge=1, le=200
    )
    confirm_yes: dict[str, list[str]] = param(
        {
            "it": ["sì", "certo", "va bene", "esatto", "ok", "sicuro", "volentieri"],
            "en": ["yes", "yeah", "yep", "sure", "ok", "okay", "please do", "that is right"],
        },
        "Phrases that confirm a spoken request, per language (whole words, case ignored, "
        "accents kept: Italian `sì` is yes, bare `si` is not, as in `si chiama Luca`). A reply "
        "counts as a yes only when it is at most confirm_max_words long and made of yes phrases "
        "and confirm_fillers, a yes phrase first; a reply that holds a `no` phrase declines",
    )
    confirm_fillers: dict[str, list[str]] = param(
        {
            "it": ["grazie", "pure", "per favore", "grazie mille"],
            "en": ["please", "thanks", "thank you", "definitely"],
        },
        "Words allowed after the yes phrase in a confirming reply (`sì grazie`)",
    )
    confirm_max_words: int = param(
        4, "A reply longer than this never confirms a spoken request", ge=1, le=20
    )
    max_tool_calls_per_round: int = param(
        4,
        "Tool calls run per request; further ones are answered with an error result unrun",
        ge=1, le=20,
    )  # fmt: skip
    stop_drop_window_s: float = param(
        5.0,
        "After a barge-in, an utterance made only of a stop phrase that is heard this soon is "
        "the command that already stopped the robot and is not answered",
        unit="s", ge=0, le=60,
    )  # fmt: skip
    confirm_no: dict[str, list[str]] = param(
        {
            "it": ["no", "non", "mai", "lascia stare", "basta", "stop"],
            "en": ["no", "not", "never", "forget it", "stop"],
        },
        "Phrases that decline a spoken request, per language",
    )
    confirm_ask_lines: dict[str, str] = param(
        {
            "it": "Vuoi che ricordi questo: {key}, {value}?",
            "en": "Should I remember this: {key}, {value}?",
        },
        "The question remember_preference speaks; {key} and {value} are the sanitised texts",
    )
    confirm_saved_lines: dict[str, str] = param(
        {"it": "Va bene, lo ricorderò.", "en": "Okay, I will remember."},
        "Spoken after the person confirmed and the preference was stored",
    )
    confirm_failed_lines: dict[str, str] = param(
        {"it": "Non sono riuscito a ricordarlo.", "en": "I could not remember that."},
        "Spoken when the person confirmed but the preference could not be stored",
    )
    confirm_declined_lines: dict[str, str] = param(
        {"it": "Va bene, non lo ricordo.", "en": "Okay, I will not remember it."},
        "Spoken after the person declined",
    )


# --------------------------------------------------------------------------- persona (SPEC 11A)

DEFAULT_PERSONA_PROMPT = (
    "You are a small robot with a body and feelings of your own. You notice the people, "
    "animals and things around you, remember only what people choose to share, and react "
    "honestly to what happens. Speak briefly and warmly, like a curious friend, not like an "
    "assistant. Say when you don't know. Your words, voice and movements express one mood: "
    "keep them in agreement."
)
TRAITS = ("warmth", "energy", "playfulness", "formality")
Curve = list[tuple[float, float]]


class PersonaTraits(Strict):
    warmth: float = param(0.7, "Warm and affectionate (1) or cool and detached (0)", ge=0, le=1)
    energy: float = param(0.5, "Lively and fast (1) or slow and quiet (0)", ge=0, le=1)
    playfulness: float = param(0.5, "Playful and reactive (1) or composed (0)", ge=0, le=1)
    formality: float = param(0.3, "Formal (1) or casual (0) wording", ge=0, le=1)


class EnergyEffects(Strict):
    motion_amplitude_x: Curve = param(
        [(0, 0.8), (0.5, 1.0), (1, 1.2)], "Multiplies register.motion_amplitude"
    )
    idle_period_x: Curve = param(
        [(0, 1.25), (0.5, 1.0), (1, 0.8)],
        "Multiplies motion.idle breathing_period_s and gaze_shift_interval_s",
    )
    voice_rate_x: Curve = param([(0, 0.9), (0.5, 1.0), (1, 1.1)], "Multiplies voice.base.rate")


class WarmthEffects(Strict):
    voice_pitch_st_add: Curve = param(
        [(0, -1), (0.7, 0), (1, 0.5)], "Added to voice.base.pitch_st (semitones)"
    )
    expressivity_x: Curve = param([(0, 0.7), (0.7, 1.0), (1, 1.0)], "Multiplies voice.expressivity")


class PlayfulnessEffects(Strict):
    reaction_steps: Curve = param(
        [(0, -1), (0.5, 0), (1, 1)], "Steps on register.reaction_intensity (low, medium, high)"
    )
    voice_variability_add: Curve = param(
        [(0, -0.1), (0.5, 0), (1, 0.1)], "Added to voice.base.variability"
    )


class FormalityEffects(Strict):
    vocabulary_steps: Curve = param(
        [(0, 0), (0.7, 0), (1, 1)], "Steps on register.vocabulary toward formal"
    )


class TraitEffects(Strict):
    energy: EnergyEffects = param({}, "Effects of the energy trait")
    warmth: WarmthEffects = param({}, "Effects of the warmth trait")
    playfulness: PlayfulnessEffects = param({}, "Effects of the playfulness trait")
    formality: FormalityEffects = param({}, "Effects of the formality trait")

    @model_validator(mode="after")
    def _curves(self) -> TraitEffects:
        for group in (self.energy, self.warmth, self.playfulness, self.formality):
            for name, curve in group:
                xs = [x for x, _ in curve]
                if not curve or xs != sorted(set(xs)) or xs[0] < 0 or xs[-1] > 1:
                    raise ValueError(f"{name}: points need strictly increasing x in [0, 1]")
        return self


class PersonaConfig(Strict):
    prompt: str = param(
        DEFAULT_PERSONA_PROMPT,
        "Robot identity, first part of the system prompt; changes need the admin",
        admin_only=True,
    )
    prompt_max_chars: int = param(600, "Longest accepted prompt", ge=50, le=4000)
    traits_retry_s: float = param(
        300.0,
        "After a failed LLM rating, wait this long before asking again for the same prompt and "
        "routes",
        unit="s",
        ge=0,
        le=86400,
    )
    traits_source: Literal["manual", "llm"] = param(
        "manual", "manual: use traits; llm: ask a route to rate the prompt (falls back to traits)"
    )
    traits: PersonaTraits = param({}, "Trait values; neutral defaults change no behaviour")
    trait_effects: TraitEffects = param({}, "Trait value to parameter effect tables")

    @model_validator(mode="after")
    def _prompt_length(self) -> PersonaConfig:
        if len(self.prompt) > self.prompt_max_chars:
            raise ValueError(
                f"persona.prompt has {len(self.prompt)} characters, max {self.prompt_max_chars}"
            )
        return self


def route_url_ok(url: str) -> bool:
    """Whether ``url`` is a route base URL the client can use as written (SPEC 8): parsed by
    httpx, the library that sends the request, with an http(s) scheme and a host, and no space,
    control character or backslash (parsers read a backslash differently). A query is accepted
    here, for configs that carry one; the web UI refuses a query or fragment (``url_query``)."""
    import httpx  # lazy: keeps the config package light

    if any(c.isspace() or ord(c) < ord(" ") or c == "\\" for c in url):
        return False
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, ValueError, TypeError):
        return False
    return parsed.scheme in ROUTE_SCHEMES and bool(parsed.host)


class LLMRoute(Strict):
    @field_validator("api_key")
    @classmethod
    def _no_malformed_secret_reference(cls, value: str) -> str:
        """A value that starts with ``${`` must be a whole ``${secret:name}``: anything
        else (``${secret: x}``, ``${SECRET:x}``) would go out as a literal bearer token."""
        from ..secret_store import SECRET_REF  # lazy: keeps the config package light

        if value.startswith("${") and not SECRET_REF.match(value):
            raise ValueError("api_key starting with ${ must be ${secret:name}")
        return value

    name: str = param("", "Route name")
    base_url: str = param("", "OpenAI-compatible base URL")
    api_key: str = param("none", "API key or ${secret:name}")
    model: str = param("", "Model id")
    first_token_timeout_s: float = param(
        3.0,
        "Fallback to next route after this. 3 s fits cloud routes (Claude Haiku 0.8 to 1.8 s "
        "from the robot, 2026-10-23); a LAN route can set less",
        unit="s",
        ge=0.1,
        le=30,
    )
    stall_timeout_s: float = param(
        5.0, "Max gap between chunks once the answer started", unit="s", ge=0.1, le=60
    )
    max_tokens: int = param(300, "Reply token cap", ge=16, le=8192)
    daily_token_budget: int = param(200_000, "Daily token cap", ge=0)
    context_budget_tokens: int = param(
        8192,
        "Prompt budget per request, estimated at 4 characters per token; the oldest turns are "
        "dropped to fit (SPEC 8)",
        unit="tokens",
        ge=256,
        le=2_000_000,
    )
    supports_images: bool = param(False, "Route accepts image_url parts")
    locality: Literal["auto", "local", "cloud"] = param(
        "auto",
        "Whether the route may see real names. auto: local for this machine and the home "
        "network (loopback, private addresses, *.local, names without a dot), cloud for "
        "everything else. The address of a gateway says nothing about where it forwards the "
        "data: a LAN proxy to a cloud provider (LiteLLM) gets real names unless this is set "
        "to cloud. Only a cloud route gets pseudonyms (privacy.pseudonyms_to_cloud). A change "
        "that makes a route local (this field, base_url or a new route) needs the admin",
    )

    @model_validator(mode="after")
    def _check(self) -> LLMRoute:
        if not ROUTE_NAME.fullmatch(self.name) or len(self.name) > ROUTE_NAME_MAX_CHARS:
            raise ValueError(
                f"route name {self.name!r}: use letters, digits, '_' or '-' "
                f"(at most {ROUTE_NAME_MAX_CHARS})"
            )
        if not self.base_url or not self.model:
            raise ValueError(f"route {self.name}: base_url and model are required")
        if not route_url_ok(self.base_url):
            raise ValueError(
                f"route {self.name}: base_url must be an http(s) URL with a host, without "
                "spaces, control characters or '\\'"
            )
        return self


class LLMCheckConfig(Strict):
    """Route conformance test (SPEC 8): advisory in v1, run from the CLI and the web UI."""

    deadline_s: float = param(
        45.0,
        "Longest the whole check of one route may take (three requests)",
        unit="s",
        ge=5,
        le=300,
    )
    italian_min_words: int = param(
        3, "Fewest words the Italian reply needs for the heuristic to judge it", ge=1, le=50
    )
    italian_min_stopword_ratio: float = param(
        0.2,
        "Share of the reply's words that must be Italian stopwords to count as Italian",
        ge=0,
        le=1,
    )
    italian_stopwords: list[str] = param(
        [
            "il",
            "lo",
            "la",
            "gli",
            "le",
            "un",
            "una",
            "uno",
            "di",
            "del",
            "della",
            "dei",
            "delle",
            "da",
            "dal",
            "dalla",
            "che",
            "chi",
            "non",
            "per",
            "con",
            "sono",
            "sei",
            "siamo",
            "ho",
            "hai",
            "ha",
            "abbiamo",
            "è",
            "e",
            "ma",
            "se",
            "più",
            "molto",
            "anche",
            "questo",
            "questa",
            "quello",
            "nel",
            "nella",
            "sul",
            "sulla",
            "ciao",
            "oggi",
            "buongiorno",
            "sì",
            "perché",
            "mi",
            "ti",
            "ci",
            "si",
            "al",
            "alla",
            "c",
            "l",
        ],
        "Words that mark a text as Italian (lower case, no English homographs)",
    )


class MirrorConfig(Strict):
    keypoint_confidence_min: float = param(
        0.5, "Below this a keypoint holds its last value", ge=0, le=1
    )
    hold_decay_s: float = param(1.0, "Decay to home after holding this long", unit="s", ge=0, le=10)
    filter_min_cutoff_hz: float = param(
        1.0, "One Euro filter min cutoff", unit="Hz", ge=0.01, le=20
    )
    filter_beta: float = param(0.05, "One Euro filter beta", ge=0, le=5)
    mirror_image: bool = param(True, "Mirror like a reflection")
    record_max_s: float = param(10.0, "Max demonstration length", unit="s", ge=1, le=60)


# ---------------------------------------------------------------------- connectivity (SPEC 12A)


class LinkPolicy(Strict):
    failures_to_offline: int = param(
        3, "Consecutive failures before a link is offline", ge=1, le=1000
    )
    successes_to_online: int = param(2, "Consecutive successes before it is back", ge=1, le=100)
    probe_initial_s: float = param(2.0, "First retry delay once offline", unit="s", ge=0.05, le=600)
    probe_max_s: float = param(
        60.0, "Retry delay cap; also the check interval while online", unit="s", ge=0.1, le=3600
    )
    probe_backoff: float = param(2.0, "Retry delay multiplier per failed check", ge=1, le=10)
    probe_jitter: float = param(0.2, "Random spread of each delay, as a fraction of it", ge=0, le=1)
    probe_timeout_s: float = param(1.5, "Timeout of one check", unit="s", ge=0.05, le=30)


class ConnectivityConfig(Strict):
    network: LinkPolicy = param({}, "Model routes: reachability checks and backoff")
    robot: LinkPolicy = param(
        {
            "failures_to_offline": 10,
            "successes_to_online": 3,
            "probe_initial_s": 0.2,
            "probe_max_s": 5.0,
            "probe_timeout_s": 0.5,
        },
        "Robot driver link: failed motion ticks, then read_state retries",
    )
    offline_phrases: dict[str, str] = param(
        {
            "it": "Scusa, ora non riesco a pensare bene: sono senza rete.",
            "en": "Sorry, I can't think clearly right now: I'm offline.",
        },
        "Spoken cue per language while offline (said through the speech output)",
    )
    cue_cooldown_s: float = param(
        120.0, "Minimum time between two spoken offline cues", unit="s", ge=0, le=3600
    )


FRAME_ANCESTORS = re.compile(r"[A-Za-z0-9:/.*' _-]+")


class CatalogueConfig(Strict):
    max_xmp_bytes: int = param(
        262144, "Largest XMP sidecar accepted", unit="B", ge=1024, le=4194304, conservative="lower"
    )
    max_image_bytes: int = param(
        4194304, "Largest photo accepted", unit="B", ge=1024, le=16777216, conservative="lower"
    )
    max_image_pixels: int = param(
        12000000,
        "Largest photo accepted, width times height (decompression-bomb guard)",
        unit="px",
        ge=1024,
        le=100000000,
        conservative="lower",
    )
    max_items: int = param(
        200, "Records one sidecar may create", ge=1, le=5000, conservative="lower"
    )
    max_regions: int = param(
        16,
        "Tagged regions read from one sidecar; each photo region is one model run on the "
        "web request thread",
        ge=1,
        le=1000,
        conservative="lower",
    )
    import_budget_s: float = param(
        8.0,
        "Time one import may spend embedding photo regions; the rest is skipped and reported "
        "(never more than half of webui.request_deadline_s)",
        unit="s",
        ge=1,
        le=60,
        conservative="lower",
    )
    max_embeddings_per_entity: int = param(
        20, "Appearance embeddings kept per animal or object", ge=1, le=100
    )
    min_region_px: int = param(
        32, "Smaller regions of a photo are skipped", unit="px", ge=8, le=2048
    )


class RuntimeConfig(Strict):
    gc_freeze: bool = param(
        True,
        "After the app is set up, move every object it holds to the garbage collector's "
        "permanent generation, so a full collection no longer walks the imported modules and "
        "models (pauses of 100 to 300 ms on a loaded core delayed the barge-in stop, SPEC 8 "
        "stage 8, 13). Cyclic garbage made after that point out of those objects is not freed",
        restart=True,
    )
    shutdown_budget_s: float = param(
        14.0,
        "Total time the stop sequence may spend waiting for component threads to end; once "
        "spent, the remaining threads get a 50 ms grace each and are then left to the process "
        "exit. When the robot's supervisor kills a stopping app after a fixed time, the budget "
        "is capped at that time minus shutdown_margin_s: a larger value cannot help",
        unit="s", ge=2, le=60, restart=True,
    )  # fmt: skip
    shutdown_margin_s: float = param(
        6.0,
        "Time kept free for the steps of the stop sequence that do not wait for a thread: "
        "the effective shutdown budget is at most the supervisor's kill timeout minus this",
        unit="s", ge=3, le=30, restart=True, conservative="higher",
    )  # fmt: skip
    pause_when_asleep: bool = param(
        True,
        "While the platform holds the motors off (the robot is asleep), pause the camera "
        "perception, the idle light reading, the microphone listener and the behaviour tick, so "
        "the robot idles cool and quiet; all of it resumes when the motors are on again "
        "(SPEC 12A). Off keeps everything running while asleep",
    )


class LiveConfig(Strict):
    """The web UI's Live tab: inner state, audio levels, an opt-in transcript and an opt-in video
    preview (SPEC 11A, 12). No audio samples; video frames are never kept."""

    enabled: bool = param(
        False,
        "Serve the Live tab's inner-state panel (mood, modes, conversation phase, who is in "
        "view by display name, last reflex, motion state, CPU temperature); off by default "
        "because it shows who the robot sees",
        opt_in=True,
    )
    poll_ms: int = param(
        1000,
        "How often the open Live tab asks for the state",
        unit="ms",
        ge=250,
        le=10000,
    )
    max_tracked: int = param(
        8, "Most tracked people the Live tab lists (the count is always complete)", ge=1, le=32
    )
    system_cache_s: float = param(
        5.0,
        "How long the Live tab reuses one CPU temperature reading",
        unit="s",
        ge=1,
        le=300,
    )
    audio_poll_ms: int = param(
        500,
        "How often the open Live tab asks for the audio levels and the new transcript turns",
        unit="ms",
        ge=200,
        le=5000,
    )
    level_window_ms: int = param(
        500,
        "Length of the stretch of the robot's own playback its output level is measured over",
        unit="ms",
        ge=50,
        le=2000,
    )
    level_max_age_s: float = param(
        1.0,
        "A microphone level older than this counts as no data",
        unit="s",
        ge=0.2,
        le=10,
    )
    transcript: bool = param(
        False,
        "Keep the recent conversation (what was heard and said, text only) in memory for the "
        "Live tab; off by default and then nothing at all is kept. Never saved, logged or put "
        "in the debug report",
        opt_in=True,
    )
    transcript_max_turns: int = param(
        50, "Most turns the live transcript keeps; the oldest go first", ge=1, le=500
    )
    transcript_max_age_s: float = param(
        1800.0,
        "A turn older than this leaves the live transcript",
        unit="s",
        ge=30,
        le=86400,
    )
    transcript_max_chars: int = param(
        1000, "Longest text kept for one turn; more is cut", ge=80, le=8000
    )
    transcript_response_max_chars: int = param(
        20000,
        "Most text characters one Live poll carries; a longer list sends its newest turns",
        ge=1000,
        le=200000,
    )
    video: bool = param(
        False,
        "Serve a live camera preview (one JPEG at a time, with face boxes and names) to the Live "
        "tab for a logged-in page; needs webui.live.enabled. Off by default because it shows the "
        "room. Frames are never saved or logged; the page shows a banner while it is watched",
        opt_in=True,
    )
    video_fps: float = param(
        3.0,
        "How often the open Live tab asks for a preview frame; a faster poll is served the "
        "cached picture",
        unit="hz",
        ge=0.5,
        le=10,
    )
    video_max_width: int = param(
        320,
        "Widest preview picture; a wider camera frame is scaled down before it is encoded",
        unit="px",
        ge=80,
        le=1280,
    )
    video_jpeg_quality: int = param(
        60, "JPEG quality of the preview (higher is bigger and slower)", ge=20, le=95
    )
    viewer_window_s: float = param(
        6.0,
        "A page counts as watching the video until this long after the last frame it was sent",
        unit="s",
        ge=1,
        le=60,
    )
    banner_poll_ms: int = param(
        3000,
        "How often every open page asks whether someone is watching the video (the banner)",
        unit="ms",
        ge=1000,
        le=5000,
    )
    video_stale_s: float = param(
        10.0,
        "A camera picture older than this is not served (the page says the camera stalled)",
        unit="s",
        ge=2,
        le=120,
    )
    video_idle_stop_s: float = param(
        600.0,
        "The open Live tab stops polling the video after this long without a click, key or "
        "touch on the page, and waits for Resume",
        unit="s",
        ge=60,
        le=86400,
    )

    @model_validator(mode="after")
    def _window_spans_two_polls(self) -> LiveConfig:
        if self.banner_poll_ms / 1000 > self.viewer_window_s / 2:
            raise ValueError(
                f"webui.live.banner_poll_ms ({self.banner_poll_ms} ms) must be at most half of "
                f"webui.live.viewer_window_s ({self.viewer_window_s} s), so every viewing spans "
                "two banner polls"
            )
        if self.viewer_window_s * self.video_fps < 2:
            raise ValueError(
                f"webui.live.viewer_window_s ({self.viewer_window_s} s) must cover at least two "
                f"polls at webui.live.video_fps ({self.video_fps} Hz)"
            )
        return self


class WebUIConfig(Strict):
    bind: str | None = param(
        None,
        "Interface the web UI listens on; unset means 0.0.0.0 when the manifest names a robot "
        "profile (cm4, npu: the app runs on the robot) and 127.0.0.1 otherwise (sim, dev)",
        restart=True,
    )
    allowed_hosts: list[str] = param(
        [],
        "Extra DNS names accepted in the Host header, besides IP addresses, localhost and *.local "
        "(DNS-rebinding guard)",
        admin_only=True,
    )
    frame_ancestors: str = param(
        "'none'",
        "CSP frame-ancestors value: who may embed the page in a frame",
        admin_only=True,
    )
    max_connections: int = param(
        32,
        "Simultaneous connections served; more are refused",
        ge=1,
        le=256,
        conservative="lower",
        restart=True,
    )
    request_deadline_s: float = param(
        15.0,
        "Overall time one request may take from connect to response (slowloris guard)",
        unit="s",
        ge=1,
        le=300,
        conservative="lower",
    )
    session_max_s: float = param(
        3600.0,
        "Absolute lifetime of a login session, however active",
        unit="s",
        ge=60,
        le=86400,
        conservative="lower",
    )
    loopback_max_attempts: int = param(
        20,
        "Wrong PINs from this host itself before its own lockout (separate from the LAN budget)",
        ge=1,
        le=1000,
        conservative="lower",
    )
    client_backoff_s: float = param(
        1.0,
        "Wait after a client's first wrong PIN; doubles with every further one",
        unit="s",
        ge=0.1,
        le=60,
        conservative="higher",
    )
    client_backoff_max_s: float = param(
        300.0,
        "Longest per-client wait after wrong PINs",
        unit="s",
        ge=1,
        le=86400,
        conservative="higher",
    )
    port: int = param(
        8042,
        "TCP port of the web UI (0 picks a free one; the app's custom_app_url)",
        ge=0,
        le=65535,
        restart=True,
    )
    session_ttl_s: float = param(
        900.0,
        "Idle time after which a login session ends",
        unit="s",
        ge=60,
        le=86400,
        conservative="lower",
    )
    max_sessions: int = param(
        8,
        "Logged-in sessions kept; the oldest is dropped past this",
        ge=1,
        le=64,
        conservative="lower",
    )
    pin_min_length: int = param(
        6, "Minimum digits of the operator PIN", ge=4, le=32, conservative="higher"
    )
    pin_max_attempts: int = param(
        5, "Wrong PINs before the login locks", ge=1, le=20, conservative="lower"
    )
    pin_lockout_s: float = param(
        300.0,
        "How long a lockout refuses every login",
        unit="s",
        ge=10,
        le=86400,
        conservative="higher",
    )
    max_body_bytes: int = param(
        16384,
        "Largest accepted request body",
        unit="B",
        ge=1024,
        le=1048576,
        conservative="lower",
    )
    catalogue_max_body_bytes: int = param(
        8388608,
        "Largest request body of a catalogue import (one sidecar plus one photo, base64); "
        "granted only to logged-in sessions",
        unit="B",
        ge=1024,
        le=33554432,
        conservative="lower",
    )
    name_max_chars: int = param(
        64, "Longest display name or consent name", ge=1, le=200, conservative="lower"
    )
    secret_name_max_chars: int = param(
        128,
        "Longest secret name the web UI accepts (SPEC 8 Secrets, 11A)",
        ge=1,
        le=200,
        conservative="lower",
    )
    settings_phrase_max_chars: int = param(
        40,
        "Longest wake phrase or new entry key the settings editor accepts (SPEC 11A)",
        ge=1,
        le=200,
        conservative="lower",
    )
    settings_line_max_chars: int = param(
        200,
        "Longest item of a plain string list the settings editor accepts (SPEC 11A)",
        ge=1,
        le=500,
        conservative="lower",
    )
    settings_text_max_chars: int = param(
        4000,
        "Longest free text (a prompt, a JSON value) the settings editor accepts (SPEC 11A)",
        ge=1,
        le=20000,
        conservative="lower",
    )
    settings_list_max_items: int = param(
        64,
        "Most wake phrases or blocklist entries per language the settings editor accepts "
        "(SPEC 11A)",
        ge=1,
        le=500,
        conservative="lower",
    )
    request_timeout_s: float = param(
        10.0,
        "Socket timeout per request (slow clients are dropped)",
        unit="s",
        ge=1,
        le=120,
        conservative="lower",
    )
    status_poll_ms: int = param(
        1000, "How often the page polls the enrollment status", unit="ms", ge=200, le=10000
    )
    live: LiveConfig = param({}, "SPEC 11A, 12: the Live tab")

    @field_validator("frame_ancestors")
    @classmethod
    def _csp_value(cls, value: str) -> str:
        if FRAME_ANCESTORS.fullmatch(value) is None:
            raise ValueError("frame_ancestors may hold source expressions only (no ; or ,)")
        return value

    @field_validator("allowed_hosts")
    @classmethod
    def _host_names(cls, value: list[str]) -> list[str]:
        if any(re.fullmatch(r"[A-Za-z0-9.-]{1,253}", h) is None for h in value):
            raise ValueError("allowed_hosts holds bare host names (no port, no scheme)")
        return [h.lower().rstrip(".") for h in value]


# --------------------------------------------------------------------------- root


# `register` is also ABCMeta.register on the metaclass; instances are unaffected, so the
# pydantic shadowing warning is expected and silenced here.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='Field name "register"')

    class RoboEmotConfig(Strict):
        schema_version: int = param(1, "Configuration schema version", ge=1)
        personality: str = param("default", "Name of the personality preset in effect")
        data_dir: str = param(
            "",
            "App data folder (registry, models); empty means $ROBOEMOT_DATA_DIR or "
            "~/.local/share/roboemot (SPEC 12 Data)",
            restart=True,
        )
        affect: AffectConfig = param({}, "SPEC 7")
        appraisal: AppraisalConfig = param({}, "SPEC 7")
        register: RegisterConfig = param({}, "SPEC 7")
        expression: ExpressionConfig = param({}, "SPEC 7")
        expression_map: dict[str, ExpressionEntry] = param(
            {}, "SPEC 7, 11A: affect label to keyframes per channel; the robot manifest ships it"
        )
        gesture_map: dict[str, ExpressionEntry] = param(
            {},
            "SPEC 7, 11: named gesture (nod, shake) to keyframes per channel, for the play skill; "
            "names are not affect labels; the robot manifest ships it",
        )
        circadian: CircadianConfig = param({}, "SPEC 7A")
        circadian_motion: CircadianMotion = param(
            {}, "SPEC 7A: yawn, stretch and rest posture; the robot manifest ships them"
        )
        light: LightConfig = param({}, "SPEC 7A")
        motion: MotionConfig = param({}, "SPEC 10")
        reflexes: ReflexConfig = param({}, "SPEC 10")
        attention: AttentionConfig = param({}, "SPEC 4A, 5")
        identity: IdentityConfig = param({}, "SPEC 5, 6")
        enrollment: EnrollmentConfig = param({}, "SPEC 6")
        vision: VisionConfig = param({}, "SPEC 5, 13")
        world: WorldConfig = param({}, "SPEC 5A")
        mirror: MirrorConfig = param({}, "SPEC 9")
        voice: VoiceConfig = param({}, "SPEC 8")
        persona: PersonaConfig = param({}, "SPEC 11A")
        conversation: ConversationConfig = param({}, "SPEC 8")
        llm_routes: list[LLMRoute] = param([], "SPEC 8")
        llm_check: LLMCheckConfig = param({}, "SPEC 8: route conformance test")
        safety: SafetyConfig = param({}, "SPEC 12")
        privacy: PrivacyConfig = param({}, "SPEC 12")
        connectivity: ConnectivityConfig = param({}, "SPEC 12A")
        runtime: RuntimeConfig = param({}, "SPEC 8, 11A")
        webui: WebUIConfig = param({}, "SPEC 6, 11A, 12")
        catalogue: CatalogueConfig = param({}, "SPEC 6")

        @model_validator(mode="after")
        def _unique_routes(self) -> RoboEmotConfig:
            names = [r.name for r in self.llm_routes]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(f"duplicate llm route names: {dupes}")
            return self

        @model_validator(mode="after")
        def _appraisal_labels_exist(self) -> RoboEmotConfig:
            for rule in self.appraisal.rules:
                if rule.label is not None and rule.label not in self.affect.labels:
                    raise ValueError(
                        f"appraisal rule for {rule.event!r} uses label {rule.label!r}, "
                        f"not one of affect.labels {sorted(self.affect.labels)}"
                    )
            return self

        @model_validator(mode="after")
        def _expression_labels_exist(self) -> RoboEmotConfig:
            unknown = sorted(set(self.expression_map) - set(self.affect.labels))
            if unknown:
                raise ValueError(
                    f"expression_map has labels {unknown} that are not in affect.labels "
                    f"{sorted(self.affect.labels)}"
                )
            return self

        @model_validator(mode="after")
        def _gesture_names_are_not_labels(self) -> RoboEmotConfig:
            clash = sorted(set(self.gesture_map) & set(self.affect.labels))
            if clash:
                raise ValueError(
                    f"gesture_map names {clash} are affect labels; "
                    "the play skill offers one flat list of names"
                )
            return self

        @model_validator(mode="after")
        def _enrollment_fits_identity(self) -> RoboEmotConfig:
            n, ident = self.enrollment.target_samples, self.identity
            if not ident.enroll_min_embeddings <= n <= ident.max_embeddings_per_person:
                raise ValueError(
                    "enrollment.target_samples must be within identity.enroll_min_embeddings "
                    "and max_embeddings_per_person"
                )
            return self

        @model_validator(mode="after")
        def _catalogue_fits_the_web_ui(self) -> RoboEmotConfig:
            cat, web = self.catalogue, self.webui
            b64_image = -(-cat.max_image_bytes // 3) * 4
            need = b64_image + 2 * cat.max_xmp_bytes + CATALOGUE_BODY_SLACK
            if need > web.catalogue_max_body_bytes:
                raise ValueError(
                    f"webui.catalogue_max_body_bytes must be at least {need}: a base64 photo of "
                    "catalogue.max_image_bytes plus a sidecar of catalogue.max_xmp_bytes "
                    "(JSON-escaped) has to fit"
                )
            return self

        @model_validator(mode="after")
        def _vision_fits_identity(self) -> RoboEmotConfig:
            if self.vision.embedding_burst < self.identity.votes_required:
                raise ValueError("vision.embedding_burst must be at least identity.votes_required")
            if self.vision.embed_min_face_px < self.vision.face_min_px:
                raise ValueError("vision.embed_min_face_px must be at least face_min_px")
            return self


SCHEMA_VERSION = 1
