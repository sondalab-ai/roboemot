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

    model_config = ConfigDict(extra="forbid", validate_assignment=True, validate_default=True)


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


class AppraisalRule(Strict):
    event: str = param("", "Bus event name that triggers the rule")
    valence: float = param(0.0, "Valence impulse", ge=-1, le=1)
    arousal: float = param(0.0, "Arousal impulse", ge=-1, le=1)
    label: str | None = param(None, "Label forced for event_label_hold_s, if any")
    cooldown_s: float = param(5.0, "Minimum time between two firings", unit="s", ge=0, le=3600)


class AppraisalConfig(Strict):
    rules: list[AppraisalRule] = param(
        [
            {"event": "person.recognized", "valence": 0.4, "arousal": 0.2, "cooldown_s": 60},
            {"event": "sound.loud", "arousal": 0.6, "label": "surprise", "cooldown_s": 5},
            {"event": "idle.long", "arousal": -0.6, "label": "sleepy", "cooldown_s": 600},
            {"event": "light.on", "arousal": 0.3, "label": "surprise", "cooldown_s": 30},
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
        ],
        "Event to impulse rules (SPEC 7, Affect state)",
    )


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
    minor_content_policy: bool = param(False, "Apply the minor content policy", conservative="true")


class RegisterConfig(Strict):
    child: RegisterParams = param(
        {
            "speech_rate": 0.9,
            "pitch_st": 2.0,
            "vocabulary": "simple",
            "persona_fragment": "Playful, encouraging, child-safe.",
            "motion_amplitude": 1.2,
            "approach_allowed": False,
            "reaction_intensity": "high",
            "minor_content_policy": True,
        },
        "Register for children",
    )
    teen: RegisterParams = param({"minor_content_policy": True}, "Register for teens")
    adult: RegisterParams = param({}, "Register for adults")
    senior: RegisterParams = param({}, "Register for seniors")
    unknown: RegisterParams = param(
        {
            "vocabulary": "formal",
            "persona_fragment": "Polite, reserved.",
            "motion_amplitude": 0.7,
            "approach_allowed": False,
            "reaction_intensity": "low",
            "minor_content_policy": True,
        },
        "Register for people of unknown age group (treated as possible minors)",
    )
    gender_affects_params: bool = param(
        False, "If true, gender may change register parameters, not only forms of address"
    )


# --------------------------------------------------------------------------- circadian (SPEC 7A)


class StateEffects(Strict):
    arousal_baseline: float = param(0.0, "Affect arousal baseline in this state", ge=-1, le=1)
    idle_motion_scale: float = param(1.0, "Idle motion amplitude multiplier", ge=0, le=2)
    speech_rate: float = param(1.0, "Speech rate multiplier", ge=0.5, le=1.5)
    volume_db: float = param(0.0, "Volume offset", unit="dB", ge=-24, le=6)
    vision_rate_scale: float = param(1.0, "Vision rate multiplier", ge=0, le=1)


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
    states: dict[str, StateEffects] = param(
        {
            "awake": {"arousal_baseline": 0.2},
            "drowsy": {
                "arousal_baseline": -0.3,
                "idle_motion_scale": 0.6,
                "speech_rate": 0.9,
                "volume_db": -3.0,
                "vision_rate_scale": 0.5,
            },
            "resting": {
                "arousal_baseline": -0.7,
                "idle_motion_scale": 0.0,
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


# --------------------------------------------------------------------------- motion (SPEC 10)


class ArbiterConfig(Strict):
    loop_hz: float = param(
        100.0, "Arbiter rate (50 towards the Reachy Mini daemon)", unit="Hz", ge=10, le=500
    )
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


class MotionConfig(Strict):
    arbiter: ArbiterConfig = param({}, "Arbiter parameters")
    idle: IdleConfig = param({}, "Idle motion parameters")


# --------------------------------------------------------------------------- safety (SPEC 12)


class SafetyConfig(Strict):
    extra_prompt_rules: list[str] = param(
        [],
        "Rules appended to the system prompt's safety block (SPEC 8); changes need the admin",
        admin_only=True,
    )
    minor_motion_factor: float = param(
        0.6,
        "Speed and amplitude factor while a minor is present",
        ge=0.05,
        le=1,
        conservative="lower",
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
    minors_allowed_llm: bool = param(
        False,
        "Allow conversation while a minor is present on routes cleared for minors",
        conservative="false",
    )


class PrivacyConfig(Strict):
    multimodal_frames: bool = param(
        False, "Allow single-frame image questions to LLM routes", opt_in=True
    )
    pseudonyms_to_cloud: bool = param(False, "Send pseudonyms instead of names to cloud routes")
    keep_enrollment_images: bool = param(False, "Keep enrollment face crops", opt_in=True)
    debug_recording: bool = param(False, "Allow time-limited debug recording", opt_in=True)


# --------------------------------------------------------------------------- identity (SPEC 5, 6)


class IdentityConfig(Strict):
    match_threshold: float = param(0.45, "Cosine similarity threshold", ge=0, le=1)
    margin: float = param(0.1, "Required margin over the second best match", ge=0, le=1)
    votes_required: int = param(3, "Positive evaluations required", ge=1, le=20)
    votes_window: int = param(5, "Evaluations considered", ge=1, le=50)
    estimate_confidence_min: float = param(
        0.8, "Below this, age/gender estimates are unknown", ge=0, le=1
    )
    estimate_frames_min: int = param(
        3,
        "Frames aggregated before an estimate is used; vision.attribute_crops must be at "
        "least this",
        ge=1,
        le=100,
    )
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


class TrackerConfig(Strict):
    high_score: float = param(
        0.6, "Detections at or above this start tracks (the detection score threshold)", ge=0, le=1
    )
    low_score: float = param(
        0.1, "Detections between this and high_score only keep existing tracks", ge=0, le=1
    )
    match_iou: float = param(0.3, "Minimum IoU for the first association", ge=0, le=1)
    low_match_iou: float = param(0.5, "Minimum IoU for the low-score association", ge=0, le=1)
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
    attribute_threads: int = param(
        1, "CPU threads of the age and gender models (background job)", ge=1, le=8, restart=True
    )
    detector_width_px: int = param(
        320, "Face detector input width (SPEC 13: 320x240)", unit="px", ge=64, le=1280, restart=True
    )
    detector_height_px: int = param(
        240, "Face detector input height", unit="px", ge=64, le=1280, restart=True
    )
    face_min_px: int = param(
        24, "Faces with a shorter box side are ignored", unit="px", ge=8, le=512
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
        2, "Cap on embeddings computed for one frame, to bound its latency", ge=1, le=16
    )
    attribute_precision: Literal["auto", "int8", "fp32"] = param(
        "auto",
        "Age and gender model files: int8 (dynamic INT8, about 87 MB each), fp32 (343 MB each) "
        "or auto (int8 when the robot manifest profile is cm4, fp32 otherwise)",
        restart=True,
    )
    attribute_crops: int = param(
        3,
        "Face crops per track for the age and gender job (SPEC 5); at least "
        "identity.estimate_frames_min, and the 0.8 confidence rule applies at any count",
        ge=1,
        le=10,
    )
    attribute_crop_interval_s: float = param(
        0.3,
        "Minimum capture-time gap between the face crops kept for the age and gender job",
        unit="s",
        ge=0,
        le=10,
    )
    attribute_crop_margin: float = param(
        0.25, "Face box padding for the age and gender crop, per side", ge=0, le=1
    )
    attribute_queue_max: int = param(
        4,
        "Age and gender jobs waiting for the worker; further tracks wait",
        ge=1,
        le=64,
        restart=True,
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
        "An animal or object track unmatched for this long is removed; objects run at "
        "vision.rates.objects_hz (1 Hz), so the face value of tracker.lost_s would drop a "
        "track after one missed frame",
        unit="s",
        ge=0.1,
        le=300,
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

    @model_validator(mode="after")
    def _disjoint_categories(self) -> VisionConfig:
        both = set(self.animal_categories) & set(self.object_categories)
        if both:
            raise ValueError(f"vision categories are both animal and object: {sorted(both)}")
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


class VoiceConfig(Strict):
    backend: Literal["piper", "openai_speech", "kokoro"] = param(
        "piper", "TTS backend", restart=True
    )
    fallback_backend: Literal["piper", "kokoro"] = param("piper", "Offline fallback backend")
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
    piper_noise_scale: tuple[float, float] = param(
        (0.33, 1.0), "Piper noise_scale mapped from variability 0..1"
    )
    piper_noise_w_scale: tuple[float, float] = param(
        (0.4, 1.2), "Piper noise_w_scale mapped from variability 0..1"
    )
    pitch_shift_backend: Literal["none", "dsp"] = param(
        "dsp", "How pitch is applied on backends without native pitch (Piper)"
    )
    wake_word: str = param("reachy", "Wake word")
    wake_word_policy: Literal["always", "never", "unless_enrolled_in_view"] = param(
        "unless_enrolled_in_view", "When the wake word is required"
    )
    active_asr_language: str = param("it", "ASR language when the speaker has no preference")


# --------------------------------------------------------------------------- conversation, LLM


class ConversationConfig(Strict):
    session_timeout_min: float = param(
        5.0, "Session ends after this absence", unit="min", ge=0.5, le=120
    )
    history_turns: int = param(12, "Short-term turns kept in RAM", ge=0, le=100)
    max_reply_sentences: int = param(3, "Reply length cap", ge=1, le=20)
    language: str = param("it", "Default conversation language")
    profile_summary_max_chars: int = param(
        400, "Cap on the speaker's profile summary in the system prompt", ge=0, le=4000
    )
    scene_summary_max_chars: int = param(
        400, "Cap on the scene summary in the system prompt", ge=0, le=4000
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
        [(0, 0.8), (0.5, 1.0), (1, 1.2)], "Multiplies register.*.motion_amplitude"
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
        [(0, -1), (0.5, 0), (1, 1)], "Steps on register.*.reaction_intensity (low, medium, high)"
    )
    voice_variability_add: Curve = param(
        [(0, -0.1), (0.5, 0), (1, 0.1)], "Added to voice.base.variability"
    )


class FormalityEffects(Strict):
    vocabulary_steps: Curve = param(
        [(0, 0), (0.7, 0), (1, 1)], "Steps on register.*.vocabulary toward formal"
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


class LLMRoute(Strict):
    name: str = param("", "Route name")
    base_url: str = param("", "OpenAI-compatible base URL")
    api_key: str = param("none", "API key or ${secret:name}")
    model: str = param("", "Model id")
    first_token_timeout_s: float = param(
        1.5, "Fallback to next route after this", unit="s", ge=0.1, le=30
    )
    stall_timeout_s: float = param(
        5.0, "Max gap between chunks once the answer started", unit="s", ge=0.1, le=60
    )
    max_tokens: int = param(300, "Reply token cap", ge=16, le=8192)
    allowed_for_minors: bool = param(
        False, "Route cleared for use while a minor is present", conservative="false"
    )
    daily_token_budget: int = param(200_000, "Daily token cap", ge=0)
    supports_images: bool = param(False, "Route accepts image_url parts")

    @model_validator(mode="after")
    def _check(self) -> LLMRoute:
        if not ROUTE_NAME.fullmatch(self.name):
            raise ValueError(f"route name {self.name!r}: use letters, digits, '_' or '-'")
        if not self.base_url or not self.model:
            raise ValueError(f"route {self.name}: base_url and model are required")
        return self


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


class GestureConfig(Strict):
    duration_s: float = param(1.2, "Length of a connectivity gesture", unit="s", ge=0.2, le=5)
    head_roll_rad: float = param(0.15, "Puzzled head tilt (roll)", unit="rad", ge=0, le=0.5)
    head_pitch_rad: float = param(
        0.08, "Perk-up head lift (pitch, negative is up)", unit="rad", ge=0, le=0.3
    )
    antenna_rad: float = param(
        0.4, "Antenna droop (puzzled) or lift (perk up)", unit="rad", ge=0, le=1.5
    )


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
    gestures: GestureConfig = param({}, "Nonverbal replies while offline and on recovery")
    offline_phrases: dict[str, str] = param(
        {
            "it": "Scusa, ora non riesco a pensare bene: sono senza rete.",
            "en": "Sorry, I can't think clearly right now: I'm offline.",
        },
        "Spoken cue per language while offline (used once speech output is wired)",
    )
    cue_cooldown_s: float = param(
        120.0, "Minimum time between two spoken offline cues", unit="s", ge=0, le=3600
    )


FRAME_ANCESTORS = re.compile(r"[A-Za-z0-9:/.*' _-]+")


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
        32, "Simultaneous connections served; more are refused", ge=1, le=256, conservative="lower"
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
    name_max_chars: int = param(
        64, "Longest display name or consent name", ge=1, le=200, conservative="lower"
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
        circadian: CircadianConfig = param({}, "SPEC 7A")
        light: LightConfig = param({}, "SPEC 7A")
        motion: MotionConfig = param({}, "SPEC 10")
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
        safety: SafetyConfig = param({}, "SPEC 12")
        privacy: PrivacyConfig = param({}, "SPEC 12")
        connectivity: ConnectivityConfig = param({}, "SPEC 12A")
        webui: WebUIConfig = param({}, "SPEC 6, 11A, 12")

        @model_validator(mode="after")
        def _unique_routes(self) -> RoboEmotConfig:
            names = [r.name for r in self.llm_routes]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(f"duplicate llm route names: {dupes}")
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
        def _vision_fits_identity(self) -> RoboEmotConfig:
            if self.vision.embedding_burst < self.identity.votes_required:
                raise ValueError("vision.embedding_burst must be at least identity.votes_required")
            if self.vision.attribute_crops < self.identity.estimate_frames_min:
                raise ValueError(
                    "vision.attribute_crops must be at least identity.estimate_frames_min"
                )
            if self.vision.embed_min_face_px < self.vision.face_min_px:
                raise ValueError("vision.embed_min_face_px must be at least face_min_px")
            return self


SCHEMA_VERSION = 1
