"""Robot manifest: parsing, validation and capability negotiation (SPEC 4, 4A, 12).

YAML uses degrees and millimetres for humans; everything is converted to radians and metres
at load. Each controllable scalar becomes a :class:`Dof` with position, velocity and
acceleration limits. A manifest without limits is rejected (SPEC 12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config.schema import SafetyConfig


class ManifestError(ValueError):
    """The manifest is invalid or unsafe."""


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JointLimits(_M):
    min_deg: float
    max_deg: float
    max_vel_dps: float | None = Field(None, gt=0)
    max_acc_dps2: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> JointLimits:
        if self.min_deg >= self.max_deg:
            raise ValueError("min_deg must be < max_deg")
        return self


class Joint(_M):
    name: str
    driver: str
    channel: int | None = None
    limits: JointLimits
    home_deg: float = 0.0
    compliance: Literal["spring_return", "hold", "limp"] = "hold"


class Pose6dLimits(_M):
    pitch_deg: tuple[float, float]
    roll_deg: tuple[float, float]
    yaw_deg: tuple[float, float]
    x_mm: tuple[float, float] | None = None
    y_mm: tuple[float, float] | None = None
    z_mm: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Pose6dLimits:
        for name in ("pitch_deg", "roll_deg", "yaw_deg", "x_mm", "y_mm", "z_mm"):
            rng = getattr(self, name)
            if rng is not None and rng[0] >= rng[1]:
                raise ValueError(f"{name}: lower bound must be < upper bound")
        return self


class PoseGroup(_M):
    type: Literal["pose6d", "joint", "joint_pair"]
    limits: Pose6dLimits | JointLimits | None = None
    names: list[str] | None = None
    home_deg: list[float] | None = None

    @model_validator(mode="after")
    def _check(self) -> PoseGroup:
        if self.type == "pose6d" and not isinstance(self.limits, Pose6dLimits):
            raise ValueError("pose6d needs pitch/roll/yaw limits")
        if self.type == "joint_pair" and (not self.names or len(self.names) != 2):
            raise ValueError("joint_pair needs exactly two names")
        return self


class Dynamics(_M):
    max_vel_dps: float | None = Field(None, gt=0)
    max_acc_dps2: float | None = Field(None, gt=0)
    max_lin_vel_mps: float | None = Field(None, gt=0)
    max_lin_acc_mps2: float | None = Field(None, gt=0)


class Constraint(_M):
    type: Literal["yaw_delta"]
    a: str
    b: str
    max_deg: float = Field(gt=0)


class Retarget(_M):
    source: str
    scale: float = 1.0
    offset_deg: float = 0.0


class Safety(_M):
    estop: dict[str, Any] | Literal["none"] = "none"
    soft_stop: Literal["hold_position"] = "hold_position"
    hard_stop: Literal["disable_motors"] = "disable_motors"
    watchdog_ms: float = Field(200, gt=0)


class Discovery(_M):
    """How the robot announces itself on the LAN (SPEC 4, Discovery)."""

    mdns_service: str = Field(pattern=r"^_[a-z0-9-]+\._(tcp|udp)$")
    match_txt: dict[str, str] = {}  # TXT values that must match, e.g. {model: ...}


class RobotInfo(_M):
    id: str
    driver: str | None = None
    profile: Literal["cm4", "npu"] | None = None
    frame: str | None = None
    connection: dict[str, Any] | None = None
    discovery: Discovery | None = None


class ManifestModel(_M):
    robot: RobotInfo
    joints: list[Joint] = []
    pose_groups: dict[str, PoseGroup] = {}
    constraints: list[Constraint] = []
    dynamics: dict[str, Dynamics] = {}
    sensors: dict[str, Any] = {}
    expressive_channels: dict[str, dict[str, Any]] = {}
    retargeting: dict[str, Retarget] = {}
    safety: Safety = Safety()
    motor_modes: dict[str, str] = {}
    kinematics: str | None = None
    behaviour: dict[str, Any] = {}  # config layer 2 (SPEC 11A)


@dataclass(frozen=True)
class Dof:
    """One controllable scalar. Angles in rad, translations in m."""

    name: str
    group: str
    min: float
    max: float
    max_vel: float
    max_acc: float
    home: float = 0.0
    measured: bool = True
    kind: Literal["angle", "translation"] = "angle"


@dataclass
class Manifest:
    model: ManifestModel
    dofs: dict[str, Dof]
    constraints: list[Constraint] = field(default_factory=list)

    @property
    def robot(self) -> RobotInfo:
        return self.model.robot

    @property
    def channels(self) -> frozenset[str]:
        """Names an expression, gesture or rest pose may drive: the angle dofs (SPEC 7)."""
        return frozenset(n for n, d in self.dofs.items() if d.kind == "angle")

    def capabilities(self) -> set[str]:
        m = self.model
        caps: set[str] = set()
        sensor_types = {k: v for k, v in m.sensors.items()}
        if any(k.startswith("camera") for k in sensor_types):
            caps.add("camera")
        if "mic_array" in sensor_types or any(k.startswith("mic") for k in sensor_types):
            caps.add("microphone")
            mic = sensor_types.get("mic_array")
            if isinstance(mic, dict) and mic.get("doa"):
                caps.add("doa")
        if "imu" in sensor_types:
            caps.add("imu")
        if "proximity" in sensor_types:
            caps.add("proximity")
        for ch in m.expressive_channels.values():
            t = ch.get("type")
            if t == "speaker":
                caps.add("speaker")
            if t in ("led_matrix", "led_strip", "leds"):
                caps.add("leds")
            if t == "display":
                caps.add("display")
        if self.dofs:
            caps.add("motion")
        if m.pose_groups:
            caps.add("pose_groups")
        if "teach" in m.motor_modes:
            caps.add("teach_mode")
        if isinstance(m.safety.estop, dict):
            caps.add("estop")
        if m.retargeting:
            caps.add("retargeting")
        return caps


def _deg(x: float) -> float:
    return math.radians(x)


def load_manifest(
    path: Path | str | dict,
    *,
    allow_unmeasured_dynamics: bool = False,
    safety: SafetyConfig | None = None,
) -> Manifest:
    """Parse and validate. ``allow_unmeasured_dynamics`` is for development only (SPEC 12)."""
    raw = path if isinstance(path, dict) else yaml.safe_load(Path(path).read_text())
    try:
        model = ManifestModel.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ManifestError(str(exc)) from exc
    safety = safety or SafetyConfig()
    dofs: dict[str, Dof] = {}

    for j in model.joints:
        lim = j.limits
        if lim.max_vel_dps is None or lim.max_acc_dps2 is None:
            raise ManifestError(f"joint {j.name}: velocity and acceleration limits are required")
        if not lim.min_deg <= j.home_deg <= lim.max_deg:
            raise ManifestError(f"joint {j.name}: home_deg outside its limits")
        dofs[j.name] = Dof(
            j.name,
            j.name,
            _deg(lim.min_deg),
            _deg(lim.max_deg),
            _deg(lim.max_vel_dps),
            _deg(lim.max_acc_dps2),
            _deg(j.home_deg),
        )

    for gname, group in model.pose_groups.items():
        dyn = model.dynamics.get(gname)
        if dyn is None:
            raise ManifestError(f"pose group {gname}: missing dynamics entry")
        measured = dyn.max_vel_dps is not None and dyn.max_acc_dps2 is not None
        if not measured and not allow_unmeasured_dynamics:
            raise ManifestError(
                f"pose group {gname}: dynamics not measured (null); measure them (M0) or load "
                "with allow_unmeasured_dynamics=True for development"
            )
        vel = (
            _deg(dyn.max_vel_dps)
            if dyn.max_vel_dps is not None
            else (safety.unmeasured_dynamics_fallback_vel)
        )
        acc = (
            _deg(dyn.max_acc_dps2)
            if dyn.max_acc_dps2 is not None
            else (safety.unmeasured_dynamics_fallback_acc)
        )
        if group.type == "pose6d":
            lim = group.limits
            assert isinstance(lim, Pose6dLimits)
            for axis in ("roll", "pitch", "yaw"):
                lo, hi = getattr(lim, f"{axis}_deg")
                dofs[f"{gname}.{axis}"] = Dof(
                    f"{gname}.{axis}", gname, _deg(lo), _deg(hi), vel, acc, 0.0, measured
                )
            lin_v, lin_a = dyn.max_lin_vel_mps, dyn.max_lin_acc_mps2
            lin_measured = lin_v is not None and lin_a is not None
            for axis in ("x", "y", "z"):
                rng = getattr(lim, f"{axis}_mm")
                if rng is None:
                    continue  # translation not controllable on this robot
                if not lin_measured and not allow_unmeasured_dynamics:
                    raise ManifestError(
                        f"pose group {gname}: max_lin_vel_mps and "
                        "max_lin_acc_mps2 required for translations"
                    )
                lv = lin_v if lin_v is not None else safety.unmeasured_dynamics_fallback_lin_vel
                la = lin_a if lin_a is not None else safety.unmeasured_dynamics_fallback_lin_acc
                dofs[f"{gname}.{axis}"] = Dof(
                    f"{gname}.{axis}",
                    gname,
                    rng[0] / 1000,
                    rng[1] / 1000,
                    lv,
                    la,
                    0.0,
                    measured and lin_measured,
                    "translation",
                )
        elif group.type == "joint":
            lim = group.limits
            if not isinstance(lim, JointLimits):
                raise ManifestError(f"pose group {gname}: joint limits required")
            dofs[gname] = Dof(
                gname, gname, _deg(lim.min_deg), _deg(lim.max_deg), vel, acc, 0.0, measured
            )
        else:  # joint_pair
            lim = group.limits
            if not isinstance(lim, JointLimits):
                raise ManifestError(f"pose group {gname}: joint_pair needs limits")
            homes = group.home_deg or [0.0, 0.0]
            if any(not lim.min_deg <= h <= lim.max_deg for h in homes):
                raise ManifestError(f"pose group {gname}: home_deg outside its limits")
            for name, home in zip(group.names or [], homes, strict=True):
                dofs[name] = Dof(
                    name,
                    gname,
                    _deg(lim.min_deg),
                    _deg(lim.max_deg),
                    vel,
                    acc,
                    _deg(home),
                    measured,
                )

    if not dofs:
        raise ManifestError("manifest declares no joints or pose groups")

    for c in model.constraints:
        for ref in (c.a, c.b):
            if ref not in dofs:
                raise ManifestError(f"constraint references unknown dof {ref}")
    for target in model.retargeting:
        if target not in dofs:
            raise ManifestError(f"retargeting references unknown dof {target}")

    _check_circadian_motion(model.behaviour.get("circadian_motion"), dofs)
    return Manifest(model, dofs, list(model.constraints))


def _check_circadian_motion(raw: Any, dofs: dict[str, Dof]) -> None:
    """Fail at load, not at the first yawn: the circadian poses and primitives (SPEC 7A) must name
    angle dofs this robot has, and a rest pose must lie inside their limits (no silent clamp)."""
    if not raw:
        return
    from ..config.schema import CircadianMotion

    try:
        motion = CircadianMotion.model_validate(raw)
    except Exception as exc:
        raise ManifestError(f"behaviour.circadian_motion: {exc}") from exc
    named = {f"rest_pose_deg.{n}": n for n in motion.rest_pose_deg}
    for label in ("yawn", "stretch"):
        entry = getattr(motion, label)
        if entry is not None:
            named |= {f"{label}.{n}": n for n in entry.channels}
    for where, name in named.items():
        dof = dofs.get(name)
        if dof is None or dof.kind != "angle":
            raise ManifestError(f"behaviour.circadian_motion.{where}: no angle dof {name!r}")
    for name, deg in motion.rest_pose_deg.items():
        if not dofs[name].min <= _deg(deg) <= dofs[name].max:
            raise ManifestError(
                f"behaviour.circadian_motion.rest_pose_deg.{name}: {deg} deg is outside the limits"
            )


# ----------------------------------------------------------------------------- negotiation


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    requires: frozenset[str]
    optional: frozenset[str] = frozenset()


@dataclass
class Negotiation:
    started: dict[str, set[str]]  # module -> optional capabilities it may use
    disabled: dict[str, set[str]]  # module -> missing required capabilities

    def log_lines(self) -> list[str]:
        lines = [f"started {m} (optional: {sorted(o) or '-'})" for m, o in self.started.items()]
        lines += [f"disabled {m}: missing {sorted(r)}" for m, r in self.disabled.items()]
        return lines


STANDARD_MODULES = (
    ModuleSpec("motion", frozenset({"motion"})),
    ModuleSpec("vision", frozenset({"camera"})),
    ModuleSpec("face_recognition", frozenset({"camera"})),
    ModuleSpec("mirror", frozenset({"camera", "motion", "retargeting"}), frozenset({"teach_mode"})),
    ModuleSpec("voice_in", frozenset({"microphone"}), frozenset({"doa"})),
    ModuleSpec("voice_out", frozenset({"speaker"})),
    ModuleSpec("conversation", frozenset({"microphone", "speaker"})),
    ModuleSpec("circadian_light", frozenset({"camera"})),
    ModuleSpec("expression", frozenset(), frozenset({"motion", "leds", "speaker", "display"})),
)


def negotiate(caps: set[str], modules: tuple[ModuleSpec, ...] = STANDARD_MODULES) -> Negotiation:
    started: dict[str, set[str]] = {}
    disabled: dict[str, set[str]] = {}
    for mod in modules:
        missing = set(mod.requires - caps)
        if missing:
            disabled[mod.name] = missing
        else:
            started[mod.name] = set(mod.optional & caps)
    return Negotiation(started, disabled)
