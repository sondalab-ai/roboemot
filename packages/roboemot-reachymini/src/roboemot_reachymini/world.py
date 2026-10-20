"""Reachy Mini world plugin: scan planner and sound-direction source (SPEC 5A, 4A).

Registered in the ``roboemot.world`` entry-point group under ``reachy-mini-wireless``.

Sound direction, from the reachy-mini 1.12 source (``media/audio_doa.py``): ``mini.media.get_DoA()``
returns ``(angle_rad, speech_detected)`` or None when the array is missing. The angle is 0 at
the left, pi/2 in front (or behind: the array reports one half-plane), pi at the right, so a
sound behind appears mirrored in front. The azimuth is ``pi/2 - angle`` (REP-103, positive
left). Assumption, to verify on the robot: the array turns with the head, so the observation is
in the head frame and the world model rotates it with the joint state at capture.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from roboemot import shutdown
from roboemot.attention.gaze import look_at
from roboemot.config.schema import RoboEmotConfig, WorldScanConfig
from roboemot.manifest import Constraint, Dof, Manifest
from roboemot.world import Observation, ObservationSink, ScanWaypoint, WorldPlugin

from .gaze import GAZE

log = logging.getLogger(__name__)

SCAN_DOFS = (GAZE.head_yaw, GAZE.body_yaw)
FULL_CIRCLE_EPS = 1e-6
SOUND_SOURCE = "doa"
JOIN_TIMEOUT_S = 2.0


def doa_azimuth(angle_rad: float) -> float:
    """Base-of-head azimuth (positive left) of an SDK direction-of-arrival angle."""
    return math.pi / 2 - angle_rad


class ReachyScanPlanner:
    """Head and body yaw sweeps within the manifest limits (SPEC 5A, Scanning).

    The camera looks where the head points, and the head can lead the body by up to the
    ``yaw_delta`` limit less the safety margin: body +-160 deg plus about 65 deg covers every
    azimuth. Waypoints are base-frame azimuths turned into head and body yaw with the same
    split as attention (``look_at``). Head pitch is left to idle breathing.
    """

    def __init__(
        self, dofs: Mapping[str, Dof], constraints: list[Constraint], cfg: RoboEmotConfig
    ) -> None:
        self.dofs, self.constraints = dofs, constraints
        self.apply_config(cfg)
        self._flip = False

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload (SPEC 11A): scan, comfort band and safety margin."""
        self.scan: WorldScanConfig = cfg.world.scan
        self.comfort_rad = math.radians(cfg.attention.head_yaw_comfort_deg)
        self.margin_rad = math.radians(cfg.safety.constraint_margin_deg)

    def plan(
        self, pose: Mapping[str, float], fov_h_rad: float, *, amplitude: float, dwell_s: float
    ) -> list[ScanWaypoint]:
        """Evenly spaced azimuths over ``amplitude`` of the circle, views overlapping.

        The spacing is at most the field of view less ``overlap_deg``. A full circle is
        planned as -pi .. pi with the duplicate rear waypoint dropped. The path runs
        monotonically from the end nearest to the head (alternating when both ends are about as
        near, so repeated sweeps do not always start on the same side), so it never
        crosses the rear seam in one move.
        """
        span = 2 * math.pi * min(1.0, max(0.0, amplitude))
        step = max(
            fov_h_rad - math.radians(self.scan.overlap_deg), fov_h_rad * self.scan.min_step_ratio
        )
        count = max(1, math.ceil(span / step)) + 1
        azimuths = [-span / 2 + k * span / (count - 1) for k in range(count)]
        if span >= 2 * math.pi - FULL_CIRCLE_EPS:
            azimuths.pop()
        head = pose.get(GAZE.head_yaw, 0.0)
        near_first, near_last = abs(head - azimuths[0]), abs(head - azimuths[-1])
        spacing = azimuths[1] - azimuths[0] if len(azimuths) > 1 else 0.0
        tie = abs(near_last - near_first) <= spacing
        reverse = self._flip if tie else near_last < near_first
        if reverse:
            azimuths.reverse()
        self._flip = not self._flip
        body = pose.get(GAZE.body_yaw, 0.0)
        out = []
        for az in azimuths:
            targets = self._targets(az, 0.0, body)
            body = targets[GAZE.body_yaw]
            out.append(ScanWaypoint(az, targets, dwell_s))
        return out

    def glance(
        self,
        pose: Mapping[str, float],
        azimuth_rad: float,
        elevation_rad: float,
        *,
        dwell_s: float,
    ) -> ScanWaypoint:
        """One waypoint that points the head at a base-frame azimuth."""
        targets = self._targets(azimuth_rad, elevation_rad, pose.get(GAZE.body_yaw, 0.0))
        return ScanWaypoint(azimuth_rad, targets, dwell_s)

    def _targets(
        self, azimuth_rad: float, elevation_rad: float, body_now: float
    ) -> dict[str, float]:
        full = look_at(
            azimuth_rad,
            elevation_rad,
            GAZE,
            self.dofs,
            self.constraints,
            comfort_rad=self.comfort_rad,
            margin_rad=self.margin_rad,
            body_now=body_now,
        )
        return {name: full[name] for name in SCAN_DOFS}


class DoASource:
    """Bearing-only ``unknown`` observations from the microphone array's direction of arrival.

    Polls ``mini.media.get_DoA()`` in a thread at ``world.scan.doa_poll_hz`` and reports a
    reading only while speech is detected. The joint state comes from ``joint_state`` (the
    measured pose); without one the reading is skipped, since it could not be rotated into the
    base frame. Sources never create entries: they confirm one or trigger a glance. Nothing
    that fails inside a poll (the SDK, the joint state, a raising subscriber) ends the thread.
    ``speaking`` (set by the app to ``SpeechActivity.speaking``) is checked on every poll, like
    the reflex engine's gate, so the robot's own voice and its echo tail are never read;
    ``mute`` is the one-off variant that takes a stamp.
    """

    name = SOUND_SOURCE

    def __init__(
        self,
        mini: Any,
        cfg: WorldScanConfig,
        joint_state: Callable[[], Mapping[str, float]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mini, self.cfg, self._joint_state, self._clock = mini, cfg, joint_state, clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._errors: dict[str, str] = {}
        self._muted_until = -math.inf
        self.speaking: Callable[[float], bool] | None = None

    def mute(self, until_stamp: float) -> None:
        """Ignore the microphones until the clock passes ``until_stamp`` (own speech playing)."""
        self._muted_until = until_stamp

    def _note_error(self, exc: Exception, what: str) -> None:
        """Log a failure of stage ``what`` once per distinct message until the stage works."""
        msg = f"{what}: {type(exc).__name__}: {exc}"
        if self._errors.get(what) != msg:
            log.warning("doa: %s", msg)
            self._errors[what] = msg

    def poll_once(self, sink: ObservationSink) -> bool:
        """Read once; hand a speech direction to ``sink``. Returns whether one was sent.

        A failure of a stage (read, joint state or sink) is logged once per distinct message,
        not per poll; it is logged again after that stage has worked once.
        """
        now = self._clock()
        if now < self._muted_until or (self.speaking is not None and self.speaking(now)):
            return False
        try:
            reading = self._mini.media.get_DoA()
        except Exception as exc:
            self._note_error(exc, "read")
            return False
        self._errors.pop("read", None)
        if reading is None:
            return False
        angle, speech = reading
        if not speech:
            return False
        try:
            state = dict(self._joint_state())
        except Exception as exc:
            self._note_error(exc, "joint state")
            return False
        self._errors.pop("joint state", None)
        obs = Observation(
            "unknown",
            self._clock(),
            self.name,
            doa_azimuth(float(angle)),
            0.0,
            confidence=self.cfg.doa_confidence,
        )
        try:
            sink(obs, state)
        except Exception as exc:
            self._note_error(exc, "subscriber")
            return False
        self._errors.pop("subscriber", None)
        return True

    def start(self, sink: ObservationSink) -> None:
        """Poll in a daemon thread until ``stop``; a second start while running is a no-op."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        thread = threading.Thread(target=self._run, args=(sink,), name="roboemot-doa", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        """Stop polling and wait briefly. A thread stuck in the SDK keeps its handle, so a later
        ``start`` does not run a second one beside it."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            shutdown.join(thread, JOIN_TIMEOUT_S)
            if not thread.is_alive():
                self._thread = None

    def _run(self, sink: ObservationSink) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once(sink)
            except Exception as exc:  # never end the thread
                self._note_error(exc, "poll")
            self._stop.wait(1.0 / self.cfg.doa_poll_hz)


def plugin(
    manifest: Manifest,
    cfg: RoboEmotConfig,
    robot: Any,
    joint_state: Callable[[], Mapping[str, float]],
) -> WorldPlugin:
    """Entry point ``roboemot.world``: the Reachy Mini planner and sound-direction source.

    ``robot`` is the connected ``ReachyMini``; ``joint_state`` is the app driver's measured
    state (no second driver is built here). Without a microphone capability no sound source
    is offered.
    """
    planner = ReachyScanPlanner(manifest.dofs, manifest.constraints, cfg)
    sources: list[Any] = []
    if "doa" in manifest.capabilities():
        sources.append(DoASource(robot, cfg.world.scan, joint_state))
    return WorldPlugin(planner, sources)
