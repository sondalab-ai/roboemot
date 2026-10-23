"""Vision pipeline: detect, track, embed, match, publish (SPEC 5, 5A, 6, 12, 13).

``process(frame)`` is synchronous and uses only the capture stamp, so a replay is
deterministic; ``LatestFrameRunner`` drives it live. Per frame, when the governor allows:

1. Faces are detected, filtered by ``face_min_px`` and tracked (stable ``track_id``). Tracks
   unmatched for the larger of ``tracker.lost_s`` and ``face_lost_periods`` detection periods
   at the governor's rate (capture time) expire before association, so a face that comes back
   after a long gap is a new track with no inherited identity or attributes.
2. A new track gets ``embedding_burst`` face evaluations, at least ``burst_min_gap_s`` apart
   (divided by the conversation scale only, so a thinking or speaking robot spends less on
   bursts; the circadian and load scales slow bursts only through the detection rate),
   stopping early once it is recognised, then one per ``1 / embedding_hz``. Only confidently
   detected faces of at least ``embed_min_face_px`` are embedded. The burst (5 by default)
   lets one missed vote still fit the SPEC 13 identity budget of 2.5 s with the 3-of-5 votes
   of SPEC 6. The embedding goes straight to ``Matcher.observe`` and is dropped; it is never
   published.
3. Per track, the first ``vision.attribute_crops`` face crops go to the attribute worker
   (skipped when the recognised person has declared age and gender); until it answers, age and
   gender are ``unknown`` (a possible minor). A job whose age comes back unknown (and no
   declared age), or that fails or is cancelled while its track lives, is retried:
   ``vision.attribute_retry_s`` of capture time after the result (or, while the robot thinks or
   speaks, at the first face frame after), the track collects new crops, until a confident age
   or a declared one stops it. An adult or senior age that follows an unknown result needs a
   second agreeing confident result. The age stays unknown, a possible minor, in between.
   Other visible faces that fall inside a crop's margin are filled with the letterbox grey
   (outside the subject's own box), so a neighbour's face cannot pull the estimate toward the
   neighbour's age.
4. Published: ``percept.face`` (no embedding), ``person.recognized`` when a track turns into a
   known person, ``percept.entity`` (kind person, with the camera-frame bearing; the joint state
   at capture travels in the message ``meta`` for the world model to rotate into the base
   frame). Optional object stage: same for animals and objects, whose DINOv2-class appearance
   embedding rides in the ``Observation`` (SPEC 5A) for world-model matching and the registry.
   ``vision.animal_categories`` and ``vision.object_categories`` map detector classes to the
   kind; any other class, ``person`` included, is ignored.

``light_tap`` (optional) receives each frame's image before anything else, for the circadian
light sampler (SPEC 7A); it measures and keeps only numbers, never the image.

``face_tap`` (optional) is called with each processed frame and its visible face tracks, on
the vision thread, for the enrollment service (SPEC 6); it must not keep the image.

The world model is fed over the bus: ``world.feed_world(bus, world)`` subscribes to
``percept.entity``. The bus keeps vision decoupled from the world model (they may live in
separate processes later) and the in-process bus calls subscribers synchronously, so replay
stays deterministic. A subscriber that raises is logged by type and skipped; it never aborts
the frame. Frames and crops are never logged or written (SPEC 12).

For the debug report (SPEC 12A) ``stats`` adds face track churn (created, ended, the creation
rate over ``vision.report_rate_window_s`` of capture time, the visible lifetime of the last
``vision.report_window`` ended tracks) and the age and gender jobs (the worker's counts and
timings, results dropped because their track ended or never sent because the queue stayed full,
retry jobs sent for an unknown age, ended tracks whose age was still unknown, capture time from a
track's first detection to its first estimate, and the age each job left the track with: adult,
minor or unknown; with retries these count jobs, not tracks).
``tracks`` also explains the churn (``track_stats``): how confirmed face tracks ended
(``ended_by``), the misses and effective ``lost_s`` behind each expiry, rebirths (a new track
born near where one ended, with gap, IoU, distance and head pose change), and the scores of
low-score detections the tracker refused or left unmatched, beside those that started tracks.
The head pose comes from the frame's joint state through the robot's ``frame_spec``.
``rate_scales`` lists the governor's multipliers by source (circadian, load, conversation)
beside their product ``rate_scale``. Numbers only; the frame thread writes them and the report
thread reads them under one lock.

``refresh`` lets the matcher add embeddings of confident matches to the registry (SPEC 6,
online refresh). ``replay`` forces it off: refresh uses the registry's wall clock and mutates
the stored galleries, so a replay would change the enrolled people.
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..bus import Bus
from ..config.schema import IdentityConfig, TrackerConfig, VisionConfig, WorldConfig
from ..registry import Estimate, Matcher, possible_minor
from ..registry.records import Attribute
from ..stats import P50, P90, summarise
from ..world import FrameSpec, Observation
from .attributes import AttributeWorker
from .geometry import CameraModel
from .governor import CONVERSATION, EPSILON_S, RateGovernor
from .stages import (
    AppearanceEmbedder,
    AttributeEstimator,
    FaceDetector,
    FaceEmbedder,
    ObjectDetector,
)
from .track_stats import TrackDiagnostics, head_pose
from .tracker import Track, Tracker
from .types import (
    LETTERBOX_GREY,
    Box,
    Detection,
    FacePercept,
    Frame,
    FrameReport,
    PersonRecognized,
)

log = logging.getLogger(__name__)

PERSON = "person"
UNKNOWN = "unknown"
NO_ESTIMATE = Estimate(UNKNOWN, "none")
MS_PER_S = 1000.0
S_PER_MIN = 60.0
SUMMARY_LEVELS = {"p50": P50, "p90": P90}
OUTCOMES = ("adult", "minor", "unknown")
CameraProvider = Callable[[int, int], CameraModel | None]
FaceTap = Callable[[Frame, list[Track]], None]
LightTap = Callable[[np.ndarray], object]


@dataclass
class _FaceState:
    generation: int
    first_seen: float = 0.0  # capture stamp of the track's first detection
    last_seen: float = 0.0  # capture stamp of its last visible frame
    evals: int = 0
    last_embed: float | None = None
    owner_id: str | None = None
    anonymous_id: str | None = None
    announced: str | None = None
    reported: bool = False
    crops: list[tuple[float, np.ndarray]] = field(default_factory=list, repr=False)
    last_crop: float | None = None
    attr: Literal["collecting", "submitted", "skipped"] = "collecting"
    age: Estimate = NO_ESTIMATE
    gender: Estimate = NO_ESTIMATE
    attr_stamp: float | None = None
    retry_at: float | None = None  # capture stamp from which an unknown age is estimated again
    retrying: bool = False  # the crops being collected are for a retry job
    in_flight: bool = False  # a job was sent and its result not absorbed yet
    jobs_done: int = 0  # results absorbed for this track
    candidate: str | None = None  # a retry's non-minor age waiting for a second agreeing result


@dataclass
class _EntityState:
    evals: int = 0
    last_embed: float | None = None
    owner_id: str | None = None
    embedding: np.ndarray | None = field(default=None, repr=False)


def _declared(attr: Attribute | None) -> str | None:
    return attr.value if attr is not None and attr.source == "declared" else None


def _inside(box: Box, left: int, top: int, shape: tuple[int, ...]) -> tuple[slice, slice]:
    """Rows and columns of ``box`` (frame pixels) within a crop at ``left, top`` of ``shape``."""
    x1, y1, x2, y2 = box
    rows = slice(min(max(0, int(y1) - top), shape[0]), min(max(0, int(y2) - top), shape[0]))
    cols = slice(min(max(0, int(x1) - left), shape[1]), min(max(0, int(x2) - left), shape[1]))
    return rows, cols


class VisionPipeline:
    """The perception module that turns camera frames into percepts; see the module docstring."""

    name = "vision"

    def __init__(
        self,
        cfg: VisionConfig,
        identity: IdentityConfig,
        bus: Bus,
        matcher: Matcher,
        *,
        face_detector: FaceDetector,
        face_embedder: FaceEmbedder,
        attribute_estimator: AttributeEstimator | None = None,
        appearance_embedder: AppearanceEmbedder | None = None,
        object_detector: ObjectDetector | None = None,
        governor: RateGovernor | None = None,
        executor: Executor | None = None,
        source: str = "vision",
        refresh: bool = True,
        camera_model: CameraProvider | None = None,
        world: WorldConfig | None = None,
        frame_spec: FrameSpec | None = None,
    ):
        if cfg.embedding_burst < identity.votes_required:
            raise ValueError("vision.embedding_burst must be at least identity.votes_required")
        self.cfg = cfg
        self.identity = identity
        self.bus = bus
        self.matcher = matcher
        self.source = source
        self.refresh = refresh
        self.camera_model = camera_model
        self.world = world or WorldConfig()
        self.face_tap: FaceTap | None = None
        self._visible: list[Track] = []
        self.light_tap: LightTap | None = None
        self.governor = governor or RateGovernor(cfg.rates)
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.appearance_embedder = appearance_embedder
        self.object_detector = object_detector
        self._lost_held = 0.0
        ids = itertools.count(1)
        self.face_tracker = Tracker(cfg.tracker, ids)
        self.frame_spec = frame_spec
        self.track_stats = TrackDiagnostics(cfg)
        self.face_tracker.observer = self.track_stats
        self.object_tracker = Tracker(self._object_tracker_cfg(), ids)
        self.worker = (
            AttributeWorker(
                attribute_estimator,
                identity,
                cfg.attribute_queue_max,
                executor,
                report_window=cfg.report_window,
                age_confidence_min=cfg.attribute_score_min,
            )
            if attribute_estimator is not None
            else None
        )
        self._generations = itertools.count(1)
        self._cameras: dict[tuple[int, int], CameraModel] = {}
        self._faces: dict[int, _FaceState] = {}
        self._entities: dict[int, _EntityState] = {}
        self._budget = 0
        self._entity_budget = 0
        self._entity_embeds = 0
        self._report_lock = threading.Lock()
        self._last_stamp: float | None = None
        self._tracks_created = 0
        self._tracks_ended = 0
        self._created_stamps: deque[float] = deque()
        self._lifetimes_s: deque[float] = deque(maxlen=cfg.report_window)
        self._to_estimate_ms: deque[float] = deque(maxlen=cfg.report_window)
        self._outcomes = dict.fromkeys(OUTCOMES, 0)
        self._attr_drops = {"dropped_queue_full": 0, "dropped_track_gone": 0}
        self._retries = 0
        self._ended_unknown = 0
        self._face_lost_held = 0.0
        matcher.registry.add_forget_listener(self._on_forget)

    def process(self, frame: Frame) -> FrameReport:
        """Run the stages that are due for one frame; frames that are not due are dropped.

        Uses only ``frame.stamp`` for time. Publishes on the bus as described in the module
        docstring and returns what ran. Tracks that expired by this stamp are dropped first,
        even when the governor is off, so state never outlives its track. Raises whatever a
        stage raises: ``LatestFrameRunner`` catches it per frame.
        """
        stamp = frame.stamp
        with self._report_lock:
            self._last_stamp = stamp
        self._absorb_attributes(stamp)
        self._tap_light(frame)
        self.face_tracker.cfg = self._face_tracker_cfg()
        self.object_tracker.cfg = self._object_tracker_cfg()
        self._expire(stamp)
        if self.governor.scale <= 0:
            return FrameReport(stamp)
        camera = self._camera(frame.image.shape[1], frame.image.shape[0])
        self._budget = self.cfg.max_embeds_per_frame
        self._entity_embeds = 0
        faces = objects = False
        if self.governor.allow("detection", stamp):
            self._process_faces(frame, camera)
            faces = True
        if (
            self.object_detector is not None
            and self.cfg.objects_enabled
            and self.governor.allow("objects", stamp)
        ):
            self._process_objects(frame, camera)
            objects = True
        embeds = self.cfg.max_embeds_per_frame - self._budget + self._entity_embeds
        return FrameReport(stamp, faces, objects, embeds)

    def _tap_light(self, frame: Frame) -> None:
        """Hand the image to the light sampler (SPEC 7A), before the governor can drop the frame
        so a resting robot still senses light. The sampler measures it at most once per
        ``light.sample_interval_s`` on this thread and keeps only numbers; a failure is logged by
        type and never costs the frame."""
        if self.light_tap is None:
            return
        try:
            self.light_tap(frame.image)
        except Exception as exc:
            log.warning("light tap failed: %s", type(exc).__name__)

    def reset(self) -> None:
        """Forget every track and its per-track state (start of a session or replay).

        Releases the matcher's votes for each track, cancels attribute jobs that have not
        started, restarts track IDs at 1 and resets the governor. A job already running
        finishes but its result belongs to an old generation and is ignored.
        """
        for track_id in (*self._faces, *self._entities):
            self.matcher.release(track_id)
        for state in self._faces.values():
            self._end_face(state)
        self.track_stats.reset(len(self._faces))
        self._faces.clear()
        self._entities.clear()
        ids = itertools.count(1)
        self.face_tracker.reset(ids)
        self.object_tracker.reset(ids)
        self._lost_held = 0.0
        self._face_lost_held = 0.0
        self.governor.reset()
        if self.worker is not None:
            self.worker.cancel_pending()

    def apply_config(
        self, vision: VisionConfig, identity: IdentityConfig, world: WorldConfig | None = None
    ) -> None:
        """Hot reload (SPEC 11A): thresholds, rates and tracker settings take effect at once, and
        so does, with ``world``, the camera field of view.

        The attribute worker gets the new ``identity`` too, so ``identity.estimate_frames_min``
        and ``estimate_confidence_min`` apply to the next job. ``vision.attribute_queue_max``,
        the age model's input size and score floor, the thread counts, the detector size and the
        models folder are restart-only (the schema marks them).
        """
        self.cfg, self.identity = vision, identity
        self.track_stats.cfg = vision
        if world is not None:
            self.world = world
        if self.worker is not None:
            self.worker.config = identity  # frames_min and the gender rule apply to new jobs
        self.governor.rates = vision.rates
        self.face_tracker.cfg = self._face_tracker_cfg()
        self.object_tracker.cfg = self._object_tracker_cfg()
        self.matcher.config = identity

    def _face_lost_s(self) -> float:
        """Lifetime of a face track at the rate the governor allows now: the larger of
        ``tracker.lost_s`` and ``face_lost_periods`` face-detection periods."""
        lost = self.cfg.tracker.lost_s
        period = self.governor.period("detection")
        if math.isfinite(period):
            lost = max(lost, self.cfg.face_lost_periods * period)
        return lost

    def _face_tracker_cfg(self) -> TrackerConfig:
        """Face tracker settings: ``vision.tracker`` with a lifetime that follows the detection
        period (SPEC 5, 13), so a slower detector (conversation, circadian or load scale) does
        not end tracks between two detections. As for objects, the longer lifetime is kept
        until the next face frame has been folded in after the rate goes back up."""
        lost = max(self._face_lost_s(), self._face_lost_held)
        return self.cfg.tracker.model_copy(update={"lost_s": lost})

    def _object_lost_s(self) -> float:
        """Lifetime of an animal or object track at the rate the governor allows now."""
        lost = self.cfg.object_lost_s
        period = self.governor.period("objects")
        if math.isfinite(period):
            lost = max(lost, self.cfg.object_lost_periods * period)
        return lost

    def _object_tracker_cfg(self) -> TrackerConfig:
        """Tracker settings for animals and objects, derived from the vision config.

        A detection at ``object_score_min`` starts a track, and there is no low-score pass: the
        detector and ``_process_objects`` already drop everything below that score, so
        ``low_score`` equals ``high_score``. A track lives for the larger of ``object_lost_s``
        and ``object_lost_periods`` object-detection periods at the rate the governor allows
        now, so a slow object stage (circadian or load scale) does not expire tracks between
        two frames. When the rate goes back up the longer lifetime is kept until the next
        object frame has been folded in, so tracks do not expire just before it can match them.
        """
        return self.cfg.tracker.model_copy(
            update={
                "high_score": self.cfg.object_score_min,
                "low_score": self.cfg.object_score_min,
                "confirm_hits": self.cfg.object_confirm_hits,
                "lost_s": max(self._object_lost_s(), self._lost_held),
            }
        )

    def stats(self) -> dict[str, Any]:
        """Counts for the debug report (SPEC 12A): live tracks, governor scale and drops, face
        track churn and the age and gender jobs (see the module docstring).

        Numbers only; no identity, embedding or image.
        """
        window_s = self.cfg.report_rate_window_s
        with self._report_lock:
            last = self._last_stamp
            recent = sum(
                1 for t in self._created_stamps if last is not None and t >= last - window_s
            )
            tracks = {
                "created": self._tracks_created,
                "ended": self._tracks_ended,
                "created_per_min": recent * S_PER_MIN / window_s,
                "rate_window_s": window_s,
                "lifetime_s": summarise(list(self._lifetimes_s), SUMMARY_LEVELS),
            }
            attributes: dict[str, Any] = {
                **self._attr_drops,
                "retries": self._retries,
                "ended_unknown": self._ended_unknown,
                "outcomes": dict(self._outcomes),
                "time_to_estimate_ms": summarise(list(self._to_estimate_ms), SUMMARY_LEVELS),
            }
        tracks.update(self.track_stats.snapshot())
        if self.worker is None:
            attributes = {"state": "off"}
        else:
            attributes = {**self.worker.snapshot(), **attributes}
        return {
            "face_tracks": len(self._faces),
            "entity_tracks": len(self._entities),
            "rate_scale": self.governor.scale,
            "rate_scales": self.governor.scales(),
            "stage_drops": dict(self.governor.dropped),
            "tracks": tracks,
            "attributes": attributes,
        }

    def close(self) -> None:
        """Stop the attribute worker: no new jobs, waiting ones cancelled, own pool shut down."""
        if self.worker is not None:
            self.worker.close()

    def _camera(self, width: int, height: int) -> CameraModel:
        key = (width, height)
        if key not in self._cameras:
            model = self.camera_model(width, height) if self.camera_model else None
            self._cameras[key] = model or CameraModel(
                width,
                height,
                self.world.camera_fov_h_deg,
                vfov_deg=self.world.camera_fov_v_deg,
            )
        return self._cameras[key]

    def _publish(self, topic: str, payload: Any, stamp: float, **meta: Any) -> bool:
        """Publish; a raising subscriber is logged by type only and reported as False."""
        try:
            self.bus.publish(topic, payload, stamp, **meta)
        except Exception as exc:
            log.warning("subscriber of %s failed: %s", topic, type(exc).__name__)
            return False
        return True

    def _on_forget(self, owner_id: str) -> None:
        """Registry forget listener: tracks of that person become unknown again, and a declared
        age that hid an unknown estimate no longer does, so the age job runs or is retried."""
        for state in self._faces.values():
            if state.owner_id == owner_id:
                state.owner_id = None
                self._unhide_age(state)
            if state.announced == owner_id:
                state.announced = None
        for entity in self._entities.values():
            if entity.owner_id == owner_id:
                entity.owner_id = None

    def _expire(self, stamp: float) -> None:
        for track_id in self.face_tracker.expire(stamp):
            self._drop_face(track_id)
        for track_id in self.object_tracker.expire(stamp):
            self._drop_entity(track_id)

    def _drop_face(self, track_id: int) -> None:
        state = self._faces.pop(track_id, None)
        self.matcher.release(track_id)
        if state is not None:
            self._end_face(state)

    def _unhide_age(self, state: _FaceState) -> None:
        """After a forget: a track whose job was skipped for declared values collects crops,
        and one whose estimate is unknown (no job in flight) is retried at once."""
        if self.worker is None:
            return
        if state.attr == "skipped":
            state.attr = "collecting"
        elif not state.in_flight and state.retry_at is None and state.age.value == UNKNOWN:
            state.retry_at = state.last_seen

    def _end_face(self, state: _FaceState) -> None:
        """Count an ended face track: its visible lifetime, a job it never sent because the
        attribute queue stayed full while its crops were ready, and whether its age was still
        unknown (a possible minor for its whole life)."""
        never_sent = (
            self.worker is not None
            and state.attr == "collecting"
            and len(state.crops) >= self.cfg.attribute_crops
        )
        unknown = self.worker is not None and self._attributes(state)[0].value == UNKNOWN
        with self._report_lock:
            self._tracks_ended += 1
            self._lifetimes_s.append(state.last_seen - state.first_seen)
            self._attr_drops["dropped_queue_full"] += never_sent
            self._ended_unknown += unknown

    def _new_face(self, track: Track, stamp: float) -> _FaceState:
        """Per-track state of a newly confirmed face track, counted for the churn report."""
        window_s = self.cfg.report_rate_window_s
        with self._report_lock:
            self._tracks_created += 1
            self._created_stamps.append(stamp)
            while self._created_stamps and self._created_stamps[0] < stamp - window_s:
                self._created_stamps.popleft()
        return _FaceState(next(self._generations), first_seen=track.first_stamp)

    def _count_outcome(self, age: Estimate) -> None:
        """Adult (adult or senior), unknown, or minor (a declared or estimated younger group)."""
        if age.value == UNKNOWN:
            outcome = "unknown"
        elif possible_minor(age.value):
            outcome = "minor"
        else:
            outcome = "adult"
        with self._report_lock:
            self._outcomes[outcome] += 1

    def _drop_entity(self, track_id: int) -> None:
        self._entities.pop(track_id, None)
        self.matcher.release(track_id)

    # ------------------------------------------------------------------ faces

    def _process_faces(self, frame: Frame, camera: CameraModel) -> None:
        min_px = self.cfg.face_min_px
        dets = [d for d in self.face_detector.detect(frame.image) if d.min_side >= min_px]
        self.track_stats.pose = head_pose(frame.joint_state, self.frame_spec)
        result = self.face_tracker.update(dets, frame.stamp)
        self._face_lost_held = self._face_lost_s()
        self._visible = result.visible
        for track_id in result.removed_ids:
            self._drop_face(track_id)
        for track in result.visible:
            state = self._faces.get(track.track_id)
            if state is None:
                state = self._faces[track.track_id] = self._new_face(track, frame.stamp)
            state.last_seen = frame.stamp
            self._evaluate_face(frame, track, state)
            self._collect_crop(frame, track, state)
            self._publish_face(frame, camera, track, state)
        if self.face_tap is not None:
            try:
                self.face_tap(frame, result.visible)
            except Exception as exc:
                log.warning("face tap failed: %s", type(exc).__name__)
        self._publish("percept.face_frame", frame.stamp, frame.stamp)

    def clear_overlay(self) -> None:
        """Forget the visible tracks the live preview would draw (the robot fell asleep)."""
        self._visible = []

    def face_overlay(self) -> list[tuple[Box, str | None]]:
        """The visible face tracks of the last processed frame as ``(box in pixels, owner_id or
        None)``, for the web UI's live preview. Read on demand from another thread; no cost per
        frame beyond one reference swap."""
        visible = self._visible
        out: list[tuple[Box, str | None]] = []
        for track in visible:
            state = self._faces.get(track.track_id)
            out.append((track.box, state.owner_id if state is not None else None))
        return out

    def preview_overlay(self, report: FrameReport) -> list[tuple[Box, str | None]] | None:
        """The live preview's boxes for the frame ``report`` is about: ``face_overlay`` when
        face detection ran on it, ``[]`` while detection is off (governor scale 0: the picture
        shows without boxes), else None (the boxes belong to an earlier frame; the runner keeps
        that one for a poll period, then shows newer frames without boxes)."""
        if report.faces_detected:
            return self.face_overlay()
        if math.isinf(self.governor.period("detection")):
            return []
        return None

    def _face_due(self, track: Track, state: _FaceState, stamp: float) -> bool:
        if track.score < self.cfg.tracker.high_score or track.detection is None:
            return False
        if track.detection.min_side < self.cfg.embed_min_face_px:
            return False
        if state.last_embed is None:
            return True
        gap = stamp - state.last_embed
        if state.owner_id is None and state.evals < self.cfg.embedding_burst:
            return gap >= self.governor.stretch(self.cfg.burst_min_gap_s, CONVERSATION) - EPSILON_S
        return gap >= self.governor.period("embedding")

    def _evaluate_face(self, frame: Frame, track: Track, state: _FaceState) -> None:
        if self._budget <= 0 or not self._face_due(track, state, frame.stamp):
            return
        self._budget -= 1
        embedding = self.face_embedder.embed(frame.image, track.detection)
        identity = self.matcher.observe(
            track.track_id,
            embedding,
            self.face_embedder.model_id,
            frame.stamp,
            refresh=self.refresh,
        )
        del embedding
        state.evals += 1
        state.last_embed = frame.stamp
        state.owner_id, state.anonymous_id = identity.owner_id, identity.anonymous_id

    def _declared_complete(self, owner_id: str | None) -> bool:
        if owner_id is None:
            return False
        try:
            person = self.matcher.registry.get_person(owner_id)
        except KeyError:
            return False
        return _declared(person.age_group) is not None and _declared(person.gender) is not None

    def _collect_crop(self, frame: Frame, track: Track, state: _FaceState) -> None:
        if self.worker is None:
            return
        self._rearm(state, frame.stamp)
        if state.attr != "collecting":
            return
        if self._declared_complete(state.owner_id):
            state.attr, state.crops = "skipped", []
            if state.jobs_done == 0:  # an earlier job already counted its outcome
                self._count_outcome(self._attributes(state)[0])
            return
        needed = self.cfg.attribute_crops
        gap = self.cfg.attribute_crop_interval_s
        if len(state.crops) < needed and (
            state.last_crop is None or frame.stamp - state.last_crop >= gap
        ):
            others = [t.box for t in self._visible if t.track_id != track.track_id]
            crop = self._crop(frame.image, track.box, others)
            if crop.size:
                state.crops.append((frame.stamp, crop))
                state.last_crop = frame.stamp
        if len(state.crops) >= needed and self.worker.submit(
            track.track_id, state.generation, state.crops
        ):
            state.crops = []
            state.attr, state.in_flight = "submitted", True
            if state.retrying:
                with self._report_lock:
                    self._retries += 1

    def _rearm(self, state: _FaceState, stamp: float) -> None:
        """Back to collecting crops once a retry is due (SPEC 5): the age stays unknown, a
        possible minor, meanwhile. Not when the age turned known since (a declared value of a
        person recognised in between), which also cancels the retry. While the robot thinks or
        speaks (conversation scale below 1) a due retry waits, ``retry_at`` kept, for the first
        detection frame after it, so the TTS keeps the CPU."""
        if state.retry_at is None or stamp < state.retry_at:
            return
        if self.governor.scale_of(CONVERSATION) < 1.0:
            return
        state.retry_at = None
        if self._attributes(state)[0].value != UNKNOWN:
            return
        state.attr, state.retrying, state.last_crop = "collecting", True, None

    def _crop(self, image: np.ndarray, box: Box, others: list[Box] | None = None) -> np.ndarray:
        """Copy of a square around the box (longer side plus margin), clipped to the frame.

        The parts of ``others`` (the other visible face boxes) inside the square are filled with
        ``LETTERBOX_GREY``; the subject's own box is then copied back, so an overlapping
        neighbour never hides the subject's face. Only the copy is changed, never ``image``.
        """
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        half = max(x2 - x1, y2 - y1) * (1 + 2 * self.cfg.attribute_crop_margin) / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left, right = max(0, int(cx - half)), min(width, int(cx + half))
        top, bottom = max(0, int(cy - half)), min(height, int(cy + half))
        crop = image[top:bottom, left:right].copy()
        if not others or not crop.size:
            return crop
        for other in others:
            crop[_inside(other, left, top, crop.shape)] = LETTERBOX_GREY
        own = _inside(box, left, top, crop.shape)
        crop[own] = image[top:bottom, left:right][own]
        return crop

    def _absorb_attributes(self, stamp: float) -> None:
        """Apply finished jobs to their tracks; ``stamp`` (this frame's) times the estimate."""
        if self.worker is None:
            return
        for result in self.worker.poll():
            state = self._faces.get(result.track_id)
            if state is None or state.generation != result.generation:
                with self._report_lock:
                    self._attr_drops["dropped_track_gone"] += 1
                continue
            first = state.jobs_done == 0
            state.jobs_done += 1
            state.in_flight = False
            state.age = self._confirmed_age(state, result.age, first)
            state.gender, state.attr_stamp = result.gender, result.stamp
            age = self._attributes(state)[0]
            self._count_outcome(age)
            if age.value == UNKNOWN:  # retried later; a confident or declared age stops it
                state.retry_at = stamp + self.cfg.attribute_retry_s
            if first:
                with self._report_lock:
                    self._to_estimate_ms.append((stamp - state.first_seen) * MS_PER_S)
        for track_id, generation in self.worker.poll_missed():
            self._retry_missed(track_id, generation, stamp)

    def _retry_missed(self, track_id: int, generation: int, stamp: float) -> None:
        """A job that failed or was cancelled while its track lives is retried like an unknown
        result (SPEC 5): ``attribute_retry_s`` later, waiting for the conversation like any
        retry; the age is untouched, so the track stays as it was (unknown: a possible minor)."""
        state = self._faces.get(track_id)
        if state is None or state.generation != generation or not state.in_flight:
            return
        state.in_flight = False
        if self._attributes(state)[0].value == UNKNOWN:
            state.retry_at = stamp + self.cfg.attribute_retry_s

    def _confirmed_age(self, state: _FaceState, age: Estimate, first: bool) -> Estimate:
        """The age a result gives the track (SPEC 5). Retrying until a crop clears the score
        floor selects for confident errors, so a non-minor (adult or senior) age that follows
        an unknown result counts only when the next confident result agrees with it; until then
        it is held as ``candidate`` and the track stays unknown, a possible minor. Any other
        result (unknown, a minor group, or the track's first result, also after a failed job)
        clears the candidate."""
        if first or age.value == UNKNOWN or age.possible_minor:
            state.candidate = None
            return age
        if state.candidate == age.value:
            state.candidate = None
            return age
        state.candidate = age.value
        return NO_ESTIMATE

    def _attributes(self, state: _FaceState) -> tuple[Estimate, Estimate]:
        """Declared values of a recognised person override the estimates (SPEC 5)."""
        age, gender = state.age, state.gender
        if state.owner_id is not None:
            try:
                person = self.matcher.registry.get_person(state.owner_id)
            except KeyError:
                return age, gender
            declared_age, declared_gender = _declared(person.age_group), _declared(person.gender)
            if declared_age is not None:
                age = Estimate(declared_age, "declared", 1.0)
            if declared_gender is not None:
                gender = Estimate(declared_gender, "declared", 1.0)
        return age, gender

    def _publish_face(
        self, frame: Frame, camera: CameraModel, track: Track, state: _FaceState
    ) -> None:
        x1, y1, x2, y2 = track.box
        azimuth, elevation = camera.bearing((x1 + x2) / 2, (y1 + y2) / 2)
        age, gender = self._attributes(state)
        percept = FacePercept(
            track_id=track.track_id,
            stamp=frame.stamp,
            box=(
                min(1.0, max(0.0, x1 / camera.width)),
                min(1.0, max(0.0, y1 / camera.height)),
                min(1.0, max(0.0, x2 / camera.width)),
                min(1.0, max(0.0, y2 / camera.height)),
            ),
            score=track.score,
            azimuth_rad=azimuth,
            elevation_rad=elevation,
            owner_id=state.owner_id,
            anonymous_id=None if state.owner_id else state.anonymous_id,
            age_group=age.value,
            age_source=age.source,
            gender=gender.value,
            gender_source=gender.source,
            attribute_confidence=age.confidence,
            possible_minor=age.possible_minor,
            new_track=not state.reported,
            attributes_stamp=state.attr_stamp,
        )
        if self._publish("percept.face", percept, frame.stamp):
            state.reported = True
        owner = state.owner_id
        if owner is not None and owner != state.announced:
            recognized = PersonRecognized(
                track_id=track.track_id,
                person_id=owner,
                stamp=frame.stamp,
                azimuth_rad=azimuth,
                elevation_rad=elevation,
                age_group=age.value,
                possible_minor=age.possible_minor,
            )
            if self._publish("person.recognized", recognized, frame.stamp):
                state.announced = owner
                self._mark_seen(owner)
        obs = Observation(
            kind=PERSON,
            stamp=frame.stamp,
            source=self.source,
            azimuth_rad=azimuth,
            elevation_rad=elevation,
            category=PERSON,
            identity=owner,
            confidence=min(1.0, max(0.0, track.score)),
        )
        self._publish("percept.entity", obs, frame.stamp, joint_state=frame.joint_state)

    def _mark_seen(self, owner_id: str) -> None:
        try:
            self.matcher.registry.mark_seen(owner_id)
        except KeyError:
            log.warning("recognised owner is no longer in the registry")

    # --------------------------------------------------------- animals, objects

    def _process_objects(self, frame: Frame, camera: CameraModel) -> None:
        assert self.object_detector is not None
        kinds: dict[str, Literal["animal", "object"]] = {
            **dict.fromkeys(self.cfg.object_categories, "object"),
            **dict.fromkeys(self.cfg.animal_categories, "animal"),
        }
        dets: list[Detection] = [
            d
            for d in self.object_detector.detect(frame.image)
            if d.score >= self.cfg.object_score_min and d.label in kinds
        ]
        result = self.object_tracker.update(dets, frame.stamp)
        self._lost_held = self._object_lost_s()
        for track_id in result.removed_ids:
            self._drop_entity(track_id)
        states = {
            t.track_id: self._entities.setdefault(t.track_id, _EntityState())
            for t in result.visible
        }
        self._entity_budget = self.cfg.max_entity_embeds_per_frame
        # Oldest embedding first (never embedded first), so a crowd of tracks takes turns
        # and new tracks are not starved by old ones.
        for track in sorted(result.visible, key=lambda t: self._embed_age_key(states[t.track_id])):
            self._evaluate_entity(frame, track, states[track.track_id])
        for track in result.visible:
            state = states[track.track_id]
            x1, y1, x2, y2 = track.box
            azimuth, elevation = camera.bearing((x1 + x2) / 2, (y1 + y2) / 2)
            obs = Observation(
                kind=kinds[track.label],
                stamp=frame.stamp,
                source=self.source,
                azimuth_rad=azimuth,
                elevation_rad=elevation,
                category=self.cfg.object_display_names.get(
                    track.label, track.label.replace("_", " ")
                ),
                identity=state.owner_id,
                confidence=min(1.0, max(0.0, track.score)),
                embedding=state.embedding,
            )
            self._publish("percept.entity", obs, frame.stamp, joint_state=frame.joint_state)

    @staticmethod
    def _embed_age_key(state: _EntityState) -> float:
        return -math.inf if state.last_embed is None else state.last_embed

    def _evaluate_entity(self, frame: Frame, track: Track, state: _EntityState) -> None:
        if self.appearance_embedder is None or self._entity_budget <= 0:
            return
        stamp = frame.stamp
        if state.last_embed is None:
            due = True
        else:
            gap = stamp - state.last_embed
            if state.owner_id is None and state.evals < self.cfg.embedding_burst:
                burst_gap = self.governor.stretch(self.cfg.entity_burst_min_gap_s, CONVERSATION)
                due = gap >= burst_gap - EPSILON_S
            else:
                due = gap >= self.governor.period("appearance")
        if not due:
            return
        self._entity_budget -= 1
        self._entity_embeds += 1
        embedding = self.appearance_embedder.embed(frame.image, track.box)
        identity = self.matcher.observe(
            track.track_id, embedding, self.appearance_embedder.model_id, stamp, kind="entity"
        )
        state.embedding, state.owner_id = embedding, identity.owner_id
        state.evals += 1
        state.last_embed = stamp
