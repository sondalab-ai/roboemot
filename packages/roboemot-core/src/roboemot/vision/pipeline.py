"""Vision pipeline: detect, track, embed, match, publish (SPEC 5, 5A, 6, 12, 13).

``process(frame)`` is synchronous and uses only the capture stamp, so a replay is
deterministic; ``LatestFrameRunner`` drives it live. Per frame, when the governor allows:

1. Faces are detected, filtered by ``face_min_px`` and tracked (stable ``track_id``). Tracks
   unmatched for the larger of ``tracker.lost_s`` and ``face_lost_periods`` detection periods
   at the governor's rate (capture time) expire before association, so a face that comes back
   after a long gap is a new track with no inherited identity.
2. A new track gets ``embedding_burst`` face evaluations, at least ``burst_min_gap_s`` apart
   (divided by the conversation scale only, so a thinking or speaking robot spends less on
   bursts; the circadian and load scales slow bursts only through the detection rate),
   stopping early once it is recognised, then one per ``1 / embedding_hz``. Only confidently
   detected faces of at least ``embed_min_face_px`` are embedded. The burst (5 by default)
   lets one missed vote still fit the SPEC 13 identity budget of 2.5 s with the 3-of-5 votes
   of SPEC 6. The embedding goes straight to ``Matcher.observe`` and is dropped; it is never
   published.
3. Published: ``percept.face`` (no embedding), ``person.recognized`` when a track turns into a
   known person, ``percept.entity`` (kind person, with the camera-frame bearing; the joint state
   at capture travels in the message ``meta`` for the world model to rotate into the base
   frame). Optional object stage: same for animals and objects, whose DINOv2-class appearance
   embedding rides in the ``Observation`` (SPEC 5A) for world-model matching and the registry.
   ``vision.animal_categories`` and ``vision.object_categories`` map detector classes to the
   kind; any other class, ``person`` included, is ignored. No age or gender is estimated (owner
   decision 2026-10-23). On the ticks the governor allows, ``object_gate`` skips object runs
   while the scene is still (``vision.object_gate``, SPEC 5, 13; see ``object_gate``). While a
   face is in view, a run may look at a crop around the faces (``vision.object_crop``, SPEC 5,
   13; see ``object_crop``): its boxes come back in frame pixels, and tracks outside the crop are
   left alone for that run.
   The appearance embedding (DINOv2-S, about 1.1 s on the CM4, inline on this thread) runs only
   when it adds information (``_entity_reason``, SPEC 5, 13): never for a kind and category no
   record holds and auto-enrollment does not collect; once for a new track, but only after it
   has lived ``entity_embed_min_age_s`` since its first detection (a cluttered room churns
   short tracks; a rebirth near a track that ended within ``rebirth_window_s`` keeps that
   track's age and last embedding, never its identity or votes, ``_new_entity``), with a
   burst of votes only while an active record of its kind could match it (again when such
   records change); then at ``rates.appearance_hz`` for a recognised track, and for an unknown
   one only when its detector box changed size or shape, after a missed object run, or after
   ``entity_reembed_floor_s``. During a shown-object episode (``governor.active("objects")``,
   SPEC 5A) only the shown track, the confirmed track nearest
   the attention target's bearing (``governor.focus``; the frame centre without one,
   ``_shown_track``), may embed, and only the votes that can recognise it: the gaze follows it
   without a freeze per run, and the others wait for the episode's end. The look-down crop
   follows the same track. A burst whose last vote aged past ``identity.vote_max_age_s``
   starts again. ``stats`` counts runs and skips by reason (``entity_embeds``).

``light_tap`` (optional) receives each frame's image before anything else, for the circadian
light sampler (SPEC 7A); it measures and keeps only numbers, never the image.

``motion_wanted`` (default never) says whether a day nap waits for motion (SPEC 7A): while it
does, every captured frame goes to the motion watch (``scene_motion``) before the governor can
drop it, with the head pose of the frame, and a confirmed change publishes
``percept.scene_motion`` (a scalar share, never the image). It needs no object stage.

``face_tap`` (optional) is called with each processed frame and its visible face tracks, on
the vision thread, for the enrollment service (SPEC 6); it must not keep the image.

The world model is fed over the bus: ``world.feed_world(bus, world)`` subscribes to
``percept.entity``. The bus keeps vision decoupled from the world model (they may live in
separate processes later) and the in-process bus calls subscribers synchronously, so replay
stays deterministic. A subscriber that raises is logged by type and skipped; it never aborts
the frame. Frames are never logged or written (SPEC 12).

For the debug report (SPEC 12A) ``stats`` adds face track churn (created, ended, the creation
rate over ``vision.report_rate_window_s`` of capture time, the visible lifetime of the last
``vision.report_window`` ended tracks).
``tracks`` also explains the churn (``track_stats``): how confirmed face tracks ended
(``ended_by``), the misses and effective ``lost_s`` behind each expiry, rebirths (a new track
born near where one ended, with gap, IoU, distance and head pose change), and the scores of
low-score detections the tracker refused or left unmatched, beside those that started tracks.
The head pose comes from the frame's joint state through the robot's ``frame_spec``.
``objects`` counts the animal and object detections below ``vision.object_score_min`` handed
to the tracker and the confirmed tracks such a detection continued (the low-score pass).
``rate_scales`` lists the governor's multipliers by source (circadian, load, conversation)
beside their product ``rate_scale``. ``stage_timing`` is what each model stage and the whole
``process`` call cost over ``vision.timing_window_s`` (``StageTimer``, its own clocks: the
values never feed a decision; a stage that raises still counts as a run, its time was spent).
Numbers only; the frame thread writes them and the report thread reads them under one lock.

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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from ..bus import Bus
from ..config.schema import (
    TIMING_WINDOW_MAX_S,
    IdentityConfig,
    ShownObjectConfig,
    TrackerConfig,
    VisionConfig,
    WorldConfig,
)
from ..registry import Matcher
from ..stats import SUMMARY_LEVELS, summarise
from ..world import FrameSpec, Observation, angular_distance
from ..world.frame import to_base_frame
from .facing import head_yaw, smoothed
from .geometry import CameraModel
from .governor import CONVERSATION, EPSILON_S, RateGovernor
from .object_crop import CropSchedule, off_centre
from .object_gate import ObjectGate
from .scene_motion import SceneMotionWatch, ScenePercept
from .stage_timing import APPEARANCE, DETECTION, EMBEDDING, OBJECTS, PROCESS, StageTimer
from .stages import (
    AppearanceEmbedder,
    FaceDetector,
    FaceEmbedder,
    ObjectDetector,
)
from .track_stats import TrackDiagnostics, head_pose
from .tracker import Track, Tracker
from .types import (
    Box,
    Detection,
    FacePercept,
    Frame,
    FrameReport,
    PersonRecognized,
    PreviewOverlay,
)

log = logging.getLogger(__name__)

PERSON = "person"
S_PER_MIN = 60.0
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
    yaws: deque[tuple[float, float]] = field(default_factory=deque)  # (capture stamp, yaw)


@dataclass
class _EntityState:
    kind: str = ""  # animal or object
    votes: int = 0  # burst embeddings voted against registry records since the burst started
    last_embed: float | None = None
    owner_id: str | None = None
    embedding: np.ndarray | None = field(default=None, repr=False)
    box: Box | None = None  # detector box at the last embedding
    last_run: int = 0  # index of the last object run the track was visible on
    reacquired: bool = False  # missed an object run since its last embedding
    born: float = 0.0  # age origin: first detection, or that of the track it is a rebirth of
    seen_box: Box | None = None  # filtered box at its last visible object run
    last_seen: float = 0.0  # capture stamp of that run


@dataclass(frozen=True)
class _EndedEntity:
    """An animal or object track that ended, kept ``rebirth_window_s`` for a rebirth: its
    place, age and last appearance embedding (never its identity or votes)."""

    kind: str
    seen_box: Box
    last_seen: float
    born: float
    last_embed: float | None
    embedding: np.ndarray | None = field(repr=False)
    box: Box | None  # detector box at the last embedding


class _Role(Enum):
    """A track's part in a shown-object episode: none running, the shown track, or another."""

    FREE = "free"
    SHOWN = "shown"
    OTHER = "other"


# Why an animal or object embedding ran, or why a visible track was skipped (debug report).
NEW, BURST, RECOGNISED, CHANGED, REACQUIRED, FLOOR = (
    "new",
    "burst",
    "recognised",
    "changed",
    "reacquired",
    "floor",
)
NO_CONSUMER, SPACING, SETTLED, BUDGET = "no_consumer", "spacing", "settled", "budget"
YOUNG, EPISODE = "young", "episode"
EMBED_RUNS = (NEW, BURST, RECOGNISED, CHANGED, REACQUIRED, FLOOR)
EMBED_SKIPS = (NO_CONSUMER, EPISODE, YOUNG, SPACING, SETTLED, BUDGET)
EntityKey = tuple[str, str]  # (kind, readable category)


def _covers(keys: frozenset[EntityKey], key: EntityKey) -> bool:
    """A record of the same kind matches its own category, or any category when it has none."""
    return key in keys or (key[0], "") in keys


@dataclass(frozen=True)
class _Records:
    """What the registry's animal and object records offer an embedding, keyed by its
    ``version``: ``voters``, ``(kind, entity_id)`` of the active records with embeddings of the
    current model (the matcher votes against these, by kind; a record of another model is left
    out); ``keep``, ``(kind, category)`` of every record, pending ones and other models
    included (a pending catalogue record is confirmed from a seen entry's embedding)."""

    version: int
    model_id: str
    voters: frozenset[tuple[str, str]]
    keep: frozenset[EntityKey]


@dataclass(frozen=True)
class _Consumers:
    """Who uses an animal or object embedding: the matcher (``records.voters``, by kind like
    ``Registry.gallery``, so a cat-or-dog label flicker still matches), a pending catalogue
    record (``records.keep``, by kind and category, as confirm checks) and auto-enrollment
    (the kinds in ``autoenroll``). The world model only breaks ties with it."""

    records: _Records
    autoenroll: frozenset[str]

    def votes(self, key: EntityKey) -> bool:
        return any(kind == key[0] for kind, _ in self.records.voters)

    def wants(self, key: EntityKey) -> bool:
        return key[0] in self.autoenroll or self.votes(key) or _covers(self.records.keep, key)


def _extent(camera: CameraModel, box: Box) -> float:
    """Angular height of a pixel box: the elevation from its top edge to its bottom edge, along
    its centre column (the world model's ``extent_rad``, SPEC 5A)."""
    x1, y1, x2, y2 = box
    _, top = camera.bearing((x1 + x2) / 2, y1)
    _, bottom = camera.bearing((x1 + x2) / 2, y2)
    return abs(top - bottom)


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
        appearance_embedder: AppearanceEmbedder | None = None,
        object_detector: ObjectDetector | None = None,
        governor: RateGovernor | None = None,
        source: str = "vision",
        refresh: bool = True,
        camera_model: CameraProvider | None = None,
        world: WorldConfig | None = None,
        frame_spec: FrameSpec | None = None,
        timer: StageTimer | None = None,
        shown: ShownObjectConfig | None = None,
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
        self._object_visible: list[Track] = []
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
        self.object_tracker = Tracker(self._object_tracker_cfg(), ids, low_confirms=False)
        self.object_gate = ObjectGate(cfg.object_gate)
        self.shown = shown or ShownObjectConfig()
        self.object_crop = CropSchedule(cfg.object_crop)
        self.scene_motion = SceneMotionWatch(cfg.scene_motion, cfg.object_gate)
        self.motion_wanted: Callable[[], bool] = lambda: False
        self._generations = itertools.count(1)
        self._cameras: dict[tuple[int, int], CameraModel] = {}
        self._faces: dict[int, _FaceState] = {}
        self._entities: dict[int, _EntityState] = {}
        self._budget = 0
        self._entity_budget = 0
        self._entity_embeds = 0
        self._object_runs = 0
        self._records: _Records | None = None
        self._ended_entities: deque[_EndedEntity] = deque()
        self._embed_runs = dict.fromkeys(EMBED_RUNS, 0)
        self._embed_skips = dict.fromkeys(EMBED_SKIPS, 0)
        self._report_lock = threading.Lock()
        self._last_stamp: float | None = None
        self._tracks_created = 0
        self._tracks_ended = 0
        self._created_stamps: deque[float] = deque()
        self._lifetimes_s: deque[float] = deque(maxlen=cfg.report_window)
        self._face_lost_held = 0.0
        self._object_low_detections = 0
        self._object_low_continued = 0
        self.timer = timer or StageTimer(TIMING_WINDOW_MAX_S)
        matcher.registry.add_forget_listener(self._on_forget)

    def process(self, frame: Frame) -> FrameReport:
        """Run the stages that are due (``_process``), timed as ``process`` for the report."""
        wall0, cpu0 = self.timer.start()
        try:
            return self._process(frame)
        finally:
            self.timer.record(PROCESS, wall0, cpu0)

    def _process(self, frame: Frame) -> FrameReport:
        """Run the stages that are due for one frame; frames that are not due are dropped.

        Uses only ``frame.stamp`` for time. Publishes on the bus as described in the module
        docstring and returns what ran. Tracks that expired by this stamp are dropped first,
        even when the governor is off, so state never outlives its track. Raises whatever a
        stage raises: ``LatestFrameRunner`` catches it per frame.
        """
        stamp = frame.stamp
        with self._report_lock:
            self._last_stamp = stamp
        self._tap_light(frame)
        self._watch_motion(frame)
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
        if self.object_detector is not None and self.cfg.objects_enabled:
            pose = head_pose(frame.joint_state, self.frame_spec)
            self.object_gate.observe(stamp, pose)
            episode = self.governor.active("objects")
            reason = None
            if self.governor.allow("objects", stamp):
                reason = self.object_gate.decide(
                    stamp,
                    frame.image,
                    pose,
                    self._object_holds(stamp, frame.image.shape[1]),
                    episode=episode,
                )
            if reason is not None:
                self._process_objects(frame, camera, reason, episode)
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

    def _watch_motion(self, frame: Frame) -> None:
        """Hand the frame to the nap's motion watch (SPEC 7A) and publish a confirmed change.
        Inactive, the watch returns at once: no pose is read and no copy is made. A failure
        (the watch or ``motion_wanted``) is logged by type and never costs the frame."""
        try:
            active = self.motion_wanted()
            pose = head_pose(frame.joint_state, self.frame_spec) if active else None
            share = self.scene_motion.offer(frame.stamp, frame.image, pose, active)
        except Exception as exc:
            log.warning("motion watch failed: %s", type(exc).__name__)
            return
        if share is not None:
            self._publish("percept.scene_motion", ScenePercept(share), frame.stamp)

    def reset(self) -> None:
        """Forget every track and its per-track state (start of a session or replay).

        Releases the matcher's votes for each track, restarts track IDs at 1 and resets the
        governor.
        """
        for track_id in (*self._faces, *self._entities):
            self.matcher.release(track_id)
        for state in self._faces.values():
            self._end_face(state)
        self.track_stats.reset(len(self._faces))
        self._faces.clear()
        self._entities.clear()
        self._ended_entities.clear()
        self._object_visible = []
        ids = itertools.count(1)
        self.face_tracker.reset(ids)
        self.object_tracker.reset(ids)
        self._lost_held = 0.0
        self._face_lost_held = 0.0
        self.governor.reset()
        self.object_gate.reset()
        self.object_crop.reset()
        self.scene_motion.reset()

    def apply_config(
        self,
        vision: VisionConfig,
        identity: IdentityConfig,
        world: WorldConfig | None = None,
        shown: ShownObjectConfig | None = None,
    ) -> None:
        """Hot reload (SPEC 11A): thresholds, rates and tracker settings take effect at once, and
        so do, with ``world``, the camera field of view and, with ``shown``, the shown-object
        region the object crop follows. The thread count, the detector size and the models
        folder are restart-only (the schema marks them).
        """
        self.cfg, self.identity = vision, identity
        self.object_crop.cfg = vision.object_crop
        if shown is not None:
            self.shown = shown
        self.track_stats.cfg = vision
        if world is not None:
            self.world = world
        self.governor.rates = vision.rates
        self.object_gate.cfg = vision.object_gate
        self.scene_motion.cfg, self.scene_motion.gate = vision.scene_motion, vision.object_gate
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
        """Lifetime of an animal or object track at the rate the governor allows now. With the
        object gate on, runs may be ``max_interval_s`` apart, so a still thing lives at least
        that plus ``object_lost_periods`` periods and keeps its ID between floor runs."""
        lost = self.cfg.object_lost_s
        period = self.governor.period("objects")
        if math.isfinite(period):
            lost = max(lost, self.cfg.object_lost_periods * period)
            gate = self.cfg.object_gate
            if gate.enabled:
                lost = max(lost, gate.max_interval_s + self.cfg.object_lost_periods * period)
        return lost

    def _object_holds(self, stamp: float, width_px: int) -> frozenset[str]:
        """Why live animal or object tracks keep the object gate on the fixed schedule."""
        gate = self.cfg.object_gate
        return self.object_tracker.holds(
            stamp, self.cfg.animal_categories, gate.moved_share * width_px, gate.moved_hold_s
        )

    def _object_tracker_cfg(self) -> TrackerConfig:
        """Tracker settings for animals and objects, derived from the vision config.

        A detection at ``object_score_min`` starts a track. One between ``object_low_score`` and
        that only continues a confirmed track matched at the object frame before (the low-score
        pass; the tracker is built with ``low_confirms`` off, so it never confirms a tentative
        track). The detector and ``_process_objects`` drop everything below ``object_low_score``.
        A track lives for the larger of ``object_lost_s`` and ``object_lost_periods``
        object-detection periods at the rate the governor allows now, so a slow object stage
        (circadian or load scale) does not expire tracks between two frames. When the rate goes
        back up the longer lifetime is kept until the next object frame has been folded in, so
        tracks do not expire just before it can match them.
        Both centre-distance gates take ``object_match_distance_heights`` (off by default).
        """
        return self.cfg.tracker.model_copy(
            update={
                "high_score": self.cfg.object_score_min,
                "low_score": self.cfg.object_low_score,
                "confirm_hits": self.cfg.object_confirm_hits,
                "match_distance_heights": self.cfg.object_match_distance_heights,
                "low_match_distance_heights": self.cfg.object_match_distance_heights,
                "lost_s": max(self._object_lost_s(), self._lost_held),
            }
        )

    def stats(self) -> dict[str, Any]:
        """Counts for the debug report (SPEC 12A): live tracks, governor scale and drops, stage
        timing and face track churn (see the module docstring).

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
                **self.track_stats.snapshot(),  # report lock, then observer lock: one report
            }
            entity_embeds = {"runs": dict(self._embed_runs), "skips": dict(self._embed_skips)}
            objects = {
                "low_score_detections": self._object_low_detections,
                "low_score_continued": self._object_low_continued,
            }
        return {
            "face_tracks": len(self._faces),
            "entity_tracks": len(self._entities),
            "rate_scale": self.governor.scale,
            "rate_scales": self.governor.scales(),
            "stage_drops": dict(self.governor.dropped),
            "object_gate": self.object_gate.snapshot(),
            "object_input": self.object_crop.snapshot(),
            "scene_motion": self.scene_motion.snapshot(),
            "stage_timing": self.timer.snapshot(self.cfg.timing_window_s),
            "tracks": tracks,
            "objects": objects,
            "entity_embeds": entity_embeds,
        }

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
        """Registry forget listener: tracks of that person become unknown again."""
        for state in self._faces.values():
            if state.owner_id == owner_id:
                state.owner_id = None
            if state.announced == owner_id:
                state.announced = None
        for entity in self._entities.values():
            if entity.owner_id == owner_id:
                entity.owner_id, entity.votes = None, 0

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

    def _end_face(self, state: _FaceState) -> None:
        """Count an ended face track and its visible lifetime."""
        with self._report_lock:
            self._tracks_ended += 1
            self._lifetimes_s.append(state.last_seen - state.first_seen)

    def _new_face(self, track: Track, stamp: float) -> _FaceState:
        """Per-track state of a newly confirmed face track, counted for the churn report."""
        window_s = self.cfg.report_rate_window_s
        with self._report_lock:
            self._tracks_created += 1
            self._created_stamps.append(stamp)
            while self._created_stamps and self._created_stamps[0] < stamp - window_s:
                self._created_stamps.popleft()
        return _FaceState(next(self._generations), first_seen=track.first_stamp)

    def _drop_entity(self, track_id: int) -> None:
        """Forget an ended track's state and votes; keep its place, age and last embedding for
        a rebirth (``_new_entity``)."""
        state = self._entities.pop(track_id, None)
        self.matcher.release(track_id)
        if state is not None and state.seen_box is not None:
            self._ended_entities.append(
                _EndedEntity(
                    state.kind,
                    state.seen_box,
                    state.last_seen,
                    state.born,
                    state.last_embed,
                    state.embedding,
                    state.box,
                )
            )

    def _new_entity(self, track: Track, kind: str, live: list[_EntityState]) -> _EntityState:
        """State of a newly confirmed animal or object track. When it is a rebirth of a track of
        its kind (born after that one's last sighting, at most ``rebirth_window_s`` later, its
        first box centre within ``rebirth_distance_heights`` heights of that one's last box), it
        takes that track's age (``born``), its last embedding, the stamp of it and the detector
        box at it, so it is embedded again only when the usual rules ask (its box changed size
        or shape, it misses a run, the floor; a burst while a record could match, since the
        votes are not carried). It never takes the identity, the owner or the votes. Accepted
        risk (owner, 2026-11-05): another thing put at the same spot within the window keeps
        the old embedding until the box rule fires. Candidates are the tracks that ended in the
        window and the live ones this run missed."""
        cfg = self.cfg
        start = track.first_stamp
        ended = self._ended_entities
        while ended and start - ended[0].last_seen > cfg.rebirth_window_s:
            ended.popleft()
        fx1, fy1, fx2, fy2 = self._measured(track)
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        best: tuple[float, _EndedEntity | _EntityState] | None = None
        for old in [*ended, *live]:
            if old.kind != kind or old.seen_box is None:
                continue
            if not 0 <= start - old.last_seen <= cfg.rebirth_window_s:
                continue
            x1, y1, x2, y2 = old.seen_box
            h = y2 - y1
            d = math.hypot(cx - (x1 + x2) / 2, cy - (y1 + y2) / 2) / h if h > 0 else math.inf
            if d <= cfg.rebirth_distance_heights and (best is None or d < best[0]):
                best = (d, old)
        if best is None:
            return _EntityState(kind, born=start)
        old = best[1]
        if isinstance(old, _EndedEntity):
            ended.remove(old)
        return _EntityState(
            kind,
            last_embed=old.last_embed,
            embedding=old.embedding,
            box=old.box,
            born=min(start, old.born),
        )

    # ------------------------------------------------------------------ faces

    def _process_faces(self, frame: Frame, camera: CameraModel) -> None:
        min_px = self.cfg.face_min_px
        wall0, cpu0 = self.timer.start()
        try:
            found = self.face_detector.detect(frame.image)
        finally:
            self.timer.record(DETECTION, wall0, cpu0)
        dets = [d for d in found if d.min_side >= min_px]
        if len(dets) < len(found):
            self.track_stats.below_min_px(len(found) - len(dets))
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
            self._track_yaw(track, state, frame.stamp)
            self._evaluate_face(frame, track, state)
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
        self._object_visible = []

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

    def object_overlay(self) -> list[tuple[Box, str, str | None]]:
        """The confirmed animal and object tracks of the last object run (on a crop, also those
        drawn before that lay outside it) that have not expired since, as ``(box in pixels,
        readable category, owner_id or None)``: the tracks the world and attention get.
        ``_drop_entity`` removes the entity state of a track that expired or was removed, and
        this filters on that state, so a dropped track is gone before the next object run.
        Object detection is slower than face detection, so a box is the tracker's latest
        estimate, not a detection on the frame it is drawn on."""
        out: list[tuple[Box, str, str | None]] = []
        for track in self._object_visible:
            state = self._entities.get(track.track_id)
            if state is not None:
                out.append((track.box, self.display_name(track.label), state.owner_id))
        return out

    def display_name(self, label: str) -> str:
        """The readable category of a detector class (``vision.object_display_names``)."""
        return self.cfg.object_display_names.get(label, label.replace("_", " "))

    def preview_overlay(self, report: FrameReport) -> PreviewOverlay:
        """The live preview's boxes for the frame ``report`` is about, always with the current
        ``object_overlay``. Face boxes: ``face_overlay`` when face detection ran on it, none
        while face detection is off (governor scale 0 or no detection rate), else unknown
        (``faces_current`` False: the face boxes belong to an earlier frame; the runner keeps
        that one for a poll period, then shows newer frames without face boxes)."""
        objects = tuple(self.object_overlay())
        if report.faces_detected:
            return PreviewOverlay(tuple(self.face_overlay()), objects)
        if math.isinf(self.governor.period("detection")):
            return PreviewOverlay(objects=objects)
        return PreviewOverlay(objects=objects, faces_current=False)

    def _track_yaw(self, track: Track, state: _FaceState, stamp: float) -> None:
        """Add this frame's head yaw (from the matched detection's landmarks) to the track's
        history; a frame without usable landmarks adds nothing."""
        det = track.detection
        cfg = self.cfg
        lm = det.landmarks if det is not None else None
        yaw = head_yaw(lm, cfg.facing_nose_depth_ratio, cfg.facing_min_eye_px)
        if yaw is not None:
            state.yaws.append((stamp, yaw))

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
        wall0, cpu0 = self.timer.start()
        try:
            embedding = self.face_embedder.embed(frame.image, track.detection)
        finally:
            self.timer.record(EMBEDDING, wall0, cpu0)
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

    def _publish_face(
        self, frame: Frame, camera: CameraModel, track: Track, state: _FaceState
    ) -> None:
        x1, y1, x2, y2 = track.box
        azimuth, elevation = camera.bearing((x1 + x2) / 2, (y1 + y2) / 2)
        extent = _extent(camera, track.box)
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
            new_track=not state.reported,
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
            facing_yaw_rad=smoothed(state.yaws, frame.stamp, self.cfg.facing_window_s),
            extent_rad=extent,
        )
        self._publish("percept.entity", obs, frame.stamp, joint_state=frame.joint_state)

    def _mark_seen(self, owner_id: str) -> None:
        try:
            self.matcher.registry.mark_seen(owner_id)
        except KeyError:
            log.warning("recognised owner is no longer in the registry")

    # --------------------------------------------------------- animals, objects

    def _fresh_face_boxes(self, stamp: float) -> list[Box]:
        """Boxes of the faces matched in the last face run, when that run is at most one
        face-detection period before ``stamp`` (the governor runs it at least that often while
        detection is on); none while detection is off."""
        period = self.governor.period("detection")
        if math.isinf(period):
            return []
        return [t.box for t in self._visible if stamp - t.last_stamp <= period]

    def _process_objects(
        self, frame: Frame, camera: CameraModel, reason: str, episode: bool
    ) -> None:
        """One object run on the full frame or on a crop (``object_crop``; ``reason`` is the
        object gate's, ``episode`` a look at a shown thing): boxes in frame pixels, tracked,
        embedded, published. On a crop, tracks outside it are left alone (unless picked up) and
        those drawn before stay drawn."""
        assert self.object_detector is not None
        kinds: dict[str, Literal["animal", "object"]] = {
            **dict.fromkeys(self.cfg.object_categories, "object"),
            **dict.fromkeys(self.cfg.animal_categories, "animal"),
        }
        height, width = frame.image.shape[:2]
        things = [t for t in self._object_visible if t.track_id in self._entities]
        if episode and self.governor.focus("objects") is not None:
            # The look-down crop follows the track the attention target names, not the centre.
            pick = self._shown_track(things, frame, camera)
            things = [t for t in things if t.track_id == pick]
        crop = self.object_crop.choose(
            frame.stamp,
            reason,
            self._fresh_face_boxes(frame.stamp),
            width,
            height,
            self.shown,
            things=[t.box for t in things],
            episode=episode,
        )
        wall0, cpu0 = self.timer.start()
        try:
            if crop is None:
                found = self.object_detector.detect(frame.image)
            else:
                found = crop.to_frame(self.object_detector.detect(crop.view(frame.image)))
        finally:
            self.timer.record(OBJECTS, wall0, cpu0)
        dets: list[Detection] = [
            d for d in found if d.score >= self.cfg.object_low_score and d.label in kinds
        ]
        region = None if crop is None else crop.trusted
        pickup = self.cfg.object_crop.pickup_distance_heights
        result = self.object_tracker.update(dets, frame.stamp, region, pickup)
        self._count_low_objects(dets, result.visible)
        kept = [t for t in result.aside if t in self._object_visible]
        self._object_visible = sorted(result.visible + kept, key=lambda t: t.track_id)
        self._lost_held = self._object_lost_s()
        for track_id in result.removed_ids:
            self._drop_entity(track_id)
        seen = {t.track_id for t in result.visible}
        missed = [s for i, s in self._entities.items() if i not in seen]
        for t in result.visible:
            if t.track_id not in self._entities:
                self._entities[t.track_id] = self._new_entity(t, kinds[t.label], missed)
        states = {t.track_id: self._entities[t.track_id] for t in result.visible}
        self._entity_budget = self.cfg.max_entity_embeds_per_frame
        self._object_runs += 1
        self._follow_aside(result.aside)
        consumers = self._entity_consumers() if self.appearance_embedder is not None else None
        if consumers is not None:
            self._follow_records(consumers.records)
        shown = self._shown_track(result.visible, frame, camera) if episode else None
        # Oldest embedding first (never embedded first), so a crowd of tracks takes turns
        # and new tracks are not starved by old ones.
        for track in sorted(result.visible, key=lambda t: self._embed_age_key(states[t.track_id])):
            key = (kinds[track.label], self.display_name(track.label))
            role = _Role.FREE
            if episode:
                role = _Role.SHOWN if track.track_id == shown else _Role.OTHER
            self._evaluate_entity(frame, track, states[track.track_id], key, consumers, role)
        for track in result.visible:
            state = states[track.track_id]
            state.seen_box, state.last_seen = track.box, frame.stamp
            x1, y1, x2, y2 = track.box
            azimuth, elevation = camera.bearing((x1 + x2) / 2, (y1 + y2) / 2)
            obs = Observation(
                kind=kinds[track.label],
                stamp=frame.stamp,
                source=self.source,
                azimuth_rad=azimuth,
                elevation_rad=elevation,
                category=self.display_name(track.label),
                identity=state.owner_id,
                confidence=min(1.0, max(0.0, track.score)),
                embedding=state.embedding,
                extent_rad=_extent(camera, track.box),
            )
            self._publish("percept.entity", obs, frame.stamp, joint_state=frame.joint_state)

    def _shown_track(self, visible: list[Track], frame: Frame, camera: CameraModel) -> int | None:
        """The track taken as the shown thing during a look at it (SPEC 5, 5A), None while no
        confirmed track is in view. Attention names a world entry, not a track, and passes the
        target's base-frame bearing (``governor.focus``): the shown track is the one whose
        bearing, rotated into the base frame with the frame's joint state, is nearest to it.
        Without a bearing (a caller that sets only the rate) it is the track nearest the frame
        centre, as the head follows the thing. Picked again on every run."""
        focus = self.governor.focus("objects")
        if focus is None:
            width, height = camera.width, camera.height
            near = min(visible, key=lambda t: off_centre(t.box, width, height), default=None)
            return None if near is None else near.track_id
        spec = self.frame_spec or FrameSpec()

        def away(t: Track) -> float:
            x1, y1, x2, y2 = t.box
            az, el = camera.bearing((x1 + x2) / 2, (y1 + y2) / 2)
            az, el = to_base_frame(az, el, frame.joint_state, spec)
            return angular_distance(az, el, *focus)

        near = min(visible, key=away, default=None)
        return None if near is None else near.track_id

    def _follow_aside(self, aside: list[Track]) -> None:
        """A crop run did not look at the tracks left outside the crop: they were not missed,
        so a track seen on every run up to this one keeps an unbroken run of sightings and a
        later sighting is no comeback (``reacquired``); a track already missed stays missed."""
        run = self._object_runs
        for track in aside:
            state = self._entities.get(track.track_id)
            if state is not None and state.last_run == run - 1:
                state.last_run = run

    def _count_low_objects(self, dets: list[Detection], visible: list[Track]) -> None:
        """Count, for the debug report, the animal and object detections below
        ``object_score_min`` handed to the tracker and the confirmed tracks one of them
        continued in this update (a visible track carries the detection it matched now)."""
        start = self.cfg.object_score_min
        low = sum(1 for d in dets if d.score < start)
        # A track starts and confirms only on a strong detection, so a weak score on a visible
        # track is always a low-pass continuation.
        continued = sum(1 for t in visible if t.score < start)
        with self._report_lock:
            self._object_low_detections += low
            self._object_low_continued += continued

    @staticmethod
    def _embed_age_key(state: _EntityState) -> float:
        return -math.inf if state.last_embed is None else state.last_embed

    def _entity_consumers(self) -> _Consumers:
        """Who can use an animal or object embedding now, checked once per object run (records
        and auto-enrollment can change while a track lives); the records are read again only
        when the registry's ``version`` moved."""
        assert self.appearance_embedder is not None
        model = self.appearance_embedder.model_id
        registry = self.matcher.registry
        records = self._records
        if records is None or records.version != registry.version or records.model_id != model:
            version = registry.version
            voters: set[tuple[str, str]] = set()
            keep: set[EntityKey] = set()
            for record in registry.entities():
                keep.add((record.kind, record.category))
                usable = record.embedding_model_id == model and len(record.embeddings)
                if record.status == "active" and usable:
                    voters.add((record.kind, record.entity_id))
            records = _Records(version, model, frozenset(voters), frozenset(keep))
        auto = self.world.autoenroll
        kinds = frozenset(auto.kinds) if auto.enabled else frozenset()
        return _Consumers(records, kinds)

    def _follow_records(self, records: _Records) -> None:
        """When the records a track could match change (one enrolled, confirmed or deleted),
        unknown tracks of that kind vote again with a new burst, and a track whose record is
        gone loses it (its votes are released)."""
        old = self._records
        self._records = records
        if old is None or old.voters == records.voters:
            return
        kinds = {kind for kind, _ in old.voters ^ records.voters}
        live = {entity_id for _, entity_id in records.voters}
        for track_id, state in self._entities.items():
            if state.owner_id is not None and state.owner_id not in live:
                self.matcher.release(track_id)
                state.owner_id, state.votes = None, 0
            elif state.owner_id is None and state.kind in kinds:
                state.votes = 0

    def _evaluate_entity(
        self,
        frame: Frame,
        track: Track,
        state: _EntityState,
        key: EntityKey,
        consumers: _Consumers | None,
        role: _Role = _Role.FREE,
    ) -> None:
        """Embed the track when ``_entity_reason`` says it is due and the frame's entity budget
        allows; count the run or the skip by reason for the debug report. ``role`` is the
        track's part in a shown-object episode. A burst whose last vote is older than
        ``identity.vote_max_age_s`` starts again: the matcher has dropped those votes."""
        run = self._object_runs
        if state.last_run and run - state.last_run > 1:
            state.reacquired = True
        state.last_run = run
        if consumers is None:
            return
        if track.score < self.cfg.object_score_min:  # a weak box is no evidence of identity
            return
        stale = state.last_embed is not None and (
            frame.stamp - state.last_embed > self.identity.vote_max_age_s
        )
        if stale and state.owner_id is None and 0 < state.votes < self.cfg.embedding_burst:
            state.votes = 0
        due, reason = self._entity_reason(track, state, frame.stamp, key, consumers, role)
        if due and self._entity_budget <= 0:
            due, reason = False, BUDGET
        with self._report_lock:
            counts = self._embed_runs if due else self._embed_skips
            counts[reason] += 1
        if not due:
            return
        assert self.appearance_embedder is not None
        voting = state.owner_id is None and consumers.votes(key)
        restart = voting and reason in (CHANGED, REACQUIRED, FLOOR)
        self._entity_budget -= 1
        self._entity_embeds += 1
        stamp = frame.stamp
        wall0, cpu0 = self.timer.start()
        try:
            embedding = self.appearance_embedder.embed(frame.image, track.box)
        finally:
            self.timer.record(APPEARANCE, wall0, cpu0)
        if restart:
            state.votes = 0  # it may look different now: a new burst of votes
        identity = self.matcher.observe(
            track.track_id, embedding, self.appearance_embedder.model_id, stamp, kind="entity"
        )
        if state.owner_id is not None and identity.owner_id is None:
            state.votes = 0  # recognition lost: vote again
        elif voting:
            state.votes += 1
        state.embedding, state.owner_id = embedding, identity.owner_id
        state.last_embed, state.box, state.reacquired = stamp, self._measured(track), False

    @staticmethod
    def _measured(track: Track) -> Box:
        """The detector's box of this run: the filtered box lags a real change by several runs
        and would trigger it twice."""
        return track.detection.box if track.detection is not None else track.box

    def _entity_reason(
        self,
        track: Track,
        state: _EntityState,
        stamp: float,
        key: EntityKey,
        c: _Consumers,
        role: _Role = _Role.FREE,
    ) -> tuple[bool, str]:
        """Whether an animal or object track is due for an embedding, and why (SPEC 5, 13).

        Checked in this order. Nothing uses the embedding of a kind and category that no record
        holds and auto-enrollment does not collect: skipped (``no_consumer``). During a
        shown-object episode every track but the shown one is skipped (``episode``). A new track
        is embedded once, but only after ``entity_embed_min_age_s`` of capture time since its
        first detection (tentative ones included; a rebirth keeps the age of the track it
        continues, ``_new_entity``), else ``young``; it
        gets a burst of votes only while an active record of its kind could match it (again
        after a re-embed of an unknown track, or when such records change). During an episode
        the shown track runs only those votes; anything else is ``episode``. After that a
        recognised track keeps ``rates.appearance_hz``; an unknown one is embedded again only
        when its box changed size or shape, when it was out of sight for an object run, or after
        ``entity_reembed_floor_s``.
        """
        cfg = self.cfg
        if not c.wants(key):
            return False, NO_CONSUMER
        if role is _Role.OTHER:
            return False, EPISODE
        if state.last_embed is None:
            if stamp - state.born < cfg.entity_embed_min_age_s - EPSILON_S:
                return False, YOUNG
            return True, NEW
        gap = stamp - state.last_embed
        if state.owner_id is None and c.votes(key) and state.votes < cfg.embedding_burst:
            burst_gap = self.governor.stretch(cfg.entity_burst_min_gap_s, CONVERSATION)
            return (True, BURST) if gap >= burst_gap - EPSILON_S else (False, SPACING)
        if role is _Role.SHOWN:
            return False, EPISODE
        if gap < self.governor.period("appearance"):
            return False, SPACING
        if state.owner_id is not None:
            return True, RECOGNISED
        if state.reacquired:
            return True, REACQUIRED
        if state.box is not None and self._box_changed(state.box, self._measured(track)):
            return True, CHANGED
        if gap >= cfg.entity_reembed_floor_s - EPSILON_S:
            return True, FLOOR
        return False, SETTLED

    def _box_changed(self, old: Box, new: Box) -> bool:
        """Whether the box area or its width over height changed by the configured factors."""
        ox1, oy1, ox2, oy2 = old
        nx1, ny1, nx2, ny2 = new
        ow, oh, nw, nh = ox2 - ox1, oy2 - oy1, nx2 - nx1, ny2 - ny1
        if min(ow, oh, nw, nh) <= 0:
            return True
        area = (nw * nh) / (ow * oh)
        aspect = (nw / nh) / (ow / oh)
        cfg = self.cfg
        return (
            max(area, 1 / area) >= cfg.entity_reembed_size_ratio
            or max(aspect, 1 / aspect) >= cfg.entity_reembed_aspect_ratio
        )
