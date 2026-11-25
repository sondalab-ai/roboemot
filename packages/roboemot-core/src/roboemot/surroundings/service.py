"""The surroundings service (SPEC 8, Surroundings snapshots; designs 2026-11-24, 2026-11-25).

The feature is on exactly while one route has ``images`` on. A room picture becomes due
(``trigger``) at a wake from sleep, at the end of a rest and after a scene change; one is owed at
start and at switch-on and retried, uncounted, about every ``room.check_interval_s`` until it
passes. Every refusal check runs first: off, ``no_vision``, ``unsupported``, ``paused``,
``daily_limit`` (the cap or a 429 today), ``offline``, ``route_unavailable``, the robot's
``gate`` (``asleep``, ``music``, ``busy``) and ``interval``. A refused trigger is counted and
nothing turns. While due, the service thread checks again every ``POLL_S``, uncounted, and
``picture_due`` (read by the scan context, SPEC 5A) is true only while they pass.

A photo sweep (``world.scan.ScanController``) then dwells at each view. While ``scan_view``
names one, the vision thread's frames (``tap``) go to that view's ``Picker``, and the face boxes
of every face run on that view's frames are kept to mask that view only. At
``world.scan_finished`` with ``photo`` the views are taken; when every planned view has a still,
sharp frame and a face run, the service thread masks them, joins them left to right by
descending azimuth (``tile``), encodes one JPEG and asks the image route for one line of text
(``describe``). Otherwise nothing is sent (``views_missing``) and the picture stays due. Going to
sleep or rest drops the views and the kept picture; at send time sleep, rest or the radio drop
it (the picture stays due), a conversation does not. A reply that arrives after the switch went
off or the route changed (the generation moved) is dropped. HTTP 400 or 415 marks the route
unsupported until its base URL, model or key changes or the switch goes off and on;
``max_failures`` failures in a row pause the pictures for ``pause_s``.

The scene check: with no picture due, the head still and ``min_interval_s`` past, about every
``check_interval_s`` a frame is compared with the sent view within ``room.view_match_deg`` of
the head's yaw; a changed share of ``room.changed_share`` is a trigger.

Privacy (SPEC 12): frames, views, JPEGs and texts stay in RAM. Logs, decision events and
``state`` carry codes, counts, ages, sizes and scores; only ``live`` (the Live card) and the
scene summary read the texts, and only ``jpeg`` the last picture sent.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import APIStatusError, APITimeoutError

from ..bus import Bus, Message
from ..circadian.model import CircadianState
from ..config.schema import RoboEmotConfig, image_route
from ..diag import REASONS, EventSink, no_events
from ..llm.router import ImagesUnsupported, NoRouteAvailable, Offline, RateLimited
from ..vision.scene_change import HeadStill, moved_share, thumbnail
from ..vision.track_stats import head_pose
from ..vision.types import Box, Frame
from ..world import FrameSpec
from .describer import EmptyDescription, describe
from .image import NoEncoder, encode_jpeg, mask_faces, shrink, tile
from .picker import Picker
from .types import Candidate

log = logging.getLogger(__name__)

EVENT_KIND = "surroundings"  # the decision events' kind (roboemot.diag)
OUTCOMES = REASONS[EVENT_KIND]  # every outcome code (roboemot.diag)
OFF = "off"  # a trigger while no route has images on: neither counted nor an event
BUSY = "busy"  # the conversation is not idle: refuses a trigger, never a picture already swept
VIEWS_MISSING = "views_missing"  # a photo sweep ended without a frame for every view
DESCRIBING = "describing"  # the last picture's outcome while its request is open
TIMEOUT, EMPTY, ERROR = "timeout", "empty", "error"  # why a description failed
DAILY_LIMIT = "daily_limit"  # today's cap reached, or the provider said 429 today
RESTING = "resting"
FULL_CIRCLE_DEG = 360.0
POLL_S = 0.1  # how often the service thread checks a due picture again
JOIN_TIMEOUT_S = 2.0
ROUND_DIGITS = 3


@dataclass
class _View:
    """One view of the photo sweep being taken: its picker and the face boxes of every face
    run on its own frames (in the shrunk frame's pixels)."""

    azimuth_rad: float
    picker: Picker
    faces: list[Box] = field(default_factory=list)
    runs: int = 0
    chosen: Candidate | None = None

    def frame(self) -> Candidate | None:
        """The sharpest chosen frame, the open window's included; None without one."""
        found = [c for c in (self.chosen, self.picker.best()) if c is not None]
        return max(found, key=lambda c: c.sharpness, default=None)


@dataclass(frozen=True)
class _Shot:
    azimuth_rad: float
    image: np.ndarray = field(repr=False)
    faces: tuple[Box, ...]
    sharpness: float


@dataclass(frozen=True)
class _Job:
    shots: tuple[_Shot, ...]
    generation: int


class Surroundings:
    """See the module docstring. ``cfg`` gives the configuration at start (``apply_config``
    keeps a copy up to date: the stack validates on every read), ``router`` the live router,
    ``gate()`` a reason code or None from the robot's state, ``scan_view`` the view the head
    dwells at (``ScanController.view``), ``views_planned`` the photo sweep's view count,
    ``vision`` whether frames and OpenCV are there, ``frame_spec`` the head's dofs, ``encode``
    the JPEG encoder (OpenCV's)."""

    def __init__(
        self,
        cfg: Callable[[], RoboEmotConfig],
        router: Callable[[], Any],
        *,
        gate: Callable[[], str | None],
        scan_view: Callable[[], tuple[int, float] | None],
        views_planned: Callable[[], int],
        vision: bool,
        frame_spec: FrameSpec | None,
        clock: Callable[[], float] = time.monotonic,
        encode: Callable[[np.ndarray, int], bytes] = encode_jpeg,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        self.router, self.gate, self.encode, self.today = router, gate, encode, today
        self.scan_view, self.views_planned = scan_view, views_planned
        self.vision, self.frame_spec, self.clock = vision, frame_spec, clock
        self.events: EventSink = no_events
        self._cfg = cfg()
        self._trust = _trust(self._cfg)
        self._lock = threading.Condition()
        # describe requests sent today and the day the provider said 429
        self._day: dt.date | None = None
        self._sent = 0
        self._limited_on: dt.date | None = None
        self._due = False  # a trigger passed: a photo sweep is wanted
        self._due_ok = False  # and the refusal checks still pass (the service thread's poll)
        self._want_first = image_route(self._cfg) is not None  # owed at start and switch-on
        self._first_tried = -math.inf
        self._views: dict[int, _View] = {}
        self._view_index: int | None = None
        self._job: _Job | None = None
        self._describing = False  # a job taken by the service thread is on its way
        self._references: list[tuple[float, np.ndarray]] = []  # (azimuth, thumbnail) per view
        self._room: tuple[str, float] | None = None  # (text, stamp)
        self._jpeg: bytes | None = None
        self._last: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []  # the pictures sent, oldest first (RAM only)
        self._latency_s: float | None = None
        self._counts = dict.fromkeys(OUTCOMES, 0)
        self._unsupported: str | None = None
        self._failures = 0
        self._paused_until = -math.inf
        self._last_room = -math.inf
        self._last_check = -math.inf
        self._generation = 0
        self._head = HeadStill()  # the vision thread's only
        self._rested = False
        self._unsubscribe: list[Callable[[], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ triggers

    def attach(self, bus: Bus) -> None:
        """Subscribe the triggers (``robot.awake``, leaving a rest), the drops
        (``robot.asleep``, entering a rest) and the end of a sweep (``world.scan_finished``)."""
        self._unsubscribe = [
            bus.subscribe("robot.awake", lambda msg: self.trigger()),
            bus.subscribe("robot.asleep", lambda msg: self._sleep()),
            bus.subscribe("circadian.state", self._on_circadian),
            bus.subscribe("world.scan_finished", self._on_scan_finished),
        ]

    def _sleep(self) -> None:
        """The robot sleeps or rests: drop the views, a picture not yet sent (it stays due)
        and the kept picture, and ignore a reply in flight (its generation moves). The texts
        stay."""
        with self._lock:
            if self._job is not None:
                self._due = True
            self._views, self._view_index, self._job = {}, None, None
            self._jpeg = None
            self._generation += 1
            self._lock.notify_all()

    def _on_circadian(self, msg: Message) -> None:
        state = msg.payload
        if not isinstance(state, CircadianState):
            return
        if state.state == RESTING:
            self._rested = True
            self._sleep()
        elif self._rested:
            self._rested = False
            self.trigger()

    def trigger(self, count: bool = True) -> str | None:
        """Make a room picture due; None when it is due (or already on its way), else the
        reason code (also counted and sent as a decision event, except ``off`` and with
        ``count`` false)."""
        with self._lock:
            if self._due or self._job is not None or self._describing:
                return None
        reason = self._refusal(self.clock())
        if reason is None:
            with self._lock:
                self._due = self._due_ok = True
                self._lock.notify_all()
            return None
        if reason != OFF and count:
            self._count(reason)
        return reason

    def picture_due(self) -> bool:
        """Whether a picture is due and could be sent now (SPEC 5A: the scan's photo sweep).
        Lock-held read only: the checks run on the service thread."""
        with self._lock:
            return self._due and self._due_ok

    def _refusal(self, now: float) -> str | None:
        cfg = self._cfg
        route = image_route(cfg)
        if route is None:
            return OFF
        if not self.vision:
            return "no_vision"
        with self._lock:
            if self._unsupported is not None:
                return "unsupported"
            if now < self._paused_until:
                return "paused"
            if self._capped():
                return DAILY_LIMIT
            last_room = self._last_room
        router = self.router()
        if router.offline:
            return "offline"
        if route.name not in {r.name for r in router.eligible()}:
            return "route_unavailable"
        reason = self.gate()
        if reason is not None:
            return reason
        if now - last_room < cfg.surroundings.room.min_interval_s:
            return "interval"
        return None

    def _capped(self) -> bool:
        """Whether today's cap is reached (lock held): the provider said 429 today, or
        ``max_per_day`` requests went out. A new day resets the count."""
        today = self.today()
        if today != self._day:
            self._day, self._sent = today, 0
        cap = self._cfg.surroundings.max_per_day
        return self._limited_on == today or bool(cap and self._sent >= cap)

    # ------------------------------------------------------------- vision thread

    def wanted(self) -> bool:
        """Whether the vision thread should call ``tap``: a due picture's view is being taken,
        or the scene check is due."""
        view = self.scan_view()
        now = self.clock()
        s = self._cfg.surroundings
        with self._lock:
            if view is not None:
                return self._due
            return (
                not self._due
                and self._job is None
                and bool(self._references)
                and image_route(self._cfg) is not None
                and self.vision
                and self._unsupported is None
                and now >= self._paused_until
                and now - self._last_room >= s.room.min_interval_s
                and now - self._last_check >= s.room.check_interval_s
            )

    def tap(self, frame: Frame, faces: list[Box] | None) -> None:
        """One frame from the vision thread (SPEC 5) with the boxes of this frame's face run
        (None: no face run on it): offer it to the view the head dwells at, or run the scene
        check."""
        cfg = self._cfg
        pose = head_pose(frame.joint_state, self.frame_spec)
        if pose is not None:
            self._head.observe(
                frame.stamp, pose, math.radians(cfg.vision.object_gate.head_still_deg)
            )
        moving = pose is not None and self._head.moving(frame.stamp, cfg.surroundings.settle_s)
        view = self.scan_view()
        if view is not None:
            self._take(view, frame, faces, moving)
        elif pose is not None and not moving:
            self._check_scene(frame, faces, pose[0])

    def _take(
        self, view: tuple[int, float], frame: Frame, faces: list[Box] | None, moving: bool
    ) -> None:
        """Offer the frame to view ``view`` (index, azimuth); a view entered again starts
        over."""
        index, azimuth = view
        small, boxes = shrink(frame.image, faces or [], self._cfg.surroundings.max_side_px)
        if not moving and small is frame.image:
            small = small.copy()  # kept past this frame: never the pipeline's buffer
        with self._lock:
            if not self._due:
                return
            entry = self._views.get(index)
            if entry is None or self._view_index != index:
                entry = self._views[index] = _View(azimuth, Picker(lambda: self._cfg.surroundings))
                self._view_index = index
            if faces is not None:
                entry.runs += 1
                entry.faces.extend(boxes)
            entry.picker.offer(small, frame.stamp, head_moving=moving)
            chosen = entry.picker.result(frame.stamp)
            if chosen is not None and (
                entry.chosen is None or chosen.sharpness > entry.chosen.sharpness
            ):
                entry.chosen = chosen

    def _on_scan_finished(self, msg: Message) -> None:
        """``world.scan_finished`` (the motion loop's thread): take the views of a photo sweep
        for the service thread when every planned view has a frame and a face run, else count
        ``views_missing`` (the picture stays due). Any other sweep drops stray views."""
        payload = msg.payload
        photo = isinstance(payload, Mapping) and payload.get("photo") is True
        planned = self.views_planned()
        with self._lock:
            views, self._views, self._view_index = self._views, {}, None
            if not photo or not self._due:
                return
            shots = [s for i in range(planned) if (s := _shot(views.get(i))) is not None]
            complete = planned > 0 and len(shots) == planned
            if complete:
                self._due = self._due_ok = False
                self._job = _Job(tuple(shots), self._generation)
                self._lock.notify_all()
        if not complete:
            self._count(VIEWS_MISSING)

    def _thumb(self, image: np.ndarray) -> np.ndarray | None:
        gate = self._cfg.vision.object_gate
        return thumbnail(image, gate.thumb_width_px, gate.thumb_height_px)

    def _check_scene(self, frame: Frame, faces: list[Box] | None, yaw: float) -> None:
        """With the head still and the check due: compare the frame with the sent view nearest
        the head's ``yaw``, when it is within ``room.view_match_deg``; a changed share of
        ``room.changed_share`` is a trigger."""
        if not self.wanted():
            return
        s = self._cfg.surroundings
        with self._lock:
            self._last_check = self.clock()
            references = self._references
        near = min(references, key=lambda r: _gap(r[0], yaw), default=None)
        if near is None or _gap(near[0], yaw) > math.radians(s.room.view_match_deg):
            return
        small, boxes = shrink(frame.image, faces or [], s.max_side_px)
        thumb = self._thumb(mask_faces(small, boxes))
        if thumb is None or thumb.shape != near[1].shape:
            return
        if moved_share(thumb, near[1], self._cfg.vision.object_gate.pixel_diff) >= (
            s.room.changed_share
        ):
            self.trigger()

    # ------------------------------------------------------------ service thread

    def start(self, bus: Bus | None = None) -> None:
        """Attach the triggers (with ``bus``) and start the service thread (once)."""
        if self._thread is not None:
            return
        if bus is not None:
            self.attach(bus)
        self._thread = threading.Thread(target=self._run, name="roboemot-surroundings", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if self._job is None:
                    self._lock.wait(POLL_S)
            try:
                self.step()
            except Exception as exc:  # a failure never stops the thread
                log.warning("surroundings: step failed: %s", type(exc).__name__)

    def step(self) -> None:
        """Send a swept picture, if any; then check a due picture again, or retry the first
        one owed."""
        with self._lock:
            job, self._job = self._job, None
            self._describing = job is not None
        if job is not None:
            try:
                self._send(job)
            finally:
                with self._lock:
                    self._describing = False
                    self._lock.notify_all()
        self._refresh(self.clock())

    def _refresh(self, now: float) -> None:
        with self._lock:
            due, first = self._due, self._want_first
            retry = now - self._first_tried >= self._cfg.surroundings.room.check_interval_s
        if due:
            ok = self._refusal(now) is None
            with self._lock:
                if self._due:
                    self._due_ok = ok
        elif first and retry:
            with self._lock:
                self._first_tried = now
            if self.trigger(count=False) is None:
                with self._lock:
                    self._want_first = False

    def _send(self, job: _Job) -> None:
        reason = self.gate()  # checked again right before the picture leaves
        if reason is not None and reason != BUSY:
            self._count(reason)
            with self._lock:
                self._due = image_route(self._cfg) is not None
            return
        try:
            jpeg, references = self._compose(job)
        except NoEncoder:
            self.vision = False
            self._count("no_vision")
            return
        self._describe(job, jpeg, references)

    def _compose(self, job: _Job) -> tuple[bytes, list[tuple[float, np.ndarray]]]:
        """The joined JPEG (views masked, left to right by descending azimuth: positive yaw
        turns left) and the thumbnail of each view for the scene check."""
        s = self._cfg.surroundings
        shots = sorted(job.shots, key=lambda shot: shot.azimuth_rad, reverse=True)
        masked = [mask_faces(shot.image, shot.faces) for shot in shots]
        joined = tile(masked, s.room.separator_px, min(s.room.view_height_px, s.max_side_px))
        jpeg = self.encode(joined, s.jpeg_quality)
        references = [
            (shot.azimuth_rad, thumb)
            for shot, image in zip(shots, masked, strict=True)
            if (thumb := self._thumb(image)) is not None
        ]
        return jpeg, references

    def _describe(self, job: _Job, jpeg: bytes, references: list[tuple[float, np.ndarray]]) -> None:
        cfg = self._cfg
        route = image_route(cfg)
        views = len(job.shots)
        span = _span_deg([shot.azimuth_rad for shot in job.shots], cfg.world.camera_fov_h_deg)
        started = self.clock()
        text, unsupported = None, None
        with self._lock:
            if route is None or job.generation != self._generation:
                return  # the switch went off, the route changed or the robot slept meanwhile
            self._capped()  # rolls the day before counting
            self._sent += 1
            self._last_room = started
            self._jpeg, self._references = jpeg, references
            self._last = {
                "views": views,
                "span_deg": span,
                "stamp": started,
                "bytes": len(jpeg),
                "sharpness": round(min(shot.sharpness for shot in job.shots), ROUND_DIGITS),
                "outcome": DESCRIBING,
                "detail": None,
            }
        detail: str | None = None
        try:
            text = asyncio.run(describe(self.router(), route, jpeg, views, span, cfg))
            outcome = "described"
        except ImagesUnsupported as exc:
            outcome, unsupported, detail = "unsupported", exc.code, exc.code
        except RateLimited as exc:
            outcome, detail = DAILY_LIMIT, exc.code
        except Offline:
            outcome = "offline"
        except NoRouteAvailable:
            outcome = "route_unavailable"
        except Exception as exc:
            outcome, detail = "failed", _failure(exc)
            if not isinstance(exc, EmptyDescription):
                log.warning("surroundings: describe failed: %s", type(exc).__name__)
        now = self.clock()
        with self._lock:
            if job.generation != self._generation:
                return  # the switch went off or the route changed meanwhile
            self._latency_s = now - started
            if self._last is not None:
                self._last.update(outcome=outcome, detail=detail)
            self._remember(now, outcome, detail, text, views, span)
            if text is not None:
                self._failures = 0
                self._room = (text, now)
            elif unsupported is not None:
                self._unsupported = unsupported
            elif outcome == DAILY_LIMIT:
                self._limited_on = self.today()
            elif outcome == "failed":
                self._failures += 1
                if self._failures >= self._cfg.surroundings.max_failures:
                    self._failures = 0
                    self._paused_until = now + self._cfg.surroundings.pause_s
            self._lock.notify_all()
        self._count(outcome)

    def _remember(
        self, now: float, outcome: str, detail: str | None, text: str | None, views: int, span: int
    ) -> None:
        """Add a picture sent to the history (lock held), at most ``history_size``."""
        self._history.append(
            {
                "stamp": now,
                "views": views,
                "span_deg": span,
                "outcome": outcome,
                "detail": detail,
                "text": text,
            }
        )
        del self._history[: max(0, len(self._history) - self._cfg.surroundings.history_size)]

    def close(self) -> None:
        """Stop the thread and drop every subscription."""
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe = []
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(JOIN_TIMEOUT_S)

    # ------------------------------------------------------------------ readers

    def room(self, now: float) -> str | None:
        """The room description, None without one or past ``room.max_age_s``."""
        with self._lock:
            room = self._room
        if room is None or now - room[1] > self._cfg.surroundings.room.max_age_s:
            return None
        return room[0]

    def jpeg(self) -> bytes | None:
        """The last picture sent (RAM only)."""
        with self._lock:
            return self._jpeg

    def state(self) -> dict[str, Any]:
        """Codes, counts, ages, sizes and scores (SPEC 12A): no text and no image."""
        now = self.clock()
        route = image_route(self._cfg)
        reach = self._reach(route)
        with self._lock:
            if route is None:
                code = OFF
            elif not self.vision:
                code = "no_vision"
            elif self._unsupported is not None:
                code = "unsupported"
            elif now < self._paused_until:
                code = "paused"
            elif self._capped():
                code = DAILY_LIMIT
            else:
                code = reach or "on"
            last = None
            if self._last is not None:
                last = {k: v for k, v in self._last.items() if k != "stamp"}
                last["age_s"] = round(now - self._last["stamp"], ROUND_DIGITS)
            return {
                "state": code,
                "route": route.name if route is not None else None,
                "due": self._due,
                "views": len(self._views),
                "counts": dict(self._counts),
                "last": last,
                "latency_s": None
                if self._latency_s is None
                else round(self._latency_s, ROUND_DIGITS),
                "unsupported": self._unsupported,
                "paused_s": round(max(0.0, self._paused_until - now), ROUND_DIGITS),
            }

    def _reach(self, route: Any) -> str | None:
        """``offline`` or ``route_unavailable`` (key missing, budget spent, unreachable) for
        the image route now, None when a request could go out."""
        if route is None:
            return None
        router = self.router()
        if router.offline:
            return "offline"
        if route.name not in {r.name for r in router.eligible()}:
            return "route_unavailable"
        return None

    def live(self) -> dict[str, Any]:
        """``state`` plus the texts (``described``), for the Live card only."""
        now = self.clock()
        with self._lock:
            history = [
                {**{k: v for k, v in h.items() if k != "stamp"},
                 "age_s": round(now - h["stamp"], ROUND_DIGITS)}
                for h in reversed(self._history)
            ]  # fmt: skip
        return {**self.state(), "described": {"room": self.room(now)}, "history": history}

    # ------------------------------------------------------------------- config

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """A stack subscriber: keep the copy. The switch going off, or the image route's base
        URL, model or key changing, drops the due picture, the views, the texts and the image,
        and ignores a reply still in flight; a changed route also clears the unsupported mark
        and the pause, and owes a first picture again when the switch is on."""
        trust = _trust(cfg)
        with self._lock:
            self._cfg = cfg
            if trust == self._trust:
                return
            self._trust = trust
            self._want_first = trust is not None
            self._first_tried = -math.inf
            self._unsupported, self._failures, self._paused_until = None, 0, -math.inf
            self._generation += 1
            self._due = self._due_ok = False
            self._views, self._view_index, self._job = {}, None, None
            self._room = None
            self._jpeg, self._last, self._references = None, None, []
            self._history = []
            self._lock.notify_all()

    def _count(self, outcome: str) -> None:
        with self._lock:
            self._counts[outcome] = self._counts.get(outcome, 0) + 1
        log.debug("surroundings: %s", outcome)
        self.events(EVENT_KIND, outcome)


def _shot(view: _View | None) -> _Shot | None:
    """The view's frame for the picture: None without a still, sharp frame or without a face
    run on its own frames (a face could go out uncovered)."""
    if view is None or view.runs == 0:
        return None
    frame = view.frame()
    if frame is None:
        return None
    return _Shot(view.azimuth_rad, frame.image, tuple(view.faces), frame.sharpness)


def _gap(a: float, b: float) -> float:
    """The angle between two bearings (rad), wrapped: -179 and 179 degrees are 2 apart."""
    return abs(math.remainder(a - b, 2 * math.pi))


def _span_deg(azimuths: list[float], fov_h_deg: float) -> int:
    """The span the views cover: their azimuth range plus one camera view, at most a circle."""
    spread = math.degrees(max(azimuths) - min(azimuths))
    return round(min(FULL_CIRCLE_DEG, spread + fov_h_deg))


def _trust(cfg: RoboEmotConfig) -> tuple[str, str, str, str] | None:
    """What the unsupported mark and the kept texts are bound to: the image route's name, base
    URL, model and key reference."""
    route = image_route(cfg)
    if route is None:
        return None
    return route.name, route.base_url, route.model, route.api_key


def _failure(exc: Exception) -> str:
    """Why a description failed, as a code (never the server's text): ``timeout``, ``empty``
    (a blank reply, often a model that thought through its whole budget), ``http_<status>``
    or ``error``."""
    if isinstance(exc, EmptyDescription):
        return EMPTY
    if isinstance(exc, TimeoutError | APITimeoutError):
        return TIMEOUT
    if isinstance(exc, APIStatusError):
        return f"http_{exc.status_code}"
    return ERROR
