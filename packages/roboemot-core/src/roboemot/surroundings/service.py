"""The surroundings service (SPEC 8, Surroundings snapshots; design 2026-11-24).

The feature is on exactly while one route has ``images`` on. A trigger asks for a snapshot
(``request``): a shown thing (the expressor, on a new shown look), the room (a wake from sleep or
rest, the end of a scan sweep, a scene change against the last room snapshot with the head
still). A request
is refused with a reason code when the feature is off or cannot run (``no_vision``,
``unsupported``, ``paused``, ``offline``, ``route_unavailable``), when the robot sleeps, rests or
the quiet hours run (``asleep``), the radio plays (``music``), a room snapshot finds the
conversation busy (``busy``) or a room snapshot came too soon (``interval``). A new shown request
replaces an open attempt.

An open attempt takes frames from the vision thread (``tap``): each is shrunk, its detected faces
masked, a shown thing cropped, and offered to the ``Picker`` with the head and thing motion. The
chosen frame is encoded once and handed to the service thread (``step``), which asks the image
route for one line of text (``describe``). A reply that arrives after the switch went off or the
route changed (the generation moved) is dropped. HTTP 400 or 415 marks the route unsupported until
its base URL, model or key changes or the switch goes off and on; ``max_failures`` failures in a
row pause the snapshots for ``pause_s``. The gate is checked again before a frame is chosen and
before it is sent; going to sleep or rest drops the attempt, the chosen frame and the kept image.

Privacy (SPEC 12): frames, crops, JPEGs and texts stay in RAM. Logs, decision events and
``state`` carry codes, counts, ages, sizes and scores; only ``live`` (the Live card) and the
scene summary read the texts, and only ``jpeg`` the last image sent.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

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
from .image import NoEncoder, crop, encode_jpeg, mask_faces, shrink
from .picker import Picker
from .types import Candidate, Kind

log = logging.getLogger(__name__)

EVENT_KIND = "surroundings"  # the decision events' kind (roboemot.diag)
OUTCOMES = REASONS[EVENT_KIND]  # every outcome code (roboemot.diag)
OFF = "off"  # a request while no route has images on: neither counted nor an event
DESCRIBING = "describing"  # the last snapshot's outcome while its request is open
TIMEOUT, EMPTY, ERROR = "timeout", "empty", "error"  # why a description failed
DAILY_LIMIT = "daily_limit"  # today's cap reached, or the provider said 429 today
RESTING = "resting"
POLL_S = 0.1  # how often the service thread looks for an attempt past its wait
JOIN_TIMEOUT_S = 2.0
ROUND_DIGITS = 3


@dataclass
class _Attempt:
    kind: Kind
    token: str
    hint: str
    picker: Picker
    generation: int


@dataclass
class _Sent:
    kind: Kind
    token: str
    hint: str
    jpeg: bytes
    sharpness: float
    generation: int


class Surroundings:
    """See the module docstring. ``cfg`` gives the configuration at start (``apply_config``
    keeps a copy up to date: the stack validates on every read), ``router`` the live router,
    ``gate(kind)`` a reason code or None from the robot's state, ``speed(entry_id)`` a shown
    thing's angular speed (deg/s) or None, ``vision`` whether frames and OpenCV are there,
    ``frame_spec`` the head's dofs, ``encode`` the JPEG encoder (OpenCV's)."""

    def __init__(
        self,
        cfg: Callable[[], RoboEmotConfig],
        router: Callable[[], Any],
        *,
        gate: Callable[[Kind], str | None],
        speed: Callable[[str], float | None],
        vision: bool,
        frame_spec: FrameSpec | None,
        clock: Callable[[], float] = time.monotonic,
        encode: Callable[[np.ndarray, int], bytes] = encode_jpeg,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        self.router, self.gate, self.speed, self.encode = router, gate, speed, encode
        self.today = today
        # describe requests sent today, all and room ones, and the day the provider said 429
        self._day: dt.date | None = None
        self._sent_all = self._sent_room = 0
        self._limited_on: dt.date | None = None
        self.vision, self.frame_spec, self.clock = vision, frame_spec, clock
        self.events: EventSink = no_events
        self._cfg = cfg()
        self._trust = _trust(self._cfg)
        self._lock = threading.Condition()
        self._attempt: _Attempt | None = None
        self._pending: _Sent | None = None
        self._shown: tuple[str, str] | None = None  # (token, text)
        self._room: tuple[str, float] | None = None  # (text, stamp)
        self._jpeg: bytes | None = None
        self._last: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []  # the snapshots sent, oldest first (RAM only)
        self._latency_s: float | None = None
        self._counts = dict.fromkeys(OUTCOMES, 0)
        self._unsupported: str | None = None
        self._failures = 0
        self._paused_until = -math.inf
        self._last_room = -math.inf
        self._last_check = -math.inf
        self._reference: np.ndarray | None = None
        self._generation = 0
        self._head = HeadStill()  # the vision thread's only
        self._rested = False
        self._describing: str | None = None  # the token of the describe request in flight
        # a first room snapshot is owed at start and at switch-on (owner, 2026-11-25)
        self._want_first = image_route(self._cfg) is not None
        self._unsubscribe: list[Callable[[], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ triggers

    def attach(self, bus: Bus) -> None:
        """Subscribe the room triggers (``robot.awake``, leaving a rest, ``world.scan_finished``)
        and the drops (``robot.asleep``, entering a rest)."""
        self._unsubscribe = [
            bus.subscribe("robot.awake", lambda msg: self.request("room")),
            bus.subscribe("robot.asleep", lambda msg: self._sleep()),
            bus.subscribe("circadian.state", self._on_circadian),
            bus.subscribe("world.scan_finished", lambda msg: self.request("room")),
        ]

    def _sleep(self) -> None:
        """The robot sleeps or rests: drop the open attempt, the chosen frame and the kept
        snapshot, and ignore a reply in flight (its generation moves). The texts stay."""
        with self._lock:
            self._attempt = self._pending = None
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
            self.request("room")

    def request(
        self, kind: Kind, token: str = "", hint: str = "", count: bool = True
    ) -> str | None:
        """Open a snapshot attempt; None when it opened, else the reason code (also counted
        and sent as a decision event, except ``off`` and with ``count`` false)."""
        now = self.clock()
        reason = self._refusal(kind, now)
        if reason is None:
            with self._lock:
                if kind == "room" and (
                    self._attempt is not None
                    or self._pending is not None
                    or self._describing is not None
                ):
                    reason = "busy"
                else:
                    self._open(kind, token, hint, now)
                    return None
        if reason != OFF and count:
            self._count(reason, kind)
        return reason

    def _refusal(self, kind: Kind, now: float) -> str | None:
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
            if self._capped(kind):
                return DAILY_LIMIT
            last_room = self._last_room
        router = self.router()
        if router.offline:
            return "offline"
        if route.name not in {r.name for r in router.eligible()}:
            return "route_unavailable"
        reason = self.gate(kind)
        if reason is not None:
            return reason
        if kind == "room" and now - last_room < cfg.surroundings.room.min_interval_s:
            return "interval"
        return None

    def _capped(self, kind: Kind | None) -> bool:
        """Whether today's cap is reached (lock held): the provider said 429 today, or
        ``max_per_day`` (and for a room snapshot ``room.max_per_day``) requests went out;
        ``kind`` None checks the overall cap only. A new day resets the counts."""
        today = self.today()
        if today != self._day:
            self._day, self._sent_all, self._sent_room = today, 0, 0
        s = self._cfg.surroundings
        if self._limited_on == today or (s.max_per_day and self._sent_all >= s.max_per_day):
            return True
        return kind == "room" and bool(s.room.max_per_day) and self._sent_room >= s.room.max_per_day

    def _open(self, kind: Kind, token: str, hint: str, now: float) -> None:
        """Open an attempt (lock held); a shown attempt replaces any open attempt."""
        s = self._cfg.surroundings
        wait_s = s.shown.wait_s if kind == "shown" else s.room.wait_s
        picker = Picker(lambda: self._cfg.surroundings, wait_s, now)
        self._attempt = _Attempt(kind, token, hint, picker, self._generation)
        if kind == "room":
            self._last_room = now
        elif self._shown is not None and self._shown[0] != token:
            self._shown = None
        self._lock.notify_all()

    def cancel(self, token: str) -> None:
        """Close the open shown attempt for ``token`` (its reaction was not queued): nothing is
        sent for it."""
        with self._lock:
            if self._attempt is not None and self._attempt.token == token:
                self._attempt = None
                self._lock.notify_all()

    def hold_until(self) -> float | None:
        """When the head hold of the open attempt ends (its start plus ``surroundings.hold_s``),
        None without an open attempt or with ``hold_s`` 0 (owner, 2026-11-25)."""
        hold_s = self._cfg.surroundings.hold_s
        with self._lock:
            attempt = self._attempt
        if attempt is None or hold_s <= 0:
            return None
        return attempt.picker.start + hold_s

    def holding(self, now: float) -> bool:
        """Whether the head should hold still at ``now``."""
        until = self.hold_until()
        return until is not None and now < until

    # ------------------------------------------------------------- vision thread

    def wanted(self) -> bool:
        """Whether the vision thread should call ``tap``: an attempt is open, or the room's
        scene check is due."""
        now = self.clock()
        s = self._cfg.surroundings
        with self._lock:
            if self._attempt is not None:
                return True
            return (
                image_route(self._cfg) is not None
                and self.vision
                and (self._reference is not None or self._want_first)
                and self._unsupported is None
                and now >= self._paused_until
                and now - self._last_room >= s.room.min_interval_s
                and now - self._last_check >= s.room.check_interval_s
            )

    def tap(self, frame: Frame, faces: list[Box], shown: Box | None) -> None:
        """One frame from the vision thread (SPEC 5): offer it to the open attempt, or run the
        room's scene check."""
        cfg = self._cfg
        s = cfg.surroundings
        pose = head_pose(frame.joint_state, self.frame_spec)
        if pose is not None:
            self._head.observe(
                frame.stamp, pose, math.radians(cfg.vision.object_gate.head_still_deg)
            )
        moving = pose is not None and self._head.moving(frame.stamp, s.settle_s)
        with self._lock:
            attempt = self._attempt
        if attempt is None:
            self._check_scene(frame, faces, moving)
            return
        image, thing_moving = self._shape(attempt, frame, faces, shown)
        with self._lock:
            if self._attempt is not attempt:
                return
            attempt.picker.offer(image, frame.stamp, head_moving=moving, object_moving=thing_moving)
            outcome = self._settle(attempt, frame.stamp)
        if outcome is not None:
            self._count(outcome, attempt.kind)

    def _shape(
        self, attempt: _Attempt, frame: Frame, faces: list[Box], shown: Box | None
    ) -> tuple[np.ndarray | None, bool]:
        """The frame as it would be sent (shrunk, faces masked, a shown thing cropped) and
        whether the shown thing moves too fast."""
        s = self._cfg.surroundings
        if attempt.kind == "room":
            small, boxes = shrink(frame.image, faces, s.max_side_px)
            return mask_faces(small, boxes), False
        if shown is None:
            return None, False
        small, boxes = shrink(frame.image, [*faces, shown], s.max_side_px)
        image = crop(mask_faces(small, boxes[:-1]), boxes[-1], s.shown.margin)
        speed = self.speed(attempt.token)
        return image, speed is not None and speed > s.shown.max_speed_deg_s

    def _settle(self, attempt: _Attempt, now: float) -> str | None:
        """Close ``attempt`` when its picker decided (lock held): the chosen frame goes to the
        service thread; a skip reason is returned for counting."""
        result = attempt.picker.result(now)
        if result is None:
            return None
        self._attempt = None
        if not isinstance(result, Candidate):
            return result
        reason = self.gate(attempt.kind)  # the robot's state may have changed since the request
        if reason is not None:
            return reason
        try:
            jpeg = self.encode(result.image, self._cfg.surroundings.jpeg_quality)
        except NoEncoder:
            self.vision = False
            return "no_vision"
        if attempt.kind == "room":
            self._reference = self._thumb(result.image)
        self._pending = _Sent(
            attempt.kind, attempt.token, attempt.hint, jpeg, result.sharpness, attempt.generation
        )
        self._lock.notify_all()
        return None

    def _thumb(self, image: np.ndarray) -> np.ndarray | None:
        gate = self._cfg.vision.object_gate
        return thumbnail(image, gate.thumb_width_px, gate.thumb_height_px)

    def _check_scene(self, frame: Frame, faces: list[Box], moving: bool) -> None:
        """With the head still and the room check due: the first room snapshot still owed (at
        start or switch-on; its refusals while the robot is busy are not counted, it tries again
        at the next check), or a view that changed by ``room.changed_share`` since the last room
        snapshot, asks for one."""
        if moving or not self.wanted():
            return
        s = self._cfg.surroundings
        small, boxes = shrink(frame.image, faces, s.max_side_px)
        thumb = self._thumb(mask_faces(small, boxes))
        with self._lock:
            self._last_check = self.clock()
            reference, first = self._reference, self._want_first
        if reference is None and first:
            if self.request("room", count=False) is None:
                with self._lock:
                    self._want_first = False
            return
        if thumb is None:
            return
        if reference is None or reference.shape != thumb.shape:
            return
        share = moved_share(thumb, reference, self._cfg.vision.object_gate.pixel_diff)
        if share >= s.room.changed_share:
            self.request("room")

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
                if self._pending is None:
                    self._lock.wait(POLL_S)
            try:
                self.step()
            except Exception as exc:  # a failure never stops the thread
                log.warning("surroundings: step failed: %s", type(exc).__name__)

    def step(self) -> None:
        """Close an attempt past its wait, then describe the chosen frame, if any."""
        now = self.clock()
        with self._lock:
            attempt = self._attempt
            expired = self._settle(attempt, now) if attempt is not None else None
            sent, self._pending = self._pending, None
        if expired is not None and attempt is not None:
            self._count(expired, attempt.kind)
        if sent is not None:
            self._describe(sent)

    def _describe(self, sent: _Sent) -> None:
        route = image_route(self._cfg)
        reason = self.gate(sent.kind)  # checked again right before the image leaves
        if reason is not None:
            self._count(reason, sent.kind)
            return
        started = self.clock()
        text, unsupported = None, None
        with self._lock:
            if route is None or sent.generation != self._generation:
                return  # the switch went off, the route changed or the robot slept meanwhile
            self._describing = sent.token
            self._capped(None)  # rolls the day before counting
            self._sent_all += 1
            self._sent_room += sent.kind == "room"
            self._jpeg = sent.jpeg
            self._last = {
                "kind": sent.kind,
                "stamp": started,
                "bytes": len(sent.jpeg),
                "sharpness": round(sent.sharpness, ROUND_DIGITS),
                "outcome": DESCRIBING,
                "detail": None,
            }
        detail: str | None = None
        try:
            text = asyncio.run(
                describe(self.router(), route, sent.kind, sent.jpeg, sent.hint, self._cfg)
            )
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
            self._describing = None
            self._lock.notify_all()
            if sent.generation != self._generation:
                return  # the switch went off or the route changed meanwhile
            self._latency_s = now - started
            if self._last is not None:
                self._last.update(outcome=outcome, detail=detail)
            self._remember(sent.kind, now, outcome, detail, text)
            if text is not None:
                self._failures = 0
                if sent.kind == "shown":
                    self._shown = (sent.token, text)
                else:
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
        self._count(outcome, sent.kind)

    def _remember(
        self, kind: Kind, now: float, outcome: str, detail: str | None, text: str | None
    ) -> None:
        """Add a snapshot sent to the history (lock held), at most ``history_size``."""
        self._history.append(
            {"kind": kind, "stamp": now, "outcome": outcome, "detail": detail, "text": text}
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

    def description(self, token: str) -> str | None:
        """The shown thing's description for the look ``token``, None without one."""
        with self._lock:
            shown = self._shown
        return shown[1] if shown is not None and shown[0] == token else None

    def wait_description(self, token: str, timeout_s: float) -> str | None:
        """``description(token)``, waiting at most ``timeout_s`` while an attempt, a chosen
        frame or a describe request for ``token`` is still open; None once none is."""
        deadline = self.clock() + max(0.0, timeout_s)
        with self._lock:
            while True:
                if self._shown is not None and self._shown[0] == token:
                    return self._shown[1]
                left = deadline - self.clock()
                if left <= 0 or not self._open_for(token):
                    return None
                self._lock.wait(min(left, POLL_S))

    def _open_for(self, token: str) -> bool:
        """Whether something for ``token`` is still on its way (lock held)."""
        return (
            (self._attempt is not None and self._attempt.token == token)
            or (self._pending is not None and self._pending.token == token)
            or self._describing == token
        )

    def room(self, now: float) -> str | None:
        """The room description, None without one or past ``room.max_age_s``."""
        with self._lock:
            room = self._room
        if room is None or now - room[1] > self._cfg.surroundings.room.max_age_s:
            return None
        return room[0]

    def jpeg(self) -> bytes | None:
        """The last snapshot sent (RAM only)."""
        with self._lock:
            return self._jpeg

    def state(self) -> dict[str, Any]:
        """Codes, counts, ages, sizes and scores (SPEC 12A): no text, token or image."""
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
            elif self._capped(None):
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
                "attempt": self._attempt.kind if self._attempt is not None else None,
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
            shown = self._shown[1] if self._shown is not None else None
            history = [
                {**{k: v for k, v in h.items() if k != "stamp"},
                 "age_s": round(now - h["stamp"], ROUND_DIGITS)}
                for h in reversed(self._history)
            ]  # fmt: skip
        described = {"shown": shown, "room": self.room(now)}
        return {**self.state(), "described": described, "history": history}

    # ------------------------------------------------------------------- config

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """A stack subscriber: keep the copy. The switch going off, or the image route's base
        URL, model or key changing, drops the attempt, the texts and the image, and ignores a
        reply still in flight; a changed route also clears the unsupported mark and the pause."""
        trust = _trust(cfg)
        with self._lock:
            self._cfg = cfg
            if trust == self._trust:
                return
            self._trust = trust
            self._want_first = trust is not None  # switched on, or another route: a new room
            self._unsupported, self._failures, self._paused_until = None, 0, -math.inf
            self._generation += 1
            self._attempt = self._pending = None
            self._shown = self._room = None
            self._jpeg, self._last, self._reference = None, None, None
            self._history = []
            self._lock.notify_all()

    def _count(self, outcome: str, kind: Kind) -> None:
        with self._lock:
            self._counts[outcome] = self._counts.get(outcome, 0) + 1
        log.debug("surroundings: %s %s", kind, outcome)
        self.events(EVENT_KIND, outcome, snapshot=kind)


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
    if isinstance(exc, TimeoutError) or type(exc).__name__ == "APITimeoutError":
        return TIMEOUT
    status = getattr(exc, "status_code", None)
    return f"http_{status}" if isinstance(status, int) else ERROR
