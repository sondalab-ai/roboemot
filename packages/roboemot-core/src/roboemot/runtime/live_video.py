"""The live camera preview for the web UI's Live tab (SPEC 4A, 11A, 12).

``LiveVideo.frame`` serves one JPEG of the newest frame the vision pump already captured: it never
opens the camera, never resumes a paused pump and never reads a frame itself. The frame comes
from ``LatestFrameRunner.latest``, which keeps one reference only while a page asked within
``webui.live.viewer_window_s``. A request encodes only when that frame is newer than the last
one encoded and the last encode is older than ``1 / video_fps``; otherwise it returns the cached
JPEG, so any number of viewers cost one encode. Face boxes and confirmed names are drawn on the
picture, the gaze target highlighted.

A viewer is a client IP that was sent a frame within the window. ``viewers`` feeds the banner every
open page shows. Start and stop of viewing are logged once each per client IP (the address only).
The one cached JPEG is dropped when viewing stops, the flag goes off or the robot sleeps; nothing
is written to disk, logs or reports.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections.abc import Callable
from typing import Any

import numpy as np

from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation.sources import person_name
from roboemot.world.types import UNKNOWN_IDENTITY

log = logging.getLogger(__name__)

# Result codes (the page maps each to a text).
LIVE_DISABLED = "live_disabled"
VIDEO_OFF = "video_off"
VIDEO_ASLEEP = "video_asleep"
VIDEO_NO_VISION = "video_no_vision"
VIDEO_NO_FRAME = "video_no_frame"

BGR_FACE = (80, 200, 80)
BGR_TARGET = (0, 215, 255)
BGR_TEXT_BG = (0, 0, 0)
LINE_PX = 1
TARGET_LINE_PX = 3
FONT_SCALE = 0.45
TEXT_THICKNESS = 1
LABEL_PAD_PX = 3
LABEL_MAX_CHARS = 24


def ascii_label(name: str) -> str:
    """``name`` folded to what cv2's Hershey font can draw (accents dropped, the rest '?')."""
    base = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return base.encode("ascii", "replace").decode("ascii")[:LABEL_MAX_CHARS]


class LiveVideo:
    """Serves the preview JPEG and tracks who is watching. ``host`` and ``p`` are the running
    ``RuntimeHost`` and ``Pipeline``; ``clock`` is the one the web UI's other Live parts use."""

    def __init__(
        self,
        host: Any,
        p: Any,
        cfg: RoboEmotConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._host, self._p, self._clock = host, p, clock
        self._cfg = cfg
        self._lock = threading.Lock()  # serialises the encode; viewers wait for the same JPEG
        self._jpeg: bytes | None = None
        self._jpeg_seq = -1
        self._jpeg_at = float("-inf")
        self._seen: dict[str, float] = {}  # client ip -> when it was last sent a frame
        self._cv2: Any = None
        self._runner: Any = None  # the runner this view asked to keep a frame (set by ``frame``)
        self._armed_until = float("-inf")  # the runner keeps a frame until then, if still armed

    def apply_config(self, change: Any) -> None:
        """``ConfigStack`` subscriber: hot reload; turning the preview off drops everything."""
        self._cfg = change.config
        live = self._cfg.webui.live
        if not (live.enabled and live.video):
            self.drop()

    # ------------------------------------------------------------------ viewers

    def _prune(self, now: float) -> None:
        """Forget clients past the window, log each stop, and free the frame when none is left."""
        window = self._cfg.webui.live.viewer_window_s
        gone = [ip for ip, at in self._seen.items() if now - at > window]
        for ip in gone:
            del self._seen[ip]
            log.info("live video view stopped (%s)", ip)
        if gone and not self._seen:
            self._release()

    def viewers(self) -> dict[str, Any]:
        """The banner's answer: whether any client was sent a frame within the window, and how
        many. Cheap: no config read, no encode."""
        live = self._cfg.webui.live
        if not (live.enabled and live.video):
            return {"video_viewers_active": False, "count": 0, "poll_ms": live.banner_poll_ms}
        with self._lock:
            self._prune(self._clock())
            count = len(self._seen)
        return {"video_viewers_active": count > 0, "count": count, "poll_ms": live.banner_poll_ms}

    # ------------------------------------------------------------------ frames

    def drop(self) -> None:
        """Viewing ends now (flag off, robot asleep, shutdown): log the stops, free the frame
        and the JPEG."""
        with self._lock:
            for ip in self._seen:
                log.info("live video view stopped (%s)", ip)
            self._seen.clear()
            self._release()

    def expire(self) -> None:
        """Called from the app tick, so the age bound holds when no page polls: once the last
        request is older than the window, log the stops and drop the frame and the JPEG. Free
        (two reads, no lock) while nothing is held."""
        if self._armed_until == float("-inf"):
            return
        with self._lock:
            now = self._clock()
            self._prune(now)
            if not self._seen and now > self._armed_until:
                self._release()

    def _release(self) -> None:
        self._jpeg, self._jpeg_seq, self._jpeg_at = None, -1, float("-inf")
        self._armed_until = float("-inf")
        if self._runner is not None:
            self._runner.unwatch()

    def _load_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError:
                return None
            self._cv2 = cv2
        return self._cv2

    def frame(self, client: str) -> tuple[bytes | None, str | None]:
        """``(jpeg, None)`` or ``(None, code)``. Codes: ``live_disabled``, ``video_off``,
        ``video_asleep`` (asleep, or the camera pump is paused: it is never resumed here),
        ``video_no_vision`` (no perception service or no cv2) and ``video_no_frame`` (watching
        has just started: the next frame the pump captures is kept)."""
        live = self._cfg.webui.live
        if not live.enabled:
            return None, LIVE_DISABLED
        if not live.video:
            return None, VIDEO_OFF
        perception = self._host.perception
        service = perception.service if perception is not None else None
        cv2 = self._load_cv2()
        if service is None or cv2 is None:
            return None, VIDEO_NO_VISION
        if self._p.sleep.asleep or service.paused:
            self.drop()
            return None, VIDEO_ASLEEP
        with self._lock:
            now = self._clock()
            self._runner = service.runner
            self._runner.watch(live.viewer_window_s)
            self._armed_until = now + live.viewer_window_s
            latest = self._runner.latest()
            if latest is None:
                self._prune(now)
                return None, VIDEO_NO_FRAME
            frame, seq = latest
            fresh = self._jpeg is not None and now - self._jpeg_at < 1.0 / live.video_fps
            if self._jpeg is None or (seq != self._jpeg_seq and not fresh):
                self._jpeg = self._encode(cv2, frame.image, live)
                self._jpeg_seq, self._jpeg_at = seq, now
            if client not in self._seen:
                log.info("live video view started (%s)", client)
            self._seen[client] = now
            self._prune(now)
            return self._jpeg, None

    # ------------------------------------------------------------------ drawing

    def _encode(self, cv2: Any, image: np.ndarray, live: Any) -> bytes:
        height, width = image.shape[:2]
        scale = min(1.0, live.video_max_width / width)
        small = (
            cv2.resize(
                image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
            )
            if scale < 1.0
            else image.copy()
        )
        self._draw(cv2, small, scale)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, live.video_jpeg_quality])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return bytes(buf)

    def _draw(self, cv2: Any, canvas: np.ndarray, scale: float) -> None:
        """Face boxes with the confirmed name (nothing for an unknown face), the gaze target in
        another colour and a thicker line."""
        pipeline = self._host.perception.pipeline
        if pipeline is None:
            return
        faces = pipeline.face_overlay()
        registry = pipeline.matcher.registry
        target = self._p.selector.target
        target_owner = None
        if target is not None:
            entry = self._p.world.get(target.entry_id) if target.entry_id else None
            if entry is not None and entry.identity not in ("", UNKNOWN_IDENTITY):
                target_owner = entry.identity
        unowned = [i for i, (_, owner) in enumerate(faces) if owner is None]
        lone = (
            unowned[0] if target is not None and target_owner is None and len(unowned) == 1 else -1
        )
        for i, (box, owner) in enumerate(faces):
            is_target = (owner is not None and owner == target_owner) or i == lone
            x1, y1, x2, y2 = (round(v * scale) for v in box)
            color = BGR_TARGET if is_target else BGR_FACE
            cv2.rectangle(
                canvas, (x1, y1), (x2, y2), color, TARGET_LINE_PX if is_target else LINE_PX
            )
            name = person_name(registry, owner)
            if name:
                self._label(cv2, canvas, ascii_label(name), x1, y1, color)

    @staticmethod
    def _label(cv2: Any, canvas: np.ndarray, text: str, x: int, y: int, color: Any) -> None:
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)
        top = max(0, y - h - base - LABEL_PAD_PX)
        cv2.rectangle(
            canvas,
            (x, top),
            (x + w + LABEL_PAD_PX * 2, top + h + base + LABEL_PAD_PX),
            BGR_TEXT_BG,
            -1,
        )
        cv2.putText(
            canvas, text, (x + LABEL_PAD_PX, top + h + LABEL_PAD_PX // 2),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, TEXT_THICKNESS, cv2.LINE_AA,
        )  # fmt: skip
