"""Object detection on a crop around the faces in view (SPEC 5, 5A, 13).

The object detector squashes its input to a fixed square, so a cup held at arm's length is a
few pixels in a full frame. While a face is in view, a run may look at a square crop around
the faces instead: the held thing is several times larger in the detector input, at the same
cost per run. The crop changes what one run looks at, never how often runs happen.

- Region: per face the shown-object region of SPEC 5A (``attention.shown``:
  ``region_width_faces`` to each side, ``region_above_faces`` above and ``region_below_faces``
  below the face centre, in face-box heights), widened by ``margin_faces`` on every side for the
  thing's own size. One crop covers the regions of all the faces in view, so any of them may be
  the holder. It is made square (the detector's input is square, so the crop is not squashed),
  at least ``min_side_share`` of the frame's shorter side, at most that side, and shifted inside
  the frame. When it would cover more than ``full_frame_share`` of the frame, the full frame runs.
  Only faces of the last face run, at most one face-detection period old, count.
- Look-down: during a look at a shown thing (SPEC 5A) the head pitches down and the holder's
  face may leave the view. With no usable face crop, the crop is then around the confirmed
  object track nearest the frame centre (the head follows the thing), ``shown_margin_heights``
  of its box height on every side, with the same size rules.
- Schedule (``CropSchedule``): the full frame runs when the switch is off, with no crop to make
  (no fresh face, no shown thing) or a large one, on the gate's ``floor`` and ``settled`` runs (a
  still thing anywhere, a new view), and when the last full run is ``full_interval_s`` old or
  more (capture time), or there was none. Every other run (``motion``, ``live_track``,
  ``episode``, ``no_pose``, the gate off) crops: ``no_pose`` is the fixed schedule, like the gate
  off.
- Boxes: ``Crop.to_frame`` adds the crop's offset, so tracking, the world, attention and the
  preview see frame pixels. A detection within ``edge_margin_share`` of the crop side from a crop
  edge inside the frame is cut by the crop and dropped: its box would start a wrong small track.
  ``Crop.trusted`` is the region where a crop run sees things whole; the object tracker leaves
  tracks outside it alone (not matched, not missed), so a run on a crop says nothing about them.
  A detection that matches no track may pick up such a track of its class within
  ``pickup_distance_heights`` (``Tracker.update``): a cup lifted into the crop keeps its track.

The crop is a view of the frame inside one call and is never kept or logged (SPEC 12). The debug
report (SPEC 12A) gets counts only.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..config.schema import ObjectCropConfig, ShownObjectConfig
from .tracker import inside
from .types import Box, Detection

FULL, CROP = "full", "crop"
FULL_REASONS = ("off", "no_face", "large", "gate", "interval")
FACE, SHOWN = "face", "shown"
CROP_REASONS = (FACE, SHOWN)
FULL_GATE_REASONS = frozenset({"floor", "settled"})


@dataclass(frozen=True)
class Crop:
    """A crop ``[x0, x1) x [y0, y1)`` of a ``width`` x ``height`` frame, in pixels; ``edge_px``
    is how close to an inner edge a box counts as cut."""

    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int
    edge_px: float

    @property
    def trusted(self) -> Box:
        """The frame region a crop run sees whole: the crop shrunk by ``edge_px`` on its edges
        inside the frame and open on the frame's own borders."""
        return (
            self.x0 + self.edge_px if self.x0 > 0 else -math.inf,
            self.y0 + self.edge_px if self.y0 > 0 else -math.inf,
            self.x1 - self.edge_px if self.x1 < self.width else math.inf,
            self.y1 - self.edge_px if self.y1 < self.height else math.inf,
        )

    def view(self, image: np.ndarray) -> np.ndarray:
        """The crop of ``image`` as a numpy view (no copy)."""
        return image[self.y0 : self.y1, self.x0 : self.x1]

    def to_frame(self, dets: Sequence[Detection]) -> list[Detection]:
        """Crop-pixel detections in frame pixels; those cut by an inner crop edge are dropped."""
        trusted = self.trusted
        out = []
        for d in dets:
            x1, y1, x2, y2 = d.box
            box = (x1 + self.x0, y1 + self.y0, x2 + self.x0, y2 + self.y0)
            if inside(box, trusted):
                out.append(replace(d, box=box))
        return out


def _region(face: Box, cfg: ObjectCropConfig, shown: ShownObjectConfig) -> Box:
    """A face's shown-object region in pixels (image y grows downward), with the margin."""
    x1, y1, x2, y2 = face
    h = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = (shown.region_width_faces + cfg.margin_faces) * h
    return (
        cx - side,
        cy - (shown.region_above_faces + cfg.margin_faces) * h,
        cx + side,
        cy + (shown.region_below_faces + cfg.margin_faces) * h,
    )


def _place(centre: float, side: int, limit: int) -> int:
    """Start of a ``side`` long span centred on ``centre`` and shifted inside ``[0, limit]``."""
    return min(max(round(centre - side / 2), 0), limit - side)


def _square(region: Box, width: int, height: int, cfg: ObjectCropConfig) -> Crop | None:
    """The square crop centred on ``region``: at least ``min_side_share`` of the frame's
    shorter side, at most that side, shifted inside the frame; None above
    ``full_frame_share`` of the frame area."""
    rx1, ry1, rx2, ry2 = region
    short = min(width, height)
    side = max(rx2 - rx1, ry2 - ry1, cfg.min_side_share * short)
    side = min(math.ceil(side), short)
    if side * side > cfg.full_frame_share * width * height:
        return None
    x0 = _place((rx1 + rx2) / 2, side, width)
    y0 = _place((ry1 + ry2) / 2, side, height)
    return Crop(x0, y0, x0 + side, y0 + side, width, height, cfg.edge_margin_share * side)


def crop_for(
    faces: Sequence[Box],
    width: int,
    height: int,
    cfg: ObjectCropConfig,
    shown: ShownObjectConfig,
) -> Crop | None:
    """The square crop around the shown-object regions of ``faces`` in a ``width`` x ``height``
    frame, or None for the full frame (no face, or a crop above ``full_frame_share``)."""
    if not faces:
        return None
    lefts, tops, rights, bottoms = zip(*(_region(f, cfg, shown) for f in faces), strict=True)
    return _square((min(lefts), min(tops), max(rights), max(bottoms)), width, height, cfg)


def _off_centre(box: Box, width: int, height: int) -> float:
    """Distance in pixels from the centre of ``box`` to the centre of the frame."""
    x1, y1, x2, y2 = box
    return math.hypot((x1 + x2 - width) / 2, (y1 + y2 - height) / 2)


def crop_around_thing(
    things: Sequence[Box], width: int, height: int, cfg: ObjectCropConfig
) -> Crop | None:
    """The square crop around the thing box nearest the frame centre (the head follows a
    shown thing), ``shown_margin_heights`` of its height on every side; None without a box or
    above ``full_frame_share``."""
    if not things:
        return None
    x1, y1, x2, y2 = min(things, key=lambda box: _off_centre(box, width, height))
    pad = cfg.shown_margin_heights * (y2 - y1)
    return _square((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height, cfg)


class CropSchedule:
    """Picks the input of each object run (full frame or a crop) and counts them; see the
    module docstring. ``cfg`` is replaced on hot reload and read at each call. One lock guards
    the state, so ``snapshot`` and ``reset`` may run on another thread."""

    def __init__(self, cfg: ObjectCropConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._clear()

    def _clear(self) -> None:
        self._last_full: float | None = None
        self._runs = dict.fromkeys((FULL, CROP), 0)
        self._full_by = dict.fromkeys(FULL_REASONS, 0)
        self._crop_by = dict.fromkeys(CROP_REASONS, 0)

    def choose(
        self,
        stamp: float,
        gate_reason: str | None,
        faces: Sequence[Box],
        width: int,
        height: int,
        shown: ShownObjectConfig,
        *,
        things: Sequence[Box] = (),
        episode: bool = False,
    ) -> Crop | None:
        """The crop for the object run at ``stamp``, or None for the full frame.
        ``gate_reason`` is why the object gate ran it; ``faces`` are the boxes of the fresh
        faces; during a look at a shown thing (``episode``) with no usable face crop, the crop
        is around the confirmed thing box (``things``) nearest the frame centre."""
        cfg = self.cfg
        with self._lock:
            crop, by = None, FACE
            if not cfg.enabled:
                why = "off"
            elif gate_reason in FULL_GATE_REASONS:
                why = "gate"
            elif self._last_full is None or stamp - self._last_full >= cfg.full_interval_s:
                why = "interval"
            else:
                crop = crop_for(faces, width, height, cfg, shown)
                if crop is None and episode:
                    crop, by = crop_around_thing(things, width, height, cfg), SHOWN
                why = "large" if faces or (episode and things) else "no_face"
            if crop is None:
                self._last_full = stamp
                self._full_by[why] += 1
            else:
                self._crop_by[by] += 1
            self._runs[FULL if crop is None else CROP] += 1
            return crop

    def snapshot(self) -> dict[str, Any]:
        """Counts for the debug report (SPEC 12A): object runs by input, full runs and crop
        runs by reason."""
        with self._lock:
            return {
                "enabled": self.cfg.enabled,
                **self._runs,
                "full_by": dict(self._full_by),
                "crop_by": dict(self._crop_by),
            }

    def reset(self) -> None:
        """Forget the last full run and the counts (new session or replay start)."""
        with self._lock:
            self._clear()
