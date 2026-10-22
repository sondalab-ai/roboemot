"""Running perception in an app: registry, pipeline, camera thread (SPEC 3, 4A, 5, 6, 12, 12A).

``open_perception`` is what an app calls once at startup. It always builds the registry (so
people can be listed and erased even when vision cannot run) and tries to load the models; if
they are missing, unverified or the ONNX Runtime is not installed, it logs once, records the
reason and returns without a pipeline. The app keeps running and the debug report says
``vision: unavailable (reason)``.

``VisionService`` owns the two threads of a running pipeline: the camera pump (reads frames,
paced by ``vision.rates.camera_hz``) and the worker (``LatestFrameRunner``, latest frame wins).
The camera is a robot driver's ``FrameSource``; this module knows nothing about a robot.

The registry store has no encryption yet (M5, SPEC 12 Data): it runs with ``allow_plaintext``,
files are protected by their 0o600 / 0o700 modes, and startup says so.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import paths, shutdown
from ..bus import Bus
from ..config.schema import RoboEmotConfig
from ..registry import FileStore, Matcher, Registry
from ..registry.enrollment import EnrollmentService
from .geometry import CameraModel
from .governor import RateGovernor
from .pipeline import VisionPipeline
from .runner import FrameSource, LatestFrameRunner, pump
from .types import Frame

log = logging.getLogger(__name__)

PUMP_RESTART_S = 1.0
JOIN_TIMEOUT_S = 5.0
CameraProvider = Callable[[int, int], CameraModel | None]
StagesLoader = Callable[[RoboEmotConfig, str | None], Any]


class CountingSource:
    """Wraps a ``FrameSource`` to count reads, frames and failures (type names only)."""

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self.reads = 0
        self.frames = 0
        self.errors = 0
        self.last_error: str | None = None

    def read(self) -> Frame | None:
        self.reads += 1
        try:
            frame = self._source.read()
        except Exception as exc:
            self.errors += 1
            self.last_error = type(exc).__name__
            raise
        if frame is not None:
            self.frames += 1
        return frame


class VisionService:
    """Camera pump thread plus the worker that runs the pipeline on the newest frame."""

    def __init__(self, pipeline: VisionPipeline, source: FrameSource, cfg: RoboEmotConfig) -> None:
        self.pipeline = pipeline
        self.cfg = cfg
        self.source = CountingSource(source)
        self.runner = LatestFrameRunner(pipeline)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None

    def pause(self) -> None:
        """Stop reading the camera until :meth:`resume` (the robot is asleep, SPEC 12A); the
        threads stay, idle, so nothing is restarted or leaked."""
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def start(self) -> None:
        """Start the worker and the camera thread."""
        self._stop.clear()
        self.runner.start()
        thread = threading.Thread(target=self._pump, name="roboemot-camera", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        """Quiet the camera thread, then stop the worker and the attribute worker.

        The camera thread is joined with a timeout and a timeout is logged: a read stuck in the
        SDK cannot be interrupted, so the caller must tear the SDK down knowing it may still be
        inside it. The worker and the pipeline are stopped even then.
        """
        self._stop.set()
        try:
            if self._thread is not None:
                shutdown.join(self._thread, JOIN_TIMEOUT_S)
                if self._thread.is_alive():
                    log.warning("camera thread did not stop within %.0f s", JOIN_TIMEOUT_S)
        finally:
            try:
                self.runner.stop()
            finally:
                self.pipeline.close()

    def _pump(self) -> None:
        """Pump frames; a failing camera read is counted, logged once per type, and retried."""
        reported: str | None = None
        while not self._stop.is_set():
            try:
                pump(
                    self.source,
                    self.runner,
                    self._stop,
                    self.cfg.vision.rates.camera_hz,
                    paused=self._paused,
                )
            except Exception as exc:
                name = type(exc).__name__
                if name != reported:
                    log.warning("camera read failed: %s", name)
                    reported = name
                self._stop.wait(PUMP_RESTART_S)

    def snapshot(self) -> dict[str, Any]:
        """Rates, drops, tracks and last error types for the debug report (SPEC 12A)."""
        return {
            "state": "paused" if self._paused.is_set() else "running",
            "camera": {
                "reads": self.source.reads,
                "frames": self.source.frames,
                "errors": self.source.errors,
                "last_error": self.source.last_error,
            },
            "frames": {
                "processed": self.runner.processed,
                "dropped": self.runner.dropped,
                "errors": self.runner.errors,
                "last_error": self.runner.last_error,
                **self.runner.rates(),
            },
            **self.pipeline.stats(),
        }


@dataclass
class Perception:
    """What ``open_perception`` returns.

    ``service`` is None while vision is unavailable; ``registry`` and ``matcher`` are None too
    when the data folder could not be opened. ``unavailable`` says why, for the debug report.
    """

    registry: Registry | None
    matcher: Matcher | None
    enrollment: EnrollmentService
    pipeline: VisionPipeline | None
    service: VisionService | None
    unavailable: str | None
    objects_unavailable: str | None = None

    def start(self) -> None:
        if self.service is not None:
            self.service.start()

    def stop(self) -> None:
        """Drop an enrollment session, stop the threads, write buffered ``last_seen`` updates.

        Each step runs whatever the others do; failures are logged by type and not raised.
        """
        steps: list[Callable[[], None]] = [self.enrollment.cancel]
        if self.service is not None:
            steps.append(self.service.stop)
        if self.registry is not None:
            steps.append(self.registry.flush)
        for step in steps:
            try:
                step()
            except Exception as exc:  # an ExceptionGroup of store errors; type only
                log.warning("perception shutdown step failed: %s", type(exc).__name__)

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload (SPEC 11A): vision, identity and enrollment settings (not restart-only)."""
        self.enrollment.cfg, self.enrollment.identity = cfg.enrollment, cfg.identity
        if self.registry is not None:
            self.registry.identity = cfg.identity
        if self.pipeline is not None:
            self.pipeline.apply_config(cfg.vision, cfg.identity, cfg.world)

    def vision_snapshot(self) -> dict[str, Any]:
        """The ``roboemot.vision`` report section."""
        if self.service is None:
            return {"state": "unavailable", "reason": self.unavailable}
        snapshot = self.service.snapshot()
        if self.objects_unavailable is not None:
            snapshot["objects"] = {"state": "unavailable", "reason": self.objects_unavailable}
        elif self.pipeline is not None:
            on = self.pipeline.object_detector is not None and self.pipeline.cfg.objects_enabled
            snapshot["objects"] = {"state": "running" if on else "off"}
        return snapshot

    def registry_snapshot(self) -> dict[str, Any]:
        """The ``roboemot.registry`` report section: counts only, no names or ids."""
        if self.registry is None:
            return {"state": "unavailable", "reason": self.unavailable}
        rows = self.registry.list()
        by_kind: dict[str, int] = {}
        for row in rows:
            by_kind[row.kind] = by_kind.get(row.kind, 0) + 1
        return {
            "store": "file, plaintext (encryption at rest arrives in M5)",
            "version": self.registry.version,
            "records": by_kind,
            "pending": sum(row.status == "pending" for row in rows),
            "quarantined": len(self.registry.quarantined()),
        }


def default_stages(cfg: RoboEmotConfig, profile: str | None) -> Any:
    """Load the ONNX backends from the models folder (needs the ``vision`` extra)."""
    from . import models
    from .onnx import load_stages

    root = models.models_dir(cfg.vision.models_dir, cfg.data_dir)
    return load_stages(root, cfg.vision, profile=profile, appearance=cfg.vision.objects_enabled)


def open_perception(
    cfg: RoboEmotConfig,
    bus: Bus,
    source: FrameSource,
    *,
    profile: str | None = None,
    camera_model: CameraProvider | None = None,
    load_stages: StagesLoader = default_stages,
) -> Perception:
    """Open the registry under the data folder and, if the models load, build the pipeline.

    ``source`` is the robot's camera; ``camera_model`` gives its intrinsics for a frame size
    (None: the field of view in ``world.camera_fov_*``); ``profile`` is the robot manifest's
    profile (``cm4`` picks the INT8 attribute models); ``load_stages`` is a seam for tests and
    other backends. Nothing here starts a thread: call ``Perception.start``.
    """
    try:
        data = paths.data_dir(cfg.data_dir)
        log.warning(
            "registry in %s is stored without encryption (pre-M5, SPEC 12 Data): "
            "protect the folder and the robot",
            data,
        )
        store = FileStore(Path(paths.registry_dir(cfg.data_dir)), allow_plaintext=True)
        registry = Registry(store, cfg.identity)
    except Exception as exc:
        reason = f"data folder: {type(exc).__name__}: {exc}"
        log.error("registry unavailable, running without recognition (%s)", reason)
        enrollment = EnrollmentService(None, cfg.enrollment, cfg.identity)
        enrollment.set_unavailable(reason)
        return Perception(None, None, enrollment, None, None, reason)
    matcher = Matcher(registry, cfg.identity)
    enrollment = EnrollmentService(registry, cfg.enrollment, cfg.identity)
    try:
        stages = load_stages(cfg, profile)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log.warning("vision unavailable, running without it (%s)", reason)
        enrollment.set_unavailable(reason)
        return Perception(registry, matcher, enrollment, None, None, reason)
    pipeline = VisionPipeline(
        cfg.vision,
        cfg.identity,
        bus,
        matcher,
        face_detector=stages.face_detector,
        face_embedder=stages.face_embedder,
        attribute_estimator=stages.attribute_estimator,
        appearance_embedder=stages.appearance_embedder,
        object_detector=stages.object_detector,
        governor=RateGovernor(cfg.vision.rates),
        camera_model=camera_model,
        world=cfg.world,
    )
    enrollment.attach(pipeline)
    return Perception(
        registry,
        matcher,
        enrollment,
        pipeline,
        VisionService(pipeline, source, cfg),
        None,
        stages.objects_unavailable,
    )


def world_snapshot(entries: list[Any], now: float) -> dict[str, Any]:
    """The ``roboemot.world`` report section from ``WorldModel.entries``.

    Bearings in degrees, age since last seen, whether the entry is an enrolled identity; no
    names, registry ids or embeddings.
    """
    from ..world.types import UNKNOWN_IDENTITY

    return {
        "count": len(entries),
        "entries": [
            {
                "entry_id": e.entry_id,
                "kind": e.kind,
                "category": e.category,
                "enrolled": bool(e.identity) and e.identity != UNKNOWN_IDENTITY,
                "azimuth_deg": math.degrees(e.azimuth_rad),
                "elevation_deg": math.degrees(e.elevation_rad),
                "confidence": e.confidence,
                "age_s": now - e.last_seen,
                "sources": sorted(e.sources),
            }
            for e in entries
        ],
    }


__all__ = [
    "CountingSource",
    "Perception",
    "VisionService",
    "default_stages",
    "open_perception",
    "world_snapshot",
]
