"""Replay of recorded camera sessions through the vision pipeline (SPEC 14, Replay).

A session is a directory with ``session.json`` and the frame files it lists::

    {"version": 1, "frames": [
        {"file": "000000.npy", "stamp": 0.0, "joint_state": {"head_yaw": 0.1},
         "labels": {"faces": [{"person_id": "alice"}]}}, ...]}

``.npy`` files hold an ``HxWx3`` uint8 BGR array and need only numpy; ``.png`` and ``.jpg``
need OpenCV (the ``vision`` extra). The library never records sessions: frames are written
to disk only by an explicit, opt-in debug recorder (SPEC 12), which this is not.

``replay`` feeds frames to the pipeline in order with their recorded capture stamps. The
pipeline reads no wall clock, so with an inline attribute executor two runs give the same
messages. Anonymous IDs are random per run; ``summarize`` renumbers them in order of first
appearance before runs are compared.

``replay`` turns the matcher's online refresh off (it would add embeddings to the registry
using its wall clock). ``Registry.mark_seen`` also stamps ``last_seen``
with the wall clock; that is metadata and does not affect the published messages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..bus import Bus, Message
from ..world import Observation
from .attributes import InlineExecutor
from .pipeline import VisionPipeline
from .types import Frame

SESSION_FILE = "session.json"
SESSION_VERSION = 1
TOPICS = ("percept.face", "percept.entity", "person.recognized")
ANON_PREFIX = "anon-"


class ReplayError(ValueError):
    """The session is malformed or a frame cannot be read."""


@dataclass(frozen=True)
class FrameRecord:
    """One recorded frame: file, capture stamp, joint state and ground-truth labels."""

    file: str
    stamp: float
    joint_state: Mapping[str, float] | None = None
    labels: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A recorded session on disk; frames are loaded one at a time."""

    root: Path
    frames: list[FrameRecord]

    def load(self, index: int) -> Frame:
        """Read frame ``index`` into a ``Frame`` carrying its recorded stamp and joint state."""
        rec = self.frames[index]
        path = (self.root / rec.file).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ReplayError(f"{rec.file}: path leaves the session folder")
        if path.suffix == ".npy":
            image = np.load(path, allow_pickle=False)
        else:
            try:
                import cv2
            except ImportError as exc:
                raise ReplayError(f"{rec.file}: reading {path.suffix} needs opencv") from exc
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ReplayError(f"{rec.file}: unreadable image")
        return Frame(image, rec.stamp, rec.joint_state)


def load_session(root: str | Path) -> Session:
    """Parse ``session.json`` in ``root``; frames must have strictly increasing stamps.

    Raises ``ReplayError`` for a missing or malformed file, a wrong version or stamps that do
    not increase. Frame files are not read until ``Session.load``.
    """
    root = Path(root)
    try:
        data = json.loads((root / SESSION_FILE).read_text())
        frames = [
            FrameRecord(f["file"], float(f["stamp"]), f.get("joint_state"), f.get("labels", {}))
            for f in data["frames"]
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReplayError(f"{root}: invalid session ({type(exc).__name__}: {exc})") from exc
    if data.get("version") != SESSION_VERSION:
        raise ReplayError(f"{root}: unsupported session version {data.get('version')!r}")
    if any(b.stamp <= a.stamp for a, b in zip(frames, frames[1:], strict=False)):
        raise ReplayError(f"{root}: frame stamps must increase")
    return Session(root, frames)


@dataclass(frozen=True)
class Recorded:
    """A message captured during a replay, with the index of the frame that caused it."""

    frame: int
    topic: str
    stamp: float
    payload: Any = field(repr=False)


@dataclass
class ReplayResult:
    """Messages published during a replay, in order, with the session's frame records."""

    frames: list[FrameRecord]
    messages: list[Recorded]


def replay(
    session: Session, pipeline: VisionPipeline, bus: Bus, topics: Sequence[str] = TOPICS
) -> ReplayResult:
    """Drive ``pipeline`` through ``session`` and collect what it publishes on ``topics``.

    Frames go in recorded order with their recorded stamps and joint states, one ``process``
    call each. To be reproducible it resets the pipeline (tracks, track IDs, governor) and the
    matcher session first, forces online refresh off for the duration (restored afterwards)
    and refuses a pipeline whose attribute worker is not inline. The subscription is removed
    afterwards even on error.
    """
    worker = pipeline.worker
    if worker is not None and not isinstance(worker.executor, InlineExecutor):
        raise ReplayError("replay needs the pipeline built with executor=InlineExecutor()")
    messages: list[Recorded] = []
    current = [0]

    def record(msg: Message) -> None:
        messages.append(Recorded(current[0], msg.topic, msg.stamp, msg.payload))

    unsubscribe = [bus.subscribe(t, record) for t in topics]
    refresh, pipeline.refresh = pipeline.refresh, False
    pipeline.reset()
    pipeline.matcher.new_session()
    try:
        for i in range(len(session.frames)):
            current[0] = i
            pipeline.process(session.load(i))
    finally:
        pipeline.refresh = refresh
        for off in unsubscribe:
            off()
    return ReplayResult(session.frames, messages)


def _plain(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Observation):
        data = {k: v for k, v in asdict(payload).items() if k != "embedding"}
        data["has_embedding"] = payload.embedding is not None
        return data
    return asdict(payload) if is_dataclass(payload) and not isinstance(payload, type) else {}


def summarize(messages: Sequence[Recorded]) -> list[dict[str, Any]]:
    """JSON-friendly copy of the messages with anonymous IDs renumbered by first appearance.

    Appearance embeddings are replaced by a ``has_embedding`` flag. Two replays of one session
    must give equal summaries; this is what the determinism tests compare.
    """
    names: dict[str, str] = {}
    out = []
    for m in messages:
        data = _plain(m.payload)
        anon = data.get("anonymous_id")
        if isinstance(anon, str) and anon.startswith(ANON_PREFIX):
            data["anonymous_id"] = names.setdefault(anon, f"{ANON_PREFIX}{len(names) + 1}")
        out.append({"frame": m.frame, "topic": m.topic, "stamp": m.stamp, **data})
    return out
