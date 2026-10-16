"""Per-track identity by voting (SPEC 6, Matching).

An evaluation compares one embedding with every stored embedding of the same model id. It
votes for the best owner when the best similarity reaches ``identity.match_threshold`` with
``identity.margin`` over the second best owner. A track is assigned to an owner once that owner
holds ``votes_required`` of the last ``votes_window`` votes. Anything else is ``unknown`` with
an anonymous ID that lives only in this object: it is never written to the registry and it
restarts with ``new_session``. The matcher shares the registry's lock and purges a forgotten
owner from every track. Never log embeddings or crops.

With a single enrolled owner there is no second best, so the margin rule cannot apply (SPEC 6
states it only "over the second best"); the threshold alone decides.
"""

from __future__ import annotations

import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

from ..config.schema import IdentityConfig
from .registry import GalleryKind, Registry

Kind = Literal["person", "entity"]
ANON_ID_HEX_CHARS = 12


@dataclass(frozen=True)
class Identity:
    """Who a track is: a registry owner, or unknown with a session-only anonymous ID."""

    track_id: int
    owner_id: str | None
    kind: Kind | None
    anonymous_id: str | None
    disagreement: bool = False

    @property
    def known(self) -> bool:
        return self.owner_id is not None


@dataclass
class _Track:
    face: deque[tuple[str | None, datetime]]
    voice: deque[tuple[str | None, datetime]]
    anonymous_id: str | None = None


@dataclass
class _Vote:
    owner_id: str | None
    best: float = 0.0


@dataclass
class Matcher:
    """Tracks votes per ``track_id`` over a registry and resolves each track to an identity."""

    registry: Registry
    config: IdentityConfig
    _tracks: dict[int, _Track] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lock = self.registry.lock
        self.registry.add_forget_listener(self._purge)

    def _purge(self, owner_id: str) -> None:
        """Turn every vote for ``owner_id`` into a miss (called when the owner is forgotten)."""
        with self._lock:
            for track in self._tracks.values():
                for votes in (track.face, track.voice):
                    kept = [(None if v == owner_id else v, at) for v, at in votes]
                    votes.clear()
                    votes.extend(kept)

    def new_session(self) -> None:
        """Forget every track and its anonymous ID; new tracks get new random IDs."""
        with self._lock:
            self._tracks.clear()

    def release(self, track_id: int) -> None:
        """Drop the state of a track that ended, so a reused track id starts clean."""
        with self._lock:
            self._tracks.pop(track_id, None)

    def _track(self, track_id: int) -> _Track:
        if track_id not in self._tracks:
            window = self.config.votes_window
            self._tracks[track_id] = _Track(deque(maxlen=window), deque(maxlen=window))
        return self._tracks[track_id]

    def _vote(self, gallery_kind: GalleryKind, embedding: np.ndarray, model_id: str) -> _Vote:
        query = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query)) if query.ndim == 1 else 0.0
        if not np.isfinite(query).all() or not np.isfinite(norm) or norm == 0:
            return _Vote(None)
        sims: dict[str, float] = {}
        for owner, rows in self.registry.gallery(gallery_kind, model_id).items():
            if rows.shape[1] == query.shape[0]:
                best_sim = float((rows @ (query / norm)).max())
                if np.isfinite(best_sim):
                    sims[owner] = best_sim
        if not sims:
            return _Vote(None)
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        best_owner, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else None
        if best < self.config.match_threshold:
            return _Vote(None, best)
        if second is not None and best - second < self.config.margin:
            return _Vote(None, best)
        return _Vote(best_owner, best)

    def _live(self, votes: deque[tuple[str | None, datetime]]) -> list[str | None]:
        """Owners of the votes younger than ``vote_max_age_s``, oldest first."""
        now, limit = self.registry.now(), self.config.vote_max_age_s
        return [v for v, at in votes if (now - at).total_seconds() <= limit]

    def _winner(self, votes: deque[tuple[str | None, datetime]]) -> str | None:
        counts = Counter(v for v in self._live(votes) if v is not None)
        if not counts:
            return None
        owner, n = counts.most_common(1)[0]
        return owner if n >= self.config.votes_required else None

    def _resolve(self, track_id: int, kind: Kind) -> Identity:
        track = self._track(track_id)
        face, voice = self._winner(track.face), self._winner(track.voice)
        disagreement = face is not None and voice is not None and face != voice
        owner = None if disagreement else (face or voice)
        if owner is not None:
            return Identity(track_id, owner, kind, None)
        if track.anonymous_id is None:
            track.anonymous_id = f"anon-{uuid.uuid4().hex[:ANON_ID_HEX_CHARS]}"
        return Identity(track_id, None, None, track.anonymous_id, disagreement)

    def _conflicted(self, track: _Track, owner_id: str) -> bool:
        """True when any recent face or voice vote on the track went to someone else."""
        live = (*self._live(track.face), *self._live(track.voice))
        return any(v is not None and v != owner_id for v in live)

    def observe(
        self,
        track_id: int,
        embedding: np.ndarray,
        model_id: str,
        *,
        kind: Kind = "person",
        refresh: bool = True,
    ) -> Identity:
        """Evaluate a face (or, for ``kind="entity"``, appearance) embedding for a track.

        After a high-confidence match of a person, the embedding is offered to the registry's
        online refresh, which applies consent, cap, interval and drift rules. No refresh is
        offered while any live vote on the track, face or voice, names someone else. Votes older
        than ``vote_max_age_s`` are ignored; neither modality ever clears the other, so a
        disagreement lasts until the stale votes age out or fresh ones agree.
        """
        with self._lock:
            track = self._track(track_id)
            vote = self._vote("entity" if kind == "entity" else "face", embedding, model_id)
            track.face.append((vote.owner_id, self.registry.now()))
            identity = self._resolve(track_id, kind)
            if (
                refresh
                and kind == "person"
                and identity.owner_id is not None
                and vote.owner_id == identity.owner_id
                and vote.best >= self.config.refresh_min_similarity
                and not self._conflicted(track, identity.owner_id)
            ):
                self.registry.add_refresh_embedding(identity.owner_id, embedding, model_id)
            return identity

    def observe_voice(self, track_id: int, embedding: np.ndarray, model_id: str) -> Identity:
        """Evaluate a speaker embedding for a track; fused with the face votes of that track."""
        with self._lock:
            track = self._track(track_id)
            owner = self._vote("voice", embedding, model_id).owner_id
            track.voice.append((owner, self.registry.now()))
            return self._resolve(track_id, "person")
