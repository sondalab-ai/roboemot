"""Per-track identity by voting (SPEC 6, Matching).

An evaluation compares one embedding with every stored embedding of the same model id. It
votes for the best owner when the best similarity reaches ``identity.match_threshold`` with
``identity.margin`` over the second best owner. A track is assigned to an owner once that owner
holds ``votes_required`` of the last ``votes_window`` votes. Anything else is ``unknown`` with
an anonymous ID that lives only in this object: it is never written to the registry and it
restarts with ``new_session``. Never log embeddings or crops.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..config.schema import IdentityConfig
from .registry import GalleryKind, Registry

Kind = Literal["person", "entity"]


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
    face: deque[str | None]
    voice: deque[str | None]
    anonymous_id: str | None = None


@dataclass
class _Vote:
    owner_id: str | None
    best: float = 0.0


@dataclass
class Matcher:
    """Tracks votes per ``track_id`` over a registry and resolves each track to an identity."""

    registry: Registry
    config: IdentityConfig | None = None
    _tracks: dict[int, _Track] = field(default_factory=dict)
    _anon_count: int = 0

    def __post_init__(self) -> None:
        self.config = self.config or self.registry.identity

    def new_session(self) -> None:
        """Forget every track and anonymous ID; the next anonymous ID starts over."""
        self._tracks.clear()
        self._anon_count = 0

    def release(self, track_id: int) -> None:
        """Drop the state of a track that ended."""
        self._tracks.pop(track_id, None)

    def _track(self, track_id: int) -> _Track:
        assert self.config is not None
        if track_id not in self._tracks:
            window = self.config.votes_window
            self._tracks[track_id] = _Track(deque(maxlen=window), deque(maxlen=window))
        return self._tracks[track_id]

    def _vote(self, gallery_kind: GalleryKind, embedding: np.ndarray, model_id: str) -> _Vote:
        assert self.config is not None
        query = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        sims: dict[str, float] = {}
        for owner, rows in self.registry.gallery(gallery_kind, model_id).items():
            if rows.shape[1] == query.shape[0] and norm > 0:
                sims[owner] = float((rows @ (query / norm)).max())
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

    def _winner(self, votes: deque[str | None]) -> str | None:
        assert self.config is not None
        counts = Counter(v for v in votes if v is not None)
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
            self._anon_count += 1
            track.anonymous_id = f"anon-{self._anon_count}"
        return Identity(track_id, None, None, track.anonymous_id, disagreement)

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
        online refresh, which applies the per-person cap. Disagreeing with voice blocks it.
        """
        assert self.config is not None
        track = self._track(track_id)
        vote = self._vote("entity" if kind == "entity" else "face", embedding, model_id)
        track.face.append(vote.owner_id)
        identity = self._resolve(track_id, kind)
        if (
            refresh
            and kind == "person"
            and identity.owner_id is not None
            and vote.owner_id == identity.owner_id
            and vote.best >= self.config.refresh_min_similarity
        ):
            self.registry.add_refresh_embedding(identity.owner_id, embedding, model_id)
        return identity

    def observe_voice(self, track_id: int, embedding: np.ndarray, model_id: str) -> Identity:
        """Evaluate a speaker embedding for a track; fused with the face votes of that track."""
        track = self._track(track_id)
        track.voice.append(self._vote("voice", embedding, model_id).owner_id)
        return self._resolve(track_id, "person")
