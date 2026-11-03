"""The attention target and its category names (SPEC 5, 5A): no imports beyond the standard
library, so consumers such as the appraisal adapter depend on this alone."""

from __future__ import annotations

from dataclasses import dataclass

SPEAKER_KEY = "speaker-bearing"
SHOWN = "shown_object"  # a thing a person shows the robot (SPEC 5A, Joint attention)
SPEAKER = "active_speaker"


@dataclass(frozen=True)
class AttentionTarget:
    """What to look at: a world entry (or a bare bearing) in the base frame.

    ``priority`` is the category that selected it; ``reason`` says why it is the target now
    (``new``, ``switched``, ``tracking``, ``held``). ``entry_id`` is None for a bearing-only
    speaker. ``category`` is the entry's detector class (empty for a bare bearing).
    """

    entry_id: str | None
    kind: str
    azimuth_rad: float
    elevation_rad: float
    priority: str
    reason: str
    stamp: float
    category: str = ""

    @property
    def key(self) -> str:
        return self.entry_id or SPEAKER_KEY
