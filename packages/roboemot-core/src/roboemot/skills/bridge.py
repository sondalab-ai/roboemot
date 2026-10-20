"""Effects of the default LLM skills (SPEC 11, LLM tools; 5 attention; 7 expression).

Core's default skills (``roboemot.skills.defaults``) act through a ``SkillEnv`` of callables;
this module builds it from the runtime's pieces. Nothing here names a joint:

- ``play`` asks the expression renderer for a label of the manifest's expression map; it is
  rendered at the next app tick through the usual scaling and the minor cap;
- ``look_at`` hints the attention selector (an entry, or a bearing for ``away``), so the gaze
  controller and the limiter still decide what the head does;
- ``express_emotion`` is the affect engine's budgeted LLM impulse (in core).

The callables run inline on the conversation's event loop (they must not block). The world model
and the registry are lock-guarded and the selector's hint is one attribute write, so no queue is
needed. Names offered for ``look_at`` are the confirmed registry names of what is in view, cut to
one line of ``conversation.summary_value_max_chars`` like the scene summary: they are data from the
registry and land in the tool schema sent to the route.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from roboemot.affect.expression import ExpressionRenderer
from roboemot.attention import AttentionSelector, SpeakerHint
from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation.summaries import one_line
from roboemot.skills.defaults import SkillEnv
from roboemot.world import Entry, WorldModel

from roboemot.conversation.sources import entry_name

log = logging.getLogger(__name__)
CATEGORY_KINDS = {"animal": "animal", "object": "object"}


class SkillBridge:
    """Looks things up for ``look_at`` and relays ``play``; see the module docstring."""

    def __init__(
        self,
        expression: ExpressionRenderer,
        selector: AttentionSelector,
        world: WorldModel,
        registry: Any,
        cfg: Callable[[], RoboEmotConfig],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._expression, self._selector, self._world = expression, selector, world
        self._registry, self._cfg, self._clock = registry, cfg, clock

    def env(self, affect: Any) -> SkillEnv:
        """The environment for ``default_skills``."""
        return SkillEnv(
            gestures=self._expression.playable,
            play=self._expression.request,
            look_names=self._names,
            look_at=self.look_at,
            affect=affect,
        )

    def _fresh(self) -> list[Entry]:
        now, stale = self._clock(), self._cfg().attention.stale_s
        return [e for e in self._world.entries if now - e.last_seen <= stale]

    def _name(self, entry: Entry) -> str | None:
        name = entry_name(self._registry, entry)
        cap = self._cfg().conversation.summary_value_max_chars
        return (one_line(name, cap) or None) if name else None

    def _names(self) -> list[str]:
        seen: dict[str, None] = {}
        for entry in self._fresh():
            name = self._name(entry)
            if name:
                seen.setdefault(name)
        return list(seen)

    @staticmethod
    def _nearest(entries: list[Entry]) -> Entry | None:
        return min(
            entries,
            key=lambda e: (e.distance_m is None, e.distance_m or 0.0, -e.confidence, e.entry_id),
            default=None,
        )

    def look_at(self, target: str) -> bool:
        """Point the gaze at ``target`` (a category or a name in view); False when there is
        nothing like it in view."""
        now = self._clock()
        fresh = self._fresh()
        if target == "away":
            return self._look_away(now)
        entry: Entry | None
        if target in ("speaker", "person"):
            people = [e for e in fresh if e.kind == "person"]
            current = self._selector.target
            attended = next(
                (e for e in people if current is not None and e.entry_id == current.entry_id), None
            )
            entry = attended if target == "speaker" and attended else self._nearest(people)
        elif target in CATEGORY_KINDS:
            entry = self._nearest([e for e in fresh if e.kind == CATEGORY_KINDS[target]])
        else:
            wanted = target.casefold()
            entry = self._nearest([e for e in fresh if (self._name(e) or "").casefold() == wanted])
        if entry is None:
            return False
        self._selector.hint_speaker(SpeakerHint(now, entry_id=entry.entry_id))
        return True

    def _look_away(self, now: float) -> bool:
        """A bearing to one side of the current target, beyond the speaker-match gate."""
        import math

        current = self._selector.target
        base = current.azimuth_rad if current is not None else 0.0
        off = math.radians(self._cfg().conversation.look_away_deg)
        self._selector.hint_speaker(
            SpeakerHint(now, azimuth_rad=base + (off if base >= 0 else -off), elevation_rad=0.0)
        )
        return True
