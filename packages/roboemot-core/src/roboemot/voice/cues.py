"""Non-verbal cues (SPEC 8, Non-verbal cues; SPEC 4A, Hierarchy gate).

Where no voice can speak a line, the robot shows a cue instead: a gesture plus a short sound.

- ``listening``: an addressed utterance's audio left for the realtime route; the robot shows it
  heard until the answer's first audio, or the turn ends without audio. Held, not one-shot.
- ``offline``: the network is down (the offline phrase's place, SPEC 12A).
- ``not_heard``: the server's transcript did not come (the robot did not get it).
- ``fallback``: no route can answer and no local voice can say the "can't think" line.

Core names no robot: the gesture is the manifest's ``gesture_map`` entry ``cue_<kind>`` (SPEC 7,
the pattern of the offline responder's ``puzzled``), the sound comes from the robot's cue
library, registered in the ``roboemot.cue_sounds`` entry-point group under the manifest
``robot.id`` (a factory taking nothing or the manifest, returning kind -> :class:`CueSound`).
A cue with neither is a no-op, counted (``unsupported``). Sounds play as raw jobs of the speech
output (``SpeechOutput.play``, like the self-soothing sound), so barge-in and the echo
reference treat them as the robot's own audio.

The listening gesture holds at its peak (the keyframe time of its largest offset) until it is
released: the answer's first audio (the speech timeline has a sentence playing), every
utterance that began it ended (``listening_end`` as often as ``listening_begin``), or
``voice.cues.listening_hold_max_s`` passed, whatever else happens. It shows only after
``listening_onset_s``, so a quick answer shows none.

Threads: ``play``, ``listening_begin`` and ``listening_end`` come from any thread (the listener,
the conversation, the offline responder); ``commands`` runs on the app tick. One lock.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

import numpy as np

from ..config.schema import CueConfig
from ..motion.core import Command, Priority
from ..motion.primitives import Player, Primitive

if TYPE_CHECKING:
    from ..manifest import Manifest

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "roboemot.cue_sounds"
CueKind = Literal["listening", "offline", "not_heard", "fallback"]
CUE_KINDS: tuple[str, ...] = get_args(CueKind)
LISTENING: CueKind = "listening"
GESTURE_PREFIX = "cue_"  # the manifest's gesture_map entry of a cue: cue_<kind>
SOURCE_PREFIX = "cue:"  # the motion source of a cue's gesture
CUE_SOUND_MAX_S = 1.0  # a cue is short: a longer sound is refused with the library
CUE_RATE_RANGE_HZ = (8000, 48000)


@dataclass(frozen=True)
class CueSound:
    """One cue sound: mono float32 samples in [-1, 1] at ``rate_hz``."""

    samples: np.ndarray
    rate_hz: int

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.rate_hz


class CueSoundError(ValueError):
    """A cue library that cannot be built or breaks the rules; refused with the manifest."""


class CueSink(Protocol):
    """What the listener, the conversation and the offline cue need of the cues."""

    def play(self, kind: CueKind) -> bool:
        """Show the one-shot cue ``kind``; whether anything played."""
        ...

    def listening_begin(self) -> None:
        """An utterance's audio left: show the listening cue after its onset."""
        ...

    def listening_end(self) -> None:
        """One utterance that began the listening cue is over (answered, dropped, failed)."""
        ...


class RawOutput(Protocol):
    """``SpeechOutput``'s raw-sound part."""

    def play(self, samples: np.ndarray, rate_hz: int, volume_db: float = 0.0) -> int | None: ...
    def cancel(self, job_id: int) -> None: ...


def validate_cue_sounds(raw: Mapping[str, Any]) -> dict[str, CueSound]:
    """Every sound of ``raw`` checked: a known kind, a :class:`CueSound` (or ``(samples,
    rate_hz)``), mono finite samples in [-1, 1], a rate in range, at most ``CUE_SOUND_MAX_S``.
    Raises :class:`CueSoundError` naming the cue."""
    out: dict[str, CueSound] = {}
    for kind, item in raw.items():
        if kind not in CUE_KINDS:
            raise CueSoundError(f"cue {kind!r}: unknown kind, expected one of {list(CUE_KINDS)}")
        sound = item if isinstance(item, CueSound) else CueSound(*item)
        samples = np.asarray(sound.samples, np.float32)
        low, high = CUE_RATE_RANGE_HZ
        if samples.ndim != 1 or not samples.size:
            raise CueSoundError(f"cue {kind!r}: samples must be mono and not empty")
        if not low <= int(sound.rate_hz) <= high:
            raise CueSoundError(f"cue {kind!r}: rate {sound.rate_hz} Hz outside {low}..{high}")
        if not np.all(np.isfinite(samples)) or float(np.max(np.abs(samples))) > 1.0:
            raise CueSoundError(f"cue {kind!r}: samples must be finite and within [-1, 1]")
        cue = CueSound(samples, int(sound.rate_hz))
        if cue.duration_s > CUE_SOUND_MAX_S:
            raise CueSoundError(f"cue {kind!r}: {cue.duration_s:.2f} s, at most {CUE_SOUND_MAX_S}")
        out[kind] = cue
    return out


def load_cue_sounds(manifest: Manifest) -> dict[str, CueSound]:
    """The validated cue library registered for ``manifest.robot.id``, or an empty one (a robot
    without one shows the gestures only). A factory that fails or a sound that breaks the rules
    raises :class:`CueSoundError`."""
    from ..plugins import _accepts_positional  # plugins imports this module

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name != manifest.robot.id:
            continue
        factory = ep.load()
        try:
            raw = factory(manifest) if _accepts_positional(factory, 1) else factory()
        except Exception as exc:
            raise CueSoundError(f"cue library of {ep.name!r}: {type(exc).__name__}: {exc}") from exc
        return validate_cue_sounds(raw)
    return {}


def peak_time(prim: Primitive) -> float:
    """Time of the keyframe with the largest offset: where the listening gesture holds."""
    best = max(prim.keyframes, key=lambda k: sum(abs(v) for v in k.values.values()))
    return best.t


class _Hold:
    """The listening gesture: plays to its peak, holds there until released, then returns.
    ``began`` is when the utterance that asked for it began it (the hold limit counts from it)."""

    def __init__(self, prim: Primitive, start: float, began: float) -> None:
        self.prim, self.start, self.began = prim, start, began
        self.peak = peak_time(prim)
        self.released: float | None = None
        self.sound: int | None = None

    def elapsed(self, now: float) -> float:
        e = now - self.start
        if e <= self.peak:
            return e
        if self.released is None:
            return self.peak
        return self.peak + max(0.0, now - max(self.released, self.start + self.peak))

    def done(self, now: float) -> bool:
        return self.released is not None and self.elapsed(now) >= self.prim.duration

    def command(self, now: float) -> Command:
        values = self.prim.sample(self.elapsed(now))
        return Command(SOURCE_PREFIX + LISTENING, Priority.EXPRESSION, values, now, additive=True)


class Cues:
    """See the module docstring. ``gesture(name)`` is ``ExpressionRenderer.gesture``;
    ``output()`` the speech output (None before it exists); ``volume_db()`` the circadian
    offset of the moment; ``muted()`` true while asleep or paused; ``speaking(now)`` whether the
    robot's speech (not a raw sound) plays at ``now``."""

    def __init__(
        self,
        cfg: CueConfig,
        *,
        gesture: Callable[[str], Primitive | None],
        sounds: Mapping[str, CueSound],
        output: Callable[[], RawOutput | None] = lambda: None,
        volume_db: Callable[[], float] = lambda: 0.0,
        muted: Callable[[], bool] = lambda: False,
        speaking: Callable[[float], bool] = lambda now: False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._gesture, self._sounds = gesture, dict(sounds)
        self._output, self._volume_db = output, volume_db
        self._muted, self._speaking, self.clock = muted, speaking, clock
        self._lock = threading.Lock()
        self._players: list[Player] = []
        self._outstanding = 0  # utterances that began the listening cue and are not over
        self._due = math.inf  # when the listening gesture shows (inf: not asked)
        self._began = math.inf  # when the oldest outstanding utterance began it
        self._hold: _Hold | None = None
        self._counts: collections.Counter[str] = collections.Counter()

    def apply_config(self, cfg: CueConfig) -> None:
        """Hot reload: the next cue reads ``cfg``."""
        with self._lock:
            self.cfg = cfg

    # ----------------------------------------------------------------- one-shot cues

    def play(self, kind: CueKind) -> bool:
        """Show ``kind`` now: its gesture (replacing one of the same kind still playing) and its
        sound. False when off, muted, or the robot has neither (counted)."""
        with self._lock:
            if not self._allowed(kind):
                return False
            prim = self._gesture(GESTURE_PREFIX + kind)
            sound = self._sounds.get(kind)
            if prim is None and sound is None:
                self._counts[f"{kind}.unsupported"] += 1
                return False
            now = self.clock()
            if prim is not None:
                source = SOURCE_PREFIX + kind
                self._players = [p for p in self._players if p.source != source]
                self._players.append(Player(prim, now, source=source))
            self._sound(kind, sound)
            self._counts[f"{kind}.played"] += 1
            return True

    def _allowed(self, kind: str) -> bool:
        """Whether a cue may show now (lock held); counts why not."""
        if not self.cfg.enabled:
            self._counts[f"{kind}.disabled"] += 1
            return False
        if self._muted():
            self._counts[f"{kind}.muted"] += 1
            return False
        return True

    def _sound(self, kind: str, sound: CueSound | None) -> int | None:
        """Queue ``sound`` on the speech output (lock held); its job id, None when not played."""
        out = self._output() if sound is not None else None
        if sound is None or out is None:
            return None
        try:
            job = out.play(sound.samples, sound.rate_hz, self.cfg.volume_db + self._volume_db())
        except Exception as exc:  # the gesture still shows
            log.warning("cues: %s sound not played (%s)", kind, type(exc).__name__)
            return None
        if job is None:
            self._counts[f"{kind}.sound_dropped"] += 1
        return job

    # ----------------------------------------------------------------- listening

    def listening_begin(self) -> None:
        """An addressed utterance's audio left: the listening gesture shows after the onset
        (a gesture already up stays)."""
        with self._lock:
            now = self.clock()
            self._outstanding += 1
            if self._hold is None or self._hold.released is not None:
                self._due = min(self._due, now + self.cfg.listening_onset_s)
                self._began = min(self._began, now)

    def listening_end(self) -> None:
        """One utterance that began the listening cue is over; the last one releases it."""
        with self._lock:
            self._outstanding = max(0, self._outstanding - 1)
            if not self._outstanding:
                self._release(self.clock(), "ended")

    def _release(self, now: float, why: str) -> None:
        """Stop showing the listening cue (lock held): a gesture returns, a pending one never
        shows, its sound is cut."""
        if self._due < math.inf:
            self._due = self._began = math.inf
            self._counts[f"{LISTENING}.skipped_{why}"] += 1
        hold = self._hold
        if hold is None or hold.released is not None:
            return
        hold.released = now
        self._counts[f"{LISTENING}.released_{why}"] += 1
        out = self._output()
        if hold.sound is not None and out is not None:
            out.cancel(hold.sound)

    def _show(self, now: float) -> None:
        """The onset passed (lock held): show the listening gesture and maybe its sound."""
        self._due = math.inf
        began, self._began = self._began, math.inf
        if not self._allowed(LISTENING):
            return
        prim = self._gesture(GESTURE_PREFIX + LISTENING)
        sound = self._sounds.get(LISTENING) if self.cfg.listening_sound else None
        if prim is None and sound is None:
            self._counts[f"{LISTENING}.unsupported"] += 1
            return
        job = self._sound(LISTENING, sound)
        self._counts[f"{LISTENING}.shown"] += 1
        if prim is not None:
            self._hold = _Hold(prim, now, began)
            self._hold.sound = job

    # ----------------------------------------------------------------- app tick

    def commands(self, now: float) -> list[Command]:
        """The cues' gesture commands for this tick (feed them to the motion loop): starts the
        listening gesture when due and releases it on the answer's first audio or the hold
        limit."""
        with self._lock:
            limit = self.cfg.listening_hold_max_s
            speaking = self._speaking(now)
            if self._due <= now:
                if speaking:
                    self._release(now, "speech")
                elif now - self._began >= limit:
                    self._outstanding = 0
                    self._release(now, "hold_max")
                else:
                    self._show(now)
            hold = self._hold
            if hold is not None and hold.released is None:
                if speaking:
                    self._release(now, "speech")
                elif now - hold.began >= limit:
                    self._outstanding = 0  # a missed end must not hold it for ever
                    self._release(now, "hold_max")
            if hold is not None and hold.done(now):
                self._hold = hold = None
            self._players = [p for p in self._players if not p.done(now)]
            out = [c for p in self._players if (c := p.command(now)) is not None]
            if hold is not None:
                out.append(hold.command(now))
            return out

    # ----------------------------------------------------------------- report

    def snapshot(self) -> dict[str, Any]:
        """Counts per cue kind, what the robot provides, and the listening state (no audio)."""
        with self._lock:
            counts = dict(self._counts)
            holding = self._hold is not None and self._hold.released is None
            outstanding = self._outstanding
        kinds: dict[str, dict[str, int]] = {k: {} for k in CUE_KINDS}
        for key, n in counts.items():
            kind, what = key.split(".", 1)
            kinds[kind][what] = n
        return {
            "enabled": self.cfg.enabled,
            "gestures": [k for k in CUE_KINDS if self._gesture(GESTURE_PREFIX + k) is not None],
            "sounds": sorted(self._sounds),
            "listening": {"holding": holding, "outstanding": outstanding},
            "counts": kinds,
        }


class NoCues:
    """The cues when there is nothing to show them with (tests, no runtime wiring)."""

    def play(self, kind: CueKind) -> bool:
        return False

    def listening_begin(self) -> None:
        """Nothing to show."""

    def listening_end(self) -> None:
        """Nothing to release."""


__all__ = [
    "CUE_KINDS",
    "ENTRY_POINT_GROUP",
    "CueKind",
    "CueSink",
    "CueSound",
    "CueSoundError",
    "Cues",
    "NoCues",
    "load_cue_sounds",
    "peak_time",
    "validate_cue_sounds",
]
