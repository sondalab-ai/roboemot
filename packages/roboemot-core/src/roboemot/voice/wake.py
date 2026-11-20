"""Whether an utterance is addressed to the robot (SPEC 8 stage 3, ``voice.wake_word_policy``).

The wake word is *required* with policy ``always``, never with ``never``, and with
``unless_enrolled_in_view`` only while no enrolled person is in view (SPEC 8 stage 2).

Every utterance that would be addressed *without* the wake word (not required, or a follow-up in
the open window) must also pass ``at_face``, the voice-at-face check (``voice.direction``): its
sound direction pointed at a visible face (else ``rejected_no_face_at_voice``) and, with
``voice.facing`` on, that face looked at the robot (else ``rejected_not_facing``, or
``rejected_no_facing_data`` when the face had no head yaw). A rejected
utterance needs the wake word, so music, a television or a video is not answered. An utterance
with the wake word never runs the check.

When required, an utterance is addressed if it contains the wake word, or it starts within
``voice.fastpath.addressed_window_s`` after the last speech of an addressed one. A follow-up
extends the window from its own last speech, but never past ``window_max_s`` from the wake word
that opened it, and when both utterances' speakers are known people they must be the same person.

While music plays at any time in the utterance (``music(start, end)``, the music detector,
SPEC 8) the wake word is required whatever the policy, and the open window does not count:
without the wake word the outcome is ``rejected_music``, before any other rule and without the
voice-at-face check. An utterance with the wake word is addressed over music as without it.

While the robot's own radio can be heard as the utterance starts (``media(start)``, the media
player's ``audible_at``, SPEC 8, Radio playback) the same holds with the outcome
``rejected_media``: only the wake word (the name or a generic wake phrase) addresses the robot, so
it never answers the lyrics of its own music (one exception, ``face_media``, below). The follow-up
window, a face in view and the policy do not count. It is asked before the music rule. While the
player holds the radio (robot speech, an open addressed turn, an announcement's reply window, and
``resume_after_s`` after them) no lyric can be heard, so the normal rules apply: "Neve" then
"stoppa la musica" is a follow-up in the window the name opened. An utterance is decided by its
start stamp, so one that started while the radio was held keeps the normal rules.

Over the audible radio the voice-at-face check can stand in for the wake word (owner decision
2026-11-20, ``tools.radio.face_addresses``): when the caller says a server transcript can
confirm it (``face_media``: a realtime audio route takes the utterance's audio) and the policy
would address it without the wake word were the radio silent (``never``, or
``unless_enrolled_in_view`` with an enrolled person in view; never under ``always``), an
utterance without the wake word whose voice came from a visible face (looking at the robot, with
``voice.facing`` on) is ``face_media``: addressed for now and opening the window as the wake word
does, but a wake the server's words must confirm (the listener). One that fails the check stays
``rejected_media``. Lyrics come from the robot's own speaker, not from a face.

A robot-initiated turn's reply window (``open_window``, an announcement; owner decision
2026-11-17) is one such hold: the media player keeps the radio paused while it is open
(``reply_window``), so the media rule does not apply in it and a follow-up extends it, never
past ``window_max_s``. A window the wake word opened is not a reply window: the radio rejoins
after the turn, and its lyrics are then rejected.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

DECISIONS = (
    "not_required",
    "wake_word",
    "window",
    "rejected_no_window",
    "rejected_other_speaker",
    "rejected_no_face_at_voice",
    "rejected_not_facing",
    "rejected_no_facing_data",
    "rejected_music",
    "rejected_media",
    "face_media",
)
# The voice-at-face check's verdicts (``at_face``); a rejection is counted ``rejected_<verdict>``.
AT_FACE = "at_face"
NO_FACE_AT_VOICE = "no_face_at_voice"
NOT_FACING = "not_facing"
NO_FACING_DATA = "no_facing_data"
AtFace = Callable[[], str]
Music = Callable[[float, float], bool]  # music at any time in [start, end] (audio clock)
Media = Callable[[float], bool]  # the robot's own radio could be heard at a stamp


class WakeGate:
    """Thread-safe addressed-window state machine; stamps are in the audio clock (seconds)."""

    def __init__(
        self,
        policy: str,
        window_s: float,
        max_s: float = math.inf,
        music: Music | None = None,
        media: Media | None = None,
    ) -> None:
        """``music`` says whether music played in [start, end] (None: never); ``media``
        whether the robot's own radio could be heard at a stamp (None: never). Both run under
        this gate's lock, like ``at_face``, so they may take only leaf locks (never this gate's
        own: ``reply_window`` takes it)."""
        self._lock = threading.Lock()
        self._music = music
        self._media = media
        self._policy = policy
        self._window = window_s
        self._max = max_s
        self._until = -math.inf
        self._opened = -math.inf
        self._reply = False  # the open window is a robot-initiated turn's reply window
        self._speaker: str | None = None
        self._decisions = dict.fromkeys(DECISIONS, 0)

    def apply(self, policy: str, window_s: float, max_s: float = math.inf) -> None:
        """Take a reloaded policy and window lengths; an open window stays open."""
        with self._lock:
            self._policy, self._window, self._max = policy, window_s, max_s

    def close_window(self) -> None:
        """Close the open follow-up window now: the next utterance needs the wake word wherever
        the policy requires it (a requested nap started, SPEC 7A). The policy is unchanged."""
        with self._lock:
            self._until = -math.inf
            self._reply = False

    def open_window(self, at: float) -> None:
        """Open the follow-up window at ``at`` (audio clock) as an addressed utterance ending
        then would: a robot-initiated turn's reply ended (SPEC 8, Conversation state), so the
        person can answer without the wake word. Any speaker may follow it. It is a reply
        window: the media player holds the radio while it is open (module docstring)."""
        with self._lock:
            self._opened, self._speaker, self._reply = at, None, True
            self._until = at + min(self._window, self._max)

    def reply_window(self, at: float) -> bool:
        """Whether a robot-initiated turn's reply window is open at ``at`` (audio clock): the
        media player keeps the radio paused while it is."""
        with self._lock:
            return self._reply_at(at)

    def _reply_at(self, at: float) -> bool:
        """``reply_window`` with the lock held."""
        return self._reply and at <= self._until

    def decide(
        self,
        *,
        wake_word: bool,
        enrolled_in_view: bool,
        start: float,
        last_speech: float,
        speaker: str | None = None,
        at_face: AtFace | None = None,
    ) -> bool:
        """Whether the utterance (first and last speech stamps, speaker id if known) is addressed.
        ``at_face`` is the voice-at-face check (its verdict, ``AT_FACE`` to pass), called only
        when the utterance would be addressed without the wake word (None: no check). It runs
        under this gate's lock, so it may take only leaf locks (``VoiceDirection`` does)."""
        return not self.outcome(
            wake_word=wake_word,
            enrolled_in_view=enrolled_in_view,
            start=start,
            last_speech=last_speech,
            speaker=speaker,
            at_face=at_face,
        ).startswith("rejected")

    def outcome(
        self,
        *,
        wake_word: bool,
        enrolled_in_view: bool,
        start: float,
        last_speech: float,
        speaker: str | None = None,
        at_face: AtFace | None = None,
        face_media: bool = False,
    ) -> str:
        """``decide`` with its outcome (one of ``DECISIONS``): a ``rejected_*`` one is not
        addressed, and says which rule rejected it (the Live tab's decision event).
        ``face_media``: over the audible radio an utterance without the wake word that passes
        ``at_face`` is ``face_media`` (a wake the server's transcript must confirm, module
        docstring) instead of ``rejected_media``, only where the policy does not require the
        wake word; the caller sets it only when that transcript can come."""
        with self._lock:
            media = self._media_at(start)
            music = media or self._music_in(start, last_speech)
            policy_required = self._required(enrolled_in_view)
            required = policy_required or music
            if not required:
                outcome = "not_required"
                if not wake_word:
                    outcome = _face_outcome(outcome, at_face)
                self._decisions[outcome] += 1
                return outcome
            if wake_word:
                self._reply = self._reply_at(start)  # inside a reply window it goes on as one
                self._opened, self._speaker = last_speech, speaker
                outcome = "wake_word"
            elif media:
                outcome = "rejected_media"
                # only where the policy would address it without the wake word (radio aside)
                face = face_media and not policy_required and at_face is not None
                if face and at_face() == AT_FACE:
                    self._reply = False
                    self._opened, self._speaker = last_speech, speaker
                    outcome = "face_media"
            else:
                outcome = self._follow(start, speaker, music)
                if outcome == "window":
                    outcome = _face_outcome(outcome, at_face)
            self._decisions[outcome] += 1
            if outcome.startswith("rejected"):
                return outcome
            self._until = min(last_speech + self._window, self._opened + self._max)
            return outcome

    def addressable(
        self,
        *,
        enrolled_in_view: bool,
        start: float,
        speaker: str | None = None,
        end: float | None = None,
    ) -> bool:
        """Whether an utterance from ``start`` to ``end`` (None: ``start``) would be addressed
        without the wake word, its voice-at-face check aside: no music in it, and the wake word
        is not required or it falls in the open window (the same person if both are known).
        Nothing is counted or changed: the listener asks it when speech starts and when it ends
        (SPEC 8, Realtime speech route), ``decide`` still decides."""
        with self._lock:
            if self._media_at(start):
                return False
            music = self._music_in(start, start if end is None else end)
            required = self._required(enrolled_in_view) or music
            return not required or self._follow(start, speaker, music) == "window"

    def policy_requires(self, enrolled_in_view: bool) -> bool:
        """Whether the policy alone requires the wake word now (music and the radio aside):
        the face rule over the radio applies only where it does not (owner decision
        2026-11-20)."""
        with self._lock:
            return self._required(enrolled_in_view)

    def _required(self, enrolled_in_view: bool) -> bool:
        """Whether the policy requires the wake word: always with ``always``, never with
        ``never``, and with ``unless_enrolled_in_view`` only while no enrolled person is in
        view (lock held). The one place both ``outcome`` and ``addressable`` ask it."""
        return self._policy == "always" or (
            self._policy == "unless_enrolled_in_view" and not enrolled_in_view
        )

    def _music_in(self, start: float, end: float) -> bool:
        """Whether music played at any time in [start, end] (lock held); False without a
        music predicate. Asked once per decision, so the rule and its reason agree."""
        return self._music is not None and self._music(start, end)

    def _media_at(self, start: float) -> bool:
        """Whether the robot's own radio could be heard at ``start`` (lock held); False without
        a predicate."""
        return self._media is not None and self._media(start)

    def _follow(self, start: float, speaker: str | None, music: bool = False) -> str:
        """Outcome of an utterance without the wake word: ``rejected_music`` while music
        plays, ``window`` when it falls in the open window (same person if both are known),
        else the ``rejected_*`` reason (lock held)."""
        if music:
            return "rejected_music"
        if start > self._until:
            return "rejected_no_window"
        if speaker is None or self._speaker is None or speaker == self._speaker:
            return "window"
        return "rejected_other_speaker"

    def snapshot(self, now: float) -> dict[str, Any]:
        """Policy, window state and decision counts for the debug report (no content)."""
        with self._lock:
            return {
                "policy": self._policy,
                "window_open": now <= self._until,
                "window_left_s": max(0.0, self._until - now) if math.isfinite(self._until) else 0.0,
                "decisions": dict(self._decisions),
            }


def _face_outcome(outcome: str, at_face: AtFace | None) -> str:
    """``outcome`` when there is no voice-at-face check or it passes, else its rejection."""
    if at_face is None:
        return outcome
    verdict = at_face()
    return outcome if verdict == AT_FACE else f"rejected_{verdict}"
