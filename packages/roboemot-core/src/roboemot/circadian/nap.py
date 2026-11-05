"""The day nap's decisions (SPEC 7A, Day nap): the idle clock with its reset rules, the face
confirmation, and when a nap starts and ends.

The idle clock adds up time outside quiet hours with no confirmed face and no addressed speech:
quiet hours pause it (idle from 22:15 and quiet hours from 22:30 make a nap at 07:15). It
restarts from zero on activity and while a nap is impossible (``blocked``: naps off, the robot
asleep, a hold, a rest of the rhythm's own, attention on a person), so a stale clock never fires a
nap the moment one becomes possible again. Presence (``held``) only holds a due nap off: a single
blip never resets the clock. Times are passed in; the caller (``CircadianRunner``) holds its lock
and acts on the model (a wake, the ``nap`` input).

A nap ends on activity (``activity``: a confirmed face, addressed speech, a conversation) and on
a stimulus (``stimulus``: the room light rising, motion in front of a still camera, a loud sound;
owner decision 2026-11-03). A stimulus only ends a nap in progress: while awake it leaves the
idle clock alone, because none of them is a person engaging with the robot (a television's bang,
a car passing the window, clouds, the robot's own moves), and a busy room would otherwise never
let it nap. ``wakes`` counts the wakes by reason for the debug report.

After a stimulus wake the next nap comes sooner (owner decision 2026-11-05): it needs only
``resume_after_s`` (``circadian.nap_resume_after_s``) of idle time instead of ``after_s``, counted
like the other from the end of the wake hold, unless a person follows: activity (a confirmed face,
addressed speech, a conversation) or a wake or hold from outside brings back ``after_s``. The
quiet hours and naps off end the pending resume too, so the next day starts with the full idle
time, and so do the end of a platform sleep and a rest of the rhythm's own (the caller's
:meth:`disarm`). Presence and attention on a person hold a due resumed nap off but keep the resume
armed (owner decision 2026-11-05). ``resumes`` counts the naps started this way.
"""

from __future__ import annotations

WAKE_REASONS = ("face", "speech", "light", "motion", "loud")


class DayNap:
    """See the module docstring."""

    def __init__(self) -> None:
        self.napping = False
        self.naps = 0  # count, for the debug report
        self.idle_s = 0.0
        self.started: float | None = None  # time the nap in progress started
        self.wakes = dict.fromkeys(WAKE_REASONS, 0)  # count by reason, for the debug report
        self.resumes = 0  # naps started after resume_after_s, for the debug report
        self.resuming = False  # the last wake was a stimulus and no person has followed
        self._last: float | None = None  # last counted time; None while paused or not started
        self._face: tuple[float, float] | None = None  # (clock time, frame stamp) of a face

    def restart(self, t: float) -> None:
        """The idle clock starts again from zero at ``t``."""
        self.idle_s, self._last = 0.0, t

    def face(self, t: float, stamp: float, window_s: float) -> bool:
        """A face seen at ``t`` in the frame of ``stamp``: whether it is confirmed, i.e. a second
        sighting from another frame within ``window_s`` of the previous one."""
        last, self._face = self._face, (t, stamp)
        return last is not None and t - last[0] <= window_s and stamp != last[1]

    def activity(self, t: float, reason: str) -> bool:
        """A confirmed face, addressed speech or a conversation at ``t`` (``reason``: ``face``
        or ``speech``): the clock restarts. Returns whether a nap was in progress (the caller
        wakes the robot); that wake is counted under ``reason``."""
        self.restart(t)
        self.resuming = False
        woke, self.napping = self.napping, False
        if woke:
            self.wakes[reason] += 1
        return woke

    def stimulus(self, t: float, reason: str) -> bool:
        """The light rising, motion or a loud sound at ``t`` (``reason``): ends a nap in progress
        like activity, arms the resume and returns True; while awake it does nothing (module
        docstring)."""
        if not (self.napping and self.activity(t, reason)):
            return False
        self.resuming = True
        return True

    def disarm(self) -> None:
        """The next nap needs the full ``after_s`` again (a platform sleep ended, the rhythm's
        own rest): a pending resume is dropped; the idle clock is left alone."""
        self.resuming = False

    def cancel(self, t: float) -> None:
        """A wake or a hold from outside (the app, the web UI): no nap, the clock restarts."""
        self.napping = self.resuming = False
        self.restart(t)

    def step(
        self,
        t: float,
        *,
        after_s: float | None,
        resume_after_s: float,
        quiet: bool,
        blocked: bool,
        held: bool,
    ) -> bool:
        """One update at ``t``. Returns whether a nap starts now.

        ``after_s`` is ``circadian.nap_after_s`` (None: naps off), ``resume_after_s``
        ``circadian.nap_resume_after_s`` (used after a stimulus wake), ``quiet`` the quiet hours,
        ``blocked`` a reason no nap is possible now (the clock restarts, a pending resume stays:
        the wake hold blocks), ``held`` presence that holds one off."""
        if quiet or after_s is None:
            self.resuming = False  # the night or naps off: the next nap needs after_s again
        if self.napping:
            if quiet or after_s is None:
                self.napping = False  # quiet hours rest the robot anyway; null: naps off now
                self.restart(t)
            return False
        if after_s is None or blocked:
            self.restart(t)
            return False
        if quiet:
            self._last = None  # paused: the time before the quiet hours is kept
            return False
        if self._last is not None:
            self.idle_s += max(0.0, t - self._last)
        self._last = t
        if self.idle_s < (resume_after_s if self.resuming else after_s) or held:
            return False
        self.napping = True
        self.naps += 1
        if self.resuming:
            self.resumes += 1
            self.resuming = False
        self.started = t
        return True
