"""The night level of the quiet hours (SPEC 7A, 8).

Inside ``circadian.quiet_hours`` every sound the speech output plays, announcements excepted,
plays at ``circadian.quiet_hours_volume_db`` in total, whatever the circadian state. Speech, cues,
clips and raw sounds already carry the state's ``volume_db``, so the night stage takes off only
the rest; the radio carries none and gets the whole level. The stage never makes a sound louder.
"""

from __future__ import annotations

from collections.abc import Callable

from roboemot.config.schema import RoboEmotConfig


class NightVolume:
    """``config`` gives the live merged config, ``quiet`` whether the quiet hours run now and
    ``state_db`` the circadian state's ``volume_db`` now."""

    def __init__(
        self,
        config: Callable[[], RoboEmotConfig],
        quiet: Callable[[], bool],
        state_db: Callable[[], float],
    ) -> None:
        self.config, self.quiet, self.state_db = config, quiet, state_db

    def active(self) -> bool:
        """Whether the night level applies now: inside the quiet hours with a level set
        (``circadian.quiet_hours_volume_db`` not null). Reads the live config on every call, so a
        hot reload takes effect on the next sound."""
        return self.quiet() and self.config().circadian.quiet_hours_volume_db is not None

    def extra_db(self, *, state_db: float | None) -> float:
        """The gain in dB the speech output adds to a sound now: 0.0 outside the night level;
        the night level less ``state_db``, the state offset the sound carries (captured when it
        was queued), never above 0.0; the whole night level for a sound that carries none
        (``None``: the radio)."""
        night = self.config().circadian.quiet_hours_volume_db
        if night is None or not self.quiet():
            return 0.0
        return night if state_db is None else min(0.0, night - state_db)
