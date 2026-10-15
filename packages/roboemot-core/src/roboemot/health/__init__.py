"""Connectivity tracking for every outside dependency (SPEC 12A, Connectivity and degradation).

A :class:`Link` turns a stream of success/failure reports into an online/offline state with
hysteresis (``failures_to_offline``, ``successes_to_online``) and schedules the next check:
every ``probe_max_s`` while online, then from ``probe_initial_s`` growing by ``probe_backoff``
up to ``probe_max_s`` while offline, each delay spread by ``probe_jitter`` so robots on one
network do not retry in lockstep. State changes are reported to ``on_change`` as bus-style
events, ``link.<name>.lost`` and ``link.<name>.restored``.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config.schema import LinkPolicy

OnChange = Callable[[str, float], None]  # (event, now)


@dataclass
class Link:
    name: str
    policy: LinkPolicy
    on_change: OnChange | None = None
    rng: random.Random = field(default_factory=random.Random)
    online: bool = True  # assumed until proven otherwise, so start-up is not "offline"
    next_check: float = 0.0
    _fails: int = 0
    _oks: int = 0
    _delay: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def report(self, ok: bool, now: float) -> bool:
        """Record one outcome (a real request or a check); returns True if the state changed."""
        with self._lock:
            changed = False
            if ok:
                self._fails, self._oks = 0, self._oks + 1
                if not self.online and self._oks >= self.policy.successes_to_online:
                    self.online, changed = True, True
            else:
                self._oks, self._fails = 0, self._fails + 1
                if self.online and self._fails >= self.policy.failures_to_offline:
                    self.online, changed = False, True
                    self._delay = 0.0
            self._schedule(now, ok)
        if changed and self.on_change is not None:
            self.on_change(f"link.{self.name}.{'restored' if self.online else 'lost'}", now)
        return changed

    def set(self, online: bool, now: float) -> bool:
        """Force the state, for aggregates whose hysteresis happened on their members."""
        with self._lock:
            changed, self.online = self.online != online, online
            self._fails = self._oks = 0
        if changed and self.on_change is not None:
            self.on_change(f"link.{self.name}.{'restored' if online else 'lost'}", now)
        return changed

    def would_restore(self) -> bool:
        """True if one more success brings the link back online."""
        return not self.online and self._oks + 1 >= self.policy.successes_to_online

    def snapshot(self, now: float) -> dict[str, object]:
        """State for diagnostics (SPEC 12A); safe to call from any thread."""
        with self._lock:
            return {
                "online": self.online,
                "consecutive_failures": self._fails,
                "consecutive_successes": self._oks,
                "next_check_in_s": max(0.0, self.next_check - now),
            }

    def due(self, now: float) -> bool:
        return now >= self.next_check

    def _schedule(self, now: float, ok: bool) -> None:
        p = self.policy
        if self.online:
            base = p.probe_max_s if ok else p.probe_initial_s  # a failure is re-checked soon
        elif ok:
            base = p.probe_initial_s  # confirm a recovery quickly
        else:
            grown = min(p.probe_max_s, self._delay * p.probe_backoff)
            self._delay = base = p.probe_initial_s if self._delay == 0 else grown
        spread = base * p.probe_jitter
        self.next_check = now + base + self.rng.uniform(-spread, spread)


__all__ = ["Link", "OnChange"]
