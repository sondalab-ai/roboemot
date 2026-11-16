"""The timers and reminders store and the runner that publishes due items (SPEC 11, Voice
tools: Scheduler store, Limits, Missed items, Due items; 11A ``tools.scheduler.*``).

:class:`Scheduler` keeps the active items in one file in the data folder, mode 0600, rewritten
atomically on every change and read at start. A change is written before it takes effect in
memory, so a failed write changes nothing. Errors are :class:`ScheduleError` codes the skills
hand to the model. The restart grace (``missed_grace_s``) applies only in :meth:`load`: while
the process runs a due item never expires.

:class:`SchedulerRunner` publishes ``tools.due`` with the ids of the due items every
``tick_s``, again every tick until each is marked fired (no defer). Until an announcer attaches
(:meth:`SchedulerRunner.set_consumer`) it logs each due id and marks it fired itself.

Logs carry ids, kinds and counts only, never a label or a reminder text (SPEC 12).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo
from datetime import time as ClockTime
from pathlib import Path
from typing import Any

from .. import paths
from ..bus import Bus
from ..config.schema import SchedulerConfig
from ..fileio import replace_file
from .schedule import ISO_DAYS, REPEATS, Item, Kind, Repeat, from_json, next_due, to_json

log = logging.getLogger(__name__)

FILE_MODE = 0o600
FILE_VERSION = 1
DUE_TOPIC = "tools.due"
JUST_BEFORE = timedelta(microseconds=1)  # so an exact first occurrence counts


def _number(item_id: str) -> int:
    """The number of an id ``t<n>`` or ``r<n>``; 0 for any other spelling (a hand edit), so
    the next id given is always above every stored one."""
    digits = item_id[1:]
    return int(digits) if digits.isascii() and digits.isdigit() else 0


def _entries(raw: dict[str, Any], key: str) -> list[Any]:
    """The list under ``key`` of the store object; empty when it is missing or not a list (a
    hand edit), so a bad file never stops the run."""
    value = raw.get(key)
    return value if isinstance(value, list) else []


class ScheduleError(ValueError):
    """A request the store refuses. ``code`` is one of ``full``, ``too_long``, ``in_past``,
    ``too_far``, ``bad_days`` and ``empty`` (a reminder with no text); the message is the code
    only, never a value."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Scheduler:
    """The active timers and reminders, persisted at ``path``. ``cfg`` reads the live
    ``tools.scheduler`` section, ``now`` the aware wall clock and ``tz`` the zone a repeating
    reminder recurs in (``circadian.timezone``). Thread-safe under one lock."""

    def __init__(
        self,
        path: Path,
        cfg: Callable[[], SchedulerConfig],
        now: Callable[[], datetime],
        tz: Callable[[], tzinfo],
    ) -> None:
        self.path, self.cfg, self.now, self.tz = path, cfg, now, tz
        self._lock = threading.RLock()
        self._items: dict[str, Item] = {}
        self._missed: list[Item] = []
        self._seq = 0  # the last id number given, kept in the file so ids never repeat

    def load(self, started: datetime) -> None:
        """Read the file (none: empty; unreadable: logged by type, empty). An item due more
        than ``missed_grace_s`` before ``started`` goes to :meth:`missed`; a repeating one also
        moves to its next occurrence after ``started``. Within the grace an item stays due."""
        raw = self._read()
        grace = timedelta(seconds=self.cfg().missed_grace_s)
        tz = self.tz()
        items: dict[str, Item] = {}
        missed = [i for i in (self._parse(r) for r in _entries(raw, "missed")) if i is not None]
        late = 0
        for item in (self._parse(r) for r in _entries(raw, "items")):
            if item is None:
                continue
            if started - item.due > grace:
                late += 1
                missed.append(item)
                nxt = next_due(item, started, tz)
                if nxt is None:
                    continue
                item = replace(item, due=nxt)
            items[item.id] = item
        seq = raw.get("seq", 0)
        used = [_number(i.id) for i in [*items.values(), *missed]]
        with self._lock:
            self._seq = max([seq if isinstance(seq, int) else 0, *used])
            self._items, self._missed = items, missed[-self.cfg().max_items :]
            if late:
                try:
                    self._write(self._items, self._missed, self._seq)
                except OSError as exc:
                    log.warning("scheduler: store not written: %s", type(exc).__name__)
        log.info("scheduler: %d active, %d missed at start", len(items), late)

    def add_timer(self, seconds: int, label: str | None) -> Item:
        """A timer due ``seconds`` from now with an optional ``label``. ``too_long`` past
        ``text_max_chars``, ``too_far`` past ``timer_max_s``, ``in_past`` for no time at all,
        ``full`` at ``max_items``."""
        cfg, text = self.cfg(), label or ""
        if len(text) > cfg.text_max_chars:
            raise ScheduleError("too_long")
        if seconds > cfg.timer_max_s:
            raise ScheduleError("too_far")
        if seconds <= 0:
            raise ScheduleError("in_past")
        now = self.now().astimezone(UTC)
        return self._add("timer", text, now + timedelta(seconds=seconds), now, "once", (), None)

    def add_reminder(
        self, text: str, due_local: datetime, repeat: Repeat, days: tuple[int, ...]
    ) -> Item:
        """A reminder at ``due_local`` (naive: in ``tz``). ``once`` must lie ahead
        (``in_past``). A repeating one recurs at that local clock time from the first matching
        day on or after ``due_local`` (never ``in_past``); ``days`` are ISO weekdays, required
        for ``days`` (``bad_days``) and ignored otherwise. ``too_long`` and ``full`` as timers."""
        cfg, tz = self.cfg(), self.tz()
        if repeat not in REPEATS:
            raise ValueError("unknown repeat")
        if not text.strip():
            raise ScheduleError("empty")
        if len(text) > cfg.text_max_chars:
            raise ScheduleError("too_long")
        wanted: tuple[int, ...] = ()
        if repeat == "days":
            wanted = tuple(sorted(set(days)))
            if not wanted or not set(wanted) <= ISO_DAYS:
                raise ScheduleError("bad_days")
        local = (due_local if due_local.tzinfo else due_local.replace(tzinfo=tz)).astimezone(tz)
        now = self.now().astimezone(UTC)
        if repeat == "once":
            due = local.astimezone(UTC)
            if due <= now:
                raise ScheduleError("in_past")
            return self._add("reminder", text, due, now, repeat, (), None)
        clock = local.time().replace(second=0, microsecond=0)
        probe = Item("", "reminder", text, now, now, repeat, wanted, clock)
        first = next_due(probe, max(now, local - JUST_BEFORE), tz)
        if first is None:  # every repeat has a day within a week: defensive only
            raise ScheduleError("bad_days")
        return self._add("reminder", text, first, now, repeat, wanted, clock)

    def cancel(self, item_id: str) -> bool:
        """Remove the active item ``item_id``; False when there is none."""
        with self._lock:
            if item_id not in self._items:
                return False
            items = {k: v for k, v in self._items.items() if k != item_id}
            self._write(items, self._missed, self._seq)
            kind = self._items[item_id].kind
            self._items = items
        log.info("scheduler: %s %s cancelled (%d active)", kind, item_id, len(items))
        return True

    def items(self) -> list[Item]:
        """The active items, soonest first."""
        with self._lock:
            return sorted(self._items.values(), key=lambda i: (i.due, i.id))

    def get(self, item_id: str) -> Item | None:
        """The active item ``item_id``, or None."""
        with self._lock:
            return self._items.get(item_id)

    def missed(self) -> list[Item]:
        """The items dropped at start because they were later than ``missed_grace_s``."""
        with self._lock:
            return list(self._missed)

    def clear_missed(self) -> None:
        """Forget the missed items (the Tools tab's dismiss)."""
        with self._lock:
            if self._missed:
                self._write(self._items, [], self._seq)
                self._missed = []

    def due(self, now: datetime) -> list[Item]:
        """The active items due at ``now``, soonest first. A due item stays due until
        :meth:`fired` or :meth:`cancel`, however late."""
        return [i for i in self.items() if i.due <= now]

    def fired(self, item_id: str) -> None:
        """``item_id`` was announced: a ``once`` item is removed, a repeating one moves to its
        next occurrence after both now and its due time (a late one does not fire again at
        once). An unknown id (cancelled meanwhile) is ignored."""
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return
            nxt = next_due(item, max(self.now().astimezone(UTC), item.due), self.tz())
            items = dict(self._items)
            if nxt is None:
                del items[item_id]
            else:
                items[item_id] = replace(item, due=nxt)
            self._write(items, self._missed, self._seq)
            self._items = items
        log.info("scheduler: %s %s fired (%s)", item.kind, item_id, "next" if nxt else "done")

    def _add(
        self,
        kind: Kind,
        text: str,
        due: datetime,
        created: datetime,
        repeat: Repeat,
        days: tuple[int, ...],
        clock: ClockTime | None,
    ) -> Item:
        """Store a new item under the next id (``t<n>`` or ``r<n>``); ``full`` at
        ``max_items``. The file is written before the item counts."""
        with self._lock:
            if len(self._items) >= self.cfg().max_items:
                raise ScheduleError("full")
            seq = self._seq + 1
            item = Item(f"{kind[0]}{seq}", kind, text, due, created, repeat, days, clock)
            items = {**self._items, item.id: item}
            self._write(items, self._missed, seq)
            self._items, self._seq = items, seq
        log.info("scheduler: %s %s added (%d active)", kind, item.id, len(items))
        return item

    def _write(self, items: dict[str, Item], missed: list[Item], seq: int) -> None:
        """Replace the store file with ``items``, ``missed`` and ``seq`` (atomic, mode 0600)."""
        body = {
            "version": FILE_VERSION,
            "seq": seq,
            "items": [to_json(i) for i in sorted(items.values(), key=lambda i: (i.due, i.id))],
            "missed": [to_json(i) for i in missed],
        }
        if not self.path.parent.exists():  # a fresh install: created private, as data_dir does
            paths.data_dir(str(self.path.parent))
        replace_file(self.path, json.dumps(body, ensure_ascii=False, indent=1), FILE_MODE)

    def _read(self) -> dict[str, Any]:
        """The store file's object; empty when there is none or it cannot be read (logged by
        exception type only: the file holds the person's texts)."""
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("scheduler: store unreadable, starting empty: %s", type(exc).__name__)
            return {}
        if not isinstance(raw, dict):
            log.warning("scheduler: store unreadable, starting empty: not an object")
            return {}
        return raw

    @staticmethod
    def _parse(raw: Any) -> Item | None:
        """One stored item, or None (logged by exception type) when it is malformed."""
        try:
            return from_json(raw)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            log.warning("scheduler: a stored item was skipped: %s", type(exc).__name__)
            return None


class SchedulerRunner:
    """A daemon thread that publishes ``tools.due`` with the due ids every ``tick_s``, again
    every tick until each is marked fired. ``set_consumer(True)`` is the announcer's attach
    point (SPEC 11, Due items): until then each due id is logged and marked fired here."""

    def __init__(self, scheduler: Scheduler, bus: Bus, cfg: Callable[[], SchedulerConfig]) -> None:
        self._scheduler, self._bus, self._cfg = scheduler, bus, cfg
        self._consumer = False
        self._failing = False  # the last tick raised: its warning is out already
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_consumer(self, attached: bool) -> None:
        """Whether an announcer consumes ``tools.due`` and calls :meth:`Scheduler.fired`
        itself. While none is attached the runner marks each due item fired (log only)."""
        self._consumer = attached

    @property
    def consumer_attached(self) -> bool:
        return self._consumer

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def tick(self) -> list[str]:
        """One check: publish the due ids (nothing when none is due) and, with no consumer,
        mark each fired. Returns the ids published."""
        due = self._scheduler.due(self._scheduler.now())
        if not due:
            return []
        ids = [i.id for i in due]
        self._bus.publish(DUE_TOPIC, ids, time.monotonic())
        if not self._consumer:
            for item in due:
                log.info("scheduler: %s %s due, no announcer: marked fired", item.kind, item.id)
                self._scheduler.fired(item.id)
        return ids

    def start(self) -> None:
        """Start the thread (once)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="roboemot-scheduler", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the thread and wait for its current tick to end."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def guarded_tick(self) -> None:
        """One :meth:`tick` that never raises. A failure (a write on a full disk) is logged by
        type once per streak of failing ticks, not every tick for the whole outage; the next
        tick that works ends the streak."""
        try:
            self.tick()
        except Exception as exc:
            if not self._failing:
                log.warning("scheduler: tick failed: %s", type(exc).__name__)
            self._failing = True
        else:
            if self._failing:
                log.info("scheduler: ticks work again")
            self._failing = False

    def _run(self) -> None:
        """Tick every ``tick_s`` (read live) until closed; a failing tick does not stop the
        next one, so one bad write never stops due items for the run."""
        while not self._stop.wait(self._cfg().tick_s):
            self.guarded_tick()
