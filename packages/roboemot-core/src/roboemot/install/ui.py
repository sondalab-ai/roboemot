"""Wizard output and prompts for the robot installer (SPEC 4A, Packaging).

Colours come from the Sondalab palette (``sondalab-palette``, optional: plain text without it)
and only when stdout is a terminal. Non-interactive runs take each prompt's default;
``assume_yes`` accepts every confirmation.
"""

from __future__ import annotations

import getpass
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

try:
    import sondalab_palette as _palette
except ImportError:  # pragma: no cover - plain output
    _palette = None


class WizardError(Exception):
    """A step failed; the message says what to do about it."""


def paint(text: str, role: str) -> str:
    """Colour ``text`` with a Sondalab role, only when stdout is a terminal."""
    return _palette.paint(text, role) if _palette and sys.stdout.isatty() else text


SPINNER = "|/-\\"
TICK_S = 1.0


def format_elapsed(seconds: float) -> str:
    """``42s``, ``1m42s``, ``1h02m05s``."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


class Progress:
    """One long remote step: call it with each output line; set ``failed`` for a soft failure
    (pip exited non-zero without raising), so the closing ``✓`` line is skipped."""

    def __init__(self) -> None:
        self.last = ""
        self.failed = False
        self.feed: Callable[[str], None] = lambda raw: None

    def __call__(self, raw: str) -> None:
        self.feed(raw)


class Ui:
    def __init__(self, interactive: bool, assume_yes: bool) -> None:
        self.interactive = interactive
        self.assume_yes = assume_yes
        self.step_no = 0
        self.steps = 0
        self.tick_s = TICK_S
        self.clock: Callable[[], float] = time.monotonic
        self._out_lock = threading.Lock()

    def title(self, text: str, note: str = "") -> None:
        print(
            f"\n{paint('RoboEmot', 'accent')} {text}" + (f"  {paint(note, 'warn')}" if note else "")
        )

    def step(self, name: str) -> None:
        self.step_no += 1
        print(f"\n{paint(f'[{self.step_no}/{self.steps}]', 'dim')} {paint(name, 'prompt')}")

    def ok(self, msg: str) -> None:
        print(f"  {paint('✓', 'ok')} {msg}")

    def info(self, msg: str) -> None:
        print(f"  {paint('·', 'dim')} {paint(msg, 'dim')}")

    def warn(self, msg: str) -> None:
        print(f"  {paint('!', 'warn')} {msg}")

    def fail(self, msg: str) -> None:
        print(f"  {paint('✗', 'err')} {msg}")

    @contextmanager
    def progress(
        self,
        label: str,
        on_line: Callable[[str], str | None] | None = None,
        done: str | None = None,
    ) -> Iterator[Progress]:
        """Announce a long remote step and show it running; yields a ``Progress`` to call per line.

        On a terminal one status line is rewritten in place (spinner, elapsed time, latest
        line) and a ticker keeps the time moving while the command is silent. Otherwise each
        line is printed plainly. ``on_line`` maps a raw output line to the text worth showing
        (None drops it). On success the step ends with a ``✓`` line carrying the total time.
        """
        keep = on_line or (lambda text: text)
        live = sys.stdout.isatty()
        state = Progress()
        start = self.clock()
        self.info(f"{label} (can take minutes on the robot)")
        stop = threading.Event()
        frame = [0]

        def draw() -> None:
            width = max(shutil.get_terminal_size().columns - 1, 20)
            spin = SPINNER[frame[0] % len(SPINNER)]
            head = f"  {spin} {label} {format_elapsed(self.clock() - start)}"
            text = f"{head}  {state.last}" if state.last else head
            if len(text) > width:
                text = text[: width - 1] + "…"
            with self._out_lock:
                sys.stdout.write(f"\r{text}\x1b[K")
                sys.stdout.flush()

        def clear() -> None:
            with self._out_lock:
                sys.stdout.write("\r\x1b[K")
                sys.stdout.flush()

        def feed(raw: str) -> None:
            shown = keep(raw)
            if not shown:
                return
            state.last = shown
            if live:
                draw()
            else:
                self.info(shown)

        def tick() -> None:
            while not stop.wait(self.tick_s):
                frame[0] += 1
                draw()

        state.feed = feed
        ticker = threading.Thread(target=tick, daemon=True)
        if live:
            draw()
            ticker.start()
        try:
            yield state
        except BaseException:
            state.failed = True
            raise
        finally:
            stop.set()
            if live:
                ticker.join()
                clear()
        if not state.failed:
            self.ok(f"{done or label} ({format_elapsed(self.clock() - start)})")

    def ask(self, question: str, default: str) -> str:
        if not self.interactive:
            return default
        try:
            answer = input(f"  {paint('?', 'prompt')} {question} {paint(f'[{default}]', 'dim')} ")
        except EOFError:
            return default
        return answer.strip() or default

    def ask_secret(self, question: str) -> str | None:
        """A hidden answer from the terminal; None when not interactive."""
        if not self.interactive:
            return None
        try:
            return getpass.getpass(f"  {paint('?', 'prompt')} {question}: ")
        except EOFError:
            return None

    def choose(self, question: str, options: list[str]) -> int:
        """Numbered pick list; returns the chosen index (the first one by default)."""
        for n, label in enumerate(options, 1):
            print(f"    {paint(str(n), 'accent')}  {label}")
        while True:
            answer = self.ask(question, "1")
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            self.warn(f"pick a number from 1 to {len(options)}")

    def confirm(self, question: str, default: bool) -> bool:
        if self.assume_yes:
            return True
        if not self.interactive:
            return default
        hint = "Y/n" if default else "y/N"
        try:
            answer = input(f"  {paint('?', 'prompt')} {question} {paint(f'[{hint}]', 'dim')} ")
        except EOFError:
            return default
        answer = answer.strip().lower()
        return default if not answer else answer in ("y", "yes")


__all__ = ["Progress", "Ui", "WizardError", "format_elapsed", "paint"]
