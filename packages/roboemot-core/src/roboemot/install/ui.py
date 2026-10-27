"""Wizard output and prompts for the robot installer (SPEC 4A, Packaging).

Follows the Sondalab Term spec (sondalab-ui docs/tui-patterns.md) through ``sondalab_palette``:

- Results are step lines on stdout: glyph, label padded to the label column, detail. Work a dry
  run only describes is ``pending``, never ``done``. The run ends with ``Summary``.
- Prompts, spinners and running lines go to stderr. An unrecognised answer asks again at most
  three times. Without a terminal on stdin nothing is asked and no default is taken: the run
  stops with an error naming the flag that answers the question (``--yes`` answers every
  confirmation).
- Errors are the three-line block (what, why, ``Try:``) on stderr; raw tool output only with
  ``--debug``.

The palette import is optional: the robot imports this module (through
``roboemot_reachymini.settings``) and has no palette. The installer itself requires it.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import TextIO

try:
    from sondalab_palette import LAYOUT, SPINNER
    from sondalab_palette import term as _term
except ImportError:  # pragma: no cover - the robot, which never runs the installer
    _term = None
    LAYOUT = SPINNER = {}

NAME = "roboemot"
TAGLINE = "Social and emotional behaviour for robots."
"""The header's description line (the README's first sentence)."""

COMMAND = "python scripts/install_on_robot.py"
"""How the installer is run; the ``Try:`` lines of its errors start with it."""

STATE_WORDS = {
    "done": "done",
    "fixed": "fixed",
    "failed": "failed",
    "warn": "warning",
    "active": "running",
    "pending": "pending",
    "skipped": "skipped",
}
"""Accessible mode writes each state as a word next to its glyph."""

if _term is not None:
    _ErrorBase = _term.TermError
else:  # pragma: no cover - the robot, which never runs the installer

    class _ErrorBase(Exception):
        def __init__(self, what: str, why: str = "", fix: str = "") -> None:
            super().__init__(what, why, fix)
            self.what, self.why, self.fix = what, why, fix

        def __str__(self) -> str:
            return f"{self.what}: {self.why}" if self.why else self.what


class WizardError(_ErrorBase):
    """A step failed: ``what`` failed, ``why`` (the cause), ``fix`` (a command to try).

    ``output`` is the raw tool output (pip, ssh, the wheel build): printed only under
    ``--debug``, never in the three-line error block.
    """

    def __init__(self, what: str, why: str = "", fix: str = "", *, output: str = "") -> None:
        super().__init__(what, why, fix)
        self.output = output


def last_line(text: str) -> str:
    """The last non-empty line of tool output: the usual one-line cause of a failure."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def app_version() -> str:
    """The installed ``roboemot-core`` version, ``unknown`` in a never-installed source tree."""
    try:
        return version("roboemot-core")
    except PackageNotFoundError:
        return "unknown"


def header() -> str:
    """The Sondalab header for stdout; "" without the palette, off a terminal and in accessible
    mode (the palette decides since 0.9.0)."""
    return "" if _term is None else str(_term.header(NAME, app_version(), TAGLINE))


def print_header() -> None:
    """Print :func:`header` (nothing when it is empty)."""
    print(header(), end="", flush=True)


class HeaderParser(argparse.ArgumentParser):
    """``argparse``'s parser with the header before the ``--help`` text."""

    def print_help(self, file=None) -> None:
        if file is None or file is sys.stdout:
            print_header()
        super().print_help(file)


def _isatty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


class Ui:
    """Installer output and prompts.

    ``interactive`` is whether stdin is a terminal a person can answer on; ``assume_yes``
    (``--yes``) answers every confirmation. ``read`` returns one typed line (``""`` at end of
    input); None reads stdin. ``command`` starts the ``Try:`` line of an error. ``vt`` False
    (Windows without VT processing) means plain text: no colour, no motion, ASCII glyphs.
    ``debug`` prints raw tool output; ``debug_flag`` is the flag an error suggests for it.
    Streams are looked up at write time, so redirecting ``sys.stdout`` works.
    """

    def __init__(
        self,
        interactive: bool,
        assume_yes: bool,
        *,
        read: Callable[[], str] | None = None,
        command: str = COMMAND,
        vt: bool = True,
        debug: bool = False,
        debug_flag: str | None = "--debug",
    ) -> None:
        self.interactive = interactive
        self.assume_yes = assume_yes
        self.read = read
        self.command = command
        self.vt = vt
        self.debug = debug
        self.debug_flag = debug_flag
        self.step_no = 0
        self.steps = 0
        self.counts = {"ok": 0, "warn": 0, "failed": 0}

    # ------------------------------------------------------------ rendering

    def mode(self, stream: TextIO) -> str:
        """The colour mode for ``stream`` (palette ``color_mode``), "none" without VT."""
        return _term.color_mode(isatty=_isatty(stream)) if self.vt else "none"

    def ascii(self) -> bool:
        return not self.vt or _term.ascii_mode()

    def paint(self, text: str, role: str, stream: TextIO | None = None) -> str:
        return _term.paint(text, role, self.mode(stream or sys.stdout))

    def _step_text(self, name: str, label: str, detail: str, stream: TextIO) -> str:
        """A step line; a detail too long for the width wraps under the detail column."""
        ascii, mode = self.ascii(), self.mode(stream)
        col = LAYOUT["indent"] + 2 + LAYOUT["labelWidth"]
        more: list[str] = []
        if detail and col + len(detail) > _term.width():
            first, *more = _term.wrap(detail, indent=col).split("\n")
            detail = first.lstrip()
        line = _term.step_line(name, label, detail, ascii, mode=mode)
        if _term.accessible():
            head = LAYOUT["indent"] + 2
            line = (
                _term.step_line(name, "", ascii=ascii, mode=mode)
                + STATE_WORDS.get(name, "running")
                + " "
                + _term.step_line(name, label, detail, ascii)[head:]
            )
        return "\n".join([line, *more])

    def _emit(self, name: str, label: str, detail: str = "", count: str | None = None) -> None:
        print(self._step_text(name, label, detail, sys.stdout), flush=True)
        if count:
            self.counts[count] += 1

    def _cont(self, text: str) -> str:
        return " " * (LAYOUT["indent"] + 2) + text

    # ------------------------------------------------------------ results (stdout)

    def title(self, text: str, note: str = "") -> None:
        """The run's first line: what runs, and a note (the dry run) in ``warn``."""
        line = f"{self.paint('RoboEmot', 'accent')} {text}"
        print(line + (f"  {self.paint(note, 'warn')}" if note else ""), flush=True)

    def step(self, name: str) -> None:
        """A numbered section heading over the step lines that follow."""
        self.step_no += 1
        heading = _term.heading(f"{self.step_no}/{self.steps} {name}", mode=self.mode(sys.stdout))
        print(f"\n{heading}", flush=True)

    def ok(self, label: str, detail: str = "") -> None:
        """A finished step (``done``): what changed or was checked."""
        self._emit("done", label, detail, "ok")

    def warn(self, label: str, detail: str = "") -> None:
        self._emit("warn", label, detail, "warn")

    def fail(self, label: str, detail: str = "") -> None:
        self._emit("failed", label, detail, "failed")

    def would(self, label: str, detail: str = "") -> None:
        """Work a dry run only describes (``pending``): never a success glyph."""
        self._emit("pending", label, detail)

    def change(self, dry_run: bool, todo: str, done: str, detail: str = "") -> None:
        """``todo`` as pending in a dry run, else ``done`` as a finished step."""
        if dry_run:
            self.would(todo, detail)
        else:
            self.ok(done, detail)

    def info(self, text: str) -> None:
        """Prose under a step, in ``dim``, wrapped from the label column."""
        col = LAYOUT["indent"] + 2
        lines = _term.wrap(text, indent=col).split("\n")
        print("\n".join(self._cont(self.paint(ln[col:], "dim")) for ln in lines), flush=True)

    def kv(self, key: str, value: str) -> None:
        """A fact, not a check: key in the label column, no glyph."""
        print(_term.kv_line(key, value), flush=True)

    def hint(self, label: str, command: str) -> None:
        """A command under a step (``label:`` in ``dim``), never wrapped: it is copied."""
        print(self._cont(f"{self.paint(label + ':', 'dim')} {command}"), flush=True)

    def next(self, command: str) -> None:
        print(_term.next_line(command, mode=self.mode(sys.stdout)), flush=True)

    def summary(self) -> None:
        """One blank line, then the four counts."""
        c = self.counts
        line = _term.summary(
            c["ok"], 0, c["warn"], c["failed"], mode=self.mode(sys.stdout), ascii=self.ascii()
        )
        print(f"\n{line}", flush=True)

    # ------------------------------------------------------------ stderr

    def error(self, error: Exception) -> None:
        """The three-line error block on stderr; raw output above it only with ``--debug``."""
        what = getattr(error, "what", str(error))
        why, fix = getattr(error, "why", ""), getattr(error, "fix", "")
        output = getattr(error, "output", "")
        if output and self.debug:
            print(output.rstrip(), file=sys.stderr)
        elif output and not fix and self.debug_flag:
            fix = f"{self.command} {self.debug_flag}"
        block = _term.error_block(what, why, fix, ascii=self.ascii(), mode=self.mode(sys.stderr))
        print(block, file=sys.stderr, flush=True)

    def active(self, label: str, detail: str = "") -> None:
        """A running line on stderr, at once: for a step that may prompt (ssh's password)."""
        print(self._step_text("active", label, detail, sys.stderr), file=sys.stderr, flush=True)

    @contextmanager
    def running(self, label: str) -> Iterator[None]:
        """A step that waits: after ``SPINNER["delayMs"]`` a spinner on stderr (a static
        running line when motion is off); cleared on exit. The caller prints the result."""
        stream = sys.stderr
        motion = self.vt and _term.motion_enabled(isatty=_isatty(stream))
        spinner = _Spinner(self, label, stream, motion)
        spinner.start()
        try:
            yield
        finally:
            spinner.stop()

    # ------------------------------------------------------------ prompts (stderr)

    def require(self, flag: str, given: bool, what: str = "") -> None:
        """Stop with :class:`MissingFlag` when ``flag`` was not given and nobody can be asked.

        Runs before the robot is touched, so an unattended run fails up front instead of
        half way through. A terminal on stdin means the question will be asked later.
        """
        _term.needs_prompt(
            flag, given=given, what=what, command=self.command, isatty=self.interactive
        )

    def _readline(self) -> str:
        try:
            return (self.read or sys.stdin.readline)()
        except EOFError:
            return ""

    def ask(self, question: str, default: str, *, flag: str) -> str:
        """A free-text answer on stderr; Enter takes ``default``, which the prompt shows.

        Without a terminal it raises :class:`MissingFlag` naming ``flag``; end of input is
        not an answer (:class:`NoValidAnswer`).
        """
        self.require(flag, False)
        q = self.paint("?", "prompt", sys.stderr)
        sys.stderr.write(f"{q} {question} {self.paint(f'[{default}]', 'dim', sys.stderr)} ")
        sys.stderr.flush()
        line = self._readline()
        if line == "":
            sys.stderr.write("\n")
            raise _term.NoValidAnswer(question, "input ended before an answer")
        return line.strip() or default

    def ask_secret(self, question: str) -> str | None:
        """A hidden answer from the terminal; None when not interactive."""
        if not self.interactive:
            return None
        try:
            return getpass.getpass(f"{self.paint('?', 'prompt', sys.stderr)} {question}: ")
        except EOFError:
            return None

    def choose(self, question: str, options: Sequence[str], *, flag: str) -> int:
        """Numbered pick list on stderr; the 0-based index typed. No default: three
        unrecognised answers or end of input raise :class:`NoValidAnswer`."""
        self.require(flag, False)
        return _term.choose(question, options, read=self._readline, mode=self.mode(sys.stderr))

    def confirm(self, question: str, default: bool, *, flag: str = "--yes") -> bool:
        """``? question [y/N]``; ``--yes`` answers yes. Without a terminal it raises
        :class:`MissingFlag` naming ``flag``. Default no for anything that changes the robot."""
        if self.assume_yes:
            return True
        self.require(flag, False)
        return _term.confirm(
            question, default=default, read=self._readline, mode=self.mode(sys.stderr)
        )


class _Spinner:
    """The thread behind :meth:`Ui.running`: nothing for the delay, then either the burst
    spinner redrawn in place every interval or, with motion off, the running line once."""

    def __init__(self, ui: Ui, label: str, stream: TextIO, motion: bool) -> None:
        self.ui, self.label, self.stream, self.motion = ui, label, stream, motion
        self.drawn = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        if self.drawn and self.motion:
            self.stream.write("\r\x1b[J")
            self.stream.flush()

    def _run(self) -> None:
        if self._stop.wait(SPINNER["delayMs"] / 1000):
            return
        self.drawn = True
        if not self.motion:
            self.ui.active(self.label)
            return
        ascii, tick = self.ui.ascii(), 0
        while True:
            frame = _term.spinner_frame(tick, ascii=ascii)
            line = _term.step_line(frame, self.label, "", ascii, mode=self.ui.mode(self.stream))
            self.stream.write(f"\x1b[?2026h\r\x1b[K{line}\x1b[?2026l")
            self.stream.flush()
            tick += 1
            if self._stop.wait(SPINNER["intervalMs"] / 1000):
                return


__all__ = [
    "COMMAND",
    "HeaderParser",
    "Ui",
    "WizardError",
    "app_version",
    "header",
    "last_line",
    "print_header",
]
