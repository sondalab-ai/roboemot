"""Wizard output and prompts for the robot installer (SPEC 4A, Packaging).

Prompts follow the Sondalab Term spec (sondalab-ui docs/tui-patterns.md): they go to stderr,
an unrecognised answer asks again at most three times, and without a terminal on stdin nothing is
asked and no default is taken: the run stops with an error naming the flag that answers the
question (``--yes`` answers every confirmation). The palette is imported lazily-safe: the robot
imports this module (through ``roboemot_reachymini.settings``) and has no palette.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version

try:
    import sondalab_palette as _palette
    from sondalab_palette import term as _term
except ImportError:  # pragma: no cover - plain output
    _palette = _term = None

TAGLINE = "Social and emotional behaviour for robots."
"""The header's description line (the README's first sentence)."""

COMMAND = "python scripts/install_on_robot.py"
"""How the installer is run; the ``Try:`` lines of its errors start with it."""

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


def paint(text: str, role: str) -> str:
    """Colour ``text`` with a Sondalab role, only when stdout is a terminal."""
    return _palette.paint(text, role) if _palette and sys.stdout.isatty() else text


def header() -> str:
    """The full Sondalab header for the installer's stdout, or "" when nothing should print:
    without the palette, in accessible mode, or when stdout is not a terminal (so piped and
    captured output stays byte-identical). On a terminal it still falls back to the one-line
    lockup with ``TERM=dumb``, in ASCII mode or when the terminal is too narrow."""
    if _term is None or _term.accessible() or not sys.stdout.isatty():
        return ""
    try:
        ver = version("roboemot-core")
    except PackageNotFoundError:  # a source tree that was never installed
        ver = "unknown"
    return str(_term.header("roboemot", ver, TAGLINE, isatty=True))


def print_header() -> None:
    """Print :func:`header` (nothing when it is empty)."""
    print(header(), end="", flush=True)


class HeaderParser(argparse.ArgumentParser):
    """``argparse``'s parser with the header before the ``--help`` text."""

    def print_help(self, file=None) -> None:
        if file is None or file is sys.stdout:
            print_header()
        super().print_help(file)


class Ui:
    """Installer output and prompts.

    ``interactive`` is whether stdin is a terminal a person can answer on; ``assume_yes``
    (``--yes``) answers every confirmation. ``read`` returns one typed line (``""`` at end of
    input); None reads stdin. ``command`` starts the ``Try:`` line of a missing-flag error.
    """

    def __init__(
        self,
        interactive: bool,
        assume_yes: bool,
        *,
        read: Callable[[], str] | None = None,
        command: str = COMMAND,
    ) -> None:
        self.interactive = interactive
        self.assume_yes = assume_yes
        self.read = read
        self.command = command
        self.step_no = 0
        self.steps = 0

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
        sys.stderr.write(f"{paint('?', 'prompt')} {question} {paint(f'[{default}]', 'dim')} ")
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
            return getpass.getpass(f"  {paint('?', 'prompt')} {question}: ")
        except EOFError:
            return None

    def choose(self, question: str, options: Sequence[str], *, flag: str) -> int:
        """Numbered pick list on stderr; the 0-based index typed. No default: three
        unrecognised answers or end of input raise :class:`NoValidAnswer`."""
        self.require(flag, False)
        return _term.choose(question, options, read=self._readline)

    def confirm(self, question: str, default: bool, *, flag: str = "--yes") -> bool:
        """``? question [y/N]``; ``--yes`` answers yes. Without a terminal it raises
        :class:`MissingFlag` naming ``flag``. Default no for anything that changes the robot."""
        if self.assume_yes:
            return True
        self.require(flag, False)
        return _term.confirm(question, default=default, read=self._readline)


__all__ = ["HeaderParser", "Ui", "WizardError", "header", "paint", "print_header"]
