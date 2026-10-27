"""Wizard output and prompts for the robot installer (SPEC 4A, Packaging).

Colours come from the Sondalab palette (``sondalab-palette``, optional: plain text without it)
and only when stdout is a terminal. Non-interactive runs take each prompt's default;
``assume_yes`` accepts every confirmation.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    import sondalab_palette as _palette
    from sondalab_palette import term as _term
except ImportError:  # pragma: no cover - plain output
    _palette = _term = None

TAGLINE = "Social and emotional behaviour for robots."
"""The header's description line (the README's first sentence)."""


class WizardError(Exception):
    """A step failed; the message says what to do about it."""


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
    def __init__(self, interactive: bool, assume_yes: bool) -> None:
        self.interactive = interactive
        self.assume_yes = assume_yes
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


__all__ = ["HeaderParser", "Ui", "WizardError", "header", "paint", "print_header"]
