"""Wizard output and prompts for the robot installer (SPEC 4A, Packaging).

Colours come from the Sondalab palette (``sondalab-palette``, optional: plain text without it)
and only when stdout is a terminal. Non-interactive runs take each prompt's default;
``assume_yes`` accepts every confirmation.
"""

from __future__ import annotations

import getpass
import sys

try:
    import sondalab_palette as _palette
except ImportError:  # pragma: no cover - plain output
    _palette = None


class WizardError(Exception):
    """A step failed; the message says what to do about it."""


def paint(text: str, role: str) -> str:
    """Colour ``text`` with a Sondalab role, only when stdout is a terminal."""
    return _palette.paint(text, role) if _palette and sys.stdout.isatty() else text


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


__all__ = ["Ui", "WizardError", "paint"]
