"""Hash-pinned requirements for the Piper virtualenv (SPEC 4A, Installer).

``piper-tts`` and every transitive dependency are locked with sha256 hashes, one lock file per
target (CPU architecture and Python minor version), shipped as package data in ``locks/``.
``scripts/lock_piper.py`` regenerates them from the version in ``roboemot.install``.
"""

from __future__ import annotations

from pathlib import Path

from .ui import WizardError

LOCK_DIR = Path(__file__).parent / "locks"

# (platform.machine(), Python "major.minor") -> uv --python-platform. The robot is aarch64; the CI
# `models` job is x86_64. 3.11 is kept on the robot side in case its image ships that Python.
TARGETS: dict[tuple[str, str], str] = {
    ("aarch64", "3.12"): "aarch64-unknown-linux-gnu",
    ("aarch64", "3.11"): "aarch64-unknown-linux-gnu",
    ("x86_64", "3.12"): "x86_64-unknown-linux-gnu",
}


def lock_name(arch: str, python: str) -> str:
    """File name of the lock for one target, e.g. ``piper-aarch64-py312.txt``."""
    return f"piper-{arch}-py{python.replace('.', '')}.txt"


def lock_path(arch: str, python: str) -> Path:
    """The lock for this target; a target without a lock is an error, never an unhashed install."""
    if (arch, python) not in TARGETS:
        known = ", ".join(f"{a} Python {p}" for a, p in sorted(TARGETS))
        raise WizardError(
            f"no hash-pinned Piper lock for {arch} Python {python} (locked: {known}); "
            "add the target to roboemot.install.piper_lock.TARGETS and run scripts/lock_piper.py"
        )
    return LOCK_DIR / lock_name(arch, python)
