"""SSH and scp to the robot through one multiplexed connection (SPEC 4A, Packaging).

The master connection is opened interactively once, so ssh asks for the password at most
once per run. The control socket lives in a short directory under ``/tmp``: its path must stay
under 104 bytes, and the macOS ``$TMPDIR`` is too long.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .ui import WizardError

CONTROL_PERSIST_S = 120
CONNECT_TIMEOUT_S = 10


@dataclass
class Remote:
    user: str
    host: str
    control_dir: Path

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def _opts(self) -> list[str]:
        return [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.control_dir}/%C",
            "-o", f"ControlPersist={CONTROL_PERSIST_S}",
            "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}",
        ]  # fmt: skip

    def run(self, command: str, check: bool = True) -> str:
        """Run ``command`` on the robot and return its stdout."""
        proc = subprocess.run(
            ["ssh", *self._opts(), self.target, command], capture_output=True, text=True
        )
        if check and proc.returncode != 0:
            raise WizardError(f"on robot: {command}\n{proc.stderr.strip() or proc.stdout.strip()}")
        return proc.stdout.strip()

    def connect(self) -> None:
        """Open the master connection interactively, so a password prompt reaches the user."""
        proc = subprocess.run(
            ["ssh", *self._opts(), self.target, "true"], stdout=subprocess.DEVNULL
        )
        if proc.returncode != 0:
            raise WizardError(
                f"ssh {self.target} failed. Check host and user. On a first connection run "
                f"`ssh {self.target}` once to accept the host key; "
                f"`ssh-copy-id {self.target}` then avoids the password."
            )

    def copy(self, files: list[Path], dest: str) -> None:
        proc = subprocess.run(
            ["scp", "-q", *self._opts(), *map(str, files), f"{self.target}:{dest}/"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise WizardError(f"scp failed: {proc.stderr.strip()}")

    def close(self) -> None:
        subprocess.run(
            ["ssh", *self._opts(), "-O", "exit", self.target], capture_output=True, check=False
        )


__all__ = ["Remote"]
