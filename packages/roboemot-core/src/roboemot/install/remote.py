"""SSH and scp to the robot through one multiplexed connection (SPEC 4A, Packaging).

The master connection is opened interactively once, so ssh asks for the password at most
once per run. The control socket lives in a short directory under ``/tmp``: its path must stay
under 104 bytes, and the macOS ``$TMPDIR`` is too long.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .ui import WizardError, last_line

CONTROL_PERSIST_S = 120
CONNECT_TIMEOUT_S = 10
LOST_CONNECTION = 255  # ssh's own exit code: it could not connect or authenticate


@dataclass
class Remote:
    user: str
    host: str
    control_dir: Path
    connected: bool = field(default=False, init=False)

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def _opts(self) -> list[str]:
        return [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.control_dir}/%C",
            "-o", f"ControlPersist={CONTROL_PERSIST_S}",
            "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}",
            # After the interactive login no ssh may prompt: a spinner may be drawing, and a
            # master that expired must fail with an error, not ask for a password.
            *(["-o", "BatchMode=yes"] if self.connected else []),
        ]  # fmt: skip

    def run(self, command: str, check: bool = True, stdin: str | None = None) -> str:
        """Run ``command`` on the robot and return its stdout.

        ``stdin`` is sent on the command's standard input, never on its command line: use it
        for secrets (the operator PIN). A failure of such a command gets a fixed ``why``; the
        command and the robot's stderr stay in ``.output`` (``--debug`` only).
        """
        proc = subprocess.run(
            ["ssh", *self._opts(), self.target, command],
            capture_output=True,
            text=True,
            input=stdin,
        )
        if check and proc.returncode != 0:
            output = proc.stderr.strip() or proc.stdout.strip()
            raw = f"$ {command}\n{output}"
            if proc.returncode == LOST_CONNECTION and self.connected:
                raise WizardError(
                    f"Lost the SSH connection to {self.target}",
                    f"ssh exited {LOST_CONNECTION}: {last_line(output) or 'no output'}",
                    output=raw,
                )
            if stdin is not None:
                why = f"a command given a value on stdin exited {proc.returncode}"
            else:
                shown = command if len(command) <= 60 else command[:59] + "…"
                why = f"`{shown}` exited {proc.returncode}: {last_line(output) or 'no output'}"
            raise WizardError("Command failed on the robot", why, output=raw)
        return proc.stdout.strip()

    def connect(self) -> None:
        """Open the master connection interactively, so a password prompt reaches the user."""
        proc = subprocess.run(
            ["ssh", *self._opts(), self.target, "true"], stdout=subprocess.DEVNULL
        )
        if proc.returncode != 0:
            raise WizardError(
                f"Could not connect to {self.target}",
                f"ssh exited {proc.returncode}. Check host and user; on a first connection, "
                f"accept the host key once, and ssh-copy-id {self.target} avoids the password",
                f"ssh {self.target}",
            )
        self.connected = True

    def copy(self, files: list[Path], dest: str) -> None:
        proc = subprocess.run(
            ["scp", "-q", *self._opts(), *map(str, files), f"{self.target}:{dest}/"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise WizardError(
                "Could not copy files to the robot",
                f"scp failed: {last_line(proc.stderr)}",
                output=proc.stderr,
            )

    def close(self) -> None:
        subprocess.run(
            ["ssh", *self._opts(), "-O", "exit", self.target], capture_output=True, check=False
        )


__all__ = ["Remote"]
