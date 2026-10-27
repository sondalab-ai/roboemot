"""SSH and scp to the robot through one multiplexed connection (SPEC 4A, Packaging).

The master connection is opened interactively once, so ssh asks for the password at most
once per run. The control socket lives in a short directory under ``/tmp``: its path must stay
under 104 bytes, and the macOS ``$TMPDIR`` is too long.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
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
            self._raise(command, proc.returncode, proc.stderr.strip() or proc.stdout.strip(), stdin)
        return proc.stdout.strip()

    def _raise(self, command: str, rc: int, output: str, stdin: str | None) -> None:
        """Raise the :class:`WizardError` for ``command`` that exited ``rc``: a lost connection,
        or a failed command whose ``why`` hides a value given on stdin; the command and its
        output stay in ``.output`` (``--debug`` only)."""
        raw = f"$ {command}\n{output}"
        if rc == LOST_CONNECTION and self.connected:
            raise WizardError(
                f"Lost the SSH connection to {self.target}",
                f"ssh exited {LOST_CONNECTION}: {last_line(output) or 'no output'}",
                output=raw,
            )
        if stdin is not None:
            why = f"a command given a value on stdin exited {rc}"
        else:
            shown = command if len(command) <= 60 else command[:59] + "…"
            why = f"`{shown}` exited {rc}: {last_line(output) or 'no output'}"
        raise WizardError("Command failed on the robot", why, output=raw)

    def stream(self, command: str, on_line: Callable[[str], None], check: bool = True) -> str:
        """Like ``run``, but hand each output line to ``on_line`` as it arrives.

        stderr is merged into stdout. Returns the whole output (stripped); a non-zero exit
        raises with that output, as ``run`` does. Whatever interrupts the read (``on_line``
        raising, Ctrl-C) kills the ssh process before the error propagates.
        """
        proc = subprocess.Popen(
            ["ssh", *self._opts(), self.target, command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        lines: list[str] = []
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                lines.append(line)
                on_line(line)
            proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        output = "\n".join(lines).strip()
        if check and proc.returncode != 0:
            self._raise(command, proc.returncode, output, None)
        return output

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
