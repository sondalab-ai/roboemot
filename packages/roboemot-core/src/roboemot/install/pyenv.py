"""The robot's Python environment, driven over SSH with pip (SPEC 4A, Packaging).

Every pip call passes ``--no-cache-dir``, so nothing lands in the robot user's pip cache,
including during dry runs. Installs are constrained to the versions already present (sent on
stdin, no file written), so other apps' packages are not upgraded unless the user agrees.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass

from .record import Versions, canonical, versions
from .remote import Remote
from .ui import Ui, WizardError, last_line

PIP_FLAGS = "--disable-pip-version-check --no-cache-dir --no-input"
LONG_FLAGS = "--progress-bar off"

# Every installed distribution -> canonical names of its unconditional requirements.
REQUIRES = r"""
import json, re
from importlib.metadata import distributions
canon = lambda s: re.sub(r"[-_.]+", "-", s).lower()
out = {}
for d in distributions():
    name = d.metadata["Name"]
    if not name:
        continue
    reqs = [r for r in (d.requires or []) if not re.search(r"extra\s*==", r)]
    out[canon(name)] = sorted({canon(re.split(r"[\s;<>=!~\[(@]", r, maxsplit=1)[0]) for r in reqs})
print(json.dumps(out))
"""

# Where each top-level module still resolves from (leftovers after an uninstall).
FIND_MODULES = r"""
import json, sys, importlib.util
found = {}
for m in sys.argv[1:]:
    spec = importlib.util.find_spec(m)
    if spec is not None:
        found[m] = list(spec.submodule_search_locations or [spec.origin])
print(json.dumps(found))
"""

SITE_PACKAGES = "import sysconfig; print(sysconfig.get_paths()['purelib'])"


def pip_progress_line(line: str) -> str | None:
    """The pip output line worth showing as progress, or None for noise (blank lines,
    ``Requirement already satisfied``, ``[notice]`` and our own exit-code marker)."""
    text = line.strip()
    if not text or text.startswith(("Requirement already satisfied", "[notice]", "__rc=")):
        return None
    return text


@dataclass
class PipResult:
    ok: bool
    output: str


@dataclass
class RemotePython:
    remote: Remote
    python: str
    ui: Ui | None = None

    def _pip(
        self,
        args: str,
        stdin: str | None = None,
        check: bool = True,
        announce: str | None = None,
        slow: bool = True,
        report: bool = True,
    ) -> PipResult:
        """Run pip. With ``announce`` (and a Ui) the output streams as live progress; without,
        it is collected and the call stays silent."""
        cmd = f"{shlex.quote(self.python)} -m pip {args}"
        if stdin is not None:
            cmd = f"printf %s {shlex.quote(stdin)} | {cmd}"
        result = self._shell(cmd, announce, pip_progress_line, slow, report)
        if check and not result.ok:
            raise WizardError(
                f"pip {args.split()[0]} failed on the robot",
                last_line(result.output),
                output=result.output,
            )
        return result

    def _shell(
        self,
        cmd: str,
        announce: str | None,
        on_line: Callable[[str], str | None] | None,
        slow: bool,
        report: bool,
    ) -> PipResult:
        """Run ``cmd`` with stderr merged and its exit code read from a marker line, so a
        failure is a result and not an exception. With ``announce`` (and a Ui) the output
        streams as live progress."""
        full = f'{cmd} 2>&1; echo "__rc=$?"'
        if announce and self.ui is not None:
            with self.ui.progress(announce, on_line, report=report, slow=slow) as progress:
                out = self.remote.stream(full, progress, check=True)
                progress.failed = not _rc_ok(out)
        else:
            out = self.remote.run(full, check=True)
        body, _, rc = out.rpartition("__rc=")
        return PipResult(rc.strip() == "0", body.strip())

    def cli(self, args: str, announce: str) -> PipResult:
        """Run ``python -m roboemot.cli <args>`` on the robot, streaming its output as live
        progress. A non-zero exit is returned, not raised."""
        cmd = f"{shlex.quote(self.python)} -m roboemot.cli {args}"
        return self._shell(
            cmd, announce, lambda line: None if line.startswith("__rc=") else line, False, True
        )

    def exists(self) -> bool:
        return self.remote.run(f"test -x {shlex.quote(self.python)} && echo yes || echo no") == (
            "yes"
        )

    def py(self, code: str, *args: str) -> str:
        quoted = " ".join(shlex.quote(a) for a in args)
        return self.remote.run(f"{shlex.quote(self.python)} -c {shlex.quote(code)} {quoted}")

    def installed(self) -> Versions:
        return versions(json.loads(self._pip(f"list --format=json {PIP_FLAGS}").output))

    def requires(self) -> dict[str, list[str]]:
        return json.loads(self.py(REQUIRES))

    def site_packages(self) -> str:
        return self.py(SITE_PACKAGES)

    @staticmethod
    def constraints(installed: Versions, exclude: set[str]) -> str:
        """Pin what is already there, except our own distributions."""
        return "".join(f"{n}=={v}\n" for n, v in sorted(installed.items()) if n not in exclude)

    def plan_install(self, targets: list[str], constraints: str | None) -> PipResult:
        """Resolve without installing; the output is pip's JSON report on success."""
        c = "-c /dev/stdin " if constraints is not None else ""
        args = f"install --dry-run --quiet --report - {PIP_FLAGS} {c}{_join(targets)}"
        return self._pip(args, stdin=constraints, check=False)

    def install(
        self, targets: list[str], constraints: str | None, report: bool = True
    ) -> PipResult:
        c = "-c /dev/stdin " if constraints is not None else ""
        return self._pip(
            f"install {LONG_FLAGS} {PIP_FLAGS} {c}{_join(targets)}",
            stdin=constraints,
            check=False,
            announce="Installing Python packages",
            report=report,
        )

    def install_locked(self, lock_file: str, report: bool = True) -> PipResult:
        """Install a hash-pinned lock file: every artifact must match, wheels only, no cache."""
        args = f"install {LONG_FLAGS} {PIP_FLAGS} --require-hashes --only-binary=:all: -r"
        return self._pip(
            f"{args} {shlex.quote(lock_file)}",
            check=False,
            announce="Installing Python packages",
            report=report,
        )

    def reinstall_own(self, files: list[str]) -> None:
        """Same version numbers on a reinstall: force our wheels in, dependencies untouched."""
        self._pip(
            f"install {LONG_FLAGS} --no-deps --force-reinstall {PIP_FLAGS} {_join(files)}",
            announce="Reinstalling the app wheels",
        )

    def uninstall(self, names: list[str]) -> None:
        if names:
            self._pip(
                f"uninstall --yes {LONG_FLAGS} {PIP_FLAGS} {_join(names)}",
                announce="Uninstalling packages",
                slow=False,
            )

    def install_exact(self, pins: dict[str, str]) -> None:
        """Put original versions back (from the package index: needs internet)."""
        if pins:
            reqs = [f"{n}=={v}" for n, v in sorted(pins.items())]
            self._pip(
                f"install {LONG_FLAGS} --no-deps {PIP_FLAGS} {_join(reqs)}",
                announce="Restoring package versions",
            )

    def find_modules(self, modules: list[str]) -> dict[str, list[str]]:
        return json.loads(self.py(FIND_MODULES, *modules))


def report_changes(report: dict, installed: Versions) -> tuple[Versions, dict[str, str]]:
    """From pip's ``--report``: distributions it would add, and those it would change
    (name -> "old -> new")."""
    added, changed = {}, {}
    for item in report.get("install", []):
        meta = item.get("metadata", {})
        name, version = canonical(meta.get("name", "")), meta.get("version", "")
        if name in installed:
            if installed[name] != version:
                changed[name] = f"{installed[name]} -> {version}"
        else:
            added[name] = version
    return added, changed


def _rc_ok(output: str) -> bool:
    return output.rpartition("__rc=")[2].strip() == "0"


def _join(items: list[str]) -> str:
    return " ".join(shlex.quote(i) for i in items)


__all__ = ["PIP_FLAGS", "PipResult", "RemotePython", "pip_progress_line", "report_changes"]
