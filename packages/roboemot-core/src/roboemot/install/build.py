"""Local wheel build for the robot installer (SPEC 4A, Packaging): runs on the PC."""

from __future__ import annotations

import posixpath
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from .record import canonical
from .ui import WizardError

CORE_WHEEL_PREFIX = "roboemot_core-"  # the wheel that defines the extras


def _meta(package_dir: Path) -> dict:
    return tomllib.loads((package_dir / "pyproject.toml").read_text())["project"]


def build_wheels(package_dirs: list[Path], out: Path) -> list[Path]:
    """Build one wheel per package (no dependencies), in the given order."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(out),
         *map(str, package_dirs)],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if proc.returncode != 0:
        raise WizardError(f"wheel build failed:\n{proc.stderr.strip()}")
    names = [_meta(d)["name"].replace("-", "_") for d in package_dirs]
    return [next(out.glob(f"{n}-*.whl")) for n in names]


def distribution_names(package_dirs: list[Path]) -> list[str]:
    return [canonical(_meta(d)["name"]) for d in package_dirs]


def third_party_requirements(package_dirs: list[Path], extras: tuple[str, ...] = ()) -> list[str]:
    """Runtime dependencies of the packages (plus the named ``extras`` they define), minus the
    packages themselves."""
    own = set(distribution_names(package_dirs))
    reqs = set()
    for d in package_dirs:
        meta = _meta(d)
        optional = meta.get("optional-dependencies", {})
        wanted = [*meta.get("dependencies", []), *(r for e in extras for r in optional.get(e, []))]
        for r in wanted:
            if canonical(re.split(r"[\s;<>=!~\[(@]", r, maxsplit=1)[0]) not in own:
                reqs.add(r)
    return sorted(reqs)


def with_extras(files: list[str], extras: tuple[str, ...]) -> list[str]:
    """``files`` with the core wheel asked for with ``extras`` (``path.whl[voice,vision]``)."""
    if not extras:
        return files
    suffix = f"[{','.join(extras)}]"
    return [f + suffix if posixpath.basename(f).startswith(CORE_WHEEL_PREFIX) else f for f in files]


def min_python(package_dirs: list[Path]) -> tuple[int, int]:
    """The strictest ``requires-python >=X.Y`` among the packages."""
    found = [(3, 0)]
    for d in package_dirs:
        m = re.search(r">=\s*(\d+)\.(\d+)", _meta(d).get("requires-python", ""))
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    return max(found)


__all__ = [
    "build_wheels",
    "distribution_names",
    "min_python",
    "third_party_requirements",
    "with_extras",
]
