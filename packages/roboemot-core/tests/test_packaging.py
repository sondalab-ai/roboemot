"""The core wheel carries every non-Python file core reads at runtime (SPEC 4A, 11A)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

CORE = Path(__file__).resolve().parents[1]
SRC = CORE / "src"
RUNTIME_DATA = (
    "roboemot/config/moved_settings.yaml",
    "roboemot/vision/models.yaml",
    "roboemot/voice/models.yaml",
    "roboemot/voice/piper_worker.py",
    "roboemot/webui/static/index.html",
    "roboemot/install/locks/piper-x86_64-py312.txt",
)


def test_the_wheel_contains_the_package_data_core_reads(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", str(CORE), "-w", str(tmp_path)],
        check=True,
    )
    (wheel,) = tmp_path.glob("roboemot_core-*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert not [p for p in RUNTIME_DATA if p not in names], "missing from the wheel"
    on_disk = {
        p.relative_to(SRC).as_posix()
        for p in (SRC / "roboemot").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    assert not sorted(on_disk - names), "files under src/roboemot that the wheel leaves out"
