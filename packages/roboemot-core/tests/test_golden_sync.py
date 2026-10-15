"""The C++ golden vectors must match the current Python reference (SPEC 10)."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_golden_vectors_up_to_date(tmp_path):
    spec = importlib.util.spec_from_file_location("gen_golden", REPO / "scripts" / "gen_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "golden.txt"
    mod.main(out)
    committed = (REPO / "rt" / "tests" / "golden_motion.txt").read_text()
    assert out.read_text() == committed, "run scripts/gen_golden.py and commit the result"
