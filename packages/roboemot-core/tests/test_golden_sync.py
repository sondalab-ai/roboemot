"""The C++ golden vectors must match the current Python reference (SPEC 10)."""

import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
# libm rounds sin/cos differently across platforms (last ulp); a stale file differs far more.
REL_TOL = 1e-12
ABS_TOL = 1e-15  # for values at or near zero


def _same_token(a: str, b: str) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL)
    except ValueError:
        return a == b


def golden_mismatches(generated: str, committed: str) -> list[int]:
    """1-based line numbers whose tokens differ beyond REL_TOL (or whose shape differs)."""
    gen, ref = generated.splitlines(), committed.splitlines()
    bad = [
        n
        for n, (g, r) in enumerate(zip(gen, ref, strict=False), 1)
        if len(g.split()) != len(r.split()) or not all(map(_same_token, g.split(), r.split()))
    ]
    if len(gen) != len(ref):
        bad.append(min(len(gen), len(ref)) + 1)
    return bad


def test_golden_vectors_up_to_date(tmp_path):
    spec = importlib.util.spec_from_file_location("gen_golden", REPO / "scripts" / "gen_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "golden.txt"
    mod.main(out)
    committed = (REPO / "rt" / "tests" / "golden_motion.txt").read_text()
    bad = golden_mismatches(out.read_text(), committed)
    assert not bad, f"lines {bad[:5]} differ: run scripts/gen_golden.py and commit the result"


@pytest.mark.parametrize(
    ("generated", "ok"),
    [
        ("cmd idle 5 0 0.77 -1 1 head.pitch 0.013922704772547137", True),  # last-ulp libm drift
        ("cmd idle 5 0 0.77 -1 1 head.pitch 0.0139227047726", False),  # stale value
        ("cmd idle 5 0 0.77 -1 1 head.yaw 0.013922704772547135", False),  # renamed dof
        ("cmd idle 5 0 0.77 -1 1 head.pitch", False),  # missing token
    ],
)
def test_golden_comparison_tolerates_only_ulp_drift(generated, ok):
    committed = "cmd idle 5 0 0.77 -1 1 head.pitch 0.013922704772547135"
    assert (golden_mismatches(generated, committed) == []) is ok
