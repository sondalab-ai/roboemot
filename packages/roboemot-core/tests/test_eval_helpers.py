"""SPEC 14: the statistics helpers of the dev-only perception report (scripts/eval/common.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[3] / "scripts" / "eval" / "common.py"
spec = importlib.util.spec_from_file_location("eval_common", COMMON)
assert spec is not None and spec.loader is not None
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def test_wilson_interval_brackets_the_rate_and_stays_in_range():
    lo, hi = common.wilson(5, 100)
    assert lo < 0.05 < hi
    assert lo >= 0.0 and hi <= 1.0
    assert common.wilson(0, 0) == (0.0, 0.0)
    lo0, hi0 = common.wilson(0, 3000)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0.0 < hi0 < 0.002  # about 3/3000
    lo1, hi1 = common.wilson(10, 10)
    assert hi1 == pytest.approx(1.0) and lo1 > 0.6


def test_rate_reports_counts_and_none_for_empty():
    r = common.rate(1, 4)
    assert (r["k"], r["n"], r["rate"]) == (1, 4, 0.25)
    assert r["ci95"][0] < 0.25 < r["ci95"][1]
    assert common.rate(0, 0)["rate"] is None


def test_percentile_is_nearest_rank():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert common.percentile(values, 0.0) == 1.0
    assert common.percentile(values, 0.5) == 3.0
    assert common.percentile(values, 1.0) == 5.0
    assert common.percentile([7.0], 0.95) == 7.0


def test_write_result_refuses_paths_and_image_names(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    path = common.write_result("ok", {"n": 3, "rate": 0.5})
    assert path.read_text().strip().startswith("{")
    for bad in ("/Users/someone/data", str(Path.home()) + "/x", "Name_0001.jpg", "face.png"):
        with pytest.raises(ValueError):
            common.write_result("bad", {"note": bad})
    assert not (tmp_path / "bad.json").exists()
