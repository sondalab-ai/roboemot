"""SPEC 5, 13: the hand-pipeline cost bench (scripts/bench/hands_cm4.py): options, statistics,
geometry and the pinned download, without the models; the real run is opt in (``-m models``)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from roboemot.config.schema import VisionConfig

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "bench" / "hands_cm4.py"
spec = importlib.util.spec_from_file_location("bench_hands", SCRIPT)
assert spec is not None and spec.loader is not None
hands = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hands  # its dataclass looks the module up there
spec.loader.exec_module(hands)


def test_defaults_follow_the_vision_pipeline():
    a = hands.parse_args([])
    assert a.threads == VisionConfig().threads
    assert a.spinning is VisionConfig().object_spinning
    assert a.frame == (1280, 720)
    assert a.variants == ["fp32", "int8"]
    assert a.cache_dir is None


def test_options_parse_and_reject_bad_values():
    a = hands.parse_args(["--variants", "int8,fp32,int8", "--frame", "640X480", "--spinning"])
    assert a.variants == ["int8", "fp32"]
    assert a.frame == (640, 480)
    assert a.spinning is True
    for bad in (
        ["--variants", "int4"],
        ["--variants", ","],
        ["--frame", "640"],
        ["--frame", "0x480"],
        ["--runs", "0"],
        ["--warmup", "-1"],
        ["--hand-px", "800"],  # larger than the 720 px frame height
    ):
        with pytest.raises(SystemExit):
            hands.parse_args(bad)


def test_every_variant_pins_a_commit_url_size_and_digest():
    for palm, hand in hands.VARIANTS.values():
        for f in (palm, hand):
            assert f.url.startswith(f"https://github.com/opencv/opencv_zoo/raw/{hands.ZOO_COMMIT}/")
            assert f.url.endswith("/" + f.name)
            assert len(f.sha256) == 64 and int(f.sha256, 16) >= 0
            assert f.size_bytes > 0
    assert len(hands.ZOO_COMMIT) == 40


def test_nearest_rank_and_summary():
    assert hands.nearest_rank([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0
    assert hands.nearest_rank([7.0], 0.9) == 7.0
    wall = [0.010] * 9 + [0.050]
    cpu = [0.020] * 10
    s = hands.summarise(wall, cpu, [0.008] * 10)
    assert s["runs"] == 10
    assert s["wall_p50_ms"] == 10.0 and s["wall_max_ms"] == 50.0
    assert s["net_p50_ms"] == 8.0
    assert s["cpu_p50_ms"] == 20.0
    assert s["cores"] == pytest.approx(0.2 / 0.14, abs=0.01)
    assert hands.summarise([0.0], [0.0], None)["cores"] is None
    assert hands.summarise([0.01], [0.0], None)["net_p50_ms"] is None


def test_letterbox_centres_a_wide_frame():
    assert hands.letterbox(1280, 720, 192) == (192, 108, 0, 42)
    assert hands.letterbox(720, 1280, 192) == (108, 192, 42, 0)
    assert hands.rotation_side(360) == 510


def _file(content: bytes) -> object:
    return hands.ModelFile(
        "m.onnx",
        "https://example.invalid/m.onnx",
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def test_fetch_keeps_a_matching_download_and_reuses_it(tmp_path):
    body = b"onnx bytes"
    calls = []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(body)

    f = _file(body)
    path = hands.fetch(f, tmp_path, urlopen)
    assert path.read_bytes() == body and not (tmp_path / "m.onnx.part").exists()
    assert hands.fetch(f, tmp_path, urlopen) == path
    assert len(calls) == 1


def test_fetch_deletes_a_download_that_does_not_match_the_pin(tmp_path):
    f = _file(b"expected")
    with pytest.raises(hands.BenchError, match="does not match"):
        hands.fetch(f, tmp_path, lambda url, timeout: io.BytesIO(b"tampered"))
    assert list(tmp_path.iterdir()) == []


def test_fetch_reports_a_network_error(tmp_path):
    def urlopen(url, timeout):
        raise OSError("no route to host")

    with pytest.raises(hands.BenchError, match="no route to host"):
        hands.fetch(_file(b"x"), tmp_path, urlopen)
    assert list(tmp_path.iterdir()) == []


STATE = {"loadavg": [1.0, 1.0, 1.0], "cpu_mhz": [1800] * 4, "throttled": "throttled=0x0"}


def test_render_lists_each_stage_and_a_variant_that_could_not_run():
    stage = hands.summarise([0.01], [0.02], [0.009])
    report = {
        "host": {"cpu": "cpu", "machine": "aarch64", "cores_visible": 4, "python": "3.12"},
        "onnxruntime": "1.x",
        "opencv": "4.x",
        "settings": {
            "threads": 2,
            "spinning": False,
            "runs": 1,
            "frame": [1280, 720],
            "hand_px": 360,
        },
        "before": STATE,
        "after": STATE,
        "results": [
            {
                "variant": "fp32",
                "inputs": {"palm": 192, "landmark": 224},
                "stages": {"palm": stage, "landmark": stage, "pass": stage},
            },
            {"variant": "int8", "error": "boom"},
        ],
    }
    text = hands.render(report)
    assert "| fp32 | palm | 192x192 | 10.0 |" in text
    assert "| fp32 | pass |  |" in text
    assert "| int8 | not run: boom |" in text
    assert "before: loadavg=[1.0, 1.0, 1.0] cpu_mhz=[1800, 1800, 1800, 1800]" in text


def test_redirects_are_followed_only_to_https():
    handler = hands.HttpsOnlyRedirect()
    req = urllib.request.Request("https://github.com/x.onnx")
    ok = handler.redirect_request(req, None, 302, "Found", {}, "https://media.example/x.onnx")
    assert ok is not None and ok.full_url == "https://media.example/x.onnx"
    with pytest.raises(urllib.error.HTTPError, match="non-https"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://media.example/x.onnx")


def test_fetch_reports_a_broken_http_response_in_one_line(tmp_path):
    import http.client

    def urlopen(url, timeout):
        raise http.client.IncompleteRead(b"partial")

    with pytest.raises(hands.BenchError, match="download of m.onnx failed") as e:
        hands.fetch(_file(b"x"), tmp_path, urlopen)
    assert "\n" not in str(e.value)
    assert list(tmp_path.iterdir()) == []


def test_cpu_mhz_reads_cpufreq_in_core_order(tmp_path):
    for core, khz in ((10, 600000), (0, 1800000), (2, 1500000)):
        f = tmp_path / f"cpu{core}" / "cpufreq" / "scaling_cur_freq"
        f.parent.mkdir(parents=True)
        f.write_text(f"{khz}\n")
    assert hands.cpu_mhz(tmp_path) == [1800, 1500, 600]
    assert hands.cpu_mhz(tmp_path / "missing") is None


def _report(results):
    stage = hands.summarise([0.01], [0.02], [0.009])
    return {
        "host": {"cpu": "cpu", "machine": "aarch64", "cores_visible": 4, "python": "3.12"},
        "onnxruntime": "1.x",
        "opencv": "4.x",
        "settings": {
            "threads": 2,
            "spinning": False,
            "runs": 1,
            "frame": [1280, 720],
            "hand_px": 360,
        },
        "before": STATE,
        "after": STATE,
        "results": [
            r
            if "error" in r
            else {
                **r,
                "inputs": {"palm": 192, "landmark": 224},
                "stages": {"palm": stage, "landmark": stage, "pass": stage},
            }
            for r in results
        ],
    }


@pytest.fixture
def temp_folder(tmp_path, monkeypatch):
    """The folder main() takes as its temporary download folder."""
    folder = tmp_path / "roboemot-hands-test"

    def mkdtemp(prefix):
        folder.mkdir()
        return str(folder)

    monkeypatch.setattr(hands.tempfile, "mkdtemp", mkdtemp)
    return folder


def _run_writes(folder, outcome):
    def run(args, path):
        assert path == folder
        (path / "model.onnx").write_bytes(b"x")
        return outcome()

    return run


def test_main_removes_the_temp_folder_after_a_run(temp_folder, monkeypatch, capsys):
    monkeypatch.setattr(
        hands, "run", _run_writes(temp_folder, lambda: _report([{"variant": "fp32"}]))
    )
    assert hands.main([]) == 0
    assert not temp_folder.exists()
    assert "| fp32 | pass |" in capsys.readouterr().out


def test_main_fails_when_no_variant_ran(temp_folder, monkeypatch, capsys):
    report = _report([{"variant": "fp32", "error": "boom"}, {"variant": "int8", "error": "boom"}])
    monkeypatch.setattr(hands, "run", _run_writes(temp_folder, lambda: report))
    assert hands.main([]) == 1
    assert "no variant ran" in capsys.readouterr().err
    assert not temp_folder.exists()


def test_main_removes_the_temp_folder_on_a_reported_error(temp_folder, monkeypatch, capsys):
    def fail():
        raise hands.BenchError("digest does not match")

    monkeypatch.setattr(hands, "run", _run_writes(temp_folder, fail))
    assert hands.main([]) == 1
    assert "error: digest does not match" in capsys.readouterr().err
    assert not temp_folder.exists()


def test_main_removes_the_temp_folder_on_an_unexpected_error(temp_folder, monkeypatch):
    def fail():
        raise RuntimeError("bug")

    monkeypatch.setattr(hands, "run", _run_writes(temp_folder, fail))
    with pytest.raises(RuntimeError, match="bug"):
        hands.main([])
    assert not temp_folder.exists()


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="POSIX signals")
@pytest.mark.parametrize("signum", [getattr(signal, "SIGHUP", 0), signal.SIGTERM])
def test_a_hangup_or_terminate_still_removes_the_temp_folder(temp_folder, monkeypatch, signum):
    before = signal.getsignal(signum)

    def hang_up():
        os.kill(os.getpid(), signum)
        raise AssertionError("the signal handler did not stop the run")

    monkeypatch.setattr(hands, "run", _run_writes(temp_folder, hang_up))
    with pytest.raises(SystemExit) as e:
        hands.main([])
    assert e.value.code == hands.SIGNAL_EXIT_BASE + signum
    assert not temp_folder.exists()
    assert signal.getsignal(signum) == before


def test_cache_dir_keeps_its_folder(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(hands, "run", _run_writes(cache, lambda: _report([{"variant": "fp32"}])))
    assert hands.main(["--cache-dir", str(cache)]) == 0
    assert (cache / "model.onnx").exists()


def test_the_script_runs_from_stdin_like_over_ssh():
    with SCRIPT.open("rb") as f:
        out = subprocess.run(
            [sys.executable, "-I", "-", "--help"], stdin=f, capture_output=True, timeout=60
        )
    assert out.returncode == 0, out.stderr
    assert b"--cache-dir" in out.stdout


@pytest.mark.models
def test_the_bench_runs_the_pinned_models(tmp_path, capsys):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    out = tmp_path / "hands.json"
    assert (
        hands.main(
            ["--runs", "3", "--warmup", "1", "--cache-dir", str(tmp_path), "--json", str(out)]
        )
        == 0
    )
    text = capsys.readouterr().out
    for variant in hands.VARIANTS:
        assert f"| {variant} | pass |" in text
    assert out.exists()
