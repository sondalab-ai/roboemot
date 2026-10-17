"""Voice registry and archive fetching (SPEC 8 stage 2; downloads are untrusted data)."""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import Path

import httpx
import pytest
import yaml
from roboemot import cli
from roboemot.vision.models import ModelError, ModelFile, check_file, fetch_file, status
from roboemot.voice import models as voice_models


def make_zip(entries: dict[str, bytes], symlink: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        for name, target in (symlink or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, target)
    return buf.getvalue()


def archive_file(body: bytes, unpacked: int = 10_000) -> ModelFile:
    return ModelFile(
        path="vosk/model-x",
        url="https://example.test/model-x.zip",
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
        archive="zip",
        unpacked_bytes=unpacked,
    )


def client(body: bytes) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))


GOOD = make_zip({"model-x/": b"", "model-x/am/final.mdl": b"weights", "model-x/conf.txt": b"c"})


def test_voice_manifest_is_complete_and_pinned():
    specs = voice_models.load_manifest()
    assert set(specs) == {
        "silero-vad",
        "vosk-small-it",
        "vosk-small-en",
        "piper-it-paola",
        "piper-en-amy",
    }
    paths = [f.path for s in specs.values() for f in s.files]
    assert len(paths) == len(set(paths))
    for spec in specs.values():
        assert spec.licence
        for f in spec.files:
            assert f.url and f.url.startswith("https://")
            assert len(f.sha256 or "") == 64 and f.size_bytes
            assert not f.path.startswith("/") and ".." not in f.path
            assert (f.archive is None) == f.path.startswith(("silero/", "piper/"))
    assert {f.archive for s in specs.values() for f in s.files} == {None, "zip"}


def test_voice_urls_are_pinned_to_a_release():
    for spec in voice_models.load_manifest().values():
        for f in spec.files:
            assert "/main/" not in f.url and "/resolve/main/" not in f.url


def test_archive_needs_unpacked_limit():
    with pytest.raises(ValueError):
        ModelFile(path="a", url="https://x/a.zip", sha256="0" * 64, archive="zip")


def test_zip_is_verified_unpacked_and_marked(tmp_path):
    f = archive_file(GOOD)
    assert fetch_file(f, tmp_path, client(GOOD)) == "downloaded"
    assert (tmp_path / "vosk/model-x/am/final.mdl").read_bytes() == b"weights"
    assert check_file(tmp_path, f) == tmp_path / "vosk/model-x"
    assert sorted(p.name for p in (tmp_path / "vosk").iterdir()) == ["model-x", "model-x.verified"]
    assert stat.S_IMODE((tmp_path / "vosk/model-x/conf.txt").stat().st_mode) == 0o600


def test_second_fetch_is_present_without_network(tmp_path):
    f = archive_file(GOOD)
    fetch_file(f, tmp_path, client(GOOD))

    def boom(request):
        raise AssertionError("network used")

    assert fetch_file(f, tmp_path, httpx.Client(transport=httpx.MockTransport(boom))) == "present"


def test_sha_mismatch_discards_everything(tmp_path):
    f = archive_file(GOOD)
    other = make_zip({"model-x/a": b"zzzzzzz"})
    f = f.model_copy(update={"size_bytes": len(other)})
    with pytest.raises(ModelError, match="sha256 mismatch"):
        fetch_file(f, tmp_path, client(other))
    assert list((tmp_path / "vosk").iterdir()) == []


@pytest.mark.parametrize(
    "entries",
    [
        {"/etc/evil": b"x"},
        {"model-x/../../evil": b"x"},
        {"../evil": b"x"},
        {"model-x/ok": b"x", "model-x\\..\\evil": b"x"},
        {"C:/evil": b"x"},
    ],
)
def test_zip_slip_is_rejected(tmp_path, entries):
    body = make_zip(entries)
    with pytest.raises(ModelError, match="unsafe path"):
        fetch_file(archive_file(body), tmp_path, client(body))
    assert not (tmp_path / "evil").exists() and not (tmp_path.parent / "evil").exists()
    assert list((tmp_path / "vosk").iterdir()) == []


def test_symlink_entry_is_rejected(tmp_path):
    body = make_zip({"model-x/ok": b"x"}, symlink={"model-x/link": "/etc/passwd"})
    with pytest.raises(ModelError, match="link"):
        fetch_file(archive_file(body), tmp_path, client(body))
    assert list((tmp_path / "vosk").iterdir()) == []


def test_two_top_level_folders_are_rejected(tmp_path):
    body = make_zip({"a/x": b"1", "b/y": b"2"})
    with pytest.raises(ModelError, match="top-level"):
        fetch_file(archive_file(body), tmp_path, client(body))


def test_unpacked_size_limit(tmp_path):
    body = make_zip({"model-x/big": b"0" * 5000})
    with pytest.raises(ModelError, match="unpacked size"):
        fetch_file(archive_file(body, unpacked=1000), tmp_path, client(body))
    assert list((tmp_path / "vosk").iterdir()) == []


def test_failed_refetch_keeps_the_old_folder(tmp_path):
    f = archive_file(GOOD)
    fetch_file(f, tmp_path, client(GOOD))
    (tmp_path / "vosk/model-x.verified").unlink()
    bad = make_zip({"/abs": b"x"})
    f2 = f.model_copy(update={"sha256": hashlib.sha256(bad).hexdigest(), "size_bytes": len(bad)})
    with pytest.raises(ModelError):
        fetch_file(f2, tmp_path, client(bad))
    assert (tmp_path / "vosk/model-x/am/final.mdl").exists()


def test_swapped_marker_is_caught(tmp_path):
    f = archive_file(GOOD)
    fetch_file(f, tmp_path, client(GOOD))
    (tmp_path / "vosk/model-x.verified").write_text('{"sha256": "0"}')
    with pytest.raises(ModelError, match="changed"):
        check_file(tmp_path, f)


def test_status_of_an_archive_model(tmp_path):
    spec = voice_models.load_manifest()["vosk-small-it"]
    assert status(spec, tmp_path) == "missing"
    assert status(spec, tmp_path, verify=True) == "missing"


def test_voice_fetch_uses_the_voice_manifest(tmp_path, monkeypatch):
    body = b"onnx"
    f = ModelFile(
        path="silero/x.onnx",
        url="https://example.test/x.onnx",
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
    )
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "x",
                        "role": "r",
                        "licence": "MIT",
                        "source": "s",
                        "files": [f.model_dump(exclude_none=True)],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(voice_models, "MANIFEST", manifest)
    report = voice_models.fetch(["x"], tmp_path / "root", client=client(body))
    assert report.files == {"silero/x.onnx": "downloaded"}
    assert voice_models.model_file(tmp_path / "root", "x") == tmp_path / "root/silero/x.onnx"


def test_model_file_refuses_unverified(tmp_path):
    with pytest.raises(ModelError):
        voice_models.model_file(tmp_path, "silero-vad")
    with pytest.raises(ModelError, match="unknown"):
        voice_models.model_file(tmp_path, "nope")


def test_models_dir_defaults_under_data_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path))
    assert voice_models.models_dir() == tmp_path / "voice-models"
    assert voice_models.models_dir("~/x") == Path("~/x").expanduser()


def test_cli_list_and_unknown_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ROBOEMOT_MODELS_DIR", str(tmp_path / "v"))
    assert cli.main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "silero-vad" in out and "yunet" in out and str(tmp_path / "voice-models") in out
    assert cli.main(["models", "fetch", "nope"]) != 0


def test_fetch_creates_a_missing_models_folder_and_its_parents(tmp_path):
    f = archive_file(GOOD)
    root = tmp_path / "not" / "yet" / "there"
    fetch_file(f, root, client(GOOD))
    assert check_file(root, f) == root / "vosk/model-x"
