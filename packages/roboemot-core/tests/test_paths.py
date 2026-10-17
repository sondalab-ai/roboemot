"""SPEC 4A Installer, 12 Data: the app data folder."""

import os
import stat
from pathlib import Path

import pytest
from roboemot import paths
from roboemot.install.record import InstallRecord
from roboemot.vision import models


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(paths.ENV_DIR, raising=False)
    monkeypatch.delenv(models.ENV_DIR, raising=False)


def test_default_is_under_the_home_folder():
    assert paths.data_path() == Path.home() / ".local" / "share" / "roboemot"
    assert paths.registry_dir() == paths.data_path() / "registry"
    assert models.models_dir("") == paths.data_path() / "models"


def test_env_overrides_the_default_and_config_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DIR, str(tmp_path / "env"))
    assert paths.data_path() == tmp_path / "env"
    assert paths.data_path(str(tmp_path / "cfg")) == tmp_path / "cfg"
    assert models.models_dir("", str(tmp_path / "cfg")) == tmp_path / "cfg" / "models"


def test_data_dir_is_created_private_whatever_the_umask(tmp_path):
    target = tmp_path / "share" / "roboemot"
    old = os.umask(0o000)
    try:
        assert paths.data_dir(str(target)) == target
    finally:
        os.umask(old)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_data_dir_tightens_a_folder_open_to_others(tmp_path, caplog):
    target = tmp_path / "roboemot"
    target.mkdir()
    target.chmod(0o755)
    with caplog.at_level("WARNING"):
        paths.data_dir(str(target))
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert "accessible to others" in caplog.text


def test_resolving_creates_nothing(tmp_path):
    paths.data_path(str(tmp_path / "x"))
    paths.registry_dir(str(tmp_path / "x"))
    assert not (tmp_path / "x").exists()


def test_old_install_records_load_without_the_data_folder_fields():
    old = {"app": "a", "platform": "p", "packages": ["x"], "version": 1}
    record = InstallRecord.from_json(old)
    assert record.data_dir is None and record.data_dir_parents == []
    assert InstallRecord.from_json(record.to_json()) == record
