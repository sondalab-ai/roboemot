import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Nothing in the core tests may write into the real home folder (SPEC 12 Data)."""
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path / "roboemot-data"))
    monkeypatch.delenv("ROBOEMOT_MODELS_DIR", raising=False)
