import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """The app writes its registry under the data folder: never into the real home folder."""
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path / "roboemot-data"))
    monkeypatch.delenv("ROBOEMOT_MODELS_DIR", raising=False)
