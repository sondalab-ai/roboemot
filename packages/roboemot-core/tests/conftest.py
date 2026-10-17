import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Nothing in the core tests may write into the real home folder (SPEC 12 Data)."""
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path / "roboemot-data"))
    monkeypatch.delenv("ROBOEMOT_MODELS_DIR", raising=False)


@pytest.fixture(autouse=True)
def fast_web_ui(monkeypatch):
    """Cheap web UI test servers: a short stop poll (``shutdown`` waits one poll interval) and
    a small scrypt cost. Production values stay in ``roboemot.webui``."""
    from roboemot.webui import auth, server

    monkeypatch.setattr(server, "SERVE_POLL_S", 0.01)
    monkeypatch.setattr(auth, "SCRYPT_N", 2**4)
