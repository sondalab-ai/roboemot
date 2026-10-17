import faulthandler

import pytest

# A thread left behind by a failed sim test (SDK client, executor worker) once kept pytest alive
# for hours after the summary. Past this, dump every thread's stack and exit.
EXIT_WATCHDOG_S = 60


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """The app writes its registry under the data folder: never into the real home folder."""
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path / "roboemot-data"))
    monkeypatch.delenv("ROBOEMOT_MODELS_DIR", raising=False)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    faulthandler.dump_traceback_later(EXIT_WATCHDOG_S, exit=True)
