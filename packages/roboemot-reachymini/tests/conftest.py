import faulthandler
import time

import pytest

# A thread left behind by a failed sim test (SDK client, executor worker) once kept pytest alive
# for hours after the summary. Past this, dump every thread's stack and exit.
EXIT_WATCHDOG_S = 60


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """The app writes its registry under the data folder: never into the real home folder."""
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path / "roboemot-data"))
    monkeypatch.delenv("ROBOEMOT_MODELS_DIR", raising=False)


class DaytimeClock:
    """Real monotonic time at a fixed noon: the circadian state does not depend on when the
    tests run (quiet hours would force resting)."""

    def monotonic(self) -> float:
        return time.monotonic()

    def hour(self) -> float:
        return 12.0


@pytest.fixture(autouse=True)
def daytime_circadian_clock(monkeypatch):
    from roboemot_reachymini import app as app_module

    monkeypatch.setattr(app_module.RoboEmotApp, "make_circadian_clock", lambda self: DaytimeClock())


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    faulthandler.dump_traceback_later(EXIT_WATCHDOG_S, exit=True)


@pytest.fixture(autouse=True)
def sdk_version_is_current(monkeypatch):
    """Unit tests run with fake SDK handles, without the SDK installed."""
    from roboemot_reachymini import app as app_module

    monkeypatch.setattr(app_module, "installed_sdk_version", lambda: "1.12.0")
