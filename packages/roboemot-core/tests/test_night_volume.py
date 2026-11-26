"""The night level of the quiet hours (SPEC 7A, 8): ``NightVolume``."""

from __future__ import annotations

from types import SimpleNamespace

from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.runtime.night import NightVolume
from roboemot.runtime.startup import night_volume


def cfg_with(night: float | None = -18.0) -> RoboEmotConfig:
    base = RoboEmotConfig()
    circ = base.circadian.model_copy(update={"quiet_hours_volume_db": night})
    return base.model_copy(update={"circadian": circ})


def test_defaults():
    c = RoboEmotConfig().circadian
    assert c.quiet_hours_volume_db == -18.0
    assert c.quiet_hours_instructions == "whisper, very softly"


def test_inactive_outside_window():
    nv = NightVolume(cfg_with, quiet=lambda: False, state_db=lambda: -6.0)
    assert not nv.active()
    assert nv.extra_db(state_db=-6.0) == 0.0
    assert nv.extra_db(state_db=None) == 0.0


def test_extra_db_takes_off_the_state_offset_the_sound_carries():
    nv = NightVolume(cfg_with, quiet=lambda: True, state_db=lambda: 0.0)
    assert nv.active()
    assert nv.extra_db(state_db=-6.0) == -12.0  # queued resting: -6 + -12 = -18
    assert nv.extra_db(state_db=0.0) == -18.0  # queued awake (woken): still -18 in total
    assert nv.extra_db(state_db=None) == -18.0  # the radio carries no state offset


def test_never_louder():
    nv = NightVolume(lambda: cfg_with(-3.0), quiet=lambda: True, state_db=lambda: -6.0)
    assert nv.extra_db(state_db=-6.0) == 0.0


def test_null_disables():
    nv = NightVolume(lambda: cfg_with(None), quiet=lambda: True, state_db=lambda: 0.0)
    assert not nv.active()
    assert nv.extra_db(state_db=None) == 0.0


def test_runtime_wiring_follows_the_runner_and_the_state():
    """``startup.night_volume``: no runner (rhythm off), no night level; with one, the runner's
    window; ``state_db`` gives the model's state offset to capture at queue time."""
    effects = SimpleNamespace(volume_db=-6.0)
    p = SimpleNamespace(circ=SimpleNamespace(model=SimpleNamespace(effects=lambda: effects)))
    host = SimpleNamespace(circadian_runner=None, circadian=True)
    nv = night_volume(host, ConfigStack(), p)
    assert not nv.active() and nv.extra_db(state_db=-6.0) == 0.0
    host.circadian_runner = SimpleNamespace(quiet_now=lambda: True)
    assert nv.active() and nv.state_db() == -6.0
    assert nv.extra_db(state_db=nv.state_db()) == -12.0


def test_reads_live_config():
    box = [cfg_with(-18.0)]
    nv = NightVolume(lambda: box[0], quiet=lambda: True, state_db=lambda: 0.0)
    box[0] = cfg_with(-24.0)
    assert nv.extra_db(state_db=None) == -24.0
