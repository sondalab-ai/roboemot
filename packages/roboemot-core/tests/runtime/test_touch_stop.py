"""SPEC 4A, 11 Radio, 12A: a hand on the robot (external control) stops the radio, then a nod."""

from __future__ import annotations

import inspect

from roboemot import diag
from roboemot.config import ConfigStack
from roboemot.runtime import diagnostics as runtime_diagnostics
from roboemot.runtime import loop as runtime_loop
from roboemot.runtime.touch_stop import TouchStop
from roboemot.webui.strings import STRINGS

TICK = 0.05  # one app tick at 20 Hz


class FakeRadio:
    def __init__(self, audible: bool = True) -> None:
        self.on = audible
        self.stops: list[str] = []

    def audible(self) -> bool:
        return self.on

    def stop(self, reason: str) -> None:
        self.stops.append(reason)
        self.on = False


def radio_cfg(**over):
    return ConfigStack(app={"tools": {"radio": over}}).effective().tools.radio


class Rig:
    def __init__(self, radio: FakeRadio | None = None, accept: bool = True, **cfg) -> None:
        self.radio = radio if radio is not None else FakeRadio()
        self.gestures: list[str] = []
        self.events: list[tuple[str, str, dict]] = []
        self.accept = accept
        self.touch = TouchStop(lambda: self.radio, self._gesture)
        self.touch.events = lambda kind, reason, **f: self.events.append((kind, reason, f))
        self.cfg = radio_cfg(**cfg)
        self.t = 100.0

    def _gesture(self, label: str) -> bool:
        self.gestures.append(label)
        return self.accept

    def run(self, seconds: float, external: bool, motors_off: bool = False) -> None:
        for _ in range(round(seconds / TICK)):
            self.touch.tick(self.t, external, self.cfg, motors_off)
            self.t += TICK


def test_defaults_are_on_and_three_tenths_of_a_second():
    cfg = radio_cfg()
    assert cfg.touch_stop is True
    assert cfg.touch_stop_min_s == 0.3


def test_external_control_over_the_minimum_stops_the_radio_once_with_its_reason():
    r = Rig()
    r.run(1.0, external=True)
    assert r.radio.stops == ["touch"]
    assert r.touch.stops == 1
    assert [(k, why) for k, why, _ in r.events] == [("media", "touch_stop")]
    assert r.events[0][2]["held_s"] >= 0.3


def test_shorter_external_control_does_not_stop_the_radio():
    r = Rig()
    r.run(0.25, external=True)
    r.run(1.0, external=False)
    r.run(0.25, external=True)  # a new run starts from zero
    assert r.radio.stops == [] and r.events == [] and r.gestures == []


def test_no_stop_while_the_radio_is_not_audible_or_absent():
    r = Rig(FakeRadio(audible=False))
    r.run(1.0, external=True)
    assert r.radio.stops == [] and r.events == []
    none = Rig()
    none.radio = None  # type: ignore[assignment]
    none.run(1.0, external=True)
    assert none.events == [] and none.touch.stops == 0


def test_the_run_counts_only_while_the_radio_is_audible():
    r = Rig(FakeRadio(audible=False))
    r.run(1.0, external=True)  # held music (robot speech): the run does not count
    r.radio.on = True
    r.run(0.25, external=True)
    assert r.radio.stops == []
    r.run(0.2, external=True)
    assert r.radio.stops == ["touch"]


def test_no_stop_with_the_opt_in_off():
    r = Rig(touch_stop=False)
    r.run(1.0, external=True)
    assert r.radio.stops == [] and r.events == []
    r.run(0.1, external=False)
    assert r.gestures == []


def test_the_nod_waits_for_the_loop_to_take_control_back():
    r = Rig()
    r.run(3.0, external=True)  # the hand stays, then the grace runs
    assert r.radio.stops == ["touch"] and r.gestures == []
    assert r.touch.snapshot()["nod_pending"] is True
    r.run(TICK, external=False)
    assert r.gestures == ["nod"]
    r.run(1.0, external=False)
    assert r.gestures == ["nod"]  # once
    assert r.touch.snapshot() == {
        "stops": 1,
        "nods": 1,
        "nods_refused": 0,
        "nods_dropped": 0,
        "nod_pending": False,
    }


def test_a_refused_nod_is_counted():
    r = Rig(accept=False)
    r.run(1.0, external=True)
    r.run(TICK, external=False)
    assert r.touch.nods == 0 and r.touch.nods_refused == 1


def test_motors_off_drops_the_nod_and_the_run():
    r = Rig()
    r.run(1.0, external=True)  # a platform sleep move with the radio on
    r.run(0.5, external=False, motors_off=True)
    r.run(0.5, external=False)
    assert r.gestures == [] and r.touch.nods_dropped == 1


def test_forget_on_sleep_drops_the_pending_nod():
    r = Rig()
    r.run(1.0, external=True)
    r.touch.forget()
    r.run(0.5, external=False)
    assert r.gestures == [] and r.touch.nods_dropped == 1


def test_the_event_is_a_declared_one_and_has_strings_in_both_languages():
    assert "touch_stop" in diag.REASONS["media"]
    cleaned = diag.clean({"held_s": 0.456, "station": "Jazz"})
    assert "station" not in cleaned
    assert cleaned["held_s"] == 0.46
    for lang in ("en", "it"):
        assert "{held_s}" in STRINGS[lang]["live_ev_media_touch_stop"]


def test_the_main_loop_and_the_report_are_wired():
    awake = inspect.getsource(runtime_loop._tick_until_stopped)
    assert "touch_stop.tick(now, p.loop.external" in awake
    assert "touch_stop.forget()" in awake
    assert "roboemot.touch_stop" in inspect.getsource(runtime_diagnostics.build_diagnostics)
