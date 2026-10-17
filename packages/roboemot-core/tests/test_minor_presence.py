"""The minor-present signal: fail-safe, on at once, off only on evidence after the hold
(SPEC 5, 12)."""

import math
from types import SimpleNamespace

from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig, SafetyConfig
from roboemot.vision.minor import MinorPresence

HOLD = 5.0
SAFETY = SafetyConfig(minor_hold_s=HOLD, minor_hold_periods=3)


def make(sink=None, period=lambda: 0.5):
    bus, t = Bus(), [100.0]
    presence = MinorPresence(bus, SAFETY, period=period, sink=sink, clock=lambda: t[0])
    return bus, t, presence


def face(bus, t, possible_minor):
    bus.publish("percept.face", SimpleNamespace(possible_minor=possible_minor), t[0])


def frame(bus, t):
    bus.publish("percept.face_frame", t[0], t[0])


def run_frames(bus, t, seconds, step=0.5):
    """Face-stage frames with no face, for ``seconds``."""
    end = t[0] + seconds
    while t[0] < end:
        frame(bus, t)
        t[0] += step


def test_starts_present_until_vision_delivered_a_full_hold_of_frames():
    bus, t, presence = make()
    assert presence.present()
    run_frames(bus, t, HOLD - 1)
    assert presence.present()
    run_frames(bus, t, 2)
    assert not presence.present()


def test_vision_off_keeps_the_caps_on_for_ever():
    bus, t, presence = make(period=lambda: math.inf)
    t[0] += 10_000
    assert presence.present()
    frame(bus, t)
    t[0] += 10_000
    assert presence.present()  # a throttled-off stage proves nothing


def test_no_frames_for_the_hold_means_present_again():
    bus, t, presence = make()
    run_frames(bus, t, HOLD + 1)
    assert not presence.present()
    t[0] += HOLD + 1  # the camera or vision thread died
    assert presence.present()
    run_frames(bus, t, HOLD - 1)  # evidence restarts from zero
    assert presence.present()
    run_frames(bus, t, 2)
    assert not presence.present()


def test_a_possible_minor_switches_on_at_once_and_releases_after_the_hold():
    bus, t, presence = make()
    run_frames(bus, t, HOLD + 1)
    face(bus, t, True)
    assert presence.present()
    t[0] += HOLD - 0.6
    frame(bus, t)
    assert presence.present()
    run_frames(bus, t, HOLD + 1)
    assert not presence.present()


def test_an_adult_face_does_not_release_early():
    bus, t, presence = make()
    face(bus, t, True)
    frame(bus, t)
    t[0] += 1
    face(bus, t, False)
    frame(bus, t)
    assert presence.present()


def test_a_payload_without_the_flag_counts_as_a_possible_minor():
    bus, t, presence = make()
    run_frames(bus, t, HOLD + 1)
    bus.publish("percept.face", SimpleNamespace(), t[0])
    assert presence.present()


def test_person_recognized_counts():
    bus, t, presence = make()
    run_frames(bus, t, HOLD + 1)
    bus.publish("person.recognized", SimpleNamespace(possible_minor=True), t[0])
    assert presence.present()


def test_the_hold_follows_a_slow_detector():
    period = [0.5]
    bus, t, presence = make(period=lambda: period[0])
    period[0] = 10.0  # throttled: the hold is 3 x 10 s
    run_frames(bus, t, 20, step=10)
    assert presence.present()
    run_frames(bus, t, 20, step=10)
    assert not presence.present()


def test_configure_applies_a_new_hold_without_reading_the_config_per_percept():
    bus, t, presence = make()
    presence.configure(SafetyConfig(minor_hold_s=20, minor_hold_periods=3))
    run_frames(bus, t, HOLD + 1)
    assert presence.present()


def test_the_sink_gets_every_change_and_a_failing_sink_does_not_stop_the_latch():
    seen = []

    def sink(state):
        seen.append(state)
        raise RuntimeError("boom")

    bus, t, presence = make(sink=sink)
    face(bus, t, True)  # must not raise into the bus
    assert presence.present() and seen == [True]
    assert presence.apply() is True and seen == [True, True]


def test_close_unsubscribes():
    bus, t, presence = make()
    run_frames(bus, t, HOLD + 1)
    presence.close()
    bus.publish("percept.face", SimpleNamespace(possible_minor=True), t[0])
    assert not presence.present()


def test_hold_params_are_conservative_safety_params():
    fields = RoboEmotConfig.model_fields["safety"].annotation.model_fields
    for name in ("minor_hold_s", "minor_hold_periods"):
        assert fields[name].json_schema_extra["conservative"] == "higher"


def test_the_state_is_read_and_written_under_one_lock():
    """A bus callback and the app tick cannot overwrite each other with a stale value."""
    held = []
    bus, t, presence = make(sink=lambda state: held.append(presence._lock.locked()))
    face(bus, t, True)
    frame(bus, t)
    presence.apply()
    assert held == [True, True, True]
