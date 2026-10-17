"""The minor-present signal: on at once, off only after the hold (SPEC 5, 12)."""

from types import SimpleNamespace

from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision.minor import MinorPresence

HOLD = 5.0


def make(on_rise=None):
    bus, t = Bus(), [100.0]
    presence = MinorPresence(bus, lambda: HOLD, clock=lambda: t[0], on_rise=on_rise)
    return bus, t, presence


def face(bus, stamp, possible_minor):
    bus.publish("percept.face", SimpleNamespace(possible_minor=possible_minor), stamp)


def test_off_until_a_possible_minor_is_seen():
    bus, t, presence = make()
    assert not presence.present()
    face(bus, t[0], False)
    assert not presence.present()


def test_switches_on_immediately_and_calls_on_rise_once():
    rises = []
    bus, t, presence = make(lambda: rises.append(1))
    face(bus, t[0], True)
    face(bus, t[0], True)
    assert presence.present() and rises == [1]


def test_stays_on_for_the_hold_then_releases():
    bus, t, presence = make()
    face(bus, t[0], True)
    t[0] += HOLD - 0.1
    assert presence.present()
    t[0] += 0.2
    assert not presence.present()


def test_an_adult_face_does_not_release_early():
    bus, t, presence = make()
    face(bus, t[0], True)
    t[0] += 1
    face(bus, t[0], False)
    assert presence.present()


def test_each_sighting_restarts_the_hold_and_a_rise_follows_a_release():
    rises = []
    bus, t, presence = make(lambda: rises.append(1))
    face(bus, t[0], True)
    t[0] += HOLD - 1
    face(bus, t[0], True)
    t[0] += HOLD - 1
    assert presence.present() and rises == [1]
    t[0] += 2
    assert not presence.present()
    face(bus, t[0], True)
    assert rises == [1, 1]


def test_person_recognized_counts_and_close_unsubscribes():
    bus, t, presence = make()
    bus.publish("person.recognized", SimpleNamespace(possible_minor=True), t[0])
    assert presence.present()
    presence.close()
    t[0] += 100
    bus.publish("percept.face", SimpleNamespace(possible_minor=True), t[0])
    assert not presence.present()


def test_hold_is_a_conservative_safety_param():
    field = RoboEmotConfig.model_fields["safety"].annotation.model_fields["minor_hold_s"]
    assert field.json_schema_extra["conservative"] == "higher" and field.default == HOLD
