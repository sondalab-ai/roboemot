"""SPEC 6, 7, 8 (Robot expression), 12: a known person who comes back, or who is there when the
robot wakes from the night, is greeted once by first name; never on a face track's churn,
never after a restart, never into an open session and never with anything from an earlier
conversation."""

from __future__ import annotations

from types import SimpleNamespace

from expressor_rig import Rig
from roboemot.circadian.model import CircadianState
from roboemot.runtime.expressor import NO_MEMORY, WAKE_EVENT, ReturnTracker, event_text
from roboemot.vision.types import FacePercept

ABSENCE = 1800.0


class Registry:
    def __init__(self, people):
        self.people = people

    def get_person(self, person_id):
        name, status = self.people[person_id]
        return SimpleNamespace(display_name=name, status=status)


def attached(registry=None, **kw) -> Rig:
    rig = Rig(**kw)
    rig.expressor.registry = registry or Registry({"p1": ("Marcello Barile", "active")})
    rig.expressor.attach()
    return rig


def face(rig: Rig, owner="p1", track=1) -> None:
    percept = FacePercept(
        track, rig.clock(), (0.4, 0.4, 0.2, 0.2), 0.9, 0.0, 0.0, owner, None, False
    )
    rig.host.bus.publish("percept.face", percept, rig.clock())


def circadian(rig: Rig, state: str, reason: str | None = None) -> None:
    rig.runner.state, rig.runner.reason = state, reason
    st = CircadianState(12.0, 0.0, 0.0, 0.0, 0.0, state, None, reason)  # type: ignore[arg-type]
    rig.host.bus.publish("circadian.state", st, rig.clock())


def queued(rig: Rig, person: str = "p1"):
    with rig.expressor._lock:
        return rig.expressor._queue.get(f"returning_person:{person}")


def test_absence_of_greet_absence_s_greets_once():
    tracker = ReturnTracker()
    assert tracker.seen("p1", 0.0, True, ABSENCE) == "none"
    assert tracker.seen("p1", ABSENCE, True, ABSENCE) == "return"
    assert tracker.seen("p1", ABSENCE + 1, True, ABSENCE) == "none"
    assert tracker.seen("p1", 3 * ABSENCE, False, ABSENCE) == "none"  # asleep: not recorded
    assert tracker.check("p1", 3 * ABSENCE, ABSENCE) == "return"


def test_a_return_is_offered_and_greets_by_first_name():
    rig = attached()
    face(rig)
    rig.clock.t = ABSENCE
    face(rig)
    request = queued(rig)
    assert request is not None and request.data == {"wake": ""}
    assert rig.expressor.handle(rig.expressor._take()) == "spoken"
    text = rig.worker.turns[0]["text"]
    assert "Marcello" in text and "Barile" not in text and "p1" not in text
    assert NO_MEMORY in text
    assert rig.worker.turns[0]["profile"] is False


def test_track_churn_in_view_never_greets():
    """Review Focus 5: the face track drops and comes back every 20 s for twice the absence;
    person.recognized fires each time. No greeting."""
    rig = attached()
    track = 0
    for second in range(int(2 * ABSENCE)):
        rig.clock.t = float(second)
        if second % 20 == 0:
            track += 1
            rig.host.bus.publish("person.recognized", SimpleNamespace(owner_id="p1"), rig.clock())
        face(rig, track=track)
    assert queued(rig) is None and rig.expressor.snapshot()["counts"] == {}


def test_first_sighting_after_restart_does_not_greet():
    """Review Focus 5: the memory is in RAM only; a fresh expressor's first sighting is not a
    return."""
    rig = attached()
    rig.clock.t = 10 * ABSENCE
    face(rig)
    assert queued(rig) is None


def test_wake_from_night_greets_once_awake_and_outside_quiet_hours():
    rig = attached()
    circadian(rig, "resting", "quiet_hours")
    for minute in range(30):  # in view through the night: not recorded
        rig.clock.t = minute * 60.0
        face(rig)
    assert queued(rig) is None
    rig.clock.t = 7 * 3600.0
    circadian(rig, "awake")  # 07:00, the quiet hours are over
    face(rig)
    request = queued(rig)
    assert request is not None and request.data == {"wake": "1"}
    rig.expressor._take()
    rig.clock.t += 1
    face(rig)
    assert queued(rig) is None  # once
    named = rig.expressor._named(request)
    assert event_text(named).endswith(WAKE_EVENT.format(who="Marcello"))


def test_woken_inside_quiet_hours_greets_only_after_they_end():
    rig = attached()
    circadian(rig, "resting", "quiet_hours")  # 22:30
    rig.clock.t = 8 * 3600.0
    rig.runner.quiet = True
    circadian(rig, "awake")  # 06:50, woken while the quiet hours still run
    rig.clock.t += 300.0
    face(rig)  # 06:55
    assert queued(rig) is None
    rig.runner.quiet = False
    rig.clock.t += 300.0
    face(rig)  # 07:00
    assert queued(rig).data == {"wake": "1"}


def test_platform_wake_after_sleep_greets():
    rig = attached()
    face(rig)
    rig.host.bus.publish("robot.asleep", None, 1.0)
    rig.asleep = True
    rig.clock.t = 60.0
    face(rig)  # asleep: not recorded
    rig.asleep = False
    rig.clock.t = 2 * 3600.0
    rig.host.bus.publish("robot.awake", None, rig.clock())
    face(rig)
    assert queued(rig).data == {"wake": "1"}


def test_motor_toggles_and_restarts_are_no_wake():
    """S4: the platform's sleep flaps (motors toggled, a daemon restart) for seconds at a time:
    shorter than wake_min_sleep_s, so no wake and no greeting, however often."""
    rig = attached()
    face(rig)
    for i in range(20):
        rig.clock.t = 10.0 + i * 20
        rig.host.bus.publish("robot.asleep", None, rig.clock())
        rig.clock.t += 5.0
        rig.host.bus.publish("robot.awake", None, rig.clock())
        face(rig)
    assert queued(rig) is None and rig.expressor._returns.last_wake is None


def test_a_short_night_rest_is_no_wake():
    rig = attached()
    face(rig)
    circadian(rig, "resting", "sleepiness")
    rig.clock.t = 600.0  # ten minutes, under wake_min_sleep_s
    circadian(rig, "awake")
    rig.clock.t = 601.0
    face(rig)
    assert queued(rig) is None


def test_robot_awake_without_a_sleep_is_no_wake():
    rig = attached()
    face(rig)
    rig.host.bus.publish("robot.awake", None, 1.0)
    rig.clock.t = 2.0
    face(rig)
    assert queued(rig) is None


def test_day_nap_is_not_a_wake():
    rig = attached()
    face(rig)
    rig.clock.t = 60.0
    circadian(rig, "resting", "nap")
    rig.clock.t = 600.0
    circadian(rig, "awake")
    face(rig)  # last seen 10 min earlier
    assert queued(rig) is None


def test_name_from_registry_first_name_only():
    people = {"p1": ("Marcello Barile", "active"), "p2": ("Ada Rossi", "pending")}
    rig = attached(Registry(people))
    first = rig.expressor._named(rig.request("returning_person", {"wake": ""}, key="p1"))
    assert first.data["who"] == "Marcello"
    pending = rig.expressor._named(rig.request("returning_person", {"wake": ""}, key="p2"))
    assert "a person you know" in event_text(pending) and "Ada" not in event_text(pending)


def test_greeting_never_carries_profile_or_history():
    """Through the chat fake: the greeting's request has no profile summary and no history,
    and its text forbids recalling an earlier conversation."""
    from test_conversation_worker import build
    from test_event_turn import audible

    chat = audible(build())
    chat.profile[0] = "PROFILE-91ab likes jazz"
    try:
        rig = attached()
        rig.worker.session = False
        request = rig.expressor._named(rig.request("returning_person", {"wake": ""}, key="p1"))
        text = event_text(request)
        future = chat.worker.reaction_turn(
            text, "it", timeout_s=5, quiet=lambda: True, profile=False
        )
        assert future.result(5.0) == "spoken"
        messages = chat.server.requests[0][1]["messages"]
        assert all("PROFILE-91ab" not in str(m["content"]) for m in messages)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "Never mention anything from an earlier conversation" in messages[-1]["content"]
    finally:
        chat.worker.close()


def test_no_greeting_into_an_open_session():
    rig = attached()
    rig.worker.session = True  # the open session holds a person's words
    request = rig.request("returning_person", {"wake": ""}, key="p1")
    assert rig.expressor.handle(request) == "dropped_session"
    assert rig.worker.turns == []


def test_two_people_back_together_are_both_greeted():
    """S5: one slot per person: neither greeting replaces the other."""
    people = {"p1": ("Marcello Barile", "active"), "p2": ("Ada Rossi", "active")}
    rig = attached(Registry(people))
    face(rig, "p1")
    face(rig, "p2", track=2)
    rig.clock.t = ABSENCE
    face(rig, "p1")
    face(rig, "p2", track=2)
    assert queued(rig, "p1") is not None and queued(rig, "p2") is not None
    assert rig.expressor.handle(rig.expressor._take()) == "spoken"
    assert rig.expressor.handle(rig.expressor._take()) == "spoken"
    texts = [t["text"] for t in rig.worker.turns]
    assert "Marcello" in texts[0] and "Ada" in texts[1]
    assert "dropped_replaced" not in rig.counts("returning_person")


def test_a_return_refused_by_the_cooldown_is_not_recorded():
    """S5: last_seen moves only when the offer was taken (or settled), so a return the
    cooldown refused is offered again once the cooldown passed."""
    app = {"expression": {"reactions": {"returning_person": {"cooldown_s": 5000}}}}
    rig = attached(app=app)
    face(rig)
    rig.clock.t = ABSENCE
    face(rig)
    rig.expressor.handle(rig.expressor._take())  # greeted: the cooldown starts
    rig.clock.t = 2 * ABSENCE + 50  # back again, inside the cooldown
    face(rig)
    assert queued(rig) is None and rig.counts("returning_person")["dropped_cooldown"] == 1
    rig.clock.t = ABSENCE + 5001  # the cooldown passed: still away since 1800 s
    face(rig)
    assert queued(rig) is not None


def test_per_person_memory_in_ram_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rig = attached()
    face(rig)
    rig.clock.t = ABSENCE
    face(rig)
    rig.expressor.handle(rig.expressor._take())
    assert list(tmp_path.iterdir()) == []
