"""SPEC 8, Robot expression: the expressor's rules, outcomes and two hard rules (never over a
conversation, never touching the music), on a fake clock with fake runtime parts."""

from __future__ import annotations

import logging

import pytest
from expressor_rig import Rig
from roboemot import diag
from roboemot.config.schema import SITUATIONS
from roboemot.llm.prompt import expression_fallback
from roboemot.runtime.expressor import (
    EVENT_PREFIX,
    OUTCOMES,
    event_text,
    fallback_text,
)


def test_spoken_when_idle_online_and_quiet():
    rig = Rig()
    assert rig.expressor.handle(rig.request("failed_action", {"station": "Radio Uno"})) == "spoken"
    (turn,) = rig.worker.turns
    assert turn["text"].startswith(EVENT_PREFIX) and "Radio Uno" in turn["text"]
    assert turn["timeout_s"] == 5.0 and turn["profile"] is True and turn["language"] == "it"
    assert turn["quiet"] is True  # the room check it hands the worker
    assert rig.host.expression.requests == ["puzzled"]
    assert rig.counts("failed_action") == {"spoken": 1}
    assert rig.events[-1][:2] == ("expression", "spoken")
    assert rig.events[-1][2]["situation"] == "failed_action"


def test_stale_request_dropped_at_its_max_age():
    rig = Rig()
    rig.clock.t = 10.5
    assert rig.expressor.handle(rig.request("didnt_catch", at=0.0)) == "dropped_stale"
    rig.clock.t = 29.0
    assert rig.expressor.handle(rig.request("failed_action", {"station": "A"}, at=0.0)) == "spoken"
    rig.clock.t = 31.0
    request = rig.request("failed_action", {"station": "A"}, at=0.0)
    assert rig.expressor.handle(request) == "dropped_stale"
    assert len(rig.worker.turns) == 1


def test_waits_for_idle_and_drops_busy_after_max_age():
    rig = Rig()
    rig.worker.busy_until = 12.0
    assert rig.expressor.handle(rig.request("didnt_catch", at=0.0)) == "dropped_busy"
    assert rig.worker.turns == []
    rig = Rig()
    rig.worker.busy_until = 5.0
    assert rig.expressor.handle(rig.request("didnt_catch", at=0.0)) == "spoken"
    assert rig.worker.turns[0]["t"] >= 5.0


def test_music_at_offer_shows_gesture_only_and_never_touches_the_player():
    rig = Rig()
    rig.media.is_playing = True
    assert rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A")) is False
    assert rig.host.expression.requests == ["puzzled"]
    assert rig.media.calls == []
    assert rig.worker.turns == [] and rig.expressor.snapshot()["queued"] == 0
    assert rig.counts("failed_action") == {"gesture_only_music": 1}
    rig.media.is_playing = False  # the music ends: nothing is said later
    assert rig.expressor.snapshot()["queued"] == 0


def test_audible_music_counts_as_music():
    rig = Rig()
    rig.media.is_audible = True
    assert rig.expressor.handle(rig.request("didnt_catch")) == "gesture_only_music"
    assert rig.worker.turns == [] and rig.media.calls == []


def test_music_starting_during_the_wait_turns_it_gesture_only():
    """Review Focus 1: the person asks for the radio while a reaction waits for the turn to
    end; the station starts before it ends. The reaction is never spoken and the player is
    never paused, stopped or turned down."""
    rig = Rig()
    rig.worker.busy_until = 4.0

    def start_music(t):
        if t >= 2.0:
            rig.media.is_playing = True

    rig.clock.waits.append(start_music)
    request = rig.shown("cup")
    assert rig.expressor.handle(request) == "gesture_only_music"
    assert rig.worker.turns == [] and rig.worker.lines == []
    assert rig.media.calls == []


def test_busy_result_goes_back_to_waiting():
    """Review Focus 2: the worker refused the first turn (a person spoke first); the
    expressor waits for idle again and tries once more inside the max age."""
    rig = Rig(answers=("busy", "spoken"))

    def busy_again(index):
        if index == 0:
            rig.worker.busy_until = 3.0

    rig.worker.on_turn = busy_again
    assert rig.expressor.handle(rig.request("didnt_catch", at=0.0)) == "spoken"
    assert len(rig.worker.turns) == 2
    assert rig.worker.turns[1]["t"] >= 3.0 and rig.worker.turns[1]["t"] <= 10.0
    assert rig.host.expression.requests == []  # didnt_catch has no gesture by default


def test_busy_result_requests_the_gesture_once():
    rig = Rig(answers=("busy", "spoken"))
    rig.expressor.handle(rig.request("failed_action", {"station": "A"}))
    assert rig.host.expression.requests == ["puzzled"]


def test_a_worker_that_keeps_refusing_is_waited_on_never_spun():
    """Regression: the idle check and the worker's refusal could disagree; a refusal then waits
    a poll before trying again, so the tries stay bounded by the max age."""
    rig = Rig(answers=("busy",))
    assert rig.expressor.handle(rig.request("didnt_catch", at=0.0)) == "dropped_busy"
    reactions = rig.stack.effective().expression.reactions
    max_age, retry = reactions.didnt_catch.max_age_s, reactions.busy_retry_s
    assert 1 < len(rig.worker.turns) <= max_age / retry + 1  # S3: backed off, not 20 a second
    gaps = [b["t"] - a["t"] for a, b in zip(rig.worker.turns, rig.worker.turns[1:], strict=False)]
    assert all(gap >= retry for gap in gaps)


def test_the_turn_is_told_when_the_reaction_no_longer_applies():
    """S1: the worker re-checks the reaction's age and validity (``fresh``) itself."""
    rig = Rig()
    turns = []

    def look(index):
        turns.append(rig.worker.turns[index]["fresh"])

    rig.worker.on_turn = look
    rig.expressor.handle(rig.request("didnt_catch", at=0.0))
    assert turns == [True]
    from expressor_rig import done

    captured = {}

    def capture(*args, fresh, **kw):
        captured["fresh"] = fresh
        return done("no_route")

    rig.worker.reaction_turn = capture
    sit = rig.stack.effective().expression.reactions.didnt_catch
    rig.expressor._turn(rig.request("didnt_catch", at=rig.clock()), sit)
    assert captured["fresh"]() is True
    rig.clock.t += 10.5
    assert captured["fresh"]() is False  # past its 10 s max age


def test_didnt_catch_is_stale_once_a_persons_turn_began():
    """S2: the person spoke again (or something else) and their turn started after the
    question's cause: the question is no longer asked."""
    rig = Rig()
    rig.worker.busy_until = 3.0

    def person_spoke(t):
        if t >= 1.0:
            rig.worker.person_at = 1.0

    rig.clock.waits.append(person_spoke)
    assert rig.expressor.handle(rig.request("didnt_catch", at=0.0)) == "dropped_stale"
    assert rig.worker.turns == []
    rig = Rig()
    rig.worker.person_at = -5.0  # an earlier turn: still asked
    assert rig.expressor.handle(rig.request("didnt_catch", at=0.0)) == "spoken"


def test_close_during_a_reaction_plays_nothing_more():
    """After close() no offline line or cue plays, and the turn asked is given up."""
    import concurrent.futures

    rig = Rig(online=True)
    pending: concurrent.futures.Future = concurrent.futures.Future()
    rig.worker.reaction_turn = lambda *a, **kw: pending
    rig.clock.waits.append(lambda t: rig.expressor.close() if t >= 0.2 else None)
    assert rig.expressor.handle(rig.request("failed_action", {"station": "A"})) == "dropped_busy"
    assert pending.cancelled()
    assert rig.worker.lines == [] and rig.host.cues.played == []
    assert rig.expressor.handle(rig.request("didnt_catch")) == "dropped_busy"
    assert rig.worker.lines == [] and rig.host.cues.played == []


def test_a_gesture_plays_once_when_music_starts_after_a_refusal():
    rig = Rig(answers=("busy",))

    def music(index):
        rig.media.is_playing = True

    rig.worker.on_turn = music
    assert rig.expressor.handle(rig.request("failed_action", {"station": "A"})) == (
        "gesture_only_music"
    )
    assert rig.host.expression.requests == ["puzzled"]


def test_a_waiting_confirmation_holds_the_reaction():
    """A confirmation question waits for the person's yes or no: the reaction waits too."""
    rig = Rig()
    rig.worker.confirm = True

    def answered(t):
        if t >= 2.0:
            rig.worker.confirm = False

    rig.clock.waits.append(answered)
    assert rig.expressor.handle(rig.request("didnt_catch", at=0.0)) == "spoken"
    assert rig.worker.turns[0]["t"] >= 2.0


@pytest.mark.parametrize("cause", ["asleep", "resting", "quiet_hours"])
def test_asleep_or_quiet_hours_while_waiting_drops_it(cause):
    """Review Focus 3: the robot falls asleep with a reaction queued: dropped, nothing said,
    the robot never woken."""
    rig = Rig()
    rig.worker.busy_until = 5.0

    def fall_asleep(t):
        if t >= 1.0:
            if cause == "asleep":
                rig.asleep = True
            elif cause == "resting":
                rig.runner.state = "resting"
            else:
                rig.runner.quiet = True

    rig.clock.waits.append(fall_asleep)
    assert rig.expressor.handle(rig.request("failed_action", {"station": "A"})) == "dropped_asleep"
    assert rig.worker.turns == [] and rig.worker.lines == []
    assert rig.runner.wakes == 0
    assert rig.host.cues.played == []


def test_resting_with_the_rhythm_off_does_not_count():
    rig = Rig()
    rig.runner.enabled = False
    rig.runner.state = "resting"
    assert rig.expressor.handle(rig.request("didnt_catch")) == "spoken"


def test_offer_while_asleep_is_dropped():
    rig = Rig()
    rig.asleep = True
    assert rig.expressor.offer(rig.request("didnt_catch")) is False
    assert rig.counts("didnt_catch") == {"dropped_asleep": 1}


def test_offline_with_local_voice_says_the_fixed_line():
    rig = Rig(online=False)
    request = rig.request("failed_action", {"station": "Radio Uno"})
    assert rig.expressor.handle(request) == "offline_line"
    template = expression_fallback("failed_action", "it", rig.stack.effective())
    assert rig.worker.lines == [(template.replace("{station}", "Radio Uno"), "it")]
    assert rig.worker.lines[0][0] == "La radio Radio Uno si è interrotta."
    assert rig.worker.turns == []


def test_offline_without_voice_plays_the_situation_cue_or_nothing():
    rig = Rig(online=False)
    rig.worker.voice = False
    assert rig.expressor.handle(rig.request("didnt_catch")) == "cue"
    assert rig.expressor.handle(rig.request("failed_action", {"station": "A"})) == "cue"
    shown = rig.shown("cup")
    assert rig.expressor.handle(shown) == "offline_silent"
    assert [kind for kind, _, _ in rig.host.cues.played] == ["not_heard", "fallback"]
    assert rig.worker.lines == []


def test_no_route_falls_back_like_offline():
    rig = Rig(answers=("no_route",))
    assert rig.expressor.handle(rig.request("didnt_catch")) == "offline_line"
    assert rig.worker.lines == [("Scusa, non ho capito. Puoi ripetere?", "it")]


def test_cooldown_per_situation_and_key():
    rig = Rig()
    assert rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A"))
    assert rig.expressor.handle(rig.expressor._take()) == "spoken"  # the cooldown starts here
    rig.clock.t = 60.0
    assert not rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A"))
    assert rig.expressor.offer(rig.request("failed_action", {"station": "B"}, key="B"))
    assert rig.counts("failed_action") == {"spoken": 1, "dropped_cooldown": 1}
    rig.clock.t = 121.0
    rig.expressor._take()
    assert rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A"))


def test_a_replaced_or_dropped_request_never_starts_the_cooldown():
    rig = Rig()
    assert rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A"))
    assert rig.expressor.offer(rig.request("failed_action", {"station": "B"}, key="B"))
    assert rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A"))  # A again
    rig.worker.busy_until = 100.0
    assert rig.expressor.handle(rig.expressor._take()) == "dropped_busy"
    rig.clock.t = 101.0
    assert rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A"))


def test_newer_request_replaces_queued_one_of_same_situation():
    rig = Rig()
    assert rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A"))
    assert rig.expressor.offer(rig.request("failed_action", {"station": "B"}, key="B"))
    assert rig.counts("failed_action") == {"dropped_replaced": 1}
    assert rig.expressor.snapshot()["queued"] == 1
    assert rig.expressor._take().data["station"] == "B"


def test_disabled_offers_nothing():
    rig = Rig(app={"expression": {"reactions": {"enabled": False}}})
    assert rig.expressor.offer(rig.request("didnt_catch")) is False
    rig = Rig(app={"expression": {"reactions": {"didnt_catch": {"enabled": False}}}})
    assert rig.expressor.offer(rig.request("didnt_catch")) is False
    assert rig.expressor.offer(rig.request("failed_action", {"station": "A"}, key="A"))


def test_waits_while_an_announcement_is_handled():
    rig = Rig()
    handling = {"until": 3.0}
    rig.host.announcer = type("A", (), {"busy": lambda self: rig.clock() < handling["until"]})()
    assert rig.expressor.handle(rig.request("didnt_catch")) == "spoken"
    assert rig.worker.turns[0]["t"] >= 3.0


def test_spoken_reaction_opens_the_follow_up_window():
    rig = Rig()
    rig.expressor.handle(rig.request("didnt_catch"))
    assert len(rig.host.voice_listener.opened) == 1 and rig.worker.after == 1
    rig = Rig(answers=("no_route",))
    rig.expressor.handle(rig.request("didnt_catch"))
    assert rig.host.voice_listener.opened == []
    rig = Rig(online=False)
    rig.expressor.handle(rig.request("didnt_catch"))
    assert rig.host.voice_listener.opened == []


def test_media_is_never_called_on_any_path():
    for online in (True, False):
        for answer in ("spoken", "no_route", "busy"):
            rig = Rig(answers=(answer, "spoken"), online=online)
            for situation, data in (
                ("didnt_catch", {}),
                ("failed_action", {"station": "A"}),
                ("shown_object", {"category": "cup"}),
                ("returning_person", {"who": "Ada"}),
            ):
                rig.expressor.offer(rig.request(situation, data, key=situation))
                rig.expressor.handle(rig.request(situation, data))
            rig.expressor.handle(rig.shown("lamp", "e9"))
            assert rig.media.calls == []


def test_logs_and_events_carry_no_data(caplog):
    caplog.set_level(logging.DEBUG, logger="roboemot")
    rig = Rig()
    rig.expressor.handle(rig.request("failed_action", {"station": "Radio Segreta"}))
    rig.expressor.handle(rig.shown("cup"))
    rig.media.is_playing = True
    rig.expressor.offer(rig.request("shown_object", {"category": "cup"}, key="cup"))
    text = caplog.text + repr(rig.events)
    assert "Radio Segreta" not in text and "cup" not in text
    for _, _, fields in rig.events:
        assert set(fields) <= {"situation", "age_s"}


def test_event_text_templates():
    rig = Rig()
    assert "{" not in event_text(rig.request("didnt_catch"))
    text = event_text(rig.request("returning_person", {"who": ""}))
    assert "a person you know" in text and "Never mention" in text
    text = event_text(rig.request("returning_person", {"who": "Ada", "wake": "1"}))
    assert "woke up" in text and "Ada" in text


def test_fallback_without_a_name_drops_the_slot_and_its_space():
    rig = Rig()
    cfg = rig.stack.effective()
    request = rig.request("returning_person", {"who": ""})
    assert fallback_text(request, "it", cfg) == "Ciao!"
    assert fallback_text(request, "en", cfg) == "Hello!"
    request = rig.request("returning_person", {"who": "Ada"})
    assert fallback_text(request, "it", cfg) == "Ciao Ada!"


def test_outcomes_and_situations_are_in_the_event_catalogue():
    assert diag.REASONS["expression"] == OUTCOMES
    assert diag.FIELDS["situation"] == SITUATIONS
    assert diag.FIELDS["age_s"] == diag.NUMBER


def test_snapshot_holds_codes_only():
    rig = Rig()
    rig.expressor.handle(rig.request("failed_action", {"station": "Radio Segreta"}))
    snap = rig.expressor.snapshot()
    assert snap["enabled"] is True and snap["queued"] == 0
    assert "Radio Segreta" not in repr(snap)


def test_the_thread_handles_an_offer():
    """The expressor's own thread takes a queued request (real time, no fake wait)."""
    import threading
    import time

    from expressor_rig import FakeWorker
    from roboemot.runtime.expressor import Expressor

    rig = Rig()
    worker = FakeWorker(time.monotonic)
    rig.host.conversation = worker
    spoke = threading.Event()
    worker.on_turn = lambda index: spoke.set()
    expressor = Expressor(
        worker, rig.host, rig.stack.effective, online=lambda: True, asleep=lambda: False
    )
    expressor.start()
    try:
        assert expressor.offer(rig.request("didnt_catch", at=time.monotonic()))
        assert spoke.wait(5.0)
    finally:
        expressor.close()
    assert expressor._thread is not None and not expressor._thread.is_alive()
