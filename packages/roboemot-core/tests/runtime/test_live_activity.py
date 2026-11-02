"""The "What's happening" block of the Live tab's state (SPEC 11A, 12; CV-10): a status code from
the parts' own state, the conditions that hold and the ring's events, only while the Live tab
and the panel are on."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from roboemot.bus import Bus
from roboemot.conversation import ConversationPhase
from roboemot.diag import BusEvents, EventRing
from roboemot.runtime.live import STATUSES
from test_live_state import cfg, live, parts


class Listener:
    def __init__(self):
        self.waiting = False

    def awaiting_transcript(self):
        return self.waiting


def rig(**live_cfg):
    config = cfg(**live_cfg)
    clock = [50.0]
    ring = EventRing(config.webui.live, clock=lambda: clock[0])
    bus = Bus()
    ring.attach(bus)
    host = SimpleNamespace(
        circadian=None,
        phase=ConversationPhase(),
        activity=ring,
        voice_listener=Listener(),
        realtime_route=SimpleNamespace(state="connected", reason=None),
    )
    p = parts()
    state = live(p=p, config=config, host=host)
    events = BusEvents(bus, clock=lambda: clock[0])
    return SimpleNamespace(state=state, host=host, p=p, events=events, clock=clock, ring=ring)


DOWN = SimpleNamespace(state="fallback_local", reason="connection_lost")


def realtime_on(r):
    """The realtime route switched on in the view's config copy (no route needs to exist)."""
    c = r.state._cfg
    route = c.voice.realtime_route.model_copy(update={"enabled": True})
    r.state._cfg = c.model_copy(
        update={"voice": c.voice.model_copy(update={"realtime_route": route})}
    )


def activity(r):
    return r.state.state()["activity"]


def test_the_block_holds_the_status_the_notes_and_the_events_newest_first():
    r = rig()
    r.events("cue", "offline", cue="offline", result="played")
    r.events("session", "reset", why="pending_calls")
    a = activity(r)
    assert a["status"] == "idle" and a["notes"] == []
    assert [e["reason"] for e in a["events"]] == ["reset", "offline"]


def test_the_panel_switch_off_sends_no_block_and_the_live_tab_off_sends_nothing():
    r = rig(activity=False)
    assert r.state.state()["activity"] is None
    r = rig(enabled=False)
    assert r.state.state() is None


@pytest.mark.parametrize(
    ("setup", "status"),
    [
        (lambda r: setattr(r.p.sleep, "asleep", True), "asleep"),
        (lambda r: r.host.phase.begin_thinking(), "waiting"),
        (lambda r: r.host.phase.set_listening(True), "listening"),
        (lambda r: setattr(r.host.voice_listener, "waiting", True), "sending"),
        (lambda r: setattr(r.p.network, "online", False), "offline"),
        (
            lambda r: setattr(r.host, "realtime_route", DOWN),
            "cloud_down",
        ),  # fmt: skip
        (lambda r: r.events("bargein", "cut"), "cut"),
    ],
)
def test_each_status_comes_from_the_parts_state(setup, status):
    r = rig()
    realtime_on(r)
    setup(r)
    assert activity(r)["status"] == status
    assert status in STATUSES


def test_answering_comes_first_while_awake():
    r = rig()
    r.host.phase = ConversationPhase(lambda now: True)
    r.host.phase.begin_thinking()
    assert activity(r)["status"] == "answering"


def test_a_cut_shows_only_for_its_hold_time():
    r = rig(activity_cut_hold_s=2.0)
    r.events("bargein", "cut")
    assert activity(r)["status"] == "cut"
    r.clock[0] += 2.5
    assert activity(r)["status"] == "idle"


def test_the_cloud_route_down_is_a_note_with_its_reason_while_the_robot_listens():
    r = rig()
    realtime_on(r)
    r.host.realtime_route = SimpleNamespace(state="fallback_local", reason="connection_lost")
    r.host.phase.set_listening(True)
    a = activity(r)
    assert a["status"] == "listening"
    assert a["notes"] == [{"code": "cloud_down", "reason": "connection_lost"}]
    r.host.realtime_route = SimpleNamespace(state="closed", reason="conversation_end")
    assert activity(r)["notes"] == []  # the robot closed it: not down


def test_a_host_without_voice_still_has_a_status():
    state = live(config=cfg())
    assert state.state()["activity"] == {"status": "idle", "notes": [], "events": []}
