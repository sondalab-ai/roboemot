"""SPEC 5 (Hand check), 11 (Radio), 12A: a fist held up to the robot stops the radio, then a nod
(owner decision 2026-11-19, on by default)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from roboemot import diag
from roboemot.bus import Bus, Message
from roboemot.config import ConfigStack
from roboemot.config.schema import HandsConfig
from roboemot.diagnostics import collect_section
from roboemot.manifest import load_manifest
from roboemot.runtime.assembly import assemble
from roboemot.runtime.diagnostics import build_diagnostics
from roboemot.runtime.host import RuntimeHost
from roboemot.runtime.radio_stop import FIST_STOP, FistStop, stop_and_nod
from roboemot.vision.hands import HandGesture, HandWatch
from roboemot.webui.strings import STRINGS
from test_media_player import JAZZ
from test_media_player import Rig as PlayerRig
from test_runtime import CONFIGS, DaytimeHost, RecordingPlatform
from test_vision_hands import FIST_HAND, FakePalm, FakePose, image

FIST = HandGesture(gesture="fist", held_s=1.04)


class StagesHost(DaytimeHost):
    """Fake vision backends, so the run has a vision pipeline to wire the seam into."""

    def load_vision_stages(self, cfg, profile):
        from roboemot.vision.onnx import Stages  # needs cv2: the vision extra

        return Stages(object(), object(), None)


class FakeRadio:
    def __init__(self, audible: bool = True, playing: bool = True) -> None:
        self.on, self.set = audible, playing
        self.stops: list[str] = []

    def audible(self) -> bool:
        return self.on

    def playing(self) -> bool:
        return self.set

    def stop(self, reason: str) -> None:
        self.stops.append(reason)
        self.on = self.set = False


def config(**radio):
    return ConfigStack(app={"tools": {"radio": radio}}).effective()


class Rig:
    def __init__(self, radio: FakeRadio | None = None, awake: bool = True, **fields) -> None:
        self.radio: FakeRadio | None = radio if radio is not None else FakeRadio()
        self.awake = awake
        self.cfg = config(**fields)
        self.gestures: list[str] = []
        self.events: list[tuple[str, str, dict]] = []
        self.host = RuntimeHost()
        self.host.expression = SimpleNamespace(
            request=lambda label, now: self.gestures.append(label) or True
        )
        self.fist = FistStop(
            lambda: self.radio,
            lambda: self.awake,
            lambda: self.cfg,
            lambda player, reason: stop_and_nod(self.host, player, reason),
        )
        self.fist.events = lambda kind, reason, **f: self.events.append((kind, reason, f))

    def hand(self, gesture=FIST) -> None:
        self.fist.on_hand(Message("percept.hand", gesture, 10.0))


def test_stop_and_nod_stops_the_station_then_nods():
    """SPEC 8, 11 Radio (owner decision 2026-11-19): the listener's voice stop stops the
    station with its reason, then asks the renderer for one nod with the time; without a
    renderer it only stops."""
    calls: list[tuple] = []
    player = SimpleNamespace(stop=lambda reason: calls.append(("stop", reason)))
    host = RuntimeHost()
    stop_and_nod(host, player, "voice_stop")
    assert calls == [("stop", "voice_stop")]
    host.expression = SimpleNamespace(
        request=lambda label, now: calls.append(("gesture", label, now is not None))
    )
    stop_and_nod(host, player, "voice_stop")
    assert calls[1:] == [("stop", "voice_stop"), ("gesture", "nod", True)]


def test_defaults():
    cfg = config()
    assert cfg.tools.radio.fist_stop is True
    assert cfg.vision.hands.enabled is True


@pytest.mark.parametrize(
    ("audible", "awake", "present", "wanted"),
    [
        (True, True, True, True),
        (False, True, True, False),  # a hold (robot speech, a turn) or nothing set
        (True, False, True, False),  # asleep
        (True, True, False, False),  # no player in this run
    ],
)
def test_the_seam_wants_the_hand_check_only_while_the_radio_is_audible_and_awake(
    audible, awake, present, wanted
):
    r = Rig(FakeRadio(audible=audible), awake=awake)
    if not present:
        r.radio = None
    assert r.fist.wanted() is wanted


def test_a_fist_while_playing_stops_the_radio_nods_and_publishes_held_s_only():
    r = Rig()
    r.hand()
    assert r.radio.stops == [FIST_STOP] == ["fist_stop"]
    assert r.gestures == ["nod"]
    assert r.events == [("media", "fist_stop", {"held_s": FIST.held_s})]
    assert r.fist.snapshot() == {"fist_stops": 1, "fist_ignored": 0, "last_held_s": 1.04}


def test_a_fist_while_the_music_is_held_still_stops_the_station_that_is_set():
    r = Rig(FakeRadio(audible=False, playing=True))  # a hold began after the check
    r.hand()
    assert r.radio.stops == ["fist_stop"]


def test_a_fist_with_no_station_does_nothing_and_counts_ignored():
    r = Rig(FakeRadio(audible=False, playing=False))
    r.hand()
    r.radio = None
    r.hand()
    assert r.gestures == [] and r.events == []
    assert r.fist.snapshot() == {"fist_stops": 0, "fist_ignored": 2, "last_held_s": None}


def test_the_field_off_stops_the_check_and_ignores_a_late_fist():
    r = Rig(fist_stop=False)
    assert r.fist.wanted() is False  # no hand check runs for the radio: no CPU spent
    r.hand()  # one already on its way
    assert r.radio.stops == [] and r.gestures == [] and r.events == []
    assert r.fist.snapshot()["fist_ignored"] == 0


def test_another_payload_on_the_topic_is_ignored():
    r = Rig()
    r.hand(gesture={"gesture": "fist"})
    assert r.radio.stops == [] and r.fist.fist_ignored == 0


def test_the_event_is_a_declared_one_and_has_strings_in_both_languages():
    assert "fist_stop" in diag.REASONS["media"]
    assert diag.clean({"held_s": 1.004, "station": "Jazz"}) == {"held_s": 1.0}
    for lang in ("en", "it"):
        assert "{held_s}" in STRINGS[lang]["live_ev_media_fist_stop"]


def test_a_fist_from_the_camera_thread_stops_the_real_player():
    """The bus calls the subscriber in the publisher's thread (the camera thread for
    ``percept.hand``): the player's stop is any-thread and ``media.stopped`` follows."""
    player = PlayerRig()
    assert player.player.play(JAZZ) == "ok"
    player.events.clear()
    host = RuntimeHost()
    nods: list[str] = []
    host.expression = SimpleNamespace(request=lambda label, now: nods.append(label) or True)
    fist = FistStop(
        lambda: player.player,
        lambda: True,
        lambda: player.cfg,
        lambda media, reason: stop_and_nod(host, media, reason),
    )
    assert fist.wanted() is True
    bus = Bus()
    bus.subscribe("percept.hand", fist.on_hand)
    camera = threading.Thread(target=bus.publish, args=("percept.hand", FIST, 1.0))
    camera.start()
    camera.join(2.0)
    assert player.events == [("media.stopped", {"station": JAZZ.name, "reason": "fist_stop"})]
    assert player.player.now_playing() is None and player.output.cuts >= 1
    assert nods == ["nod"] and fist.wanted() is False


def test_the_assembled_runtime_wires_the_seam_the_topic_and_the_report():
    stack = ConfigStack()
    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=stack.effective().safety,
    )
    pytest.importorskip("cv2")
    platform = RecordingPlatform()
    host = StagesHost()
    p = assemble(host, stack, manifest, platform, platform.bind(object()))
    try:
        assert p.fist_stop is not None and p.perception.pipeline is not None
        assert p.perception.pipeline.hands_wanted == p.fist_stop.wanted
        assert p.fist_stop.wanted() is False  # no player before start-up
        host.media = FakeRadio()
        assert p.fist_stop.wanted() is True
        p.loop.motors_off = True  # asleep: the sleep gate's own rule
        assert p.fist_stop.wanted() is False
        p.loop.motors_off = False
        p.bus.publish("percept.hand", FIST, 1.0)
        assert host.media.stops == ["fist_stop"]
        sources = build_diagnostics(host, stack, manifest, p).sources
        (source,) = [s for s in sources if s.name == "roboemot.fist_stop"]
        assert collect_section(source)["data"]["fist_stops"] == 1
    finally:
        p.perception.stop()


def test_a_hot_reload_of_the_field_takes_effect_without_a_restart():
    """SPEC 11A: tools.radio.fist_stop is read on each call, so a web UI change applies at
    once (the models, loaded at start, stay)."""
    stack = ConfigStack()
    radio = FakeRadio()
    fist = FistStop(lambda: radio, lambda: True, stack.effective, lambda r, why: r.stop(why))
    assert fist.wanted() is True
    stack.set("tools.radio.fist_stop", False, source="test")
    assert fist.wanted() is False
    fist.on_hand(Message("percept.hand", FIST, 1.0))
    assert radio.stops == []
    stack.set("tools.radio.fist_stop", True, source="test")
    assert fist.wanted() is True
    fist.on_hand(Message("percept.hand", FIST, 2.0))
    assert radio.stops == ["fist_stop"]


@pytest.mark.parametrize("silenced", [True, False])
def test_one_held_fist_stops_the_radio_once(silenced):
    """SPEC 5, Hold: a fist kept up after the stop is the same hold, so it never fires a
    second stop, whether the stop ends the seam (the music is gone) or not (a radio that
    stays audible)."""

    class Stubborn(FakeRadio):
        def stop(self, reason: str) -> None:
            self.stops.append(reason)
            if silenced:
                self.on = self.set = False

    radio = Stubborn()
    fist = FistStop(lambda: radio, lambda: True, lambda: config(), lambda r, why: r.stop(why))
    watch = HandWatch(HandsConfig(), FakePalm(), FakePose(FIST_HAND))
    dt = 1 / 30
    for i in range(round(6.0 / dt)):  # six seconds of a fist in view
        for stamp, event in watch.step(i * dt, image(), fist.wanted()):
            fist.on_hand(Message("percept.hand", event, stamp))
    assert radio.stops == ["fist_stop"]
    assert fist.fist_stops == 1 and fist.fist_ignored == 0
