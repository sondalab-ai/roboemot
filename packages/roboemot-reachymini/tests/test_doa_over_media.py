"""The sound direction over the robot's own radio (SPEC 8, Radio playback; owner decision
2026-11-20, PR 289 fix round). The robot's report: over the radio ``direction.degraded.silent``
7 and ``degraded_rejected`` 7. The radio opened the own-sound window that gates the DoA source,
so no bearing reached the voice-at-face check and the face rule never passed.

With a real ``SpeechActivity``, media pushed through the Reachy speaker and the real DoA source:
behind a hardware echo canceller (the host wires ``direction_speaking`` to
``sounding_except_media``) the direction is read over the radio, marked ``listener_only``, and
only the listener's check sees it; the world model and the reflexes do not. Without the
canceller nothing is read over the radio, as before. Robot speech and the listening blip still
gate it either way."""

from __future__ import annotations

import numpy as np
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.voice.activity import SpeechActivity
from roboemot.world import WorldModel, feed_world
from roboemot_reachymini.gaze import FRAME
from roboemot_reachymini.voice import LatestBearing
from roboemot_reachymini.world import DoASource
from test_speaker_pacing import RATE, paced
from test_world_plugin import FakeMini

CFG = RoboEmotConfig()
BEARING = 1.0  # rad, sensor frame: the person speaking over the radio
AGE_S = 1.0


def rig(echo_cancelled: bool):
    """The speaker playing the radio into a real ``SpeechActivity``, the real DoA source gated by
    it as the host wires it, its observations published on a bus, the listener's
    ``LatestBearing`` and a world model fed from that bus."""
    act = SpeechActivity(0)
    sp, _, clock = paced(lead_ms=50, activity=act)
    mini = FakeMini([(BEARING, True)] * 8)
    src = DoASource(mini, CFG.world.scan, lambda: {}, clock)
    src.speaking = act.sounding
    src.direction_speaking = act.sounding_except_media if echo_cancelled else None
    bus = Bus()
    bearing = LatestBearing(bus, FRAME)
    world = WorldModel(CFG.world, FRAME, bus)
    observed: list = []  # what reached the world model (and so attention and the scan)
    feed_world(bus, world, lambda obs, result: observed.append(obs))

    def sink(obs, joint_state):
        bus.publish("percept.entity", obs, obs.stamp, joint_state=joint_state)

    return act, sp, clock, src, sink, bearing, observed


def radio(sp, seconds=1.0):
    sp.push_media(np.full(int(RATE * seconds), 0.1, np.float32), RATE)


def test_behind_an_echo_canceller_the_listener_reads_the_direction_over_the_radio():
    act, sp, clock, src, sink, bearing, observed = rig(echo_cancelled=True)
    radio(sp)
    assert act.sounding(clock.t) and not act.sounding_except_media(clock.t)
    assert not act.speaking(clock.t)
    assert src.poll_once(sink)
    assert bearing.get(clock.t, AGE_S, listener=True) is not None  # the voice-at-face check
    assert bearing.get(clock.t, AGE_S) is None  # the fast path's head turn: as before
    assert observed == []  # the world model, attention and the scan never see it


def test_without_an_echo_canceller_nothing_is_read_over_the_radio():
    act, sp, clock, src, sink, bearing, _ = rig(echo_cancelled=False)
    radio(sp)
    assert not src.poll_once(sink)
    assert len(src._mini.readings) == 8  # the microphones were not even read
    assert bearing.get(clock.t, AGE_S, listener=True) is None


def test_robot_speech_and_the_blip_still_gate_it_behind_the_canceller():
    act, sp, clock, src, sink, bearing, _ = rig(echo_cancelled=True)
    sp.push_audio_sample(np.full(RATE, 0.1, np.float32), RATE)
    assert not src.poll_once(sink)
    act2, sp2, clock2, src2, sink2, _, _ = rig(echo_cancelled=True)
    sp2.push_aside(np.full(RATE // 10, 0.1, np.float32), RATE)
    assert act2.sounding_except_media(clock2.t) and not src2.poll_once(sink2)


def test_without_the_radio_a_reading_reaches_everyone_as_before():
    _, _, clock, src, sink, bearing, observed = rig(echo_cancelled=True)
    assert src.poll_once(sink)
    assert bearing.get(clock.t, AGE_S) is not None
    assert bearing.get(clock.t, AGE_S, listener=True) is not None
    assert len(observed) == 1 and not observed[0].listener_only


def test_the_media_window_ends_with_the_radio_and_a_reset():
    act, sp, clock, *_ = rig(echo_cancelled=True)
    radio(sp, 0.5)
    assert act.sounding(clock.t)
    act.reset(clock.t)
    assert not act.sounding(clock.t + 0.01)
