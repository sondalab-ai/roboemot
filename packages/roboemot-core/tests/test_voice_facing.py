"""Facing check (SPEC 8 stage 2): without the wake word, the face at the voice must look at the
robot; a video on a screen in the owner's direction is not answered."""

from __future__ import annotations

import json
import math

import pytest
from roboemot.config.schema import VoiceDirectionConfig, VoiceFacingConfig
from roboemot.voice.direction import DirectionSample, VoiceDirection, matched_yaw
from roboemot.voice.facing import CpuLoad, FacingCheck
from roboemot.voice.wake import AT_FACE, NO_FACE_AT_VOICE, NO_FACING_DATA, NOT_FACING, WakeGate

FACE = math.radians(10.0)
OTHER = math.radians(-40.0)
LOOKING = math.radians(5.0)
AWAY = math.radians(40.0)  # looking at a screen beside the robot


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Cpu:
    """Cumulative (busy, total) seconds that grow at ``percent`` per call."""

    def __init__(self, percent=10.0):
        self.percent, self.busy, self.total = percent, 0.0, 0.0

    def __call__(self):
        self.busy += self.percent / 100.0
        self.total += 1.0
        return self.busy, self.total


def busy_facing():
    """A facing check whose CPU has been above the threshold for longer than the hold."""
    clock, cpu = Clock(), Cpu(percent=99.0)
    check = facing(cpu=cpu, clock=clock, cpu_hold_s=1.0)
    check.poll()
    clock.t = 1.0
    check.poll()
    clock.t = 2.0
    return check


def facing(cpu=None, clock=None, **cfg):
    cfg = {"enabled": True, **cfg}
    load = CpuLoad(reader=cpu or Cpu(), clock=clock or Clock())
    return FacingCheck(VoiceFacingConfig(**cfg), load)


def at_voice(yaw, *, bearing=FACE, faces=(FACE,), yaws=None, muted=False):
    """A sample with the voice at ``bearing`` and the faces' yaws (``yaw`` for each)."""
    return DirectionSample(
        bearing, tuple(faces), True, muted, yaws or (yaw,) * len(faces), engaged_yaw_rad=yaw
    )


def run(samples, *, check=None, doa=True, muted=False, **cfg):
    check = check if check is not None else facing()
    feed = iter(samples)
    d = VoiceDirection(
        VoiceDirectionConfig(**cfg), lambda now: next(feed), doa_available=doa, facing=check
    )
    d.begin()
    for k in range(len(samples)):
        d.sample(float(k), muted=muted)
    return d, d.check()


def outcomes(d):
    return d.snapshot()["facing"]["checks"]


# ---- the gate's verdicts


def test_a_matched_face_looking_at_the_robot_passes():
    d, verdict = run([at_voice(LOOKING)] * 6)
    assert verdict == AT_FACE
    assert outcomes(d)["passed"] == 1
    assert d.snapshot()["facing"]["yaw_deg"]["median"] == pytest.approx(5.0)


def test_a_matched_face_looking_away_is_rejected():
    d, verdict = run([at_voice(AWAY)] * 6)
    assert verdict == NOT_FACING
    assert outcomes(d)["not_facing"] == 1
    assert d.snapshot()["checks"]["passed"] == 1  # the voice was at the face


def test_the_face_at_the_voice_counts_not_another_one():
    s = at_voice(None, faces=(FACE, OTHER), yaws=(AWAY, LOOKING))  # voice from the first face
    assert matched_yaw(s, math.radians(25.0)) == AWAY
    assert run([s] * 4)[1] == NOT_FACING


def test_a_voice_off_every_face_is_rejected_before_the_facing_check():
    d, verdict = run([at_voice(LOOKING, bearing=math.radians(80.0))] * 4)
    assert verdict == NO_FACE_AT_VOICE and sum(outcomes(d).values()) == 0


def test_min_share_and_max_yaw_come_from_config():
    mixed = [at_voice(LOOKING)] * 4 + [at_voice(AWAY)] * 6
    assert run(mixed)[1] == NOT_FACING  # 0.4 < 0.5
    assert run(mixed, check=facing(min_share=0.4))[1] == AT_FACE
    assert run([at_voice(AWAY)] * 4, check=facing(max_yaw_deg=45.0))[1] == AT_FACE


def test_muted_degraded_path_needs_the_attended_face_to_look_at_the_robot():
    seen = [DirectionSample(None, (FACE,), True, False, (AWAY,), AWAY)] * 5
    d, verdict = run(seen, muted=True)  # a barge-in over the robot's voice: engaged rule
    assert verdict == NOT_FACING
    assert d.snapshot()["degraded"]["muted"] == 1
    looking = [DirectionSample(None, (FACE,), True, False, (LOOKING,), LOOKING)] * 5
    assert run(looking, muted=True)[1] == AT_FACE


def test_a_face_at_the_voice_with_no_yaw_is_rejected_for_its_own_reason():
    d, verdict = run([at_voice(None)] * 5)
    assert verdict == NO_FACING_DATA
    assert outcomes(d)["no_data"] == 1
    assert d.snapshot()["facing"]["yaw_deg"]["count"] == 0


@pytest.mark.parametrize("muted,doa", [(True, True), (False, False)])  # muted, no_source
def test_degraded_paths_reject_on_any_fallback(muted, doa):
    no_yaw = [DirectionSample(None, (FACE,), True, False, (None,), None)] * 5
    d, verdict = run(no_yaw, muted=muted, doa=doa)
    assert verdict == NOT_FACING and outcomes(d)["no_data"] == 1
    looking = [DirectionSample(None, (FACE,), True, False, (LOOKING,), LOOKING)] * 5
    d, verdict = run(looking, muted=muted, doa=doa, check=busy_facing())
    assert verdict == NOT_FACING and outcomes(d)["cpu_high"] == 1


def test_a_busy_cpu_leaves_a_voice_heard_at_the_face_to_the_voice_at_face_check():
    d, verdict = run([at_voice(AWAY)] * 5, check=busy_facing())
    assert verdict == AT_FACE and outcomes(d)["cpu_high"] == 1
    assert run([at_voice(AWAY, bearing=1.5)] * 5, check=busy_facing())[1] == NO_FACE_AT_VOICE


def test_facing_checks_break_down_the_passed_voice_at_face_checks():
    check = facing()
    for samples, muted in [
        ([at_voice(LOOKING)] * 3, False),
        ([at_voice(AWAY)] * 3, False),
        ([at_voice(None)] * 3, False),
        ([at_voice(LOOKING, bearing=1.5)] * 3, False),  # off the face: no facing check
        ([DirectionSample(None, (FACE,), True, False, (AWAY,), AWAY)] * 3, True),
    ]:
        d, _ = run(samples, check=check, muted=muted)
    snap = d.snapshot()  # the last run's direction counts, the shared facing counts
    assert sum(snap["facing"]["checks"].values()) == 4


def test_samples_without_a_yaw_do_not_count_against_the_face():
    d, verdict = run([at_voice(LOOKING)] * 2 + [at_voice(None)] * 8)
    assert verdict == AT_FACE and outcomes(d)["passed"] == 1


def test_disabled_keeps_the_previous_behaviour_exactly():
    check = facing(enabled=False)
    d, verdict = run([at_voice(AWAY)] * 6, check=check)
    assert verdict == AT_FACE
    assert sum(outcomes(d).values()) == 0
    assert run([at_voice(AWAY)] * 6, check=check, tolerance_deg=1.0)[1] == AT_FACE  # 0 deg off
    assert run([at_voice(AWAY, bearing=1.5)] * 6, check=check)[1] == NO_FACE_AT_VOICE


def test_reload_turns_it_on_and_off():
    check = facing(enabled=False)
    d = VoiceDirection(
        VoiceDirectionConfig(), lambda now: at_voice(AWAY), doa_available=True, facing=check
    )
    d.apply_config(VoiceDirectionConfig(), VoiceFacingConfig(enabled=True))
    d.begin()
    d.sample(0.0)
    assert d.check() == NOT_FACING


# ---- CPU fallback


def test_cpu_above_the_threshold_for_the_hold_time_falls_back_and_is_counted():
    clock, cpu = Clock(), Cpu(percent=97.0)
    check = facing(cpu=cpu, clock=clock, cpu_hold_s=3.0)
    check.poll()  # baseline at t=0
    for t in (1.0, 2.0):
        clock.t = t
        check.poll()
    assert check.decide([AWAY]) == "not_facing"  # busy for 2 s only: still judged
    clock.t = 3.0
    assert check.decide([AWAY]) == "cpu_high"  # busy since t=0, 3 s
    snap = check.snapshot()
    assert snap["checks"]["cpu_high"] == 1 and snap["checks"]["not_facing"] == 1
    assert snap["cpu_percent"] == pytest.approx(97.0)


def test_one_quiet_sample_ends_the_cpu_fallback():
    clock, cpu = Clock(), Cpu(percent=97.0)
    check = facing(cpu=cpu, clock=clock, cpu_hold_s=2.0)
    check.poll()
    clock.t = 2.0
    assert check.decide([AWAY]) == "cpu_high"
    cpu.percent, clock.t = 20.0, 3.0
    assert check.decide([AWAY]) == "not_facing"


def test_a_busy_sample_after_a_long_gap_restarts_the_stretch():
    clock, cpu = Clock(), Cpu(percent=97.0)
    check = facing(cpu=cpu, clock=clock, cpu_hold_s=3.0)
    check.poll()  # an utterance minutes ago
    clock.t = 300.0
    assert check.decide([AWAY]) == "not_facing"  # the stretch starts now, not at t=0
    clock.t = 302.0
    assert check.decide([AWAY]) == "not_facing"
    clock.t = 303.0
    assert check.decide([AWAY]) == "cpu_high"


def test_cpu_is_sampled_at_most_every_cpu_sample_s():
    clock, cpu = Clock(), Cpu()
    check = facing(cpu=cpu, clock=clock, cpu_sample_s=1.0)
    for k in range(10):
        clock.t = k * 0.1
        check.poll()
    assert cpu.total == 1.0  # one read in the first second
    clock.t = 1.0
    check.poll()
    assert cpu.total == 2.0


def test_a_failing_cpu_reader_never_falls_back():
    def boom():
        raise OSError

    check = FacingCheck(VoiceFacingConfig(enabled=True), CpuLoad(reader=boom, clock=Clock()))
    assert check.decide([AWAY]) == "not_facing"


def test_the_default_reader_gives_busy_within_total():
    from roboemot.voice.facing import cpu_times

    busy, total = cpu_times()
    assert 0.0 <= busy <= total


# ---- the wake gate


def test_the_gate_counts_not_facing_and_the_wake_word_bypasses_it():
    gate = WakeGate("never", 5.0)
    calls = []

    def check():
        calls.append(1)
        return NOT_FACING

    assert not gate.decide(
        wake_word=False, enrolled_in_view=False, start=0.0, last_speech=1.0, at_face=check
    )
    assert gate.decide(
        wake_word=True, enrolled_in_view=False, start=2.0, last_speech=3.0, at_face=check
    )
    assert len(calls) == 1
    decisions = gate.snapshot(0.0)["decisions"]
    assert decisions["rejected_not_facing"] == 1 and decisions["not_required"] == 1


def test_a_window_follow_up_not_facing_is_rejected():
    gate = WakeGate("always", 5.0)
    gate.decide(wake_word=True, enrolled_in_view=False, start=0.0, last_speech=1.0)
    assert not gate.decide(
        wake_word=False,
        enrolled_in_view=False,
        start=2.0,
        last_speech=3.0,
        at_face=lambda: NOT_FACING,
    )
    assert gate.snapshot(0.0)["decisions"]["rejected_not_facing"] == 1


def test_snapshot_is_json_numbers_only():
    d, _ = run([at_voice(LOOKING)] * 3)
    snap = d.snapshot()["facing"]
    assert json.loads(json.dumps(snap)) == snap
    assert set(snap["checks"]) == {"passed", "not_facing", "cpu_high", "no_data"}
    assert set(snap["yaw_deg"]) == {"count", "min", "median", "p90", "max", "last"}


def test_facing_on_with_the_voice_at_face_check_off_is_rejected():
    from pydantic import ValidationError
    from roboemot.config.schema import RoboEmotConfig

    with pytest.raises(ValidationError, match="voice.direction.enabled"):
        RoboEmotConfig.model_validate(
            {"voice": {"facing": {"enabled": True}, "direction": {"enabled": False}}}
        )
    RoboEmotConfig.model_validate({"voice": {"direction": {"enabled": False}}})  # facing off


def test_the_gate_counts_no_facing_data_on_its_own():
    gate = WakeGate("never", 5.0)
    assert not gate.decide(
        wake_word=False,
        enrolled_in_view=False,
        start=0.0,
        last_speech=1.0,
        at_face=lambda: NO_FACING_DATA,
    )
    assert gate.snapshot(0.0)["decisions"]["rejected_no_facing_data"] == 1
