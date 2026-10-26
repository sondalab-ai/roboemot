"""SPEC 7B: the idle runner of the self-soothing sound (schedule, pick, event, cancel)."""

from __future__ import annotations

import random

import pytest
from roboemot.config.schema import APPRAISAL_EVENTS, RoboEmotConfig, SootheConfig
from roboemot.soothe import SootheRunner

RATE = 8000
LENGTH_S = 1.0
SHORT = {"length_s": (LENGTH_S, LENGTH_S)}
INTERVAL, JITTER, IDLE_MIN, COOLDOWN = 100.0, 0.5, 10.0, 30.0
STEP = 0.5


def variant(label, sound="hum", **extra):
    return {"label": label, "sound": sound, "shape": SHORT, **extra}


def config(**extra) -> SootheConfig:
    base = {
        "interval_s": INTERVAL,
        "jitter": JITTER,
        "idle_min_s": IDLE_MIN,
        "cooldown_s": COOLDOWN,
        "rate_hz": RATE,
        "variants": [
            variant("calm"),
            variant("joy", "blirp"),
            variant("sleepy", states=["drowsy"]),
        ],
    }
    return SootheConfig(**{**base, **extra})


class Output:
    def __init__(self, full=False):
        self.full, self.played, self.cancelled, self.next_id = full, [], [], 1

    def play(self, samples, rate_hz, volume_db=0.0):
        if self.full:
            return None
        self.played.append((samples, rate_hz, volume_db))
        self.next_id += 1
        return self.next_id - 1

    def cancel(self, job_id):
        self.cancelled.append(job_id)


class Rig:
    def __init__(self, cfg=None, seed=1, full=False, volume_db=0.0):
        self.out, self.events, self.now = Output(full), [], 0.0
        self.runner = SootheRunner(
            cfg or config(),
            self.out,
            on_event=lambda *a: self.events.append(a),
            rng=random.Random(seed),
            volume_db=lambda: volume_db,
        )

    def run(self, seconds, idle=True, state="awake", quiet=False):
        for _ in range(int(seconds / STEP)):
            self.runner.tick(self.now, idle, state, quiet)
            self.now += STEP

    def run_until_sound(self, limit=10_000.0, **kw):
        """Tick until a new sound is played; ``self.now - STEP`` is then its start."""
        n = len(self.out.played)
        end = self.now + limit
        while len(self.out.played) == n and self.now < end:
            self.run(STEP, **kw)
        return len(self.out.played) > n


def test_no_sound_before_idle_min_and_the_first_is_an_interval_later():
    rig = Rig()
    rig.run(IDLE_MIN - STEP)
    assert not rig.out.played
    assert rig.run_until_sound()
    assert rig.now - STEP >= IDLE_MIN + INTERVAL * (1 - JITTER) - STEP


@pytest.mark.parametrize("seed", range(6))
def test_gaps_between_sounds_stay_inside_the_jitter_band(seed):
    rig = Rig(config(cooldown_s=0), seed=seed)
    starts = []
    for _ in range(4):
        assert rig.run_until_sound()
        starts.append(rig.now - STEP)
    gaps = [b - a - LENGTH_S for a, b in zip(starts, starts[1:], strict=False)]
    assert all(INTERVAL * (1 - JITTER) - STEP <= g <= INTERVAL * (1 + JITTER) + STEP for g in gaps)


def test_the_schedule_is_reproducible_from_the_seed():
    one, two = Rig(seed=7), Rig(seed=7)
    for rig in (one, two):
        rig.run_until_sound()
    assert one.now == two.now
    assert one.out.played[0][0].tobytes() == two.out.played[0][0].tobytes()


def test_no_sound_while_not_idle():
    rig = Rig()
    rig.run(INTERVAL * 5, idle=False)
    assert not rig.out.played and not rig.events


def test_a_non_idle_tick_resets_the_idle_clock_and_the_schedule():
    rig = Rig()
    rig.run(IDLE_MIN + 1)  # scheduled, not yet due
    rig.run(STEP, idle=False)
    n = rig.now
    assert rig.run_until_sound()
    assert rig.now - STEP - n >= IDLE_MIN + INTERVAL * (1 - JITTER) - STEP


@pytest.mark.parametrize(("state", "quiet"), [("resting", False), ("awake", True)])
def test_nothing_plays_when_resting_or_in_quiet_hours(state, quiet):
    rig = Rig()
    rig.run(INTERVAL * 6, state=state, quiet=quiet)
    assert not rig.out.played and not rig.events
    assert rig.run_until_sound()  # the schedule goes on: it plays once the state allows


def test_a_state_without_an_eligible_variant_plays_nothing():
    rig = Rig(config(variants=[variant("sleepy", states=["drowsy"])]))
    rig.run(INTERVAL * 6, state="awake")
    assert not rig.out.played
    assert rig.run_until_sound(state="drowsy")


def test_only_variants_of_the_current_state_are_picked():
    rig = Rig(config(cooldown_s=0))
    for _ in range(12):
        rig.run_until_sound(state="awake")
    assert {e[2]["label"] for e in rig.events} == {"calm", "joy"}
    drowsy = Rig(config(cooldown_s=0), seed=3)
    for _ in range(30):
        drowsy.run_until_sound(state="drowsy")
    assert "sleepy" in {e[2]["label"] for e in drowsy.events}


def test_the_same_variant_never_plays_twice_in_a_row_unless_it_is_the_only_one():
    rig = Rig(config(cooldown_s=0))
    for _ in range(25):
        rig.run_until_sound()
    picks = [(e[2]["label"], e[2]["sound"]) for e in rig.events]
    assert all(a != b for a, b in zip(picks, picks[1:], strict=False))
    solo = Rig(config(cooldown_s=0, variants=[variant("calm")]))
    for _ in range(3):
        solo.run_until_sound()
    assert [e[2]["label"] for e in solo.events] == ["calm"] * 3


def test_the_event_carries_label_and_sound_and_the_sound_plays_at_the_circadian_volume():
    rig = Rig(config(variants=[variant("joy", "blirp")]), volume_db=-6.0)
    assert rig.run_until_sound()
    ((name, at, payload),) = rig.events
    assert name == "self.soothe" and payload == {"label": "joy", "sound": "blirp"}
    assert at == rig.now - STEP
    samples, rate_hz, volume_db = rig.out.played[0]
    assert rate_hz == RATE and volume_db == -6.0
    assert samples.size == pytest.approx(RATE * LENGTH_S, abs=RATE * 0.01)


def test_a_full_queue_emits_no_event_and_retries_at_the_next_schedule():
    rig = Rig(full=True)
    rig.run(INTERVAL * 3)
    assert not rig.events and not rig.out.played
    rig.out.full = False
    assert rig.run_until_sound() and len(rig.events) == 1


def test_idle_ending_mid_sound_cancels_that_sound_and_starts_the_cooldown():
    rig = Rig()
    assert rig.run_until_sound()
    rig.run(STEP, idle=False)  # half-way through the sound
    assert rig.out.cancelled == [1]
    cut = rig.now - STEP
    assert rig.run_until_sound()
    assert rig.now - STEP - cut >= COOLDOWN


def test_no_cancel_once_the_sound_has_finished():
    rig = Rig()
    assert rig.run_until_sound()
    rig.run(LENGTH_S + STEP)
    rig.run(STEP, idle=False)
    assert rig.out.cancelled == []


def test_the_next_sound_never_starts_inside_the_cooldown_after_a_played_one():
    rig = Rig(config(interval_s=30, jitter=0, cooldown_s=300))
    assert rig.run_until_sound()
    first = rig.now - STEP
    assert rig.run_until_sound()
    assert rig.now - STEP - first >= LENGTH_S + 300


def test_disabled_plays_nothing_and_cuts_a_sound_in_flight():
    rig = Rig()
    assert rig.run_until_sound()
    rig.runner.apply_config(config(enabled=False))
    rig.run(STEP)
    assert rig.out.cancelled == [1]
    n = len(rig.out.played)
    rig.run(INTERVAL * 6)
    assert len(rig.out.played) == n


def test_apply_config_takes_effect_at_the_next_schedule():
    rig = Rig(config(interval_s=1000, jitter=0))
    rig.run(IDLE_MIN + 1)
    rig.runner.apply_config(config(interval_s=30, jitter=0, variants=[variant("curiosity")]))
    assert rig.run_until_sound(limit=200)
    assert rig.events[0][2]["label"] == "curiosity"


def test_sounding_covers_the_sound_and_its_tail_and_close_cuts_it():
    rig = Rig()
    assert not rig.runner.sounding(0.0, 5.0)
    assert rig.run_until_sound()
    start = rig.now - STEP
    assert rig.runner.sounding(start + LENGTH_S / 2)
    assert not rig.runner.sounding(start + LENGTH_S + 1.0)
    assert rig.runner.sounding(start + LENGTH_S + 1.0, 2.0)
    rig.runner.close()
    assert rig.out.cancelled == [1]


def test_self_soothe_is_an_appraisal_event_and_every_default_label_has_a_rule():
    cfg = RoboEmotConfig()
    assert "self.soothe" in APPRAISAL_EVENTS
    rules = {r.when["label"]: r for r in cfg.appraisal.rules if r.event == "self.soothe"}
    assert {v.label for v in cfg.soothe.variants} == set(rules)
    for label, rule in rules.items():
        assert rule.label == label and 0.1 <= rule.valence <= 0.15 and rule.arousal == 0
        assert label in cfg.affect.labels
