"""SPEC 7B: the idle runner of the self-soothing sound (schedule, pick, event, cancel)."""

from __future__ import annotations

import random
import time

import pytest
from roboemot.config.schema import (
    APPRAISAL_EVENTS,
    RegisterParams,
    RoboEmotConfig,
    SootheConfig,
    StateEffects,
)
from roboemot.soothe import SootheRunner
from roboemot.voice.speech import SpeechOutput
from test_speech import FakeBackend, TimelineSink, wait_for

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
    """A fake speech output: a job is live from ``play`` until its length has passed on the
    rig's clock, or until it is cancelled or ``drop`` (a barge-in) is called."""

    def __init__(self, rig, full=False):
        self.rig, self.full, self.played, self.cancelled, self.next_id = rig, full, [], [], 1
        self.until: dict[int, float] = {}

    def play(self, samples, rate_hz, volume_db=0.0):
        if self.full:
            return None
        self.played.append((samples, rate_hz, volume_db))
        self.until[self.next_id] = self.rig.now + samples.size / rate_hz
        self.next_id += 1
        return self.next_id - 1

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        self.until.pop(job_id, None)

    def live(self, job_id):
        return self.rig.now < self.until.get(job_id, -1.0)

    def drop(self):
        self.until.clear()


class Rig:
    def __init__(self, cfg=None, seed=1, full=False, volume_db=0.0):
        self.now, self.events = 0.0, []
        self.out = Output(self, full)
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
    assert rig.now - STEP >= max(IDLE_MIN, INTERVAL * (1 - JITTER))


def test_the_first_sound_is_the_later_of_idle_min_and_the_drawn_interval():
    rig = Rig(config(interval_s=30, jitter=0, idle_min_s=200))
    rig.run(200 - STEP)
    assert not rig.out.played
    assert rig.run_until_sound() and rig.now - STEP == 200
    rig = Rig(config(interval_s=60, jitter=0, idle_min_s=20))
    assert rig.run_until_sound() and rig.now - STEP == 60  # measured from the start of idle


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
    assert rig.now - STEP - n >= max(IDLE_MIN, INTERVAL * (1 - JITTER))


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


def test_a_stop_that_drops_the_sound_ends_it_without_a_cancel_and_starts_the_cooldown():
    rig = Rig()
    assert rig.run_until_sound()
    assert rig.runner.sounding(rig.now)
    rig.run(STEP)
    rig.out.drop()  # barge-in: the output no longer has the job
    rig.run(STEP)
    assert rig.out.cancelled == [] and rig.runner.job is None
    assert not rig.runner.sounding(rig.now)
    cut = rig.now - STEP
    assert rig.run_until_sound()
    assert rig.now - STEP - cut >= COOLDOWN


def test_the_sound_counts_as_ours_until_the_output_has_played_it_out():
    rig = Rig()
    assert rig.run_until_sound()
    start = rig.now - STEP
    rig.out.until[1] = start + LENGTH_S * 5  # a slow sink: the output still holds the sound
    rig.run(LENGTH_S * 3)  # the tick estimate is long past
    assert rig.runner.sounding(rig.now) and rig.out.cancelled == []
    rig.out.until[1] = 0.0
    rig.run(STEP)
    assert not rig.runner.sounding(rig.now) and rig.runner.sounding(rig.now, 5.0)


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


def test_sounding_covers_the_tail_after_the_sound():
    rig = Rig()
    assert not rig.runner.sounding(0.0, 5.0)
    assert rig.run_until_sound()
    assert rig.runner.sounding(rig.now)
    rig.run(LENGTH_S + STEP)  # played out
    assert not rig.runner.sounding(rig.now)
    assert rig.runner.sounding(rig.now, 2.0)
    assert not rig.runner.sounding(rig.now + 3.0, 2.0)


def test_close_cuts_a_sound_still_playing():
    rig = Rig()
    assert rig.run_until_sound()
    rig.runner.close()
    assert rig.out.cancelled == [1] and rig.runner.job is None


def test_self_soothe_is_an_appraisal_event_and_every_default_label_has_a_rule():
    cfg = RoboEmotConfig()
    assert "self.soothe" in APPRAISAL_EVENTS
    rules = {r.when["label"]: r for r in cfg.appraisal.rules if r.event == "self.soothe"}
    assert {v.label for v in cfg.soothe.variants} == set(rules)
    for label, rule in rules.items():
        assert rule.label == label and 0.1 <= rule.valence <= 0.15 and rule.arousal == 0
        assert label in cfg.affect.labels


OFFSET_S = 1000.0  # the runner's clock runs ahead of the real one: the schedule needs no wait
TAIL_S = 0.2
REAL_LENGTH_S = 0.4


class RealRig:
    """The runner over a real ``SpeechOutput`` and a timeline sink (a real sound of 0.4 s)."""

    def __init__(self):
        self.sink = TimelineSink()
        self.out = SpeechOutput(RoboEmotConfig(), self.sink, FakeBackend())
        self.out.start()
        self.cancelled = []
        real_cancel = self.out.cancel
        self.out.cancel = lambda job: (self.cancelled.append(job), real_cancel(job))[1]
        cfg = config(
            interval_s=30,
            jitter=0,
            idle_min_s=0,
            cooldown_s=0,
            variants=[variant("calm", shape={"length_s": (REAL_LENGTH_S, REAL_LENGTH_S)})],
        )
        self.runner = SootheRunner(cfg, self.out, on_event=lambda *a: None, rng=random.Random(1))

    def now(self) -> float:
        return time.monotonic() + OFFSET_S

    def idle(self) -> bool:
        """The audio part of the app predicate."""
        if self.out.waiting(exclude=self.runner.job) > 0:
            return False
        speaking = time.monotonic() < self.sink.until + TAIL_S
        return self.runner.sounding(self.now(), TAIL_S) or not speaking

    def start_sound(self) -> int:
        self.runner.tick(self.now(), True, "awake", False)  # draws the schedule
        self.runner.tick(self.now() + self.runner.cfg.interval_s, True, "awake", False)
        assert self.runner.job is not None
        return self.runner.job


def test_a_real_output_never_makes_the_robot_non_idle_through_the_sound_or_its_end():
    rig = RealRig()
    try:
        job = rig.start_sound()
        deadline = time.monotonic() + REAL_LENGTH_S + TAIL_S + 0.3
        seen = []
        while time.monotonic() < deadline:
            seen.append(rig.idle())
            rig.runner.tick(rig.now() + rig.runner.cfg.interval_s, seen[-1], "awake", False)
            time.sleep(0.01)
        assert rig.sink.pushed and all(seen)
        assert rig.cancelled == [] and rig.runner.job is None and not rig.out.live(job)
    finally:
        rig.out.close()


def test_a_stop_that_cuts_the_real_sound_is_seen_without_a_cancel():
    rig = RealRig()
    try:
        job = rig.start_sound()
        assert wait_for(lambda: rig.sink.pushed)
        rig.out.stop()  # barge-in
        assert wait_for(lambda: not rig.out.live(job))
        later = rig.now() + rig.runner.cfg.interval_s
        rig.runner.tick(later, True, "awake", False)
        assert rig.cancelled == [] and rig.runner.job is None
        assert not rig.runner.sounding(later)
    finally:
        rig.out.close()


def test_a_sentence_queued_behind_the_sound_ends_idle_and_cancels_only_the_sound():
    rig = RealRig()
    try:
        job = rig.start_sound()
        assert wait_for(lambda: rig.sink.pushed)
        rig.out.say(
            "Ciao.",
            language="it",
            register=RegisterParams(),
            circadian=StateEffects(),
            mood=None,
            intensity=0.0,
        )
        assert rig.out.waiting(exclude=job) == 1 and not rig.idle()
        rig.runner.tick(rig.now() + rig.runner.cfg.interval_s, False, "awake", False)
        assert rig.cancelled == [job]
    finally:
        rig.out.close()
