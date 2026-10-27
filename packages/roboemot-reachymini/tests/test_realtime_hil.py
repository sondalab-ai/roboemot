"""Realtime speech route on the robot: the SR-5 measurements (SPEC 8, Realtime speech route;
SPEC 13 latency budgets; SPEC 15, To verify on the robot).

Run by the owner, never in CI (``hil`` is deselected by default, and every test skips without
``ROBOEMOT_HIL_WEBUI`` and ``ROBOEMOT_HIL_PIN``), from a machine on the robot's LAN, against the
app running on the robot with the realtime route on: ``mode: platform``, ``input: audio``, gate
``addressed``, the platform's allocator deployed, ``privacy.realtime_audio`` on::

    export ROBOEMOT_HIL_PIN   # the web UI PIN, typed into the shell: never in a file
    ROBOEMOT_HIL_WEBUI=http://<robot>:8042 ROBOEMOT_HIL_ROUTE=pollen \\
        pytest -m hil -s -p no:randomly packages/roboemot-reachymini/tests/test_realtime_hil.py

The PIN is never printed. Each test reads the debug report (``GET /api/debug``, the admin
session) before and after a step the owner performs when asked, prints the figures for SPEC 13
and asserts on the change. The tests run in file order: the one that restarts the app (the
listener's two rounds) comes last. A 401 (an app restart, the session's maximum age) logs in
again with the PIN from the environment, waiting for the app to come back. Nothing here reads
or prints audio, transcripts, replies, keys or identifiers: the report holds none.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.hil

WEBUI = os.environ.get("ROBOEMOT_HIL_WEBUI", "")
PIN = os.environ.get("ROBOEMOT_HIL_PIN", "")
ROUTE = os.environ.get("ROBOEMOT_HIL_ROUTE", "pollen")
TURNS = int(os.environ.get("ROBOEMOT_HIL_TURNS", "10"))
# SPEC 13, cm4: end of user speech to first audio (p95) and barge-in to stop.
FIRST_AUDIO_BUDGET_MS = 2000.0
BARGE_IN_BUDGET_MS = 200.0
NEVE_TRIES = 5
NEVE_MIN_HEARD = 4  # "Neve" heard in at least 4 of 5 tries: the HIL's pass line
BARGE_INS = 5
TOOL_TURNS = 3
TV_SECONDS = 60
REQUEST_TIMEOUT_S = 10.0
RESTART_WAIT_S = 180.0  # how long a 401 or a refused connection waits for the app to come back
RETRY_S = 2.0
CHECK_POLL_S = 0.5
CHECK_MARGIN_S = 15.0
CPU_SAMPLE_EVERY_S = 3.0
SEND_TIMEOUTS_PER_CLOSE = 4  # the check's connect (1) and close (about 3) send timeouts


class Robot:
    """The app's web UI with the admin session: the debug report, settings and the route check.
    A 401 or a refused connection (the app restarted, the session aged out) logs in again with
    the PIN from the environment, for up to ``RESTART_WAIT_S``."""

    def __init__(self, base_url: str, pin: str) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT_S)
        self._pin = pin
        self.csrf = ""
        self._login()

    def _login(self) -> None:
        end = time.monotonic() + RESTART_WAIT_S
        while True:
            try:
                resp = self.http.post("/api/login", json={"pin": self._pin})
                if resp.status_code == 200:
                    self.csrf = resp.json()["csrf"]
                    return
                problem = f"HTTP {resp.status_code}"
            except httpx.TransportError as exc:
                problem = type(exc).__name__
            if time.monotonic() > end:
                raise AssertionError(f"login: {problem}")
            time.sleep(RETRY_S)

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
        for _ in range(2):
            try:
                headers = {"X-CSRF-Token": self.csrf} if method == "POST" else {}
                resp = self.http.request(method, path, json=body, headers=headers)
            except httpx.TransportError:
                self._login()
                continue
            if resp.status_code != 401:
                return resp
            self._login()
        raise AssertionError(f"{method} {path}: still not logged in")

    def report(self) -> dict[str, Any]:
        resp = self.call("GET", "/api/debug")
        assert resp.status_code == 200, f"debug: HTTP {resp.status_code}"
        return resp.json()["sections"]

    def section(self, name: str) -> dict[str, Any]:
        sec = self.report()[name]
        assert sec["ok"], f"{name}: {sec.get('error')}"
        return sec["data"]

    def setting(self, path: str) -> Any:
        fields = self.call("GET", "/api/settings").json()["fields"]
        return next(f["value"] for f in fields if f["path"] == path)

    def set(self, path: str, value: Any) -> None:
        resp = self.call("POST", "/api/settings", {"path": path, "value": value, "pin": self._pin})
        assert resp.status_code == 200, f"{path}: HTTP {resp.status_code}"

    def check(self, route: str) -> dict[str, Any]:
        """Run the Routes tab's realtime check of ``route`` and return its report. The wait is
        the check's own bound (SPEC 8, Realtime check) plus a margin."""
        send = float(self.setting("voice.realtime_route.send_timeout_s"))
        connect = float(self.setting("voice.realtime_route.connect_timeout_s"))
        deadline = float(self.setting("llm_check.deadline_s"))
        wait_s = max(deadline, connect + send) + SEND_TIMEOUTS_PER_CLOSE * send + CHECK_MARGIN_S
        resp = self.call("POST", f"/api/routes/{route}/check", {})
        assert resp.status_code == 202, f"check: HTTP {resp.status_code}"
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            rows = self.call("GET", "/api/routes").json()["routes"]
            row = next(r for r in rows if r["name"] == route)
            if not row["running"]:
                assert row["last_check"] is not None, "no report: the route changed meanwhile"
                return row["last_check"]
            time.sleep(CHECK_POLL_S)
        raise AssertionError(f"the realtime check did not finish in {wait_s:g} s")


@pytest.fixture(scope="module")
def robot() -> Robot:
    if not WEBUI:
        pytest.skip("no robot: set ROBOEMOT_HIL_WEBUI to the robot's web UI")
    if not PIN:
        pytest.skip("no PIN: export ROBOEMOT_HIL_PIN (typed into the shell, never a file)")
    bot = Robot(WEBUI, PIN)
    rt = bot.section("roboemot.realtime_route")
    if rt.get("state") in (None, "off", "blocked"):
        pytest.skip(f"the realtime route is not usable ({rt.get('state')}, {rt.get('reason')})")
    assert rt["mode"] == "platform" and rt["gate"] == "addressed", rt
    return bot


def owner(step: str) -> str:
    """Ask the owner to do ``step`` and wait for Enter (needs ``-s`` and a terminal); returns
    what they typed."""
    if not sys.stdin.isatty():
        pytest.skip("this step needs the owner: run with -s on a terminal")
    return input(f"\n>>> {step}\n    Press Enter when done. ").strip().lower()


def record(name: str, value: Any) -> None:
    """Print a figure for SPEC 13 / SPEC 15 (numbers and codes only)."""
    print(f"HIL {name}: {json.dumps(value)}")


def grew(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def test_end_of_speech_to_transcript_and_to_first_audio_p50_p95(robot):
    """Restart the app first (the figures cover its last 50 turns). Speak ``TURNS`` addressed
    utterances (looking at the robot, engaged), one at a time, waiting for each reply. Records
    transcript and first-audio p50/p95; first audio p95 must be within the cm4 budget (2.0 s)."""
    before = robot.section("roboemot.realtime_route")
    owner(f"Say {TURNS} short questions to the robot, looking at it; wait for each answer.")
    after = robot.section("roboemot.realtime_route")
    latency = after["latency_ms"]
    record("transcript_ms", latency["transcript"])
    record("first_audio_ms", latency["first_audio"])
    record("server_stages_ms", latency["server"])
    assert grew(before, after, "audio_sent") >= TURNS - 1
    assert latency["first_audio"]["count"] >= TURNS - 1
    assert latency["first_audio"]["p95"] <= FIRST_AUDIO_BUDGET_MS


def test_no_underruns_on_the_servers_audio(robot):
    """Five replies played from the server's audio: no underrun on the speaker."""
    before = robot.section("roboemot.voice")["speech"]
    owner("Ask five questions whose answers are a few sentences long; let each reply finish.")
    after = robot.section("roboemot.voice")["speech"]
    record("underruns", grew(before, after, "underruns"))
    record("late_sentences", grew(before, after, "late_sentences"))
    assert grew(before, after, "underruns") == 0


def test_barge_in_stops_the_servers_voice_within_200_ms(robot):
    """Interrupt ``BARGE_INS`` long replies by speaking over them, one at a time. Each stop is
    read on its own (``last_ms`` right after it): the lifetime maximum would carry a slow stop
    from before this test."""
    rt = robot.section("roboemot.realtime_route")
    stops: list[float] = []
    for i in range(BARGE_INS):
        before = robot.section("roboemot.voice").get("bargein", {})
        owner(f"[{i + 1}/{BARGE_INS}] Ask for a long story, then talk over the robot.")
        after = robot.section("roboemot.voice").get("bargein", {})
        if grew(before, after, "triggers") < 1:
            continue  # no barge-in this time: nothing to time
        stops.append(after["latency"]["onset_to_stop"]["last_ms"])
    rt_after = robot.section("roboemot.realtime_route")
    record("barge_in_onset_to_stop_ms", stops)
    record("route_cuts", grew(rt, rt_after, "cuts"))
    assert stops, "no barge-in was detected"
    assert grew(rt, rt_after, "cuts") >= 1
    assert max(stops) <= BARGE_IN_BUDGET_MS


def test_nod_to_phrase_sync_on_the_returned_audio(robot):
    """The speaking motion follows the server's audio (energy frames, phrase gaps): the owner
    watches three replies and judges whether nods and beats land on the phrases."""
    before = robot.section("roboemot.voice")["speech"]
    owner("Ask three questions and watch the head while the robot answers.")
    after = robot.section("roboemot.voice")["speech"]
    assert grew(before, after, "spoken") >= 1
    answer = owner("Did the nods and beats land on the phrases? Type y or n.")
    record("nod_sync", answer)
    assert answer == "y"


def test_neve_as_the_local_wake_word_looking_away(robot):
    """(a) The Vosk path: not engaged (looking away for a while first), the utterance is not
    addressable, so Vosk decodes it locally and the wake gate reads the local text. This is the
    local wake word with the route on, not the server's recognition."""
    before = robot.section("roboemot.voice")
    owner(f'{NEVE_TRIES} times: look away for ten seconds, then say "ciao Neve, come stai?".')
    after = robot.section("roboemot.voice")
    woke = grew(before["wake"]["decisions"], after["wake"]["decisions"], "wake_word")
    record("neve_local_wake_word", woke)
    assert woke >= NEVE_MIN_HEARD


def test_neve_in_the_servers_transcript_facing_the_robot(robot):
    """(b) The server path: engaged and facing the robot, the utterance is addressable, its
    audio goes, and the listener counts the server transcripts that held the name
    (``realtime_audio.wake_word_heard``)."""
    before = robot.section("roboemot.voice").get("realtime_audio", {})
    owner(f'{NEVE_TRIES} times, looking at the robot: say "Neve, come stai?" and wait.')
    after = robot.section("roboemot.voice").get("realtime_audio", {})
    heard = grew(before, after, "wake_word_heard")
    record("neve_server", {"heard": heard, "transcribed": grew(before, after, "transcribed")})
    assert heard >= NEVE_MIN_HEARD


def test_client_chosen_item_ids_are_deleted_and_truncation_is_confirmed(robot):
    """The platform's server takes the robot's own item ids. The realtime check deletes its
    ``input_text`` item; then, with ``conversation.history_turns`` at 1 for the round (set and
    restored here), tool turns push ``input_text`` and ``function_call_output`` items out of
    the history and the server confirms each delete; a barge-in's truncate is confirmed too.
    Watch ``state`` during the check: it opens a second session next to the live one. Also
    records the connect host for ``KEY_HOSTS``."""
    state = robot.section("roboemot.realtime_route")["state"]
    report = robot.check(ROUTE)
    record(
        "realtime_check",
        {
            k: report.get(k)
            for k in (
                "allocator_host",
                "connect_host",
                "header_to_connect_host",
                "platform_id_sent",
                "update_seen",
                "first_audio_ms",
                "item_delete",
                "failure",
            )
        },  # fmt: skip
    )
    record(
        "live_state_around_check",
        [state, robot.section("roboemot.realtime_route")["state"]],
    )
    assert report["update_seen"] is True, report
    assert report["item_delete"] == "accepted", report
    turns = robot.setting("conversation.history_turns")
    robot.set("conversation.history_turns", 1)
    try:
        before = robot.section("roboemot.realtime_route")
        owner(
            f'Ask "che ore sono?" {TOOL_TURNS} times (a tool call each), then ask for a long '
            "story and talk over the robot (a truncate)."
        )
        after = robot.section("roboemot.realtime_route")
    finally:
        robot.set("conversation.history_turns", turns)
    figures = {
        k: grew(before, after, k)
        for k in (
            "tool_calls",
            "items_deleted",
            "items_delete_confirmed",
            "items_truncated",
            "items_truncate_confirmed",
            "errors_unrelated",
        )
    }
    record("item_ids", figures)
    assert figures["tool_calls"] >= 1
    assert figures["items_deleted"] >= 1
    assert figures["items_delete_confirmed"] >= figures["items_deleted"]
    assert figures["items_truncated"] >= 1
    # A server that never sends conversation.item.truncated leaves truncation unverified
    # (SPEC 15): this assert then fails and the figure above is the record.
    assert figures["items_truncate_confirmed"] >= figures["items_truncated"]
    assert figures["errors_unrelated"] == 0


def test_tv_with_nobody_engaged_sends_nothing(robot):
    """A television talking in the room, nobody engaged or in view: no audio leaves."""
    before = robot.section("roboemot.realtime_route")
    owner(
        f"Leave the room (nobody in view), play a TV programme with speech near the robot for "
        f"{TV_SECONDS} s, then come back without speaking."
    )
    after = robot.section("roboemot.realtime_route")
    record("tv_audio_sent", grew(before, after, "audio_sent"))
    record("tv_upload_s", after["upload_s_today"] - before["upload_s_today"])
    assert grew(before, after, "audio_sent") == 0
    assert after["upload_s_today"] == before["upload_s_today"]


# The last test: it restarts the app twice (a 401 then logs in again).


def _cpu(report: dict[str, Any]) -> float | None:
    process = report.get("process", {}).get("data")
    rows = process if isinstance(process, list) else [process]
    values = [r.get("cpu_percent") for r in rows if isinstance(r, dict)]
    return next((v for v in values if v is not None), None)


def _sampled(robot: Robot, step: Callable[[], Any]) -> list[float]:
    """The app's CPU percent, sampled every ``CPU_SAMPLE_EVERY_S`` while ``step`` runs (each
    sample reads the report, which costs a little CPU itself)."""
    samples: list[float] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(CPU_SAMPLE_EVERY_S):
            try:
                value = _cpu(robot.report())
            except AssertionError:
                continue
            if value is not None:
                samples.append(value)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        step()
    finally:
        stop.set()
        thread.join(REQUEST_TIMEOUT_S)
    return samples


def _share(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    """The listener's share over the round: the change of its busy seconds over the change of
    its audio seconds (``realtime.total``), not the last 60 s window."""
    a0, a1 = before["audio_s"], after["audio_s"]
    if a1 <= a0:
        return None
    return (after["share"] * a1 - before["share"] * a0) / (a1 - a0)


def test_listener_share_and_cpu_with_vosk_skipped_versus_local(robot):
    """Two rounds of the same five utterances: realtime on (Vosk skipped), then the route off
    (local Vosk). Records the listener's share over each round (from its totals) and the app's
    CPU sampled during each round; both shares must stay within ``voice.listener.budget_share``.
    The owner restarts the app between the rounds and after the test."""
    rounds = {}
    for mode in ("realtime", "local"):
        if mode == "local":
            owner("In the web UI turn voice.realtime_route.enabled off and restart the app.")
        before = robot.section("roboemot.voice")
        cpu = _sampled(
            robot, lambda m=mode: owner(f"[{m}] Say five short questions, looking at the robot.")
        )
        after = robot.section("roboemot.voice")
        load = after["realtime"]
        rounds[mode] = {
            "share": _share(before["realtime"]["total"], load["total"]),
            "budget_share": load["budget_share"],
            "vosk_skipped": grew(
                before.get("realtime_audio", {}), after.get("realtime_audio", {}), "vosk_skipped"
            ),
            "cpu_percent_mean": statistics.fmean(cpu) if cpu else None,
            "cpu_samples": len(cpu),
        }
    record("listener_rounds", rounds)
    owner("Turn voice.realtime_route.enabled back on and restart the app.")
    assert rounds["realtime"]["vosk_skipped"] >= 1
    for mode, r in rounds.items():
        assert r["share"] is not None and r["share"] <= r["budget_share"], (mode, r)
