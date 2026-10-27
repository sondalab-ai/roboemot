"""Realtime speech route on the robot: the SR-5 measurements (SPEC 8, Realtime speech route;
SPEC 13 latency budgets; SPEC 15, To verify on the robot).

Run by the owner, never in CI (``hil`` is deselected by default, and every test skips without
``ROBOEMOT_HIL_WEBUI``), from a machine on the robot's LAN, against the app running on the robot
with the realtime route on: ``mode: platform``, ``input: audio``, gate ``addressed``, the
platform's allocator deployed, ``privacy.realtime_audio`` on::

    ROBOEMOT_HIL_WEBUI=http://<robot>:8042 ROBOEMOT_HIL_ROUTE=pollen \\
        pytest -m hil -s packages/roboemot-reachymini/tests/test_realtime_hil.py

The PIN is asked on the terminal (or read from ``ROBOEMOT_HIL_PIN``) and never printed. Each
test reads the debug report (``GET /api/debug``, the admin session) before and after a step the
owner performs when asked, prints the figures for SPEC 13 and asserts on the change. Nothing
here reads or prints audio, transcripts, replies, keys or identifiers: the report holds none.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import time
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.hil

WEBUI = os.environ.get("ROBOEMOT_HIL_WEBUI", "")
ROUTE = os.environ.get("ROBOEMOT_HIL_ROUTE", "pollen")
TURNS = int(os.environ.get("ROBOEMOT_HIL_TURNS", "10"))
# SPEC 13, cm4: end of user speech to first audio (p95) and barge-in to stop.
FIRST_AUDIO_BUDGET_MS = 2000.0
BARGE_IN_BUDGET_MS = 200.0
NEVE_MIN_SHARE = 0.8  # "Neve" heard in at least 4 of 5 tries: the HIL's pass line
TV_SECONDS = 60
REQUEST_TIMEOUT_S = 10.0
CHECK_POLL_S = 0.5
CHECK_WAIT_S = 60.0


class Robot:
    """The app's web UI with the admin session: the debug report and the route check."""

    def __init__(self, base_url: str, pin: str) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT_S)
        resp = self.http.post("/api/login", json={"pin": pin})
        assert resp.status_code == 200, f"login: HTTP {resp.status_code}"
        self.csrf = resp.json()["csrf"]

    def report(self) -> dict[str, Any]:
        resp = self.http.get("/api/debug")
        assert resp.status_code == 200, f"debug: HTTP {resp.status_code}"
        return resp.json()["sections"]

    def section(self, name: str) -> dict[str, Any]:
        sec = self.report()[name]
        assert sec["ok"], f"{name}: {sec.get('error')}"
        return sec["data"]

    def check(self, route: str) -> dict[str, Any]:
        """Run the Routes tab's realtime check of ``route`` and return its report."""
        resp = self.http.post(
            f"/api/routes/{route}/check", json={}, headers={"X-CSRF-Token": self.csrf}
        )
        assert resp.status_code == 202, f"check: HTTP {resp.status_code} {resp.text}"
        end = time.monotonic() + CHECK_WAIT_S
        while time.monotonic() < end:
            rows = self.http.get("/api/routes").json()["routes"]
            row = next(r for r in rows if r["name"] == route)
            if not row["running"]:
                return row["last_check"]
            time.sleep(CHECK_POLL_S)
        raise AssertionError("the realtime check did not finish")


@pytest.fixture(scope="module")
def robot() -> Robot:
    if not WEBUI:
        pytest.skip("no robot: set ROBOEMOT_HIL_WEBUI to the robot's web UI")
    pin = os.environ.get("ROBOEMOT_HIL_PIN") or (
        getpass.getpass("web UI PIN: ") if sys.stdin.isatty() else ""
    )
    if not pin:
        pytest.skip("no PIN: run with -s on a terminal, or set ROBOEMOT_HIL_PIN")
    bot = Robot(WEBUI, pin)
    rt = bot.section("roboemot.realtime_route")
    if rt.get("state") in (None, "off", "blocked"):
        pytest.skip(f"the realtime route is not usable ({rt.get('state')}, {rt.get('reason')})")
    assert rt["mode"] == "platform" and rt["gate"] == "addressed", rt
    return bot


def owner(step: str) -> None:
    """Ask the owner to do ``step`` and wait for Enter (needs ``-s`` and a terminal)."""
    if not sys.stdin.isatty():
        pytest.skip("this step needs the owner: run with -s on a terminal")
    input(f"\n>>> {step}\n    Press Enter when done. ")


def record(name: str, value: Any) -> None:
    """Print a figure for SPEC 13 / SPEC 15 (numbers and codes only)."""
    print(f"HIL {name}: {json.dumps(value)}")


def grew(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def test_end_of_speech_to_transcript_and_to_first_audio_p50_p95(robot):
    """Speak ``TURNS`` addressed utterances (looking at the robot, engaged), one at a time,
    waiting for each reply. Records transcript and first-audio p50/p95; first audio p95 must be
    within the cm4 budget (2.0 s)."""
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


def test_listener_share_and_cpu_with_vosk_skipped_versus_local(robot):
    """Two rounds of the same five utterances: realtime on (Vosk skipped), then the route off
    from the web UI (local Vosk). Records the listener's last-window share and the app's CPU in
    each; both shares must stay within ``voice.listener.budget_share``."""
    rounds = {}
    for mode in ("realtime", "local"):
        if mode == "local":
            owner("In the web UI turn voice.realtime_route.enabled off and restart the app.")
        voice = robot.section("roboemot.voice")
        owner(f"[{mode}] Say five short questions to the robot, looking at it.")
        after = robot.section("roboemot.voice")
        load = after["realtime"]
        rounds[mode] = {
            "share": (load["last_window"] or {}).get("share"),
            "budget_share": load["budget_share"],
            "vosk_skipped": grew(
                voice.get("realtime_audio", {}), after.get("realtime_audio", {}), "vosk_skipped"
            ),
            "cpu_percent": _cpu(robot.report()),
        }
        assert not load["over_budget"], rounds
    record("listener_rounds", rounds)
    assert rounds["realtime"]["vosk_skipped"] >= 1
    owner("Turn voice.realtime_route.enabled back on and restart the app.")


def _cpu(report: dict[str, Any]) -> float | None:
    process = report.get("process", {}).get("data")
    rows = process if isinstance(process, list) else [process]
    values = [r.get("cpu_percent") for r in rows if isinstance(r, dict)]
    return next((v for v in values if v is not None), None)


def test_no_underruns_on_the_servers_audio(robot):
    """Five replies played from the server's audio: no underrun on the speaker."""
    before = robot.section("roboemot.voice")["speech"]
    owner("Ask five questions whose answers are a few sentences long; let each reply finish.")
    after = robot.section("roboemot.voice")["speech"]
    record("underruns", grew(before, after, "underruns"))
    record("late_sentences", grew(before, after, "late_sentences"))
    assert grew(before, after, "underruns") == 0


def test_barge_in_stops_the_servers_voice_within_200_ms(robot):
    """Interrupt five long replies by speaking over them. The local barge-in stops the speaker
    within 200 ms of the first speech frame and the route cuts the response."""
    voice = robot.section("roboemot.voice")["bargein"]
    rt = robot.section("roboemot.realtime_route")
    owner("Five times: ask for a long story, then talk over the robot while it speaks.")
    after = robot.section("roboemot.voice")["bargein"]
    rt_after = robot.section("roboemot.realtime_route")
    stop = after["latency"]["onset_to_stop"]
    record("barge_in_onset_to_stop_ms", stop)
    record("route_cuts", grew(rt, rt_after, "cuts"))
    assert grew(voice, after, "triggers") >= 1
    assert grew(rt, rt_after, "cuts") >= 1
    assert stop["max_ms"] <= BARGE_IN_BUDGET_MS


def test_nod_to_phrase_sync_on_the_returned_audio(robot):
    """The speaking motion follows the server's audio (energy frames, phrase gaps): the owner
    watches three replies and judges whether nods and beats land on the phrases."""
    before = robot.section("roboemot.voice")["speech"]
    owner("Ask three questions and watch the head while the robot answers.")
    after = robot.section("roboemot.voice")["speech"]
    assert grew(before, after, "spoken") >= 1
    owner("If the nods and beats landed on the phrases, press Enter; otherwise Ctrl-C.")
    record("nod_sync", "owner: in sync")


def test_neve_is_recognised_by_the_server(robot):
    """Five times, not engaged (looking away for a while first), say "ciao Neve" and a question:
    the wake decision comes from the server's transcript."""
    before = robot.section("roboemot.voice")
    owner('Five times: look away for ten seconds, then say "ciao Neve, come stai?".')
    after = robot.section("roboemot.voice")
    woke = grew(before["wake"]["decisions"], after["wake"]["decisions"], "wake_word")
    transcribed = grew(
        before.get("realtime_audio", {}), after.get("realtime_audio", {}), "transcribed"
    )
    record("neve", {"wake_word": woke, "transcribed": transcribed})
    assert woke >= NEVE_MIN_SHARE * 5


def test_client_chosen_item_ids_are_deleted_and_truncation_is_honoured(robot):
    """The platform's server accepts the robot's own item ids: the realtime check deletes its
    ``input_text`` item; a tool call's ``function_call_output`` items and the barge-in's
    ``conversation.item.truncate`` draw no server error. Also records the connect host for
    ``KEY_HOSTS``."""
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
    assert report["update_seen"] is True, report
    assert report["item_delete"] == "accepted", report
    before = robot.section("roboemot.realtime_route")
    owner(
        'Ask "che ore sono?" three times (a tool call), then ask for a long story and talk '
        "over the robot (a truncate)."
    )
    after = robot.section("roboemot.realtime_route")
    record("tool_calls", grew(before, after, "tool_calls"))
    record("items_deleted", grew(before, after, "items_deleted"))
    record("errors_unrelated", grew(before, after, "errors_unrelated"))
    assert grew(before, after, "tool_calls") >= 1
    assert grew(before, after, "cuts") >= 1
    assert grew(before, after, "errors_unrelated") == 0  # no delete or truncate refused


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
