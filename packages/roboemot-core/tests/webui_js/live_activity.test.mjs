// The Live tab's "What's happening" panel (SPEC 11A, 12, 12A): the status line and the decision
// feed come with the state poll as codes and numbers; the page builds the sentences in its own
// language and shows nothing while the server sends no `activity` block. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const POLL_MS = 15;

const BASE = {
  poll_ms: POLL_MS,
  mood: { label: "joy", valence: 0, arousal: 0, intensity: 0, circadian: null },
  modes: { asleep: false, paused: [], online: true, people: 0, persona: "default" },
  conversation: { phase: "idle", route: null, elapsed_s: null },
  attention: { count: 0, faces: [] },
  reflexes: { last: null },
  motion: { state: "writing" },
  system: { cpu_temp_c: null },
  video: { enabled: false, poll_ms: 333 },
};

const T0 = Date.UTC(2026, 10, 2, 9, 30, 0) / 1000;
const EVENTS = [
  {
    kind: "cue", reason: "no_voice", cue: "fallback", result: "played", failure: "refused",
    detail: "function_call_output_pending", reset: "pending_calls", t: T0 + 4, seq: 5, count: 1,
  },
  { kind: "session", reason: "reset", why: "pending_calls", counter: "pending_call_resets", t: T0 + 3, seq: 4, count: 1 },
  { kind: "unanswered", reason: "not_facing", facing: "not_facing", yaw_deg: 41.5, limit_deg: 30, policy: "never", t: T0 + 2, seq: 3, count: 3 },
  {
    kind: "bargein", reason: "cut", fired: "vad", confirmed: "onset", level_db: -21.25, robot_playing: true,
    moving: false, echo_corr: 0.12, t: T0 + 1, seq: 2, count: 1,
  },
  { kind: "unanswered", reason: "skipped", conversation: "in_progress", t: T0, seq: 1, count: 1 },
  { kind: "mystery", reason: "new_code", t: T0, seq: 0, count: 1 },
];

const ACTIVITY = {
  status: "cloud_down",
  notes: [{ code: "cloud_down", reason: "network" }],
  events: EVENTS,
};

const EN = {
  live_activity: "What's happening",
  live_activity_empty: "Nothing to explain yet.",
  live_activity_count: "x{n}",
  live_status_cloud_down: "Cloud voice down",
  live_status_listening: "Listening",
  live_note_cloud_down: "The cloud voice is down ({reason}).",
  live_yes: "yes",
  live_no: "no",
  live_none: "-",
  live_ev_unknown: "{kind}: {reason}",
  live_ev_cue_fallback: "Fallback sound: {why}",
  live_why_no_voice: "the cloud voice could not answer and there is no local voice",
  live_fail_refused: "the cloud refused the answer ({detail})",
  live_ev_reset_after: "session reset ({reset})",
  live_reset_pending_calls: "a tool call was left open",
  live_ev_session_reset: "Cloud session reset: {why}",
  live_ev_unanswered_not_facing: "Not answered: not facing the robot (head turned {yaw_deg} deg, limit {limit_deg} deg)",
  live_ev_bargein_cut: "Cut by a barge-in (confirmed by {confirmed}): level {level_db} dB, robot playing {robot_playing}, head moving {moving}, echo {echo_corr}",
  live_ev_unanswered_skipped: "Not answered: the cloud model skipped it (conversation: {conversation})",
  live_conv_in_progress: "in progress",
};

const IT = {
  ...EN,
  live_activity: "Cosa succede",
  live_status_cloud_down: "Voce cloud non disponibile",
  live_note_cloud_down: "La voce cloud non è disponibile ({reason}).",
  live_yes: "sì",
  live_ev_cue_fallback: "Suono di ripiego: {why}",
  live_why_no_voice: "la voce cloud non poteva rispondere e non c'è una voce locale",
  live_fail_refused: "il cloud ha rifiutato la risposta ({detail})",
  live_ev_reset_after: "sessione azzerata ({reset})",
  live_reset_pending_calls: "una chiamata a uno strumento è rimasta aperta",
  live_ev_unanswered_not_facing: "Senza risposta: non guardavi il robot (testa girata di {yaw_deg} gradi, limite {limit_deg} gradi)",
  live_ev_bargein_cut: "Interrotto da chi parlava (confermato da {confirmed}): livello {level_db} dB, robot in riproduzione {robot_playing}, testa in movimento {moving}, eco {echo_corr}",
};

async function livePage(t, { strings = EN, language = "en", activity = ACTIVITY } = {}) {
  const page = await loadPage({
    "GET /api/session": () => ({ authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 1000 }),
    "GET /api/strings": { language, strings },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/state": () => ({ ...BASE, activity: typeof activity === "function" ? activity() : activity }),
    "GET /api/live/audio": { audio_poll_ms: 1000, levels: { mic: { state: "off", db: null }, out: { state: "off", db: null }, floor_db: null }, transcript: { enabled: false, epoch: "", rev: 0, min_seq: 0, max_turns: 0, turns: [] } },
  });
  t.after(() => page.window.close());
  page.document.querySelector('#tabs button[data-tab="live"]').click();
  await settle();
  return page;
}

const rows = (page) => [...page.document.querySelectorAll("#live-events li")].map((li) => li.textContent);

test("the panel sits above the cards with the status line, the conditions and the feed", async (t) => {
  const page = await livePage(t);
  const panel = page.document.querySelector("#live-activity");
  assert.ok(panel.nextElementSibling.id === "live-cards", "above the cards");
  assert.equal(page.document.querySelector("#live-cards").children.length, 9, "the cards are unchanged");
  assert.ok(panel.textContent.includes("What's happening"));
  assert.equal(page.document.querySelector("#live-status").textContent, "Cloud voice down");
  assert.ok(panel.textContent.includes("The cloud voice is down (network)."));
  assert.equal(rows(page).length, EVENTS.length);
});

test("each event reads as a plain sentence with its reason and numbers, newest first", async (t) => {
  const page = await livePage(t);
  const [cue, reset, facing, cut, skip, unknown] = rows(page);
  assert.ok(cue.includes("Fallback sound: the cloud voice could not answer and there is no local voice"));
  assert.ok(cue.includes("the cloud refused the answer (function_call_output_pending)"));
  assert.ok(cue.includes("session reset (a tool call was left open)"));
  assert.ok(reset.includes("Cloud session reset: a tool call was left open"));
  assert.ok(facing.includes("head turned 41.5 deg, limit 30 deg"), facing);
  assert.ok(facing.includes("x3"), "a repeated decision shows its count");
  assert.ok(cut.includes("confirmed by onset"));
  assert.ok(cut.includes("level -21.3 dB, robot playing yes, head moving no, echo 0.1"), cut);
  assert.ok(skip.includes("(conversation: in progress)"));
  assert.ok(unknown.includes("mystery: new_code"), "an event the page has no sentence for still shows");
  for (const row of rows(page)) assert.match(row, /\d{1,2}[:.]\d{2}/, "each row has a local time");
});

test("the panel follows the page language", async (t) => {
  const page = await livePage(t, { strings: IT, language: "it" });
  const panel = page.document.querySelector("#live-activity");
  assert.ok(panel.textContent.includes("Cosa succede"));
  assert.equal(page.document.querySelector("#live-status").textContent, "Voce cloud non disponibile");
  const [cue, , facing, cut] = rows(page);
  assert.ok(cue.includes("Suono di ripiego: la voce cloud non poteva rispondere"));
  assert.ok(cue.includes("il cloud ha rifiutato la risposta (function_call_output_pending)"));
  assert.ok(cue.includes("sessione azzerata (una chiamata a uno strumento è rimasta aperta)"));
  assert.ok(facing.includes("testa girata di 41,5 gradi, limite 30 gradi"), "a decimal comma in Italian");
  assert.ok(cut.includes("robot in riproduzione sì"));
});

test("an empty feed says so, and no activity block shows no panel", async (t) => {
  const page = await livePage(t, { activity: { status: "listening", notes: [], events: [] } });
  assert.equal(page.document.querySelector("#live-status").textContent, "Listening");
  assert.equal(page.document.querySelector("#live-events-empty").textContent, "Nothing to explain yet.");
  const off = await livePage(t, { activity: null });
  assert.equal(off.document.querySelector("#live-activity").children.length, 0);
  assert.equal(off.document.querySelector("#live-cards").children.length, 9);
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test("the status line is one node across polls and changes only with the status", async (t) => {
  let status = "cloud_down";
  const page = await livePage(t, { activity: () => ({ ...ACTIVITY, status }) });
  const line = page.document.querySelector("#live-status");
  assert.equal(line.getAttribute("role"), "status");
  const badgeNode = line.firstElementChild;
  await sleep(POLL_MS * 4);
  assert.equal(page.document.querySelector("#live-status"), line);
  assert.equal(line.firstElementChild, badgeNode, "an unchanged status is not rewritten");
  status = "listening";
  await sleep(POLL_MS * 4);
  assert.equal(page.document.querySelector("#live-status"), line, "the same live region");
  assert.equal(line.textContent, "Listening");
});

test("a new event is added above the rows already shown, which stay the same nodes", async (t) => {
  let events = EVENTS.slice(1);
  const page = await livePage(t, { activity: () => ({ ...ACTIVITY, events }) });
  const before = [...page.document.querySelectorAll("#live-events li")];
  assert.equal(before.length, EVENTS.length - 1);
  events = EVENTS;
  await sleep(POLL_MS * 4);
  const after = [...page.document.querySelectorAll("#live-events li")];
  assert.equal(after.length, EVENTS.length);
  assert.ok(after[0].textContent.includes("Fallback sound"), "newest first");
  assert.ok(after.slice(1).every((row, i) => row === before[i]), "the older rows are kept");
});
