// Behaviour of the Live tab's audio cards (SPEC 11A, 12): two level meters in dBFS and the opt-in
// transcript, polled from /api/live/audio with a since-cursor. Text goes through textContent only.
// Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const POLL_MS = 15;
const WAIT_MS = POLL_MS * 8;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const STATE = {
  poll_ms: 1000,
  mood: { label: "joy", valence: 0.25, arousal: -0.1, intensity: 0.4, circadian: null },
  modes: { asleep: false, paused: [], online: true, minor_present: false, register: null, persona: "default" },
  conversation: { phase: "idle", route: null, elapsed_s: null },
  attention: { count: 0, faces: [] },
  reflexes: { last: null },
  motion: { state: "writing" },
  system: { cpu_temp_c: null },
};

const turn = (seq, who, text, extra = {}) => ({ seq, t: 1000 + seq, who, text, language: "en", interrupted: false, ...extra });
const levels = (mic = -23.4, out = -40.0, floor = -55.2) => ({
  mic: { state: "ok", db: mic }, out: { state: "ok", db: out }, floor_db: floor,
});
const block = (turns, extra = {}) => ({
  enabled: true, epoch: "a.0", seq: turns.length ? turns.at(-1).seq : 0, min_seq: turns.length ? turns[0].seq : 0,
  max_turns: 50, turns, ...extra,
});
const AUDIO = { audio_poll_ms: POLL_MS, levels: levels(), transcript: block([turn(1, "user", "hello"), turn(2, "robot", "hi there")]) };

const STRINGS = {
  live_db: "{n} dBFS",
  live_floor: "Noise floor: {n} dBFS",
  live_level_paused: "paused",
  live_level_no_data: "no signal",
  live_level_unavailable: "not available",
  live_transcript_off: "Transcript is off.",
  live_transcript_empty: "Nothing said yet.",
  live_transcript_user: "Heard",
  live_transcript_robot: "Said",
  live_transcript_cut: "cut off",
  live_transcript_clear: "Clear",
};

async function livePage(t, audio = AUDIO, extra = {}) {
  let loggedIn = true;
  const page = await loadPage({
    "GET /api/session": () => ({ authenticated: loggedIn, csrf: "tok", pin_set: true, status_poll_ms: 1000 }),
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/state": STATE,
    "GET /api/live/audio": audio,
    "POST /api/logout": () => { loggedIn = false; return { ok: true }; },
    ...extra,
  });
  t.after(() => page.window.close());
  return page;
}

// A server that, like the real one, returns only the turns newer than the `after` cursor.
function ringServer(ring) {
  return (body, path) => {
    const after = Number(new URL(path, "http://x").searchParams.get("after"));
    const cursor = after <= ring.seq ? after : 0;
    return { ...AUDIO, transcript: { ...ring, turns: ring.all.filter((x) => x.seq > cursor) } };
  };
}
const ringOf = (all, extra = {}) => ({
  enabled: true, epoch: "a.0", all, seq: all.at(-1)?.seq ?? 0, min_seq: all[0]?.seq ?? 0, max_turns: 50, ...extra,
});

const audioCalls = (page) => page.calls.filter((c) => c.path.startsWith("/api/live/audio"));
const openLive = async (page) => {
  page.document.querySelector('#tabs button[data-tab="live"]').click();
  await settle();
};
const visibility = (page, value) => {
  Object.defineProperty(page.document, "visibilityState", { configurable: true, get: () => value });
  page.document.dispatchEvent(new page.window.Event("visibilitychange"));
};
const turns = (page) => [...page.document.querySelectorAll("#live-transcript .live-turn")];

test("the levels card shows both meters with their dBFS and the noise floor", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  const card = page.document.querySelector("#live-levels");
  assert.ok(card, "levels card");
  const text = card.textContent;
  assert.ok(text.includes("-23.4 dBFS") && text.includes("-40.0 dBFS"), text);
  assert.ok(text.includes("Noise floor: -55.2 dBFS"), text);
  const bars = [...card.querySelectorAll('[role="progressbar"]')];
  assert.equal(bars.length, 2);
  assert.ok(bars[0].getAttribute("aria-valuenow") > bars[1].getAttribute("aria-valuenow"), "louder is fuller");
});

test("paused, no data and unavailable levels show words, not numbers", async (t) => {
  const audio = { ...AUDIO, levels: { mic: { state: "paused", db: null }, out: { state: "unavailable", db: null }, floor_db: null } };
  const page = await livePage(t, audio);
  await openLive(page);
  const text = page.document.querySelector("#live-levels").textContent;
  assert.ok(text.includes("paused") && text.includes("not available"), text);
  assert.ok(!text.includes("dBFS"), text);
  for (const bar of page.document.querySelectorAll("#live-levels [role=progressbar]")) {
    assert.equal(bar.getAttribute("aria-valuenow"), "0");
  }
});

test("the transcript lists turns oldest first, newest last, with who said it", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  const rows = turns(page);
  assert.equal(rows.length, 2);
  assert.ok(rows[0].textContent.includes("hello") && rows[0].textContent.includes("Heard"));
  assert.ok(rows[1].textContent.includes("hi there") && rows[1].textContent.includes("Said"));
  assert.ok(rows[0].classList.contains("live-turn--user") && rows[1].classList.contains("live-turn--robot"));
});

test("an interrupted reply is marked", async (t) => {
  const audio = { ...AUDIO, transcript: block([turn(1, "robot", "Uno.", { interrupted: true })]) };
  const page = await livePage(t, audio);
  await openLive(page);
  assert.ok(turns(page)[0].textContent.includes("cut off"));
});

test("hostile text is shown as text: no element, no handler, no markup", async (t) => {
  const evil = '<img src=x onerror="window.pwned=1"><script>window.pwned=2</script>&amp;';
  const audio = { ...AUDIO, transcript: block([turn(1, "user", evil), turn(2, "robot", "</div><b>x</b>")]) };
  const page = await livePage(t, audio);
  await openLive(page);
  const box = page.document.querySelector("#live-transcript");
  assert.equal(box.querySelectorAll("img, script, b").length, 0);
  assert.ok(box.textContent.includes(evil));
  assert.ok(box.textContent.includes("</div><b>x</b>"));
  assert.equal(page.window.pwned, undefined);
});

test("the page script never builds markup from transcript text", () => {
  const src = readFileSync(new URL("../../src/roboemot/webui/static/live.js", import.meta.url), "utf8");
  assert.ok(!/innerHTML|outerHTML|insertAdjacentHTML|document\.write/.test(src));
});

test("each poll asks only for turns after the last one it has and appends them", async (t) => {
  const ring = ringOf([turn(1, "user", "one")]);
  const page = await livePage(t, undefined, { "GET /api/live/audio": ringServer(ring) });
  await openLive(page);
  assert.equal(audioCalls(page)[0].path, "/api/live/audio?after=0");
  assert.equal(turns(page).length, 1);
  ring.all.push(turn(2, "robot", "two"));
  ring.seq = 2;
  await sleep(WAIT_MS);
  assert.deepEqual(turns(page).map((r) => r.dataset.seq), ["1", "2"]);
  assert.ok(audioCalls(page).some((c) => c.path === "/api/live/audio?after=1"));
  await sleep(WAIT_MS);
  assert.equal(turns(page).length, 2, "nothing new, nothing redrawn");
  assert.ok(audioCalls(page).at(-1).path.endsWith("after=2"));
});

test("a new turn scrolls the log to the bottom", async (t) => {
  const ring = ringOf([turn(1, "user", "one")]);
  const page = await livePage(t, undefined, { "GET /api/live/audio": ringServer(ring) });
  await openLive(page);
  const box = page.document.querySelector("#live-transcript");
  Object.defineProperty(box, "scrollHeight", { configurable: true, value: 500 });
  box.scrollTop = 0;
  ring.all.push(turn(2, "robot", "two"));
  ring.seq = 2;
  await sleep(WAIT_MS);
  assert.equal(box.scrollTop, 500);
});

test("turns the server dropped (min_seq) and an over-long list leave the page too", async (t) => {
  const ring = ringOf([turn(1, "user", "a"), turn(2, "robot", "b"), turn(3, "user", "c")]);
  const page = await livePage(t, undefined, { "GET /api/live/audio": ringServer(ring) });
  await openLive(page);
  assert.equal(turns(page).length, 3);
  ring.all.splice(0, 2, ...[]);
  ring.all.push(turn(4, "robot", "d"));
  Object.assign(ring, { seq: 4, min_seq: 3 });
  await sleep(WAIT_MS);
  assert.deepEqual(turns(page).map((r) => r.dataset.seq), ["3", "4"]);
  ring.all.push(turn(5, "user", "e"), turn(6, "robot", "f"));
  Object.assign(ring, { seq: 6, min_seq: 3, max_turns: 3 });
  await sleep(WAIT_MS);
  assert.deepEqual(turns(page).map((r) => r.dataset.seq), ["4", "5", "6"]);
});

test("a new epoch (cleared, restarted or switched off and on) starts the list over", async (t) => {
  const ring = ringOf([turn(1, "user", "old")]);
  const page = await livePage(t, undefined, { "GET /api/live/audio": ringServer(ring) });
  await openLive(page);
  assert.ok(turns(page)[0].textContent.includes("old"));
  Object.assign(ring, { epoch: "b.0", all: [turn(2, "user", "fresh")], seq: 2, min_seq: 2 });
  await sleep(WAIT_MS);
  assert.deepEqual(turns(page).map((r) => r.textContent.includes("fresh")), [true]);
});

test("the Clear button posts and empties the list", async (t) => {
  const page = await livePage(t, undefined, { "POST /api/live/transcript/clear": { ok: true } });
  await openLive(page);
  const button = page.document.querySelector("#live-transcript-clear");
  assert.ok(button && button.textContent === "Clear");
  button.click();
  await settle();
  assert.ok(page.calls.some((c) => c.method === "POST" && c.path === "/api/live/transcript/clear"));
  assert.equal(turns(page).length, 0);
});

test("with the transcript off the card explains it and hides Clear", async (t) => {
  const audio = { ...AUDIO, transcript: { enabled: false, epoch: "", seq: 0, min_seq: 0, max_turns: 0, turns: [] } };
  const page = await livePage(t, audio);
  await openLive(page);
  assert.ok(page.document.querySelector("#live-transcript-card").textContent.includes("Transcript is off."));
  assert.equal(page.document.querySelector("#live-transcript-clear").hidden, true);
  assert.equal(turns(page).length, 0);
});

test("an empty transcript says so", async (t) => {
  const page = await livePage(t, { ...AUDIO, transcript: block([]) });
  await openLive(page);
  assert.ok(page.document.querySelector("#live-transcript-card").textContent.includes("Nothing said yet."));
});

test("the audio poll runs at the server's interval, only on the open tab, and stops with it", async (t) => {
  const page = await livePage(t);
  await sleep(WAIT_MS);
  assert.equal(audioCalls(page).length, 0);
  await openLive(page);
  await sleep(WAIT_MS);
  assert.ok(audioCalls(page).length >= 3, `polled ${audioCalls(page).length} times`);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await settle();
  const seen = audioCalls(page).length;
  await sleep(WAIT_MS);
  assert.equal(audioCalls(page).length, seen);
});

test("a hidden page stops the audio poll and a visible one resumes it", async (t) => {
  const page = await livePage(t, undefined, { "GET /api/live/audio": ringServer(ringOf([turn(1, "user", "hello"), turn(2, "robot", "hi there")])) });
  await openLive(page);
  await sleep(WAIT_MS);
  visibility(page, "hidden");
  await settle();
  const seen = audioCalls(page).length;
  await sleep(WAIT_MS);
  assert.equal(audioCalls(page).length, seen);
  visibility(page, "visible");
  await sleep(WAIT_MS);
  assert.ok(audioCalls(page).length >= seen + 2);
  assert.equal(turns(page).length, 2, "the list survived and no turn is doubled");
});

test("logging out stops the audio poll", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  page.document.querySelector("#logout").click();
  await settle();
  await sleep(POLL_MS * 2);
  const seen = audioCalls(page).length;
  await sleep(WAIT_MS);
  assert.equal(audioCalls(page).length, seen);
});

test("the state poll's own timer does not stop the audio poll", async (t) => {
  const page = await livePage(t, undefined, { "GET /api/live/state": { ...STATE, poll_ms: 20 } });
  await openLive(page);
  await sleep(POLL_MS * 20);
  const stateCalls = page.calls.filter((c) => c.path === "/api/live/state").length;
  assert.ok(stateCalls >= 5 && audioCalls(page).length >= 8, `${stateCalls} state, ${audioCalls(page).length} audio`);
});

test("live_disabled removes the audio cards and the poll comes back when it is switched on", async (t) => {
  let enabled = false;
  const page = await livePage(t, undefined, {
    "GET /api/live/audio": () => (enabled ? AUDIO : reply(404, { error: "live_disabled" })),
  });
  const timers = [];
  const real = page.window.setTimeout.bind(page.window);
  page.window.setTimeout = (fn, ms, ...rest) => { timers.push({ fn, ms }); return real(() => {}, 1e9); };
  await openLive(page);
  assert.equal(page.document.querySelector("#live-levels"), null);
  assert.equal(page.document.querySelector("#notice").hidden, true, "not an error banner");
  const retry = timers.filter((x) => x.ms === 5000).at(-1);
  assert.ok(retry, "retry armed at the slow delay");
  enabled = true;
  retry.fn();
  await settle();
  assert.ok(page.document.querySelector("#live-levels"));
});

test("an expired session on the audio poll goes to the login", async (t) => {
  let sessions = 0;
  const page = await livePage(t, undefined, {
    "GET /api/session": () => ({ authenticated: sessions++ === 0, csrf: "tok", pin_set: true, status_poll_ms: 1000 }),
    "GET /api/live/audio": reply(401, { error: "unauthorized" }),
  });
  await openLive(page);
  await settle();
  assert.equal(page.document.querySelector("#tabs").hidden, true);
});
