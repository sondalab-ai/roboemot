// Behaviour of the Live tab (SPEC 11A, 12): it polls /api/live/state only while the tab is open and
// the page is visible, stops on leaving the tab, hiding the page and logging out, and shows how to
// turn the feature on when the server says it is off. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const POLL_MS = 15;
const WAIT_MS = POLL_MS * 8;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const STATE = {
  poll_ms: POLL_MS,
  mood: { label: "joy", valence: 0.25, arousal: -0.1, intensity: 0.4, circadian: { state: "awake", sleepiness: 0.2 } },
  modes: { asleep: false, paused: [], online: true, minor_present: false, register: "adult", persona: "default" },
  conversation: { phase: "thinking", route: "lan", elapsed_s: 1.234 },
  attention: { count: 2, faces: [{ name: "Mia", target: true }, { name: null, target: false }] },
  reflexes: { last: { name: "startle", age_s: 2.5 } },
  motion: { state: "writing" },
  system: { cpu_temp_c: 51.2 },
  video: { enabled: false, poll_ms: 333 },
};

const AUDIO = {
  audio_poll_ms: POLL_MS * 4,
  levels: { mic: { state: "ok", db: -30 }, out: { state: "ok", db: -60 }, floor_db: null },
  transcript: { enabled: false, epoch: "", rev: 0, min_seq: 0, max_turns: 0, turns: [] },
};

const STRINGS = {
  live_reflex_last: "Last: {name}, {n} s ago",
  live_cpu_temp: "CPU temperature: {n} C",
  live_people: "People: {n}",
  live_elapsed: "Waiting for {n} s",
};

async function livePage(t, extra = {}) {
  let loggedIn = true;
  const page = await loadPage({
    "GET /api/session": () => ({ authenticated: loggedIn, csrf: "tok", pin_set: true, status_poll_ms: 1000 }),
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/state": STATE,
    "GET /api/live/audio": AUDIO,
    "POST /api/logout": () => { loggedIn = false; return { ok: true }; },
    ...extra,
  });
  t.after(() => page.window.close()); // jsdom timers would keep the runner alive
  return page;
}

const polls = (page) => page.calls.filter((c) => c.path === "/api/live/state").length;
const openLive = async (page) => {
  page.document.querySelector('#tabs button[data-tab="live"]').click();
  await settle();
};
const visibility = (page, value) => {
  Object.defineProperty(page.document, "visibilityState", { configurable: true, get: () => value });
  page.document.dispatchEvent(new page.window.Event("visibilitychange"));
};

test("the Live tab sits next to the others and draws one card per group", async (t) => {
  const page = await livePage(t);
  const tabs = [...page.document.querySelectorAll("#tabs button")].map((b) => b.dataset.tab);
  assert.deepEqual(tabs, ["enroll", "registry", "catalogue", "routes", "settings", "live", "debug"]);
  await openLive(page);
  const grid = page.document.querySelector("#live-cards");
  assert.equal(grid.children.length, 8);
  const text = grid.textContent;
  for (const expected of ["joy", "0.25", "lan", "Mia", "startle", "writing", "51.2", "adult"]) {
    assert.ok(text.includes(expected) || text.toLowerCase().includes(expected), expected);
  }
  assert.equal(page.document.querySelector("#notice").hidden, true);
});

test("it polls at the interval the server names, and not before the tab is opened", async (t) => {
  const page = await livePage(t);
  await sleep(WAIT_MS);
  assert.equal(polls(page), 0);
  await openLive(page);
  await sleep(WAIT_MS);
  assert.ok(polls(page) >= 3, `polled ${polls(page)} times`);
});

test("leaving the tab stops the polling", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await settle();
  const seen = polls(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), seen);
});

test("a hidden page stops the polling and a visible one resumes it", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  visibility(page, "hidden");
  await settle();
  const seen = polls(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), seen, "no request while hidden");
  visibility(page, "visible");
  await sleep(WAIT_MS);
  assert.ok(polls(page) >= seen + 2, "resumed");
});

test("a page that is hidden when the tab opens makes no request until it is visible", async (t) => {
  const page = await livePage(t);
  visibility(page, "hidden");
  await openLive(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), 0);
  visibility(page, "visible");
  await sleep(WAIT_MS);
  assert.ok(polls(page) >= 1);
});

test("a reply that arrives after the page was hidden neither draws nor re-arms the timer", async (t) => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const page = await livePage(t, { "GET /api/live/state": async () => { await gate; return STATE; } });
  await openLive(page);
  assert.equal(polls(page), 1);
  visibility(page, "hidden");
  release();
  await sleep(WAIT_MS);
  assert.equal(polls(page), 1);
  assert.equal(page.document.querySelector("#live-cards").children.length, 0, "nothing drawn from the retired request");
});

test("a reply that arrives after the tab was left neither draws nor re-arms the timer", async (t) => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const page = await livePage(t, { "GET /api/live/state": async () => { await gate; return STATE; } });
  await openLive(page);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await settle();
  release();
  await sleep(WAIT_MS);
  assert.equal(polls(page), 1);
});

test("logging out stops the polling", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  page.document.querySelector("#logout").click();
  await settle();
  assert.equal(page.document.querySelector("#tabs").hidden, true, "back at the login");
  await sleep(POLL_MS * 2);
  const seen = polls(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), seen);
  assert.ok(page.calls.some((c) => c.path === "/api/logout"));
});

test("a re-opened tab runs one timer, not one per visit", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await openLive(page);
  const before = polls(page);
  await sleep(POLL_MS * 10);
  const rate = (polls(page) - before) / 10;
  assert.ok(rate <= 1.6, `about one request per interval, got ${rate}`);
});

test("live_disabled shows how to turn it on and keeps retrying at the slow delay", async (t) => {
  let enabled = false;
  const page = await livePage(t, {
    "GET /api/live/state": () => (enabled ? STATE : reply(404, { error: "live_disabled" })),
    "GET /api/live/audio": () => (enabled ? AUDIO : reply(404, { error: "live_disabled" })),
  });
  const timers = [];
  const real = page.window.setTimeout.bind(page.window);
  page.window.setTimeout = (fn, ms, ...rest) => { timers.push({ fn, ms }); return real(() => {}, 1e9); };
  await openLive(page);
  assert.equal(polls(page), 1);
  assert.ok(page.document.querySelector("#live-cards").textContent.includes("live_disabled_body"));
  assert.equal(page.document.querySelector("#notice").hidden, true, "not an error banner");
  const retries = timers.filter((x) => x.ms === 5000);
  assert.ok(retries.length >= 1, "retry armed at the slow delay");
  enabled = true;
  retries.forEach((x) => x.fn());
  await settle();
  assert.equal(page.document.querySelector("#live-cards").children.length, 8, "self-healed");
});

test("coming back into view keeps the last cards until the next reply", async (t) => {
  let release = null;
  let hold = false;
  const page = await livePage(t, {
    "GET /api/live/state": async () => { if (hold) await new Promise((r) => { release = r; }); return STATE; },
  });
  await openLive(page);
  await sleep(POLL_MS * 2);
  visibility(page, "hidden");
  hold = true;
  visibility(page, "visible");
  await settle();
  assert.equal(page.document.querySelector("#live-cards").children.length, 8, "cards kept while the poll is pending");
  release();
  await settle();
  visibility(page, "hidden");
  await settle();
});

test("a failed poll shows the error and retries", async (t) => {
  let n = 0;
  const page = await livePage(t, {
    "GET /api/live/state": () => (n++ === 0 ? reply(500, { error: "generic" }) : STATE),
  });
  await openLive(page);
  assert.equal(page.document.querySelector("#notice").hidden, false);
  assert.equal(polls(page), 1);
});

test("an expired session goes back to the login", async (t) => {
  let expired = false;
  const page = await livePage(t, {
    "GET /api/session": () => ({ authenticated: !expired, csrf: "tok", pin_set: true }),
    "GET /api/live/state": () => { expired = true; return reply(401, { error: "unauthorized" }); },
  });
  await openLive(page);
  await sleep(POLL_MS * 2);
  assert.equal(page.document.querySelector("#tabs").hidden, true);
  const seen = polls(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), seen, "no polling at the login");
});

test("everything the state holds goes through textContent", async (t) => {
  const hostile = { ...STATE, attention: { count: 1, faces: [{ name: "<img src=x onerror=alert(1)>", target: false }] } };
  const page = await livePage(t, { "GET /api/live/state": hostile });
  await openLive(page);
  const grid = page.document.querySelector("#live-cards");
  const images = [...grid.querySelectorAll("img")];
  assert.ok(images.every((i) => i.id === "live-video-img" && !i.hasAttribute("onerror")), "only the preview <img>");
  assert.ok(grid.textContent.includes("<img src=x"));
});
