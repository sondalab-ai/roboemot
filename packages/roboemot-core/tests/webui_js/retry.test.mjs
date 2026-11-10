// Polls outside the Live tab keep going (SPEC 11A, 12): the enrollment status poll and the routes
// poll (while a route check runs) retry a failed request at the slow delay, mark the cards on screen
// as old, take back their toast when a reply comes, and stop when the tab is left. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const SLOW_MS = 5000; // LIVE_RETRY_MS in live.js
const STRINGS = { err_generic: "Something failed.", stale_since: "Last updated at {at}." };

// The page's timers, held instead of run: `fire(ms)` runs the newest one armed with that delay.
function heldTimers(window) {
  const held = [];
  const real = window.setTimeout.bind(window);
  window.setTimeout = (fn, ms) => { held.push({ fn, ms }); return real(() => {}, 1e9); };
  return {
    armed: (ms) => held.filter((x) => x.ms === ms).length,
    async fire(ms) {
      const at = held.map((x) => x.ms).lastIndexOf(ms);
      assert.ok(at >= 0, "a timer of " + ms + " ms is armed");
      const [{ fn }] = held.splice(at, 1);
      await fn();
      await settle();
      await settle();
    },
  };
}

// Answers from `seq` in turn, the last one again once they run out; a function answer is called.
function sequence(seq) {
  let n = 0;
  return () => {
    const next = seq[Math.min(n++, seq.length - 1)];
    return typeof next === "function" ? next() : next;
  };
}

const alertText = (page) => page.document.querySelector("#notice-alert").textContent;

test("enrollment: a failed status request is retried at the slow delay and the cards are marked as old", async () => {
  let timers = null;
  const collecting = { state: "collecting", frames_collected: 2, frames_needed: 5, can_complete: false, hints: [] };
  const statusCalls = () => page.calls.filter((c) => c.path === "/api/enrollment/status").length;
  const page = await loadPage({
    "GET /api/session": { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 50 },
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/enrollment/status": sequence([collecting, reply(503, { error: "generic" }), reply(503, { error: "generic" }), collecting]),
  }, { before: (w) => { timers = heldTimers(w); } });
  await settle();
  const box = page.document.querySelector("#enroll-cards");
  const note = page.document.querySelector("#enroll-stale");
  assert.equal(statusCalls(), 1);
  assert.ok(box.textContent.includes("2 / 5"));

  await timers.fire(50);
  assert.equal(statusCalls(), 2);
  assert.equal(alertText(page), "Something failed.", "the failure is reported once");
  assert.equal(box.getAttribute("aria-busy"), "true", "the cards on screen are old");
  assert.equal(note.hidden, false);
  assert.ok(note.textContent.startsWith("Last updated at"), note.textContent);
  assert.ok(box.textContent.includes("2 / 5"), "the last cards stay");
  assert.equal(timers.armed(SLOW_MS), 1, "retried at the slow delay");

  page.document.querySelector("#notice-alert").replaceChildren(); // dismissed by the user
  await timers.fire(SLOW_MS);
  assert.equal(statusCalls(), 3);
  assert.equal(alertText(page), "", "a failure that goes on is not reported again");
  assert.equal(timers.armed(SLOW_MS), 1);

  page.run("notice('Something failed.', true)");
  await timers.fire(SLOW_MS);
  assert.equal(statusCalls(), 4);
  assert.equal(box.getAttribute("aria-busy"), null, "current again");
  assert.equal(note.hidden, true);
  assert.equal(alertText(page), "", "the poll takes back its own toast");
  assert.equal(timers.armed(50), 1, "back at the normal rate");
  page.run("stopPolling()");
});

test("enrollment: a retry armed before the tab was left sends nothing", async () => {
  let timers = null;
  const page = await loadPage({
    "GET /api/session": { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 50 },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/enrollment/status": reply(503, { error: "generic" }),
    "GET /api/registry": { records: [] },
    "GET /api/registry/quarantined": { keys: [] },
  }, { before: (w) => { timers = heldTimers(w); } });
  await settle();
  assert.equal(timers.armed(SLOW_MS), 1, "the first failure is retried too");
  page.document.querySelector('#tabs [data-tab="registry"]').click();
  await settle();
  const before = page.calls.filter((c) => c.path === "/api/enrollment/status").length;
  await timers.fire(SLOW_MS);
  assert.equal(page.calls.filter((c) => c.path === "/api/enrollment/status").length, before);
});

const ROUTE = { name: "cloud", model: "m", endpoint: "h", locality: "cloud", kind: "chat", key: "none", removable: false };

test("routes: a failed poll while a check runs is retried at the slow delay, keeps the cards and recovers", async () => {
  let timers = null;
  const running = { routes: [{ ...ROUTE, running: true }], name_max_chars: 32 };
  const done = { routes: [{ ...ROUTE, running: false }], name_max_chars: 32 };
  const page = await loadPage({
    "GET /api/session": { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 },
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/routes": sequence([running, () => { throw new TypeError("Failed to fetch"); }, running, done]),
    "GET /api/secrets": { secrets: [], max_name_chars: 32, max_value_chars: 64 },
  }, { before: (w) => { timers = heldTimers(w); } });
  page.document.querySelector('#tabs [data-tab="routes"]').click();
  await settle();
  await settle();
  const grid = page.document.querySelector("#routes-cards");
  assert.ok(grid, "the route cards are drawn");
  assert.equal(timers.armed(2000), 1, "the poll runs while the check runs");

  await timers.fire(2000);
  assert.equal(page.document.querySelector("#routes-cards"), grid, "the cards stay on screen");
  assert.equal(grid.getAttribute("aria-busy"), "true");
  assert.equal(page.document.querySelector("#routes-stale").hidden, false);
  assert.equal(page.document.querySelector("#view-failed"), null, "no failed-view card over the cards");
  assert.ok(alertText(page).length > 0, "the failure is reported");
  assert.equal(timers.armed(SLOW_MS), 1, "retried at the slow delay");

  await timers.fire(SLOW_MS);
  const fresh = page.document.querySelector("#routes-cards");
  assert.notEqual(fresh, grid, "redrawn");
  assert.equal(fresh.getAttribute("aria-busy"), null);
  assert.equal(page.document.querySelector("#routes-stale").hidden, true);
  assert.equal(alertText(page), "", "the poll takes back its own toast");
  assert.equal(timers.armed(2000), 1, "still polling while the check runs");

  await timers.fire(2000);
  assert.equal(timers.armed(2000), 0, "the check is over: no more polls");
  assert.equal(timers.armed(SLOW_MS), 0);
});

test("routes: a retry armed before the tab was left sends nothing", async () => {
  let timers = null;
  const page = await loadPage({
    "GET /api/session": { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/routes": sequence([{ routes: [{ ...ROUTE, running: true }], name_max_chars: 32 }, reply(503, { error: "generic" })]),
    "GET /api/secrets": { secrets: [], max_name_chars: 32, max_value_chars: 64 },
  }, { before: (w) => { timers = heldTimers(w); } });
  page.document.querySelector('#tabs [data-tab="routes"]').click();
  await settle();
  await settle();
  await timers.fire(2000);
  assert.equal(timers.armed(SLOW_MS), 1);
  page.document.querySelector('#tabs [data-tab="enroll"]').click();
  await settle();
  const before = page.calls.filter((c) => c.path === "/api/routes").length;
  await timers.fire(SLOW_MS);
  assert.equal(page.calls.filter((c) => c.path === "/api/routes").length, before);
});
