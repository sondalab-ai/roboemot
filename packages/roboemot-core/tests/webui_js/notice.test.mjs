// The page's feedback (SPEC 11A, 12): `notice()` shows one kit toast in the toast stack that is
// always in the page, a success in its polite region and an error in its alert region. Run with
// `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const toasts = (page) => [...page.document.querySelectorAll("#notice .sl-toast")];

test("both live regions are in the page, empty, before the first toast", async () => {
  const page = await loadPage();
  const regions = [...page.document.querySelectorAll("#notice > .sl-toast-stack__live")];
  assert.deepEqual(regions.map((r) => [r.getAttribute("aria-live"), r.getAttribute("role")]), [["polite", null], [null, "alert"]]);
  assert.ok(regions.every((r) => r.childNodes.length === 0));
  assert.ok(!page.document.querySelector("main").contains(page.document.getElementById("notice")), "not at the top of the page");
});

test("a success goes to the polite region, an error to the alert region, and \"\" clears", async () => {
  const page = await loadPage();
  page.run('notice("Saved.")');
  assert.equal(page.document.querySelector("#notice-polite .sl-toast--success").textContent, "Saved.");
  page.run('notice("Bad key.", true)');
  assert.equal(toasts(page).length, 1, "a later message replaces the toast");
  assert.equal(page.document.querySelector("#notice-alert .sl-toast--danger").textContent, "Bad key.");
  page.run('notice("")');
  assert.equal(toasts(page).length, 0);
});

test("an error equal to the last one is shown again: a new toast, even after a dismiss", async () => {
  const page = await loadPage();
  page.run('notice("Cannot reach the robot.", true)');
  const first = toasts(page)[0];
  page.run('notice("Cannot reach the robot.", true)');
  assert.equal(toasts(page).length, 1, "it replaces the toast");
  assert.notEqual(toasts(page)[0], first, "a new node: the alert region speaks again");
  toasts(page)[0].querySelector(".sl-toast__dismiss").click();
  assert.equal(toasts(page).length, 0);
  page.run('notice("Cannot reach the robot.", true)');
  assert.equal(toasts(page).length, 1, "a later action that fails the same way is seen");
});

test("a Save that fails like the dismissed poll failure still shows its error", async () => {
  const page = await loadPage({
    "GET /api/session": () => ({ authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 }),
    "GET /api/strings": { language: "en", strings: { err_unreachable: "Cannot reach the robot." } },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/live/state": () => { throw new TypeError("Failed to fetch"); },
    "GET /api/live/audio": () => { throw new TypeError("Failed to fetch"); },
  });
  page.document.querySelector('#tabs button[data-tab="live"]').click();
  await settle();
  assert.equal(toasts(page)[0].textContent, "Cannot reach the robot.");
  toasts(page)[0].querySelector(".sl-toast__dismiss").click();
  assert.equal(toasts(page).length, 0);
  page.window.fetch = async () => { throw new TypeError("Failed to fetch"); };
  await page.run('guarded(() => api("POST", "/api/settings", { path: "x", value: 1 }))');
  assert.equal(toasts(page).length, 1);
  assert.equal(toasts(page)[0].textContent, "Cannot reach the robot.");
});

test("a success leaves after its reading time, an error stays", async () => {
  const page = await loadPage();
  const timers = [];
  const real = page.window.setTimeout.bind(page.window);
  page.window.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return real(() => {}, 1e9); };
  page.run('notice("Bad key.", true)');
  assert.equal(timers.length, 0, "no timer for an error");
  page.run('notice("Saved.")');
  assert.deepEqual(timers.map((x) => x.ms), [8000]);
  timers[0].fn();
  assert.equal(toasts(page).length, 0);
});

test("a success the user has focused or points at waits for them", async () => {
  const page = await loadPage();
  const timers = [];
  const real = page.window.setTimeout.bind(page.window);
  page.window.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return real(() => {}, 1e9); };
  page.run('notice("Saved.")');
  const dismiss = page.document.querySelector("#notice .sl-toast__dismiss");
  dismiss.focus();
  timers.shift().fn();
  assert.equal(toasts(page).length, 1, "kept while it has the focus");
  assert.equal(page.document.activeElement, dismiss);
  assert.deepEqual(timers.map((x) => x.ms), [8000], "asked again later");
  dismiss.blur();
  timers.shift().fn();
  assert.equal(toasts(page).length, 0);
});

test("the dismiss button has an accessible name in the page language", async () => {
  const page = await loadPage({ "GET /api/strings": { language: "it", strings: { notice_dismiss: "Chiudi" } } });
  page.run('notice("Salvato.")');
  const dismiss = page.document.querySelector("#notice .sl-toast__dismiss");
  assert.equal(dismiss.getAttribute("aria-label"), "Chiudi");
  assert.equal(dismiss.textContent, "", "an icon only: the name is the label");
  dismiss.click();
  assert.equal(toasts(page).length, 0);
});
