// Loading and retry (SPEC 11A, 12A): a tab that waits for its first answer says so, a tab that draws
// at once does not flash it, and a page that cannot load offers to try again. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const SESSION = { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 };
const ENROLL = { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] };

async function shellPage(routes) {
  return loadPage({
    "GET /api/session": SESSION,
    "GET /api/strings": { language: "en", strings: { loading: "Loading..." } },
    "GET /api/enrollment/status": ENROLL,
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    ...routes,
  });
}

const line = (page) => page.document.querySelector("#view-loading");
const view = (page) => page.document.querySelector("#view");
const open = (page, tab) => page.document.querySelector(`#tabs button[data-tab="${tab}"]`).click();

test("a tab that waits for its answer shows the loading line and a busy view until it draws", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const page = await shellPage({
    "GET /api/debug": async () => { await gate; return { sections: {} }; },
  });
  open(page, "debug");
  assert.equal(line(page).hidden, false);
  assert.equal(line(page).textContent, "Loading...");
  assert.equal(view(page).getAttribute("aria-busy"), "true");
  release();
  await settle();
  await settle();
  assert.ok(view(page).hasChildNodes(), "the view drew");
  assert.equal(line(page).hidden, true);
  assert.equal(line(page).textContent, "");
  assert.equal(view(page).getAttribute("aria-busy"), null);
});

test("a tab that draws at once never shows the loading line", async () => {
  const page = await shellPage({
    "GET /api/live/state": () => new Promise(() => {}),
    "GET /api/live/audio": () => new Promise(() => {}),
  });
  const shown = [];
  new page.window.MutationObserver(() => shown.push(line(page).hidden)).observe(line(page), { attributes: true });
  open(page, "live");
  assert.equal(line(page).hidden, true);
  await settle();
  assert.deepEqual(shown, [], "not even for a moment");
  assert.equal(view(page).getAttribute("aria-busy"), null);
});

test("a tab whose request fails leaves no loading line behind", async () => {
  const page = await shellPage({ "GET /api/debug": reply(500, { error: "generic" }) });
  open(page, "debug");
  await settle();
  await settle();
  assert.equal(line(page).hidden, true);
  assert.equal(view(page).getAttribute("aria-busy"), null);
});

test("a view that fails before it draws shows the error in the view with a retry that loads it", async () => {
  let down = true;
  const page = await shellPage({
    "GET /api/strings": { language: "en", strings: { loading: "Loading...", retry: "Try again", err_settings_unavailable: "Settings are not available." } },
    "GET /api/settings": () => (down ? reply(503, { error: "settings_unavailable" }) : { fields: [], pin_required: true }),
  });
  open(page, "settings");
  await settle();
  await settle();
  const failed = view(page).querySelector("#view-failed");
  assert.ok(failed && failed.classList.contains("sl-card"), "a kit card in the view");
  assert.ok(failed.textContent.includes("Settings are not available."));
  // shown once (the card), announced once: the alert region holds the text for assistive technology only
  assert.equal(page.document.querySelectorAll("#notice .sl-toast").length, 0, "no toast repeats the card");
  const alert = page.document.getElementById("notice-alert");
  assert.equal(alert.textContent, "Settings are not available.");
  assert.ok(alert.firstElementChild.classList.contains("sr-only"));
  assert.equal(alert.querySelector("button"), null, "nothing focusable that nobody sees");
  const retry = failed.querySelector("button");
  assert.equal(retry.textContent, "Try again");
  down = false;
  retry.click();
  await settle();
  await settle();
  assert.equal(view(page).querySelector("#view-failed"), null);
  assert.ok(view(page).querySelector("#settings-search"), "the view drew");
  assert.equal(page.calls.filter((c) => c.path === "/api/settings").length, 2);
});

test("a view that drew and then failed keeps its content", async () => {
  const page = await shellPage({});
  await page.run("guardedView(async () => { newView(); document.getElementById('view').append(el('p', { id: 'drawn' }, 'x')); throw new Error('generic'); })");
  assert.ok(view(page).querySelector("#drawn"), "the content stays");
  assert.equal(view(page).querySelector("#view-failed"), null);
  assert.equal(page.document.querySelectorAll("#notice-alert .sl-toast").length, 1, "with no card, the toast says it");
});

test("a session that expired while a view loads goes to the login, not to the error card", async () => {
  let authenticated = true;
  const page = await shellPage({
    "GET /api/session": () => ({ ...SESSION, authenticated }),
    "GET /api/debug": () => { authenticated = false; return reply(401, { error: "unauthorized" }); },
  });
  open(page, "debug");
  await settle();
  await settle();
  assert.equal(view(page).querySelector("#view-failed"), null);
  assert.ok(page.document.querySelector("#pin"), "the login is drawn");
});

test("a later tab keeps its own loading line when the earlier one answers", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const page = await shellPage({
    "GET /api/debug": async () => { await gate; return { sections: {} }; },
    "GET /api/catalogue": () => new Promise(() => {}),
  });
  open(page, "debug");
  open(page, "catalogue");
  release();
  await settle();
  await settle();
  assert.equal(line(page).hidden, false, "catalogue is still waiting");
});

test("a page that cannot load offers to try again, and the retry loads it", async () => {
  let down = true;
  const page = await loadPage({
    "GET /api/strings": () => { if (down) throw new TypeError("Failed to fetch"); return { language: "en", strings: {} }; },
    "GET /api/session": { authenticated: false, pin_set: true },
  });
  const retry = page.document.querySelector("#boot-retry");
  assert.equal(retry.textContent, "Try again");
  assert.equal(page.document.querySelector("#notice").textContent, "Cannot reach the robot.");
  retry.click();
  await settle();
  assert.equal(page.document.querySelector("#notice").textContent, "Cannot reach the robot.", "still down: said again");
  down = false;
  page.document.querySelector("#boot-retry").click();
  await settle();
  assert.ok(page.document.querySelector("#pin"), "the login is drawn");
  assert.equal(page.document.querySelector("#boot-retry"), null);
  assert.equal(page.document.querySelector("#notice").textContent, "");
});
