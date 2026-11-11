// The tab strip (SPEC 11A, 12): manual activation as in the WAI-ARIA tabs pattern (arrows, Home and
// End move the focus only; a click, or Enter and Space through the native button, opens the tab),
// the tab in the URL fragment, the page title and the selected tab kept in view. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const SESSION = { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 };
const STRINGS = { title: "RoboEmot", tab_enroll: "Enrollment", tab_live: "Live", tab_debug: "Diagnostics", tabs_label: "Sections" };

function routes(extra = {}) {
  return {
    "GET /api/session": SESSION,
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/live/state": () => new Promise(() => {}),
    "GET /api/live/audio": () => new Promise(() => {}),
    "GET /api/debug": { sections: {} },
    ...extra,
  };
}

// Records scrollIntoView calls (jsdom has none) before the page scripts run.
const scrolls = [];
const before = (window) => {
  window.Element.prototype.scrollIntoView = function (opts) { scrolls.push({ node: this, opts }); };
};

const tabs = (page) => [...page.document.querySelectorAll("#tabs [role=tab]")];
const tab = (page, name) => page.document.querySelector(`#tabs [data-tab="${name}"]`);
const selected = (page) => page.document.querySelector('#tabs [aria-selected="true"]').dataset.tab;
const key = (page, k) => page.document.activeElement.dispatchEvent(new page.window.KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));
const viewCalls = (page) => page.calls.filter((c) => c.path !== "/api/live/viewers").length;

test("arrows, Home and End move the focus only; the roving tabindex follows it", async () => {
  const page = await loadPage(routes(), { before });
  tab(page, "enroll").focus();
  const calls = viewCalls(page);
  key(page, "ArrowRight");
  assert.equal(page.document.activeElement, tab(page, "registry"));
  key(page, "ArrowLeft");
  key(page, "ArrowLeft");
  assert.equal(page.document.activeElement, tabs(page).at(-1), "wraps to the last");
  key(page, "Home");
  assert.equal(page.document.activeElement, tabs(page)[0]);
  key(page, "End");
  assert.equal(page.document.activeElement, tab(page, "debug"));
  assert.deepEqual(tabs(page).map((b) => b.tabIndex), tabs(page).map((b) => (b.dataset.tab === "debug" ? 0 : -1)));
  for (let i = 0; i < 3; i++) key(page, "ArrowLeft"); // across Live: its polls must not start
  await settle();
  assert.equal(selected(page), "enroll", "no tab opened");
  assert.equal(viewCalls(page), calls, "no view was fetched");
  assert.equal(page.calls.filter((c) => c.path === "/api/live/state").length, 0);
});

test("a click opens the tab (Enter and Space reach it through the native button)", async () => {
  const page = await loadPage(routes(), { before });
  for (const b of tabs(page)) assert.equal(b.tagName + ":" + b.type, "BUTTON:button");
  tab(page, "debug").click();
  await settle();
  assert.equal(selected(page), "debug");
  assert.equal(tab(page, "debug").tabIndex, 0);
});

test("focus that leaves the strip puts the roving stop back on the selected tab", async () => {
  const page = await loadPage(routes(), { before });
  tab(page, "enroll").focus();
  key(page, "ArrowRight");
  page.document.getElementById("logout").focus();
  assert.equal(page.document.activeElement.id, "logout");
  assert.equal(tab(page, "enroll").tabIndex, 0);
  assert.equal(tab(page, "registry").tabIndex, -1);
});

test("the tab list has an accessible name", async () => {
  const page = await loadPage(routes(), { before });
  assert.equal(page.document.getElementById("tabs").getAttribute("aria-label"), "Sections");
});

test("the open tab is in the URL fragment, replaced, not pushed", async () => {
  const page = await loadPage(routes(), { before });
  const length = page.window.history.length;
  tab(page, "debug").click();
  await settle();
  assert.equal(page.window.location.hash, "#debug");
  tab(page, "enroll").click();
  assert.equal(page.window.location.hash, "#enroll");
  assert.equal(page.window.history.length, length, "no history entry per tab");
});

test("the fragment opens its tab at start, and an unknown one the default tab", async () => {
  const live = await loadPage(routes(), { before, url: "http://localhost/#debug" });
  await settle();
  assert.equal(selected(live), "debug");
  const unknown = await loadPage(routes(), { before, url: "http://localhost/#nope" });
  assert.equal(selected(unknown), "enroll");
  assert.equal(unknown.window.location.hash, "#enroll");
});

test("a fragment changed by hand or a bookmark shows its tab; nothing happens while logged out", async () => {
  const page = await loadPage(routes(), { before });
  page.window.location.hash = "#debug";
  page.window.dispatchEvent(new page.window.HashChangeEvent("hashchange"));
  await settle();
  assert.equal(selected(page), "debug");
  const out = await loadPage(routes({ "GET /api/session": { authenticated: false, pin_set: true } }), { before });
  out.window.location.hash = "#debug";
  out.window.dispatchEvent(new out.window.HashChangeEvent("hashchange"));
  await settle();
  assert.ok(out.document.querySelector("#pin"), "still the login");
});

test("the page title names the tab, then the product", async () => {
  const page = await loadPage(routes(), { before });
  assert.equal(page.document.title, "Enrollment · RoboEmot");
  tab(page, "debug").click();
  assert.equal(page.document.title, "Diagnostics · RoboEmot");
});

test("the selected tab is brought into view in the strip", async () => {
  scrolls.length = 0;
  const page = await loadPage(routes(), { before });
  tab(page, "debug").click();
  const last = scrolls.at(-1);
  assert.equal(last.node, tab(page, "debug"));
  assert.deepEqual({ ...last.opts }, { inline: "nearest", block: "nearest" });
});

test("after login the focus is on the selected tab", async () => {
  let authenticated = false;
  const page = await loadPage(routes({
    "GET /api/session": () => ({ ...SESSION, authenticated }),
    "POST /api/login": () => { authenticated = true; return { csrf: "tok" }; },
  }), { before });
  page.document.querySelector("#pin").value = "1234";
  page.document.querySelector("#pin").closest("form").dispatchEvent(new page.window.Event("submit", { cancelable: true }));
  await settle();
  await settle();
  assert.equal(page.document.activeElement, tab(page, "enroll"));
});
