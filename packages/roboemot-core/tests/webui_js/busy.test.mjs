// No double submit (SPEC 11A, 12): the forms that do not go through the PIN dialog (login, consent,
// catalogue import, secret set) and the route check button are disabled and busy while their
// request runs, so a second tap on slow Wi-Fi sends nothing. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const SESSION = { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 };
const BASE = {
  "GET /api/session": SESSION,
  "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
  "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
};

// A route answer that waits until `open()` is called.
function gate() {
  let open;
  const until = new Promise((resolve) => { open = resolve; });
  return { open, wait: (answer) => async () => { await until; return answer; } };
}

const posts = (page, path) => page.calls.filter((c) => c.method === "POST" && c.path === path).length;

test("login: a second press while the first request runs sends nothing; a failure gives the button back, focused", async () => {
  const g = gate();
  const page = await loadPage({
    "GET /api/session": { authenticated: false, pin_set: true },
    "POST /api/login": g.wait(reply(403, { error: "bad_pin" })),
  });
  const submit = page.document.querySelector("#view button[type=submit]");
  page.document.querySelector("#pin").value = "1234";
  submit.focus();
  submit.click();
  await settle();
  assert.equal(submit.disabled, true);
  assert.equal(submit.getAttribute("aria-busy"), "true");
  submit.click();
  submit.closest("form").dispatchEvent(new page.window.Event("submit", { cancelable: true })); // Enter in the field
  await settle();
  assert.equal(posts(page, "/api/login"), 1);
  g.open();
  await settle();
  await settle();
  assert.equal(submit.disabled, false);
  assert.equal(submit.getAttribute("aria-busy"), null);
  assert.equal(page.document.activeElement, submit, "the focus stays on the button");
});

async function routesPage(extra) {
  const page = await loadPage({
    ...BASE,
    "GET /api/routes": { routes: [{ name: "cloud", model: "m", endpoint: "h", locality: "cloud", kind: "chat", key: "none", removable: false, running: false }], name_max_chars: 32 },
    "GET /api/secrets": { secrets: [], max_name_chars: 32, max_value_chars: 64 },
    ...extra,
  });
  page.document.querySelector('#tabs [data-tab="routes"]').click();
  await settle();
  await settle();
  return page;
}

test("route check: one request per press while it runs, and the button is active again after success", async () => {
  const g = gate();
  const page = await routesPage({ "POST /api/routes/cloud/check": g.wait({}) });
  const check = [...page.document.querySelectorAll("#view button")].find((b) => b.textContent === "route_check");
  check.click();
  await settle();
  assert.equal(check.disabled, true);
  check.click();
  assert.equal(posts(page, "/api/routes/cloud/check"), 1);
  g.open();
  await settle();
  await settle();
  assert.equal(check.disabled, false, "given back, even though the view was redrawn");
  assert.equal(check.getAttribute("aria-busy"), null);
});

test("route check: a button that starts disabled (a check running) stays disabled", async () => {
  const page = await loadPage({ ...BASE });
  page.run("window.__b = button('x', 'plain', null, { attrs: { disabled: true } })");
  await page.run("whileBusy(window.__b, async () => {})");
  assert.equal(page.window.__b.disabled, true);
});

test("secret set: a second submit sends nothing, and a failure gives the button back", async () => {
  const g = gate();
  const page = await routesPage({ "POST /api/secrets/set": g.wait(reply(403, { error: "bad_pin" })) });
  const form = page.document.querySelector("#secret-name").closest("form");
  const submit = form.querySelector("button[type=submit]");
  page.document.querySelector("#secret-name").value = "k";
  page.document.querySelector("#secret-value").value = "v";
  page.document.querySelector("#secret-pin").value = "1";
  submit.click();
  await settle();
  form.dispatchEvent(new page.window.Event("submit", { cancelable: true }));
  await settle();
  assert.equal(posts(page, "/api/secrets/set"), 1);
  g.open();
  await settle();
  await settle();
  assert.equal(submit.disabled, false);
});

test("catalogue import: a second submit sends nothing while the import runs", async () => {
  const g = gate();
  const page = await loadPage({
    ...BASE,
    "GET /api/catalogue": { records: [], live: [], limits: { max_xmp_bytes: 1e6, max_image_bytes: 1e6, max_items: 10 }, tombstones: 0 },
    "POST /api/catalogue/import": g.wait(reply(422, { error: "xmp_invalid" })),
  });
  page.document.querySelector('#tabs [data-tab="catalogue"]').click();
  await settle();
  await settle();
  const xmp = page.document.querySelector("#xmp_file");
  Object.defineProperty(xmp, "files", { value: [{ size: 3, text: async () => "<x/>" }] });
  const form = xmp.closest("form");
  const submit = form.querySelector("button[type=submit]");
  submit.click();
  await settle();
  await settle();
  form.dispatchEvent(new page.window.Event("submit", { cancelable: true }));
  await settle();
  assert.equal(posts(page, "/api/catalogue/import"), 1);
  g.open();
  await settle();
  await settle();
  assert.equal(submit.disabled, false);
});

test("consent: a second submit sends nothing while the enrollment completes", async () => {
  const g = gate();
  const page = await loadPage({
    ...BASE,
    "GET /api/enrollment/status": { state: "ready", frames_collected: 5, frames_needed: 5, can_complete: true, hints: [] },
    "GET /api/registry": { records: [] },
    "POST /api/enrollment/complete": g.wait(reply(400, { error: "consent_required" })),
  });
  await settle();
  await settle();
  const form = page.document.querySelector("#display_name").closest("form");
  const submit = form.querySelector("button[type=submit]");
  page.document.querySelector("#display_name").value = "Mia";
  page.document.querySelector("#consent_who").value = "Mia";
  submit.click();
  await settle();
  assert.equal(submit.disabled, true);
  form.dispatchEvent(new page.window.Event("submit", { cancelable: true }));
  await settle();
  assert.equal(posts(page, "/api/enrollment/complete"), 1);
  g.open();
  await settle();
  await settle();
  assert.equal(submit.disabled, false);
  page.run("stopPolling()");
});
