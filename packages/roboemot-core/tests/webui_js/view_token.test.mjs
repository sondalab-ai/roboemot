// A tab view whose API call finishes after the user left the tab draws nothing (SPEC 11A): only the
// latest view owns #view. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

/** Runs every pending promise chain of the page: the stubbed API answers without real time. */
async function flush() {
  for (let i = 0; i < 10; i++) await settle();
}

/** A route reply held until `open()` is called. */
function gate(body) {
  let open;
  const held = new Promise((resolve) => { open = resolve; });
  return { open, route: async () => { await held; return typeof body === "function" ? body() : body; } };
}

const FAST = {
  "GET /api/registry": { records: [] },
  "GET /api/registry/quarantined": { keys: [] },
  "GET /api/catalogue": { records: [], live: [], tombstones: 0, limits: { max_xmp_bytes: 1, max_image_bytes: 1 } },
  "GET /api/routes": { routes: [] },
  "GET /api/secrets": { secrets: [], max_value_chars: 64 },
  "GET /api/debug": { sections: {} },
  "GET /api/settings": { fields: [], pin_required: true },
};

const RUNNING = { name: "cloud", model: "m", endpoint: "api.example.com", locality: "cloud", kind: "chat", key: "none", removable: false, running: true, last_check: null };

// Each view with the route that holds it, a title it draws, and another tab to switch to while it waits.
const CASES = [
  { view: "registry", path: "GET /api/registry", title: "registry_title", other: "debug" },
  { view: "catalogue", path: "GET /api/catalogue", title: "catalogue_intro", other: "debug" },
  { view: "routes", path: "GET /api/routes", title: "routes_intro", other: "debug" },
  { view: "debug", path: "GET /api/debug", title: "debug_title", other: "registry" },
  { view: "settings", path: "GET /api/settings", title: "settings_intro", other: "debug" },
];

/** Opens `view`, switches to `other` while the held answer waits, then lets the answer through. */
async function dropView(c, routes = {}, answer = FAST[c.path]) {
  const held = gate(answer);
  const page = await loadPage({ ...FAST, ...routes, [c.path]: held.route });
  page.run(`showTab("${c.view}")`);
  await settle();
  page.run(`showTab("${c.other}")`);
  await flush();
  const drawn = page.document.querySelector("#view").innerHTML;
  assert.ok(drawn.length > 0, "the other tab drew");
  held.open();
  await flush();
  return { page, drawn };
}

for (const c of CASES) {
  test(`${c.view}: an answer that arrives after a tab switch does not draw over the new tab`, async () => {
    const { page, drawn } = await dropView(c);
    assert.equal(page.document.querySelector("#view").innerHTML, drawn);
    assert.equal(page.run("state.timer"), null, "no poll of the old view");
  });

  test(`${c.view}: an answer that arrives while the tab is still open draws it`, async () => {
    const held = gate(FAST[c.path]);
    const page = await loadPage({ ...FAST, [c.path]: held.route });
    page.run(`showTab("${c.view}")`);
    await settle();
    held.open();
    await flush();
    assert.equal(page.document.querySelector("#view").textContent.split(c.title).length - 1, 1);
  });
}

for (const c of CASES.filter((x) => ["registry", "debug"].includes(x.view))) {
  test(`${c.view}: an error of a view dropped by a tab switch shows no notice`, async () => {
    const { page, drawn } = await dropView(c, {}, reply(500, { error: "generic" }));
    assert.equal(page.document.querySelector("#notice").hidden, true);
    assert.equal(page.document.querySelector("#view").innerHTML, drawn);
  });
}

test("routes: a secret store error that arrives after a tab switch shows no notice", async () => {
  const c = { view: "routes", path: "GET /api/secrets", other: "debug" };
  const { page, drawn } = await dropView(c, {}, reply(500, { error: "secret_store" }));
  assert.equal(page.document.querySelector("#notice").hidden, true);
  assert.equal(page.document.querySelector("#view").innerHTML, drawn);
});

test("routes: a call dropped after its second answer draws nothing and arms no poll", async () => {
  const c = { view: "routes", path: "GET /api/secrets", other: "debug" };
  const { page, drawn } = await dropView(c, { "GET /api/routes": { routes: [RUNNING] } });
  assert.equal(page.document.querySelector("#view").innerHTML, drawn);
  assert.equal(page.run("state.timer"), null);
});

test("routes: a dropped call with a running check arms no poll", async () => {
  const c = { view: "routes", path: "GET /api/routes", other: "debug" };
  const { page } = await dropView(c, {}, { routes: [RUNNING] });
  assert.equal(page.run("state.timer"), null);
});

test("routes: a running check on the open tab arms one poll", async () => {
  const page = await loadPage({ ...FAST, "GET /api/routes": { routes: [RUNNING] } });
  page.run('showTab("routes")');
  await flush();
  assert.notEqual(page.run("state.timer"), null);
  page.run("stopPolling()");
});

test("two overlapping calls of one view draw one copy", async () => {
  const held = gate(FAST["GET /api/registry"]);
  const page = await loadPage({ ...FAST, "GET /api/registry": held.route });
  page.run("registryView(); registryView();");
  await settle();
  held.open();
  await flush();
  const text = page.document.querySelector("#view").textContent;
  assert.equal(text.split("registry_title").length - 1, 1);
});

test("logout drops a view still waiting for its answer", async () => {
  const held = gate(FAST["GET /api/registry"]);
  const page = await loadPage({ ...FAST, "GET /api/registry": held.route });
  page.run("registryView(); resetSettingsState();");
  held.open();
  await flush();
  assert.equal(page.document.querySelector("#view").children.length, 0);
});

test("the login view drops a view still waiting for its answer", async () => {
  const held = gate(FAST["GET /api/debug"]);
  const page = await loadPage({ ...FAST, "GET /api/debug": held.route });
  page.run("debugView(); loginView();");
  const drawn = page.document.querySelector("#view").innerHTML;
  held.open();
  await flush();
  assert.equal(page.document.querySelector("#view").innerHTML, drawn);
});

/** A settings page whose second catalogue answer (the reload) is held. */
async function settingsReloadPage() {
  let calls = 0;
  const held = gate({ fields: [], pin_required: true, reloaded: true });
  const page = await loadPage({
    ...FAST,
    "GET /api/settings": () => (++calls === 2 ? held.route() : { fields: [], pin_required: true }),
  });
  await page.run("settingsView()");
  page.run("settingsReload(null).catch((e) => { window.__err = String(e); })");
  await settle();
  return { page, held };
}

test("a settings reload answered after a logout does not apply to the new state", async () => {
  const { page, held } = await settingsReloadPage();
  page.run("resetSettingsState()");
  held.open();
  await flush();
  assert.equal(page.run("settingsState.data"), null);
  assert.equal(page.run("window.__err"), undefined);
});

test("a settings reload answered after a new settings view does not apply to it", async () => {
  const { page, held } = await settingsReloadPage();
  page.run("settingsView()");
  await flush();
  held.open();
  await flush();
  assert.equal(page.run("settingsState.data.reloaded"), undefined);
  assert.equal(page.run("window.__err"), undefined);
});

test("a settings reload on the current settings view applies its answer", async () => {
  const { page, held } = await settingsReloadPage();
  held.open();
  await flush();
  assert.equal(page.run("settingsState.data.reloaded"), true);
  assert.equal(page.run("window.__err"), undefined);
});

test("enrollment: the view reopened while an old status call waits keeps updating", async () => {
  // every answer counts one more frame, so the visible box shows whether the latest poll drew it
  let n = 0;
  const status = () => ({ state: "collecting", frames_collected: 100 + n++, frames_needed: 999, hints: [], can_complete: false });
  const first = gate(status);
  const second = gate(status);
  let third;
  const thirdServed = new Promise((resolve) => { third = resolve; });
  let calls = 0;
  const page = await loadPage({
    ...FAST,
    "GET /api/enrollment/status": () => {
      calls += 1;
      if (calls === 1) return first.route();
      if (calls === 2) return second.route();
      if (calls === 3) { third(); return status(); }
      return new Promise(() => {}); // later polls wait forever, so the box keeps the third answer
    },
  });
  page.run("state.session.status_poll_ms = 1");
  page.run('showTab("enroll")');
  await settle();
  page.run('showTab("debug")');
  await settle();
  page.run('showTab("enroll")');
  await settle();
  first.open();
  second.open();
  await thirdServed;
  await flush();
  assert.match(page.document.querySelector("#view").textContent, /frames: 102 \/ 999/);
  page.run("stopPolling()");
});

test("enrollment: a dropped view does not keep the consent form it was loading", async () => {
  const held = gate({ records: [] });
  const page = await loadPage({
    ...FAST,
    "GET /api/enrollment/status": { state: "ready", frames_collected: 1, frames_needed: 1, hints: [], can_complete: true },
    "GET /api/registry": held.route,
  });
  page.run("state.session.status_poll_ms = 60000");
  page.run('showTab("enroll")');
  await flush();
  page.run('showTab("debug")');
  await flush();
  held.open();
  await flush();
  assert.equal(page.run("state.form"), null);
  assert.equal(page.run("state.timer"), null);
});

test("catalogue: an import whose redraw is dropped by a tab switch shows no error", async () => {
  let calls = 0;
  const held = gate(FAST["GET /api/catalogue"]);
  const page = await loadPage({
    ...FAST,
    "GET /api/catalogue": () => (++calls === 1 ? FAST["GET /api/catalogue"] : held.route()),
    "POST /api/catalogue/import": { imported: [], skipped: [] },
  });
  page.run('showTab("catalogue")');
  await flush();
  const xmp = page.document.querySelector("#xmp_file");
  Object.defineProperty(xmp, "files", { value: [{ size: 1, text: async () => "<x/>" }] });
  xmp.closest("form").dispatchEvent(new page.window.Event("submit", { bubbles: true, cancelable: true }));
  await flush();
  assert.equal(calls, 2, "the import redraws the tab");
  page.run('showTab("debug")');
  await settle();
  held.open();
  await flush();
  assert.equal(page.document.querySelector("#notice").hidden, true);
  assert.equal(page.document.querySelector("#import_result"), null);
});
