// A tab view whose API call finishes after the user left the tab draws nothing (SPEC 11A): only the
// latest view owns #view. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

// Each view with the route that holds it, a title it draws, and another tab to switch to while it waits.
const CASES = [
  { view: "registry", path: "GET /api/registry", title: "registry_title", other: "debug" },
  { view: "catalogue", path: "GET /api/catalogue", title: "catalogue_intro", other: "debug" },
  { view: "routes", path: "GET /api/routes", title: "routes_intro", other: "debug" },
  { view: "debug", path: "GET /api/debug", title: "debug_title", other: "registry" },
  { view: "settings", path: "GET /api/settings", title: "settings_intro", other: "debug" },
];

for (const c of CASES) {
  test(`${c.view}: an answer that arrives after a tab switch does not draw over the new tab`, async () => {
    const held = gate(FAST[c.path]);
    const page = await loadPage({ ...FAST, [c.path]: held.route });
    page.run(`showTab("${c.view}")`);
    await settle();
    page.run(`showTab("${c.other}")`);
    await settle();
    const view = page.document.querySelector("#view");
    const drawn = view.innerHTML;
    assert.ok(drawn.length > 0, "the other tab drew");
    held.open();
    await sleep(20);
    assert.equal(view.innerHTML, drawn);
    assert.equal(page.run("state.timer"), null, "no poll of the old view");
  });

  test(`${c.view}: an answer that arrives while the tab is still open draws it`, async () => {
    const held = gate(FAST[c.path]);
    const page = await loadPage({ ...FAST, [c.path]: held.route });
    page.run(`showTab("${c.view}")`);
    await settle();
    held.open();
    await sleep(20);
    assert.equal(page.document.querySelector("#view").textContent.split(c.title).length - 1, 1);
  });
}

test("two overlapping calls of one view draw one copy", async () => {
  const held = gate(FAST["GET /api/registry"]);
  const page = await loadPage({ ...FAST, "GET /api/registry": held.route });
  page.run("registryView(); registryView();");
  await settle();
  held.open();
  await sleep(20);
  const text = page.document.querySelector("#view").textContent;
  assert.equal(text.split("registry_title").length - 1, 1);
});

test("logout drops a view still waiting for its answer", async () => {
  const held = gate(FAST["GET /api/registry"]);
  const page = await loadPage({ ...FAST, "GET /api/registry": held.route });
  page.run("registryView(); resetSettingsState();");
  held.open();
  await sleep(20);
  assert.equal(page.document.querySelector("#view").children.length, 0);
});

test("the login view drops a view still waiting for its answer", async () => {
  const held = gate(FAST["GET /api/debug"]);
  const page = await loadPage({ ...FAST, "GET /api/debug": held.route });
  page.run("debugView(); loginView();");
  const drawn = page.document.querySelector("#view").innerHTML;
  held.open();
  await sleep(20);
  assert.equal(page.document.querySelector("#view").innerHTML, drawn);
});

test("a settings reload answered after a logout does not apply to the new state", async () => {
  let calls = 0;
  const held = gate({ fields: [], pin_required: true });
  const page = await loadPage({
    ...FAST,
    "GET /api/settings": () => (++calls === 1 ? { fields: [], pin_required: true } : held.route()),
  });
  await page.run("settingsView()");
  page.run("settingsReload(null).catch((e) => { window.__err = String(e); })");
  await settle();
  page.run("resetSettingsState()");
  held.open();
  await sleep(20);
  assert.equal(page.run("settingsState.data"), null);
  assert.equal(page.run("window.__err"), undefined);
});

test("enrollment: the view reopened while an old status call waits keeps updating", async () => {
  // every answer counts one more frame, so the visible box shows whether the latest poll drew it
  let n = 0;
  const status = () => ({ state: "collecting", frames_collected: 100 + n++, frames_needed: 999, hints: [], can_complete: false });
  const first = gate(status);
  const second = gate(status);
  let calls = 0;
  const page = await loadPage({
    ...FAST,
    "GET /api/enrollment/status": () => {
      calls += 1;
      return calls === 1 ? first.route() : calls === 2 ? second.route() : status();
    },
  });
  page.run("state.session.status_poll_ms = 5");
  page.run('showTab("enroll")');
  await settle();
  page.run('showTab("debug")');
  await settle();
  page.run('showTab("enroll")');
  await settle();
  first.open();
  await settle();
  second.open();
  await sleep(60);
  assert.ok(calls > 4, "the poll went on");
  const latest = 100 + n - 1;
  assert.match(page.document.querySelector("#view").textContent, new RegExp("frames: " + latest + " / 999"));
  page.run("stopPolling()");
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
  await settle();
  const xmp = page.document.querySelector("#xmp_file");
  Object.defineProperty(xmp, "files", { value: [{ size: 1, text: async () => "<x/>" }] });
  xmp.closest("form").dispatchEvent(new page.window.Event("submit", { bubbles: true, cancelable: true }));
  await sleep(10);
  assert.equal(calls, 2, "the import redraws the tab");
  page.run('showTab("debug")');
  await settle();
  held.open();
  await sleep(20);
  assert.equal(page.document.querySelector("#notice").hidden, true);
  assert.equal(page.document.querySelector("#import_result"), null);
});
