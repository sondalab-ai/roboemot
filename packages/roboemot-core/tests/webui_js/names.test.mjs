// Accessible names of repeated controls (SPEC 11A, 12): a button repeated on each row says its row,
// and its name starts with its visible text (WCAG 2.5.3); action columns have a header for assistive
// technology; the hint of a disabled Remove is visible and linked. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const STRINGS = {
  named_action: "{action}: {name}", table_actions: "Actions",
  rename: "Rename", forget: "Forget (erase everything)", merge: "Merge into...", delete: "Delete",
  secret_remove: "Remove", secret_in_use_hint: "Choose another key or no key on those routes first.",
  settings_save: "Save", settings_reset_layer: "Reset to lower layer", settings_remove_entry: "Remove entry",
  settings_restart_badge: "Restart needed", settings_restart_goto: "Go to Restart RoboEmot", settings_restart: "Restart RoboEmot",
};
const BASE = {
  "GET /api/session": { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 },
  "GET /api/strings": { language: "en", strings: STRINGS },
  "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
  "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
};

// The name assistive technology gives the control, and its visible text.
const named = (node) => ({ name: node.getAttribute("aria-label") || node.textContent, shown: node.textContent });

function labelInName(node, row) {
  const { name, shown } = named(node);
  assert.ok(name.startsWith(shown), `"${name}" starts with "${shown}"`);
  assert.ok(name.includes(row), `"${name}" names ${row}`);
}

const hiddenHeads = (table) => [...table.querySelectorAll("thead th")].map((th) => {
  const sr = th.querySelector(".sr-only");
  return sr ? sr.textContent : null;
});

test("registry: each row's buttons name the row, and the action column has a header", async () => {
  const page = await loadPage({
    ...BASE,
    "GET /api/registry": { records: [
      { id: "p1", kind: "person", display_name: "Mia", status: "active", last_seen: null },
      { id: "o1", kind: "object", display_name: "Lamp", status: "active", last_seen: null },
      { id: "o2", kind: "object", display_name: "Mug", status: "active", last_seen: null },
    ] },
    "GET /api/registry/quarantined": { keys: [] },
  });
  page.document.querySelector('#tabs [data-tab="registry"]').click();
  await settle();
  await settle();
  const row = (id) => page.document.querySelector(`tr[data-row="${id}"]`);
  const names = (id) => [...row(id).querySelectorAll("button")].map((b) => named(b).name);
  assert.deepEqual(names("p1"), ["Rename: Mia", "Forget (erase everything): Mia"]);
  assert.deepEqual(names("o1"), ["Rename: Lamp", "Merge into... Mug: Lamp", "Delete: Lamp"]);
  for (const b of page.document.querySelectorAll("tbody button")) labelInName(b, b.closest("tr").cells[0].textContent);
  assert.equal(hiddenHeads(page.document.querySelector("table")).at(-1), "Actions");
});

test("secrets: Remove names the secret; a secret in use says why it cannot go, in the row", async () => {
  const page = await loadPage({
    ...BASE,
    "GET /api/routes": { routes: [], name_max_chars: 32 },
    "GET /api/secrets": { secrets: [
      { name: "cloud_key", set: true, used_by: ["cloud"] },
      { name: "spare", set: true, used_by: [] },
    ], max_name_chars: 32, max_value_chars: 64 },
  });
  await page.run("routesView()");
  const remove = (name) => page.document.querySelector(`tr[data-row="${name}"] button`);
  labelInName(remove("spare"), "spare");
  labelInName(remove("cloud_key"), "cloud_key");
  const used = remove("cloud_key");
  assert.equal(used.disabled, true);
  assert.equal(used.getAttribute("title"), null, "not a tooltip only");
  const hint = page.document.getElementById(used.getAttribute("aria-describedby"));
  assert.ok(hint && hint.classList.contains("hint"), "linked to a visible muted line");
  assert.equal(hint.textContent, STRINGS.secret_in_use_hint);
  assert.ok(used.closest("tr").contains(hint), "in the row");
  assert.equal(remove("spare").getAttribute("aria-describedby"), null);
  assert.equal(hiddenHeads(page.document.querySelector("table")).at(-1), "Actions");
});

test("registry: Purge names the quarantined key", async () => {
  const page = await loadPage({
    ...BASE,
    "GET /api/strings": { language: "en", strings: { ...STRINGS, purge: "Purge" } },
    "GET /api/registry": { records: [] },
    "GET /api/registry/quarantined": { keys: ["k1", "k2"] },
  });
  await page.run("registryView()");
  const purge = [...page.document.querySelectorAll("li[data-row] button")];
  assert.deepEqual(purge.map((b) => named(b).name), ["Purge: k1", "Purge: k2"]);
  for (const b of purge) labelInName(b, b.closest("li").dataset.row);
});

test("catalogue: Activate and every Confirm name their record", async () => {
  const page = await loadPage({
    ...BASE,
    "GET /api/strings": { language: "en", strings: { ...STRINGS, catalogue_activate: "Activate from photo", catalogue_confirm: "This is {name}" } },
    "GET /api/catalogue": {
      records: [
        { id: "c1", kind: "object", display_name: "Lamp", status: "pending", origin: "xmp", has_appearance: true, category: null },
        { id: "c2", kind: "object", display_name: "Mug", status: "pending", origin: "xmp", has_appearance: true, category: null },
      ],
      live: [{ kind: "object", category: "cup", entry_id: "e-7", identified: false }],
      limits: { max_xmp_bytes: 1e6, max_image_bytes: 1e6, max_items: 10 }, tombstones: 0,
    },
  });
  await page.run("catalogueView()");
  for (const id of ["c1", "c2"]) {
    const row = page.document.querySelector(`tr[data-row="${id}"]`);
    const shown = row.cells[0].textContent;
    const buttons = [...row.querySelectorAll("button")];
    assert.equal(buttons.length, 2);
    for (const b of buttons) {
      labelInName(b, shown);
      assert.ok(b.hasAttribute("aria-label"), named(b).name);
    }
    assert.equal(named(buttons[0]).name, "Activate from photo: " + shown);
  }
});

test("routes: Check, Edit and Remove name their route", async () => {
  const route = (name) => ({ name, model: "m", endpoint: "h", base_url: "http://h", locality: "cloud", kind: "chat", key: "none",
    secret: null, removable: true, running: false, last_check: null });
  const page = await loadPage({
    ...BASE,
    "GET /api/strings": { language: "en", strings: { ...STRINGS, route_check: "Check", route_edit: "Edit", route_remove: "Remove" } },
    "GET /api/routes": { routes: [route("cloud"), route("local")], name_max_chars: 32 },
    "GET /api/secrets": { secrets: [], max_name_chars: 32, max_value_chars: 64 },
  });
  await page.run("routesView()");
  for (const name of ["cloud", "local"]) {
    const row = page.document.querySelector(`[data-row="${name}"] .row`);
    const names = [...row.querySelectorAll(":scope > button")].map((b) => named(b).name);
    assert.deepEqual(names, [`Check: ${name}`, `Edit: ${name}`, `Remove: ${name}`]);
  }
});

test("settings: Add and Remove of a list name the setting", async () => {
  const page = await loadPage({
    ...BASE,
    "GET /api/strings": { language: "en", strings: { ...STRINGS, settings_add: "Add entry", settings_remove: "Remove", settings_remove_n: "Remove row {n}" } },
    "GET /api/settings": { pin_required: true, fields: [
      d("voice.tags", "string_list", { value: ["x"] }),
      d("voice.pairs", "tuple_list", { value: [[1, 2]], length: 2, item_kind: "number" }),
    ] },
  });
  await page.run("settingsView()");
  page.document.querySelector("#grp-core-voice").click();
  await settle();
  const buttons = (path) => [...page.document.querySelectorAll(`[data-path="${path}"] button:not(.sl-chip)`)]
    .filter((b) => !b.closest(".row:last-child") || b.textContent !== "settings_save");
  const add = (path) => buttons(path).find((b) => b.textContent === "Add entry");
  for (const path of ["voice.tags", "voice.pairs"]) {
    labelInName(add(path), path.split(".").pop());
  }
  const remove = buttons("voice.pairs").find((b) => b.textContent === "Remove");
  assert.ok(named(remove).name.startsWith("Remove row 1"), named(remove).name);
  labelInName(remove, "pairs");
});

// Every header cell of a table has text: visible on a data column, hidden on the action column.
const headTexts = (table) => [...table.querySelectorAll("thead th")].map((th) => th.textContent.trim());

test("registry, catalogue and secrets: every column has a header", async () => {
  const page = await loadPage({
    ...BASE,
    "GET /api/strings": { language: "en", strings: { ...STRINGS, display_name: "Name", last_seen: "Last seen",
      secret_name: "Name", table_kind_status: "Kind and status", table_kind_origin: "Kind and origin",
      table_secret_status: "Status", table_secret_used_by: "Used by" } },
    "GET /api/registry": { records: [{ id: "p1", kind: "person", display_name: "Mia", status: "active", last_seen: null }] },
    "GET /api/registry/quarantined": { keys: [] },
    "GET /api/catalogue": { records: [{ id: "c1", kind: "object", display_name: "Lamp", status: "active", origin: "xmp", has_appearance: false, category: null }],
      live: [], limits: { max_xmp_bytes: 1e6, max_image_bytes: 1e6, max_items: 10 }, tombstones: 0 },
    "GET /api/routes": { routes: [], name_max_chars: 32 },
    "GET /api/secrets": { secrets: [{ name: "k", set: true, used_by: [] }], max_name_chars: 32, max_value_chars: 64 },
  });
  await page.run("registryView()");
  assert.deepEqual(headTexts(page.document.querySelector("table")), ["Name", "Kind and status", "Last seen", "Actions"]);
  await page.run("catalogueView()");
  assert.deepEqual(headTexts(page.document.querySelector("table")), ["Name", "Kind and origin", "Actions"]);
  await page.run("routesView()");
  assert.deepEqual(headTexts(page.document.querySelector("table")), ["Name", "Status", "Used by", "Actions"]);
});

function d(path, kind, extra = {}) {
  return {
    path, kind, nullable: false, value: null, default: null, source: "schema", ge: null, le: null, unit: null,
    restart: false, admin_only: false, opt_in: false, conservative: null, description: "", choices: null,
    item_kind: null, length: null, key: null, entry: null, secret: false, private: false, writable: true, removable: null, ...extra,
  };
}

test("settings: Save, Reset and Remove entry name the setting; the restart link keeps its visible text", async () => {
  const page = await loadPage({
    ...BASE,
    "GET /api/settings": { pin_required: true, fields: [
      d("voice.gain", "number", { value: 2, default: 1, source: "runtime", restart: true }),
      d("voice.moods", "dict_of_model", { value: ["calm"], writable: false, removable: ["calm"], entry: [d("rate", "number")] }),
      d("voice.moods.calm.rate", "number", { value: 1 }),
    ], robot: { id: "r", title: "R", actions: ["restart"], fields: [] } },
  });
  await page.run("settingsView()");
  page.document.querySelector("#grp-core-voice").click();
  await settle();
  const gain = page.document.querySelector('[data-path="voice.gain"]');
  const [save, reset] = gain.querySelectorAll(".row:last-child button");
  labelInName(save, "voice.gain");
  labelInName(reset, "voice.gain");
  assert.equal(named(save).name, "Save: voice.gain");
  const entry = page.document.querySelector('[data-entry="voice.moods.calm"] .row > button');
  assert.equal(named(entry).name, "Remove entry: voice.moods.calm");
  const link = gain.querySelector("a[href='#settings-restart']");
  assert.ok(link, "the badge links to the Restart button");
  labelInName(link, "Go to Restart RoboEmot");
  assert.equal(named(link).name, "Restart needed: Go to Restart RoboEmot");
});
