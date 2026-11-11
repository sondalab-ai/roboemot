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
