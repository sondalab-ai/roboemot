// Focus around the one dialog (SPEC 11A, 12): it opens on its first field, else on Cancel; after it
// closes and the view is drawn again, the focus goes back to the equal control of the same row, or
// to the heading of the card that held it, never to <body>. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const PIN = "482913";
const SESSION = { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000, name_max_chars: 64 };
const BASE = {
  "GET /api/session": SESSION,
  "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
  "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
};

const submit = (window, form) => form.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));

async function confirmDialog(page, fill = {}) {
  const dialog = page.document.querySelector("dialog");
  if (fill.pin !== undefined) dialog.querySelector("#dialog-pin").value = fill.pin;
  if (fill.value !== undefined) dialog.querySelector("#dialog-value").value = fill.value;
  submit(page.window, dialog.querySelector("form"));
  for (let i = 0; i < 4; i++) await settle();
}

test("the dialog opens on its first field, and a confirm with no field on Cancel", async () => {
  const page = await loadPage(BASE);
  const opened = (opts) => {
    page.run("ask(" + opts + ")");
    const focus = page.document.activeElement;
    page.document.querySelector("dialog").close();
    return focus;
  };
  assert.equal(opened("{ title: 'x', message: 'm', danger: true, run: () => {} }").textContent, "dialog_cancel");
  assert.equal(opened("{ title: 'x', message: 'm', pin: true, danger: true, run: () => {} }").id, "dialog-pin");
  assert.equal(opened("{ title: 'x', input: { label: 'n', value: 'a', max: '9' }, run: () => {} }").id, "dialog-value");
});

function registry(records) {
  return {
    ...BASE,
    "GET /api/registry": () => ({ records: records() }),
    "GET /api/registry/quarantined": { keys: [] },
  };
}

const rowButton = (page, id, text) =>
  [...page.document.querySelectorAll(`tr[data-row="${id}"] button`)].find((b) => b.textContent === text);

test("after a rename the focus is on the Rename button of the same row, drawn again", async () => {
  let name = "Mia";
  const records = () => [
    { id: "p1", kind: "person", display_name: name, status: "active", last_seen: null },
    { id: "p2", kind: "person", display_name: "Leo", status: "active", last_seen: null },
  ];
  const page = await loadPage({
    ...registry(records),
    // real names: the accessible name holds the row's name, which the rename changes
    "GET /api/strings": { language: "en", strings: { named_action: "{action}: {name}", rename: "Rename", forget: "Forget" } },
    "POST /api/registry/p1/rename": (b) => { name = b.display_name; return {}; },
  });
  page.document.querySelector('#tabs [data-tab="registry"]').click();
  await settle();
  await settle();
  const opener = rowButton(page, "p1", "Rename");
  assert.equal(opener.getAttribute("aria-label"), "Rename: Mia");
  opener.focus();
  opener.click();
  await confirmDialog(page, { value: "Mila" });
  const focus = page.document.activeElement;
  assert.notEqual(opener.isConnected, true, "the row was drawn again");
  assert.equal(focus, rowButton(page, "p1", "Rename"), "the equal control of the same row, though its name changed");
  assert.ok(page.document.querySelector('tr[data-row="p1"]').textContent.includes("Mila"));
});

test("after a forget the row is gone and the focus is on the heading of its card, focusable only meanwhile", async () => {
  let gone = false;
  const records = () => (gone ? [] : [{ id: "p1", kind: "person", display_name: "Mia", status: "active", last_seen: null }]);
  const page = await loadPage({
    ...registry(records),
    "POST /api/registry/p1/forget": () => { gone = true; return {}; },
  });
  page.document.querySelector('#tabs [data-tab="registry"]').click();
  await settle();
  await settle();
  const opener = rowButton(page, "p1", "forget");
  opener.focus();
  opener.click();
  await confirmDialog(page, { pin: PIN });
  const focus = page.document.activeElement;
  assert.equal(focus.tagName, "H2");
  assert.equal(focus.textContent, "registry_title");
  assert.equal(focus.getAttribute("tabindex"), "-1");
  page.document.querySelector('#tabs [data-tab="registry"]').focus();
  assert.equal(focus.getAttribute("tabindex"), null, "not a tab stop once the focus moved on");
});

test("a cancelled dialog changes nothing, and a later redraw does not move the focus", async () => {
  const records = () => [{ id: "p1", kind: "person", display_name: "Mia", status: "active", last_seen: null }];
  const page = await loadPage(registry(records));
  page.document.querySelector('#tabs [data-tab="registry"]').click();
  await settle();
  await settle();
  const opener = rowButton(page, "p1", "rename");
  opener.focus();
  opener.click();
  page.document.querySelector("dialog").close();
  await settle();
  const other = page.document.querySelector('#tabs [data-tab="registry"]');
  other.focus();
  await page.run("guardedView(registryView)");
  assert.equal(page.document.activeElement, other, "focus that is somewhere is left there");
});

function d(path, kind, extra = {}) {
  return {
    path, kind, nullable: false, value: null, default: null, source: "schema", ge: null, le: null, unit: null,
    restart: false, admin_only: false, opt_in: false, conservative: null, description: "", choices: null,
    item_kind: null, length: null, key: null, entry: null, secret: false, private: false, writable: true, removable: null, ...extra,
  };
}

test("settings: after an entry is removed the focus is on its group's toggle", async () => {
  let removed = false;
  const fields = () => [
    d("voice.moods", "dict_of_model", { value: removed ? [] : ["calm"], writable: false, removable: ["calm"], entry: [d("rate", "number")] }),
    ...(removed ? [] : [d("voice.moods.calm.rate", "number", { value: 1 })]),
  ];
  const page = await loadPage({
    "GET /api/settings": () => ({ fields: fields(), pin_required: true }),
    "POST /api/settings/entry/remove": () => { removed = true; return { changed: true, layer: "runtime", persisted: true }; },
  });
  await page.run("settingsView()");
  page.document.querySelector("#grp-core-voice").click();
  await settle();
  const opener = [...page.document.querySelectorAll('[data-entry="voice.moods.calm"] button')].find((b) => b.textContent === "settings_remove_entry");
  opener.focus();
  opener.click();
  await settle();
  await confirmDialog(page, { pin: PIN });
  assert.equal(page.document.querySelector('[data-entry="voice.moods.calm"]'), null);
  assert.equal(page.document.activeElement, page.document.querySelector("#grp-core-voice"));
});
