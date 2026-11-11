// The Edited marker (SPEC 11A, 12): a setting whose control holds an unsaved edit says so in its tag
// list and carries data-dirty; both go when the value is back to the stored one or after a save, and
// stay across a filter change (the row is kept). Nothing is announced. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const PIN = "482913";

function d(path, kind, extra = {}) {
  return {
    path, kind, nullable: false, value: null, default: null, source: "schema", ge: null, le: null, unit: null,
    restart: false, admin_only: false, opt_in: false, conservative: null, description: "", choices: null,
    item_kind: null, length: null, key: null, entry: null, secret: false, private: false, writable: true, removable: null, ...extra,
  };
}

async function page(routes = {}) {
  let volume = 3;
  const fields = () => [
    d("voice.volume", "number", { value: volume, default: 3, ge: 0, le: 10 }),
    d("voice.label", "text", { value: "hi", default: "hi" }),
    d("voice.tags", "string_list", { value: ["x"], default: ["x"] }),
    d("voice.loud", "bool", { value: false, default: false }),
  ];
  const p = await loadPage({
    "GET /api/strings": { language: "en", strings: { settings_edited: "Edited" } },
    "GET /api/settings": () => ({ fields: fields(), pin_required: true }),
    "POST /api/settings": (b) => { volume = b.value; return { changed: true, layer: "runtime", persisted: true, applied: true }; },
    ...routes,
  });
  await p.run("settingsView()");
  p.document.querySelector("#grp-core-voice").click();
  await settle();
  return p;
}

const row = (p, path) => p.document.querySelector(`[data-path="${path}"]`);
const marked = (p, path) => {
  const r = row(p, path);
  const tags = r.querySelectorAll("[data-edited]");
  assert.equal(tags.length > 0, r.hasAttribute("data-dirty"), "tag and data-dirty go together");
  return tags.length === 1 && tags[0].textContent === "Edited" && tags[0].classList.contains("sl-tag");
};
const type = (p, input, value) => { input.value = value; input.dispatchEvent(new p.window.Event("input", { bubbles: true })); };

test("an edit marks the row; typing the stored value back unmarks it", async () => {
  const p = await page();
  const input = row(p, "voice.volume").querySelector("input[type=number]");
  assert.equal(marked(p, "voice.volume"), false, "nothing edited yet");
  type(p, input, "7");
  assert.equal(marked(p, "voice.volume"), true);
  assert.equal(row(p, "voice.label").hasAttribute("data-dirty"), false, "only the edited row");
  type(p, input, "3");
  assert.equal(marked(p, "voice.volume"), false, "back to the stored value");
  assert.equal(row(p, "voice.volume").querySelector("[role=status], [aria-live]"), null, "nothing is announced");
});

test("a switch, a chip removed and a slider each count as an edit", async () => {
  const p = await page();
  row(p, "voice.loud").querySelector("input[type=checkbox]").click();
  assert.equal(marked(p, "voice.loud"), true);
  row(p, "voice.tags").querySelector(".sl-chip").click();
  assert.equal(marked(p, "voice.tags"), true);
  const slider = row(p, "voice.volume").querySelector("input[type=range]");
  type(p, slider, "9");
  assert.equal(marked(p, "voice.volume"), true);
});

test("a save clears the marker of the saved row and keeps the others", async () => {
  const p = await page();
  type(p, row(p, "voice.volume").querySelector("input[type=number]"), "7");
  type(p, row(p, "voice.label").querySelector("input"), "hello");
  const save = [...row(p, "voice.volume").querySelectorAll("button")].find((b) => b.textContent === "settings_save");
  save.click();
  await settle();
  const dialog = p.document.querySelector("dialog");
  dialog.querySelector("#dialog-pin").value = PIN;
  dialog.querySelector("form").dispatchEvent(new p.window.Event("submit", { cancelable: true }));
  for (let i = 0; i < 4; i++) await settle();
  assert.equal(marked(p, "voice.volume"), false, "saved: the stored value is the shown one");
  assert.equal(row(p, "voice.volume").querySelector("input[type=number]").value, "7");
  assert.equal(marked(p, "voice.label"), true, "the other edit is still unsaved");
});

test("the marker stays across a filter change", async () => {
  const p = await page();
  type(p, row(p, "voice.label").querySelector("input"), "hello");
  const search = p.document.querySelector("#settings-search");
  search.value = "label";
  search.dispatchEvent(new p.window.Event("input"));
  await new Promise((r) => setTimeout(r, 200));
  assert.equal(row(p, "voice.volume"), null, "filtered out");
  assert.equal(marked(p, "voice.label"), true);
  search.value = "";
  search.dispatchEvent(new p.window.Event("input"));
  await new Promise((r) => setTimeout(r, 200));
  assert.equal(marked(p, "voice.label"), true);
  assert.equal(row(p, "voice.label").querySelector("input").value, "hello");
});
