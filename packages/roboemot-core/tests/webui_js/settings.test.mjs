// Behaviour of the settings page (SPEC 11A, 12): controls per descriptor kind, lazy groups, search,
// PIN per write, reveal, reset, entries, the robot section and error mapping. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle, storageDump } from "./harness.mjs";

const PIN = "482913";

const submit = (window, form) =>
  form.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));

/** A full descriptor (every key present, as the server sends it). */
function d(path, kind, extra = {}) {
  return {
    path, kind, nullable: false, value: null, default: null, source: "schema", ge: null, le: null, unit: null,
    restart: false, admin_only: false, opt_in: false, conservative: null, description: "", choices: null,
    item_kind: null, length: null, key: null, entry: null, secret: false, private: false, writable: true, removable: null, ...extra,
  };
}

const ENTRY = [
  d("rate", "number", { value: null, ge: 0, le: 2 }),
  d("label", "text", { value: "" }),
  d("secret_token", "text", { secret: true, value: "***" }),
];

function fields() {
  return [
    d("voice.volume_db", "number", { value: 3.5, default: 0, ge: -12, le: 12, unit: "dB", source: "runtime", description: "Speech volume" }),
    d("voice.count", "int", { value: 3, default: 3, ge: 1, le: 9 }),
    d("voice.flag", "bool", { value: true, default: false, restart: true }),
    d("voice.mode", "enum", { value: "b", default: "a", choices: ["a", "b", "c"] }),
    d("voice.label", "text", { value: "hello", default: "hello" }),
    d("voice.tags", "string_list", { value: ["x", "y"], default: ["x"] }),
    d("voice.kinds", "enum_list", { value: ["animal"], default: ["animal"], choices: ["animal", "object"] }),
    d("voice.pair", "tuple", { value: [1, 2], default: [1, 2], item_kind: "number", length: 2 }),
    d("voice.pairs", "tuple_list", { value: [[0, 1], [1, 2]], default: [[0, 1]], item_kind: "number", length: 2 }),
    d("voice.names", "string_dict", { value: { a: "b" }, default: { a: "b" } }),
    d("voice.gains", "number_dict", { value: { lat: 1 }, default: { lat: 1 } }),
    d("voice.blob", "json", { value: { k: [1, 2] }, default: { k: [1, 2] } }),
    d("voice.opt", "number", { value: null, default: null, nullable: true, ge: 0, le: 5 }),
    d("voice.opt2", "number", { value: 3, default: null, nullable: true, ge: 0, le: 5 }),
    d("voice.api_key", "text", { value: "***", secret: true, writable: false }),
    d("voice.fixed", "number", { value: 7, default: 7, writable: false, ge: 0, le: 9 }),
    d("voice.moods", "dict_of_model", { value: ["calm"], entry: ENTRY, writable: false, removable: ["calm"] }),
    d("voice.moods.calm.rate", "number", { value: 1, default: 1, ge: 0, le: 2, source: "runtime" }),
    d("voice.moods.calm.label", "text", { value: "c" }),
    d("appraisal.rules", "keyed_list", { value: ["0"], key: "event", writable: false,
      entry: [d("event", "text", { value: "" }), d("valence", "number", { value: null, ge: -1, le: 1 })] }),
    d("appraisal.rules.0.event", "text", { value: "user.greeted" }),
    d("appraisal.rules.0.valence", "number", { value: 0.5, ge: -1, le: 1 }),
    d("conversation.confirm_yes", "string_list_dict", { value: ["en"], writable: false }),
    d("conversation.confirm_yes.en", "string_list", { value: ["yes"], default: ["yes"] }),
    d("privacy.note", "text", { value: "***", default: "***", private: true, admin_only: true }),
    d("safety.blocklist_word", "string_list", { value: ["bad"], default: ["bad"], conservative: "superset" }),
  ];
}

const ROBOT = {
  id: "reachy-mini-wireless", title: "Reachy Mini",
  fields: [
    d("robot.reachy-mini-wireless.limits.max_speed", "number", { value: 2, writable: false, source: "manifest" }),
    d("robot.reachy-mini-wireless.dynamics.accel", "number", { value: null, writable: false, source: "manifest" }),
    d("expression_map", "dict_of_model", { value: [], entry: ENTRY, writable: false }),
    d("robot.limit", "number", { value: 4, default: 4, writable: false, ge: 0, le: 9 }),
  ],
};

async function settingsPage(extra = {}, data = { fields: fields(), pin_required: true }) {
  const page = await loadPage({ "GET /api/settings": data, ...extra });
  await page.run("settingsView()");
  return page;
}

const group = (page, id) => page.document.querySelector("#grp-" + id);
async function open(page, id) {
  group(page, id).click();
  await settle();
}
const body = (page, id) => page.document.querySelector("#grp-" + id + "-body");
const row = (page, path) => page.document.querySelector(`[data-path="${path}"]`);
const buttons = (node, text) => [...node.querySelectorAll("button")].filter((b) => b.textContent === text);
const button = (node, text) => buttons(node, text)[0];

async function answerPin(page, pin = PIN) {
  const dialog = page.document.querySelector("dialog");
  dialog.querySelector("#dialog-pin").value = pin;
  submit(page.window, dialog.querySelector("form"));
  await settle();
  await settle();
  return dialog;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const typeQuery = async (page, text) => {
  const search = page.document.querySelector("#settings-search");
  search.value = text;
  search.dispatchEvent(new page.window.Event("input"));
  await sleep(200);
};

const posts = (page, path) => page.calls.filter((c) => c.method === "POST" && c.path === path);

test("groups are collapsed and render nothing until opened", async () => {
  const page = await settingsPage();
  const doc = page.document;
  assert.ok(doc.querySelector("nav.sl-nav"));
  assert.equal(doc.querySelectorAll(".sl-nav__item").length, 5, "voice, appraisal, conversation, privacy, safety");
  assert.equal(group(page, "core-voice").getAttribute("aria-expanded"), "false");
  assert.equal(body(page, "core-voice").hidden, true);
  assert.equal(doc.querySelectorAll(".setting input, .setting select, .setting textarea").length, 0);
  await open(page, "core-voice");
  assert.equal(group(page, "core-voice").getAttribute("aria-expanded"), "true");
  assert.ok(doc.querySelectorAll("#grp-core-voice-body input").length > 10);
  assert.equal(body(page, "core-privacy").children.length, 0, "other groups stay empty");
});

test("every kind gets its control", async () => {
  const page = await settingsPage();
  await open(page, "core-voice");
  const n = (path, sel) => row(page, path).querySelectorAll(sel).length;

  const num = row(page, "voice.volume_db");
  assert.equal(num.querySelector('input[type="number"]').value, "3.5");
  assert.equal(num.querySelector('input[type="range"].sl-slider').value, "3.5");
  assert.ok(num.querySelector('input[type="range"]').style.getPropertyValue("--sl-slider-value"));
  assert.equal(num.textContent.includes("dB"), true, "unit shown");
  assert.equal(num.querySelector("label").getAttribute("for"), num.querySelector('input[type="number"]').id);

  assert.equal(row(page, "voice.count").querySelector('input[type="number"]').getAttribute("step"), "1");
  const sw = row(page, "voice.flag").querySelector("label.sl-switch input");
  assert.equal(sw.getAttribute("role"), "switch");
  assert.equal(sw.checked, true);
  const select = row(page, "voice.mode").querySelector(".sl-select select");
  assert.deepEqual([...select.options].map((o) => o.textContent), ["a", "b", "c"]);
  assert.equal(select.options[select.selectedIndex].textContent, "b");
  assert.equal(row(page, "voice.label").querySelector('input[type="text"]').value, "hello");
  assert.deepEqual([...row(page, "voice.tags").querySelectorAll(".sl-chip")].map((c) => c.textContent), ["x", "y"]);
  assert.deepEqual([...row(page, "voice.kinds").querySelectorAll(".sl-chip")].map((c) => c.textContent), ["animal"]);
  assert.equal(n("voice.kinds", "select option"), 1, "only the choices not yet chosen");
  assert.equal(n("voice.pair", 'input[type="number"]'), 2);
  assert.equal(n("voice.pairs", 'input[type="number"]'), 4, "two tuples of two");
  assert.equal(n("voice.names", 'input[type="text"]'), 2, "key and value");
  assert.equal(n("voice.gains", 'input[type="number"]'), 1);
  const json = row(page, "voice.blob");
  assert.ok(json.querySelector("pre"));
  assert.equal(json.querySelectorAll("input, select, textarea, button").length, 0, "json is read-only");
  const opt = row(page, "voice.opt");
  assert.equal(opt.querySelector('input[type="checkbox"]').checked, true, "null toggle on");
  assert.equal(opt.querySelector('input[type="number"]').disabled, true);

  assert.ok(row(page, "voice.moods"), "dict_of_model container");
  assert.ok(row(page, "conversation.confirm_yes") === null, "other groups are not rendered");
  await open(page, "core-conversation");
  assert.ok(row(page, "conversation.confirm_yes"), "string_list_dict container");
  assert.equal(row(page, "conversation.confirm_yes").querySelectorAll(".sl-chip").length, 0, "language entries are lazy");
  await open(page, "core-appraisal");
  assert.ok(row(page, "appraisal.rules"), "keyed_list container");
});

test("read-only, secret and identity rows have no editing control", async () => {
  const page = await settingsPage();
  await open(page, "core-voice");
  for (const path of ["voice.api_key", "voice.fixed"]) {
    assert.equal(row(page, path).querySelectorAll("input, select, textarea, button").length, 0, path);
  }
  assert.equal(row(page, "voice.api_key").querySelector(".setting-value").textContent, "***");
  assert.ok(row(page, "voice.api_key").textContent.includes("settings_secret"));
  assert.ok(row(page, "voice.fixed").textContent.includes("settings_readonly"));
  await open(page, "core-appraisal");
  const entry = group(page, "core-appraisal").closest("section").querySelector(".entry-block .sl-disclosure");
  entry.click();
  assert.equal(row(page, "appraisal.rules.0.event").querySelectorAll("input, select, textarea, button").length, 0, "identity is fixed");
  assert.equal(row(page, "appraisal.rules.0.valence").querySelectorAll("input").length > 0, true);
});

test("a robot section appears as its own group, and is absent without one", async () => {
  const without = await settingsPage();
  assert.ok(!without.document.body.textContent.includes("Reachy Mini"));
  assert.equal(without.document.querySelectorAll(".sl-nav__group").length, 1);

  const nullRobot = await settingsPage({}, { fields: fields(), robot: null, pin_required: true });
  assert.equal(nullRobot.document.querySelector("#grp-robot"), null);

  const page = await settingsPage({}, { fields: fields(), robot: ROBOT, pin_required: true });
  const groups = [...page.document.querySelectorAll(".sl-nav__group")].map((g) => g.textContent);
  assert.deepEqual(groups, ["settings_core", "Reachy Mini"]);
  assert.equal(page.document.querySelector("#settings-robot-actions").children.length, 0, "no action without robot.actions");
  assert.equal(page.document.querySelector("#settings-restart"), null, "no restart button without the action");
  assert.equal(without.document.querySelector("#settings-robot-actions"), null);
  await open(page, "robot-limits");
  await open(page, "robot-dynamics");
  await open(page, "robot-robot");
  await open(page, "robot-expression_map");
  assert.equal(row(page, "robot.reachy-mini-wireless.limits.max_speed").querySelectorAll("input").length, 0, "hardware limits are read-only");
  assert.ok(row(page, "robot.reachy-mini-wireless.dynamics.accel").textContent.includes("settings_not_measured"));
  assert.equal(row(page, "robot.limit").querySelectorAll("input").length, 0);
  assert.ok(row(page, "expression_map"), "behaviour container");
});

test("search matches the path or the description and opens the matching groups", async () => {
  const page = await settingsPage({ "GET /api/strings": { language: "en", strings: { settings_matches: "{n} shown" } } });
  await typeQuery(page, "speech volume");
  assert.equal(group(page, "core-voice").getAttribute("aria-expanded"), "true");
  assert.deepEqual([...body(page, "core-voice").querySelectorAll(".setting")].map((r) => r.dataset.path), ["voice.volume_db"]);
  assert.equal(page.document.querySelector("#grp-core-privacy").closest("section").hidden, true);
  assert.equal(page.document.querySelector('.sl-nav__item[data-group="core-privacy"]').hidden, true);
  assert.equal(page.document.querySelector("#settings-count").textContent, "1 shown");

  await typeQuery(page, "valence");
  assert.equal(group(page, "core-appraisal").closest("section").hidden, false, "found through a child of a container");

  await typeQuery(page, "");
  assert.equal(group(page, "core-voice").getAttribute("aria-expanded"), "false", "back to what was opened by hand");
});

test("the changed-only filter keeps runtime values and values that differ from the default", async () => {
  const page = await settingsPage();
  const only = page.document.querySelector("#settings-changed");
  only.checked = true;
  only.dispatchEvent(new page.window.Event("change"));
  const shown = [...page.document.querySelectorAll(".setting")].map((r) => r.dataset.path);
  assert.ok(shown.includes("voice.volume_db"), "runtime source");
  assert.ok(shown.includes("voice.flag"), "value differs from default");
  assert.ok(shown.includes("voice.tags"));
  assert.ok(!shown.includes("voice.label"), "same as default");
  assert.ok(!shown.includes("voice.count"));
  assert.ok(shown.includes("voice.moods"), "a container with a runtime entry");
  assert.equal(page.document.querySelector("#grp-core-privacy").closest("section").hidden, true);
});

test("badges show the source layer, restart, admin and private", async () => {
  const page = await settingsPage();
  await open(page, "core-voice");
  assert.ok(row(page, "voice.volume_db").querySelector(".sl-tag:not(.sl-tag--plain)").textContent.length > 0);
  const restart = row(page, "voice.flag").querySelector(".sl-badge--warning");
  assert.ok(restart, "restart badge shown");
  assert.equal(restart.closest("a"), null, "no link without the restart action");
  assert.equal(page.document.querySelector("#settings-restart"), null);
  await open(page, "core-privacy");
  const text = row(page, "privacy.note").textContent;
  assert.ok(text.includes("settings_admin") && text.includes("settings_private"));
});

test("save: the PIN is asked, path/value/PIN are posted, and the response layer and restart show", async () => {
  let attempt = 0;
  const page = await settingsPage({
    "GET /api/strings": { language: "en", strings: { err_bad_pin: "Wrong PIN." } },
    "POST /api/settings": () => (++attempt === 1
      ? reply(403, { error: "bad_pin" })
      : { path: "voice.flag", changed: true, value: false, layer: "app", restart_required: true, persisted: true }),
  });
  await open(page, "core-voice");
  const box = row(page, "voice.flag");
  const sw = box.querySelector("input");
  sw.checked = false;
  button(box, "settings_save").click();
  await settle();
  assert.equal(posts(page, "/api/settings").length, 0, "nothing is sent before the PIN");
  const dialog = await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings")[0].body, { path: "voice.flag", value: false, pin: PIN });
  assert.equal(dialog.querySelector("#dialog-pin").getAttribute("aria-invalid"), "true");
  assert.equal(dialog.querySelector("#dialog-error").textContent, "Wrong PIN.");
  await answerPin(page);
  assert.equal(page.document.querySelector("dialog"), null);
  const notice = page.document.querySelector("#notice").textContent;
  assert.ok(notice.includes("settings_layer_app"), "permanent app-layer save is announced");
  assert.ok(notice.includes("settings_restart_needed"));
  assert.ok(page.document.querySelector("#settings-restart-hint").textContent.includes("settings_restart_pending"));
  assert.ok(!storageDump(page.window).includes(PIN));
  assert.ok(!page.document.documentElement.outerHTML.includes(PIN));
});

test("save: each write asks again, and an unchanged value sends nothing", async () => {
  const page = await settingsPage({ "POST /api/settings": { changed: true, layer: "runtime", persisted: true } });
  await open(page, "core-voice");
  const box = row(page, "voice.label");
  button(box, "settings_save").click();
  await settle();
  assert.equal(page.document.querySelector("dialog"), null);
  assert.equal(page.document.querySelector("#notice").textContent, "settings_unchanged");
  const input = box.querySelector("input");
  for (const v of ["one", "two"]) {
    row(page, "voice.label").querySelector("input").value = v;
    button(row(page, "voice.label"), "settings_save").click();
    await settle();
    assert.ok(page.document.querySelector("dialog"), "PIN asked for " + v);
    await answerPin(page);
  }
  assert.equal(posts(page, "/api/settings").length, 2);
  assert.ok(input);
});

test("save: a client pre-check flags the field and sends nothing", async () => {
  const page = await settingsPage();
  await open(page, "core-voice");
  const box = row(page, "voice.volume_db");
  const input = box.querySelector('input[type="number"]');
  input.value = "99";
  button(box, "settings_save").click();
  await settle();
  const err = box.querySelector(".sl-field__error");
  assert.equal(err.hidden, false);
  assert.equal(err.textContent, "err_value_range");
  assert.equal(input.getAttribute("aria-invalid"), "true");
  assert.ok(input.getAttribute("aria-describedby").split(" ").includes(err.id));
  assert.equal(page.document.querySelector("dialog"), null);
});

test("controls send the right value shapes", async () => {
  const page = await settingsPage({ "POST /api/settings": { changed: true, layer: "runtime", persisted: true } });
  await open(page, "core-voice");
  const send = async (path, change) => {
    const box = row(page, path);
    change(box);
    button(box, "settings_save").click();
    await settle();
    await answerPin(page);
    const sent = posts(page, "/api/settings");
    return sent[sent.length - 1].body.value;
  };
  const input = (box, n = 0) => box.querySelectorAll("input")[n];
  assert.equal(await send("voice.volume_db", (b) => { input(b, 1).value = "-2.5"; }), -2.5);
  assert.equal(await send("voice.count", (b) => { b.querySelector('input[type="number"]').value = "5"; }), 5);
  assert.equal(await send("voice.mode", (b) => { b.querySelector("select").value = "2"; }), "c");
  assert.deepEqual(await send("voice.tags", (b) => {
    b.querySelector(".sl-chip").click();
    const add = b.querySelector('input[type="text"]');
    add.value = " new   one ";
    button(b, "settings_add").click();
  }), ["y", "new one"]);
  assert.deepEqual(await send("voice.kinds", (b) => { button(b, "settings_add").click(); }), ["animal", "object"]);
  assert.deepEqual(await send("voice.pair", (b) => { b.querySelectorAll("input")[1].value = "9"; }), [1, 9]);
  assert.deepEqual(await send("voice.pairs", (b) => { b.querySelectorAll("input")[0].value = "0.5"; }), [[0.5, 1], [1, 2]]);
  assert.deepEqual(await send("voice.names", (b) => { b.querySelectorAll("input")[1].value = "z"; }), { a: "z" });
  assert.deepEqual(await send("voice.gains", (b) => {
    button(b, "settings_add").click();
    const ins = b.querySelectorAll('input[type="text"]');
    ins[ins.length - 1].value = "lon";
    const nums = b.querySelectorAll('input[type="number"]');
    nums[nums.length - 1].value = "4";
  }), { lat: 1, lon: 4 });
  assert.equal(await send("voice.opt", (b) => { b.querySelector('input[type="checkbox"]').click(); b.querySelector('input[type="number"]').value = "2"; }), 2);
  assert.equal(await send("voice.opt2", (b) => { b.querySelector('input[type="checkbox"]').click(); }), null);
});

test("the slider and the number input stay in step", async () => {
  const page = await settingsPage();
  await open(page, "core-voice");
  const box = row(page, "voice.volume_db");
  const slider = box.querySelector('input[type="range"]');
  const number = box.querySelector('input[type="number"]');
  slider.value = "-4";
  slider.dispatchEvent(new page.window.Event("input"));
  assert.equal(number.value, "-4");
  number.value = "6";
  number.dispatchEvent(new page.window.Event("input"));
  assert.equal(slider.value, "6");
  assert.match(slider.style.getPropertyValue("--sl-slider-value"), /^75%$/);
});

test("a superset list can lose words (the server asks for the PIN) and is resettable from the runtime layer", async () => {
  const flat = fields().map((f) => (f.path === "safety.blocklist_word" ? { ...f, source: "runtime" } : f));
  const page = await settingsPage({ "POST /api/settings": { changed: true, layer: "app", persisted: true } }, { fields: flat, pin_required: true });
  await open(page, "core-safety");
  const box = row(page, "safety.blocklist_word");
  assert.equal(box.querySelector(".sl-chip").disabled, false);
  assert.equal(buttons(box, "settings_reset_layer").length, 1);
  box.querySelector(".sl-chip").click();
  button(box, "settings_save").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings")[0].body, { path: "safety.blocklist_word", value: [], pin: PIN });
});

test("reset to the lower layer is offered for runtime values and goes through the PIN", async () => {
  const page = await settingsPage({
    "POST /api/settings/reset": { path: "voice.volume_db", changed: true, value: 0, layer: "runtime", restart_required: false, persisted: true },
  });
  await open(page, "core-voice");
  assert.equal(buttons(row(page, "voice.label"), "settings_reset_layer").length, 0, "not a runtime value");
  button(row(page, "voice.volume_db"), "settings_reset_layer").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings/reset")[0].body, { path: "voice.volume_db", pin: PIN });
  assert.ok(page.document.querySelector("#notice").textContent.includes("settings_reset_done"));
  assert.ok(page.calls.filter((c) => c.method === "GET" && c.path === "/api/settings").length >= 2, "the list is reloaded");
});

test("private rows are masked until the PIN reveals them, and the values stay in memory", async () => {
  const revealed = d("privacy.note", "text", { value: "secret note", default: "d", private: true, admin_only: true });
  const page = await settingsPage({ "POST /api/settings/reveal": { fields: [revealed] } });
  await open(page, "core-privacy");
  const box = () => row(page, "privacy.note");
  assert.ok(!box().textContent.includes("***"));
  assert.equal(box().querySelectorAll("input").length, 0, "no editing control while hidden");
  button(box(), "settings_reveal").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings/reveal")[0].body, { pin: PIN });
  assert.equal(box().querySelector("input").value, "secret note");
  assert.equal(buttons(box(), "settings_reveal").length, 0);
  assert.ok(!storageDump(page.window).includes("secret note"));
  assert.ok(!storageDump(page.window).includes(PIN));
});

test("entries: they open lazily, are removed only when the runtime holds them, and go through the PIN", async () => {
  const page = await settingsPage({
    "POST /api/settings/entry/remove": { path: "voice.moods.calm", changed: true, layer: "runtime", restart_required: false, persisted: true },
  });
  await open(page, "core-voice");
  const block = page.document.querySelector('[data-entry="voice.moods.calm"]');
  assert.ok(block);
  assert.equal(row(page, "voice.moods.calm.rate"), null, "entry fields render when the entry opens");
  block.querySelector(".sl-disclosure").click();
  assert.ok(row(page, "voice.moods.calm.rate").querySelector('input[type="range"]'));
  button(block, "settings_remove_entry").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings/entry/remove")[0].body, { path: "voice.moods.calm", pin: PIN });
  assert.ok(page.document.querySelector("#notice").textContent.includes("settings_entry_removed"));
});

test("entries the server does not list as removable have no Remove button, whatever their source", async () => {
  const flat = fields().map((f) => (f.path === "voice.moods" ? { ...f, removable: [] } : f));
  const page = await settingsPage({}, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  const block = page.document.querySelector('[data-entry="voice.moods.calm"]');
  assert.equal(buttons(block, "settings_remove_entry").length, 0);
});

test("entries: the add form is built from the entry descriptors and posts key, value and PIN", async () => {
  const page = await settingsPage({
    "POST /api/settings/entry": { path: "voice.moods", changed: true, layer: "runtime", restart_required: true, persisted: true },
  });
  await open(page, "core-voice");
  const box = row(page, "voice.moods");
  const toggle = [...box.querySelectorAll(".sl-disclosure")].find((b) => b.textContent === "settings_add_entry");
  toggle.click();
  const adder = [...box.querySelectorAll(".nested")].pop();
  assert.ok(adder);
  assert.equal(adder.querySelectorAll("input").length >= 3, true);
  assert.ok(!adder.textContent.includes("secret_token"), "secret fields are not offered");
  const name = adder.querySelector('input[type="text"]');
  name.value = "angry";
  button(adder, "settings_add_entry").click();
  await settle();
  assert.equal(adder.querySelector(".sl-field__error").textContent, "err_value_type", "required number left empty");
  assert.equal(posts(page, "/api/settings/entry").length, 0);
  adder.querySelector('input[type="number"]').value = "1.5";
  button(adder, "settings_add_entry").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings/entry")[0].body, { path: "voice.moods", key: "angry", value: { rate: 1.5, label: "" }, pin: PIN });
  assert.ok(page.document.querySelector("#notice").textContent.includes("settings_restart_needed"));
});

test("a keyed list's add form carries the identity in the value, not as a key", async () => {
  const page = await settingsPage({
    "POST /api/settings/entry": { path: "appraisal.rules", changed: true, layer: "runtime", restart_required: false, persisted: true },
  });
  await open(page, "core-appraisal");
  const box = row(page, "appraisal.rules");
  [...box.querySelectorAll(".sl-disclosure")].find((b) => b.textContent === "settings_add_entry").click();
  const form = [...box.querySelectorAll(".nested")].pop();
  const [ident, valence] = form.querySelectorAll("input:not([type=range])");
  ident.value = "user.left";
  valence.value = "-0.5";
  button(form, "settings_add_entry").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings/entry")[0].body, { path: "appraisal.rules", value: { event: "user.left", valence: -0.5 }, pin: PIN });
});

test("a language map has an add form; string_list_dict has none", async () => {
  const flat = fields().concat([d("voice.keywords", "dict_of_model", { value: ["en"], entry: ENTRY, writable: true })]);
  const page = await settingsPage({}, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  assert.equal(buttons(row(page, "voice.keywords"), "settings_add_entry").length, 1);
  await open(page, "core-conversation");
  assert.equal(buttons(row(page, "conversation.confirm_yes"), "settings_add_entry").length, 0);
});

test("server codes map to the strings of the page language, and unknown ones to the generic text", async () => {
  const strings = {
    err_setting_guardrail: "Refused: it would loosen a rule.", err_value_range: "Out of range.", err_generic: "Something went wrong.",
  };
  for (const [code, want] of [["setting_guardrail", strings.err_setting_guardrail], ["value_range", strings.err_value_range], ["never_heard_of", strings.err_generic]]) {
    const page = await settingsPage({
      "GET /api/strings": { language: "it", strings },
      "POST /api/settings": reply(403, { error: code }),
    });
    await open(page, "core-voice");
    const box = row(page, "voice.label");
    box.querySelector("input").value = "other";
    [...box.querySelectorAll("button")].find((b) => b.textContent === "settings_save").click();
    await settle();
    const dialog = await answerPin(page);
    assert.equal(dialog.querySelector("#dialog-error").textContent, want, code);
    assert.equal(dialog.querySelector("#dialog-error").hidden, false);
  }
});

test("the legacy rendering is gone and the page copes with an empty catalogue", async () => {
  const page = await settingsPage({}, { fields: [], pin_required: true });
  assert.equal(page.document.querySelector("nav.sl-nav"), null);
  assert.ok(page.document.querySelector("#view").textContent.includes("settings_none"));
});

test("nav items open their group and mark it as current", async () => {
  const page = await settingsPage();
  const item = page.document.querySelector('.sl-nav__item[data-group="core-safety"]');
  item.click();
  assert.equal(item.getAttribute("aria-current"), "true");
  assert.equal(group(page, "core-safety").getAttribute("aria-expanded"), "true");
  assert.ok(body(page, "core-safety").children.length > 0);
});

test("the debug tab splits robot sections from the core ones", async () => {
  const page = await loadPage({
    "GET /api/debug": { sections: { system: { scope: "local" }, platform: { scope: "robot", ok: true } } },
  });
  await page.run("debugView()");
  const blocks = [...page.document.querySelectorAll(".sl-term-block__out")].map((b) => JSON.parse(b.textContent));
  assert.equal(blocks.length, 2);
  assert.deepEqual(Object.keys(blocks[0].sections), ["system"]);
  assert.deepEqual(Object.keys(blocks[1]), ["platform"]);
});

const authed = {
  "GET /api/session": { authenticated: true, pin_set: true, csrf: "tok" },
  "GET /api/enrollment/status": { state: "idle" },
};

test("logout clears the revealed values, the catalogue and the DOM before the API call, even if it fails", async () => {
  const revealed = d("privacy.note", "text", { value: "TOPSECRET", private: true });
  const page = await loadPage({
    ...authed,
    "GET /api/settings": { fields: fields(), pin_required: true },
    "POST /api/settings/reveal": { fields: [revealed] },
    "POST /api/logout": reply(500, { error: "generic" }),
  });
  await page.run("showTab('settings')");
  await settle();
  await open(page, "core-privacy");
  button(row(page, "privacy.note"), "settings_reveal").click();
  await settle();
  await answerPin(page);
  assert.ok(page.document.body.textContent.includes("TOPSECRET") || row(page, "privacy.note").querySelector("input").value === "TOPSECRET");
  page.document.querySelector("#logout").click();
  await settle();
  assert.ok(posts(page, "/api/logout").length === 1);
  assert.equal(page.run("revealedSettings.size"), 0);
  assert.equal(page.run("settingsState.groups.length"), 0);
  assert.equal(page.run("settingsState.data"), null);
  assert.equal(page.run("settingsState.query"), "");
  assert.equal(page.run("settingsState.open.size + settingsState.collapsed.size"), 0);
  const dump = page.run("JSON.stringify([...revealedSettings], null, 0) + JSON.stringify(settingsState.data)");
  assert.ok(!dump.includes("TOPSECRET"));
  assert.ok(!page.document.documentElement.outerHTML.includes("TOPSECRET"));
  for (const i of page.document.querySelectorAll("input")) assert.notEqual(i.value, "TOPSECRET");
});

test("entry keys that are prefixes of each other keep their own fields", async () => {
  const flat = [
    d("voice.moods", "dict_of_model", { value: ["a", "a.b"], entry: ENTRY, writable: false }),
    d("voice.moods.a.rate", "number", { value: 1, ge: 0, le: 2 }),
    d("voice.moods.a.b.rate", "number", { value: 2, ge: 0, le: 2 }),
  ];
  const page = await settingsPage({}, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  for (const entry of page.document.querySelectorAll("[data-entry] .sl-disclosure")) entry.click();
  const inside = (key) => [...page.document.querySelector(`[data-entry="voice.moods.${key}"]`).querySelectorAll(".setting")].map((r) => r.dataset.path);
  assert.deepEqual(inside("a"), ["voice.moods.a.rate"]);
  assert.deepEqual(inside("a.b"), ["voice.moods.a.b.rate"]);
});

test("search is debounced, needs two characters to open groups, and caps what it opens", async () => {
  const bulk = Array.from({ length: 60 }, (_, n) => d("bulk.f" + n, "bool", { value: false, description: "needle " + n }));
  const page = await settingsPage({ "GET /api/strings": { language: "en", strings: { settings_show_all: "All {n}" } } }, { fields: fields().concat(bulk), pin_required: true });
  const search = page.document.querySelector("#settings-search");
  search.value = "needle";
  search.dispatchEvent(new page.window.Event("input"));
  assert.equal(group(page, "core-bulk").getAttribute("aria-expanded"), "false", "nothing before the delay");
  await sleep(200);
  assert.equal(group(page, "core-bulk").getAttribute("aria-expanded"), "true");
  assert.equal(body(page, "core-bulk").querySelectorAll(".setting").length, 50);
  const all = button(body(page, "core-bulk"), "All 60");
  assert.ok(all);
  all.click();
  assert.equal(body(page, "core-bulk").querySelectorAll(".setting").length, 60);

  await typeQuery(page, "n");
  assert.equal(group(page, "core-bulk").getAttribute("aria-expanded"), "false", "one character does not auto-open");
});

test("unsaved edits survive a collapse and a filter change", async () => {
  const page = await settingsPage();
  await open(page, "core-voice");
  row(page, "voice.label").querySelector("input").value = "draft";
  group(page, "core-voice").click();
  group(page, "core-voice").click();
  assert.equal(row(page, "voice.label").querySelector("input").value, "draft");
  await typeQuery(page, "voice.label");
  await typeQuery(page, "");
  assert.equal(row(page, "voice.label").querySelector("input").value, "draft");
});

test("secret json is never printed", async () => {
  const flat = [d("voice.cfg", "json", { value: { token: "LEAK" }, secret: true, writable: false })];
  const page = await settingsPage({}, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  assert.ok(!page.document.body.textContent.includes("LEAK"));
  assert.equal(row(page, "voice.cfg").querySelector("pre"), null);
});

test("the add form skips json fields and sends their default", async () => {
  const entry = [d("rate", "number", { value: null, ge: 0, le: 2 }), d("channels", "json", { value: { a: 1 } })];
  const flat = [d("voice.moods", "dict_of_model", { value: [], entry, writable: false })];
  const page = await settingsPage({ "POST /api/settings/entry": { changed: true, layer: "runtime", persisted: true } }, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  const box = row(page, "voice.moods");
  [...box.querySelectorAll(".sl-disclosure")].find((b) => b.textContent === "settings_add_entry").click();
  const form = [...box.querySelectorAll(".nested")].pop();
  assert.equal(form.querySelectorAll("textarea").length, 0);
  form.querySelector('input[type="text"]').value = "k";
  form.querySelector('input[type="number"]').value = "1";
  button(form, "settings_add_entry").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings/entry")[0].body.value, { channels: { a: 1 }, rate: 1 });
});

test("dict rows reject prototype keys, blank keys and duplicates", async () => {
  const page = await settingsPage();
  await open(page, "core-voice");
  const box = row(page, "voice.names");
  const save = async (keys) => {
    const rows = box.querySelectorAll(".entry .entry");
    const have = rows.length;
    for (let n = have; n < keys.length; n++) button(box, "settings_add").click();
    const ins = [...box.querySelectorAll('input[type="text"]')];
    keys.forEach((k, n) => { ins[n * 2].value = k; ins[n * 2 + 1].value = "v"; });
    button(box, "settings_save").click();
    await settle();
    return box.querySelector(".sl-field__error").textContent;
  };
  assert.equal(await save(["__proto__"]), "err_key_invalid");
  assert.equal(await save(["constructor"]), "err_key_invalid");
  assert.equal(await save(["x", "x"]), "err_key_duplicate");
  assert.equal(page.document.querySelector("dialog"), null);
});

test("an enum outside its choices shows a blank and never sends choice 0", async () => {
  const flat = [d("voice.mode", "enum", { value: "zzz", choices: ["a", "b"] })];
  const page = await settingsPage({}, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  const box = row(page, "voice.mode");
  const select = box.querySelector("select");
  assert.equal(select.value, "");
  button(box, "settings_save").click();
  await settle();
  assert.equal(box.querySelector(".sl-field__error").textContent, "err_value_choice");
  assert.equal(page.document.querySelector("dialog"), null);
});

test("no slider when the limits are equal; chips are not toggles; the description is linked", async () => {
  const flat = [
    d("voice.fixed", "number", { value: 1, ge: 1, le: 1 }),
    d("voice.tags", "string_list", { value: ["x"], description: "Tags" }),
    d("voice.label", "text", { value: "a", description: "A label" }),
  ];
  const page = await settingsPage({}, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  assert.equal(row(page, "voice.fixed").querySelector('input[type="range"]'), null);
  assert.equal(row(page, "voice.tags").querySelector(".sl-chip").hasAttribute("aria-pressed"), false);
  const input = row(page, "voice.label").querySelector("input");
  const desc = row(page, "voice.label").querySelector(".hint");
  assert.equal(input.getAttribute("aria-describedby"), desc.id);
});

test("hostile text in descriptions, keys and the robot title stays text", async () => {
  const evil = '<img src=x onerror="window.pwned=1">';
  const flat = [
    d("voice.moods", "dict_of_model", { value: [evil], entry: ENTRY, writable: false, description: evil }),
    d("voice.moods." + evil + ".rate", "number", { value: 1, ge: 0, le: 2 }),
    d("voice.names", "string_dict", { value: { [evil]: evil } }),
  ];
  const robot = { id: "r", title: evil, fields: [d("robot.r.limits.a", "number", { value: 1, writable: false })], actions: [] };
  const page = await settingsPage({}, { fields: flat, robot, pin_required: true });
  await open(page, "core-voice");
  await open(page, "robot-limits");
  assert.equal(page.document.querySelectorAll("img").length, 0);
  assert.equal(page.window.pwned, undefined);
  assert.ok(page.document.body.textContent.includes(evil));
});

test("the null toggle disables the control, restores it, and sends null or the value", async () => {
  const page = await settingsPage({ "POST /api/settings": { changed: true, layer: "runtime", persisted: true } });
  await open(page, "core-voice");
  const box = row(page, "voice.opt2");
  const toggle = box.querySelector('input[type="checkbox"]');
  const num = box.querySelector('input[type="number"]');
  assert.equal(toggle.checked, false);
  assert.equal(num.disabled, false);
  toggle.click();
  assert.equal(num.disabled, true);
  assert.equal(box.querySelector('input[type="range"]').disabled, true);
  toggle.click();
  assert.equal(num.disabled, false);
  num.value = "4";
  button(box, "settings_save").click();
  await settle();
  await answerPin(page);
  assert.equal(posts(page, "/api/settings")[0].body.value, 4);
});

test("a tuple with null items renders blanks and refuses to save until filled", async () => {
  const flat = [d("voice.pair", "tuple", { value: [null, 2], item_kind: "number", length: 2 })];
  const page = await settingsPage({}, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  const box = row(page, "voice.pair");
  const ins = box.querySelectorAll('input[type="number"]');
  assert.equal(ins[0].value, "");
  button(box, "settings_save").click();
  await settle();
  assert.equal(box.querySelector(".sl-field__error").textContent, "err_value_type");
});

test("the debug tab works without a robot section", async () => {
  const page = await loadPage({ "GET /api/debug": { sections: { system: { scope: "local" } } } });
  await page.run("debugView()");
  const blocks = page.document.querySelectorAll(".sl-term-block__out");
  assert.equal(blocks.length, 1);
  assert.deepEqual(Object.keys(JSON.parse(blocks[0].textContent).sections), ["system"]);
  assert.ok(![...page.document.querySelectorAll("h3")].some((h) => h.textContent === "settings_robot"));
});

const RESTARTABLE = { ...ROBOT, actions: ["restart"], restart_effects: [
  { id: "zero_pose", params: {} }, { id: "remote_session_drop", params: {} }, { id: "kill_on_timeout", params: { timeout_s: 20 } },
] };
const restartPage = (extra = {}) =>
  settingsPage({ "POST /api/robot/restart": { ok: true }, ...extra }, { fields: fields(), robot: RESTARTABLE, pin_required: true });

test("restart: the button exists only with the action, the dialog lists the side effects, the PIN is posted", async () => {
  const page = await restartPage();
  const btn = page.document.querySelector("#settings-robot-actions #settings-restart");
  assert.equal(btn.textContent, "settings_restart");
  btn.click();
  await settle();
  const dialog = page.document.querySelector("dialog");
  assert.ok(dialog, "confirm dialog");
  assert.equal(dialog.querySelector("#dialog-title").textContent, "settings_restart_title");
  assert.deepEqual([...dialog.querySelectorAll("li")].map((li) => li.textContent), [
    "settings_restart_effect_zero_pose", "settings_restart_effect_remote_session_drop", "settings_restart_effect_kill_on_timeout",
  ]);
  assert.equal(posts(page, "/api/robot/restart").length, 0, "nothing is sent before the PIN");
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/robot/restart").map((c) => c.body), [{ pin: PIN }]);
  assert.equal(page.document.querySelector("#settings-restart").disabled, true, "no second restart after success");
  assert.equal(page.document.querySelector("dialog"), null);
  assert.equal(page.document.querySelector("#notice").textContent, "settings_restart_started");
  assert.ok(!storageDump(page.window).includes(PIN));
  assert.ok(!page.document.documentElement.outerHTML.includes(PIN));
});

test("restart: cancelling sends nothing, and a wrong PIN or an unavailable restart is explained in the dialog", async () => {
  let reply404 = false;
  const page = await restartPage({
    "GET /api/strings": { language: "en", strings: { err_bad_pin: "Wrong PIN.", err_restart_unavailable: "Not now." } },
    "POST /api/robot/restart": () => (reply404 ? reply(404, { error: "restart_unavailable" }) : reply(401, { error: "bad_pin" })),
  });
  const start = async () => { page.document.querySelector("#settings-restart").click(); await settle(); };
  await start();
  buttons(page.document.querySelector("dialog"), "dialog_cancel")[0].click();
  await settle();
  assert.equal(page.document.querySelector("dialog"), null);
  assert.equal(posts(page, "/api/robot/restart").length, 0);
  await start();
  let dialog = await answerPin(page);
  assert.equal(dialog.querySelector("#dialog-error").textContent, "Wrong PIN.");
  reply404 = true;
  await answerPin(page);
  assert.equal(dialog.querySelector("#dialog-error").textContent, "Not now.");
  assert.notEqual(page.document.querySelector("#notice").textContent, "settings_restart_started");
});

test("restart: the badge of a restart field links to the button; without the action it is plain", async () => {
  const page = await restartPage();
  await open(page, "core-voice");
  const link = row(page, "voice.flag").querySelector(".sl-badge--warning").closest("a");
  assert.ok(link, "badge is a link");
  assert.equal(link.getAttribute("href"), "#settings-restart");
  assert.equal(link.getAttribute("aria-label"), "settings_restart_goto");
  assert.equal(link.getAttribute("title"), "settings_restart_goto");
  let focused = null;
  const btn = page.document.querySelector("#settings-restart");
  btn.focus = () => { focused = btn; };
  const click = new page.window.MouseEvent("click", { bubbles: true, cancelable: true });
  link.dispatchEvent(click);
  assert.equal(click.defaultPrevented, true);
  assert.equal(focused, btn, "the link moves focus to the button");
  assert.equal(page.document.querySelector("dialog"), null, "the link does not start a restart by itself");
});

test("restart: effect text and the robot title stay text, never markup", async () => {
  const page = await settingsPage({}, {
    fields: fields(), pin_required: true,
    robot: { ...RESTARTABLE, restart_effects: [{ id: "<img src=x onerror=alert(1)>", params: {} }] },
  });
  page.document.querySelector("#settings-restart").click();
  await settle();
  const dialog = page.document.querySelector("dialog");
  assert.equal(dialog.querySelectorAll("img").length, 0);
  assert.equal(dialog.querySelectorAll("li").length, 1);
});

test("restart: the {timeout_s} placeholder is filled from the effect params", async () => {
  const page = await restartPage({
    "GET /api/strings": { language: "en", strings: { settings_restart_effect_kill_on_timeout: "killed after {timeout_s} s" } },
  });
  page.document.querySelector("#settings-restart").click();
  await settle();
  const items = [...page.document.querySelectorAll("dialog li")].map((li) => li.textContent);
  assert.ok(items.includes("killed after 20 s"));
});

test("restart: the button is rebuilt on refresh, so it appears and disappears with robot.actions", async () => {
  const page = await restartPage();
  assert.ok(page.document.querySelector("#settings-restart"));
  page.run("settingsState.data.robot.actions = []; refreshSettings()");
  assert.equal(page.document.querySelector("#settings-restart"), null);
  page.run("settingsState.data.robot.actions = ['restart']; refreshSettings()");
  assert.ok(page.document.querySelector("#settings-restart"));
});

test("a revealed private container brings its removable list", async () => {
  const hidden = d("safety.minor_lines", "dict_of_model", { value: [], private: true, admin_only: true, removable: [], writable: false });
  const shown = d("safety.minor_lines", "dict_of_model", { value: ["xx"], private: true, admin_only: true, removable: ["xx"], entry: ENTRY, writable: false });
  const page = await settingsPage({ "POST /api/settings/reveal": { fields: [shown, d("safety.minor_lines.xx.rate", "number", { value: 1, private: true, admin_only: true })] } },
    { fields: [hidden], pin_required: true });
  await open(page, "core-safety");
  button(row(page, "safety.minor_lines"), "settings_reveal").click();
  await settle();
  await answerPin(page);
  const block = page.document.querySelector('[data-entry="safety.minor_lines.xx"]');
  assert.equal(buttons(block, "settings_remove_entry").length, 1);
});

test("saving one row keeps the unsaved edits of the other rows", async () => {
  let saved = false;
  const page = await settingsPage({
    "GET /api/settings": () => ({
      fields: fields().map((f) => (f.path === "voice.count" && saved ? { ...f, value: 5, source: "runtime" } : f)),
      pin_required: true,
    }),
    "POST /api/settings": () => { saved = true; return { path: "voice.count", changed: true, value: 5, layer: "runtime", persisted: true }; },
  });
  await open(page, "core-voice");
  row(page, "voice.label").querySelector("input").value = "draft";
  row(page, "voice.count").querySelector('input[type="number"]').value = "5";
  button(row(page, "voice.count"), "settings_save").click();
  await settle();
  await answerPin(page);
  assert.equal(row(page, "voice.count").querySelector('input[type="number"]').value, "5");
  assert.equal(row(page, "voice.count").dataset.source, "runtime", "the saved row shows the new descriptor");
  assert.equal(row(page, "voice.label").querySelector("input").value, "draft", "the other row keeps its edit");
});

const URL_ROW = d("llm_routes.cloud.base_url", "text", { secret: true, writable: true, value: "***", default: "***" });

test("a write-only URL shows an empty box, sends nothing when left empty and posts a whole new value", async () => {
  const page = await settingsPage({ "POST /api/settings": { path: URL_ROW.path, changed: true, layer: "runtime", persisted: true } },
    { fields: [URL_ROW], pin_required: true });
  await open(page, "core-llm_routes");
  const box = row(page, URL_ROW.path);
  const input = box.querySelector("input");
  assert.equal(input.value, "");
  assert.equal(input.getAttribute("placeholder"), "***");
  assert.ok(box.textContent.includes("settings_write_only_hint"));
  assert.ok(!box.textContent.includes("settings_default"), "no masked default is printed");
  button(box, "settings_save").click();
  await settle();
  assert.equal(page.document.querySelector("dialog"), null);
  assert.ok(page.document.querySelector("#notice").textContent.includes("settings_unchanged"));
  input.value = "https://cloud.example/v2";
  button(box, "settings_save").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings")[0].body, { path: URL_ROW.path, value: "https://cloud.example/v2", pin: PIN, ack_endpoint: "https://cloud.example" });
});

test("a route moving to another host: the dialog names the host before the first submit, which carries the acknowledgement", async () => {
  const page = await settingsPage({ "POST /api/settings": { path: URL_ROW.path, changed: true, layer: "runtime", persisted: true } },
    { fields: [URL_ROW], pin_required: true });
  await open(page, "core-llm_routes");
  const box = row(page, URL_ROW.path);
  box.querySelector("input").value = "https://Evil.example:8443/v1";
  button(box, "settings_save").click();
  await settle();
  const dialog = page.document.querySelector("dialog");
  assert.ok(dialog.textContent.includes("settings_route_move_message"), "the warning is shown first");
  assert.equal(posts(page, "/api/settings").length, 0, "nothing is sent before the PIN");
  await answerPin(page);
  assert.equal(posts(page, "/api/settings")[0].body.ack_endpoint, "https://evil.example:8443");
});

test("a route move: the 422 reply is the fallback and its endpoint is acknowledged on the retry", async () => {
  const sent = [];
  const page = await settingsPage({
    "GET /api/strings": { language: "en", strings: { err_endpoint_ack_required: "Key goes to {host}." } },
    "POST /api/settings": (body) => {
      sent.push(body);
      return body.ack_endpoint === "https://evil.example"
        ? { path: URL_ROW.path, changed: true, layer: "runtime", persisted: true }
        : reply(422, { error: "endpoint_ack_required", field: "value", endpoint: "https://evil.example" });
    },
  }, { fields: [URL_ROW], pin_required: true });
  await open(page, "core-llm_routes");
  const box = row(page, URL_ROW.path);
  box.querySelector("input").value = "https://evil.example:443/v1";
  button(box, "settings_save").click();
  await settle();
  const dialog = await answerPin(page);
  assert.equal(dialog.querySelector("#dialog-error").textContent, "Key goes to https://evil.example.");
  await answerPin(page);
  assert.equal(page.document.querySelector("dialog"), null);
  assert.equal(sent[0].ack_endpoint, "https://evil.example:443");
  assert.equal(sent[1].ack_endpoint, "https://evil.example");
});

test("a route whose host does not change opens no warning when the old value is shown", async () => {
  const plain = { ...URL_ROW, secret: false, value: "https://cloud.example/v1", default: "https://cloud.example/v1" };
  const page = await settingsPage({ "POST /api/settings": { path: plain.path, changed: true, layer: "runtime", persisted: true } },
    { fields: [plain], pin_required: true });
  await open(page, "core-llm_routes");
  const box = row(page, plain.path);
  box.querySelector("input").value = "https://cloud.example/v2";
  button(box, "settings_save").click();
  await settle();
  assert.ok(!page.document.querySelector("dialog").textContent.includes("settings_route_move_message"));
});

test("the restart availability flipping updates the badge in place and keeps unsaved edits", async () => {
  const page = await restartPage();
  await open(page, "core-voice");
  const label = () => row(page, "voice.label").querySelector("input");
  label().value = "draft";
  const flag = row(page, "voice.flag");
  assert.ok(flag.querySelector("a"), "linked while the action exists");
  page.run("settingsState.data.robot.actions = []; refreshSettings()");
  assert.equal(row(page, "voice.flag").querySelector("a"), null, "plain badge without the action");
  assert.ok(row(page, "voice.flag").querySelector(".sl-badge--warning"));
  assert.equal(label().value, "draft");
  page.run("settingsState.data.robot.actions = ['restart']; refreshSettings()");
  assert.ok(row(page, "voice.flag").querySelector("a"));
  assert.equal(label().value, "draft");
});

test("a language can be added to the blocklist: key and words are posted with the PIN", async () => {
  const lists = [
    d("safety.minor_output_blocklist", "string_list_dict", { value: ["en"], private: true, admin_only: true, writable: false }),
    d("safety.minor_output_blocklist.en", "string_list", { value: ["bad"], private: true, admin_only: true }),
  ];
  const revealed = lists.map((f) => ({ ...f }));
  const page = await settingsPage({
    "POST /api/settings/reveal": { fields: revealed },
    "POST /api/settings/entry": { path: lists[0].path, changed: true, layer: "runtime", persisted: true },
  }, { fields: lists, pin_required: true });
  await open(page, "core-safety");
  button(row(page, lists[0].path), "settings_reveal").click();
  await settle();
  await answerPin(page);
  const container = row(page, lists[0].path);
  const toggles = container.querySelectorAll(".sl-disclosure");
  toggles[toggles.length - 1].click();
  const forms = container.querySelectorAll(".nested");
  const form = forms[forms.length - 1];
  form.querySelector('input[type="text"]').value = "de";
  const words = form.querySelectorAll('input[type="text"]')[1];
  words.value = "mist";
  button(form, "settings_add").click();
  button(form, "settings_add_entry").click();
  await settle();
  await answerPin(page);
  assert.deepEqual(posts(page, "/api/settings/entry")[0].body, { path: lists[0].path, key: "de", value: ["mist"], pin: PIN });
});

