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
    item_kind: null, length: null, key: null, entry: null, secret: false, private: false, writable: true, ...extra,
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
    d("voice.moods", "dict_of_model", { value: ["calm"], entry: ENTRY, writable: false }),
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
  assert.equal(page.document.querySelector("#settings-robot-actions").children.length, 0, "no actions rendered yet");
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
  const search = page.document.querySelector("#settings-search");
  search.value = "speech volume";
  search.dispatchEvent(new page.window.Event("input"));
  assert.equal(group(page, "core-voice").getAttribute("aria-expanded"), "true");
  assert.deepEqual([...body(page, "core-voice").querySelectorAll(".setting")].map((r) => r.dataset.path), ["voice.volume_db"]);
  assert.equal(page.document.querySelector("#grp-core-privacy").closest("section").hidden, true);
  assert.equal(page.document.querySelector('.sl-nav__item[data-group="core-privacy"]').hidden, true);
  assert.equal(page.document.querySelector("#settings-count").textContent, "1 shown");

  search.value = "valence";
  search.dispatchEvent(new page.window.Event("input"));
  assert.equal(group(page, "core-appraisal").closest("section").hidden, false, "found through a child of a container");

  search.value = "";
  search.dispatchEvent(new page.window.Event("input"));
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
  const restart = row(page, "voice.flag").querySelector('a[href="#settings-restart"]');
  assert.ok(restart, "restart badge links to the hook");
  const hook = page.document.querySelector("#settings-restart");
  assert.equal(hook.disabled, true, "placeholder until the restart action exists");
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
  assert.equal(input.getAttribute("aria-describedby"), err.id);
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

test("a superset list cannot lose the words it already has", async () => {
  const page = await settingsPage();
  await open(page, "core-safety");
  const box = row(page, "safety.blocklist_word");
  assert.equal(box.querySelector(".sl-chip").disabled, true);
  assert.equal(buttons(box, "settings_reset_layer").length, 0);
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

test("logging out forgets the revealed values", async () => {
  const revealed = d("privacy.note", "text", { value: "secret note", private: true });
  const page = await settingsPage({ "POST /api/settings/reveal": { fields: [revealed] } });
  await open(page, "core-privacy");
  button(row(page, "privacy.note"), "settings_reveal").click();
  await settle();
  await answerPin(page);
  assert.equal(page.run("revealedSettings.size"), 1);
  page.run("forgetRevealedSettings()");
  assert.equal(page.run("revealedSettings.size"), 0);
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

test("entries without a runtime value have no Remove button", async () => {
  const flat = fields().map((f) => (f.path.startsWith("voice.moods.") ? { ...f, source: "schema" } : f));
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

test("no add form for the language maps or for string_list_dict", async () => {
  const flat = fields().concat([d("voice.keywords", "dict_of_model", { value: ["en"], entry: ENTRY, writable: false })]);
  const page = await settingsPage({}, { fields: flat, pin_required: true });
  await open(page, "core-voice");
  assert.equal(buttons(row(page, "voice.keywords"), "settings_add_entry").length, 0);
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
