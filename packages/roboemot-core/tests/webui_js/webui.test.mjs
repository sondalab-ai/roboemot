// Behaviour of the web UI page script (SPEC 12 Web UI, 14): the dialog helper `ask()`, the PIN
// handling and the settings editor. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle, storageDump } from "./harness.mjs";

const SECRET_PIN = "482913";

const submit = (window, form) =>
  form.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));

/** Opens `ask(opts)` in the page; returns the promise and the dialog parts. */
function openAsk(page, opts) {
  page.window.__opts = opts;
  const promise = page.run("ask(window.__opts)");
  const dialog = page.document.querySelector("dialog");
  return {
    promise,
    dialog,
    form: dialog.querySelector("form"),
    pin: dialog.querySelector("#dialog-pin"),
    error: dialog.querySelector("#dialog-error"),
    cancel: dialog.querySelector(".sl-dialog__actions button[type=button]"),
    submitButton: dialog.querySelector(".sl-dialog__actions button[type=submit]"),
  };
}

test("ask(): Cancel closes without ever calling run, and resolves null", async () => {
  const page = await loadPage();
  let ran = 0;
  const d = openAsk(page, { title: "t", pin: true, run: async () => { ran++; } });
  d.pin.value = SECRET_PIN;
  d.cancel.click();
  assert.equal(await d.promise, null);
  assert.equal(ran, 0);
  assert.equal(page.document.querySelector("dialog"), null, "dialog removed from the page");
});

test("ask(): Esc and Cancel are refused while the request is in flight, then it resolves true", async () => {
  const page = await loadPage();
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const d = openAsk(page, { title: "t", pin: true, run: () => gate });
  d.pin.value = SECRET_PIN;
  submit(page.window, d.form);
  await settle();
  assert.equal(d.cancel.disabled, true);
  assert.equal(d.submitButton.disabled, true);
  const esc = new page.window.Event("cancel", { cancelable: true });
  d.dialog.dispatchEvent(esc);
  assert.equal(esc.defaultPrevented, true, "Esc is prevented while pending");
  assert.ok(d.dialog.hasAttribute("open"));
  release();
  assert.equal(await d.promise, true);
  assert.equal(page.document.querySelector("dialog"), null);
});

test("ask(): Esc is allowed again once the request has failed", async () => {
  const page = await loadPage();
  const d = openAsk(page, {
    title: "t", pin: true,
    run: async () => { throw Object.assign(new Error("bad_pin"), { status: 403 }); },
  });
  d.pin.value = SECRET_PIN;
  submit(page.window, d.form);
  await settle();
  const esc = new page.window.Event("cancel", { cancelable: true });
  d.dialog.dispatchEvent(esc);
  assert.equal(esc.defaultPrevented, false);
  d.cancel.click();
  assert.equal(await d.promise, null);
});

test("PIN: read once, cleared before the call runs, never stored", async () => {
  const page = await loadPage();
  let seen = null;
  let inputAtCall = null;
  const d = openAsk(page, {
    title: "t", pin: true,
    run: async (args) => { seen = args.pin; inputAtCall = d.pin.value; },
  });
  d.pin.value = SECRET_PIN;
  submit(page.window, d.form);
  assert.equal(d.pin.value, "", "the input is cleared as soon as the PIN is read");
  assert.equal(await d.promise, true);
  assert.equal(seen, SECRET_PIN);
  assert.equal(inputAtCall, "");
  assert.ok(!storageDump(page.window).includes(SECRET_PIN));
  assert.ok(!page.document.documentElement.outerHTML.includes(SECRET_PIN));
});

test("PIN: a wrong PIN stays cleared, is flagged on the PIN field, and nothing is stored", async () => {
  const page = await loadPage({ "GET /api/strings": { language: "en", strings: { err_bad_pin: "Wrong PIN." } } });
  const d = openAsk(page, {
    title: "t", pin: true,
    run: async () => { throw Object.assign(new Error("bad_pin"), { status: 403 }); },
  });
  d.pin.value = SECRET_PIN;
  submit(page.window, d.form);
  await settle();
  assert.equal(d.pin.value, "");
  assert.equal(d.pin.getAttribute("aria-invalid"), "true");
  assert.equal(d.pin.getAttribute("aria-describedby"), "dialog-error");
  assert.equal(d.error.hidden, false);
  assert.equal(d.error.textContent, "Wrong PIN.");
  assert.ok(d.dialog.hasAttribute("open"), "stays open so the PIN can be retried");
  assert.ok(!storageDump(page.window).includes(SECRET_PIN));
  d.cancel.click();
  await d.promise;
});

test("PIN: the login form clears the PIN field and sends it only in the request body", async () => {
  const page = await loadPage({
    "POST /api/login": { csrf: "tok" },
    "GET /api/session": { authenticated: false, pin_set: true },
  });
  const input = page.document.querySelector("#pin");
  input.value = SECRET_PIN;
  submit(page.window, input.closest("form"));
  await settle();
  assert.equal(input.value, "");
  const login = page.calls.find((c) => c.path === "/api/login");
  assert.deepEqual(login.body, { pin: SECRET_PIN });
  assert.ok(!storageDump(page.window).includes(SECRET_PIN));
});

const LIST = {
  path: "voice.keywords.stop", value: ["stop", "basta"], default: ["stop"], needs_admin: false,
  schema: { type: "array", minItems: 1, maxItems: 3, items: { maxLength: 8 } },
};
const LINE = {
  path: "voice.wake_word", value: "robo", default: null, needs_admin: true,
  schema: { type: ["string", "null"], maxLength: 6 },
};

async function settingsPage(extra = {}) {
  const page = await loadPage({ "GET /api/settings": { settings: [LIST, LINE] }, ...extra });
  await page.run("settingsView()");
  return page;
}

const editButton = (page, path) =>
  page.document.querySelector(`[data-path="${path}"] button`);

test("settings: the editor is built from the schema's limits", async () => {
  const page = await settingsPage();
  const doc = page.document;
  assert.ok(doc.querySelector('[data-path="voice.keywords.stop"] .setting-value'));
  editButton(page, "voice.keywords.stop").click();
  const box = doc.querySelector('[data-path="voice.keywords.stop"]');
  const inputs = [...box.querySelectorAll("input")];
  assert.equal(inputs.length, 2);
  assert.ok(inputs.every((i) => i.getAttribute("maxlength") === "8"));
  const add = [...box.querySelectorAll("button")].find((b) => b.textContent === "settings_add");
  add.click();
  assert.equal(box.querySelectorAll("input").length, 3);
  assert.equal(add.disabled, true, "maxItems reached");
  const removes = [...box.querySelectorAll(".entry button")];
  removes.slice(1).forEach((b) => b.click());
  const last = box.querySelector(".entry button");
  assert.equal(last.disabled, true, "minItems reached");

  editButton(page, "voice.wake_word").click();
  const line = doc.querySelector('[data-path="voice.wake_word"] input');
  assert.equal(line.getAttribute("maxlength"), "6");
  assert.equal(line.value, "robo");
});

test("settings: a value over the limit is flagged on its field and nothing is sent", async () => {
  const page = await settingsPage();
  editButton(page, "voice.wake_word").click();
  const box = page.document.querySelector('[data-path="voice.wake_word"]');
  const input = box.querySelector("input");
  input.value = "much too long";
  submit(page.window, box.querySelector("form"));
  await settle();
  const failure = box.querySelector("#set-error-1");
  assert.equal(failure.hidden, false);
  assert.equal(failure.textContent, "err_line_too_long");
  assert.equal(input.getAttribute("aria-invalid"), "true");
  assert.equal(input.getAttribute("aria-describedby"), "set-error-1");
  assert.equal(page.document.querySelector("dialog"), null, "no PIN prompt for an invalid value");
  assert.ok(!page.calls.some((c) => c.method === "POST" && c.path === "/api/settings"));
});

test("settings: an over-long list entry is flagged on that entry only", async () => {
  const page = await settingsPage();
  editButton(page, "voice.keywords.stop").click();
  const box = page.document.querySelector('[data-path="voice.keywords.stop"]');
  const [first, second] = box.querySelectorAll("input");
  second.value = "waytoolongphrase";
  submit(page.window, box.querySelector("form"));
  await settle();
  assert.equal(second.getAttribute("aria-invalid"), "true");
  assert.equal(first.getAttribute("aria-invalid"), null);
});

test("settings: save asks for the PIN, posts path/value/PIN, and a server refusal maps to the PIN field", async () => {
  let attempt = 0;
  const page = await settingsPage({
    "GET /api/strings": { language: "en", strings: { err_bad_pin: "Wrong PIN." } },
    "POST /api/settings": () => (++attempt === 1 ? reply(403, { error: "bad_pin" }) : { changed: true, persisted: true }),
  });
  editButton(page, "voice.wake_word").click();
  const box = page.document.querySelector('[data-path="voice.wake_word"]');
  box.querySelector("input").value = "  nuovo  ";
  submit(page.window, box.querySelector("form"));
  await settle();

  const dialog = page.document.querySelector("dialog");
  const pin = dialog.querySelector("#dialog-pin");
  pin.value = SECRET_PIN;
  submit(page.window, dialog.querySelector("form"));
  await settle();
  assert.deepEqual(page.calls.filter((c) => c.path === "/api/settings" && c.method === "POST")[0].body,
    { path: "voice.wake_word", value: "nuovo", pin: SECRET_PIN });
  assert.equal(pin.getAttribute("aria-invalid"), "true");
  assert.equal(dialog.querySelector("#dialog-error").textContent, "Wrong PIN.");

  pin.value = SECRET_PIN;
  submit(page.window, dialog.querySelector("form"));
  await settle();
  assert.equal(page.document.querySelector("dialog"), null, "closed after the accepted save");
  assert.equal(page.document.querySelector("#notice").textContent, "saved");
  assert.ok(!storageDump(page.window).includes(SECRET_PIN));
});

test("settings: an unchanged value is not sent and asks for no PIN", async () => {
  const page = await settingsPage();
  editButton(page, "voice.wake_word").click();
  const box = page.document.querySelector('[data-path="voice.wake_word"]');
  submit(page.window, box.querySelector("form"));
  await settle();
  assert.equal(page.document.querySelector("dialog"), null);
  assert.equal(page.document.querySelector("#notice").textContent, "settings_unchanged");
});
