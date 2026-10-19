// Behaviour of the web UI page script (SPEC 12 Web UI, 14): the dialog helper `ask()`, the PIN
// handling. The settings page is covered in settings.test.mjs. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle, storageDump } from "./harness.mjs";

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
