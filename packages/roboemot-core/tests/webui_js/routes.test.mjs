// Behaviour of the Routes tab (SPEC 8 Routes and Secrets, 11A, 12): the write-only key form, the
// route form and its key choice. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle, storageDump } from "./harness.mjs";

const PIN = "482913";
const CANARY = "sk-JS-CANARY-3d8f1a";

const submit = (window, form) =>
  form.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));

const SECRETS = {
  secrets: [
    { name: "cloud_key", set: true, used_by: ["cloud"] },
    { name: "spare", set: true, used_by: [] },
  ],
  max_value_chars: 4096,
};

async function routesPage(extra = {}) {
  const page = await loadPage({
    "GET /api/routes": { routes: [] },
    "GET /api/secrets": SECRETS,
    ...extra,
  });
  await page.run("routesView()");
  return page;
}

const posts = (page, path) => page.calls.filter((c) => c.method === "POST" && c.path === path);

function fillSecret(page, name = "new_key", value = CANARY, pin = PIN) {
  const doc = page.document;
  doc.querySelector("#secret-name").value = name;
  doc.querySelector("#secret-value").value = value;
  doc.querySelector("#secret-pin").value = pin;
  return doc.querySelector("#secret-value").closest("form");
}

const noTrace = (page, text) => {
  assert.ok(!page.document.documentElement.outerHTML.includes(text), "not in the DOM");
  assert.ok(!storageDump(page.window).includes(text), "not in web storage");
};

test("secrets: the value field is a password input that autocomplete leaves alone", async () => {
  const page = await routesPage();
  const value = page.document.querySelector("#secret-value");
  assert.equal(value.type, "password");
  assert.equal(value.getAttribute("autocomplete"), "off");
  assert.equal(value.getAttribute("maxlength"), "4096", "the server's limit, not a literal in the page");
  assert.equal(value.closest("form").getAttribute("autocomplete"), "off");
});

test("secrets: value and PIN are cleared before the request is sent and are in no trace after success", async () => {
  let seen = null;
  const page = await routesPage({
    "POST /api/secrets/set": (b) => { seen = b; return { name: b.name, set: true }; },
  });
  const form = fillSecret(page);
  submit(page.window, form);
  assert.equal(page.document.querySelector("#secret-value").value, "", "emptied synchronously");
  assert.equal(page.document.querySelector("#secret-pin").value, "");
  await settle();
  await settle();
  assert.deepEqual(seen, { name: "new_key", value: CANARY, pin: PIN });
  assert.equal(page.document.querySelector("#secret-value").value, "");
  noTrace(page, CANARY);
  noTrace(page, PIN);
});

test("secrets: after an error the fields stay empty and the page has no trace of the value", async () => {
  const page = await routesPage({
    "GET /api/strings": { language: "en", strings: { err_secret_value_invalid: "Bad key." } },
    "POST /api/secrets/set": reply(422, { error: "secret_value_invalid", field: "value" }),
  });
  submit(page.window, fillSecret(page));
  await settle();
  await settle();
  assert.equal(page.document.querySelector("#secret-value").value, "");
  assert.equal(page.document.querySelector("#secret-pin").value, "");
  assert.equal(page.document.querySelector("#notice").textContent, "Bad key.");
  noTrace(page, CANARY);
  noTrace(page, PIN);
});

test("secrets: the list shows names and users; a key a route uses cannot be removed", async () => {
  const page = await routesPage();
  const rows = [...page.document.querySelectorAll("tbody tr")];
  assert.equal(rows.length, 2);
  assert.match(rows[0].textContent, /cloud_key/);
  assert.equal(rows[0].querySelector("button").disabled, true);
  assert.equal(rows[1].querySelector("button").disabled, false);
});

test("secrets: Remove asks for the PIN and posts only the name and PIN", async () => {
  const page = await routesPage({ "POST /api/secrets/remove": { name: "spare", set: false } });
  page.document.querySelectorAll("tbody tr")[1].querySelector("button").click();
  await settle();
  const dialog = page.document.querySelector("dialog");
  dialog.querySelector("#dialog-pin").value = PIN;
  submit(page.window, dialog.querySelector("form"));
  await settle();
  await settle();
  assert.deepEqual(posts(page, "/api/secrets/remove")[0].body, { name: "spare", pin: PIN });
  noTrace(page, PIN);
});

test("secrets: the card is absent when the server has no secret store", async () => {
  const page = await routesPage({ "GET /api/secrets": reply(503, { error: "secrets_unavailable" }) });
  assert.equal(page.document.querySelector("#secret-value"), null);
  assert.ok(page.document.querySelector("#view").textContent.length > 0);
});
