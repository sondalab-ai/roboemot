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

// ---------------------------------------------------------------- the route form

const STRINGS = { language: "en", strings: { route_key_message: "Key to {host}", route_minors_hint: "NOT CLEARABLE" } };
const ROUTE = {
  name: "lan", model: "m", endpoint: "api.example.com", base_url: "https://api.example.com/v1", locality: "auto",
  key: "none", allowed_for_minors: false, running: false, secret: null, last_check: null, removable: false,
};
const ADDED = { ...ROUTE, name: "cloud", key: "secret", secret: { name: "cloud_key", set: true }, locality: "cloud", removable: true };

async function formsPage(list, extra = {}) {
  return routesPage({
    "GET /api/strings": STRINGS,
    "GET /api/routes": { routes: list, name_max_chars: 32 },
    ...extra,
  });
}

async function answerPin(page, pin = PIN) {
  const dialog = page.document.querySelector("dialog");
  dialog.querySelector("#dialog-pin").value = pin;
  submit(page.window, dialog.querySelector("form"));
  await settle();
  await settle();
  return dialog;
}

const sel = (page, id) => page.document.querySelector("#" + id);

test("route form: a new route defaults to cloud, cannot be cleared for minors and offers only stored keys", async () => {
  const page = await formsPage([]);
  assert.equal(sel(page, "route-new-locality").value, "cloud");
  assert.equal(sel(page, "route-new-minors").disabled, true);
  assert.match(page.document.querySelector("#view").textContent, /NOT CLEARABLE/);
  const options = [...sel(page, "route-new-key").options].map((o) => o.value);
  assert.deepEqual(options, ["", "cloud_key", "spare"]);
  assert.equal(sel(page, "route-new-key").value, "");
});

test("route form: adding with a key names the host in the PIN dialog and sends the acknowledgement", async () => {
  let seen = null;
  const page = await formsPage([], { "POST /api/routes/add": (b) => { seen = b; return { changed: true }; } });
  sel(page, "route-new-name").value = "cloud2";
  sel(page, "route-new-url").value = "https://API.Cloudco.example/v1";
  sel(page, "route-new-model").value = "big";
  sel(page, "route-new-key").value = "cloud_key";
  submit(page.window, sel(page, "route-new-name").closest("form"));
  await settle();
  const dialog = page.document.querySelector("dialog");
  assert.match(dialog.textContent, /Key to https:\/\/api\.cloudco\.example/);
  await answerPin(page);
  assert.deepEqual(seen, {
    name: "cloud2", base_url: "https://API.Cloudco.example/v1", model: "big", locality: "cloud",
    allowed_for_minors: false, secret: "cloud_key", pin: PIN, ack_endpoint: "https://api.cloudco.example",
  });
  noTrace(page, PIN);
});

test("route form: adding without a key sends no acknowledgement and no secret", async () => {
  let seen = null;
  const page = await formsPage([], { "POST /api/routes/add": (b) => { seen = b; return { changed: true }; } });
  sel(page, "route-new-name").value = "plain";
  sel(page, "route-new-url").value = "https://x.example/v1";
  sel(page, "route-new-model").value = "m";
  submit(page.window, sel(page, "route-new-name").closest("form"));
  await settle();
  await answerPin(page);
  assert.equal(seen.secret, null);
  assert.ok(!("ack_endpoint" in seen));
});

test("route form: only a route of the config file can be cleared for minors; only an added one can be removed", async () => {
  const page = await formsPage([ROUTE, ADDED]);
  assert.equal(sel(page, "route-lan-minors").disabled, false);
  assert.equal(sel(page, "route-cloud-minors").disabled, true);
  const cards = [...page.document.querySelectorAll("section.sl-card")].filter((c) => /lan|cloud/.test(c.querySelector("h2").textContent));
  const removers = (c) => [...c.querySelectorAll("button")].filter((b) => b.textContent === "route_remove");
  assert.equal(removers(cards.find((c) => c.querySelector("h2").textContent === "lan")).length, 0);
  assert.equal(removers(cards.find((c) => c.querySelector("h2").textContent === "cloud")).length, 1);
  assert.equal(sel(page, "route-cloud-key").value, "cloud_key", "the current key is preselected");
});

test("route form: an edit sends only what changed, one request after the other, with one PIN entry", async () => {
  const order = [];
  const page = await formsPage([ROUTE], {
    "POST /api/settings": (b) => { order.push(b.path); return { changed: true }; },
    "POST /api/routes/key": (b) => { order.push("key"); return { changed: true }; },
  });
  sel(page, "route-lan-model").value = "m2";
  sel(page, "route-lan-key").value = "spare";
  sel(page, "route-lan-minors").checked = true;
  submit(page.window, sel(page, "route-lan-model").closest("form"));
  await settle();
  await answerPin(page);
  assert.deepEqual(order, ["key", "llm_routes.lan.allowed_for_minors", "llm_routes.lan.model"]);
  const key = page.calls.find((c) => c.path === "/api/routes/key").body;
  assert.deepEqual(key, { name: "lan", secret: "spare", pin: PIN, ack_endpoint: "https://api.example.com" });
  noTrace(page, PIN);
});

test("route form: moving a keyed route sends the new origin as the acknowledgement", async () => {
  const page = await formsPage([ADDED], { "POST /api/settings": { changed: true } });
  sel(page, "route-cloud-url").value = "https://new.example/v1";
  submit(page.window, sel(page, "route-cloud-url").closest("form"));
  await settle();
  assert.match(page.document.querySelector("dialog").textContent, /Key to https:\/\/new\.example/);
  await answerPin(page);
  const call = page.calls.find((c) => c.path === "/api/settings");
  assert.deepEqual(call.body, { path: "llm_routes.cloud.base_url", value: "https://new.example/v1", pin: PIN, ack_endpoint: "https://new.example" });
});

test("route form: a refused step leaves the dialog open with the error and no PIN behind", async () => {
  const page = await formsPage([ROUTE], {
    "GET /api/strings": { language: "en", strings: { ...STRINGS.strings, err_secret_unknown: "No such key." } },
    "POST /api/routes/key": reply(422, { error: "secret_unknown" }),
  });
  sel(page, "route-lan-key").value = "spare";
  submit(page.window, sel(page, "route-lan-key").closest("form"));
  await settle();
  const dialog = await answerPin(page);
  assert.ok(dialog.hasAttribute("open"));
  assert.equal(dialog.querySelector("#dialog-error").textContent, "No such key.");
  assert.equal(dialog.querySelector("#dialog-pin").value, "");
});

test("route form: an unchanged edit says so and sends nothing", async () => {
  const page = await formsPage([ROUTE]);
  submit(page.window, sel(page, "route-lan-model").closest("form"));
  await settle();
  assert.equal(page.document.querySelector("dialog"), null);
  assert.equal(page.calls.filter((c) => c.method === "POST").length, 0);
});

test("route form: the poll does not redraw while a field has the focus", async () => {
  const page = await formsPage([{ ...ROUTE, running: true }]);
  const input = sel(page, "route-new-name");
  input.focus();
  input.value = "typing";
  await page.run("refreshRoutes()");
  assert.equal(sel(page, "route-new-name").value, "typing");
});
