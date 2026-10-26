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

const STRINGS = { language: "en", strings: { route_key_message: "Key to {host}" } };
const ROUTE = {
  name: "lan", model: "m", endpoint: "api.example.com", base_url: "https://api.example.com/v1", locality: "auto",
  key: "none", running: false, secret: null, last_check: null, removable: false,
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

test("route form: a new route defaults to cloud, has no minors flag and offers only stored keys", async () => {
  const page = await formsPage([]);
  assert.equal(sel(page, "route-new-locality").value, "cloud");
  assert.equal(sel(page, "route-new-minors"), null);
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
    name: "cloud2", base_url: "https://API.Cloudco.example/v1", model: "big", locality: "cloud", kind: "chat",
    secret: "cloud_key", pin: PIN, ack_endpoint: "https://api.cloudco.example",
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

test("route form: the kind is chosen when a route is added, chat by default", async () => {
  let seen = null;
  const page = await formsPage([], { "POST /api/routes/add": (b) => { seen = b; return { changed: true }; } });
  assert.equal(sel(page, "route-new-kind").value, "chat");
  sel(page, "route-new-name").value = "voice";
  sel(page, "route-new-url").value = "https://rt.example/v1";
  sel(page, "route-new-model").value = "s2s";
  sel(page, "route-new-kind").value = "realtime";
  submit(page.window, sel(page, "route-new-name").closest("form"));
  await settle();
  await answerPin(page);
  assert.equal(seen.kind, "realtime");
});

test("route form: a realtime route shows its kind, cannot be checked and its kind is not editable", async () => {
  const RT = { ...ADDED, name: "voice", kind: "realtime" };
  const page = await formsPage([ROUTE, RT]);
  const card = [...page.document.querySelectorAll("section.sl-card")].find((c) => c.querySelector("h2").textContent === "voice");
  assert.match(card.textContent, /route_kind_realtime/);
  const check = [...card.querySelectorAll("button")].find((b) => b.textContent === "route_check");
  assert.equal(check.disabled, true);
  assert.equal(sel(page, "route-voice-kind"), null);
  const lan = [...page.document.querySelectorAll("section.sl-card")].find((c) => c.querySelector("h2").textContent === "lan");
  assert.equal([...lan.querySelectorAll("button")].find((b) => b.textContent === "route_check").disabled, false);
});

test("route form: only a route added here can be removed", async () => {
  const page = await formsPage([ROUTE, ADDED]);
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
  submit(page.window, sel(page, "route-lan-model").closest("form"));
  await settle();
  await answerPin(page);
  assert.deepEqual(order, ["key", "llm_routes.lan.model"]);
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

test("route form: a change of the path only, on the same host, is still sent", async () => {
  const page = await formsPage([ROUTE], { "POST /api/settings": { changed: true } });
  sel(page, "route-lan-url").value = "https://api.example.com/v2";
  submit(page.window, sel(page, "route-lan-url").closest("form"));
  await settle();
  await answerPin(page);
  const call = page.calls.find((c) => c.path === "/api/settings");
  assert.deepEqual(call.body, { path: "llm_routes.lan.base_url", value: "https://api.example.com/v2", pin: PIN });
});

test("route form: a path-only change of a keyed route carries the origin as the acknowledgement", async () => {
  const page = await formsPage([ADDED], { "POST /api/settings": { changed: true } });
  sel(page, "route-cloud-url").value = "https://api.example.com/v2";
  submit(page.window, sel(page, "route-cloud-url").closest("form"));
  await settle();
  await answerPin(page);
  const call = page.calls.find((c) => c.path === "/api/settings");
  assert.equal(call.body.ack_endpoint, "https://api.example.com");
});

test("route form: an http host that is not this machine adds the unencrypted warning", async () => {
  const page = await formsPage([], { "GET /api/strings": { language: "en", strings: { route_key_message: "Key to {host}", route_key_insecure: "CLEAR" } } });
  const fill = (url) => {
    sel(page, "route-new-name").value = "n";
    sel(page, "route-new-url").value = url;
    sel(page, "route-new-model").value = "m";
    sel(page, "route-new-key").value = "cloud_key";
    submit(page.window, sel(page, "route-new-name").closest("form"));
  };
  fill("http://192.168.1.9/v1");
  await settle();
  assert.match(page.document.querySelector("dialog").textContent, /CLEAR/);
  page.document.querySelector("dialog").close();
  fill("http://localhost:8080/v1");
  await settle();
  assert.doesNotMatch(page.document.querySelector("dialog").textContent, /CLEAR/);
});

test("route form: the host is acknowledged in punycode", async () => {
  let seen = null;
  const page = await formsPage([], { "POST /api/routes/add": (b) => { seen = b; return { changed: true }; } });
  sel(page, "route-new-name").value = "n";
  sel(page, "route-new-url").value = "https://аpple.example/v1";
  sel(page, "route-new-model").value = "m";
  sel(page, "route-new-key").value = "cloud_key";
  submit(page.window, sel(page, "route-new-name").closest("form"));
  await settle();
  assert.match(page.document.querySelector("dialog").textContent, /xn--/);
  await answerPin(page);
  assert.match(seen.ack_endpoint, /^https:\/\/xn--[a-z0-9-]+\.example$/);
});

test("route form: when the server names another endpoint the dialog text is rebuilt from it", async () => {
  const page = await formsPage([], {
    "GET /api/strings": { language: "en", strings: { route_key_message: "Key to {host}" } },
    "POST /api/routes/add": reply(422, { error: "endpoint_ack_required", endpoint: "https://server.example" }),
  });
  sel(page, "route-new-name").value = "n";
  sel(page, "route-new-url").value = "https://client.example/v1";
  sel(page, "route-new-model").value = "m";
  sel(page, "route-new-key").value = "cloud_key";
  submit(page.window, sel(page, "route-new-name").closest("form"));
  await settle();
  await answerPin(page);
  assert.match(page.document.querySelector("dialog").textContent, /Key to https:\/\/server\.example/);
});

// ---------------------------------------------------------------- R3 SF4: the order of an edit

function editRecorder(fail = {}) {
  const order = [];
  const handler = (label) => (b) => {
    order.push({ label, body: b });
    if (fail[label] && fail[label]-- > 0) return reply(429, { error: "busy" });
    return { changed: true };
  };
  return {
    order,
    routes: {
      "GET /api/strings": { language: "en", strings: { ...STRINGS.strings, route_partial: "Saved already: {items}.", route_key: "Key", route_url: "URL", err_busy: "Busy." } },
      "POST /api/settings": (b) => handler(b.path.split(".").pop())(b),
      "POST /api/routes/key": (b) => handler(b.secret === null ? "unbind" : "bind")(b),
    },
  };
}

test("route edit: moving a keyed route and dropping its key unbinds the key first, with no acknowledgement", async () => {
  const rec = editRecorder();
  const page = await formsPage([ADDED], rec.routes);
  sel(page, "route-cloud-url").value = "https://new.example/v1";
  sel(page, "route-cloud-key").value = "";
  submit(page.window, sel(page, "route-cloud-url").closest("form"));
  await settle();
  assert.doesNotMatch(page.document.querySelector("dialog").textContent, /Key to/, "no key goes anywhere");
  await answerPin(page);
  assert.deepEqual(rec.order.map((s) => s.label), ["unbind", "base_url"]);
  assert.deepEqual(rec.order[0].body, { name: "cloud", secret: null, pin: PIN });
  assert.ok(!("ack_endpoint" in rec.order[1].body), "the route has no key when it moves");
});

test("route edit: moving a keyed route to another key never sends the old key to the new host", async () => {
  const rec = editRecorder();
  const page = await formsPage([ADDED], rec.routes);
  sel(page, "route-cloud-url").value = "https://new.example/v1";
  sel(page, "route-cloud-key").value = "spare";
  submit(page.window, sel(page, "route-cloud-url").closest("form"));
  await settle();
  assert.match(page.document.querySelector("dialog").textContent, /Key to https:\/\/new\.example/);
  await answerPin(page);
  assert.deepEqual(rec.order.map((s) => s.label), ["unbind", "base_url", "bind"]);
  assert.ok(!("ack_endpoint" in rec.order[1].body));
  assert.deepEqual(rec.order[2].body, { name: "cloud", secret: "spare", pin: PIN, ack_endpoint: "https://new.example" });
});

test("route edit: a failing step stops the rest, names what was saved, and a retry resumes there", async () => {
  const rec = editRecorder({ base_url: 1 });
  const page = await formsPage([ADDED], rec.routes);
  sel(page, "route-cloud-url").value = "https://new.example/v1";
  sel(page, "route-cloud-key").value = "";
  sel(page, "route-cloud-model").value = "bigger";
  submit(page.window, sel(page, "route-cloud-url").closest("form"));
  await settle();
  const dialog = await answerPin(page);
  assert.deepEqual(rec.order.map((s) => s.label), ["unbind", "base_url"], "the model was not sent");
  assert.ok(dialog.hasAttribute("open"));
  assert.equal(dialog.querySelector("#dialog-error").textContent, "Busy. Saved already: Key.");
  await answerPin(page);
  assert.deepEqual(rec.order.map((s) => s.label), ["unbind", "base_url", "base_url", "model"], "no step is sent twice once done");
});

test("route edit: moving a keyed LAN route to the cloud sets it to cloud before anything else (review ZG)", async () => {
  // A local route refuses a new base_url while it is still local: the locality must go first, and a
  // refusal there must leave the key bound (nothing unbound yet).
  const rec = editRecorder();
  const LOCAL = { ...ADDED, name: "box", base_url: "http://192.168.1.20:1234/v1", locality: "local" };
  const page = await formsPage([LOCAL], rec.routes);
  sel(page, "route-box-url").value = "https://api.cloud.example/v1";
  sel(page, "route-box-key").value = "spare";
  sel(page, "route-box-locality").value = "cloud";
  submit(page.window, sel(page, "route-box-url").closest("form"));
  await settle();
  await answerPin(page);
  assert.deepEqual(rec.order.map((s) => s.label), ["locality", "unbind", "base_url", "bind"]);
  assert.equal(rec.order[0].body.value, "cloud");
});

test("route edit: a refused locality step unbinds nothing", async () => {
  const rec = editRecorder({ locality: 1 });
  const LOCAL = { ...ADDED, name: "box", base_url: "http://192.168.1.20:1234/v1", locality: "local" };
  const page = await formsPage([LOCAL], rec.routes);
  sel(page, "route-box-url").value = "https://api.cloud.example/v1";
  sel(page, "route-box-key").value = "";
  sel(page, "route-box-locality").value = "cloud";
  submit(page.window, sel(page, "route-box-url").closest("form"));
  await settle();
  await answerPin(page);
  assert.deepEqual(rec.order.map((s) => s.label), ["locality"]);
});
