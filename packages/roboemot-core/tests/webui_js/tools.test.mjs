// The Tools tab (SPEC 11, 11A, 12): the timers and reminders from GET /api/tools/state with their
// cancel behind the PIN, the missed items, what plays, one card per tool with its switch (the
// admin opt-ins confirmed through the batch preview) and its settings saved through the batch
// save, the place search as a POST and the station editor. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const SKILLS = ["play", "go_to_sleep", "timer_set", "timer_cancel", "reminder_set", "reminder_cancel", "schedule_list", "weather", "radio_play", "radio_stop", "radio_volume"];

const STATE = {
  poll_ms: 60000,
  tools: [
    { name: "sleep", enabled: true, available: true },
    { name: "scheduler", enabled: true, available: true },
    { name: "weather", enabled: false, available: false },
    { name: "radio", enabled: true, available: true },
  ],
  items: [
    { id: "t3", kind: "timer", text: "pasta", due_local: "2026-11-17T10:10:00", repeat: "once", days: [] },
    { id: "r4", kind: "reminder", text: "call grandma", due_local: "2026-11-18T08:30:00", repeat: "daily", days: [] },
  ],
  missed: [{ id: "t1", kind: "timer", text: "", due_local: "2026-11-16T09:00:00", repeat: "once", days: [] }],
  playing: { name: "Jazz", volume_db: -12 },
};

const row = (path, kind, value, extra = {}) => ({
  path, kind, value, default: value, writable: true, admin_only: false, opt_in: false, private: false, advanced: false,
  secret: false, source: "default", description: path + " text", nullable: false, ...extra,
});

const FIELDS = [
  row("conversation.llm_skills", "string_list", SKILLS),
  row("circadian.location", "number_dict", "***", { private: true, admin_only: true, nullable: true }),
  row("tools.scheduler.max_items", "int", 20, { advanced: true, ge: 1, le: 200 }),
  row("tools.weather.enabled", "bool", false, { opt_in: true }),
  row("tools.weather.place_name", "text", "***", { private: true, admin_only: true }),
  row("tools.weather.forecast_days", "int", 3, { ge: 1, le: 7 }),
  row("tools.radio.enabled", "bool", true, { opt_in: true }),
  // The catalogue keys "RTL 102.5" by its index (a dot in the name): the segment is "1".
  row("tools.radio.stations", "keyed_list", ["Jazz", "1"], { key: "name", writable: false, removable: ["Jazz", "1"], admin_only: true }),
  row("tools.radio.stations.1.name", "text", "RTL 102.5", { admin_only: true }),
  row("tools.radio.stations.1.url", "text", "https://stream.example.org/rtl.mp3", { admin_only: true }),
  row("tools.radio.stations.Jazz.name", "text", "Jazz", { admin_only: true }),
  row("tools.radio.stations.Jazz.url", "text", "https://stream.example.org/jazz.mp3", { admin_only: true }),
  row("tools.radio.stations.Jazz.genres", "string_list", ["jazz"], { admin_only: true }),
  row("tools.radio.stations.Jazz.language", "text", null, { admin_only: true, nullable: true }),
  row("tools.radio.dance", "bool", true),
  row("tools.media.volume_db", "number", -12, { ge: -40, le: 0, unit: "dB" }),
];

const EN = {
  tab_tools: "Tools", tools_cancel: "Cancel", tools_kind_timer: "Timer", tools_kind_reminder: "Reminder", pin: "PIN",
  tools_repeat_days: "on {days}", tools_weekday_1: "Monday", tools_weekday_3: "Wednesday",
  tools_radio_test_ok: "{name} plays.", tools_cancelled: "Cancelled.", tools_playing_line: "{name} at {volume} dB",
};

async function toolsPage(t, { state = STATE, routes = {} } = {}) {
  let body = state;
  const p = await loadPage({
    "GET /api/session": () => ({ authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 }),
    "GET /api/strings": { language: "en", strings: EN },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/settings": { fields: FIELDS, robot: null, pin_required: true },
    "GET /api/settings/voice": reply(404, { error: "not_found" }),
    "GET /api/tools/state": () => body,
    ...routes,
  });
  t.after(() => p.window.close());
  p.document.querySelector('#tabs button[data-tab="tools"]').click();
  for (let i = 0; i < 4; i++) await settle();
  p.setState = (next) => { body = next; };
  return p;
}

async function submitDialog(p, pin) {
  const dialog = p.document.querySelector("dialog[open]");
  const field = dialog.querySelector("#dialog-pin");
  if (field) field.value = pin;
  dialog.querySelector("form").dispatchEvent(new p.window.Event("submit", { cancelable: true }));
  for (let i = 0; i < 4; i++) await settle();
}

test("the tab sits after Emotions and lists the timers and reminders", async (t) => {
  const p = await toolsPage(t);
  const tabs = [...p.document.querySelectorAll("#tabs button")].map((b) => b.dataset.tab);
  assert.equal(tabs[tabs.indexOf("emotions") + 1], "tools");
  const rows = [...p.document.querySelectorAll("#tools-items li")].map((li) => li.textContent);
  assert.equal(rows.length, 2);
  assert.ok(rows[0].includes("Timer") && rows[0].includes("pasta"), rows[0]);
  assert.ok(rows[1].includes("Reminder") && rows[1].includes("call grandma"), rows[1]);
  assert.equal(p.document.querySelectorAll("#tools-missed li").length, 1);
  assert.ok(p.document.querySelector("#tools-playing").textContent.includes("Jazz"));
});

test("the cancel button asks for the PIN and sends the item id", async (t) => {
  let sent = null;
  const p = await toolsPage(t, { routes: { "POST /api/tools/cancel": (b) => { sent = b; return { ok: true }; } } });
  p.document.querySelector('#tools-items [data-id="r4"]').click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(sent, { id: "r4", pin: "123456" });
  const call = p.calls.find((c) => c.path === "/api/tools/cancel");
  assert.equal(call.method, "POST");
  assert.equal(call.headers["X-CSRF-Token"], "tok");
  assert.equal(p.document.getElementById("notice").textContent, "Cancelled.");
});

test("a poll is marked as activity only after input", async (t) => {
  const p = await toolsPage(t);
  await p.run("toolsRefresh()");
  await settle();
  const polls = p.calls.filter((c) => c.path === "/api/tools/state");
  assert.ok(polls.length >= 2);
  assert.equal(polls[polls.length - 1].headers["X-RoboEmot-Active"], undefined, "no input since the last marked poll");
});

test("without a scheduler the list says so instead of an error", async (t) => {
  const tools = STATE.tools.map((x) => (x.name === "scheduler" ? { ...x, available: false } : x));
  const p = await toolsPage(t, { state: { ...STATE, tools, items: [], missed: [] } });
  const box = p.document.querySelector("#tools-items-card");
  assert.ok(box.textContent.includes("tools_items_unavailable"), box.textContent);
  assert.equal(p.document.querySelector("#tools-items"), null);
});

test("with the scheduler switched off, items that will still fire stay listed with Cancel", async (t) => {
  const tools = STATE.tools.map((x) => (x.name === "scheduler" ? { ...x, enabled: false, available: false } : x));
  const p = await toolsPage(t, { state: { ...STATE, tools } });
  assert.equal(p.document.querySelectorAll("#tools-items li").length, 2);
  assert.ok(p.document.querySelector('#tools-items [data-id="t3"]'));
});

test("turning weather on shows the opt-in, then saves the switch in one batch", async (t) => {
  const sent = [];
  const p = await toolsPage(t, {
    routes: {
      "POST /api/settings/batch/preview": (b) => ({ changes: b.changes.map((c) => ({ path: c.path, old: false, new: c.value, restart: false, opt_in: true })) }),
      "POST /api/settings/batch": (b) => { sent.push(b); return { changed: true, layer: "runtime", results: [] }; },
    },
  });
  const sw = p.document.querySelector("#tools-switch-weather");
  sw.checked = true;
  sw.dispatchEvent(new p.window.Event("change"));
  await settle();
  await submitDialog(p, "123456");
  const confirm = p.document.querySelector("dialog[open]");
  assert.ok(confirm.textContent.includes("tools_opt_in"), confirm.textContent);
  await submitDialog(p, null);
  assert.deepEqual(sent, [{ changes: [{ path: "tools.weather.enabled", value: true }], pin: "123456" }]);
});

test("the scheduler switch writes conversation.llm_skills without its skills", async (t) => {
  let previewed = null;
  const p = await toolsPage(t, {
    routes: { "POST /api/settings/batch/preview": (b) => { previewed = b; return { changes: [] }; } },
  });
  const sw = p.document.querySelector("#tools-switch-scheduler");
  sw.checked = false;
  sw.dispatchEvent(new p.window.Event("change"));
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(previewed.changes, [{ path: "conversation.llm_skills", value: ["play", "go_to_sleep", "weather", "radio_play", "radio_stop", "radio_volume"] }]);
});

test("a card's settings save through the batch with only what changed", async (t) => {
  let sent = null;
  const p = await toolsPage(t, { routes: { "POST /api/settings/batch": (b) => { sent = b; return { changed: true, layer: "runtime", results: [] }; } } });
  const input = p.document.querySelector('[data-tools-card="weather"] input[type=number]');
  input.value = "5";
  p.document.querySelector("#tools-save-weather").click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(sent, { changes: [{ path: "tools.weather.forecast_days", value: 5 }], pin: "123456" });
});

test("the place search posts the query and a result saves the location and its name together", async (t) => {
  let searched = null;
  let saved = null;
  const p = await toolsPage(t, {
    routes: {
      "POST /api/tools/geocode": (b) => { searched = b; return { places: [{ name: "Como", admin: "Lombardia", country: "Italia", lat: 45.8, lon: 9.08 }] }; },
      "POST /api/settings/batch": (b) => { saved = b; return { changed: true, layer: "runtime", results: [] }; },
    },
  });
  p.document.querySelector("#tools-place-query").value = "Como";
  p.document.querySelector("#tools-place-search").click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(searched, { query: "Como", pin: "123456" });
  assert.ok(!p.calls.some((c) => c.path.includes("Como")), "the query never goes in a URL");
  p.document.querySelector("#tools-place-results button").click();
  await settle();
  await submitDialog(p, "654321");
  assert.deepEqual(saved, {
    changes: [
      { path: "circadian.location", value: { lat: 45.8, lon: 9.08 } },
      { path: "tools.weather.place_name", value: "Como, Lombardia, Italia" },
    ],
    pin: "654321",
  });
});

test("the station editor adds a station and test-plays one by name", async (t) => {
  let added = null;
  let tested = null;
  const p = await toolsPage(t, {
    routes: {
      "POST /api/settings/entry": (b) => { added = b; return { changed: true, layer: "runtime" }; },
      "POST /api/tools/radio/test": (b) => { tested = b; return { status: "ok" }; },
    },
  });
  p.document.querySelector("#tools-station-new-name").value = "Rock";
  p.document.querySelector("#tools-station-new-url").value = "https://stream.example.org/rock.mp3";
  p.document.querySelector("#tools-station-new-genres").value = "rock, pop";
  p.document.querySelector("#tools-station-add").click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(added, {
    path: "tools.radio.stations",
    value: { name: "Rock", url: "https://stream.example.org/rock.mp3", genres: ["rock", "pop"], language: null },
    pin: "123456",
  });
  p.document.querySelector('[data-station="Jazz"] [data-action="test"]').click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(tested, { name: "Jazz", pin: "123456" });
  assert.equal(p.document.getElementById("notice").textContent, "Jazz plays.");
});

test("station URLs show only in the editor, never from the state poll", async (t) => {
  const p = await toolsPage(t);
  const editor = p.document.querySelector('[data-station="Jazz"]');
  assert.ok(editor.querySelector('input[value="https://stream.example.org/jazz.mp3"]') || editor.textContent.includes("jazz.mp3"));
  assert.ok(!p.document.querySelector("#tools-live").textContent.includes("example.org"));
});

test("a station keyed by its index shows and test-plays its real name", async (t) => {
  let tested = null;
  let removed = null;
  const p = await toolsPage(t, {
    routes: {
      "POST /api/tools/radio/test": (b) => { tested = b; return { status: "ok" }; },
      "POST /api/settings/entry/remove": (b) => { removed = b; return { changed: true, layer: "runtime" }; },
    },
  });
  const box = p.document.querySelector('[data-station="RTL 102.5"]');
  assert.ok(box, "the editor names the station by its name row");
  assert.equal(box.querySelector("h4").textContent, "RTL 102.5");
  box.querySelector('[data-action="test"]').click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(tested, { name: "RTL 102.5", pin: "123456" });
  box.querySelector('[data-action="remove"]').click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(removed, { path: "tools.radio.stations.1", pin: "123456" }, "paths keep the segment");
});

test("an unavailable tool's status reads as words, with no stray comma", async (t) => {
  const tools = STATE.tools.map((x) => (x.name === "radio" ? { ...x, available: false } : x));
  const p = await toolsPage(t, { state: { ...STATE, tools } });
  const text = p.document.querySelector("#tools-status-radio").textContent;
  assert.equal(text, "tools_status_unavailable tools_unavailable_radio");
});

test("repeating days show as weekday names", async (t) => {
  const items = [{ id: "r9", kind: "reminder", text: "gym", due_local: "2026-11-18T08:30:00", repeat: "days", days: [1, 3] }];
  const p = await toolsPage(t, { state: { ...STATE, items } });
  assert.ok(p.document.querySelector("#tools-items li").textContent.includes("on Monday, Wednesday"));
});

// ---------------------------------------------------------------- the station catalogue (SPEC 11, Radio)

const shipped = (name, url, extra = {}) => ({
  name, url, genres: ["rock"], language: null, operator: "Op " + name, policy_url: "https://op.example.org/terms",
  official: true, bitrate_kbps: 128, added: false, ...extra,
});

const CATALOGUE = {
  groups: [
    { id: "jazz", stations: [shipped("Jazz One", "https://a.example.org/j1")] },
    {
      id: "rock",
      stations: [
        shipped("Rock Here", "https://a.example.org/r0", { added: true }),
        shipped("Rock New", "https://a.example.org/r1"),
        shipped("Rock Taken", "https://a.example.org/r2", { official: false }),
      ],
    },
  ],
};

const CATALOGUE_EN = {
  tools_catalogue_unofficial: "Unofficial stream.", tools_catalogue_added_n: "{n} stations added.",
  tools_catalogue_added_n_one: "{n} station added.", tools_catalogue_skipped_n: "{n} already in the list.",
  tools_catalogue_group_rock: "Rock",
};

async function catalogueOpen(t, routes = {}) {
  const p = await toolsPage(t, { routes: { "GET /api/tools/radio/catalogue": CATALOGUE, ...routes } });
  p.run(`Object.assign(state.strings, ${JSON.stringify(CATALOGUE_EN)})`);
  p.document.querySelector("#tools-catalogue-toggle").click();
  return p;
}

test("the catalogue is a closed section of the radio card, grouped by genre", async (t) => {
  const p = await toolsPage(t, { routes: { "GET /api/tools/radio/catalogue": CATALOGUE } });
  const radio = p.document.querySelector('[data-tools-card="radio"]');
  const toggle = radio.querySelector("#tools-catalogue-toggle");
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
  assert.equal(radio.querySelector("#tools-catalogue").hidden, true);
  toggle.click();
  assert.equal(radio.querySelector("#tools-catalogue").hidden, false);
  assert.deepEqual([...radio.querySelectorAll("[data-catalogue-group]")].map((g) => g.dataset.catalogueGroup), ["jazz", "rock"]);
  const here = radio.querySelector('[data-catalogue-station="Rock Here"]');
  assert.ok(here.querySelector("[data-added]"), "a station already listed shows as added");
  assert.equal(here.querySelector('[data-action="add"]'), null, "and has no second Add");
});

test("Add posts exactly the station's fields behind the PIN and keeps the section open", async (t) => {
  let added = null;
  const p = await catalogueOpen(t, { "POST /api/settings/entry": (b) => { added = b; return { changed: true, layer: "runtime" }; } });
  p.document.querySelector('[data-catalogue-station="Jazz One"] [data-action="add"]').click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(added, {
    path: "tools.radio.stations",
    value: { name: "Jazz One", url: "https://a.example.org/j1", genres: ["rock"], language: null },
    pin: "123456",
  });
  assert.equal(p.document.querySelector("#tools-catalogue").hidden, false, "the rebuild keeps the section open");
  assert.ok(p.calls.filter((c) => c.path === "/api/tools/radio/catalogue").length >= 2, "read again after the add");
});

test("Add all asks for the PIN once, skips the listed station and counts a server duplicate", async (t) => {
  const sent = [];
  const p = await catalogueOpen(t, {
    "POST /api/settings/entry": (b) => {
      sent.push(b);
      return b.value.name === "Rock Taken" ? reply(422, { error: "entry_exists", field: "value" }) : { changed: true, layer: "runtime" };
    },
  });
  p.document.querySelector('[data-catalogue-group="rock"] [data-action="add-all"]').click();
  await settle();
  await submitDialog(p, "123456");
  assert.deepEqual(sent.map((b) => [b.value.name, b.pin]), [["Rock New", "123456"], ["Rock Taken", "123456"]]);
  assert.equal(p.document.querySelector("dialog[open]"), null, "one PIN dialog for the group");
  assert.equal(p.document.getElementById("notice").textContent, "1 station added. 2 already in the list.");
});

test("Add all stops at a wrong PIN without trying the other stations", async (t) => {
  const sent = [];
  const p = await catalogueOpen(t, { "POST /api/settings/entry": (b) => { sent.push(b); return reply(401, { error: "bad_pin" }); } });
  p.document.querySelector('[data-catalogue-group="rock"] [data-action="add-all"]').click();
  await settle();
  await submitDialog(p, "000000");
  assert.equal(sent.length, 1);
});

test("an unofficial stream carries its note and every station links its operator's terms", async (t) => {
  const p = await catalogueOpen(t);
  const notes = [...p.document.querySelectorAll("[data-unofficial]")].map((n) => n.closest("[data-catalogue-station]").dataset.catalogueStation);
  assert.deepEqual(notes, ["Rock Taken"]);
  assert.equal(p.document.querySelector("[data-unofficial]").textContent, "Unofficial stream.");
  const links = [...p.document.querySelectorAll("#tools-catalogue a")];
  assert.equal(links.length, 4);
  for (const a of links) {
    assert.equal(a.getAttribute("href"), "https://op.example.org/terms");
    assert.equal(a.getAttribute("target"), "_blank");
    assert.equal(a.getAttribute("rel"), "noopener noreferrer");
  }
});

test("without the catalogue the station editor still works", async (t) => {
  const p = await toolsPage(t, { routes: { "GET /api/tools/radio/catalogue": reply(503, { error: "tools_unavailable" }) } });
  assert.equal(p.document.querySelector("#tools-catalogue-toggle"), null);
  assert.ok(p.document.querySelector("#tools-station-add"));
});
