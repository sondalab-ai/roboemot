"use strict";
// Tools tab (SPEC 11, 11A, 12): the voice tools. The timers and reminders, the missed items and
// what the radio plays come from GET /api/tools/state (a passive poll, at `webui.tools.poll_ms`);
// one card per tool shows its switch, a short status and its settings, read from GET /api/settings.
// Every write re-enters the PIN: cancel, clear the missed items, stop and test-play the radio and
// the place search are POSTs of their own (the query travels in the body, never in a URL). The
// switches and the settings are saved through the batch settings save, an admin opt-in confirmed
// through the batch preview first (settingsPreviewedSave). The station editor adds and removes an
// entry through the settings entry routes and saves its fields through the batch; the station
// catalogue (GET /api/tools/radio/catalogue, read with the settings, not polled) adds through the
// same entry route. A station URL shows only in the editor and the catalogue, never in the polled
// state. Shares the helpers of app.js, settings.js
// and live.js, so it loads before app.js and calls them at run time.
//
// Polling: the timer is `state.timer` (stopPolling clears it), only while the tab is open and the
// document visible; `toolsState.serial` retires a request in flight when the tab is left or hidden.
// The tool cards are built once per settings read and keep their edits across polls.
//
// Skills (SPEC 11A, 12; design 2026-11-24-skill-packs 5.6): the skill packs from
// GET /api/tools/skill_packs, read when the tab opens and after each write (not polled). Turning
// one on first shows its text (GET /api/tools/skill_packs/<name>); every write re-enters the PIN
// and sends `confirm: true` once the dialog is confirmed. The upload reads the file as base64.

const TOOLS_RETRY_MS = 5000;
const TOOLS_LLM_SKILLS = "conversation.llm_skills";
const TOOLS_LOCATION = "circadian.location";
const TOOLS_PLACE_NAME = "tools.weather.place_name";
const TOOLS_STATIONS = "tools.radio.stations";
// Each tool's card: the switch (a setting, or its skills in `conversation.llm_skills`, SPEC 11) and
// the settings sections it shows.
const TOOLS_CARDS = {
  sleep: { skills: ["go_to_sleep"], sections: [] },
  scheduler: { skills: ["timer_set", "timer_cancel", "reminder_set", "reminder_cancel", "schedule_list"], sections: ["tools.scheduler."] },
  weather: { path: "tools.weather.enabled", sections: ["tools.weather."] },
  radio: { path: "tools.radio.enabled", sections: ["tools.radio.", "tools.media."] },
};
// Rows a card shows in a control of its own, not among its settings.
const TOOLS_OWN_ROWS = new Set(["tools.weather.enabled", "tools.radio.enabled", TOOLS_PLACE_NAME, TOOLS_STATIONS]);
const toolsState = { serial: 0, rows: null, cards: null, live: null, tools: [], catalogue: null, catalogueOpen: false };

const toolsId = (path) => "tools-" + path.replace(/[^A-Za-z0-9]+/g, "-");
const toolsName = (name) => t("tools_name_" + name);

// ---------------------------------------------------------------- the polled part

function toolsItemRow(item, missed) {
  const what = item.text || t("tools_no_label");
  const days = item.days.map((d) => t("tools_weekday_" + d)).join(", ");
  const repeat = item.repeat === "days" ? t("tools_repeat_days", { days }) : t("tools_repeat_" + item.repeat);
  const parts = [badge(t("tools_kind_" + item.kind), item.kind === "timer" ? "info" : "success"), " ",
    el("span", {}, what), " ", el("span", { class: "sl-eyebrow" }, when(item.due_local) + " · " + repeat)];
  if (!missed) {
    parts.push(" ", button(t("tools_cancel"), "danger", () => guarded(() => toolsCancel(item)),
      { small: true, attrs: { "data-id": item.id, "aria-label": rowName(t("tools_cancel"), what) } }));
  }
  return el("li", {}, parts);
}

function toolsItemsCard(d) {
  const tool = d.tools.find((x) => x.name === "scheduler");
  // Items set before the switch went off still fire, so they always show with their Cancel.
  let body;
  if (d.items.length) body = el("ul", { class: "sl-list", id: "tools-items" }, d.items.map((i) => toolsItemRow(i, false)));
  else if (!tool || !tool.available) body = el("p", { class: "hint" }, t(tool && !tool.enabled ? "tools_items_off" : "tools_items_unavailable"));
  else body = el("p", { class: "hint" }, t("tools_items_none"));
  const node = card(t("tab_tools"), t("tools_items"), body);
  node.id = "tools-items-card";
  return node;
}

function toolsMissedCard(d) {
  const node = d.missed.length
    ? card(t("tab_tools"), t("tools_missed"), el("p", { class: "hint" }, t("tools_missed_intro")),
      el("ul", { class: "sl-list", id: "tools-missed" }, d.missed.map((i) => toolsItemRow(i, true))),
      el("div", { class: "row" }, button(t("tools_missed_clear"), "plain", () => guarded(toolsMissedClear), { small: true })))
    : card(t("tab_tools"), t("tools_missed"), el("p", { class: "hint" }, t("tools_missed_none")));
  node.id = "tools-missed-card";
  return node;
}

function toolsPlayingCard(d) {
  const p = d.playing;
  const node = p === null
    ? card(t("tab_tools"), t("tools_playing"), el("p", { class: "hint", id: "tools-playing" }, t("tools_playing_none")))
    : card(t("tab_tools"), t("tools_playing"),
      el("p", { id: "tools-playing" }, t("tools_playing_line", { name: p.name, volume: num(p.volume_db, 1) })),
      el("div", { class: "row" }, button(t("tools_radio_stop"), "danger", () => guarded(toolsRadioStop), { small: true })));
  node.id = "tools-playing-card";
  return node;
}

// The status of each tool card, updated in place (the cards keep their edits).
function toolsStatus(d) {
  toolsState.tools = d.tools;
  for (const tool of d.tools) {
    const box = document.getElementById("tools-status-" + tool.name);
    if (!box) continue;
    const [key, tone] = !tool.enabled ? ["tools_status_off", "info"] : tool.available ? ["tools_status_ready", "success"] : ["tools_status_unavailable", "warning"];
    const text = t(key) + (tool.enabled && !tool.available ? " " + t("tools_unavailable_" + tool.name) : "");
    if (box.dataset.shown === text) continue;
    box.dataset.shown = text;
    box.replaceChildren(badge(t(key), tone), ...(tool.enabled && !tool.available ? [" ", t("tools_unavailable_" + tool.name)] : []));
  }
}

function toolsRender(d) {
  liveKeep(toolsState.live, [toolsItemsCard(d), toolsMissedCard(d), toolsPlayingCard(d)]);
  toolsStatus(d);
}

function toolsActive(serial) {
  return serial === toolsState.serial && state.tab === "tools" && state.csrf !== null && document.visibilityState === "visible";
}

async function toolsPoll(serial) {
  clearTimeout(state.timer);
  state.timer = null;
  if (!toolsActive(serial)) return;
  let delay = TOOLS_RETRY_MS;
  try {
    const data = await api("GET", "/api/tools/state", undefined, liveActivity());
    if (!toolsActive(serial)) return;
    toolsRender(data);
    pollGood(toolsState.live);
    delay = data.poll_ms || TOOLS_RETRY_MS;
  } catch (err) {
    if (!toolsActive(serial)) return;
    if (err.status === 401 && err.message === "unauthorized") { await boot(); return; }
    if (err.message === "tools_unavailable") {
      pollReset(toolsState.live);
      toolsState.live.replaceChildren(card(t("tab_tools"), t("tab_tools"), el("p", {}, explain(err))));
      return;
    }
    pollFailed(toolsState.live, explain(err));
  }
  state.timer = setTimeout(() => toolsPoll(serial), delay);
}

// Polls again now (after a write), on a new serial so a poll in flight is retired.
function toolsRefresh() {
  return toolsPoll(++toolsState.serial);
}

// ---------------------------------------------------------------- the polled writes

async function toolsPinned(title, message, confirm, call, danger) {
  let out = null;
  const done = await ask({ title, message, pin: true, confirm, danger, run: async ({ pin }) => { out = await call(pin); } });
  return done ? out : null;
}

async function toolsCancel(item) {
  const what = item.text || t("tools_kind_" + item.kind);
  const out = await toolsPinned(t("tools_cancel"), t("tools_cancel_message", { what }), t("tools_cancel"),
    (pin) => api("POST", "/api/tools/cancel", { id: item.id, pin }), true);
  if (!out) return;
  notice(t("tools_cancelled"));
  await toolsRefresh();
}

async function toolsMissedClear() {
  const out = await toolsPinned(t("tools_missed_clear"), t("tools_missed_clear_message"), t("tools_missed_clear"),
    (pin) => api("POST", "/api/tools/missed/clear", { pin }));
  if (!out) return;
  notice(t("tools_missed_cleared"));
  await toolsRefresh();
}

async function toolsRadioStop() {
  const out = await toolsPinned(t("tools_radio_stop"), t("tools_radio_stop_message"), t("tools_radio_stop"),
    (pin) => api("POST", "/api/tools/radio/stop", { pin }), true);
  if (!out) return;
  notice(t("tools_radio_stopped"));
  await toolsRefresh();
}

async function toolsRadioTest(name) {
  const out = await toolsPinned(t("tools_station_test"), t("tools_station_test_message", { name }), t("tools_station_test"),
    (pin) => api("POST", "/api/tools/radio/test", { name, pin }));
  if (!out) return;
  notice(t("tools_radio_test_" + out.status, { name }), out.status !== "ok");
  await toolsRefresh();
}

// ---------------------------------------------------------------- tool cards

const toolsRow = (path) => (toolsState.rows ? toolsState.rows.get(path) : undefined);

// Whether a tool's switch is on: its setting, or all of its skills in `conversation.llm_skills`.
function toolsSwitchOn(name) {
  const spec = TOOLS_CARDS[name];
  if (spec.path) return toolsRow(spec.path) ? toolsRow(spec.path).value === true : null;
  const skills = toolsRow(TOOLS_LLM_SKILLS);
  return skills && Array.isArray(skills.value) ? spec.skills.every((s) => skills.value.includes(s)) : null;
}

// The batch that turns a tool on or off.
function toolsSwitchChanges(name, on) {
  const spec = TOOLS_CARDS[name];
  if (spec.path) return [{ path: spec.path, value: on }];
  const listed = toolsRow(TOOLS_LLM_SKILLS).value;
  const value = on ? [...listed, ...spec.skills.filter((s) => !listed.includes(s))] : listed.filter((s) => !spec.skills.includes(s));
  return [{ path: TOOLS_LLM_SKILLS, value }];
}

// A switch saves through the batch preview: the PIN dialog shows what changes and names an admin
// opt-in it turns on or off (SPEC 12) before the save, like the voice switch.
async function toolsSwitch(name, on) {
  await settingsPreviewedSave({
    title: t("tools_switch_title", { tool: toolsName(name) }),
    changes: toolsSwitchChanges(name, on),
    confirm: t(on ? "tools_switch_on" : "tools_switch_off"),
    same: t("settings_unchanged"),
    message: (diff, wanted) => [t("voice_switch_diff"), ...optInLines(diff, wanted, "tools_opt_in", "tools_opt_out")].join(" "),
    reload: toolsReload,
  });
}

function toolsSwitchControl(name) {
  const on = toolsSwitchOn(name);
  if (on === null) return [];
  const input = el("input", { type: "checkbox", id: "tools-switch-" + name, role: "switch", class: "sl-switch__input", checked: on });
  input.addEventListener("change", () => {
    const wanted = input.checked;
    input.checked = on; // the saved value shows until the save went through (toolsReload)
    guarded(() => toolsSwitch(name, wanted));
  });
  return el("label", { class: "sl-switch" }, input, el("span", { class: "sl-switch__track" }), el("span", { class: "sl-switch__label" }, t("tools_switch_label")));
}

// The editable leaf rows of a card: under its sections, not its own controls, not a station field.
function toolsCardRows(name) {
  if (!toolsState.rows) return [];
  const prefixes = TOOLS_CARDS[name].sections;
  return [...toolsState.rows.values()].filter((r) => prefixes.some((p) => r.path.startsWith(p)) && !TOOLS_OWN_ROWS.has(r.path) &&
    !r.path.startsWith(TOOLS_STATIONS + ".") && r.writable && !r.secret && !r.private && !SETTING_CONTAINERS.includes(r.kind) && r.kind !== "json");
}

// One settings row of a card: the control the settings view builds for it, with its name and text.
function toolsSetting(row, controls) {
  const id = toolsId(row.path);
  const ctl = buildControl(row, id, { label: row.path });
  controls.push({ row, ctl });
  const name = el("code", { class: "sl-code" }, row.path);
  const head = ctl.labelFor ? el("label", { class: "sl-field__label", for: ctl.labelFor }, name) : el("span", { class: "sl-field__label" }, ctl.ownsLabel ? [] : name);
  return el("div", { class: "setting stack", "data-path": row.path }, head, row.description ? el("p", { class: "hint" }, row.description) : [], ctl.node);
}

// The closed-by-default fold of a card's advanced rows.
function toolsFold(id, rows) {
  const body = el("div", { class: "stack", id, hidden: true }, rows);
  const toggle = el("button", { type: "button", class: "sl-disclosure", "aria-expanded": "false", "aria-controls": id }, t("settings_advanced"));
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    body.hidden = !open;
  });
  return el("div", { class: "stack" }, el("h3", { class: "setting-entry" }, toggle), body);
}

// The settings of a card with one Save: only the rows whose value changed go in the batch.
function toolsSettings(name) {
  const rows = toolsCardRows(name);
  if (!rows.length) return [];
  const controls = [];
  const basic = rows.filter((r) => !r.advanced).map((r) => toolsSetting(r, controls));
  const advanced = rows.filter((r) => r.advanced).map((r) => toolsSetting(r, controls));
  const save = async () => {
    const changes = [];
    for (const { row, ctl } of controls) {
      const out = ctl.read();
      if (out.code) { notice(row.path + ": " + (out.code === "value_range" ? rangeError(row) : t("err_" + out.code)), true); return; }
      if (JSON.stringify(out.value) !== JSON.stringify(row.value)) changes.push({ path: row.path, value: out.value });
    }
    if (!changes.length) { notice(t("settings_unchanged")); return; }
    await settingsBatchSave(changes.map((c) => c.path).join(", "), changes, "saved", toolsReload);
  };
  return [markedEyebrow("h3", t("tools_settings")), basic, advanced.length ? toolsFold("tools-advanced-" + name, advanced) : [],
    el("div", { class: "row" }, button(t("settings_save"), "primary", () => guarded(save), { small: true, attrs: { id: "tools-save-" + name } }))];
}

// ---------------------------------------------------------------- weather: the place

function toolsPlaceLine() {
  const location = toolsRow(TOOLS_LOCATION);
  if (!location || location.value === null) return el("p", { class: "hint", id: "tools-place" }, t("tools_place_none"));
  const shown = revealedSettings.get(TOOLS_PLACE_NAME);
  if (shown) return el("p", { id: "tools-place" }, t("tools_place", { place: shown.value || t("tools_place_unnamed") }));
  return el("div", { class: "row", id: "tools-place" }, el("span", { class: "hint" }, t("tools_place_hidden")),
    button(t("settings_reveal"), "plain", () => guarded(toolsReveal), { small: true }));
}

// The private place name, read behind the PIN into the page's revealed values (dropped at logout).
async function toolsReveal() {
  const out = await pinnedCall(t("settings_reveal_message"), t("settings_reveal"), (pin) => api("POST", "/api/settings/reveal", { pin }));
  if (!out) return;
  for (const f of out.fields) revealedSettings.set(f.path, f);
  toolsBuild();
}

const toolsPlaceText = (p) => [p.name, p.admin, p.country].filter(Boolean).join(", ");

async function toolsPlaceSearch(query, results) {
  if (!query.trim()) { notice(t("err_query_invalid"), true); return; }
  const out = await pinnedCall(t("tools_place_search_message"), t("tools_place_search"),
    (pin) => api("POST", "/api/tools/geocode", { query, pin }));
  if (!out) return;
  results.replaceChildren(out.places.length
    ? el("ul", { class: "sl-list" }, out.places.map((p) => el("li", {}, toolsPlaceText(p), " ",
      button(t("tools_place_use"), "plain", () => guarded(() => toolsPlaceUse(p)), { small: true, attrs: { "aria-label": rowName(t("tools_place_use"), toolsPlaceText(p)) } }))))
    : el("p", { class: "hint" }, t("tools_place_results_none")));
}

// A chosen place writes the location and its name in one batch (SPEC 11A).
async function toolsPlaceUse(place) {
  const name = toolsPlaceText(place);
  await settingsBatchSave(t("tools_place_use_message", { place: name }), [
    { path: TOOLS_LOCATION, value: { lat: place.lat, lon: place.lon } },
    { path: TOOLS_PLACE_NAME, value: name },
  ], "saved", toolsReload);
}

function toolsWeatherExtra() {
  const query = textInput("tools-place-query", { autocomplete: "off" }); // the server owns the length limit
  const results = el("div", { class: "stack", id: "tools-place-results" });
  return [markedEyebrow("h3", t("tools_place_title")), toolsPlaceLine(),
    field("tools-place-query", t("tools_place_search_label"), query, t("tools_place_search_hint")),
    el("div", { class: "row" }, button(t("tools_place_search"), "plain", () => guarded(() => toolsPlaceSearch(query.value, results)), { small: true, attrs: { id: "tools-place-search" } })),
    results];
}

// ---------------------------------------------------------------- radio: the station editor

// One station of the editor. `key` is its catalogue path segment, which is its index when the name
// has a dot, is all digits or is a duplicate; the name shown and sent is the entry's `name` row.
function toolsStation(key, removable) {
  const base = TOOLS_STATIONS + "." + key + ".";
  const own = toolsRow(base + "name");
  const name = own && typeof own.value === "string" ? own.value : key;
  const controls = [];
  const fields = ["url", "genres", "language"].map((k) => toolsRow(base + k)).filter(Boolean).map((r) => toolsSetting(r, controls));
  const save = async () => {
    const changes = [];
    for (const { row, ctl } of controls) {
      const out = ctl.read();
      if (out.code) { notice(row.path + ": " + t("err_" + out.code), true); return; }
      if (JSON.stringify(out.value) !== JSON.stringify(row.value)) changes.push({ path: row.path, value: out.value });
    }
    if (!changes.length) { notice(t("settings_unchanged")); return; }
    await settingsBatchSave(changes.map((c) => c.path).join(", "), changes, "saved", toolsReload);
  };
  const remove = async () => {
    const out = await pinnedCall(t("tools_station_remove_message", { name }), t("tools_station_remove"),
      (pin) => api("POST", "/api/settings/entry/remove", { path: TOOLS_STATIONS + "." + key, pin }));
    if (!out) return;
    notice(t("tools_station_removed", { name }));
    await toolsReload();
  };
  const named = (label, action) => ({ small: true, attrs: { "data-action": action, "aria-label": rowName(label, name) } });
  return el("div", { class: "stack", "data-station": name }, el("h4", { class: "setting-entry" }, name), fields,
    el("div", { class: "row" },
      button(t("settings_save"), "primary", () => guarded(save), named(t("settings_save"), "save")),
      button(t("tools_station_test"), "plain", () => guarded(() => toolsRadioTest(name)), named(t("tools_station_test"), "test")),
      removable ? button(t("tools_station_remove"), "danger", () => guarded(remove), named(t("tools_station_remove"), "remove")) : []),
    removable ? [] : el("p", { class: "hint" }, t("tools_station_lower")));
}

function toolsStationAdd() {
  const input = (key, props) => textInput("tools-station-new-" + key, Object.assign({ autocomplete: "off" }, props));
  const name = input("name");
  const url = input("url", { type: "url" });
  const genres = input("genres");
  const language = input("language");
  const add = async () => {
    const value = {
      name: name.value.trim(), url: url.value.trim(),
      genres: genres.value.split(",").map((g) => g.trim()).filter(Boolean),
      language: language.value.trim() || null,
    };
    if (!value.name || !value.url) { notice(t("tools_station_needs"), true); return; }
    const out = await pinnedCall(t("tools_station_add_message", { name: value.name }), t("tools_station_add"),
      (pin) => api("POST", "/api/settings/entry", { path: TOOLS_STATIONS, value, pin }));
    if (!out) return;
    notice(t("tools_station_added", { name: value.name }));
    await toolsReload();
  };
  return el("div", { class: "stack" }, el("h4", { class: "setting-entry" }, t("tools_station_new")),
    field("tools-station-new-name", t("tools_station_name"), name),
    field("tools-station-new-url", t("tools_station_url"), url, t("tools_station_url_hint")),
    field("tools-station-new-genres", t("tools_station_genres"), genres, t("tools_station_genres_hint")),
    field("tools-station-new-language", t("tools_station_language"), language),
    el("div", { class: "row" }, button(t("tools_station_add"), "plain", () => guarded(add), { small: true, attrs: { id: "tools-station-add" } })));
}

function toolsRadioExtra() {
  const list = toolsRow(TOOLS_STATIONS);
  if (!list) return [];
  const keys = Array.isArray(list.value) ? list.value : [];
  const removable = new Set(list.removable || []);
  return [markedEyebrow("h3", t("tools_stations")), el("p", { class: "hint" }, t("tools_stations_intro")),
    keys.length ? keys.map((k) => toolsStation(k, removable.has(k))) : el("p", { class: "hint" }, t("tools_stations_none")),
    toolsStationAdd(), toolsCatalogue()];
}

// ---------------------------------------------------------------- radio: the station catalogue

// The shipped catalogue (GET /api/tools/radio/catalogue, SPEC 11, Radio), grouped by genre. Add
// posts one station through the settings entry route like the add form; it becomes a normal
// runtime station (editable, removable). A station already listed (by name or URL) shows as added.
// null when the catalogue cannot be read: the section is then left out.
async function toolsCatalogueFetch() {
  try {
    return await api("GET", "/api/tools/radio/catalogue");
  } catch (err) {
    if (err.status === 401) throw err;
    return null;
  }
}

// What the add route takes: exactly a station's fields, never the catalogue-only ones.
const toolsCatalogueValue = (s) => ({ name: s.name, url: s.url, genres: s.genres, language: s.language });
const toolsCatalogueAdd = (s, pin) => api("POST", "/api/settings/entry", { path: TOOLS_STATIONS, value: toolsCatalogueValue(s), pin });
// A refusal about the PIN or the session, not the station: Add all stops on it at any station. At the
// first one the dialog shows it (nothing went in); later, the notice reports what went in, the cause
// and how many were not tried.
const TOOLS_PIN_ERRORS = new Set(["bad_pin", "locked", "busy", "unauthorized", "csrf", "pin_not_set"]);

async function toolsCatalogueAddOne(s) {
  const out = await pinnedCall(t("tools_station_add_message", { name: s.name }), t("tools_station_add"), (pin) => toolsCatalogueAdd(s, pin));
  if (!out) return;
  notice(t("tools_station_added", { name: s.name }));
  await toolsReload();
}

// One PIN for the group, then one add per station not yet listed. A station the server already
// has (entry_exists) counts as skipped; another station refusal counts as failed and the rest go
// on. A PIN or session refusal stops the run (TOOLS_PIN_ERRORS). An expired session after some
// adds shows the counts first, then the login (boot), so the partial result is not lost silently.
async function toolsCatalogueAddAll(group) {
  const label = t("tools_catalogue_group_" + group.id);
  const pending = group.stations.filter((s) => !s.added);
  const counts = { added: 0, skipped: group.stations.length - pending.length, failed: 0, stopped: null, untried: 0 };
  if (!pending.length) { notice(t("tools_catalogue_all_added", { group: label })); return; }
  const out = await pinnedCall(tn("tools_catalogue_add_all_message", pending.length, { group: label }), t("tools_catalogue_add_all"), async (pin) => {
    for (const [i, s] of pending.entries()) {
      try {
        await toolsCatalogueAdd(s, pin);
        counts.added += 1;
      } catch (err) {
        if (TOOLS_PIN_ERRORS.has(err.message)) {
          if (i === 0) throw err;
          counts.stopped = err;
          counts.untried = pending.length - i;
          break;
        }
        if (err.message === "entry_exists") counts.skipped += 1;
        else counts.failed += 1;
      }
    }
    return counts;
  });
  if (!out) return;
  const parts = [tn("tools_catalogue_added_n", out.added)];
  if (out.skipped) parts.push(tn("tools_catalogue_skipped_n", out.skipped));
  if (out.failed) parts.push(tn("tools_catalogue_failed_n", out.failed));
  if (out.stopped) parts.push(tn("tools_catalogue_stopped", out.untried, { reason: explain(out.stopped) }));
  notice(parts.join(" "), out.failed > 0 || out.stopped !== null);
  if (out.stopped && out.stopped.status === 401 && out.stopped.message === "unauthorized") { await boot(); return; }
  await toolsReload();
}

// Only an https policy page becomes a link (the server checks it at load too).
function toolsHttps(url) {
  try {
    return new URL(url).protocol === "https:";
  } catch (e) {
    return false;
  }
}

function toolsCatalogueStation(s) {
  const facts = [s.operator, s.genres.join(", "), s.language, s.bitrate_kbps + " kbps"].filter(Boolean).join(" · ");
  const action = s.added
    ? el("span", { class: "sl-tag sl-tag--plain", "data-added": "" }, t("tools_catalogue_added"))
    : button(t("tools_station_add"), "plain", () => guarded(() => toolsCatalogueAddOne(s)),
      { small: true, attrs: { "data-action": "add", "aria-label": rowName(t("tools_station_add"), s.name) } });
  return el("li", { "data-catalogue-station": s.name },
    el("div", { class: "row" }, el("strong", {}, s.name), action),
    el("p", { class: "hint" }, facts),
    s.official ? [] : el("p", { class: "hint", "data-unofficial": "" }, t("tools_catalogue_unofficial")),
    toolsHttps(s.policy_url)
      ? el("p", { class: "hint" }, el("a", { href: s.policy_url, target: "_blank", rel: "noopener noreferrer" }, t("tools_catalogue_policy", { operator: s.operator })))
      : []);
}

function toolsCatalogueGroup(group) {
  const label = t("tools_catalogue_group_" + group.id);
  return el("div", { class: "stack", "data-catalogue-group": group.id },
    el("div", { class: "row" }, el("h4", { class: "setting-entry" }, label),
      button(t("tools_catalogue_add_all"), "plain", () => guarded(() => toolsCatalogueAddAll(group)),
        { small: true, attrs: { "data-action": "add-all", "aria-label": rowName(t("tools_catalogue_add_all"), label) } })),
    el("ul", { class: "stack" }, group.stations.map(toolsCatalogueStation)));
}

// The collapsible section; its open state outlives the rebuild after each add.
function toolsCatalogue() {
  const data = toolsState.catalogue;
  if (!data || !data.groups.length) return [];
  const id = "tools-catalogue";
  const open = toolsState.catalogueOpen;
  const body = el("div", { class: "stack", id, hidden: !open },
    el("p", { class: "hint" }, t("tools_catalogue_intro")), data.groups.map(toolsCatalogueGroup));
  const toggle = el("button", { type: "button", class: "sl-disclosure", id: "tools-catalogue-toggle", "aria-expanded": String(open), "aria-controls": id }, t("tools_catalogue"));
  toggle.addEventListener("click", () => {
    toolsState.catalogueOpen = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(toolsState.catalogueOpen));
    body.hidden = !toolsState.catalogueOpen;
  });
  return el("div", { class: "stack" }, el("h3", { class: "setting-entry" }, toggle), body);
}

// ---------------------------------------------------------------- view

function toolsCard(name) {
  const extra = { weather: toolsWeatherExtra, radio: toolsRadioExtra }[name];
  const node = card(t("tab_tools"), toolsName(name),
    el("p", { class: "hint" }, t("tools_about_" + name)),
    el("p", { id: "tools-status-" + name }), toolsSwitchControl(name),
    toolsState.rows ? [] : el("p", { class: "hint" }, t("tools_settings_unavailable")),
    extra && toolsState.rows ? extra() : [], toolsSettings(name));
  node.setAttribute("data-tools-card", name);
  return node;
}

// The tool cards, built again from the settings rows read last; the statuses come from the poll.
function toolsBuild() {
  const box = document.getElementById("tools-cards");
  if (!box) return;
  box.replaceChildren(...Object.keys(TOOLS_CARDS).map(toolsCard));
  if (toolsState.tools.length) toolsStatus({ tools: toolsState.tools });
}

// The settings rows the cards read; null when the settings cannot be read (the cards then show
// their status only).
async function toolsRowsFetch() {
  try {
    const data = await api("GET", "/api/settings");
    return new Map([...data.fields, ...(data.robot ? data.robot.fields : [])].map((f) => [f.path, f]));
  } catch (err) {
    if (err.status === 401) throw err;
    return null;
  }
}

// After a settings write from this tab: the rows read again, the cards rebuilt and the state polled.
async function toolsReload() {
  const current = viewToken();
  const [rows, catalogue] = await Promise.all([toolsRowsFetch(), toolsCatalogueFetch()]);
  if (!current()) return;
  toolsState.rows = rows;
  toolsState.catalogue = catalogue;
  toolsBuild();
  await toolsRefresh();
}

// ---------------------------------------------------------------- skills

const SKILLS = "/api/tools/skill_packs";

function skillsRow(pack) {
  const tones = { on: "success", off: "info", invalid: "warning", missing: "warning", too_large: "warning" };
  const parts = [el("strong", {}, pack.name), " ", badge(t("skills_state_" + pack.state), tones[pack.state] || "info")];
  if (pack.description) parts.push(" ", el("span", {}, pack.description));
  if (pack.state === "invalid") parts.push(" ", el("span", { class: "hint" }, t("skills_reason_" + pack.reason)));
  if (pack.state === "too_large") parts.push(" ", el("span", { class: "hint" }, t("skills_too_large", { route: pack.route })));
  if (pack.ignored_files && pack.ignored_files.length) {
    parts.push(" ", el("span", { class: "hint" }, t("skills_ignored", { files: pack.ignored_files.join(", ") })));
  }
  const actions = [];
  if (pack.state === "off") actions.push(button(t("skills_turn_on"), "primary", () => guarded(() => skillsEnable(pack.name, true)), { small: true }));
  if (pack.state === "on" || pack.state === "too_large") {
    actions.push(button(t("skills_turn_off"), "plain", () => guarded(() => skillsEnable(pack.name, false)), { small: true }));
  }
  if (pack.state !== "missing") actions.push(button(t("skills_remove"), "danger", () => guarded(() => skillsRemove(pack.name)), { small: true }));
  return el("li", { "data-skill": pack.name }, parts, actions.length ? el("div", { class: "row" }, actions) : []);
}

function skillsCard(data) {
  const upload = el("input", { type: "file", accept: ".md,.zip", id: "skills-file", class: "sl-input",
    onchange: (ev) => { const f = ev.target.files[0]; ev.target.value = ""; if (f) guarded(() => skillsUploadFile(f)); } });
  const active = data.active
    ? el("p", { id: "skills-active" }, t("skills_active", { name: data.active }), " ",
      button(t("skills_end"), "danger", () => guarded(skillsEnd), { small: true }))
    : [];
  const list = data.packs.length
    ? el("ul", { class: "sl-list", id: "skills-list" }, data.packs.map(skillsRow))
    : el("p", { class: "hint", id: "skills-list" }, t("skills_none"));
  return card(t("tab_tools"), t("skills_title"),
    el("p", { class: "hint" }, t("skills_intro")),
    data.tools_listed ? [] : el("p", { class: "hint" }, t("skills_tools_missing")),
    active, list,
    el("div", { class: "row" }, field("skills-file", t("skills_add"), upload),
      button(t("skills_rescan"), "plain", () => guarded(skillsRescan), { small: true })));
}

async function skillsLoad() {
  const box = document.getElementById("skills-section");
  if (!box) return;
  try {
    box.replaceChildren(skillsCard(await api("GET", SKILLS)));
  } catch (err) {
    if (err.status === 401) throw err;
    box.replaceChildren(card(t("tab_tools"), t("skills_title"), el("p", { class: "hint" }, explain(err))));
  }
}

// One PIN-confirmed Skills write; the section is read again after it.
async function skillsWrite(title, message, confirmLabel, call, opts = {}) {
  const done = await ask({ title, message, items: opts.items, pin: true, confirm: confirmLabel, danger: opts.danger,
    run: async ({ pin }) => { await call(pin); } });
  if (done) {
    if (opts.notice) notice(opts.notice);
    await skillsLoad();
  }
  return done;
}

async function skillsEnable(name, on) {
  const path = `${SKILLS}/${encodeURIComponent(name)}/enable`;
  const call = (pin) => api("POST", path, { on, pin, confirm: true });
  if (!on) return skillsWrite(t("skills_turn_off"), t("skills_off_message", { name }), t("skills_turn_off"), call);
  const detail = await api("GET", `${SKILLS}/${encodeURIComponent(name)}`);
  return skillsWrite(t("skills_turn_on"), t("skills_on_message", { name }), t("skills_turn_on"), call,
    { items: [detail.body], notice: t("skills_on_done", { name }) });
}

async function skillsRemove(name) {
  return skillsWrite(t("skills_remove"), t("skills_remove_message", { name }), t("skills_remove"),
    (pin) => api("POST", `${SKILLS}/${encodeURIComponent(name)}/remove`, { pin, confirm: true }), { danger: true });
}

async function skillsEnd() {
  return skillsWrite(t("skills_end"), t("skills_end_message"), t("skills_end"),
    (pin) => api("POST", `${SKILLS}/end`, { pin }), { danger: true });
}

async function skillsRescan() {
  return skillsWrite(t("skills_rescan"), t("skills_rescan_message"), t("skills_rescan"),
    (pin) => api("POST", `${SKILLS}/rescan`, { pin }));
}

function skillsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(Object.assign(new Error("bad_upload"), { status: 0 }));
    reader.readAsDataURL(file);
  });
}

// Upload one file; a name already installed asks once more, to replace it.
async function skillsUploadFile(file) {
  const data = await skillsBase64(file);
  const send = (replace) => (pin) => api("POST", `${SKILLS}/upload`,
    { filename: file.name, data_b64: data, replace, pin, confirm: true });
  let exists = false;
  const done = await ask({ title: t("skills_add"), message: t("skills_add_message", { file: file.name }), pin: true,
    confirm: t("skills_add"), run: async ({ pin }) => {
      try { await send(false)(pin); } catch (err) { if (err.message !== "exists") throw err; exists = true; }
    } });
  if (done && exists) {
    await skillsWrite(t("skills_replace"), t("skills_replace_message", { file: file.name }), t("skills_replace"), send(true),
      { danger: true, notice: t("skills_added") });
    return;
  }
  if (done) {
    notice(t("skills_added"));
    await skillsLoad();
  }
}

async function toolsView() {
  const current = newView();
  const view = document.getElementById("view");
  view.replaceChildren();
  // Built before the first reply, so a first poll that fails has a grid to report for (pollFailed).
  toolsState.live = el("div", { class: "live-grid", id: "tools-live" });
  pollForget();
  view.append(el("div", { class: "stack" }, el("p", { class: "hint" }, t("tools_intro")), staleNote("tools-stale", ["tools-live"]),
    toolsState.live, el("div", { class: "stack", id: "tools-cards" }), el("div", { class: "stack", id: "skills-section" })));
  const [rows, catalogue] = await Promise.all([toolsRowsFetch(), toolsCatalogueFetch()]);
  if (!current()) return;
  toolsState.rows = rows;
  toolsState.catalogue = catalogue;
  toolsBuild();
  await skillsLoad();
  await toolsRefresh();
}

document.addEventListener("visibilitychange", () => {
  if (state.tab !== "tools" || state.csrf === null) return;
  if (document.visibilityState === "visible") guarded(toolsRefresh);
  else {
    toolsState.serial += 1;
    stopPolling();
  }
});
