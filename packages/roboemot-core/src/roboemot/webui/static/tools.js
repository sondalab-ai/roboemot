"use strict";
// Tools tab (SPEC 11, 11A, 12): the voice tools. The timers and reminders, the missed items and
// what the radio plays come from GET /api/tools/state (a passive poll, at `webui.tools.poll_ms`);
// one card per tool shows its switch, a short status and its settings, read from GET /api/settings.
// Every write re-enters the PIN: cancel, clear the missed items, stop and test-play the radio and
// the place search are POSTs of their own (the query travels in the body, never in a URL). The
// switches and the settings are saved through the batch settings save, an admin opt-in confirmed
// through the batch preview first (settingsPreviewedSave). The station editor adds and removes an
// entry through the settings entry routes and saves its fields through the batch. A station URL
// shows only in the editor, never in the polled state. Shares the helpers of app.js, settings.js
// and live.js, so it loads before app.js and calls them at run time.
//
// Polling: the timer is `state.timer` (stopPolling clears it), only while the tab is open and the
// document visible; `toolsState.serial` retires a request in flight when the tab is left or hidden.
// The tool cards are built once per settings read and keep their edits across polls.

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
const toolsState = { serial: 0, rows: null, cards: null, live: null, tools: [] };

const toolsId = (path) => "tools-" + path.replace(/[^A-Za-z0-9]+/g, "-");
const toolsName = (name) => t("tools_name_" + name);

// ---------------------------------------------------------------- the polled part

function toolsItemRow(item, missed) {
  const what = item.text || t("tools_no_label");
  const repeat = item.repeat === "days" ? t("tools_repeat_days", { days: item.days.join(", ") }) : t("tools_repeat_" + item.repeat);
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
    box.replaceChildren(badge(t(key), tone), tool.enabled && !tool.available ? [" ", t("tools_unavailable_" + tool.name)] : []);
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
  const query = textInput("tools-place-query", { maxlength: "100", autocomplete: "off" });
  const results = el("div", { class: "stack", id: "tools-place-results" });
  return [markedEyebrow("h3", t("tools_place_title")), toolsPlaceLine(),
    field("tools-place-query", t("tools_place_search_label"), query, t("tools_place_search_hint")),
    el("div", { class: "row" }, button(t("tools_place_search"), "plain", () => guarded(() => toolsPlaceSearch(query.value, results)), { small: true, attrs: { id: "tools-place-search" } })),
    results];
}

// ---------------------------------------------------------------- radio: the station editor

function toolsStation(name, removable) {
  const base = TOOLS_STATIONS + "." + name + ".";
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
      (pin) => api("POST", "/api/settings/entry/remove", { path: TOOLS_STATIONS + "." + name, pin }));
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
  const names = Array.isArray(list.value) ? list.value : [];
  const removable = new Set(list.removable || []);
  return [markedEyebrow("h3", t("tools_stations")), el("p", { class: "hint" }, t("tools_stations_intro")),
    names.length ? names.map((n) => toolsStation(n, removable.has(n))) : el("p", { class: "hint" }, t("tools_stations_none")),
    toolsStationAdd()];
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
  const rows = await toolsRowsFetch();
  if (!current()) return;
  toolsState.rows = rows;
  toolsBuild();
  await toolsRefresh();
}

async function toolsView() {
  const current = newView();
  const view = document.getElementById("view");
  view.replaceChildren();
  // Built before the first reply, so a first poll that fails has a grid to report for (pollFailed).
  toolsState.live = el("div", { class: "live-grid", id: "tools-live" });
  pollForget();
  view.append(el("div", { class: "stack" }, el("p", { class: "hint" }, t("tools_intro")), staleNote("tools-stale", ["tools-live"]),
    toolsState.live, el("div", { class: "stack", id: "tools-cards" })));
  const rows = await toolsRowsFetch();
  if (!current()) return;
  toolsState.rows = rows;
  toolsBuild();
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
