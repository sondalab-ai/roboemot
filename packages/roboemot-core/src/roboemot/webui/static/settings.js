"use strict";
// Settings page (SPEC 11A, 12): a mirror of the field catalogue served by /api/settings. Controls are
// generated from each descriptor's kind; groups render when opened. Shares the helpers of app.js
// (el, t, api, ask, notice, ...), so this file loads first and only calls them at run time.
// Text goes through textContent and styles through the CSSOM: the CSP forbids markup injection and style attributes.

const SETTING_CONTAINERS = ["dict_of_model", "keyed_list", "string_list_dict"];
const SETTING_ENTRY_KINDS = ["dict_of_model", "keyed_list"];
const SETTING_SLIDER_STEPS = 100;
const SETTING_DEFAULT_MAX_CHARS = 120;
// Mirrors LANGUAGE_MAPS in webui/settings.py: a language is never added from the web. The server refuses it anyway.
const SETTING_LANGUAGE_MAPS = ["voice.keywords", "safety.minor_lines"];
const SETTING_SEARCH_DEBOUNCE_MS = 150;
const SETTING_SEARCH_MIN_CHARS = 2;
const SETTING_AUTO_OPEN_MAX = 50;
const SETTING_UNSAFE_KEYS = ["__proto__", "constructor", "prototype"];

const settingsState = {
  query: "", changedOnly: false, open: new Set(), collapsed: new Set(),
  groups: [], restartPending: false, restarting: false, serial: 0, data: null,
};
let revealedSettings = new Map(); // private fields shown after a PIN check, kept in memory only

// Drops everything the page learned: revealed private values, the catalogue and the view state.
// Called on logout before the API call, so nothing survives a failed logout request.
function resetSettingsState() {
  revealedSettings = new Map();
  Object.assign(settingsState, {
    query: "", changedOnly: false, open: new Set(), collapsed: new Set(), groups: [], restartPending: false, restarting: false, data: null,
  });
  clearTimeout(settingsState.timer);
  settingsState.timer = null;
  const view = document.getElementById("view");
  if (view) view.replaceChildren();
}

// ---------------------------------------------------------------- model

// Rows of the catalogue with the revealed private fields merged in by path (a revealed container
// brings entry rows the plain list never had).
function mergeRevealed(fields) {
  const merged = new Map(fields.map((f) => [f.path, f]));
  for (const [path, shown] of revealedSettings) {
    const base = merged.get(path);
    merged.set(path, base ? { ...base, value: shown.value, default: shown.default, entry: shown.entry } : shown);
  }
  return [...merged.values()];
}

// Nodes in a tree: a row sits under the nearest container that is a path prefix of it.
// Returns the top-level nodes in schema order.
function settingsTree(fields) {
  const rows = mergeRevealed(fields);
  const nodes = new Map(rows.map((row) => [row.path, { row, kids: [], parent: null }]));
  const top = [];
  for (const node of nodes.values()) {
    let parent = null;
    let path = node.row.path;
    while (!parent && path.includes(".")) {
      path = path.slice(0, path.lastIndexOf("."));
      const found = nodes.get(path);
      if (found && SETTING_CONTAINERS.includes(found.row.kind)) parent = found;
    }
    node.parent = parent;
    (parent ? parent.kids : top).push(node);
  }
  return top;
}

function settingsSections(top, scope, robot) {
  const order = [];
  const by = new Map();
  const plugin = scope === "robot" ? "robot." + robot.id + "." : null;
  for (const node of top) {
    const path = node.row.path;
    const name = plugin && path.startsWith(plugin) ? path.slice(plugin.length).split(".")[0] : path.split(".")[0];
    if (!by.has(name)) { by.set(name, []); order.push(name); }
    by.get(name).push(node);
  }
  return order.map((name) => ({ id: scope + "-" + name, scope, title: name, nodes: by.get(name) }));
}

const nodeText = (node) => (node.row.path + " " + (node.row.description || "")).toLowerCase();
const isChanged = (node) => (SETTING_CONTAINERS.includes(node.row.kind)
  ? node.kids.some(isChanged)
  : node.row.source === "runtime" || JSON.stringify(node.row.value) !== JSON.stringify(node.row.default));

function nodeMatches(node, words, changedOnly) {
  const hit = (n) => words.every((w) => nodeText(n).includes(w)) || n.kids.some(hit);
  return hit(node) && (!changedOnly || isChanged(node));
}

const anyRuntime = (node) => node.row.source === "runtime" || node.kids.some(anyRuntime);

// ---------------------------------------------------------------- small helpers

const nextSettingId = () => "set-" + settingsState.serial++;
const settingHidden = (row) => row.private && !revealedSettings.has(row.path);
const firstControl = (node) => node.querySelector("input, select, textarea");

function formatValue(value) {
  if (value === null || value === undefined) return "null";
  if (Array.isArray(value) && value.every((x) => typeof x === "string")) return value.join(", ") || "—";
  if (typeof value === "object" || Array.isArray(value)) return JSON.stringify(value);
  return value === "" ? "—" : String(value);
}

function clip(text) {
  return text.length > SETTING_DEFAULT_MAX_CHARS ? text.slice(0, SETTING_DEFAULT_MAX_CHARS) + "…" : text;
}

function setNested(target, path, value) {
  const parts = path.split(".");
  let at = target;
  for (const part of parts.slice(0, -1)) at = at[part] = at[part] || {};
  at[parts[parts.length - 1]] = value;
}

function fieldControl(inner) {
  return el("span", { class: FIELD_FX }, inner);
}

function paintSlider(slider) {
  const span = Number(slider.max) - Number(slider.min);
  const pct = span > 0 ? ((Number(slider.value) - Number(slider.min)) / span) * 100 : 0;
  slider.style.setProperty("--sl-slider-value", Math.min(100, Math.max(0, pct)) + "%");
}

// ---------------------------------------------------------------- controls
// Each builder returns {node, read(), labelFor?, ownsLabel?}. read() gives {value} or {code}, with the
// server's own error codes so the page maps them like a refusal. The server stays the authority.

function numberInput(desc, id, label) {
  const integer = desc.kind === "int";
  return el("input", {
    type: "number", id, class: "sl-field__input", step: integer ? "1" : "any",
    min: desc.ge, max: desc.le, value: desc.value === null || desc.value === undefined ? "" : String(desc.value),
    "aria-label": label,
  });
}

function readNumber(desc, input) {
  if (input.value.trim() === "") return { code: "value_type" };
  const n = Number(input.value);
  if (!Number.isFinite(n) || (desc.kind === "int" && !Number.isInteger(n))) return { code: "value_type" };
  if ((desc.ge !== null && desc.ge !== undefined && n < desc.ge) || (desc.le !== null && desc.le !== undefined && n > desc.le)) {
    return { code: "value_range" };
  }
  return { value: n };
}

function numberControl(desc, id, opts) {
  const input = numberInput(desc, id, opts.label);
  const parts = [fieldControl(input)];
  const bounded = desc.ge !== null && desc.ge !== undefined && desc.le !== null && desc.le !== undefined && desc.ge < desc.le;
  if (bounded && !opts.plain) {
    const step = desc.kind === "int" ? 1 : 10 ** Math.floor(Math.log10((desc.le - desc.ge) / SETTING_SLIDER_STEPS));
    const slider = el("input", {
      type: "range", class: "sl-slider", min: desc.ge, max: desc.le, step: String(step),
      value: input.value || String(desc.ge), "aria-label": t("settings_slider", { path: opts.label }),
    });
    paintSlider(slider);
    slider.addEventListener("input", () => { input.value = slider.value; paintSlider(slider); });
    input.addEventListener("input", () => {
      if (input.value.trim() === "") return;
      slider.value = input.value;
      paintSlider(slider);
    });
    parts.unshift(slider);
  }
  if (desc.unit && !opts.plain) parts.push(el("span", { class: "hint" }, desc.unit));
  return { node: el("div", { class: "row" }, parts), labelFor: id, main: input, read: () => readNumber(desc, input) };
}

function boolControl(desc, id, opts) {
  const input = el("input", { type: "checkbox", id, role: "switch", class: "sl-switch__input", checked: desc.value === true });
  const node = el("label", { class: "sl-switch" }, input, el("span", { class: "sl-switch__track" }),
    el("span", { class: "sl-switch__label" }, opts.label));
  return { node, main: input, ownsLabel: true, read: () => ({ value: input.checked }) };
}

function enumControl(desc, id) {
  const choices = desc.choices || [];
  const known = choices.some((c) => c === desc.value);
  const select = el("select", { id, class: "sl-field__input" },
    known ? [] : el("option", { value: "", selected: true }, "—"),
    choices.map((c, n) => el("option", { value: String(n), selected: c === desc.value }, String(c))));
  return {
    node: el("span", { class: FIELD_FX }, el("span", { class: "sl-select" }, select)),
    labelFor: id, main: select,
    read: () => (select.value === "" ? { code: "value_choice" } : { value: choices[Number(select.value)] }),
  };
}

function textControl(desc, id) {
  const multi = typeof desc.value === "string" && desc.value.includes("\n");
  const input = multi
    ? el("textarea", { id, class: "sl-field__input", rows: "3" }, desc.value)
    : el("input", { type: "text", id, class: "sl-field__input", value: desc.value === null || desc.value === undefined ? "" : String(desc.value) });
  return { node: fieldControl(input), labelFor: id, main: input, read: () => ({ value: input.value }) };
}

// A list of removable rows plus an Add button. `make(value, n)` builds one row's control.
function rowsControl(desc, id, opts, make, blank, initial, canRemove) {
  const list = el("div", { class: "stack" });
  const rows = [];
  const add = button(t("settings_add"), "plain", () => { addRow(blank(), true); firstControl(rows[rows.length - 1].box).focus(); }, { small: true });
  const label = () => rows.forEach((r, n) => r.relabel(n + 1));
  const addRow = (value, fresh) => {
    const control = make(value, id + "-" + settingsState.serial++);
    const remove = button(t("settings_remove"), "plain", () => {
      rows.splice(rows.indexOf(entry), 1);
      box.remove();
      label();
    }, { small: true });
    remove.disabled = !(fresh || canRemove);
    const box = el("div", { class: "entry" }, control.node, remove);
    const entry = { control, box, relabel: (n) => remove.setAttribute("aria-label", t("settings_remove_n", { n })) };
    rows.push(entry);
    list.append(box);
    label();
  };
  initial.forEach((v) => addRow(v, false));
  return {
    node: el("div", { class: "stack", role: "group", "aria-label": opts.label }, list, el("div", { class: "row" }, add)),
    rows,
  };
}

function itemDesc(desc) {
  return { kind: desc.item_kind || "text", ge: null, le: null, unit: null, nullable: false, value: null, choices: null };
}

function tupleControl(desc, id, opts, value) {
  const item = itemDesc(desc);
  const cells = Array.from({ length: desc.length || 0 }, (_, n) => {
    const part = { ...item, value: value ? value[n] : null };
    const label = t("settings_tuple_item", { n: n + 1, total: desc.length });
    if (item.kind !== "text") return numberControl(part, id + "-" + n, { label, plain: true });
    const input = el("input", { type: "text", class: "sl-field__input", id: id + "-" + n, value: part.value === null ? "" : part.value, "aria-label": label });
    return { node: fieldControl(input), read: () => ({ value: input.value }) };
  });
  return {
    node: el("div", { class: "row", role: "group", "aria-label": opts.label }, cells.map((c) => c.node)),
    read: () => {
      const out = cells.map((c) => c.read());
      const bad = out.find((o) => o.code);
      return bad || { value: out.map((o) => o.value) };
    },
  };
}

function tupleListControl(desc, id, opts) {
  const blank = () => null;
  const made = rowsControl(desc, id, opts, (value, rid) => tupleControl(desc, rid, opts, value), blank, desc.value || [], true);
  return {
    node: made.node,
    read: () => {
      const out = made.rows.map((r) => r.control.read());
      return out.find((o) => o.code) || { value: out.map((o) => o.value) };
    },
  };
}

function chipsControl(desc, id, opts) {
  const items = [...(desc.value || [])];
  const original = new Set(items);
  const append = desc.conservative === "superset";
  const chips = el("div", { class: "sl-chips", role: "group", "aria-label": opts.label });
  const paint = () => {
    chips.replaceChildren(...items.map((item) => {
      const chip = el("button", {
        type: "button", class: "sl-chip" + PRESS,
        "aria-label": t("settings_remove_item", { item: String(item) }),
        onclick: () => { items.splice(items.indexOf(item), 1); paint(); },
      }, String(item));
      chip.disabled = append && original.has(item);
      return chip;
    }));
  };
  let entry;
  let adder;
  const take = () => {
    const text = desc.kind === "enum_list" ? desc.choices[Number(entry.value)] : entry.value.trim().replace(/\s+/g, " ");
    if (text === undefined || text === "") return;
    if (!items.includes(text)) items.push(text);
    if (desc.kind === "string_list") entry.value = "";
    paint();
    if (desc.kind === "enum_list") fill();
  };
  const fill = () => {
    entry.replaceChildren(...(desc.choices || []).filter((c) => !items.includes(c)).map((c) => el("option", { value: String(desc.choices.indexOf(c)) }, String(c))));
    adder.disabled = !entry.options.length;
  };
  if (desc.kind === "enum_list") {
    entry = el("select", { class: "sl-field__input", id, "aria-label": t("settings_new_item") });
    adder = button(t("settings_add"), "plain", take, { small: true });
    fill();
  } else {
    entry = el("input", { type: "text", class: "sl-field__input", id, "aria-label": t("settings_new_item") });
    entry.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); take(); } });
    adder = button(t("settings_add"), "plain", take, { small: true });
  }
  const adding = el("div", { class: "entry" }, desc.kind === "enum_list"
    ? el("span", { class: FIELD_FX }, el("span", { class: "sl-select" }, entry)) : fieldControl(entry), adder);
  paint();
  return { node: el("div", { class: "stack" }, chips, adding), labelFor: id, main: entry, read: () => ({ value: [...items] }) };
}

function dictControl(desc, id, opts) {
  const numeric = desc.kind === "number_dict";
  const entries = Object.entries(desc.value || {});
  const made = rowsControl(desc, id, opts, ([key, value], rid) => {
    const name = el("input", { type: "text", class: "sl-field__input", id: rid + "-k", value: key, "aria-label": t("settings_dict_key") });
    const valueDesc = { kind: numeric ? "number" : "text", ge: null, le: null, nullable: false, value };
    const control = numeric
      ? numberControl(valueDesc, rid + "-v", { label: t("settings_dict_value"), plain: true })
      : (() => { const c = textControl(valueDesc, rid + "-v"); c.main.setAttribute("aria-label", t("settings_dict_value")); return c; })();
    return {
      node: el("div", { class: "entry" }, fieldControl(name), control.node),
      key: () => name.value.trim(), read: control.read,
    };
  }, () => ["", numeric ? null : ""], entries, false);
  return {
    node: made.node,
    read: () => {
      const out = Object.create(null);
      for (const { control } of made.rows) {
        const key = control.key();
        const got = control.read();
        if (key === "" && (got.code || got.value === "")) continue;
        if (key === "" || SETTING_UNSAFE_KEYS.includes(key)) return { code: "key_invalid" };
        if (got.code) return got;
        if (Object.hasOwn(out, key)) return { code: "key_duplicate" };
        out[key] = got.value;
      }
      return { value: { ...out } };
    },
  };
}

function nullable(desc, ctl, id) {
  const off = el("input", { type: "checkbox", id: id + "-null", checked: desc.value === null });
  const sync = () => {
    for (const c of ctl.node.querySelectorAll("input, select, textarea, button")) c.disabled = off.checked;
  };
  off.addEventListener("change", sync);
  sync();
  return { ...ctl, node: el("div", { class: "stack" }, ctl.node, check(t("settings_null"), off)), read: () => (off.checked ? { value: null } : ctl.read()) };
}

// A control for the descriptor `desc` (a catalogue row, or one field of an entry to add).
function buildControl(desc, id, opts) {
  const kinds = {
    number: numberControl, int: numberControl, bool: boolControl, enum: enumControl, text: textControl,
    string_list: chipsControl, enum_list: chipsControl, tuple_list: tupleListControl,
    string_dict: dictControl, number_dict: dictControl,
    tuple: (d, i, o) => tupleControl(d, i, o, d.value),
  };
  const ctl = kinds[desc.kind](desc, id, opts);
  return desc.nullable ? nullable(desc, ctl, id) : ctl;
}

// ---------------------------------------------------------------- rows

function settingTags(row, editable) {
  const source = state.strings["settings_source_" + row.source] || row.source;
  const tags = [el("span", { class: row.source === "runtime" ? "sl-tag" : "sl-tag sl-tag--plain" }, source)];
  if (row.restart) {
    const badge = el("span", { class: "sl-badge sl-badge--warning" }, t("settings_restart_badge"));
    tags.push(canRestart() ? el("a", { href: "#settings-restart", onclick: focusRestart, "aria-label": t("settings_restart_goto"), title: t("settings_restart_goto") }, badge) : badge);
  }
  if (row.admin_only) tags.push(el("span", { class: "sl-tag", title: t("settings_admin_hint") }, t("settings_admin")));
  if (row.private) tags.push(el("span", { class: "sl-tag sl-tag--plain" }, t("settings_private")));
  if (row.secret) tags.push(el("span", { class: "sl-tag sl-tag--plain" }, t("settings_secret")));
  if (!editable && !row.secret && !row.private) tags.push(el("span", { class: "sl-tag sl-tag--plain" }, t("settings_readonly")));
  return tags;
}

const canRestart = () => {
  const robot = settingsState.data && settingsState.data.robot;
  return !!robot && (robot.actions || []).includes("restart");
};

function focusRestart(ev) {
  ev.preventDefault();
  const btn = document.getElementById("settings-restart");
  if (!btn) return;
  if (btn.scrollIntoView) btn.scrollIntoView();
  btn.focus();
}

// The Restart RoboEmot action: PIN and a dialog that lists what the robot does, then the server
// answers before it restarts. The session ends with the process, so the page says to sign in again.
async function restartRobot(robot) {
  const effects = (robot.restart_effects || []).map((e) => t("settings_restart_effect_" + e.id, e.params));
  const done = await pinnedCall(t("settings_restart_message"), t("settings_restart_confirm"),
    (pin) => api("POST", "/api/robot/restart", { pin }), effects, t("settings_restart_title"));
  if (done) {
    settingsState.restartPending = false;
    settingsState.restarting = true;
    refreshSettings();
    notice(t("settings_restart_started"));
  }
}

// The action button, or nothing when the robot offers no restart. It stays disabled once a restart was accepted.
function restartButton() {
  if (!canRestart()) return [];
  const robot = settingsState.data.robot;
  const btn = button(t("settings_restart"), "plain", () => guarded(() => restartRobot(robot)), { small: true, attrs: { id: "settings-restart" } });
  btn.disabled = settingsState.restarting;
  return btn;
}

function isIdentity(node) {
  const parent = node.parent;
  if (!parent || parent.row.kind !== "keyed_list") return false;
  const name = node.row.path.slice(parent.row.path.length + 1).split(".").pop();
  return name === parent.row.key || (parent.row.key === "event" && name === "when");
}

const canEdit = (node) => {
  const row = node.row;
  return row.writable && !row.secret && !SETTING_CONTAINERS.includes(row.kind) && row.kind !== "json" && !isIdentity(node);
};

const labelOf = (node) => (node.parent ? node.row.path.slice(node.parent.row.path.length + 1) : node.row.path);

// Runs a write behind the PIN dialog. Returns the response, or null when the dialog was cancelled.
async function pinnedCall(message, confirm, call, items, title) {
  let result = null;
  const done = await ask({
    title: title || t("settings_pin_title"), message, items, pin: true, confirm,
    run: async ({ pin }) => { result = await call(pin); },
  });
  return done ? result : null;
}

function writeNotice(result, verbKey) {
  if (!result.changed) { notice(t("settings_unchanged")); return; }
  const parts = [t(verbKey)];
  if (result.layer === "app") parts.push(t("settings_layer_app"));
  else if (result.layer === "runtime") parts.push(t("settings_layer_runtime"));
  if (result.persisted === false) parts.push(t("settings_not_persisted"));
  if (result.restart_required) { settingsState.restartPending = true; parts.push(t("settings_restart_needed")); }
  notice(parts.join(" "), result.persisted === false);
}

function settingLeaf(node) {
  const row = node.row;
  const hidden = settingHidden(row);
  const editable = canEdit(node) && !hidden;
  const id = nextSettingId();
  const ctl = editable ? buildControl(row, id, { label: labelOf(node) }) : null;
  const failure = el("p", { class: "sl-field__error", id: id + "-err", role: "alert", hidden: true });
  const name = el("code", { class: "sl-code" }, labelOf(node));
  const title = ctl && ctl.labelFor
    ? el("label", { class: "sl-field__label", for: ctl.labelFor }, name)
    : el("span", { class: "sl-field__label" }, ctl && ctl.ownsLabel ? [] : name);
  const parts = [el("div", { class: "row" }, title, settingTags(row, editable))];
  if (row.description) parts.push(el("p", { class: "hint", id: id + "-d" }, row.description));
  if (row.conservative === "superset") parts.push(el("p", { class: "hint" }, t("settings_superset_hint")));
  if (hidden) {
    parts.push(el("p", { class: "hint setting-value" }, t("settings_hidden")),
      el("div", { class: "row" }, button(t("settings_reveal"), "plain", () => guarded(revealSettings), { small: true })));
  } else if (!ctl) {
    const shown = row.secret ? "***" : row.value === null && !row.writable ? t("settings_not_measured") : null;
    parts.push(row.kind === "json" && !row.secret
      ? el("pre", { class: "pre-scroll setting-json setting-value" }, JSON.stringify(row.value, null, 2))
      : el("p", { class: "setting-value" }, shown === null ? formatValue(row.value) : shown));
    if (row.kind === "json" && !row.secret && row.writable) parts.push(el("p", { class: "hint" }, t("settings_json_readonly")));
  } else {
    parts.push(ctl.node);
    const first = ctl.main || firstControl(ctl.node);
    if (first && row.description) first.setAttribute("aria-describedby", id + "-d");
    if (JSON.stringify(row.default) !== JSON.stringify(row.value) && row.default !== null) {
      parts.push(el("p", { class: "hint" }, t("settings_default") + ": " + clip(formatValue(row.default))));
    }
    parts.push(failure, settingActions(node, ctl, failure));
  }
  return el("div", { class: "setting stack", "data-path": row.path, "data-source": row.source }, parts);
}

function settingActions(node, ctl, failure) {
  const row = node.row;
  const descId = row.description ? failure.id.replace(/-err$/, "-d") : null;
  const mark = (code) => {
    failure.hidden = !code;
    failure.textContent = code ? t("err_" + code) : "";
    const target = ctl.main || firstControl(ctl.node);
    if (!target) return;
    if (code) { target.setAttribute("aria-invalid", "true"); target.setAttribute("aria-describedby", [descId, failure.id].filter(Boolean).join(" ")); }
    else {
      target.removeAttribute("aria-invalid");
      if (descId) target.setAttribute("aria-describedby", descId); else target.removeAttribute("aria-describedby");
    }
  };
  const save = async () => {
    const out = ctl.read();
    mark(out.code || null);
    if (out.code) return;
    if (JSON.stringify(out.value) === JSON.stringify(row.value)) { notice(t("settings_unchanged")); return; }
    const result = await pinnedCall(row.path, t("settings_save"),
      (pin) => api("POST", "/api/settings", { path: row.path, value: out.value, pin }));
    if (!result) return;
    if (revealedSettings.has(row.path) && result.changed) {
      revealedSettings.set(row.path, { ...revealedSettings.get(row.path), value: out.value });
    }
    writeNotice(result, "saved");
    await settingsReload(result.changed ? row.path : null);
  };
  const reset = async () => {
    const result = await pinnedCall(row.path, t("settings_reset_layer"),
      (pin) => api("POST", "/api/settings/reset", { path: row.path, pin }));
    if (!result) return;
    if (revealedSettings.has(row.path) && result.changed) {
      revealedSettings.set(row.path, { ...revealedSettings.get(row.path), value: result.value });
    }
    writeNotice(result, "settings_reset_done");
    await settingsReload(row.path);
  };
  const actions = [button(t("settings_save"), "primary", () => guarded(save), { small: true })];
  if (row.source === "runtime" && row.conservative !== "superset") {
    actions.push(button(t("settings_reset_layer"), "plain", () => guarded(reset), { small: true }));
  }
  return el("div", { class: "row" }, actions);
}

// ---------------------------------------------------------------- containers (entries)

function entryKeys(node) {
  return Array.isArray(node.row.value) ? node.row.value.map(String) : [];
}

// The entry a kid belongs to: the longest entry key that is a path prefix (keys `a` and `a.b` can both exist).
function entryOf(node, path) {
  const base = node.row.path + ".";
  return entryKeys(node)
    .filter((key) => path === base + key || path.startsWith(base + key + "."))
    .sort((a, b) => b.length - a.length)[0];
}

function entryKids(node, key) {
  return node.kids.filter((k) => entryOf(node, k.row.path) === key);
}

// A nested, lazily filled disclosure: `fill()` builds the body the first time it is opened.
function nestedDisclosure(openKey, title, fill) {
  const id = nextSettingId();
  const body = el("div", { id, class: "stack nested", hidden: true });
  const toggle = el("button", { type: "button", class: "sl-disclosure", "aria-expanded": "false", "aria-controls": id }, title);
  let filled = false;
  const show = (on) => {
    toggle.setAttribute("aria-expanded", String(on));
    body.hidden = !on;
    if (on && !filled) { filled = true; body.replaceChildren(...fill()); }
    if (on) settingsState.open.add(openKey); else settingsState.open.delete(openKey);
  };
  toggle.addEventListener("click", () => show(toggle.getAttribute("aria-expanded") !== "true"));
  if (settingsState.open.has(openKey)) show(true);
  return { toggle, body };
}

function entryBlock(node, key) {
  const kids = entryKids(node, key);
  const identity = kids.find((k) => isIdentity(k));
  const title = identity && identity.row.value ? key + " · " + formatValue(identity.row.value) : key;
  const { toggle, body } = nestedDisclosure("entry:" + node.row.path + "." + key, title, () => kids.map(settingNode));
  const removable = SETTING_ENTRY_KINDS.includes(node.row.kind) && !node.row.secret && kids.some(anyRuntime);
  const path = node.row.path + "." + key;
  const remove = removable
    ? button(t("settings_remove_entry"), "danger", () => guarded(async () => {
      const result = await pinnedCall(path, t("settings_remove_entry"),
        (pin) => api("POST", "/api/settings/entry/remove", { path, pin }));
      if (!result) return;
      revealedSettings = new Map();
      writeNotice(result, "settings_entry_removed");
      await settingsReload(null);
    }), { small: true })
    : [];
  return el("div", { class: "entry-block stack", "data-entry": path }, el("div", { class: "row" }, el("h4", { class: "setting-entry" }, toggle), remove), body);
}

// The form that adds an entry, built from the container's `entry` descriptors (paths relative to the entry).
function addEntryForm(node) {
  const row = node.row;
  const keyed = row.kind === "keyed_list";
  const failure = el("p", { class: "sl-field__error", role: "alert", hidden: true, id: nextSettingId() + "-err" });
  const id = nextSettingId();
  const name = keyed ? null : el("input", { type: "text", class: "sl-field__input", id: id + "-name", required: true });
  const fixed = (row.entry || []).filter((e) => e.kind === "json" && !e.secret);
  const fields = (row.entry || []).filter((e) => !e.secret && e.kind !== "json" && !SETTING_CONTAINERS.includes(e.kind))
    .map((e) => ({ desc: e, ctl: buildControl(e, id + "-" + settingsState.serial++, { label: e.path }) }));
  const show = (code) => { failure.hidden = !code; failure.textContent = code ? t("err_" + code) : ""; };
  const submit = () => guarded(async () => {
    const value = {};
    for (const e of fixed) setNested(value, e.path, e.value);
    for (const { desc, ctl } of fields) {
      const out = ctl.read();
      if (out.code) { show(out.code); return; }
      setNested(value, desc.path, out.value);
    }
    const key = keyed ? undefined : name.value.trim();
    if (!keyed && key === "") { show("key_invalid"); return; }
    show(null);
    const body = keyed ? { path: row.path, value } : { path: row.path, key, value };
    const result = await pinnedCall(row.path, t("settings_add_entry"), (pin) => api("POST", "/api/settings/entry", { ...body, pin }));
    if (!result) return;
    revealedSettings = new Map();
    writeNotice(result, "settings_entry_added");
    await settingsReload(null);
  });
  return el("div", { class: "stack" },
    name ? field(id + "-name", t("settings_entry_name"), name) : [],
    fields.map(({ desc, ctl }) => el("div", { class: "stack", role: "group", "aria-label": desc.path },
      ctl.labelFor ? el("label", { class: "sl-field__label", for: ctl.labelFor }, el("code", { class: "sl-code" }, desc.path))
        : el("span", { class: "sl-field__label" }, ctl.ownsLabel ? [] : el("code", { class: "sl-code" }, desc.path)),
      ctl.node)),
    failure, el("div", { class: "row" }, button(t("settings_add_entry"), "primary", submit, { small: true })));
}

function settingContainer(node) {
  const row = node.row;
  const hidden = settingHidden(row);
  const addable = SETTING_ENTRY_KINDS.includes(row.kind) && !row.secret && row.entry && !SETTING_LANGUAGE_MAPS.includes(row.path) && !hidden;
  const parts = [el("div", { class: "row" }, el("code", { class: "sl-code" }, labelOf(node)), settingTags(row, true))];
  if (row.description) parts.push(el("p", { class: "hint" }, row.description));
  if (hidden) {
    parts.push(el("p", { class: "hint setting-value" }, t("settings_hidden")),
      el("div", { class: "row" }, button(t("settings_reveal"), "plain", () => guarded(revealSettings), { small: true })));
  } else {
    const keys = entryKeys(node);
    parts.push(keys.length ? keys.map((key) => entryBlock(node, key)) : el("p", { class: "hint" }, t("settings_no_entries")));
    if (addable) {
      const { toggle, body } = nestedDisclosure("add:" + row.path, t("settings_add_entry"), () => [addEntryForm(node)]);
      parts.push(el("div", { class: "stack" }, toggle, body));
    }
  }
  return el("div", { class: "setting stack", "data-path": row.path, "data-source": row.source }, parts);
}

const settingNode = (node) => (SETTING_CONTAINERS.includes(node.row.kind) ? settingContainer(node) : settingLeaf(node));

// ---------------------------------------------------------------- reveal and reload

async function revealSettings() {
  const result = await pinnedCall(t("settings_reveal_message"), t("settings_reveal"),
    (pin) => api("POST", "/api/settings/reveal", { pin }));
  if (!result) return;
  revealedSettings = new Map(result.fields.map((f) => [f.path, f]));
  settingsApplyModel(settingsState.data);
  refreshSettings();
}

async function settingsReload(changedPath) {
  settingsState.data = await api("GET", "/api/settings");
  settingsApplyModel(settingsState.data);
  refreshSettings();
  if (!changedPath) return;
  const shown = document.querySelector('[data-path="' + CSS.escape(changedPath) + '"]');
  const focus = shown && shown.querySelector("input, select, textarea, button");
  if (focus) focus.focus();
  const name = shown && shown.querySelector(".sl-code");
  if (name && window.roboemotFx) {
    try { window.roboemotFx.burst(name); } catch (e) { /* effects are optional */ }
  }
}

function settingsApplyModel(data) {
  const core = settingsSections(settingsTree(data.fields), "core", null);
  const robot = data.robot ? settingsSections(settingsTree(data.robot.fields), "robot", data.robot) : [];
  const by = new Map([...core, ...robot].map((s) => [s.id, s]));
  for (const g of settingsState.groups) {
    const section = by.get(g.id);
    g.nodes = section ? section.nodes : [];
    g.shown = null;
    g.cache = new Map();
  }
}

// ---------------------------------------------------------------- groups, search, nav

// Rows are built once per catalogue load and kept by path, so unsaved edits survive a collapse or a filter change.
function cachedRow(g, node) {
  if (!g.cache.has(node.row.path)) g.cache.set(node.row.path, settingNode(node));
  return g.cache.get(node.row.path);
}

function refreshGroup(g) {
  const words = settingsState.query.toLowerCase().split(/\s+/).filter(Boolean);
  const filtering = words.length > 0 || settingsState.changedOnly;
  const hits = g.nodes.filter((n) => nodeMatches(n, words, settingsState.changedOnly));
  g.count.textContent = String(hits.length);
  g.navCount.textContent = String(hits.length);
  const gone = filtering && hits.length === 0;
  g.section.hidden = gone;
  g.navItem.hidden = gone;
  const auto = filtering && hits.length > 0 && (settingsState.changedOnly || settingsState.query.trim().length >= SETTING_SEARCH_MIN_CHARS);
  const manual = settingsState.open.has(g.id);
  const open = !gone && !settingsState.collapsed.has(g.id) && (manual || auto);
  g.toggle.setAttribute("aria-expanded", String(open));
  g.body.hidden = !open;
  if (!open) return hits.length;
  const capped = !manual && !g.showAll && hits.length > SETTING_AUTO_OPEN_MAX;
  const shown = capped ? hits.slice(0, SETTING_AUTO_OPEN_MAX) : hits;
  const key = shown.map((n) => n.row.path).join("|") + (capped ? "+" : "");
  if (g.shown !== key) {
    g.shown = key;
    const more = capped
      ? [el("div", { class: "row" }, button(t("settings_show_all", { n: hits.length }), "plain", () => { g.showAll = true; refreshGroup(g); }, { small: true }))]
      : [];
    g.body.replaceChildren(...shown.map((n, i) => [i ? el("hr", { class: "sl-rule" }) : [], cachedRow(g, n)]).flat(), ...more);
  }
  return hits.length;
}

function refreshSettings() {
  const total = settingsState.groups.reduce((sum, g) => sum + refreshGroup(g), 0);
  const status = document.getElementById("settings-count");
  if (status) status.textContent = t("settings_matches", { n: total });
  const actions = document.getElementById("settings-robot-actions");
  if (actions) actions.replaceChildren(restartButton());
  const hint = document.getElementById("settings-restart-hint");
  if (hint) hint.textContent = t(settingsState.restartPending ? "settings_restart_pending" : "settings_restart_hint");
}

function settingsGroup(section) {
  const id = "grp-" + section.id;
  const count = el("span", { class: "sl-nav__count" }, "0");
  const navCount = el("span", { class: "sl-nav__count" }, "0");
  const body = el("div", { class: "stack", id: id + "-body", hidden: true });
  const toggle = el("button", { type: "button", class: "sl-disclosure", id, "aria-expanded": "false", "aria-controls": id + "-body" },
    el("code", { class: "sl-code" }, section.title), count);
  const navItem = el("button", { type: "button", class: "sl-nav__item", "data-group": section.id }, section.title, navCount);
  const g = { id: section.id, scope: section.scope, nodes: section.nodes, count, navCount, body, toggle, navItem, shown: null, cache: new Map(), showAll: false,
    section: el("section", { class: "sl-card settings-group" }, el("h3", { class: "setting-group" }, toggle), body) };
  toggle.addEventListener("click", () => {
    const on = toggle.getAttribute("aria-expanded") !== "true";
    if (on) { settingsState.open.add(g.id); settingsState.collapsed.delete(g.id); }
    else { settingsState.open.delete(g.id); settingsState.collapsed.add(g.id); }
    refreshGroup(g);
  });
  navItem.addEventListener("click", () => {
    settingsState.open.add(g.id);
    settingsState.collapsed.delete(g.id);
    for (const other of settingsState.groups) other.navItem.removeAttribute("aria-current");
    navItem.setAttribute("aria-current", "true");
    refreshGroup(g);
    if (g.section.scrollIntoView) g.section.scrollIntoView();
    toggle.focus();
  });
  return g;
}

function settingsToolbar() {
  const query = el("input", { type: "search", id: "settings-search", class: "sl-field__input", value: settingsState.query, autocomplete: "off" });
  query.addEventListener("input", () => {
    clearTimeout(settingsState.timer);
    settingsState.timer = setTimeout(() => {
      settingsState.query = query.value;
      settingsState.collapsed.clear();
      for (const g of settingsState.groups) g.showAll = false;
      refreshSettings();
    }, SETTING_SEARCH_DEBOUNCE_MS);
  });
  const only = el("input", { type: "checkbox", id: "settings-changed", checked: settingsState.changedOnly });
  only.addEventListener("change", () => {
    settingsState.changedOnly = only.checked;
    settingsState.collapsed.clear();
    refreshSettings();
  });
  return el("div", { class: "stack" },
    field("settings-search", t("settings_search"), query, t("settings_search_hint")),
    el("div", { class: "row" }, check(t("settings_changed_only"), only),
      el("p", { class: "hint", id: "settings-count", role: "status", "aria-live": "polite" })));
}

async function settingsView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const data = await api("GET", "/api/settings");
  settingsState.data = data;
  const robot = data.robot || null;
  const sections = [
    ...settingsSections(settingsTree(data.fields), "core", null),
    ...(robot ? settingsSections(settingsTree(robot.fields), "robot", robot) : []),
  ];
  settingsState.groups = sections.map(settingsGroup);
  const hook = el("p", { class: "hint", id: "settings-restart-hint" });
  const intro = card(t("tab_settings"), t("tab_settings"), el("p", { class: "hint" }, t("settings_intro")), settingsToolbar(), hook);
  if (!sections.length) {
    view.append(intro, empty(t("tab_settings"), t("settings_none")));
    refreshSettings();
    return;
  }
  const coreGroups = settingsState.groups.filter((g) => g.scope === "core");
  const robotGroups = settingsState.groups.filter((g) => g.scope === "robot");
  const nav = el("nav", { class: "sl-nav settings-nav", "aria-label": t("settings_nav") },
    el("div", { class: "sl-nav__group" }, t("settings_core")), coreGroups.map((g) => g.navItem),
    robot ? [el("div", { class: "sl-nav__group" }, robot.title), robotGroups.map((g) => g.navItem)] : []);
  const groups = el("div", { class: "stack", id: "settings-groups" }, settingsState.groups.map((g) => g.section));
  const robotActions = robot ? el("div", { class: "row", id: "settings-robot-actions" }) : [];
  view.append(intro, robotActions, el("div", { class: "settings-layout" }, nav, groups));
  refreshSettings();
}
