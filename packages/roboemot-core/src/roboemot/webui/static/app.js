"use strict";
// RoboEmot web UI on the vendored Sondalab kit. All text goes through textContent: names from the
// registry are never HTML. Styles are set through the CSSOM only (the CSP forbids style attributes).

const ROUTES_POLL_MS = 2000;
const state = { strings: {}, lang: undefined, csrf: null, session: {}, tab: "enroll", timer: null, form: null, view: 0 };

// #view has one owner: the latest view. A view takes `newView()` before its first await and draws
// nothing once a later view, a tab switch, login or logout took #view (`current()` is false); an action
// inside a view takes `viewToken()`, which does not take #view from the view it acts in.
function newView() {
  state.view += 1;
  return viewToken();
}

function viewToken() {
  const mine = state.view;
  return () => mine === state.view;
}

const t = (key, vars) => {
  let text = state.strings[key] || key;
  for (const [k, v] of Object.entries(vars || {})) text = text.replace("{" + k + "}", () => String(v));
  return text;
};

// A server code in the page language: the string `prefix + code` (spaces and dashes as "_") when
// the page has one, else the code as it is (a name the owner chose, such as a custom label).
function word(prefix, code) {
  const key = prefix + String(code).replace(/[ -]/g, "_");
  return Object.prototype.hasOwnProperty.call(state.strings, key) ? state.strings[key] : String(code);
}

// A date (ms, ISO text or Date) in the page language, not the browser's; `timeOnly` drops the day.
function when(value, timeOnly) {
  const d = new Date(value);
  return timeOnly ? d.toLocaleTimeString(state.lang) : d.toLocaleString(state.lang);
}

// The only text the page holds itself, for when it cannot load its strings: the same keys and
// texts as webui/strings.py (tests/test_webui_page.py checks that they match).
const PAGE_HELD = {
  en: { err_unreachable: "Cannot reach the robot.", notice_dismiss: "Dismiss", retry: "Try again" },
  it: { err_unreachable: "Impossibile raggiungere il robot.", notice_dismiss: "Chiudi", retry: "Riprova" },
};

// `key` from the loaded strings, else from PAGE_HELD in the page (or browser) language.
function held(key) {
  if (state.strings[key]) return state.strings[key];
  const lang = state.lang || String(navigator.language || "").slice(0, 2);
  return (PAGE_HELD[lang] || PAGE_HELD.en)[key];
}

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") node.className = v;
    else if (k === "style") for (const [name, value] of Object.entries(v)) node.style.setProperty(name, value);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) node.append(c instanceof Node ? c : document.createTextNode(c));
  return node;
}

// ---------------------------------------------------------------- kit building blocks

const PRESS = " sl-fx-press";
const BUTTONS = {
  primary: "sl-btn sl-btn--primary" + PRESS,
  danger: "sl-btn sl-btn--ghost sl-btn--danger" + PRESS,
  plain: "sl-btn sl-btn--ghost" + PRESS,
};

// Aurora means attention lives here: the focused field, the selected tab, work that is running.
// The state itself stays in the markup (aria-busy, text); the light only renders it.
const FIELD_FX = "sl-field__control sl-fx-aurora sl-fx-aurora--field";

function busy(node, on) {
  if (!on) return node;
  node.classList.add("sl-fx-aurora", "sl-fx-aurora--wide");
  node.setAttribute("data-sl-fx-live", "busy");
  node.setAttribute("aria-busy", "true");
  return node;
}

function button(label, kind, onclick, extra) {
  const small = extra && extra.small ? " sl-btn--sm" : "";
  const props = Object.assign({ type: "button", class: BUTTONS[kind || "plain"] + small, onclick: onclick }, extra && extra.attrs);
  return el("button", props, label);
}

function submitButton(label) {
  return el("button", { type: "submit", class: BUTTONS.primary }, label);
}

// The kit's eyebrow glyphs (ui-kit 0.42), in the order the heads of a view cycle through them.
// "" is the bare sl-eyebrow--mark; the others name the modifier worn with it. Six glyphs, so two
// neighbouring heads never share one. High contrast paints none of them and leaves no gap.
const MARK_GLYPHS = ["", "square", "tri", "circle", "half", "stair"];

function markedEyebrow(tag, text) {
  return el(tag, { class: "sl-eyebrow sl-eyebrow--mark" }, text);
}

// Gives every eyebrow under root its glyph by document position, so a head never repeats the one
// just before it. A view that re-renders (the routes poll) keeps each head's glyph only while its
// number of heads stays the same; a head added or removed shifts the glyphs of the ones after it.
function assignMarks(root) {
  root.querySelectorAll(".sl-eyebrow--mark").forEach((node, i) => {
    const glyph = MARK_GLYPHS[i % MARK_GLYPHS.length];
    for (const g of MARK_GLYPHS.slice(1)) node.classList.toggle("sl-eyebrow--mark-" + g, g === glyph);
  });
}

function card(eyebrow, title, ...children) {
  return el("section", { class: "sl-card" },
    eyebrow ? markedEyebrow("p", eyebrow) : [],
    el("h2", { class: "sl-card__title" }, title),
    el("div", { class: "sl-card__body stack" }, children));
}

function textInput(id, props) {
  return el("input", Object.assign({ type: "text", id: id, class: "sl-field__input" }, props));
}

function field(id, label, control, hint) {
  if (hint) control.setAttribute("aria-describedby", id + "-hint");
  const inner = control.tagName === "SELECT" ? el("span", { class: "sl-select" }, control) : control;
  return el("div", { class: "sl-field" },
    el("label", { class: "sl-field__label", for: id }, label),
    el("span", { class: FIELD_FX }, inner),
    hint ? el("p", { class: "sl-field__hint", id: id + "-hint" }, hint) : []);
}

function check(label, input) {
  input.classList.add("sl-check__input");
  return el("label", { class: "sl-check" }, input,
    el("span", { class: "sl-check__box" }), el("span", { class: "sl-check__label" }, label));
}

function badge(text, tone) {
  return el("span", { class: "sl-badge sl-badge--" + tone }, text);
}

function progress(label, value, max) {
  const top = Math.max(max, 1);
  const shown = Math.min(value, top);
  return el("div", {
    class: "sl-progress", role: "progressbar", "aria-label": label,
    "aria-valuemin": 0, "aria-valuemax": top, "aria-valuenow": shown,
    style: { "--sl-progress": Math.round((shown / top) * 100) + "%" },
  }, el("span", { class: "sl-progress__bar" }));
}

function empty(eyebrow, body) {
  return el("div", { class: "sl-empty" }, markedEyebrow("span", eyebrow), el("p", { class: "sl-empty__body" }, body));
}

function table(head, rows) {
  return el("div", { class: "table-scroll" }, el("table", { class: "sl-table" },
    el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))),
    el("tbody", {}, rows)));
}

// A success toast leaves on its own after this long; an error stays until it is cleared, replaced or
// dismissed. The kit gives no duration: this leaves time to read a two-line message twice.
const NOTICE_SUCCESS_MS = 8000;
const toastState = { error: null, timer: null };
const SVG_NS = "http://www.w3.org/2000/svg";
const TOAST_ICONS = { success: "M3.5 8.5l3 3 6-7", danger: "M8 3.5v5M8 12v0.5", close: "M4 4l8 8M12 4l-8 8" };

// A 16-unit stroked glyph from TOAST_ICONS, hidden from assistive technology.
function toastIcon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  const path = document.createElementNS(SVG_NS, "path");
  for (const [k, v] of [["viewBox", "0 0 16 16"], ["aria-hidden", "true"], ["focusable", "false"]]) svg.setAttribute(k, v);
  for (const [k, v] of [["d", TOAST_ICONS[name]], ["fill", "none"], ["stroke", "currentColor"], ["stroke-width", "2"], ["stroke-linecap", "round"], ["stroke-linejoin", "round"]]) path.setAttribute(k, v);
  svg.append(path);
  return svg;
}

// One toast at a time in the kit's toast stack (#notice, always in the page): a success in its
// polite region, an error in its alert region; "" clears. It always shows what it is given (an
// error equal to the last one is a new toast, announced again): a poll that keeps failing speaks
// once because it calls this only when it starts failing (pollFailed, live.js).
function notice(message, isError) {
  clearTimeout(toastState.timer);
  toastState.timer = null;
  toastState.error = isError && message ? message : null;
  for (const region of document.querySelectorAll("#notice .sl-toast-stack__live")) region.replaceChildren();
  if (!message) return;
  const tone = isError ? "danger" : "success";
  const toast = el("div", { class: "sl-toast sl-toast--" + tone },
    el("span", { class: "sl-toast__mark", "aria-hidden": "true" }, toastIcon(tone)),
    el("div", { class: "sl-toast__content" }, el("p", { class: "sl-toast__title" }, message)));
  const dismiss = () => { clearTimeout(toastState.timer); toast.remove(); };
  toast.append(el("button", { type: "button", class: "sl-icon-btn sl-toast__dismiss", "aria-label": held("notice_dismiss"), onclick: dismiss },
    toastIcon("close")));
  document.getElementById(isError ? "notice-alert" : "notice-polite").append(toast);
  // A toast the user is pointing at or has focused waits: removing it would drop the focus.
  const leave = () => {
    if (toast.matches(":focus-within, :hover")) toastState.timer = setTimeout(leave, NOTICE_SUCCESS_MS);
    else dismiss();
  };
  if (!isError) toastState.timer = setTimeout(leave, NOTICE_SUCCESS_MS);
}

// Clears the error `message` if it is still the one reported last: a poll that recovers takes back
// its own failure, never a newer message from something else.
function noticeWithdraw(message) {
  if (message && message === toastState.error) notice("");
}

// ---------------------------------------------------------------- api

// `fetch`, with the one failure that has no response named as such: it rejects with the code
// "unreachable" (network down, robot off, Wi-Fi lost), which `explain` words for the page.
async function request(path, init) {
  try {
    return await fetch(path, init);
  } catch (e) {
    throw new Error("unreachable");
  }
}

async function api(method, path, body, extra) {
  const headers = { ...extra };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET" && state.csrf) headers["X-CSRF-Token"] = state.csrf;
  const resp = await request(path, {
    method,
    headers,
    credentials: "same-origin",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = {};
  try { data = await resp.json(); } catch (e) { data = {}; }
  if (!resp.ok) {
    const err = new Error(data.error || "generic");
    err.status = resp.status;
    err.endpoint = data.endpoint;
    err.retryAfter = Number(resp.headers.get("Retry-After"));
    throw err;
  }
  return data;
}

function explain(err) {
  const known = state.strings["err_" + err.message];
  let text = known || t("err_generic");
  if (err.endpoint) text = text.replace("{host}", () => err.endpoint);
  if (err.saved) text += " " + err.saved;
  const wait = err.retryAfter;
  if (err.status === 429 && wait > 0 && wait <= 60) text += " " + t("dialog_retry_in", { n: wait });
  return text;
}

async function guarded(fn) {
  try {
    await fn();
  } catch (err) {
    if (err.status === 401 && err.message === "unauthorized") { await boot(); return; }
    notice(explain(err), true);
  }
}

// `guarded` for a call that draws a view. The view takes #view before its first await, so the token
// read right after the call starts is its own; an error of a view that was dropped since (a tab
// switch, a later view, logout) is not shown on the tab the user is on now. A view that failed
// before it drew anything also says so in #view, where the user looks, besides the toast.
async function guardedView(view) {
  let current = null;
  try {
    const pending = view();
    current = viewToken();
    await pending;
  } catch (err) {
    if (current && !current()) return;
    if (err.status === 401 && err.message === "unauthorized") { await boot(); return; }
    notice(explain(err), true);
    viewFailed(explain(err));
  }
}

// An empty #view after a failed load: a kit card with the error and a retry of the tab. A view
// that drew content keeps it (its polls handle their own failures).
function viewFailed(text) {
  const view = document.getElementById("view");
  if (view.hasChildNodes()) return;
  const box = card(null, t("tab_" + state.tab), el("p", {}, text),
    el("div", { class: "row" }, button(held("retry"), "primary", () => showTab(state.tab))));
  box.id = "view-failed";
  view.append(box);
}

function stopPolling() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
  liveStopAudio(true);
  liveStopVideo(true);
}

// ---------------------------------------------------------------- dialogs

// One modal for every confirmation, rename and PIN prompt. `run({pin, value})` is the API call; the
// dialog stays open with the error shown inline until it succeeds or is cancelled. The PIN lives in
// the input only: it is read once, passed to `run` and cleared. Resolves true on success, else null.
function ask(opts) {
  return new Promise((resolve) => {
    const dialog = el("dialog", { class: "sl-dialog", "aria-labelledby": "dialog-title" });
    const pin = opts.pin
      ? textInput("dialog-pin", { type: "password", autocomplete: "off", inputmode: "numeric", required: true, maxlength: "64" })
      : null;
    const value = opts.input
      ? textInput("dialog-value", { value: opts.input.value || "", maxlength: opts.input.max, required: true })
      : null;
    const failure = el("p", { class: "sl-field__error", id: "dialog-error", role: "alert", hidden: true });
    const cancel = button(t("dialog_cancel"), "plain", () => dialog.close());
    const submit = el("button", { type: "submit", class: BUTTONS[opts.danger ? "danger" : "primary"] }, opts.confirm || t("dialog_confirm"));
    let done = null;
    let inflight = null;
    const mark = (input) => {
      for (const i of [pin, value]) if (i) { i.removeAttribute("aria-invalid"); i.removeAttribute("aria-describedby"); }
      if (input) { input.setAttribute("aria-invalid", "true"); input.setAttribute("aria-describedby", "dialog-error"); }
    };
    const form = el("form", {
      class: "dialog-form",
      onsubmit: async (ev) => {
        ev.preventDefault();
        const args = { pin: pin ? pin.value : null, value: value ? value.value : null };
        if (pin) pin.value = "";
        failure.hidden = true;
        mark(null);
        submit.disabled = true;
        cancel.disabled = true;
        inflight = Promise.resolve(opts.run(args)).then(() => { done = true; });
        try {
          await inflight;
          dialog.close();
        } catch (err) {
          if (err.status === 401 && err.message === "unauthorized") { dialog.close(); await boot(); return; }
          failure.textContent = explain(err);
          failure.hidden = false;
          const pinProblem = ["bad_pin", "locked", "busy", "pin_not_set"].includes(err.message);
          const target = (pinProblem || !value) ? pin : value;
          mark(target);
          if (target) target.focus();
        } finally {
          inflight = null;
          submit.disabled = false;
          cancel.disabled = false;
        }
      },
    },
    opts.message ? el("p", {}, opts.message) : [],
    opts.items ? el("ul", {}, opts.items.map((item) => el("li", {}, item))) : [],
    value ? field("dialog-value", opts.input.label, value) : [],
    pin ? field("dialog-pin", t("pin"), pin) : [],
    failure,
    el("div", { class: "sl-dialog__actions" }, cancel, submit));
    dialog.append(el("h2", { class: "sl-dialog__title", id: "dialog-title" }, opts.title), el("div", { class: "sl-dialog__body" }, form));
    // Esc and the Cancel button are refused while the request is pending: the outcome would be
    // lost. If the dialog closes anyway, the result still waits for the request to settle, so the
    // caller refreshes its view.
    dialog.addEventListener("cancel", (ev) => { if (inflight) ev.preventDefault(); });
    dialog.addEventListener("close", () => {
      if (pin) pin.value = "";
      dialog.remove();
      (inflight || Promise.resolve()).catch(() => {}).then(() => resolve(done));
    });
    document.body.append(dialog);
    dialog.showModal();
  });
}

// ---------------------------------------------------------------- login and setup

// The login card is alone on its screen, so it carries the page's one art piece: a corner of
// quarter discs behind the title's far end. Decorative; high contrast paints nothing there.
function loginCard(body) {
  const box = card(null, t("login_title"), body);
  box.classList.add("sl-art-corner");
  return box;
}

function loginView() {
  newView();
  const view = document.getElementById("view");
  view.replaceChildren();
  if (!state.session.pin_set) {
    view.append(loginCard(el("p", {}, t("no_pin"))));
    return;
  }
  const pin = textInput("pin", { type: "password", autocomplete: "off", inputmode: "numeric" });
  const form = el("form", {
    class: "stack",
    onsubmit: (ev) => {
      ev.preventDefault();
      guarded(async () => {
        const data = await api("POST", "/api/login", { pin: pin.value });
        pin.value = "";
        state.csrf = data.csrf;
        notice("");
        await boot();
        const tab = selectedTab(); // the login form is gone: the focus starts on the open tab
        if (tab) tab.focus();
      });
    },
  }, field("pin", t("pin"), pin), el("div", { class: "row" }, submitButton(t("login"))));
  view.append(loginCard(form));
  pin.focus();
}

// ---------------------------------------------------------------- enrollment

function enrollmentView() {
  const current = newView();
  const view = document.getElementById("view");
  view.replaceChildren();
  const box = el("div", { class: "stack" });
  view.append(box);
  const poll = async () => {
    if (!current()) return;
    stopPolling();
    try {
      const s = await api("GET", "/api/enrollment/status");
      if (!current()) return;
      await renderEnrollment(box, s, current);
      if (current() && ["collecting", "ready"].includes(s.state)) {
        state.timer = setTimeout(poll, state.session.status_poll_ms || 1000);
      }
    } catch (err) {
      if (!current()) return;
      if (err.status === 401) { await boot(); return; }
      notice(explain(err), true);
    }
  };
  state.refresh = poll;
  poll();
}

async function renderEnrollment(box, s, current) {
  const stateKey = "state_" + s.state;
  const tone = { collecting: "info", ready: "success", completing: "info", completed: "success" }[s.state] || "warning";
  const parts = [
    el("p", {}, t("enroll_intro")),
    el("p", {}, t("state") + ": ", badge(state.strings[stateKey] || s.state, tone)),
  ];
  if (!["collecting", "ready", "completing"].includes(s.state)) {
    parts.push(el("div", { class: "row" }, button(t("enroll_start"), "primary",
      () => guarded(async () => { await api("POST", "/api/enrollment/start", {}); await state.refresh(); }))));
    box.replaceChildren(card(t("tab_enroll"), t("tab_enroll"), parts));
    return;
  }
  parts.push(
    el("p", {}, t("frames") + ": " + s.frames_collected + " / " + s.frames_needed),
    progress(t("frames"), s.frames_collected, s.frames_needed),
  );
  if (s.hints.length) {
    parts.push(el("ul", {}, s.hints.map((h) => el("li", {}, state.strings["hint_" + h] || h))));
  }
  parts.push(el("div", { class: "row" }, button(t("enroll_cancel"), "plain",
    () => guarded(async () => { await api("POST", "/api/enrollment/cancel", {}); await state.refresh(); }))));
  const cards = [busy(card(t("tab_enroll"), t("tab_enroll"), parts), s.state === "collecting")];
  if (s.can_complete) {
    if (!state.form) {
      const reg = await api("GET", "/api/registry");
      if (!current()) return;
      state.form = consentForm(reg.records.filter((r) => r.kind === "person" && r.status === "pending"));
    }
    cards.push(state.form);
  } else {
    state.form = null;
  }
  box.replaceChildren(...cards);
}

function consentForm(pendingPeople) {
  const max = String(state.session.name_max_chars || 64);
  const name = textInput("display_name", { maxlength: max, required: true });
  const who = textInput("consent_who", { maxlength: max, required: true });
  const scopes = ["face", "voice", "refresh"].map((s) =>
    check(t("scope_" + s), el("input", { type: "checkbox", value: s, checked: s !== "voice" })));
  const agreed = el("input", { type: "checkbox", id: "agreed" });
  const link = el("select", { id: "pending_id", class: "sl-field__input" },
    el("option", { value: "" }, t("pending_none")),
    pendingPeople.map((r) => el("option", { value: r.id }, r.display_name)));
  const form = el("form", {
    class: "stack",
    onsubmit: (ev) => {
      ev.preventDefault();
      guarded(async () => {
        await api("POST", "/api/enrollment/complete", {
          display_name: name.value,
          consent_who: who.value,
          scope: scopes.map((l) => l.firstChild).filter((c) => c.checked).map((c) => c.value),
          agreed: agreed.checked,
          pending_id: link.value || null,
        });
        notice(t("saved"));
        await state.refresh();
      });
    },
  },
  el("div", { class: "sl-callout sl-callout--info" },
    el("div", { class: "sl-callout__body stack" }, el("p", {}, t("consent_biometric")), el("p", {}, t("consent_cloud")))),
  pendingPeople.length ? field("pending_id", t("pending_link"), link) : [],
  field("display_name", t("display_name"), name),
  field("consent_who", t("consent_who"), who),
  el("fieldset", { class: "stack" }, el("legend", { class: "sl-field__label" }, t("scope_title")), el("div", { class: "check-list" }, scopes)),
  check(t("agree"), agreed),
  el("div", { class: "row" }, submitButton(t("enroll_complete"))));
  return card(t("tab_enroll"), t("consent_title"), form);
}

// ---------------------------------------------------------------- registry

const STATUS_TONE = { pending: "warning", active: "success" };

async function registryView() {
  const current = newView();
  const view = document.getElementById("view");
  view.replaceChildren();
  const [list, quarantined] = await Promise.all([
    api("GET", "/api/registry"), api("GET", "/api/registry/quarantined"),
  ]);
  if (!current()) return;
  const rows = list.records;
  const entities = rows.filter((r) => r.kind !== "person");
  const body = rows.length === 0 ? empty(t("tab_registry"), t("registry_empty")) : table(
    [t("display_name"), "", t("last_seen"), ""],
    rows.map((r) => el("tr", {},
      el("td", {}, r.display_name),
      el("td", {}, el("span", { class: "sl-tag sl-tag--plain" }, t("kind_" + r.kind)), " ",
        badge(state.strings["status_" + r.status] || r.status, STATUS_TONE[r.status] || "info")),
      el("td", {}, r.last_seen ? when(r.last_seen) : t("never")),
      el("td", { class: "actions" }, el("div", { class: "row" }, rowActions(r, entities))))));
  view.append(card(t("tab_registry"), t("registry_title"), body));
  view.append(card(t("tab_registry"), t("quarantined_title"),
    quarantined.keys.length === 0 ? empty(t("quarantined_title"), t("quarantined_none")) :
      el("ul", {}, quarantined.keys.map((k) => el("li", {}, k + " ", button(t("purge"), "danger", () => purgeKey(k), { small: true }))))));
}

async function purgeKey(key) {
  const done = await ask({
    title: t("purge"), message: t("confirm_purge"), pin: true, danger: true, confirm: t("purge"),
    run: ({ pin }) => api("POST", "/api/registry/quarantined/purge", { key: key, confirm: true, pin: pin }),
  });
  if (done) await guardedView(registryView);
}

function rowActions(r, entities) {
  const reload = () => guardedView(registryView);
  const buttons = [button(t("rename"), "plain", async () => {
    const done = await ask({
      title: t("rename"), confirm: t("rename"),
      input: { label: t("new_name"), value: r.display_name, max: String(state.session.name_max_chars || 64) },
      run: ({ value }) => api("POST", "/api/registry/" + r.id + "/rename", { display_name: value }),
    });
    if (done) await reload();
  }, { small: true })];
  if (r.kind === "person") {
    buttons.push(button(t("forget"), "danger", async () => {
      const done = await ask({
        title: t("forget"), message: t("confirm_forget", { name: r.display_name }), pin: true, danger: true, confirm: t("forget"),
        run: ({ pin }) => api("POST", "/api/registry/" + r.id + "/forget", { confirm: true, pin: pin }),
      });
      if (done) await reload();
    }, { small: true }));
  } else {
    for (const other of entities.filter((o) => o.id !== r.id)) {
      buttons.push(button(t("merge") + " " + other.display_name, "plain", async () => {
        const done = await ask({
          title: t("merge"), message: t("confirm_merge", { drop: r.display_name, keep: other.display_name }),
          danger: true, confirm: t("merge"),
          run: () => api("POST", "/api/registry/merge", { keep_id: other.id, drop_id: r.id }),
        });
        if (done) await reload();
      }, { small: true }));
    }
    buttons.push(button(t("delete"), "danger", async () => {
      const done = await ask({
        title: t("delete"), message: t("confirm_delete", { name: r.display_name }), danger: true, confirm: t("delete"),
        run: () => api("POST", "/api/registry/" + r.id + "/delete", { confirm: true }),
      });
      if (done) await reload();
    }, { small: true }));
  }
  return buttons;
}

// ---------------------------------------------------------------- catalogue, diagnostics

async function toBase64(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function importReport(report) {
  const created = report.items.filter((i) => i.created).length;
  const parts = [el("p", {}, t("catalogue_report", { created: created, existing: report.items.length - created }))];
  if (report.photo !== "none") parts.push(el("p", { class: "hint" }, t("photo_" + report.photo)));
  if (report.truncated) parts.push(el("p", { class: "hint" }, t("catalogue_truncated")));
  if (report.conflicts.length) {
    parts.push(el("p", {}, t("catalogue_conflicts") + ":"),
      el("ul", {}, report.conflicts.map((c) => el("li", {}, c.name + ": " + (state.strings["conflict_" + c.code] || c.code)))));
  }
  return parts;
}

async function catalogueView() {
  const current = newView();
  const view = document.getElementById("view");
  view.replaceChildren();
  const data = await api("GET", "/api/catalogue");
  if (!current()) return;
  const xmp = el("input", { type: "file", id: "xmp_file", class: "sl-field__input", accept: ".xmp,application/rdf+xml,text/xml" });
  const photo = el("input", { type: "file", id: "photo_file", class: "sl-field__input", accept: "image/jpeg,image/png" });
  const result = el("div", { id: "import_result", role: "status" });
  const form = el("form", {
    class: "stack",
    onsubmit: (ev) => {
      ev.preventDefault();
      guarded(async () => {
        const file = xmp.files[0];
        if (!file) { notice(t("err_xmp_required"), true); return; }
        if (file.size > data.limits.max_xmp_bytes) { notice(t("err_xmp_too_large"), true); return; }
        const body = { xmp: await file.text() };
        if (photo.files[0]) {
          if (photo.files[0].size > data.limits.max_image_bytes) { notice(t("err_image_too_large"), true); return; }
          body.image_b64 = await toBase64(photo.files[0]);
        }
        const report = await api("POST", "/api/catalogue/import", body);
        notice("");
        await catalogueView();
        const shown = document.getElementById("import_result");
        if (shown) shown.replaceChildren(...importReport(report));
      });
    },
  },
  field("xmp_file", t("catalogue_xmp"), xmp),
  field("photo_file", t("catalogue_photo"), photo),
  el("div", { class: "row" }, submitButton(t("catalogue_import"))));

  const confirmBy = (record, entryId) => guarded(async () => {
    await api("POST", "/api/catalogue/" + record.id + "/confirm", entryId ? { entry_id: entryId } : {});
    notice(t("saved"));
    await catalogueView();
  });
  const pending = data.records.filter((r) => r.status === "pending");
  const origin = (r) => t("origin_" + r.origin) + ", " + (r.has_appearance ? t("has_appearance") : t("no_appearance"));
  const records = data.records.length === 0 ? empty(t("catalogue_pending"), t("catalogue_none")) : table(
    [t("display_name"), "", ""],
    data.records.map((r) => el("tr", {},
      el("td", {}, r.display_name),
      el("td", {}, el("span", { class: "sl-tag sl-tag--plain" }, t("kind_" + r.kind) + (r.category ? " (" + word("category_", r.category) + ")" : "")), " " + origin(r)),
      el("td", { class: "actions" }, el("div", { class: "row" },
        r.status === "pending" && r.has_appearance ?
          button(t("catalogue_activate"), "plain", () => confirmBy(r, null), { small: true }) : [],
        data.live.filter((e) => r.status === "pending" && e.kind === r.kind && (!r.category || e.category === r.category))
          .map((e) => button(t("catalogue_confirm", { name: r.display_name }) + " ← " + word("category_", e.category) + " #" + e.entry_id.split("-").pop(),
            "plain", () => confirmBy(r, e.entry_id), { small: true })))))));
  const live = data.live.length === 0 ? empty(t("catalogue_live"), t("catalogue_live_none")) :
    el("ul", {}, data.live.map((e) => el("li", {}, t("kind_" + e.kind) + " " + word("category_", e.category) + " #" + e.entry_id.split("-").pop() +
      (e.identified ? " ✓" : ""))));
  const clearCandidates = () => ask({
    title: t("catalogue_clear"), message: t("catalogue_clear_confirm"), pin: true, danger: true, confirm: t("catalogue_clear"),
    run: async ({ pin }) => {
      const out = await api("POST", "/api/catalogue/candidates/clear", { confirm: true, pin: pin });
      notice(t("catalogue_cleared", { n: out.cleared }));
    },
  });
  const clearTombstones = () => ask({
    title: t("tombstones_clear", { n: data.tombstones }), message: t("tombstones_clear_confirm"), pin: true, danger: true,
    confirm: t("tombstones_clear", { n: data.tombstones }),
    run: async ({ pin }) => {
      const out = await api("POST", "/api/catalogue/tombstones/clear", { confirm: true, pin: pin });
      notice(t("tombstones_cleared", { n: out.cleared }));
    },
  });
  view.append(
    card(t("tab_catalogue"), t("tab_catalogue"), el("p", { class: "hint" }, t("catalogue_intro")), form, result),
    card(t("tab_catalogue"), t("catalogue_pending"), pending.length ? el("p", { class: "hint" }, t("catalogue_pick")) : [], records),
    card(t("tab_catalogue"), t("catalogue_live"), live),
    card(t("tab_catalogue"), t("catalogue_clear"), el("div", { class: "row" },
      button(t("catalogue_clear"), "danger", async () => { if (await clearCandidates()) await guardedView(catalogueView); }),
      data.tombstones ? button(t("tombstones_clear", { n: data.tombstones }), "plain",
        async () => { if (await clearTombstones()) await guardedView(catalogueView); }) : [])));
}

// ---------------------------------------------------------------- secrets (write-only)

// The key is typed once: it is read from the field and the field is emptied before the request is
// sent, and the PIN likewise, so a failure never leaves either in the page. The page never holds a
// stored value: the list has names and the routes that use them.
function secretsCard(data, reload) {
  const name = textInput("secret-name", { required: true, maxlength: String(data.max_name_chars), autocomplete: "off", autocapitalize: "off", spellcheck: "false" });
  const value = textInput("secret-value", {
    type: "password", required: true, autocomplete: "off", autocapitalize: "off", spellcheck: "false",
    maxlength: String(data.max_value_chars),
  });
  const pin = textInput("secret-pin", { type: "password", autocomplete: "off", inputmode: "numeric", required: true, maxlength: "64" });
  const form = el("form", {
    class: "stack", autocomplete: "off",
    onsubmit: (ev) => {
      ev.preventDefault();
      const body = { name: name.value.trim(), value: value.value, pin: pin.value };
      value.value = "";
      pin.value = "";
      guarded(async () => {
        try {
          await api("POST", "/api/secrets/set", body);
        } finally {
          body.value = "";
          body.pin = "";
        }
        name.value = "";
        notice(t("secret_saved", { name: body.name }));
        await reload();
      });
    },
  },
  field("secret-name", t("secret_name"), name, t("secret_name_hint")),
  field("secret-value", t("secret_value"), value, t("secret_value_hint")),
  field("secret-pin", t("pin"), pin),
  el("div", { class: "row" }, submitButton(t("secret_save"))));
  const remove = (s) => guarded(async () => {
    const done = await ask({
      title: t("secret_remove_title", { name: s.name }), message: t("secret_remove_message"),
      pin: true, danger: true, confirm: t("secret_remove"),
      run: async ({ pin: entered }) => { await api("POST", "/api/secrets/remove", { name: s.name, pin: entered }); },
    });
    if (done) { notice(t("secret_removed", { name: s.name })); await reload(); }
  });
  const rows = data.secrets.map((s) => el("tr", {},
    el("td", {}, s.name),
    el("td", {}, badge(t(s.set ? "secret_set" : "secret_unset"), s.set ? "success" : "warning")),
    el("td", {}, s.used_by.length ? t("secret_used_by", { routes: s.used_by.join(", ") }) : t("secret_unused")),
    el("td", {}, button(t("secret_remove"), "danger", () => remove(s), { small: true, attrs: { disabled: s.used_by.length > 0, title: s.used_by.length ? t("secret_in_use_hint") : null } }))));
  return card(t("tab_routes"), t("secrets_title"), el("p", { class: "hint" }, t("secrets_intro")),
    rows.length ? table([t("secret_name"), "", "", ""], rows) : el("p", { class: "hint" }, t("secrets_none")),
    form);
}

// ---------------------------------------------------------------- route form (add and edit)

const ROUTE_LOCALITIES = ["auto", "local", "cloud"];
const ROUTE_KINDS = ["chat", "realtime"]; // chosen when a route is added; shown read-only after
const ROUTE_KEY_LITERAL = "@literal"; // not a valid secret name: a key written in the config file

// One form for a new route (`route` null) and for an existing one. A key is chosen from the stored
// names, never typed. `save(values, form)` runs the PIN dialog and the calls.
function routeForm(route, secrets, nameMax, save) {
  const id = (k) => "route-" + (route ? route.name : "new") + "-" + k;
  const name = route ? null : textInput(id("name"), { required: true, maxlength: String(nameMax), autocomplete: "off", spellcheck: "false" });
  const url = textInput(id("url"), { type: "url", required: !route, autocomplete: "off", spellcheck: "false", placeholder: t(route ? "route_url_keep" : "route_url_placeholder") });
  const model = textInput(id("model"), { required: true, autocomplete: "off", spellcheck: "false", value: route ? route.model : "" });
  const locality = el("select", { id: id("locality"), class: "sl-field__input" },
    ROUTE_LOCALITIES.map((l) => el("option", { value: l }, t("route_locality_" + l))));
  locality.value = route ? route.locality : "cloud";
  const kind = route ? null : el("select", { id: id("kind"), class: "sl-field__input" },
    ROUTE_KINDS.map((k) => el("option", { value: k }, t("route_kind_" + k))));
  if (kind) kind.value = "chat";
  const current = !route || route.key === "none" ? "" : (route.key === "secret" ? route.secret.name : ROUTE_KEY_LITERAL);
  const names = secrets.filter((s) => s.set).map((s) => s.name);
  const keys = [["", t("route_key_none")]];
  if (current === ROUTE_KEY_LITERAL) keys.push([ROUTE_KEY_LITERAL, t("route_key_literal")]);
  else if (current && !names.includes(current)) keys.push([current, current + " (" + t("secret_unset") + ")"]);
  for (const n of names) keys.push([n, n]);
  const key = el("select", { id: id("key"), class: "sl-field__input" }, keys.map(([v, label]) => el("option", { value: v }, label)));
  key.value = current;
  const form = el("form", {
    class: "stack", autocomplete: "off",
    onsubmit: (ev) => {
      ev.preventDefault();
      guarded(() => save({
        name: name ? name.value.trim() : route.name, base_url: url.value.trim(), model: model.value.trim(),
        locality: locality.value, key: key.value, ...(kind ? { kind: kind.value } : {}),
      }));
    },
  },
  name ? field(id("name"), t("route_name"), name) : [],
  field(id("url"), t("route_url"), url, route ? t("route_url_hint_edit") : null),
  field(id("model"), t("route_model"), model),
  field(id("locality"), t("route_locality"), locality, t("route_locality_hint")),
  kind ? field(id("kind"), t("route_kind"), kind, t("route_kind_hint")) : [],
  field(id("key"), t("route_key"), key, t("route_key_hint")),
  el("div", { class: "row" }, submitButton(t(route ? "route_save" : "route_add"))));
  return form;
}

// A key sent over http to a host that is not this machine crosses the network unencrypted.
function insecureHost(origin) {
  const m = /^http:[/][/](\[[^\]]*\]|[^:/]*)/.exec(origin);
  return Boolean(m) && !(m[1] === "localhost" || m[1] === "[::1]" || /^127\./.test(m[1]));
}

// Calls for a new route or for the changed fields of an existing one, behind one PIN dialog. A route
// with a key sends it to the host of its URL: the dialog names that host and the call carries it as
// the acknowledgement (the server names the one it expects when it differs).
// An edit is several calls (R3 SF4), ordered so a key never reaches a host the owner did not mean:
// a locality change to cloud first (stricter; a local route refuses a new URL), then a key that
// changes is unbound when it goes away or the route moves, then the URL moves (with no key bound,
// no acknowledgement), then the new key is bound to the new host, then the rest. The
// first failure stops the rest; the error names what was already saved, and a retry resumes there.
async function saveRoute(route, v) {
  let ack = null;
  const origin = routeOrigin(v.base_url || (route && route.base_url) || "");
  const keyed = v.key !== "" && v.key !== ROUTE_KEY_LITERAL;
  const current = !route || route.key === "none" ? "" : (route.key === "secret" ? route.secret.name : ROUTE_KEY_LITERAL);
  const keyChanged = route && v.key !== current;
  const urlChanged = route && v.base_url && v.base_url !== (route.base_url || "");
  const moved = urlChanged && routeOrigin(v.base_url) !== routeOrigin(route.base_url || "");
  const unbind = keyChanged && current !== "" && (!keyed || moved);
  const target = origin || ack;
  const steps = []; // [label, call(pin)]
  const mapAck = async (call) => {
    try {
      return await call();
    } catch (err) {
      if (err.message === "endpoint_ack_required" && err.endpoint) {
        ack = err.endpoint;
        const text = document.querySelector("dialog .sl-dialog__body p");
        if (text) text.textContent = ackMessage(ack);
      }
      throw err;
    }
  };
  const bindKey = (secret) => (pin) => mapAck(() => api("POST", "/api/routes/key", {
    name: route.name, secret, pin,
    ...(secret !== null && (ack || origin) ? { ack_endpoint: ack || origin } : {}),
  }));
  if (!route) {
    steps.push([v.name, (pin) => mapAck(() => api("POST", "/api/routes/add", {
      name: v.name, base_url: v.base_url, model: v.model, locality: v.locality, kind: v.kind || "chat",
      secret: keyed ? v.key : null, pin,
      ...(keyed && (ack || origin) ? { ack_endpoint: ack || origin } : {}),
    }))]);
  } else {
    const keyAtMove = current !== "" && !unbind; // the key still bound when the URL moves
    const path = (f) => "llm_routes." + route.name + "." + f;
    const set = (f, value) => (pin) => mapAck(() => api("POST", "/api/settings", { path: path(f), value, pin,
      ...(f === "base_url" && keyAtMove && (ack || routeOrigin(value)) ? { ack_endpoint: ack || routeOrigin(value) } : {}) }));
    // To cloud is stricter and goes first: a local route refuses a new URL while it is local.
    const toCloud = v.locality !== route.locality && v.locality === "cloud";
    if (toCloud) steps.push([t("route_locality"), set("locality", v.locality)]);
    if (unbind) steps.push([t("route_key"), bindKey(null)]);
    if (urlChanged) steps.push([t("route_url"), set("base_url", v.base_url)]);
    if (keyChanged && keyed) steps.push([t("route_key"), bindKey(v.key)]);
    if (v.locality !== route.locality && !toCloud) steps.push([t("route_locality"), set("locality", v.locality)]);
    if (v.model !== route.model) steps.push([t("route_model"), set("model", v.model)]);
    if (!steps.length) { notice(t("settings_unchanged")); return; }
  }
  const sends = keyed || (moved && current !== "" && !keyChanged);
  const realtime = (route ? route.kind : v.kind) === "realtime";
  const extraHosts = realtime ? (state.realtimeKeyHosts || []) : [];
  const ackMessage = (host) => t("route_key_message", { host }) + (insecureHost(host) ? " " + t("route_key_insecure") : "") +
    (extraHosts.length ? " " + t("route_key_hosts", { hosts: extraHosts.join(", ") }) : "");
  const message = sends && target ? ackMessage(target) : (route ? route.name : v.name);
  let next = 0; // steps before this index are saved: a retry of the dialog resumes here
  let notApplied = false;
  let ran = false;
  await pinnedCall(message, t(route ? "route_save" : "route_add"), async (pin) => {
    for (; next < steps.length; next++) {
      try {
        const result = await steps[next][1](pin);
        if (result && result.applied === false) notApplied = true;
      } catch (err) {
        if (route && next > 0) {
          const saved = [...new Set(steps.slice(0, next).map(([label]) => label))].join(", ");
          err.saved = t("route_partial", { items: saved });
        }
        throw err;
      }
    }
    ran = true;
    return {};
  });
  if (ran) {
    const done = t(route ? "route_saved" : "route_added", { name: route ? route.name : v.name });
    notice(notApplied ? done + " " + t("settings_not_applied") : done, notApplied);
  }
  await routesView();
}

const REALTIME_FACTS = [
  "mode", "allocator_host", "connect_host", "header_to_connect_host", "platform_id_sent",
  "update_seen", "first_audio_ms", "item_delete", "charged", "failure",
  "server_error_code", "server_error_about",
];

const REALTIME_WORDS = { mode: "rt_mode_", header_to_connect_host: "rt_header_", item_delete: "rt_item_", failure: "live_code_" };

// One fact of a realtime check: yes / no, a worded code, or a host or server code as it is.
function realtimeFact(name, value) {
  if (value === null || value === undefined) return t("live_none");
  if (typeof value === "boolean") return t(value ? "live_yes" : "live_no");
  return REALTIME_WORDS[name] ? word(REALTIME_WORDS[name], value) : String(value);
}

// A check's reason in the page language from its `code` and `params`. A report kept from before
// the codes has none: only its status shows (its English reason is not). `why` and `reasoned`
// add a clause.
function checkReason(c) {
  if (!c.code) return "";
  const p = Object.assign({}, c.params);
  if (p.failure !== undefined) p.failure = word("live_code_", p.failure);
  const parts = [t("check_reason_" + c.code, p)];
  if (p.why) parts.push(t("check_why_" + p.why, { detail: p.detail || "" }));
  if (p.withheld) parts.push(t("check_withheld"));
  if (p.reasoned) parts.push(t("check_reasoned", { n: p.reasoned }));
  return parts.join("; ");
}

async function routesView() {
  const current = newView();
  const view = document.getElementById("view");
  const data = await api("GET", "/api/routes");
  if (!current()) return;
  state.realtimeKeyHosts = data.realtime_key_hosts || [];
  let secrets = null;
  let secretsError = null;
  try {
    secrets = await api("GET", "/api/secrets");
  } catch (err) {
    secretsError = err;
  }
  if (!current()) return;
  if (secretsError) {
    if (secretsError.message === "secret_store") notice(explain(secretsError), true);
    else if (secretsError.message !== "secrets_unavailable") throw secretsError;
  }
  const secretList = secrets ? secrets.secrets : [];
  view.replaceChildren();
  const verdict = (c) => el("li", {}, t("check_" + c.name) + ": ", badge(t("status_" + c.status), c.status === "pass" ? "success" : (c.status === "fail" ? "danger" : "info")), " " + checkReason(c));
  // A realtime check names hosts and outcome codes only (SPEC 8, 12): codes are worded, hosts
  // and the server's own error codes are shown as they come.
  const realtimeFacts = (report) => report.kind !== "realtime" ? [] :
    [el("ul", {}, REALTIME_FACTS.map((f) => el("li", {}, t("rt_" + f) + ": " + realtimeFact(f, report[f]))))];
  const show = (box, report) => {
    const head = report.error ? t("route_error_" + report.error) : t(report.ok ? "route_ok" : "route_failed");
    box.replaceChildren(
      el("p", {}, head + " " + t("route_at", { at: when(report.checked_at) })),
      el("ul", {}, report.checks.map(verdict)),
      ...realtimeFacts(report));
  };
  const cards = data.routes.map((r) => {
    const box = el("div", { role: "status" });
    if (r.running) box.append(el("p", { class: "hint" }, t("route_running")));
    else if (r.last_check) show(box, r.last_check);
    else box.append(el("p", { class: "hint" }, t("route_never")));
    const run = button(t("route_check"), "plain", () => guarded(async () => {
      await api("POST", "/api/routes/" + encodeURIComponent(r.name) + "/check", {});
      await routesView();
    }), { attrs: { disabled: r.running } });
    const edit = el("div", { hidden: true }, routeForm(r, secretList, data.name_max_chars, (v) => saveRoute(r, v)));
    const toggle = button(t("route_edit"), "plain", () => { edit.hidden = !edit.hidden; });
    const remove = button(t("route_remove"), "danger", () => guarded(async () => {
      const done = await ask({
        title: t("route_remove_title", { name: r.name }), message: t("route_remove_message"), pin: true, danger: true,
        confirm: t("route_remove"),
        run: async ({ pin }) => { await api("POST", "/api/routes/remove", { name: r.name, pin }); },
      });
      if (done) { notice(t("route_removed", { name: r.name })); await routesView(); }
    }));
    return busy(card(t("tab_routes"), r.name,
      el("p", { class: "hint" }, r.model + " @ " + r.endpoint + ", " + t("route_locality_" + r.locality) + ", " + t("route_kind_" + (r.kind || "chat"))),
      r.secret ? el("p", { class: "hint" }, t(r.secret.set ? "route_secret_set" : "route_secret_missing", { name: r.secret.name })) :
        (r.key === "literal" ? el("p", { class: "hint" }, t("route_key_literal")) : []),
      el("div", { class: "row" }, run, toggle, r.removable ? remove : []), box, edit), r.running);
  });
  const addCard = card(t("tab_routes"), t("route_add_title"), el("p", { class: "hint" }, t("route_add_intro")),
    routeForm(null, secretList, data.name_max_chars, (v) => saveRoute(null, v)));
  // Names kept as typed (admin PIN, one request): shown here only until the view is redrawn.
  const refusedBox = el("div", { role: "status" });
  const showRefused = async () => {
    let out = null;
    const done = await ask({
      title: t("refused_title"), message: t("refused_hint"), pin: true, confirm: t("refused_show"),
      run: async ({ pin }) => { out = await api("POST", "/api/pseudonyms/refused", { pin: pin }); },
    });
    if (!done || !out) return;
    refusedBox.replaceChildren(!out.enabled ? el("p", { class: "hint" }, t("refused_off")) :
      (out.names.length ? el("ul", {}, out.names.map((n) => el("li", {}, n))) : el("p", { class: "hint" }, t("refused_none"))));
  };
  view.append(card(t("tab_routes"), t("tab_routes"), el("p", { class: "hint" }, t("routes_intro"))),
    ...(cards.length ? cards : [empty(t("tab_routes"), t("routes_none"))]),
    addCard,
    ...(secrets ? [secretsCard(secrets, routesView)] : []),
    card(t("tab_routes"), t("refused_title"), el("p", { class: "hint" }, t("refused_hint")),
      el("div", { class: "row" }, button(t("refused_show"), "plain", () => guarded(showRefused))), refusedBox));
  if (data.routes.some((r) => r.running)) {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => guardedView(refreshRoutes), ROUTES_POLL_MS);
  }
}

// The poll redraws the page; it waits while a form field has the focus so typing is not lost.
async function refreshRoutes() {
  const focus = document.activeElement;
  const typing = focus && document.getElementById("view").contains(focus) && ["INPUT", "SELECT", "TEXTAREA"].includes(focus.tagName);
  if (!typing) { await routesView(); return; }
  if (state.tab !== "routes") return;
  clearTimeout(state.timer);
  state.timer = setTimeout(() => guardedView(refreshRoutes), ROUTES_POLL_MS);
}

async function debugView() {
  const current = newView();
  const view = document.getElementById("view");
  view.replaceChildren();
  const report = await api("GET", "/api/debug");
  if (!current()) return;
  const sections = report.sections || {};
  const robotOnly = Object.fromEntries(Object.entries(sections).filter(([, s]) => s && s.scope === "robot"));
  const coreOnly = Object.fromEntries(Object.entries(sections).filter(([, s]) => !(s && s.scope === "robot")));
  const block = (cmd, data) => el("article", { class: "sl-term-block" },
    el("header", { class: "sl-term-block__head" },
      el("span", { class: "sl-term-block__cmd" }, el("span", { class: "sl-term-block__prompt" }, "$"), " " + cmd)),
    el("pre", { class: "sl-term-block__out pre-scroll" }, JSON.stringify(data, null, 2)));
  const hasRobot = Object.keys(robotOnly).length > 0;
  view.append(card(t("tab_debug"), t("debug_title"),
    el("div", { class: "row" }, button(t("debug_refresh"), "plain", () => guardedView(debugView))),
    markedEyebrow("h3", t("settings_core")),
    block("GET /api/debug", hasRobot ? { ...report, sections: coreOnly } : report),
    hasRobot ? [markedEyebrow("h3", t("settings_robot")), block("GET /api/debug (robot)", robotOnly)] : []));
}

// ---------------------------------------------------------------- shell

const TABS = {
  enroll: enrollmentView,
  registry: registryView,
  catalogue: catalogueView,
  routes: routesView,
  settings: settingsView,
  live: liveView,
  emotions: emotionsView,
  debug: debugView,
};

// While a tab waits for its first answer: #view is busy (aria-busy, what assistive technology
// reads) and a muted line says so on screen.
function viewLoading(on) {
  const line = document.getElementById("view-loading");
  if (line.hidden === !on) return;
  line.textContent = on ? t("loading") : "";
  line.hidden = !on;
  const view = document.getElementById("view");
  if (on) view.setAttribute("aria-busy", "true");
  else view.removeAttribute("aria-busy");
}

// The tab's view starts on an empty #view. A view that draws before its first await shows no
// loading line; the others show it until they put something in #view (the observer below) or
// settle, unless a later view took #view meanwhile. Resolves once the view settled.
function showTab(name) {
  stopPolling();
  newView();
  state.tab = name;
  const view = document.getElementById("view");
  view.setAttribute("aria-labelledby", "tab-" + name);
  view.replaceChildren();
  notice("");
  for (const b of tabButtons()) {
    const on = b.dataset.tab === name;
    b.setAttribute("aria-selected", String(on));
    b.classList.toggle("sl-fx-aurora", on);
    b.classList.toggle("sl-fx-aurora--linear", on);
  }
  const tab = selectedTab();
  tabRove(tab);
  // On a narrow screen the strip scrolls sideways: the open tab stays in sight.
  if (tab && tab.scrollIntoView) tab.scrollIntoView({ inline: "nearest", block: "nearest" });
  history.replaceState(null, "", "#" + name);
  document.title = t("tab_" + name) + " · " + t("title");
  const done = guardedView(TABS[name]);
  const current = viewToken();
  viewLoading(!view.hasChildNodes());
  done.then(() => { if (current()) viewLoading(false); });
  return done;
}

const DEFAULT_TAB = "enroll";
const tabButtons = () => [...document.querySelectorAll("#tabs [role=tab]")];
const selectedTab = () => document.querySelector('#tabs [aria-selected="true"]');

// The tab a fragment names ("#live"), or null for one that names none.
function tabOf(hash) {
  const name = String(hash || "").replace(/^#/, "");
  return Object.hasOwn(TABS, name) ? name : null;
}

// Roving tabindex: `target` is the one tab the Tab key reaches.
function tabRove(target) {
  for (const b of tabButtons()) b.tabIndex = b === target ? 0 : -1;
}

// Manual activation (WAI-ARIA tabs pattern): arrows, Home and End move the focus only, so moving
// across Live does not start its polls on the robot. A click opens the focused tab; Enter and
// Space click it, as on any button.
function tabKeys(ev) {
  const tabs = tabButtons();
  const at = tabs.indexOf(document.activeElement);
  const from = at >= 0 ? at : tabs.indexOf(selectedTab());
  const to = { ArrowRight: from + 1, ArrowLeft: from - 1, Home: 0, End: tabs.length - 1 }[ev.key];
  if (to === undefined) return;
  ev.preventDefault();
  const next = tabs[(to + tabs.length) % tabs.length];
  tabRove(next);
  next.focus();
}

// Focus that leaves the strip puts its tab stop back on the open tab, where Tab enters it again.
function tabLeave(ev) {
  if (!ev.currentTarget.contains(ev.relatedTarget)) tabRove(selectedTab());
}
document.getElementById("tabs").addEventListener("focusout", tabLeave);

// A bookmark or a fragment typed by hand opens its tab; nothing while logged out.
window.addEventListener("hashchange", () => {
  const name = tabOf(location.hash);
  if (state.csrf === null || document.getElementById("tabs").hidden || !name || name === state.tab) return;
  showTab(name);
});

function themeControl() {
  const box = document.getElementById("theme");
  const theme = window.roboemotTheme;
  if (!theme) { box.hidden = true; return; }
  box.setAttribute("aria-label", t("theme"));
  const paint = () => {
    for (const b of box.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.mode === theme.mode()));
  };
  box.replaceChildren(...theme.modes.map((mode) =>
    el("button", { type: "button", class: "sl-segmented__item", "data-mode": mode, onclick: () => { theme.set(mode); paint(); } },
      t("theme_" + mode.replace("-", "_")))));
  paint();
}

function shell() {
  const tabs = document.getElementById("tabs");
  tabs.hidden = false;
  tabs.onkeydown = tabKeys;
  tabs.setAttribute("aria-label", t("tabs_label"));
  const asked = location.hash.replace(/^#/, "");
  if (asked) state.tab = tabOf(asked) || DEFAULT_TAB;
  tabs.replaceChildren(...Object.keys(TABS).map((name) =>
    el("button", { type: "button", role: "tab", id: "tab-" + name, "aria-controls": "view", class: "sl-tabs__tab", "data-tab": name, onclick: () => showTab(name) }, t("tab_" + name))));
  const out = document.getElementById("logout");
  out.hidden = false;
  out.textContent = t("logout");
  out.onclick = () => guarded(async () => {
    resetSettingsState();
    await api("POST", "/api/logout", {});
    state.csrf = null;
    await boot();
  });
  liveBannerStart();
  return showTab(state.tab);
}

// Takes the page strings of GET /api/strings: the text, the `lang` of the document, the title and
// the theme control, which is drawn outside the views.
function applyStrings(strings) {
  state.strings = strings.strings;
  state.lang = strings.language;
  document.documentElement.lang = strings.language;
  document.title = t("title");
  document.getElementById("title").textContent = t("title");
  themeControl();
}

// The page language changed (a saved `conversation.language`): loads its strings and redraws the
// shell on the current tab, so no text stays in the old language. Resolves once the tab has drawn.
async function relanguage() {
  applyStrings(await api("GET", "/api/strings"));
  return shell();
}

async function boot() {
  stopPolling();
  liveBannerStop();
  const [strings, session] = await Promise.all([api("GET", "/api/strings"), api("GET", "/api/session")]);
  state.session = session;
  applyStrings(strings);
  if (session.authenticated) {
    state.csrf = session.csrf;
    shell();
  } else {
    state.csrf = null;
    resetSettingsState();
    document.getElementById("tabs").hidden = true;
    document.getElementById("logout").hidden = true;
    loginView();
  }
}

// Views render asynchronously and in parts, so the glyphs follow the DOM rather than the code
// that builds it. Mutation callbacks run before the next paint: a head never shows the wrong one.
const viewRoot = document.getElementById("view");
new MutationObserver(() => assignMarks(viewRoot)).observe(viewRoot, { childList: true, subtree: true });
new MutationObserver(() => { if (viewRoot.hasChildNodes()) viewLoading(false); }).observe(viewRoot, { childList: true });

// Loads the page; a failure says the robot is out of reach and offers to try again.
function start() {
  boot().catch(bootFailed);
}

// The strings may be missing here: the text comes from PAGE_HELD.
function bootFailed() {
  notice(held("err_unreachable"), true);
  const view = document.getElementById("view");
  view.replaceChildren(el("div", { class: "row" }, button(held("retry"), "primary", () => {
    view.replaceChildren();
    notice("");
    start();
  }, { attrs: { id: "boot-retry" } })));
}

start();
