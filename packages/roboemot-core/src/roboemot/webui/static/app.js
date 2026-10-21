"use strict";
// RoboEmot web UI on the vendored Sondalab kit. All text goes through textContent: names from the
// registry are never HTML. Styles are set through the CSSOM only (the CSP forbids style attributes).

const ROUTES_POLL_MS = 2000;
const state = { strings: {}, csrf: null, session: {}, tab: "enroll", timer: null, form: null };

const t = (key, vars) => {
  let text = state.strings[key] || key;
  for (const [k, v] of Object.entries(vars || {})) text = text.replace("{" + k + "}", () => String(v));
  return text;
};

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

function card(eyebrow, title, ...children) {
  return el("section", { class: "sl-card" },
    eyebrow ? el("p", { class: "sl-eyebrow" }, eyebrow) : [],
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
  return el("div", { class: "sl-empty" }, el("span", { class: "sl-eyebrow" }, eyebrow), el("p", { class: "sl-empty__body" }, body));
}

function table(head, rows) {
  return el("div", { class: "table-scroll" }, el("table", { class: "sl-table" },
    el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))),
    el("tbody", {}, rows)));
}

function notice(message, isError) {
  const n = document.getElementById("notice");
  n.setAttribute("aria-live", isError ? "assertive" : "polite");
  n.hidden = !message;
  n.className = message ? "sl-callout sl-callout--" + (isError ? "danger" : "success") : "";
  n.replaceChildren(...(message ? [el("div", { class: "sl-callout__body" }, message)] : []));
}

// ---------------------------------------------------------------- api

async function api(method, path, body) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET" && state.csrf) headers["X-CSRF-Token"] = state.csrf;
  const resp = await fetch(path, {
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
    err.detail = data.detail;
    err.endpoint = data.endpoint;
    err.retryAfter = Number(resp.headers.get("Retry-After"));
    throw err;
  }
  return data;
}

function explain(err) {
  const known = state.strings["err_" + err.message];
  let text = known || t("err_generic");
  if (err.message === "consent_refused" && err.detail) text += " " + err.detail;
  if (err.endpoint) text = text.replace("{host}", () => err.endpoint);
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

function stopPolling() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
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

function loginView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  if (!state.session.pin_set) {
    view.append(card(null, t("login_title"), el("p", {}, t("no_pin"))));
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
      });
    },
  }, field("pin", t("pin"), pin), el("div", { class: "row" }, submitButton(t("login"))));
  view.append(card(null, t("login_title"), form));
  pin.focus();
}

// ---------------------------------------------------------------- enrollment

function enrollmentView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const box = el("div", { class: "stack" });
  view.append(box);
  const poll = async () => {
    stopPolling();
    try {
      const s = await api("GET", "/api/enrollment/status");
      await renderEnrollment(box, s);
      if (state.tab === "enroll" && ["collecting", "ready"].includes(s.state)) {
        state.timer = setTimeout(poll, state.session.status_poll_ms || 1000);
      }
    } catch (err) {
      if (err.status === 401) { await boot(); return; }
      notice(explain(err), true);
    }
  };
  state.refresh = poll;
  poll();
}

async function renderEnrollment(box, s) {
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
      state.form = consentForm(reg.records.filter((r) => r.kind === "person" && r.status === "pending"));
    }
    cards.push(state.form);
  } else {
    state.form = null;
  }
  box.replaceChildren(...cards);
}

function selectInput(id, values, prefix) {
  return el("select", { id, class: "sl-field__input" }, values.map((v) => el("option", { value: v }, t(prefix + v))));
}

function consentForm(pendingPeople) {
  const max = String(state.session.name_max_chars || 64);
  const name = textInput("display_name", { maxlength: max, required: true });
  const who = textInput("consent_who", { maxlength: max, required: true });
  const guardian = textInput("guardian", { maxlength: max });
  const age = selectInput("age_group", ["unknown", "child", "teen", "adult", "senior"], "age_");
  const gender = selectInput("gender", ["unknown", "female", "male"], "gender_");
  const minor = () => !["adult", "senior"].includes(age.value);
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
          consent_who: minor() ? guardian.value : who.value,
          guardian: guardian.value,
          age_group: age.value,
          gender: gender.value,
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
  field("age_group", t("age_group"), age),
  field("gender", t("gender"), gender),
  field("guardian", t("guardian"), guardian, t("guardian_hint")),
  el("fieldset", { class: "stack" }, el("legend", { class: "sl-field__label" }, t("scope_title")), el("div", { class: "check-list" }, scopes)),
  check(t("agree"), agreed),
  el("div", { class: "row" }, submitButton(t("enroll_complete"))));
  return card(t("tab_enroll"), t("consent_title"), form);
}

// ---------------------------------------------------------------- registry

const STATUS_TONE = { pending: "warning", active: "success" };

async function registryView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const [list, quarantined] = await Promise.all([
    api("GET", "/api/registry"), api("GET", "/api/registry/quarantined"),
  ]);
  const rows = list.records;
  const entities = rows.filter((r) => r.kind !== "person");
  const body = rows.length === 0 ? empty(t("tab_registry"), t("registry_empty")) : table(
    [t("display_name"), "", t("last_seen"), ""],
    rows.map((r) => el("tr", {},
      el("td", {}, r.display_name),
      el("td", {}, el("span", { class: "sl-tag sl-tag--plain" }, t("kind_" + r.kind)), " ",
        badge(state.strings["status_" + r.status] || r.status, STATUS_TONE[r.status] || "info")),
      el("td", {}, r.last_seen ? new Date(r.last_seen).toLocaleString() : t("never")),
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
  if (done) await guarded(registryView);
}

function rowActions(r, entities) {
  const reload = () => guarded(registryView);
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
  const view = document.getElementById("view");
  view.replaceChildren();
  const data = await api("GET", "/api/catalogue");
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
        document.getElementById("import_result").replaceChildren(...importReport(report));
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
      el("td", {}, el("span", { class: "sl-tag sl-tag--plain" }, t("kind_" + r.kind) + (r.category ? " (" + r.category + ")" : "")), " " + origin(r)),
      el("td", { class: "actions" }, el("div", { class: "row" },
        r.status === "pending" && r.has_appearance ?
          button(t("catalogue_activate"), "plain", () => confirmBy(r, null), { small: true }) : [],
        data.live.filter((e) => r.status === "pending" && e.kind === r.kind && (!r.category || e.category === r.category))
          .map((e) => button(t("catalogue_confirm", { name: r.display_name }) + " ← " + e.category + " #" + e.entry_id.split("-").pop(),
            "plain", () => confirmBy(r, e.entry_id), { small: true })))))));
  const live = data.live.length === 0 ? empty(t("catalogue_live"), t("catalogue_live_none")) :
    el("ul", {}, data.live.map((e) => el("li", {}, t("kind_" + e.kind) + " " + e.category + " #" + e.entry_id.split("-").pop() +
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
      button(t("catalogue_clear"), "danger", async () => { if (await clearCandidates()) await guarded(catalogueView); }),
      data.tombstones ? button(t("tombstones_clear", { n: data.tombstones }), "plain",
        async () => { if (await clearTombstones()) await guarded(catalogueView); }) : [])));
}

// ---------------------------------------------------------------- secrets (write-only)

// The key is typed once: it is read from the field and the field is emptied before the request is
// sent, and the PIN likewise, so a failure never leaves either in the page. The page never holds a
// stored value: the list has names and the routes that use them.
function secretsCard(data, reload) {
  const name = textInput("secret-name", { required: true, maxlength: "128", autocomplete: "off", autocapitalize: "off", spellcheck: "false" });
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
const ROUTE_KEY_LITERAL = "@literal"; // not a valid secret name: a key written in the config file

// One form for a new route (`route` null) and for an existing one. A key is chosen from the stored
// names, never typed. `canClear`: the route may be cleared for minors (a route added from this page
// cannot: the server refuses that loosening). `save(values, form)` runs the PIN dialog and the calls.
function routeForm(route, secrets, nameMax, save) {
  const id = (k) => "route-" + (route ? route.name : "new") + "-" + k;
  const name = route ? null : textInput(id("name"), { required: true, maxlength: String(nameMax), autocomplete: "off", spellcheck: "false" });
  const url = textInput(id("url"), { type: "url", required: !route, autocomplete: "off", spellcheck: "false", placeholder: t(route ? "route_url_keep" : "route_url_placeholder") });
  const model = textInput(id("model"), { required: true, autocomplete: "off", spellcheck: "false", value: route ? route.model : "" });
  const locality = el("select", { id: id("locality"), class: "sl-field__input" },
    ROUTE_LOCALITIES.map((l) => el("option", { value: l }, t("route_locality_" + l))));
  locality.value = route ? route.locality : "cloud";
  const clearable = route && !route.removable;
  const minors = el("input", { type: "checkbox", id: id("minors"), disabled: !clearable });
  minors.checked = Boolean(route && route.allowed_for_minors);
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
        locality: locality.value, allowed_for_minors: minors.checked, key: key.value,
      }));
    },
  },
  name ? field(id("name"), t("route_name"), name) : [],
  field(id("url"), t("route_url"), url, route ? t("route_url_hint_edit") : null),
  field(id("model"), t("route_model"), model),
  field(id("locality"), t("route_locality"), locality, t("route_locality_hint")),
  check(t("route_minors_label"), minors),
  el("p", { class: "hint" }, t(clearable ? "route_minors_hint_edit" : "route_minors_hint")),
  field(id("key"), t("route_key"), key, t("route_key_hint")),
  el("div", { class: "row" }, submitButton(t(route ? "route_save" : "route_add"))));
  return form;
}

// Calls for a new route or for the changed fields of an existing one, behind one PIN dialog. A route
// with a key sends it to the host of its URL: the dialog names that host and the call carries it as
// the acknowledgement (the server names the one it expects when it differs).
async function saveRoute(route, v) {
  let ack = null;
  const origin = routeOrigin(v.base_url || (route && route.base_url) || "");
  const keyed = v.key !== "" && v.key !== ROUTE_KEY_LITERAL;
  const urlChanged = route && v.base_url && v.base_url !== (route.base_url || "");
  const moved = urlChanged && routeOrigin(v.base_url) !== routeOrigin(route.base_url || "");
  const target = origin || ack;
  const steps = [];
  const mapAck = async (call) => {
    try {
      return await call();
    } catch (err) {
      if (err.message === "endpoint_ack_required" && err.endpoint) ack = err.endpoint;
      throw err;
    }
  };
  if (!route) {
    steps.push((pin) => mapAck(() => api("POST", "/api/routes/add", {
      name: v.name, base_url: v.base_url, model: v.model, locality: v.locality,
      allowed_for_minors: v.allowed_for_minors, secret: keyed ? v.key : null, pin,
      ...(keyed && (ack || origin) ? { ack_endpoint: ack || origin } : {}),
    })));
  } else {
    const path = (f) => "llm_routes." + route.name + "." + f;
    const set = (f, value) => (pin) => mapAck(() => api("POST", "/api/settings", { path: path(f), value, pin,
      ...(f === "base_url" && route.key !== "none" && (ack || routeOrigin(value)) ? { ack_endpoint: ack || routeOrigin(value) } : {}) }));
    if (urlChanged) steps.push(set("base_url", v.base_url));
    if (v.key !== (route.key === "none" ? "" : (route.key === "secret" ? route.secret.name : ROUTE_KEY_LITERAL))) {
      steps.push((pin) => mapAck(() => api("POST", "/api/routes/key", {
        name: route.name, secret: keyed ? v.key : null, pin,
        ...(keyed && (ack || origin) ? { ack_endpoint: ack || origin } : {}),
      })));
    }
    if (v.locality !== route.locality) steps.push(set("locality", v.locality));
    if (!route.removable && v.allowed_for_minors !== route.allowed_for_minors) steps.push(set("allowed_for_minors", v.allowed_for_minors));
    if (v.model !== route.model) steps.push(set("model", v.model));
    if (!steps.length) { notice(t("settings_unchanged")); return; }
  }
  const sends = keyed || (moved && route.key !== "none");
  const message = sends && target ? t("route_key_message", { host: target }) : (route ? route.name : v.name);
  let ran = false;
  await pinnedCall(message, t(route ? "route_save" : "route_add"), async (pin) => {
    for (const step of steps) await step(pin);
    ran = true;
    return {};
  });
  if (ran) notice(t(route ? "route_saved" : "route_added", { name: route ? route.name : v.name }));
  await routesView();
}

async function routesView() {
  const view = document.getElementById("view");
  const data = await api("GET", "/api/routes");
  let secrets = null;
  try {
    secrets = await api("GET", "/api/secrets");
  } catch (err) {
    if (err.message === "secret_store") notice(explain(err), true);
    else if (err.message !== "secrets_unavailable") throw err;
  }
  const secretList = secrets ? secrets.secrets : [];
  view.replaceChildren();
  const verdict = (c) => el("li", {}, t("check_" + c.name) + ": ", badge(t("status_" + c.status), c.status === "pass" ? "success" : (c.status === "fail" ? "danger" : "info")), " " + c.reason);
  const show = (box, report) => {
    const head = report.error ? t("route_error_" + report.error) : t(report.ok ? "route_ok" : "route_failed");
    box.replaceChildren(
      el("p", {}, head + " " + t("route_at", { at: report.checked_at })),
      el("ul", {}, report.checks.map(verdict)));
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
      el("p", { class: "hint" }, r.model + " @ " + r.endpoint + ", " + t("route_locality_" + r.locality) + (r.allowed_for_minors ? ", " + t("route_minors") : "")),
      !r.allowed_for_minors ? el("p", { class: "hint" }, t("route_minors_off")) : [],
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
  if (state.tab === "routes" && data.routes.some((r) => r.running)) {
    state.timer = setTimeout(() => guarded(refreshRoutes), ROUTES_POLL_MS);
  }
}

// The poll redraws the page; it waits while a form field has the focus so typing is not lost.
async function refreshRoutes() {
  const focus = document.activeElement;
  const typing = focus && document.getElementById("view").contains(focus) && ["INPUT", "SELECT", "TEXTAREA"].includes(focus.tagName);
  if (!typing) { await routesView(); return; }
  if (state.tab === "routes") state.timer = setTimeout(() => guarded(refreshRoutes), ROUTES_POLL_MS);
}

async function debugView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const report = await api("GET", "/api/debug");
  const sections = report.sections || {};
  const robotOnly = Object.fromEntries(Object.entries(sections).filter(([, s]) => s && s.scope === "robot"));
  const coreOnly = Object.fromEntries(Object.entries(sections).filter(([, s]) => !(s && s.scope === "robot")));
  const block = (cmd, data) => el("article", { class: "sl-term-block" },
    el("header", { class: "sl-term-block__head" },
      el("span", { class: "sl-term-block__cmd" }, el("span", { class: "sl-term-block__prompt" }, "$"), " " + cmd)),
    el("pre", { class: "sl-term-block__out pre-scroll" }, JSON.stringify(data, null, 2)));
  const hasRobot = Object.keys(robotOnly).length > 0;
  view.append(card(t("tab_debug"), t("debug_title"),
    el("div", { class: "row" }, button(t("debug_refresh"), "plain", () => guarded(debugView))),
    el("h3", { class: "sl-eyebrow" }, t("settings_core")),
    block("GET /api/debug", hasRobot ? { ...report, sections: coreOnly } : report),
    hasRobot ? [el("h3", { class: "sl-eyebrow" }, t("settings_robot")), block("GET /api/debug (robot)", robotOnly)] : []));
}

// ---------------------------------------------------------------- shell

const TABS = {
  enroll: enrollmentView,
  registry: registryView,
  catalogue: catalogueView,
  routes: routesView,
  settings: settingsView,
  debug: debugView,
};

function showTab(name) {
  stopPolling();
  state.tab = name;
  document.getElementById("view").setAttribute("aria-labelledby", "tab-" + name);
  notice("");
  for (const b of document.querySelectorAll("#tabs button")) {
    const on = b.dataset.tab === name;
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
    b.classList.toggle("sl-fx-aurora", on);
    b.classList.toggle("sl-fx-aurora--linear", on);
  }
  guarded(async () => { await TABS[name](); });
}

function tabKeys(ev) {
  const names = Object.keys(TABS);
  const step = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
  if (!step) return;
  ev.preventDefault();
  const next = names[(names.indexOf(state.tab) + step + names.length) % names.length];
  showTab(next);
  document.querySelector('#tabs button[data-tab="' + next + '"]').focus();
}

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
  showTab(state.tab);
}

async function boot() {
  stopPolling();
  const [strings, session] = await Promise.all([api("GET", "/api/strings"), api("GET", "/api/session")]);
  state.strings = strings.strings;
  state.session = session;
  document.documentElement.lang = strings.language;
  document.title = t("title");
  document.getElementById("title").textContent = t("title");
  themeControl();
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

boot().catch(() => notice("Cannot reach the robot.", true));
