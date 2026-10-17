"use strict";
// RoboEmot web UI. All text goes through textContent: names from the registry are never HTML.

const state = { strings: {}, csrf: null, session: {}, tab: "enroll", timer: null, form: null };

const t = (key, vars) => {
  let text = state.strings[key] || key;
  for (const [k, v] of Object.entries(vars || {})) text = text.replace("{" + k + "}", v);
  return text;
};

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(c));
  return node;
}

function notice(message, isError) {
  const n = document.getElementById("notice");
  n.textContent = message || "";
  n.className = isError ? "error" : "";
}

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
    throw err;
  }
  return data;
}

function explain(err) {
  const known = state.strings["err_" + err.message];
  let text = known || t("err_generic");
  if (err.message === "consent_refused" && err.detail) text += " " + err.detail;
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

// ---------------------------------------------------------------- login and setup

function loginView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  if (!state.session.pin_set) {
    view.append(el("section", {}, el("h2", {}, t("login_title")), el("p", {}, t("no_pin"))));
    return;
  }
  const pin = el("input", { type: "password", id: "pin", autocomplete: "off", inputmode: "numeric" });
  const form = el("form", {
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
  },
  el("h2", {}, t("login_title")),
  el("label", { for: "pin" }, t("pin")),
  pin,
  el("p", {}, el("button", { type: "submit", class: "primary" }, t("login"))));
  view.append(form);
  pin.focus();
}

// ---------------------------------------------------------------- enrollment

function enrollmentView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const box = el("section", {});
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
  box.replaceChildren(el("h2", {}, t("tab_enroll")), el("p", { class: "muted" }, t("enroll_intro")));
  const stateKey = "state_" + s.state;
  box.append(el("p", {}, t("state") + ": " + (state.strings[stateKey] || s.state)));
  if (!["collecting", "ready", "completing"].includes(s.state)) {
    box.append(el("button", {
      class: "primary", type: "button",
      onclick: () => guarded(async () => { await api("POST", "/api/enrollment/start", {}); await state.refresh(); }),
    }, t("enroll_start")));
    return;
  }
  const max = Math.max(s.frames_needed, 1);
  box.append(
    el("p", {}, t("frames") + ": " + s.frames_collected + " / " + s.frames_needed),
    el("progress", { max: max, value: Math.min(s.frames_collected, max) }),
  );
  if (s.hints.length) {
    box.append(el("ul", {}, s.hints.map((h) => el("li", {}, state.strings["hint_" + h] || h))));
  }
  if (s.can_complete) {
    if (!state.form) {
      const reg = await api("GET", "/api/registry");
      state.form = consentForm(reg.records.filter((r) => r.kind === "person" && r.status === "pending"));
    }
    box.append(state.form);
  } else {
    state.form = null;
  }
  box.append(el("p", {}, el("button", {
    type: "button",
    onclick: () => guarded(async () => { await api("POST", "/api/enrollment/cancel", {}); await state.refresh(); }),
  }, t("enroll_cancel"))));
}

function select(id, values, prefix) {
  return el("select", { id }, values.map((v) => el("option", { value: v }, t(prefix + v))));
}

function consentForm(pendingPeople) {
  const max = String(state.session.name_max_chars || 64);
  const name = el("input", { type: "text", id: "display_name", maxlength: max, required: true });
  const who = el("input", { type: "text", id: "consent_who", maxlength: max, required: true });
  const guardian = el("input", { type: "text", id: "guardian", maxlength: max });
  const age = select("age_group", ["unknown", "child", "teen", "adult", "senior"], "age_");
  const gender = select("gender", ["unknown", "female", "male"], "gender_");
  const minor = () => !["adult", "senior"].includes(age.value);
  const scopes = ["face", "voice", "refresh"].map((s) =>
    el("label", { class: "check" }, el("input", { type: "checkbox", value: s, checked: s !== "voice" }), t("scope_" + s)));
  const agreed = el("input", { type: "checkbox", id: "agreed" });
  const link = el("select", { id: "pending_id" },
    el("option", { value: "" }, t("pending_none")),
    pendingPeople.map((r) => el("option", { value: r.id }, r.display_name)));
  const form = el("form", {
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
  el("h2", {}, t("consent_title")),
  el("div", { class: "consent" }, el("p", {}, t("consent_biometric")), el("p", {}, t("consent_cloud"))),
  pendingPeople.length ? [el("label", { for: "pending_id" }, t("pending_link")), link] : [],
  el("label", { for: "display_name" }, t("display_name")), name,
  el("label", { for: "consent_who" }, t("consent_who")), who,
  el("label", { for: "age_group" }, t("age_group")), age,
  el("label", { for: "gender" }, t("gender")), gender,
  el("label", { for: "guardian" }, t("guardian")), guardian,
  el("p", { class: "muted" }, t("guardian_hint")),
  el("p", {}, t("scope_title")), scopes,
  el("label", { class: "check" }, agreed, t("agree")),
  el("p", {}, el("button", { type: "submit", class: "primary" }, t("enroll_complete"))));
  return form;
}

// ---------------------------------------------------------------- registry

async function registryView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const [list, quarantined] = await Promise.all([
    api("GET", "/api/registry"), api("GET", "/api/registry/quarantined"),
  ]);
  const rows = list.records;
  state.pinField = el("input", { type: "password", id: "confirm_pin", autocomplete: "off", inputmode: "numeric" });
  const entities = rows.filter((r) => r.kind !== "person");
  const table = rows.length === 0 ? el("p", { class: "muted" }, t("registry_empty")) : el("table", {},
    el("tr", {}, el("th", {}, t("display_name")), el("th", {}, ""), el("th", {}, t("last_seen")), el("th", {}, "")),
    rows.map((r) => el("tr", {},
      el("td", {}, r.display_name),
      el("td", {}, t("kind_" + r.kind) + ", " + (state.strings["status_" + r.status] || r.status)),
      el("td", {}, r.last_seen ? new Date(r.last_seen).toLocaleString() : t("never")),
      el("td", { class: "actions" }, rowActions(r, entities)))));
  view.append(el("section", {}, el("h2", {}, t("registry_title")),
    el("label", { for: "confirm_pin" }, t("pin_confirm")), state.pinField, table));
  view.append(el("section", {}, el("h2", {}, t("quarantined_title")),
    quarantined.keys.length === 0 ? el("p", { class: "muted" }, t("quarantined_none")) :
      el("ul", {}, quarantined.keys.map((k) => el("li", {}, k + " ", el("button", {
        type: "button", class: "danger",
        onclick: () => confirm(t("confirm_purge")) && guarded(async () => {
          await api("POST", "/api/registry/quarantined/purge", { key: k, confirm: true, pin: takePin() });
          await registryView();
        }),
      }, t("purge")))))));
}

function takePin() {
  const value = state.pinField.value;
  state.pinField.value = "";
  return value;
}

function rowActions(r, entities) {
  const reload = () => registryView();
  const buttons = [el("button", {
    type: "button",
    onclick: () => {
      const next = prompt(t("new_name"), r.display_name);
      if (next === null) return;
      guarded(async () => { await api("POST", "/api/registry/" + r.id + "/rename", { display_name: next }); await reload(); });
    },
  }, t("rename"))];
  if (r.kind === "person") {
    buttons.push(el("button", {
      type: "button", class: "danger",
      onclick: () => confirm(t("confirm_forget", { name: r.display_name })) && guarded(async () => {
        await api("POST", "/api/registry/" + r.id + "/forget", { confirm: true, pin: takePin() });
        await reload();
      }),
    }, t("forget")));
  } else {
    for (const other of entities.filter((o) => o.id !== r.id)) {
      buttons.push(el("button", {
        type: "button",
        onclick: () => guarded(async () => {
          await api("POST", "/api/registry/merge", { keep_id: other.id, drop_id: r.id });
          await reload();
        }),
      }, t("merge") + " " + other.display_name));
    }
    buttons.push(el("button", {
      type: "button", class: "danger",
      onclick: () => confirm(t("confirm_delete", { name: r.display_name })) && guarded(async () => {
        await api("POST", "/api/registry/" + r.id + "/delete", { confirm: true });
        await reload();
      }),
    }, t("delete")));
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
  if (report.photo !== "none") parts.push(el("p", { class: "muted" }, t("photo_" + report.photo)));
  if (report.truncated) parts.push(el("p", { class: "muted" }, t("catalogue_truncated")));
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
  state.pinField = el("input", { type: "password", id: "confirm_pin", autocomplete: "off", inputmode: "numeric" });
  const xmp = el("input", { type: "file", id: "xmp_file", accept: ".xmp,application/rdf+xml,text/xml" });
  const photo = el("input", { type: "file", id: "photo_file", accept: "image/jpeg,image/png" });
  const result = el("div", { id: "import_result", role: "status" });
  const form = el("form", {
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
  el("label", { for: "xmp_file" }, t("catalogue_xmp")), xmp,
  el("label", { for: "photo_file" }, t("catalogue_photo")), photo,
  el("p", {}, el("button", { type: "submit", class: "primary" }, t("catalogue_import"))));

  const confirmBy = (record, entryId) => guarded(async () => {
    await api("POST", "/api/catalogue/" + record.id + "/confirm", entryId ? { entry_id: entryId } : {});
    notice(t("saved"));
    await catalogueView();
  });
  const pending = data.records.filter((r) => r.status === "pending");
  const origin = (r) => t("origin_" + r.origin) + ", " + (r.has_appearance ? t("has_appearance") : t("no_appearance"));
  const records = data.records.length === 0 ? el("p", { class: "muted" }, t("catalogue_none")) : el("table", {},
    data.records.map((r) => el("tr", {},
      el("td", {}, r.display_name),
      el("td", {}, t("kind_" + r.kind) + (r.category ? " (" + r.category + ")" : "") + ", " + origin(r)),
      el("td", { class: "actions" },
        r.status === "pending" && r.has_appearance ?
          el("button", { type: "button", onclick: () => confirmBy(r, null) }, t("catalogue_activate")) : [],
        data.live.filter((e) => r.status === "pending" && e.kind === r.kind && (!r.category || e.category === r.category))
          .map((e) => el("button", { type: "button", onclick: () => confirmBy(r, e.entry_id) },
            t("catalogue_confirm", { name: r.display_name }) + " ← " + e.category + " #" + e.entry_id.split("-").pop()))))));
  const live = data.live.length === 0 ? el("p", { class: "muted" }, t("catalogue_live_none")) :
    el("ul", {}, data.live.map((e) => el("li", {}, t("kind_" + e.kind) + " " + e.category + " #" + e.entry_id.split("-").pop() +
      (e.identified ? " ✓" : ""))));
  view.append(
    el("section", {}, el("h2", {}, t("tab_catalogue")), el("p", { class: "muted" }, t("catalogue_intro")), form, result),
    el("section", {}, el("h3", {}, t("catalogue_pending")), pending.length ? el("p", { class: "muted" }, t("catalogue_pick")) : [], records),
    el("section", {}, el("h3", {}, t("catalogue_live")), live),
    el("section", {},
      el("label", { for: "confirm_pin" }, t("pin_confirm")), state.pinField,
      el("button", {
        type: "button", class: "danger",
        onclick: () => confirm(t("catalogue_clear_confirm")) && guarded(async () => {
          const out = await api("POST", "/api/catalogue/candidates/clear", { confirm: true, pin: takePin() });
          notice(t("catalogue_cleared", { n: out.cleared }));
        }),
      }, t("catalogue_clear")),
      data.tombstones ? el("button", {
        type: "button",
        onclick: () => confirm(t("tombstones_clear_confirm")) && guarded(async () => {
          const out = await api("POST", "/api/catalogue/tombstones/clear", { confirm: true });
          notice(t("tombstones_cleared", { n: out.cleared }));
          await catalogueView();
        }),
      }, t("tombstones_clear", { n: data.tombstones })) : []));
}

async function debugView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const report = await api("GET", "/api/debug");
  view.append(el("section", {},
    el("h2", {}, t("debug_title")),
    el("button", { type: "button", onclick: () => guarded(debugView) }, t("debug_refresh")),
    el("pre", {}, JSON.stringify(report, null, 2))));
}

// ---------------------------------------------------------------- shell

const TABS = {
  enroll: enrollmentView,
  registry: registryView,
  catalogue: catalogueView,
  debug: debugView,
};

function showTab(name) {
  stopPolling();
  state.tab = name;
  notice("");
  for (const b of document.querySelectorAll("#tabs button")) {
    b.setAttribute("aria-current", String(b.dataset.tab === name));
  }
  guarded(async () => { await TABS[name](); });
}

function shell() {
  const tabs = document.getElementById("tabs");
  tabs.hidden = false;
  tabs.replaceChildren(...Object.keys(TABS).map((name) =>
    el("button", { type: "button", "data-tab": name, onclick: () => showTab(name) }, t("tab_" + name))));
  const out = document.getElementById("logout");
  out.hidden = false;
  out.textContent = t("logout");
  out.onclick = () => guarded(async () => { await api("POST", "/api/logout", {}); state.csrf = null; await boot(); });
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
  if (session.authenticated) {
    state.csrf = session.csrf;
    shell();
  } else {
    state.csrf = null;
    document.getElementById("tabs").hidden = true;
    document.getElementById("logout").hidden = true;
    loginView();
  }
}

boot().catch(() => notice("Cannot reach the robot.", true));
