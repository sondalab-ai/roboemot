"use strict";
// Emotions tab (SPEC 7, 11, 11A, 12): the affect, the register, the persona's effects, the
// circadian state and the impulses that moved the affect, polled from GET /api/emotions/state at
// the rate it names; plus two test controls behind the PIN dialog: play an expression or gesture
// (POST /api/emotions/play, the `play` skill's path) and fire an appraisal rule (POST
// /api/emotions/inject). Codes and numbers only: no person, frame or audio. Shares the helpers of
// app.js and live.js (el, t, word, api, ask, card, badge, progress, notice, explain, liveRow,
// liveActivity, state), so it loads before app.js and calls them at run time.
//
// Polling: the timer is `state.timer` (stopPolling clears it), only while the tab is open and the
// document visible; `emoState.serial` retires a request in flight when the tab is left or hidden.
// The control cards are built once per view and kept across polls, so a choice is not lost.

const EMO_RETRY_MS = 5000;
const EMO_PERCENT = 100;
const emoState = { serial: 0, controls: null, grid: null }; // grid: the state cards of the open view

function emoNumber(value) {
  return value.toFixed(2);
}

// A signed value in [-1, 1] as a bar from 0 to 100 (0 at the middle), with its number.
function emoAxis(label, value) {
  return [liveRow(label, emoNumber(value)), progress(label, Math.round(((value + 1) / 2) * EMO_PERCENT), EMO_PERCENT)];
}

function emoAffectCard(a) {
  return card(t("tab_emotions"), t("emo_affect"),
    el("p", {}, badge(word("affect_label_", a.label), "info")),
    emoAxis(t("live_valence"), a.valence),
    emoAxis(t("live_arousal"), a.arousal),
    liveRow(t("live_intensity"), emoNumber(a.intensity)),
    progress(t("live_intensity"), Math.round(a.intensity * EMO_PERCENT), EMO_PERCENT),
    el("p", { class: "hint" }, t("emo_baseline", { v: emoNumber(a.baseline_valence), a: emoNumber(a.baseline_arousal) })));
}

// Parameter rows: the config name as code (an identifier), the value as a number, yes / no or code.
function emoParams(params) {
  return el("ul", { class: "sl-list" }, Object.entries(params).map(([k, v]) => el("li", {},
    el("code", { class: "sl-code" }, k), " ",
    typeof v === "boolean" ? t(v ? "live_yes" : "live_no") : (typeof v === "string" ? el("code", { class: "sl-code" }, v) : String(v)))));
}

function emoRegisterCard(d) {
  const r = d.register;
  const persona = [liveRow(t("live_persona"), el("code", { class: "sl-code" }, d.persona.name)), emoParams(d.persona.effects)];
  if (r === null) return card(t("tab_emotions"), t("emo_register"), el("p", { class: "hint" }, t("emo_register_none")), markedEyebrow("h3", t("emo_persona")), persona);
  const { people, ...params } = r;
  return card(t("tab_emotions"), t("emo_register"),
    el("p", { class: "hint" }, t("emo_register_intro")),
    liveRow(t("emo_people"), String(people)), emoParams(params),
    markedEyebrow("h3", t("emo_persona")), el("p", { class: "hint" }, t("emo_persona_intro")), persona);
}

function emoCircadianCard(c) {
  if (c === null) return card(t("tab_emotions"), t("emo_circadian"), el("p", { class: "hint" }, t("live_circ_off")));
  const parts = [el("p", {}, badge(word("live_circ_", c.state), "info"))];
  if (c.sleepiness !== null) {
    parts.push(liveRow(t("emo_sleepiness"), emoNumber(c.sleepiness)), progress(t("emo_sleepiness"), Math.round(c.sleepiness * EMO_PERCENT), EMO_PERCENT));
  }
  parts.push(liveRow(t("emo_circ_amplitude"), emoNumber(c.amplitude)));
  return card(t("tab_emotions"), t("emo_circadian"), parts);
}

function emoExpressionCard(e) {
  const body = e.last_label === null ? el("p", { class: "hint" }, t("emo_expression_none"))
    : el("p", {}, t("emo_expression_line", { label: emoName(e.last_label), amp: emoNumber(e.last_amplitude), n: e.started }));
  return card(t("tab_emotions"), t("emo_expression"), body);
}

// An expression or gesture name in the page language (an owner's own name as written).
function emoName(name) {
  return Object.prototype.hasOwnProperty.call(state.strings, "affect_label_" + name) ? word("affect_label_", name) : word("gesture_", name);
}

function emoMovedRow(m) {
  const source = m.rule === null ? t("emo_llm") : t("emo_rule", { n: m.rule });
  return el("li", { class: "emo-moved" },
    el("span", { class: "sl-eyebrow" }, t("emo_ago", { n: m.age_s.toFixed(1) })), " ",
    m.rule === null ? [] : [el("code", { class: "sl-code" }, m.event), " "], source, " → ",
    m.label === null ? t("emo_no_label") : word("affect_label_", m.label), ": ",
    t("emo_delta", { dv: emoSigned(m.dv), da: emoSigned(m.da) }));
}

function emoSigned(value) {
  return (value > 0 ? "+" : "") + emoNumber(value);
}

function emoMovedCard(moved) {
  return card(t("tab_emotions"), t("emo_moved"),
    moved.length ? el("ul", { class: "sl-list", id: "emo-moved" }, moved.map(emoMovedRow)) : el("p", { class: "hint", id: "emo-moved-none" }, t("emo_moved_none")),
    el("p", { class: "hint" }, t("emo_moved_note")));
}

// ---------------------------------------------------------------- controls

// A select with a hint, built once; `options` is [[value, label]] or {group: [[value, label]]}.
function emoSelect(id, groups) {
  const select = el("select", { id, class: "sl-field__input" });
  for (const [group, items] of groups) {
    const parent = group ? el("optgroup", { label: group }) : select;
    for (const [value, label] of items) parent.append(el("option", { value: String(value) }, label));
    if (group) select.append(parent);
  }
  return select;
}

// Refills a select only when its choices changed, keeping the one picked.
function emoSync(select, key, build) {
  if (select.dataset.key === key) return;
  const picked = select.value;
  select.replaceChildren(...build().children);
  select.dataset.key = key;
  if ([...select.options].some((o) => o.value === picked)) select.value = picked;
}

function emoControlsBuild() {
  const play = emoSelect("emo-play", []);
  const rule = emoSelect("emo-rule", []);
  const refused = el("p", { class: "sl-callout sl-callout--warning", id: "emo-refused", role: "status", hidden: true });
  const playBtn = button(t("emo_play"), "primary", () => guarded(() => emoPlay(play.value)), { attrs: { id: "emo-play-button" } });
  const ruleBtn = button(t("emo_inject"), "plain", () => guarded(() => emoInject(Number(rule.value))), { attrs: { id: "emo-inject-button" } });
  const node = card(t("tab_emotions"), t("emo_controls"),
    el("p", { class: "hint" }, t("emo_controls_intro")), refused,
    field("emo-play", t("emo_play_label"), play, t("emo_play_hint")), el("div", { class: "row" }, playBtn),
    field("emo-rule", t("emo_inject_label"), rule, t("emo_inject_hint")), el("div", { class: "row" }, ruleBtn));
  node.id = "emo-controls";
  emoState.controls = { node, play, rule, refused, buttons: [playBtn, ruleBtn] };
}

function emoControlsCard(d) {
  if (emoState.controls === null) emoControlsBuild();
  const c = emoState.controls;
  const labels = d.playable.labels.map((n) => [n, emoName(n)]);
  const gestures = d.playable.gestures.map((n) => [n, emoName(n)]);
  emoSync(c.play, JSON.stringify(d.playable), () => emoSelect("", [[t("emo_group_labels"), labels], [t("emo_group_gestures"), gestures]].filter(([, items]) => items.length)));
  const rules = d.rules.map((r) => [r.index, t("emo_rule_option", { n: r.index, event: r.event, label: r.label === null ? t("emo_no_label") : word("affect_label_", r.label) })]);
  emoSync(c.rule, JSON.stringify(d.rules), () => emoSelect("", [[null, rules]]));
  const off = !d.controls.available;
  c.refused.hidden = !off;
  c.refused.textContent = off ? t("err_" + d.controls.refused) : "";
  for (const b of c.buttons) b.disabled = off;
  return c.node;
}

async function emoPlay(label) {
  if (!label) return;
  let out = null;
  const done = await ask({
    title: t("emo_play"), message: t("emo_play_message", { name: emoName(label) }), pin: true, confirm: t("emo_play"),
    run: async ({ pin }) => { out = await api("POST", "/api/emotions/play", { label, pin }); },
  });
  if (done && out) notice(t(out.queued ? "emo_play_queued" : "emo_play_not_queued", { name: emoName(label) }), !out.queued);
}

async function emoInject(index) {
  if (!Number.isInteger(index)) return;
  let out = null;
  const done = await ask({
    title: t("emo_inject"), message: t("emo_inject_message", { n: index }), pin: true, confirm: t("emo_inject"),
    run: async ({ pin }) => { out = await api("POST", "/api/emotions/inject", { rule: index, pin }); },
  });
  if (done && out) notice(t(out.fired ? "emo_inject_fired" : "emo_inject_cooldown", { n: index }), !out.fired);
}

// ---------------------------------------------------------------- view and poll

// The state cards are rebuilt on each poll and only those that changed replace theirs (liveKeep);
// the control card is updated in place (a redraw would close an open list and drop the focus).
function emoRender(box, d) {
  const controls = emoControlsCard(d);
  const cards = emoState.grid;
  liveKeep(cards, [emoAffectCard(d.affect), emoExpressionCard(d.expression), emoCircadianCard(d.circadian),
    emoRegisterCard(d), emoMovedCard(d.moved)]);
  if (cards.parentNode !== box || controls.parentNode !== box) box.replaceChildren(cards, controls);
}

function emoActive(serial) {
  return serial === emoState.serial && state.tab === "emotions" && state.csrf !== null && document.visibilityState === "visible";
}

async function emoPoll(box, serial) {
  clearTimeout(state.timer);
  state.timer = null;
  if (!emoActive(serial)) return;
  let delay = EMO_RETRY_MS;
  try {
    const data = await api("GET", "/api/emotions/state", undefined, liveActivity());
    if (!emoActive(serial)) return;
    emoRender(box, data);
    pollGood(emoState.grid);
    delay = data.poll_ms || EMO_RETRY_MS;
  } catch (err) {
    if (!emoActive(serial)) return;
    if (err.status === 401 && err.message === "unauthorized") { await boot(); return; }
    if (err.message === "emotions_disabled") {
      // Off until the owner turns it on: no retry. Entering the tab or the page coming back into
      // view asks again (emotionsView, the visibility handler below).
      emoOff(box, card(t("tab_emotions"), t("emo_disabled_title"), el("p", {}, t("emo_disabled_body"))));
      return;
    }
    if (err.message === "emotions_unavailable") emoOff(box, card(t("tab_emotions"), t("tab_emotions"), el("p", {}, explain(err))));
    else pollFailed(emoState.grid, explain(err));
  }
  state.timer = setTimeout(() => emoPoll(box, serial), delay);
}

// A state the server named (off, unavailable) replaces the cards with `note`: nothing is stale,
// and the controls are built again when the cards come back.
function emoOff(box, note) {
  emoState.controls = null;
  pollReset(emoState.grid);
  box.replaceChildren(note);
}

function emotionsView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  emoState.controls = null;
  // Built before the first reply, so a first poll that fails has a grid to report for (pollFailed).
  emoState.grid = el("div", { class: "live-grid", id: "emo-state" });
  pollForget();
  const box = el("div", { class: "stack", id: "emo-cards" });
  view.append(el("div", { class: "stack" }, el("p", { class: "hint" }, t("emo_intro")), staleNote("emo-stale", ["emo-state"]), box));
  return emoPoll(box, ++emoState.serial);
}

document.addEventListener("visibilitychange", () => {
  if (state.tab !== "emotions" || state.csrf === null) return;
  if (document.visibilityState === "visible") {
    const box = document.getElementById("emo-cards");
    guarded(async () => { await (box ? emoPoll(box, ++emoState.serial) : emotionsView()); });
  } else {
    emoState.serial += 1;
    stopPolling();
  }
});
