"use strict";
// Live tab (SPEC 11A, 12): the robot's inner state, polled from GET /api/live/state. State only: no
// media, no transcript (later slices add their own cards). Shares the helpers of app.js (el, t, api,
// card, badge, progress, empty, notice, explain, stopPolling, boot, state), so this file loads first
// and only calls them at run time. Text goes through textContent and styles through the CSSOM.
//
// Polling: one timer in `state.timer` (so leaving the tab, logging out and showTab stop it), only
// while the document is visible. `liveState.serial` retires a request that was in flight when the
// tab was left, hidden or reopened, so a late reply neither draws nor re-arms the timer.

const LIVE_RETRY_MS = 5000;
const LIVE_PERCENT = 100;
const liveState = { serial: 0 };

// Each card is { key, render(data) -> Node }. Slices T (audio, transcript) and U (video) add theirs
// here and read their own part of `data` (or poll their own endpoint from `render`).
const LIVE_CARDS = [
  { key: "mood", render: liveMoodCard },
  { key: "conversation", render: liveConversationCard },
  { key: "modes", render: liveModesCard },
  { key: "attention", render: liveAttentionCard },
  { key: "reflexes", render: liveReflexesCard },
  { key: "motion", render: liveMotionCard },
  { key: "system", render: liveSystemCard },
];

function liveRow(label, ...value) {
  return el("p", { class: "live-row" }, el("span", { class: "sl-eyebrow" }, label), " ", ...value);
}

function liveNumber(value) {
  return value.toFixed(2);
}

function liveMoodCard(d) {
  const m = d.mood;
  const parts = [
    el("p", {}, badge(m.label, "info")),
    liveRow(t("live_valence"), liveNumber(m.valence)),
    liveRow(t("live_arousal"), liveNumber(m.arousal)),
    liveRow(t("live_intensity"), liveNumber(m.intensity)),
  ];
  if (m.circadian) {
    const name = state.strings["live_circ_" + m.circadian.state] || m.circadian.state;
    parts.push(liveRow(t("live_energy"), name));
    if (m.circadian.sleepiness !== null) {
      parts.push(progress(t("live_energy"), Math.round((1 - m.circadian.sleepiness) * LIVE_PERCENT), LIVE_PERCENT));
    }
  } else {
    parts.push(el("p", { class: "hint" }, t("live_circ_off")));
  }
  return card(t("tab_live"), t("live_mood"), parts);
}

function liveConversationCard(d) {
  const c = d.conversation;
  const tone = { idle: "info", listening: "info", thinking: "warning", speaking: "success" }[c.phase] || "info";
  const parts = [el("p", {}, badge(state.strings["live_phase_" + c.phase] || c.phase, tone))];
  if (c.elapsed_s !== null) {
    parts.push(liveRow(t("live_route"), c.route === null ? t("live_route_pending") : c.route));
    parts.push(el("p", { class: "hint" }, t("live_elapsed", { n: c.elapsed_s.toFixed(1) })));
  }
  return card(t("tab_live"), t("live_conversation"), parts);
}

function liveModesCard(d) {
  const m = d.modes;
  const paused = m.paused.map((p) => state.strings["live_part_" + p] || p).join(", ");
  return card(t("tab_live"), t("live_modes"),
    el("div", { class: "row" },
      badge(t(m.asleep ? "live_asleep" : "live_awake"), m.asleep ? "warning" : "success"),
      badge(t(m.online ? "live_online" : "live_offline"), m.online ? "success" : "danger"),
      badge(t(m.minor_present ? "live_minor_yes" : "live_minor_no"), m.minor_present ? "warning" : "info")),
    liveRow(t("live_paused"), paused || t("live_paused_none")),
    liveRow(t("live_register"), m.register === null ? "-" : m.register),
    liveRow(t("live_persona"), m.persona));
}

function liveAttentionCard(d) {
  const a = d.attention;
  const parts = [el("p", {}, t("live_people", { n: a.count }))];
  if (a.count === 0) parts.push(el("p", { class: "hint" }, t("live_nobody")));
  if (a.faces.length) {
    parts.push(el("ul", {}, a.faces.map((f) =>
      el("li", {}, f.name === null ? t("live_unknown") : f.name, f.target ? [" ", badge(t("live_gaze"), "info")] : []))));
  }
  if (a.count > a.faces.length) parts.push(el("p", { class: "hint" }, t("live_more", { n: a.count - a.faces.length })));
  return card(t("tab_live"), t("live_attention"), parts);
}

function liveReflexesCard(d) {
  const last = d.reflexes.last;
  return card(t("tab_live"), t("live_reflexes"), el("p", {},
    last ? t("live_reflex_last", { name: last.name, n: last.age_s.toFixed(1) }) : t("live_reflex_none")));
}

function liveMotionCard(d) {
  const s = d.motion.state;
  const tone = { writing: "success", external: "warning", motors_off: "info" }[s] || "info";
  return card(t("tab_live"), t("live_motion"), el("p", {}, badge(state.strings["live_motion_" + s] || s, tone)));
}

function liveSystemCard(d) {
  const temp = d.system.cpu_temp_c;
  return card(t("tab_live"), t("live_system"), el("p", {},
    temp === null ? t("live_cpu_temp_none") : t("live_cpu_temp", { n: temp })));
}

function liveDisabledView(box) {
  box.replaceChildren(card(t("tab_live"), t("live_disabled_title"), el("p", {}, t("live_disabled_body"))));
}

function liveRender(box, data) {
  box.replaceChildren(...LIVE_CARDS.map((c) => c.render(data)));
}

// True while a request started for `serial` may still draw and re-arm the timer.
function liveActive(serial) {
  return serial === liveState.serial && state.tab === "live" && state.csrf !== null && document.visibilityState === "visible";
}

async function livePoll(box, serial) {
  stopPolling();
  if (!liveActive(serial)) return;
  let delay = LIVE_RETRY_MS;
  try {
    const data = await api("GET", "/api/live/state");
    if (!liveActive(serial)) return;
    notice("");
    liveRender(box, data);
    delay = data.poll_ms || LIVE_RETRY_MS;
  } catch (err) {
    if (!liveActive(serial)) return;
    if (err.status === 401 && err.message === "unauthorized") { await boot(); return; }
    if (err.message === "live_disabled") liveDisabledView(box);
    else if (err.message === "live_unavailable") box.replaceChildren(card(t("tab_live"), t("live_title"), el("p", {}, explain(err))));
    else notice(explain(err), true);
  }
  state.timer = setTimeout(() => livePoll(box, serial), delay);
}

function liveView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const box = el("div", { class: "live-grid", id: "live-cards" });
  view.append(el("div", { class: "stack" }, el("p", { class: "hint" }, t("live_intro")), box));
  const serial = ++liveState.serial;
  return livePoll(box, serial);
}

// Back in view: keep the cards on screen (their last values) and only restart the poll.
function liveResume() {
  const box = document.getElementById("live-cards");
  if (!box) return liveView();
  return livePoll(box, ++liveState.serial);
}

// Hidden page: stop at once and retire the request in flight. Visible again: redraw and resume.
document.addEventListener("visibilitychange", () => {
  if (state.tab !== "live" || state.csrf === null) return;
  if (document.visibilityState === "visible") {
    guarded(async () => { await liveResume(); });
  } else {
    liveState.serial += 1;
    stopPolling();
  }
});
