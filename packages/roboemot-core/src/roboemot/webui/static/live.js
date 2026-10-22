"use strict";
// Live tab (SPEC 11A, 12): the robot's inner state, polled from GET /api/live/state, plus the audio
// cards (two level meters and the opt-in transcript), polled from GET /api/live/audio at their own
// rate. Levels and text only: no sound, no media. Shares the helpers of app.js (el, t, api,
// card, badge, progress, empty, notice, explain, stopPolling, boot, state), so this file loads first
// and only calls them at run time. Text goes through textContent and styles through the CSSOM.
//
// Video card: polls GET /api/live/frame.jpg at the rate the state names (`video.poll_ms`), only
// while the tab is open and visible and the card is not paused. Each JPEG becomes an object URL on
// the <img>; the previous URL is revoked at once, and every URL is revoked when polling stops.
// Banner: while logged in, every page polls GET /api/live/viewers and shows the kit callout
// #live-video-banner while someone is watching the video (any tab, any browser).
//
// Polling: the state timer is `state.timer` and the audio timer `liveState.audioTimer` (stopPolling
// clears both, so leaving the tab, logging out and showTab stop them), only while the document is
// visible. `liveState.serial` retires a request that was in flight when the
// tab was left, hidden or reopened, so a late reply neither draws nor re-arms the timer.

const LIVE_RETRY_MS = 5000;
const LIVE_PERCENT = 100;
const LIVE_METER_FLOOR_DB = -60;
const liveState = { serial: 0, audioTimer: null, after: 0, epoch: null, ui: null };
const liveVideo = { timer: null, ui: null, url: null, paused: false, chain: 0, running: null, pollMs: LIVE_RETRY_MS };
const liveBanner = { timer: null, serial: 0 };

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
  { key: "video", render: liveVideoCard },
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
  clearTimeout(state.timer);
  state.timer = null;
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

// ---------------------------------------------------------------- video card

function liveRevokeVideo() {
  if (liveVideo.url !== null) URL.revokeObjectURL(liveVideo.url);
  liveVideo.url = null;
  if (liveVideo.ui) {
    liveVideo.ui.img.removeAttribute("src");
    liveVideo.ui.img.hidden = true;
  }
}

// `drop`: also forget the card and the picture (leaving the tab, hiding the page, logging out).
function liveStopVideo(drop) {
  clearTimeout(liveVideo.timer);
  liveVideo.timer = null;
  liveVideo.running = null;
  liveVideo.chain += 1;
  if (drop) {
    liveRevokeVideo();
    liveVideo.ui = null;
    liveVideo.paused = false;
  }
}

function liveVideoText(text) {
  const ui = liveVideo.ui;
  if (!ui) return;
  ui.text.textContent = text;
  ui.text.hidden = text === "";
}

function liveShowFrame(blob) {
  const ui = liveVideo.ui;
  if (!ui) return;
  const old = liveVideo.url;
  liveVideo.url = URL.createObjectURL(blob);
  ui.img.src = liveVideo.url;
  ui.img.hidden = false;
  if (old !== null) URL.revokeObjectURL(old);
  liveVideoText("");
}

function liveVideoBuild() {
  const img = el("img", { class: "live-video", alt: t("live_video_alt"), id: "live-video-img" });
  img.hidden = true;
  const text = el("p", { class: "hint", id: "live-video-text" });
  text.hidden = true;
  const pause = button(t("live_video_pause"), "plain", () => liveVideoToggle(), { small: true, attrs: { id: "live-video-pause" } });
  const node = card(t("tab_live"), t("live_video"), img, text, pause, el("p", { class: "hint" }, t("live_video_note")));
  node.id = "live-video-card";
  liveVideo.ui = { node, img, text, pause };
}

function liveVideoToggle() {
  const ui = liveVideo.ui;
  if (!ui) return;
  liveVideo.paused = !liveVideo.paused;
  ui.pause.textContent = t(liveVideo.paused ? "live_video_resume" : "live_video_pause");
  if (liveVideo.paused) {
    liveStopVideo(false);
    liveRevokeVideo();
    liveVideoText(t("live_video_paused"));
  } else {
    liveVideoStart();
  }
}

function liveVideoStart() {
  const serial = liveState.serial;
  if (liveVideo.running === serial || liveVideo.paused) return;
  liveVideo.running = serial;
  liveVideo.chain += 1;
  liveVideoPoll(serial, liveVideo.chain);
}

// True while the request chain `chain` (started for `serial`) may still draw and re-arm its timer.
function liveVideoLive(serial, chain) {
  return liveActive(serial) && liveVideo.chain === chain && !liveVideo.paused;
}

function liveVideoCard(d) {
  if (liveVideo.ui === null) liveVideoBuild();
  const v = d.video;
  if (!v.enabled) {
    liveStopVideo(false);
    liveRevokeVideo();
    liveVideoText(t("live_video_off"));
    liveVideo.ui.pause.hidden = true;
    return liveVideo.ui.node;
  }
  liveVideo.ui.pause.hidden = false;
  liveVideo.pollMs = v.poll_ms;
  liveVideoStart();
  return liveVideo.ui.node;
}

async function liveVideoPoll(serial, chain) {
  liveVideo.timer = null;
  if (!liveVideoLive(serial, chain)) return;
  let delay = liveVideo.pollMs;
  try {
    const resp = await fetch("/api/live/frame.jpg", { method: "GET", credentials: "same-origin", cache: "no-store" });
    if (!liveVideoLive(serial, chain)) return;
    if (resp.ok) {
      const blob = await resp.blob();
      if (!liveVideoLive(serial, chain)) return;
      liveShowFrame(blob);
    } else {
      let code = "generic";
      try { code = (await resp.json()).error || code; } catch (e) { code = "generic"; }
      if (!liveVideoLive(serial, chain)) return;
      if (resp.status === 401 && code === "unauthorized") { await boot(); return; }
      liveRevokeVideo();
      liveVideoText(state.strings["err_" + code] || t("err_generic"));
      if (code !== "video_no_frame") delay = Math.max(delay, LIVE_RETRY_MS);
    }
  } catch (e) {
    if (!liveVideoLive(serial, chain)) return;
    liveRevokeVideo();
    liveVideoText(t("err_generic"));
    delay = Math.max(delay, LIVE_RETRY_MS);
  }
  liveVideo.timer = setTimeout(() => liveVideoPoll(serial, chain), delay);
}

// ---------------------------------------------------------------- banner

function liveBannerStop() {
  clearTimeout(liveBanner.timer);
  liveBanner.timer = null;
  liveBanner.serial += 1;
  const box = document.getElementById("live-video-banner");
  if (box) box.hidden = true;
}

function liveBannerActive(serial) {
  return serial === liveBanner.serial && state.csrf !== null && document.visibilityState === "visible";
}

async function liveBannerPoll(serial) {
  clearTimeout(liveBanner.timer);
  liveBanner.timer = null;
  if (!liveBannerActive(serial)) return;
  let delay = LIVE_RETRY_MS;
  try {
    const data = await api("GET", "/api/live/viewers");
    if (!liveBannerActive(serial)) return;
    const box = document.getElementById("live-video-banner");
    box.hidden = !data.active;
    delay = data.poll_ms || LIVE_RETRY_MS;
  } catch (err) {
    if (!liveBannerActive(serial)) return;
    if (err.status === 401 && err.message === "unauthorized") { await boot(); return; }
  }
  liveBanner.timer = setTimeout(() => liveBannerPoll(serial), delay);
}

// Logged in: ask now, then at the interval the server names (called by the page shell).
function liveBannerStart() {
  liveBannerStop();
  const box = document.getElementById("live-video-banner");
  const body = box.querySelector(".sl-callout__body");
  body.textContent = t("live_video_banner");
  liveBannerPoll(liveBanner.serial);
}

document.addEventListener("visibilitychange", () => {
  if (state.csrf === null) return;
  if (document.visibilityState === "visible") liveBannerStart();
  else liveBannerStop();
});

// ---------------------------------------------------------------- audio cards

// `drop`: also forget the drawn rows (leaving the tab, logging out), so their text is not kept.
function liveStopAudio(drop) {
  clearTimeout(liveState.audioTimer);
  liveState.audioTimer = null;
  if (drop) liveState.ui = null;
}

function liveMeter(label) {
  const bar = progress(label, 0, LIVE_PERCENT);
  const text = el("span", { class: "hint" });
  return { bar, text, node: el("div", { class: "live-meter" }, el("span", { class: "sl-eyebrow" }, label), " ", text, bar) };
}

function liveSetMeter(meter, level) {
  const ok = level.state === "ok" && level.db !== null;
  const span = -LIVE_METER_FLOOR_DB;
  const pct = ok ? Math.round(Math.min(Math.max((level.db - LIVE_METER_FLOOR_DB) / span, 0), 1) * LIVE_PERCENT) : 0;
  meter.bar.setAttribute("aria-valuenow", pct);
  meter.bar.style.setProperty("--sl-progress", pct + "%");
  meter.text.textContent = ok ? t("live_db", { n: level.db.toFixed(1) }) : t("live_level_" + level.state);
}

function liveAudioBuild(box) {
  const mic = liveMeter(t("live_mic"));
  const out = liveMeter(t("live_out"));
  const floor = el("p", { class: "hint" });
  const levelsCard = card(t("tab_live"), t("live_levels"), mic.node, out.node, floor, el("p", { class: "hint" }, t("live_levels_note")));
  levelsCard.id = "live-levels";
  const list = el("div", { class: "live-transcript", id: "live-transcript", role: "log", "aria-live": "polite", tabindex: "0" });
  const off = el("p", { class: "hint" }, t("live_transcript_off"));
  const none = el("p", { class: "hint" }, t("live_transcript_empty"));
  const clear = button(t("live_transcript_clear"), "plain", () => guarded(liveClearTranscript), { small: true, attrs: { id: "live-transcript-clear" } });
  const textCard = card(t("tab_live"), t("live_transcript"), off, none, list, clear, el("p", { class: "hint" }, t("live_transcript_note")));
  textCard.id = "live-transcript-card";
  box.replaceChildren(levelsCard, textCard);
  liveState.ui = { mic, out, floor, list, off, none, clear };
}

function liveAudioReset(box) {
  box.replaceChildren();
  liveState.ui = null;
  liveState.after = 0;
  liveState.epoch = null;
}

function liveTurnRow(turn) {
  const who = turn.who === "robot" ? "robot" : "user";
  const stamp = new Date(turn.t * 1000).toLocaleTimeString();
  const label = stamp + " " + t("live_transcript_" + who);
  const parts = [el("span", { class: "sl-eyebrow" }, label), " ", turn.text];
  if (turn.interrupted) parts.push(" ", badge(t("live_transcript_cut"), "warning"));
  return el("p", { class: "live-turn live-turn--" + who, "data-seq": turn.seq }, parts);
}

// A reply grows and can be trimmed after the page first saw it: the same `seq` comes again with its
// new text and replaces its row; an empty text (nothing of it was heard) removes the row.
function liveUpsertTurn(list, turn) {
  const key = String(turn.seq);
  const old = [...list.children].find((row) => row.dataset.seq === key);
  if (turn.text === "") { if (old) old.remove(); return; }
  if (old) old.replaceWith(liveTurnRow(turn));
  else list.append(liveTurnRow(turn));
}

function liveRenderTranscript(ui, tr) {
  ui.off.hidden = tr.enabled;
  ui.clear.hidden = !tr.enabled;
  ui.list.hidden = !tr.enabled;
  if (!tr.enabled) {
    ui.none.hidden = true;
    ui.list.replaceChildren();
    liveState.after = 0;
    liveState.epoch = tr.epoch;
    return;
  }
  if (liveState.epoch !== tr.epoch) ui.list.replaceChildren();
  liveState.epoch = tr.epoch;
  for (const turn of tr.turns) liveUpsertTurn(ui.list, turn);
  while (ui.list.firstChild && Number(ui.list.firstChild.dataset.seq) < tr.min_seq) ui.list.firstChild.remove();
  while (tr.max_turns > 0 && ui.list.children.length > tr.max_turns) ui.list.firstChild.remove();
  liveState.after = tr.rev;
  ui.none.hidden = ui.list.children.length > 0;
  if (tr.turns.length) ui.list.scrollTop = ui.list.scrollHeight;
}

function liveAudioRender(box, data) {
  if (liveState.ui === null || !box.contains(liveState.ui.list)) liveAudioBuild(box);
  const ui = liveState.ui;
  liveSetMeter(ui.mic, data.levels.mic);
  liveSetMeter(ui.out, data.levels.out);
  ui.floor.hidden = data.levels.floor_db === null;
  ui.floor.textContent = data.levels.floor_db === null ? "" : t("live_floor", { n: data.levels.floor_db.toFixed(1) });
  liveRenderTranscript(ui, data.transcript);
}

async function liveClearTranscript() {
  await api("POST", "/api/live/transcript/clear", {});
  if (liveState.ui) liveState.ui.list.replaceChildren();
  if (liveState.ui) liveState.ui.none.hidden = false;
}

async function liveAudioPoll(box, serial) {
  liveStopAudio();
  if (!liveActive(serial)) return;
  let delay = LIVE_RETRY_MS;
  try {
    const epoch = liveState.epoch ? "&epoch=" + encodeURIComponent(liveState.epoch) : "";
    const data = await api("GET", "/api/live/audio?after=" + liveState.after + epoch);
    if (!liveActive(serial)) return;
    liveAudioRender(box, data);
    delay = data.audio_poll_ms || LIVE_RETRY_MS;
  } catch (err) {
    if (!liveActive(serial)) return;
    if (err.status === 401 && err.message === "unauthorized") { await boot(); return; }
    if (err.message === "live_disabled" || err.message === "live_unavailable") liveAudioReset(box);
    else notice(explain(err), true);
  }
  liveState.audioTimer = setTimeout(() => liveAudioPoll(box, serial), delay);
}

function liveView() {
  const view = document.getElementById("view");
  view.replaceChildren();
  const box = el("div", { class: "live-grid", id: "live-cards" });
  const audio = el("div", { class: "live-grid", id: "live-audio" });
  view.append(el("div", { class: "stack" }, el("p", { class: "hint" }, t("live_intro")), box, audio));
  liveAudioReset(audio);
  const serial = ++liveState.serial;
  return Promise.all([livePoll(box, serial), liveAudioPoll(audio, serial)]);
}

// Back in view: keep the cards on screen (their last values) and only restart the polls.
function liveResume() {
  const box = document.getElementById("live-cards");
  const audio = document.getElementById("live-audio");
  if (!box || !audio) return liveView();
  liveState.after = 0; // turns may have changed while hidden: ask for the full list again
  liveState.epoch = null;
  const serial = ++liveState.serial;
  return Promise.all([livePoll(box, serial), liveAudioPoll(audio, serial)]);
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
