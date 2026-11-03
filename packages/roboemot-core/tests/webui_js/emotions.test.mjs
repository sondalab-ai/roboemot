// The Emotions tab (SPEC 7, 11, 11A, 12): the state cards from GET /api/emotions/state, the poll
// marked as activity only after input, and the two test controls behind the PIN dialog. Run with
// `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const STATE = {
  poll_ms: 60000,
  affect: { label: "joy", valence: 0.4, arousal: -0.2, intensity: 0.3, baseline_valence: 0, baseline_arousal: -0.1 },
  register: { people: 1, motion_amplitude: 0.8, approach_allowed: true, reaction_intensity: "normal" },
  persona: { name: "default", effects: { motion_amplitude_x: 1.1, reaction_steps: 0 } },
  circadian: { state: "drowsy", sleepiness: 0.6, amplitude: 0.5 },
  expression: { last_label: "nod", last_amplitude: 0.42, started: 3 },
  moved: [
    { age_s: 1.5, event: "sound.loud", rule: 3, label: "surprise", dv: 0, da: 0.6 },
    { age_s: 9, event: "llm", rule: null, label: null, dv: 0.1, da: 0 },
  ],
  controls: { available: true, refused: null },
  playable: { labels: ["joy", "calm"], gestures: ["nod"] },
  rules: [{ index: 0, event: "person.known", label: "calm" }, { index: 3, event: "sound.loud", label: "surprise" }],
};

const EN = {
  tab_emotions: "Emotions", emo_affect: "Affect", affect_label_joy: "joy", affect_label_surprise: "surprise",
  affect_label_calm: "calm", gesture_nod: "nod", emo_rule: "rule #{n}", emo_llm: "the conversation",
  emo_no_label: "no label", emo_delta: "valence {dv}, arousal {da}", emo_ago: "{n} s ago",
  emo_play: "Play", emo_inject: "Fire the rule", emo_play_queued: "{name} will play in a moment.",
  emo_inject_cooldown: "Rule #{n} did not fire: it is in its cooldown.", err_robot_asleep: "The robot is asleep: tests are off.",
  emo_expression_line: "{label}, amplitude {amp}; {n} played since the start", live_circ_drowsy: "drowsy",
  emo_rule_option: "#{n} {event} → {label}", pin: "PIN",
};

async function emotionsPage(t, { state = STATE, routes = {} } = {}) {
  let body = state;
  const p = await loadPage({
    "GET /api/session": () => ({ authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 }),
    "GET /api/strings": { language: "en", strings: EN },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/emotions/state": () => body,
    ...routes,
  });
  t.after(() => p.window.close());
  p.document.querySelector('#tabs button[data-tab="emotions"]').click();
  await settle();
  await settle();
  p.setState = (next) => { body = next; };
  return p;
}

async function confirmPin(p, pin) {
  const dialog = p.document.querySelector("dialog");
  dialog.querySelector("#dialog-pin").value = pin;
  dialog.querySelector("form").dispatchEvent(new p.window.Event("submit", { cancelable: true }));
  await settle();
  await settle();
}

test("the tab sits after Live and shows the state in words", async (t) => {
  const p = await emotionsPage(t);
  const tabs = [...p.document.querySelectorAll("#tabs button")].map((b) => b.dataset.tab);
  assert.equal(tabs[tabs.indexOf("live") + 1], "emotions");
  const text = p.document.querySelector("#view").textContent;
  assert.ok(text.includes("joy") && text.includes("drowsy"));
  assert.ok(text.includes("nod, amplitude 0.42; 3 played since the start"));
  const rows = [...p.document.querySelectorAll("#emo-moved li")].map((li) => li.textContent);
  assert.ok(rows[0].includes("sound.loud") && rows[0].includes("rule #3 → surprise: valence 0.00, arousal +0.60"), rows[0]);
  assert.ok(rows[1].includes("the conversation → no label"), rows[1]);
});

test("a poll is marked as activity only after input", async (t) => {
  const p = await emotionsPage(t);
  await p.run("emoPoll(document.getElementById('emo-cards'), emoState.serial)");
  await settle();
  const polls = p.calls.filter((c) => c.path === "/api/emotions/state");
  assert.ok(polls.length >= 2);
  assert.equal(polls[polls.length - 1].headers["X-RoboEmot-Active"], undefined, "no input since the last marked poll");
});

test("play asks for the PIN and sends the label through the play endpoint", async (t) => {
  let sent = null;
  const p = await emotionsPage(t, { routes: { "POST /api/emotions/play": (b) => { sent = b; return { queued: true }; } } });
  const select = p.document.querySelector("#emo-play");
  assert.deepEqual([...select.options].map((o) => o.value), ["joy", "calm", "nod"]);
  select.value = "nod";
  p.document.querySelector("#emo-play-button").click();
  await settle();
  await confirmPin(p, "123456");
  assert.deepEqual(sent, { label: "nod", pin: "123456" });
  const call = p.calls.find((c) => c.path === "/api/emotions/play");
  assert.equal(call.headers["X-CSRF-Token"], "tok");
  assert.equal(p.document.getElementById("notice").textContent, "nod will play in a moment.");
});

test("inject sends the rule index and says when the cooldown held it", async (t) => {
  let sent = null;
  const p = await emotionsPage(t, { routes: { "POST /api/emotions/inject": (b) => { sent = b; return { fired: false }; } } });
  const select = p.document.querySelector("#emo-rule");
  select.value = "3";
  p.document.querySelector("#emo-inject-button").click();
  await settle();
  await confirmPin(p, "123456");
  assert.deepEqual(sent, { rule: 3, pin: "123456" });
  assert.equal(p.document.getElementById("notice").textContent, "Rule #3 did not fire: it is in its cooldown.");
});

test("while the robot cannot take a test the buttons are off and say why", async (t) => {
  const p = await emotionsPage(t, { state: { ...STATE, controls: { available: false, refused: "robot_asleep" } } });
  assert.ok(p.document.querySelector("#emo-play-button").disabled);
  assert.ok(p.document.querySelector("#emo-inject-button").disabled);
  const why = p.document.querySelector("#emo-refused");
  assert.equal(why.hidden, false);
  assert.equal(why.textContent, "The robot is asleep: tests are off.");
});

test("a later poll keeps the picked choice and the same control card", async (t) => {
  const p = await emotionsPage(t);
  const card = p.document.querySelector("#emo-controls");
  p.document.querySelector("#emo-play").value = "calm";
  p.setState({ ...STATE, affect: { ...STATE.affect, label: "calm" } });
  await p.run("emoPoll(document.getElementById('emo-cards'), emoState.serial)");
  await settle();
  assert.equal(p.document.querySelector("#emo-controls"), card);
  assert.equal(p.document.querySelector("#emo-play").value, "calm");
});
