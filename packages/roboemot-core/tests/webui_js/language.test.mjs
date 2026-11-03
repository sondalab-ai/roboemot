// One language per page (SPEC 11A, 12): a pseudo-locale run. Every page string becomes a marker
// ("⟦key⟧", its placeholders kept), each view is drawn from fixture replies, and every text the
// page shows must then be a marker, a number or symbol, or a value the fixture names on purpose
// (a name the owner gave, a host, a server error code). Hard-coded English in the script or a
// server code shown raw leaves text outside the markers and fails here. Run with `npm test`.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

// The page strings as Python holds them: strings.py is run on its own (it imports nothing), so
// the dicts are read exactly, never parsed from the source. PYTHON overrides the interpreter.
function pageStrings() {
  const path = fileURLToPath(new URL("../../src/roboemot/webui/strings.py", import.meta.url));
  const dump = "import json, runpy, sys; d = runpy.run_path(sys.argv[1]); print(json.dumps(d['STRINGS']))";
  const out = JSON.parse(execFileSync(process.env.PYTHON || "python3", ["-c", dump, path], { encoding: "utf8" }));
  assert.ok(Object.keys(out.en).length > 100 && Object.keys(out.it).length > 100, "strings.py read");
  return out;
}

// Each key as a marker with the placeholders its text uses, so the page still fills them.
function pseudoStrings(strings) {
  return Object.fromEntries(Object.entries(strings).map(([key, text]) => {
    const vars = [...new Set([...text.matchAll(/\{([a-z_]+)\}/g)].map((m) => m[1]))];
    return [key, "⟦" + key + vars.map((v) => "{" + v + "}").join("") + "⟧"];
  }));
}

const STRINGS = pseudoStrings(pageStrings().en);
const SYMBOLS = /^[\s\d.,:;()\-–—#%/+×←→✓·…$*@[\]=_]*$/;
const ATTRIBUTES = ["title", "aria-label", "placeholder", "alt"]; // text a person reads or hears

function leftover(text, allowed) {
  let rest = text;
  for (let prev = null; prev !== rest;) { prev = rest; rest = rest.replace(/⟦[^⟦⟧]*⟧/g, " "); }
  for (const value of allowed) rest = rest.split(value).join(" ");
  return SYMBOLS.test(rest) ? null : JSON.stringify(text);
}

// The texts of `root` (text nodes and the readable attributes) with the markers taken out, minus
// the allowed fixture values, that are still not numbers or symbols.
function leftovers(root, allowed) {
  const out = [];
  const walker = root.ownerDocument.createTreeWalker(root, 4 /* SHOW_TEXT */);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) out.push(leftover(node.nodeValue, allowed));
  for (const node of [root, ...root.querySelectorAll("*")]) {
    for (const name of ATTRIBUTES) {
      if (node.hasAttribute(name)) out.push(leftover(node.getAttribute(name), allowed));
    }
  }
  return out.filter(Boolean);
}

const T0 = Date.UTC(2026, 10, 2, 9, 30, 0) / 1000;
const LIVE_STATE = {
  poll_ms: 60000,
  mood: { label: "joy", valence: 0.1, arousal: -0.2, intensity: 0.3, circadian: { state: "drowsy", sleepiness: 0.4 } },
  modes: { asleep: false, paused: ["vision"], online: true, people: 1, persona: "default" },
  conversation: { phase: "thinking", route: "cloud", elapsed_s: 1.5 },
  attention: { count: 2, faces: [{ name: "Mia", target: true }, { name: null, target: false }] },
  reflexes: { last: { name: "startle-loud", age_s: 2.5 } },
  motion: { state: "external" },
  system: { cpu_temp_c: 51.2 },
  video: { enabled: false, poll_ms: 333, idle_stop_ms: 60000 },
  audio_egress: true,
  activity: {
    status: "cloud_down",
    notes: [{ code: "offline", reason: null }, { code: "cloud_down", reason: "connection_lost" }],
    events: [
      { kind: "cue", reason: "no_voice", cue: "fallback", result: "muted", failure: "closed", detail: "timeout", reset: "expired", t: T0, seq: 3, count: 1 },
      { kind: "bargein", reason: "cut", fired: "vad", confirmed: "onset", level_db: -21.25, robot_playing: true, moving: false, echo_corr: 0.12, t: T0, seq: 2, count: 2 },
      { kind: "session", reason: "connect_failed", why: "connect_timeout", t: T0, seq: 1, count: 1 },
      { kind: "turn", reason: "error", failure: "exception", detail: "runtime_error", t: T0, seq: 0, count: 1 },
    ],
  },
};
const LIVE_AUDIO = {
  audio_poll_ms: 60000,
  levels: { mic: { state: "ok", db: -30 }, out: { state: "paused", db: null }, floor_db: -50 },
  transcript: { enabled: true, epoch: "e", rev: 1, min_seq: 0, max_turns: 10, turns: [{ seq: 1, who: "user", text: "ciao robot", t: T0, interrupted: true }] },
};

async function page(t, routes, tab) {
  const p = await loadPage({
    "GET /api/session": () => ({ authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 }),
    "GET /api/strings": { language: "it", strings: STRINGS },
    "GET /api/enrollment/status": { state: "collecting", frames_collected: 2, frames_needed: 5, can_complete: false, hints: ["no_face", "turn_head"] },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    ...routes,
  });
  t.after(() => p.window.close());
  if (tab) {
    p.document.querySelector(`#tabs button[data-tab="${tab}"]`).click();
    await settle();
    await settle();
  }
  return p;
}

test("the shell and the enrollment tab show page strings only", async (t) => {
  const p = await page(t, {}, null);
  assert.deepEqual(leftovers(p.document.body, ["RoboEmot"]), []);
});

test("the Live tab words every code: label, reflex, notes, events, levels", async (t) => {
  const p = await page(t, { "GET /api/live/state": LIVE_STATE, "GET /api/live/audio": LIVE_AUDIO }, "live");
  assert.ok(p.document.querySelector("#live-events"), "the feed is drawn");
  // names and values the owner or the speaker gave: shown as they are
  assert.deepEqual(leftovers(p.document.body, ["Mia", "ciao robot", "cloud", "default"]), []);
});

test("the Live tab words the alone state, the nap and the resting reasons (CV-16)", async (t) => {
  const reasons = ["nap", "quiet_hours", "sleepiness", "hold"];
  for (const [i, status] of ["napping", "resting", "alone", "idle"].entries()) {
    const state = {
      ...LIVE_STATE,
      mood: { ...LIVE_STATE.mood, circadian: { state: "resting", reason: reasons[i], sleepiness: 0.9 } },
      modes: { ...LIVE_STATE.modes, alone: true },
      activity: {
        status,
        notes: [],
        events: [
          { kind: "alone", reason: "start", t: T0, seq: 4, count: 1 },
          { kind: "alone", reason: "face", t: T0, seq: 3, count: 1 },
          { kind: "alone", reason: "sound", t: T0, seq: 2, count: 1 },
          { kind: "nap", reason: "start", t: T0, seq: 1, count: 1 },
          { kind: "nap", reason: "face", t: T0, seq: 0, count: 1 },
          { kind: "nap", reason: "speech", t: T0, seq: 5, count: 1 },
        ],
      },
    };
    const p = await page(t, { "GET /api/live/state": state, "GET /api/live/audio": LIVE_AUDIO }, "live");
    assert.deepEqual(leftovers(p.document.body, ["Mia", "ciao robot", "cloud", "default"]), [], status);
  }
});

test("the Routes tab words the check reasons and the realtime facts", async (t) => {
  const chat = {
    name: "cloud", model: "m1", endpoint: "api.example.com", locality: "cloud", kind: "chat", key: "none", removable: false, running: false,
    last_check: {
      ok: false, error: "", checked_at: "2026-10-27T10:00:00+00:00", checks: [
        { name: "streaming", status: "fail", reason: "x", code: "no_stream", params: { why: "http", detail: "HTTP 404 at /v1/chat", reasoned: 12 } },
        { name: "tool_round_trip", status: "skip", reason: "x", code: "budget_spent", params: {} },
        { name: "first_token_latency", status: "pass", reason: "x", code: "latency", params: { seconds: 0.5, limit: 4 } },
        { name: "italian", status: "fail", reason: "x", code: "italian_ratio", params: { ratio: 10, need: 30 } },
      ],
    },
  };
  const rt = {
    name: "voice", model: "m2", endpoint: "rt.example.net", locality: "cloud", kind: "realtime", key: "none", removable: false, running: false,
    last_check: {
      kind: "realtime", ok: false, error: "", checked_at: "2026-10-27T10:00:00+00:00", mode: "platform",
      allocator_host: "alloc.example.net", connect_host: null, header_to_connect_host: "not_sent", platform_id_sent: false,
      update_seen: false, first_audio_ms: null, item_delete: "not_run", charged: 0, failure: "connect_failed",
      server_error_code: "invalid_value", server_error_about: "session.update",
      checks: [{ name: "realtime_connect", status: "fail", reason: "connect_failed", code: "session_failed", params: { failure: "connect_failed" } }],
    },
  };
  const p = await page(t, {
    "GET /api/routes": { routes: [chat, rt], name_max_chars: 32 },
    "GET /api/secrets": { secrets: [], max_name_chars: 32, max_value_chars: 64 },
  }, "routes");
  const allowed = ["cloud", "voice", "m1", "m2", "api.example.com", "rt.example.net", "alloc.example.net", "HTTP 404 at /v1/chat", "invalid_value", "session.update", "https://api.example.com/v1"];
  const body = p.document.querySelector("#view");
  assert.ok(body.textContent.includes("⟦check_reason_no_stream⟧"), "the reasons are drawn");
  assert.ok(body.textContent.includes("⟦check_why_httpHTTP 404 at /v1/chat⟧"));
  assert.deepEqual(leftovers(body, allowed), []);
});

test("the Catalogue tab words the categories", async (t) => {
  const p = await page(t, {
    "GET /api/catalogue": {
      records: [{ id: "r1", display_name: "Fido", kind: "animal", category: "dog", status: "pending", origin: "xmp", has_appearance: false, last_seen: null }],
      live: [{ entry_id: "animal-dog-7", kind: "animal", category: "potted plant", age_s: 1, identified: true }],
      limits: { max_xmp_bytes: 1, max_image_bytes: 1, max_items: 1 }, tombstones: 0,
    },
  }, "catalogue");
  assert.deepEqual(leftovers(p.document.querySelector("#view"), ["Fido"]), []);
});

test("the Emotions tab words labels, gestures and states; config names stay as code", async (t) => {
  const p = await page(t, {
    "GET /api/emotions/state": {
      poll_ms: 60000,
      affect: { label: "joy", valence: 0.4, arousal: -0.2, intensity: 0.3, baseline_valence: 0, baseline_arousal: -0.1 },
      register: { people: 1, motion_amplitude: 0.8, approach_allowed: true, reaction_intensity: "normal" },
      persona: { name: "default", effects: { motion_amplitude_x: 1.1 } },
      circadian: { state: "drowsy", sleepiness: 0.6, amplitude: 0.5 },
      expression: { last_label: "nod", last_amplitude: 0.42, started: 3 },
      moved: [{ age_s: 1.5, event: "sound.loud", rule: 3, label: "surprise", dv: 0, da: 0.6 }, { age_s: 9, event: "llm", rule: null, label: null, dv: 0.1, da: 0 }],
      controls: { available: false, refused: "external_control" },
      playable: { labels: ["joy", "calm"], gestures: ["nod", "perk_up"] },
      rules: [{ index: 3, event: "sound.loud", label: "surprise" }],
    },
  }, "emotions");
  const view = p.document.querySelector("#view");
  assert.ok(view.querySelector("#emo-controls"), "the tab is drawn");
  for (const code of view.querySelectorAll("code")) code.remove(); // config names and values
  // the rule list names the event (a config name) inside the option sentence
  assert.deepEqual(leftovers(view, ["sound.loud"]), []);
});

const ROW = {
  kind: "number", value: 1, default: 1, source: "defaults", description: "⟦a description⟧", writable: true,
  restart: false, admin_only: false, private: false, secret: false, opt_in: false, unit: "s", ge: 0, le: 10,
  nullable: false, choices: null, conservative: null,
};

test("the Settings tab words sources, tags and the robot section; values stay as code", async (t) => {
  const p = await page(t, {
    "GET /api/settings": {
      pin_required: true,
      fields: [
        { ...ROW, path: "affect.half_life_s", restart: true },
        { ...ROW, path: "affect.update_hz", source: "runtime", admin_only: true },
        { ...ROW, path: "webui.json_thing", kind: "json", value: { a: 1 }, default: { a: 1 }, writable: false },
      ],
      robot: {
        id: "bot", title: "Bot One", actions: ["restart"], restart_effects: [{ id: "kill_on_timeout", params: { timeout_s: 20 } }],
        fields: [
          { ...ROW, path: "robot.bot.limits.head.max_deg", source: "manifest", writable: false },
          { ...ROW, path: "robot.bot.realtime_session.allocator_host", kind: "text", value: null, default: null, source: "config", writable: false, nullable: true },
        ],
      },
    },
  }, "settings");
  const view = p.document.querySelector("#view");
  for (const button of view.querySelectorAll(".sl-disclosure")) button.click(); // open every group
  await settle();
  assert.ok(view.querySelector('[data-path="robot.bot.limits.head.max_deg"]'), "the robot rows are drawn");
  // config names (code, the group nav), values and units stay as they are
  for (const node of view.querySelectorAll("code, .setting-value, pre, .sl-nav__item, .row > span.hint")) node.remove();
  // an input is labelled by its config path (an identifier) for screen readers
  const paths = ["affect.half_life_s", "affect.update_hz"];
  assert.deepEqual(leftovers(view, ["⟦a description⟧", "Bot One", ...paths]), []);
});

test("the Registry tab words kinds and statuses", async (t) => {
  const p = await page(t, {
    "GET /api/registry": { records: [
      { id: "p1", display_name: "Mia", kind: "person", status: "active", last_seen: "2026-10-27T10:00:00+00:00" },
      { id: "a1", display_name: "Fido", kind: "animal", status: "pending", last_seen: null },
    ] },
    "GET /api/registry/quarantined": { keys: [] },
  }, "registry");
  const view = p.document.querySelector("#view");
  assert.ok(view.textContent.includes("Fido"));
  assert.deepEqual(leftovers(view, ["Mia", "Fido"]), []);
});

test("the page names its failure to load in a page language", async (t) => {
  const p = await loadPage({ "GET /api/strings": () => { throw new Error("down"); } });
  t.after(() => p.window.close());
  const text = p.document.getElementById("notice").textContent;
  assert.ok(["Cannot reach the robot.", "Impossibile raggiungere il robot."].includes(text), text);
});
