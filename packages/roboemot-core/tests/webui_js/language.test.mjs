// One language per page (SPEC 11A, 12): a pseudo-locale run. Every page string becomes a marker
// ("⟦key⟧", its placeholders kept), each view is drawn from fixture replies, and every text the
// page shows must then be a marker, a number or symbol, or a value the fixture names on purpose
// (a name the owner gave, a host, a server error code). Hard-coded English in the script or a
// server code shown raw leaves text outside the markers and fails here. Run with `npm test`.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

// The default-language block of strings.py: each key with the placeholders its text uses.
function pseudoStrings() {
  const src = readFileSync(new URL("../../src/roboemot/webui/strings.py", import.meta.url), "utf8");
  const block = src.slice(src.indexOf("EN: dict[str, str] = {"), src.indexOf("IT: dict[str, str] = {"));
  const out = {};
  const parts = block.split(/\n    "([a-z0-9_]+)": /);
  for (let i = 1; i < parts.length; i += 2) {
    const vars = [...new Set([...parts[i + 1].matchAll(/\{([a-z_]+)\}/g)].map((m) => m[1]))];
    out[parts[i]] = "⟦" + parts[i] + vars.map((v) => "{" + v + "}").join("") + "⟧";
  }
  return out;
}

const STRINGS = pseudoStrings();
const SYMBOLS = /^[\s\d.,:;()\-–—#%/+×←✓·…$*@[\]=_]*$/;

// The text nodes of `root` with the markers taken out, minus the allowed fixture values.
function leftovers(root, allowed) {
  const out = [];
  const walker = root.ownerDocument.createTreeWalker(root, 4 /* SHOW_TEXT */);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    let rest = node.nodeValue;
    for (let prev = null; prev !== rest;) { prev = rest; rest = rest.replace(/⟦[^⟦⟧]*⟧/g, " "); }
    for (const value of allowed) rest = rest.split(value).join(" ");
    if (!SYMBOLS.test(rest)) out.push(JSON.stringify(node.nodeValue));
  }
  return out;
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

test("the page names its failure to load in a page language", async (t) => {
  const p = await loadPage({ "GET /api/strings": () => { throw new Error("down"); } });
  t.after(() => p.window.close());
  const text = p.document.getElementById("notice").textContent;
  assert.ok(["Cannot reach the robot.", "Impossibile raggiungere il robot."].includes(text), text);
});
