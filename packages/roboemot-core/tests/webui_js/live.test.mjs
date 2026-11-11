// Behaviour of the Live tab (SPEC 11A, 12): it polls /api/live/state only while the tab is open and
// the page is visible, stops on leaving the tab, hiding the page and logging out, and shows how to
// turn the feature on when the server says it is off. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const POLL_MS = 15;
const WAIT_MS = POLL_MS * 8;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const STATE = {
  poll_ms: POLL_MS,
  mood: { label: "joy", valence: 0.25, arousal: -0.1, intensity: 0.4, circadian: { state: "awake", sleepiness: 0.2 } },
  modes: { asleep: false, paused: [], online: true, people: 1, persona: "default" },
  conversation: { phase: "thinking", route: "lan", elapsed_s: 1.234 },
  attention: { count: 2, faces: [{ name: "Mia", target: true }, { name: null, target: false }] },
  reflexes: { last: { name: "startle", age_s: 2.5 } },
  motion: { state: "writing" },
  system: { cpu_temp_c: 51.2 },
  video: { enabled: false, poll_ms: 333 },
};

const AUDIO = {
  audio_poll_ms: POLL_MS * 4,
  levels: { mic: { state: "ok", db: -30 }, out: { state: "ok", db: -60 }, floor_db: null },
  transcript: { enabled: false, epoch: "", rev: 0, min_seq: 0, max_turns: 0, turns: [] },
};

const STRINGS = {
  live_reflex_last: "Last: {name}, {n} s ago",
  live_cpu_temp: "CPU temperature: {n} C",
  live_cpu_load: "CPU load: {n} %",
  live_cpu_freq_max: "CPU clock: {n} of {max} MHz",
  live_memory: "Memory used: {n} %",
  live_disk: "Disk: {free} GB free ({n} % used)",
  live_wifi: "Wi-Fi: {n} dBm, link quality {q}",
  live_process_mem: "RoboEmot memory: {n} MB, {threads} threads",
  live_throttle: "Throttling",
  live_throttle_active: "Now: {names}",
  live_throttle_flag_under_voltage: "under-voltage",
  live_people: "People: {n}",
  live_elapsed: "Waiting for {n} s",
  live_bpm: "{n} BPM",
  live_dance_accent_ago: "{n} s ago",
  live_dance_accent_ago_min: "{n} min ago",
  live_dance_accent_ago_h: "{n} h ago",
  live_dance_state_active: "dancing",
  live_dance_state_idle: "still",
  live_dance_source_room: "room music",
};

async function livePage(t, extra = {}) {
  let loggedIn = true;
  const page = await loadPage({
    "GET /api/session": () => ({ authenticated: loggedIn, csrf: "tok", pin_set: true, status_poll_ms: 1000 }),
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/state": STATE,
    "GET /api/live/audio": AUDIO,
    "POST /api/logout": () => { loggedIn = false; return { ok: true }; },
    ...extra,
  });
  t.after(() => page.window.close()); // jsdom timers would keep the runner alive
  return page;
}

const polls = (page) => page.calls.filter((c) => c.path === "/api/live/state").length;
// Waits for the page's own poll timer to have asked `n` times, however slow the machine is: the
// stubbed fetch has no timers, so each poll is drawn in the timer turn that asked it.
const POLLS_GIVE_UP_MS = 5000; // a page that stopped polling fails here instead of hanging
const pollsReach = async (page, n) => {
  const end = Date.now() + POLLS_GIVE_UP_MS;
  while (polls(page) < n) {
    assert.ok(Date.now() < end, `${polls(page)} of ${n} polls`);
    await sleep(1);
  }
};
const openLive = async (page) => {
  page.document.querySelector('#tabs button[data-tab="live"]').click();
  await settle();
};
const visibility = (page, value) => {
  Object.defineProperty(page.document, "visibilityState", { configurable: true, get: () => value });
  page.document.dispatchEvent(new page.window.Event("visibilitychange"));
};

test("the Live tab sits next to the others and draws one card per group", async (t) => {
  const page = await livePage(t);
  const tabs = [...page.document.querySelectorAll("#tabs button")].map((b) => b.dataset.tab);
  assert.deepEqual(tabs, ["enroll", "registry", "catalogue", "routes", "settings", "live", "emotions", "debug"]);
  await openLive(page);
  const grid = page.document.querySelector("#live-cards");
  assert.equal(grid.children.length, 9);
  const text = grid.textContent;
  for (const expected of ["joy", "0.25", "lan", "Mia", "startle", "writing", "51.2"]) {
    assert.ok(text.includes(expected) || text.toLowerCase().includes(expected), expected);
  }
  assert.equal(page.document.querySelector("#notice").textContent, "");
});

test("the system card shows the hardware readings", async (t) => {
  const system = {
    cpu_temp_c: 51.2,
    cpu: { percent: 12.5, freq_mhz: 1500, freq_max_mhz: 1800 },
    memory: { percent: 41.3, swap_percent: null },
    throttling: { now: ["under_voltage"], since_boot: ["under_voltage"] },
    disk: { free_gb: 8, percent: 62.5 },
    wifi: { signal_dbm: -52, link_quality: 58 },
    process: { cpu_percent: null, rss_mb: 150, threads: 21 },
  };
  const page = await livePage(t, { "GET /api/live/state": { ...STATE, system } });
  await openLive(page);
  const text = page.document.querySelector("#live-cards").textContent;
  for (const expected of [
    "CPU load: 12.5 %", "CPU clock: 1500 of 1800 MHz", "Memory used: 41.3 %",
    "Disk: 8 GB free (62.5 % used)", "Wi-Fi: -52 dBm, link quality 58",
    "RoboEmot memory: 150 MB, 21 threads", "Now: under-voltage",
  ]) {
    assert.ok(text.includes(expected), expected);
  }
  assert.ok(!text.includes("live_swap") && !text.includes("live_process_cpu"), "a null reading draws no line");
});

test("the system card draws no hardware line where the OS reports none", async (t) => {
  const page = await livePage(t, { "GET /api/live/state": { ...STATE, system: { cpu_temp_c: 51.2, cpu: null, throttling: null, disk: null, wifi: null } } });
  await openLive(page);
  const text = page.document.querySelector("#live-cards").textContent;
  for (const hidden of ["CPU load", "Throttling", "Disk", "Wi-Fi", "RoboEmot memory"]) {
    assert.ok(!text.includes(hidden), hidden);
  }
});

test("the modes card shows a warning while microphone audio may leave the robot", async (t) => {
  const page = await livePage(t, {
    "GET /api/live/state": { ...STATE, audio_egress: true },
    "GET /api/strings": { language: "en", strings: { ...STRINGS, live_audio_egress: "AUDIO OUT" } },
  });
  await openLive(page);
  assert.ok(page.document.querySelector("#live-cards").textContent.includes("AUDIO OUT"));
});

test("no audio warning in text mode", async (t) => {
  const page = await livePage(t, {
    "GET /api/live/state": { ...STATE, audio_egress: false },
    "GET /api/strings": { language: "en", strings: { ...STRINGS, live_audio_egress: "AUDIO OUT" } },
  });
  await openLive(page);
  assert.ok(!page.document.querySelector("#live-cards").textContent.includes("AUDIO OUT"));
});

test("the mood card says why the robot rests", async (t) => {
  const page = await livePage(t, {
    "GET /api/live/state": { ...STATE, mood: { ...STATE.mood, circadian: { state: "resting", reason: "nap", sleepiness: 0.4 } } },
    "GET /api/strings": { language: "en", strings: { ...STRINGS, live_circ_resting: "resting", live_circ_reason_nap: "A NAP" } },
  });
  await openLive(page);
  assert.ok(page.document.querySelector("#live-cards").textContent.includes("resting (A NAP)"));
});

test("the modes card says when the robot is alone and holding still", async (t) => {
  const strings = { ...STRINGS, live_alone: "ALONE STILL" };
  const alone = await livePage(t, {
    "GET /api/live/state": { ...STATE, modes: { ...STATE.modes, alone: true } },
    "GET /api/strings": { language: "en", strings },
  });
  await openLive(alone);
  assert.ok(alone.document.querySelector("#live-cards").textContent.includes("ALONE STILL"));
  const company = await livePage(t, { "GET /api/strings": { language: "en", strings } });
  await openLive(company);
  assert.ok(!company.document.querySelector("#live-cards").textContent.includes("ALONE STILL"));
});

test("the modes card names the motor state, not sleep", async (t) => {
  const strings = { ...STRINGS, live_motors_on: "MOTORS ON", live_motors_off: "MOTORS OFF" };
  const on = await livePage(t, { "GET /api/strings": { language: "en", strings } });
  await openLive(on);
  const text = on.document.querySelector("#live-cards").textContent;
  assert.ok(text.includes("MOTORS ON") && !text.includes("MOTORS OFF"));
  const off = await livePage(t, {
    "GET /api/live/state": { ...STATE, modes: { ...STATE.modes, asleep: true } },
    "GET /api/strings": { language: "en", strings },
  });
  await openLive(off);
  assert.ok(off.document.querySelector("#live-cards").textContent.includes("MOTORS OFF"));
});

test("the modes card shows the circadian nap apart from the motors, and nothing while awake", async (t) => {
  const strings = { ...STRINGS, live_motors_on: "MOTORS ON", live_rest_napping: "NAP BADGE", live_rest_drowsy: "DROWSY BADGE" };
  const nap = await livePage(t, {
    "GET /api/live/state": { ...STATE, modes: { ...STATE.modes, rest: "napping" } },
    "GET /api/strings": { language: "en", strings },
  });
  await openLive(nap);
  const text = nap.document.querySelector("#live-cards").textContent;
  assert.ok(text.includes("NAP BADGE") && text.includes("MOTORS ON"));
  const awake = await livePage(t, {
    "GET /api/live/state": { ...STATE, modes: { ...STATE.modes, rest: null } },
    "GET /api/strings": { language: "en", strings },
  });
  await openLive(awake);
  const quiet = awake.document.querySelector("#live-cards").textContent;
  assert.ok(!quiet.includes("NAP BADGE") && !quiet.includes("DROWSY BADGE"));
});

for (const [rest, label] of [["drowsy", "DROWSY BADGE"], ["resting", "REST BADGE"]]) {
  test(`the modes card shows the ${rest} badge`, async (t) => {
    const strings = { ...STRINGS, live_rest_drowsy: "DROWSY BADGE", live_rest_resting: "REST BADGE", live_rest_napping: "NAP BADGE" };
    const page = await livePage(t, {
      "GET /api/live/state": { ...STATE, modes: { ...STATE.modes, rest } },
      "GET /api/strings": { language: "en", strings },
    });
    await openLive(page);
    const text = page.document.querySelector("#live-cards").textContent;
    assert.ok(text.includes(label), label);
    assert.ok(!text.includes("NAP BADGE"));
  });
}

test("an old snapshot without the rest field draws the modes card with no rest badge", async (t) => {
  const { rest, ...modes } = { ...STATE.modes };
  assert.equal(rest, undefined);
  assert.ok(!("rest" in modes));
  const strings = { ...STRINGS, live_motors_on: "MOTORS ON", live_rest_drowsy: "DROWSY BADGE", live_rest_resting: "REST BADGE", live_rest_napping: "NAP BADGE" };
  const page = await livePage(t, {
    "GET /api/live/state": { ...STATE, modes },
    "GET /api/strings": { language: "en", strings },
  });
  await openLive(page);
  const text = page.document.querySelector("#live-cards").textContent;
  assert.ok(text.includes("MOTORS ON"));
  for (const label of ["DROWSY BADGE", "REST BADGE", "NAP BADGE", "undefined"]) assert.ok(!text.includes(label), label);
  assert.equal(page.document.querySelector("#live-cards").children.length, 9);
});

test("it polls at the interval the server names, and not before the tab is opened", async (t) => {
  const page = await livePage(t);
  await sleep(WAIT_MS);
  assert.equal(polls(page), 0);
  await openLive(page);
  await sleep(WAIT_MS);
  assert.ok(polls(page) >= 3, `polled ${polls(page)} times`);
});

test("leaving the tab stops the polling", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await settle();
  const seen = polls(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), seen);
});

test("a hidden page stops the polling and a visible one resumes it", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  visibility(page, "hidden");
  await settle();
  const seen = polls(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), seen, "no request while hidden");
  visibility(page, "visible");
  await sleep(WAIT_MS);
  assert.ok(polls(page) >= seen + 2, "resumed");
});

test("a page that is hidden when the tab opens makes no request until it is visible", async (t) => {
  const page = await livePage(t);
  visibility(page, "hidden");
  await openLive(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), 0);
  visibility(page, "visible");
  await sleep(WAIT_MS);
  assert.ok(polls(page) >= 1);
});

test("a reply that arrives after the page was hidden neither draws nor re-arms the timer", async (t) => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const page = await livePage(t, { "GET /api/live/state": async () => { await gate; return STATE; } });
  await openLive(page);
  assert.equal(polls(page), 1);
  visibility(page, "hidden");
  release();
  await sleep(WAIT_MS);
  assert.equal(polls(page), 1);
  assert.equal(page.document.querySelector("#live-cards").children.length, 0, "nothing drawn from the retired request");
});

test("a reply that arrives after the tab was left neither draws nor re-arms the timer", async (t) => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const page = await livePage(t, { "GET /api/live/state": async () => { await gate; return STATE; } });
  await openLive(page);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await settle();
  release();
  await sleep(WAIT_MS);
  assert.equal(polls(page), 1);
});

test("logging out stops the polling", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  page.document.querySelector("#logout").click();
  await settle();
  assert.equal(page.document.querySelector("#tabs").hidden, true, "back at the login");
  await sleep(POLL_MS * 2);
  const seen = polls(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), seen);
  assert.ok(page.calls.some((c) => c.path === "/api/logout"));
});

test("a re-opened tab runs one timer, not one per visit", async (t) => {
  const page = await livePage(t);
  await openLive(page);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await openLive(page);
  const before = polls(page);
  await sleep(POLL_MS * 10);
  const rate = (polls(page) - before) / 10;
  assert.ok(rate <= 1.6, `about one request per interval, got ${rate}`);
});

const audioPolls = (page) => page.calls.filter((c) => c.path.startsWith("/api/live/audio")).length;

test("live_disabled shows how to turn it on and stops both polls", async (t) => {
  const page = await livePage(t, {
    "GET /api/live/state": reply(404, { error: "live_disabled" }),
    "GET /api/live/audio": reply(404, { error: "live_disabled" }),
  });
  const timers = [];
  const real = page.window.setTimeout.bind(page.window);
  page.window.setTimeout = (fn, ms, ...rest) => { timers.push({ fn, ms }); return real(fn, ms, ...rest); };
  await openLive(page);
  assert.equal(polls(page), 1);
  assert.ok(page.document.querySelector("#live-cards").textContent.includes("live_disabled_body"));
  assert.equal(page.document.querySelector("#notice").textContent, "", "not an error banner");
  assert.deepEqual(timers.filter((x) => x.ms === 5000 || x.ms === POLL_MS), [], "no retry armed");
  assert.equal(page.run("state.timer"), null);
  assert.equal(page.run("liveState.audioTimer"), null);
  await sleep(WAIT_MS);
  assert.equal(polls(page), 1);
  assert.equal(audioPolls(page), 1);
});

test("a disabled Live tab asks again when it is entered again or the page comes back into view", async (t) => {
  let enabled = false;
  const page = await livePage(t, {
    "GET /api/live/state": () => (enabled ? STATE : reply(404, { error: "live_disabled" })),
    "GET /api/live/audio": () => (enabled ? AUDIO : reply(404, { error: "live_disabled" })),
  });
  await openLive(page);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await openLive(page);
  assert.equal(polls(page), 2, "entering the tab asks again");
  assert.equal(audioPolls(page), 2);
  enabled = true;
  visibility(page, "hidden");
  visibility(page, "visible");
  await settle();
  assert.equal(page.document.querySelector("#live-cards").children.length, 9, "back in view: on again");
  assert.ok(page.document.querySelector("#live-levels"), "the audio cards too");
  visibility(page, "hidden");
});

test("live_unavailable and a network failure keep retrying at the slow delay", async (t) => {
  for (const failure of [() => reply(503, { error: "live_unavailable" }), () => { throw new TypeError("Failed to fetch"); }]) {
    const page = await livePage(t, { "GET /api/live/state": failure, "GET /api/live/audio": failure });
    const timers = [];
    const real = page.window.setTimeout.bind(page.window);
    page.window.setTimeout = (fn, ms, ...rest) => { timers.push({ fn, ms }); return real(() => {}, 1e9); };
    await openLive(page);
    assert.equal(timers.filter((x) => x.ms === 5000).length, 2, "both polls retry");
  }
});

test("coming back into view keeps the last cards until the next reply", async (t) => {
  let release = null;
  let hold = false;
  const page = await livePage(t, {
    "GET /api/live/state": async () => { if (hold) await new Promise((r) => { release = r; }); return STATE; },
  });
  await openLive(page);
  await sleep(POLL_MS * 2);
  visibility(page, "hidden");
  hold = true;
  visibility(page, "visible");
  await settle();
  assert.equal(page.document.querySelector("#live-cards").children.length, 9, "cards kept while the poll is pending");
  release();
  await settle();
  visibility(page, "hidden");
  await settle();
});

test("a failed poll shows the error and retries", async (t) => {
  let n = 0;
  const page = await livePage(t, {
    "GET /api/live/state": () => (n++ === 0 ? reply(500, { error: "generic" }) : STATE),
  });
  await openLive(page);
  assert.notEqual(page.document.querySelector("#notice").textContent, "");
  assert.equal(polls(page), 1);
});

test("an expired session goes back to the login", async (t) => {
  let expired = false;
  const page = await livePage(t, {
    "GET /api/session": () => ({ authenticated: !expired, csrf: "tok", pin_set: true }),
    "GET /api/live/state": () => { expired = true; return reply(401, { error: "unauthorized" }); },
  });
  await openLive(page);
  await sleep(POLL_MS * 2);
  assert.equal(page.document.querySelector("#tabs").hidden, true);
  const seen = polls(page);
  await sleep(WAIT_MS);
  assert.equal(polls(page), seen, "no polling at the login");
});

test("everything the state holds goes through textContent", async (t) => {
  const hostile = { ...STATE, attention: { count: 1, faces: [{ name: "<img src=x onerror=alert(1)>", target: false }] } };
  const page = await livePage(t, { "GET /api/live/state": hostile });
  await openLive(page);
  const grid = page.document.querySelector("#live-cards");
  const images = [...grid.querySelectorAll("img")];
  assert.ok(images.every((i) => i.id === "live-video-img" && !i.hasAttribute("onerror")), "only the preview <img>");
  assert.ok(grid.textContent.includes("<img src=x"));
});

test("a Live poll carries the activity mark only after a click, key or touch (R3 SF1)", async (t) => {
  const page = await livePage(t);
  const marked = (c) => c.headers["X-RoboEmot-Active"] === "1";
  await openLive(page); // the click that opened the tab is input
  await sleep(WAIT_MS);
  const live = () => page.calls.filter((c) => c.path.startsWith("/api/live/") && c.path !== "/api/live/viewers");
  assert.equal(live().filter(marked).length, 1, "one poll reports the click, the rest are passive");
  const before = live().length;
  page.document.dispatchEvent(new page.window.Event("keydown"));
  await sleep(WAIT_MS);
  const after = live().slice(before);
  assert.ok(after.length > 1);
  assert.equal(after.filter(marked).length, 1);
  assert.ok(page.calls.filter((c) => c.path === "/api/live/viewers").every((c) => !marked(c)), "the banner poll stays passive");
});

const BEAT = {
  music: { playing: true, tempo_bpm: 120, confidence: 0.75, beat_age_s: 0.125 },
  dance: { state: "active", source: "room", bob: 0.5, sway: 0.25, last_accent_age_s: 3.25 },
};
const beatCard = (page) => page.document.querySelector("#live-beat-card");
const stillClock = (page) => { page.window.performance.now = () => 1000; };
const bars = (node) => [...node.querySelectorAll(".sl-progress")].map((b) => b.getAttribute("aria-valuenow"));

test("the beat card shows the tempo, the music, the dance and pulses from the server's phase", async (t) => {
  const page = await livePage(t, { "GET /api/live/state": { ...STATE, ...BEAT } });
  stillClock(page);
  await openLive(page);
  const node = beatCard(page);
  for (const expected of ["120 BPM", "live_yes", "dancing", "room music", "3.3 s ago"]) {
    assert.ok(node.textContent.includes(expected), expected);
  }
  assert.deepEqual(bars(node), ["75", "50", "25"]);
  const pulse = node.querySelector("#live-pulse");
  assert.ok(pulse.classList.contains("live-pulse--on"));
  assert.equal(pulse.style.getPropertyValue("--live-beat-period"), "0.500s");
  assert.equal(pulse.style.getPropertyValue("--live-beat-delay"), "-0.125s");
});

test("the beat card redraws the pulse on each poll, at the new phase", async (t) => {
  let age = 0.1;
  const page = await livePage(t, {
    "GET /api/live/state": () => ({ ...STATE, ...BEAT, music: { ...BEAT.music, beat_age_s: age } }),
  });
  stillClock(page);
  await openLive(page);
  const delay = () => beatCard(page).querySelector("#live-pulse").style.getPropertyValue("--live-beat-delay");
  assert.equal(delay(), "-0.100s");
  age = 0.3;
  await sleep(WAIT_MS);
  assert.equal(delay(), "-0.300s");
});

for (const [name, data] of [
  ["null", { music: null, dance: null }],
  ["absent", {}],
]) {
  test(`the beat card is idle and static with ${name} blocks`, async (t) => {
    const page = await livePage(t, { "GET /api/live/state": { ...STATE, ...data } });
    await openLive(page);
    const node = beatCard(page);
    assert.ok(node.textContent.includes("still"));
    assert.deepEqual(bars(node), ["0", "0", "0"]);
    const pulse = node.querySelector("#live-pulse");
    assert.ok(!pulse.classList.contains("live-pulse--on"));
    assert.equal(pulse.style.getPropertyValue("--live-beat-period"), "");
  });
}

test("the beat card is idle without a beat and hides the source of a dance fading out", async (t) => {
  const page = await livePage(t, {
    "GET /api/live/state": {
      ...STATE,
      music: { playing: false, tempo_bpm: null, confidence: 0.2, beat_age_s: null },
      dance: { state: "active", source: null, bob: 0.1, sway: 0, last_accent_age_s: null },
    },
  });
  await openLive(page);
  const node = beatCard(page);
  assert.ok(!node.querySelector("#live-pulse").classList.contains("live-pulse--on"));
  assert.ok(!node.textContent.includes("room music"));
  assert.ok(node.textContent.includes("dancing"));
});

test("the pulse follows the dance's own beat while the music block has none", async (t) => {
  const page = await livePage(t, {
    "GET /api/live/state": {
      ...STATE,
      music: { playing: false, tempo_bpm: null, confidence: 0, beat_age_s: null },
      dance: { state: "active", source: "own", tempo_bpm: 100, beat_age_s: 0.2, bob: 0.5, sway: 0.5, last_accent_age_s: null },
    },
  });
  stillClock(page);
  await openLive(page);
  const node = beatCard(page);
  const pulse = node.querySelector("#live-pulse");
  assert.ok(pulse.classList.contains("live-pulse--on"));
  assert.equal(pulse.style.getPropertyValue("--live-beat-period"), "0.600s");
  assert.equal(pulse.style.getPropertyValue("--live-beat-delay"), "-0.200s");
  assert.ok(node.textContent.includes("100 BPM"));
});

test("the dance outranks the music while active, and an idle dance leaves the pulse to the music", async (t) => {
  const dance = { ...BEAT.dance, tempo_bpm: 60, beat_age_s: 0.5 };
  const active = await livePage(t, { "GET /api/live/state": { ...STATE, music: BEAT.music, dance } });
  stillClock(active);
  await openLive(active);
  assert.equal(beatCard(active).querySelector("#live-pulse").style.getPropertyValue("--live-beat-period"), "1.000s");
  const idle = await livePage(t, { "GET /api/live/state": { ...STATE, music: BEAT.music, dance: { ...dance, state: "idle" } } });
  stillClock(idle);
  await openLive(idle);
  assert.equal(beatCard(idle).querySelector("#live-pulse").style.getPropertyValue("--live-beat-period"), "0.500s");
});

test("the pulse phase is corrected for half the round trip and the time since the reply", async (t) => {
  let clock = 1000;
  const page = await livePage(t, {
    "GET /api/live/state": () => { clock += 100; return { ...STATE, ...BEAT, poll_ms: 60000 }; },
  });
  page.window.performance.now = () => { const value = clock; clock += 10; return value; };
  await openLive(page);
  // sent 1000, received 1110, drawn 1120: 55 ms + 10 ms on top of the reported 125 ms
  assert.equal(beatCard(page).querySelector("#live-pulse").style.getPropertyValue("--live-beat-delay"), "-0.190s");
});

test("a tempo of 0 shows no tempo and no pulse", async (t) => {
  const page = await livePage(t, {
    "GET /api/live/state": { ...STATE, music: { playing: true, tempo_bpm: 0, confidence: 0.5, beat_age_s: 0.1 }, dance: null },
  });
  await openLive(page);
  const node = beatCard(page);
  assert.ok(!node.querySelector("#live-pulse").classList.contains("live-pulse--on"));
  assert.ok(node.textContent.includes("live_none"));
  assert.ok(!node.textContent.includes("Infinity"));
});

test("a malformed or partial beat block renders the empty state and keeps the cards", async (t) => {
  for (const blocks of [
    { music: { playing: true, confidence: 0.5 }, dance: { state: "active", bob: 0.5 } },
    { music: { playing: true, tempo_bpm: "fast", beat_age_s: {}, confidence: "x" }, dance: { last_accent_age_s: "soon", bob: null } },
    { music: 5, dance: "broken" },
  ]) {
    const page = await livePage(t, { "GET /api/live/state": { ...STATE, ...blocks } });
    await openLive(page);
    const node = beatCard(page);
    assert.ok(node, "the card is drawn");
    assert.equal(page.document.querySelector("#live-cards").children.length, 9);
    assert.ok(!node.querySelector("#live-pulse").classList.contains("live-pulse--on"));
    assert.ok(node.textContent.includes("live_none"));
    assert.equal(page.document.querySelector("#notice").textContent, "");
  }
});

test("a long accent age reads in minutes, then hours", async (t) => {
  for (const [age, expected] of [[59.5, "59.5 s ago"], [125, "2 min ago"], [7300, "2 h ago"], [12345.6, "3 h ago"]]) {
    const page = await livePage(t, {
      "GET /api/live/state": { ...STATE, ...BEAT, dance: { ...BEAT.dance, last_accent_age_s: age } },
    });
    await openLive(page);
    assert.ok(beatCard(page).textContent.includes(expected), expected);
  }
});

test("a failed poll marks the cards as old with the time of the last good update, once", async (t) => {
  let down = false;
  const page = await livePage(t, {
    "GET /api/strings": { language: "en", strings: { ...STRINGS, stale_since: "Last updated at {at}.", err_unreachable: "Cannot reach the robot." } },
    "GET /api/live/state": () => { if (down) throw new TypeError("Failed to fetch"); return STATE; },
  });
  await openLive(page);
  const grid = page.document.querySelector("#live-cards");
  const note = page.document.querySelector("#live-stale");
  assert.equal(grid.getAttribute("aria-busy"), null);
  assert.equal(note.hidden, true);
  const goodAt = page.run("when(pollMarks.get(document.getElementById('live-cards')).goodMs, true)");
  down = true;
  await sleep(POLL_MS * 3);
  assert.equal(grid.getAttribute("aria-busy"), "true");
  assert.equal(note.hidden, false);
  assert.ok(note.classList.contains("sl-callout--warning"));
  assert.equal(note.textContent, `Last updated at ${goodAt}.`);
  assert.equal(page.document.querySelectorAll("#live-stale").length, 1, "one callout");
  assert.equal(grid.children.length, 9, "the last values stay");
  const toast = page.document.querySelector("#notice .sl-toast");
  assert.equal(toast.textContent, "Cannot reach the robot.");
  await sleep(POLL_MS * 3);
  assert.equal(page.document.querySelector("#notice .sl-toast"), toast, "reported once");
  down = false;
  visibility(page, "hidden");
  visibility(page, "visible"); // the retry waits 5 s: coming back into view polls at once
  await sleep(POLL_MS * 2);
  assert.equal(grid.getAttribute("aria-busy"), null);
  assert.equal(note.hidden, true);
  assert.equal(page.document.querySelector("#notice").textContent, "", "the poll takes back its own failure");
});

test("a recovering poll leaves a newer message from something else", async (t) => {
  let down = true;
  const page = await livePage(t, { "GET /api/live/state": () => (down ? reply(500, { error: "generic" }) : STATE) });
  await openLive(page);
  page.run('notice("Saved.")');
  down = false;
  visibility(page, "hidden");
  visibility(page, "visible");
  await sleep(POLL_MS * 2);
  assert.equal(page.document.querySelector("#notice").textContent, "Saved.");
});

// Counts the toasts put in the alert region: each one is an announcement.
function alerts(page) {
  const seen = { n: 0 };
  new page.window.MutationObserver((list) => {
    for (const m of list) seen.n += [...m.addedNodes].filter((n) => n.classList && n.classList.contains("sl-toast")).length;
  }).observe(page.document.getElementById("notice-alert"), { childList: true });
  return seen;
}

const UNREACHABLE = { "GET /api/strings": { language: "en", strings: { ...STRINGS, err_unreachable: "Cannot reach the robot." } } };

test("a poll that fails five times announces its failure once", async (t) => {
  const page = await livePage(t, { ...UNREACHABLE, "GET /api/live/state": () => { throw new TypeError("Failed to fetch"); } });
  const seen = alerts(page);
  await openLive(page);
  for (let i = 0; i < 4; i++) await page.run("livePoll(document.getElementById('live-cards'), liveState.serial)");
  assert.equal(page.calls.filter((c) => c.path === "/api/live/state").length, 5);
  assert.equal(seen.n, 1, "one announcement for one failure");
  assert.equal(page.document.querySelector("#notice").textContent, "Cannot reach the robot.");
});

test("two polls failing with the same text: one announcement, and the toast stays until both recover", async (t) => {
  let stateDown = true;
  let audioDown = true;
  const page = await livePage(t, {
    ...UNREACHABLE,
    "GET /api/live/state": () => { if (stateDown) throw new TypeError("Failed to fetch"); return STATE; },
    "GET /api/live/audio": () => { if (audioDown) throw new TypeError("Failed to fetch"); return AUDIO; },
  });
  const seen = alerts(page);
  await openLive(page);
  assert.equal(seen.n, 1, "the second poll does not announce the same text again");
  audioDown = false;
  await page.run("liveAudioPoll(document.getElementById('live-audio'), liveState.serial)");
  assert.equal(page.document.querySelector("#notice").textContent, "Cannot reach the robot.", "the state poll still fails");
  await page.run("liveAudioPoll(document.getElementById('live-audio'), liveState.serial)");
  await page.run("livePoll(document.getElementById('live-cards'), liveState.serial)");
  assert.equal(seen.n, 1);
  stateDown = false;
  await page.run("livePoll(document.getElementById('live-cards'), liveState.serial)");
  assert.equal(page.document.querySelector("#notice").textContent, "", "both recovered: the toast goes");
});

test("a poll that fails again after a recovery announces again", async (t) => {
  let down = true;
  const page = await livePage(t, { ...UNREACHABLE, "GET /api/live/state": () => { if (down) throw new TypeError("Failed to fetch"); return STATE; } });
  const seen = alerts(page);
  await openLive(page);
  down = false;
  await page.run("livePoll(document.getElementById('live-cards'), liveState.serial)");
  down = true;
  await page.run("livePoll(document.getElementById('live-cards'), liveState.serial)");
  assert.equal(seen.n, 2);
});

test("el() styles through the CSSOM, which isEqualNode compares as the style attribute", async (t) => {
  const page = await livePage(t);
  assert.equal(page.run('el("span", { style: { "--a": "1s" } }).getAttribute("style")'), "--a: 1s;");
  assert.equal(page.run('el("span", { style: { "--a": "1s" } }).isEqualNode(el("span", { style: { "--a": "1s" } }))'), true);
  assert.equal(page.run('el("span", { style: { "--a": "1s" } }).isEqualNode(el("span", { style: { "--a": "2s" } }))'), false);
});

test("a poll with equal data keeps every card node; a changed value replaces only its card", async (t) => {
  let valence = 0.25;
  const page = await livePage(t, { "GET /api/live/state": () => ({ ...STATE, mood: { ...STATE.mood, valence } }) });
  await openLive(page);
  const grid = page.document.querySelector("#live-cards");
  const first = [...grid.children];
  await pollsReach(page, 3); // two more polls with equal data
  assert.ok([...grid.children].every((node, i) => node === first[i]), "same nodes after equal polls");
  valence = 0.5;
  await pollsReach(page, polls(page) + 1); // one poll asked after the change
  const now = [...grid.children];
  assert.equal(now.length, 9);
  assert.notEqual(now[0], first[0], "the mood card is new");
  assert.ok(now[0].textContent.includes("0.50"));
  assert.ok(now.slice(1).every((node, i) => node === first[i + 1]), "the other cards are the same nodes");
});

test("a beating pulse replaces the beat card on each poll (the resync), and only that card", async (t) => {
  const page = await livePage(t, { "GET /api/live/state": { ...STATE, ...BEAT } });
  await openLive(page);
  const grid = page.document.querySelector("#live-cards");
  const first = [...grid.children];
  const pulse = page.document.querySelector("#live-pulse");
  const index = first.findIndex((n) => n.id === "live-beat-card");
  await sleep(POLL_MS * 4);
  assert.notEqual(page.document.querySelector("#live-pulse"), pulse, "a new pulse element restarts the animation");
  assert.ok([...grid.children].every((node, i) => i === index || node === first[i]));
});
