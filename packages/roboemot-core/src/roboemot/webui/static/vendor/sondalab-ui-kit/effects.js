/* @sondalab/ui-kit — effects.js
   Hand-authored Tier 1 driver for effects.css: the press light and its spill,
   the glass's specular edge, the
   glitch's timing and the aurora's one-per-document
   grant.
   Contract: docs/specs/2026-09-08-effects-layer-design.md

   Importing this module does nothing. mount() is explicit, which is what makes
   the IDE opt-in possible: a Theia surface that never calls it pays nothing.

   COLOUR: sl-audit reads CSS only, so a literal in this file would pass the gate
   while breaking the design-system contract. Every hue is read from a token with
   getComputedStyle and re-read on theme change. scripts/effects-js.test.mjs
   enforces the absence of literals. */

// The 8x8 Bayer wipe and the cell size are the kit's pure dither maths.
import {
  CELL_PX, CONTROL_CELL_PX, CHECK_INSET, CHECK_BOX_CELLS, wipeFrame, CHECK_CELLS, TICK_CELLS, DOT_FRAMES, SWITCH_CELLS, TICK_START, tickCount, dotFrame, knobCells,
} from "./dither-core.js";

const REDUCE = "(prefers-reduced-motion: reduce)";
const CAP_MS = 1500;                      // hard ceiling on any single burst
const rand = (a, b) => a + Math.random() * (b - a);

/* ── coordinator state (spec §Decision 5) ─────────────────────────────────────
   One ambient aurora and one glitch burst at a time. Busy auroras are exempt:
   they are bound to real work and several can legitimately run at once. */
let liveEl = null;
let armed = false;

/** How many glitch bursts may run at once, and how many are running.
 *
 *  The budget was a single boolean until 2026-09-12, when the only consumer
 *  was the showcase and one burst at a time was plainly right there. A real
 *  page can want more: sondalab.eu runs an ambient burst on its hero claim,
 *  another on each interlude marker and a hover burst on every product row,
 *  and with a ceiling of one most of them are simply dropped — the effect
 *  thins out rather than reading as restraint.
 *
 *  So the ceiling is the consumer's, and the DEFAULT IS STILL ONE: a surface
 *  that says nothing behaves exactly as before, and raising it is a decision
 *  someone made on purpose. A burst over the ceiling is still dropped, never
 *  queued — a queue produces a stutter train, which reads as breakage. */
let burstBudget = 1;
let burstsRunning = 0;

function onPointerMove(ev) {
  if (!(ev.target instanceof Element)) return;
  onSpecPointer(ev);
}

/* ── glass: the specular edge ─────────────────────────────────────────────────
   The pointer is a light above the surface (docs/specs/2027-improve-aesthetics.md,
   re-scope ruling 4, and R1's pick). Every glass host near the pointer gets
   two custom properties and nothing else — --slfx-spec-a, the light's bearing
   as a conic angle, and --slfx-spec-i, how lit its edge is — and effects.css
   draws a narrow arc on the border facing the light from them.

   Two springs per host, both frame-rate independent:
   - PRESENCE eases toward how close the pointer is: 1 on or over the box,
     falling to 0 at SPEC_REACH px beyond its edge, so the edge wakes before
     the pointer arrives and settles after it leaves.
   - DIRECTION is sprung as a VECTOR in box-normalised space, never as an
     angle: outside the box it has length 1; over the box it shortens, because
     a light overhead grazes the edge less. A pointer crossing the element
     therefore dims the arc, turns it and brightens it again on the far side,
     instead of the arc snapping 180° in a frame — the owner's note on the
     first probe was exactly that jump.
   The loop runs only while some edge is lit and stops itself when all have
   settled; a host is only measured while the IntersectionObserver still
   marks it near the viewport. */
const SPEC_HOSTS = ".sl-fx-glass";
/** How far beyond its box a host starts to catch the light, in CSS px. */
const SPEC_REACH = 140;
/** Per-frame spring rates at 60 fps; scaled by the real frame time. */
const SPEC_DIR_K = 0.06;
const SPEC_PRES_K = 0.08;
/** Below this the edge is dark and the host is released. */
const SPEC_SETTLE = 0.004;
/** Both springs closer to their targets than this, on every lit host, and the
 *  loop stops until the pointer moves or the page scrolls or resizes. */
const SPEC_REST = 0.001;
/** The vector length below which the arc is off, and above which it is full:
 *  the smoothstep that dims it through a crossing. */
const SPEC_DIM_LO = 0.12;
const SPEC_DIM_HI = 0.85;

const specHosts = new Set();
const specState = new Map();
let specX = 0;
let specY = 0;
let specIn = false;
let specRafId = 0;
let specPrevT = 0;

function collectSpecHosts(root) {
  if (root instanceof Element && root.matches(SPEC_HOSTS)) specHosts.add(root);
  for (const el of root.querySelectorAll(SPEC_HOSTS)) specHosts.add(el);
}

function onSpecPointer(ev) {
  specX = ev.clientX;
  specY = ev.clientY;
  specIn = true;
  ensureSpecLoop();
}

/** The pointer left the window: every edge goes dark on its own spring. */
function onSpecOut(ev) {
  if (ev.relatedTarget) return;
  specIn = false;
  ensureSpecLoop();
}

/** The pointer's direction from the host's centre, in box-normalised space,
 *  clamped to the unit circle.
 *  @returns {[number, number]} */
function specDirection(tx, ty, hw, hh) {
  let nx = tx / hw;
  let ny = ty / hh;
  const len = Math.hypot(nx, ny);
  if (len > 1) { nx /= len; ny /= len; }
  return [nx, ny];
}

function releaseSpec(el) {
  specState.delete(el);
  el.classList.remove("is-spec-lit");
  el.style.removeProperty("--slfx-spec-a");
  el.style.removeProperty("--slfx-spec-i");
  if (!el.getAttribute("style")) el.removeAttribute("style");
}

/** Whether a host needs measuring this frame: lit already, or near enough
 *  for the pointer to light it. Releases a host inside an opt-out region.
 *  Reads no layout. */
function specCandidate(el) {
  const s = specState.get(el);
  if (!s && (!specIn || !el.hasAttribute("data-sl-fx-near"))) return false;
  if (el.closest('[data-sl-fx="off"]')) {
    if (s) releaseSpec(el);
    return false;
  }
  return true;
}

/** One host, one frame, from a box measured before any host was written.
 *  @returns {0|1|2} 0 dark and released, 1 lit and still moving, 2 lit and at rest */
function stepSpec(el, r, kd, kp) {
  let s = specState.get(el);
  const hw = r.width / 2;
  const hh = r.height / 2;
  if (!hw || !hh) return 0;
  const tx = specX - (r.left + hw);
  const ty = specY - (r.top + hh);
  const gap = Math.hypot(Math.max(Math.abs(tx) - hw, 0), Math.max(Math.abs(ty) - hh, 0));
  const want = specIn ? Math.max(0, 1 - gap / SPEC_REACH) : 0;
  const [nx, ny] = specDirection(tx, ty, hw, hh);
  if (!s) {
    if (want <= 0) return 0;
    // Born pointing at the light, so the first frame does not sweep in.
    s = { dx: nx, dy: ny, pres: 0, a: "", i: "" };
    specState.set(el, s);
    el.classList.add("is-spec-lit");
  }
  s.dx += (nx - s.dx) * kd;
  s.dy += (ny - s.dy) * kd;
  s.pres += (want - s.pres) * kp;
  if (want === 0 && s.pres < SPEC_SETTLE) {
    releaseSpec(el);
    return 0;
  }
  const t = Math.min(1, Math.max(0, (Math.hypot(s.dx, s.dy) - SPEC_DIM_LO) / (SPEC_DIM_HI - SPEC_DIM_LO)));
  const lit = s.pres * t * t * (3 - 2 * t);
  // Conic angles start at 12 o'clock and run clockwise; atan2 on screen axes
  // starts at 3 o'clock, hence the quarter turn. Back in pixel space first, so
  // the bearing on a wide card is the pointer's, not the unit square's.
  const bearing = (Math.atan2(s.dy * r.height, s.dx * r.width) * 180) / Math.PI + 90;
  // Unchanged strings are not written: every write restyles the host's
  // subtree, since both properties inherit.
  const a = `${bearing.toFixed(1)}deg`;
  const i = lit.toFixed(3);
  if (a !== s.a) { el.style.setProperty("--slfx-spec-a", a); s.a = a; }
  if (i !== s.i) { el.style.setProperty("--slfx-spec-i", i); s.i = i; }
  const rest = Math.abs(nx - s.dx) < SPEC_REST && Math.abs(ny - s.dy) < SPEC_REST && Math.abs(want - s.pres) < SPEC_REST;
  return rest ? 2 : 1;
}

function integrateSpec(now) {
  const dt = Math.min((now - specPrevT) / 1000 || 1 / 60, 1 / 20);
  specPrevT = now;
  const kd = 1 - (1 - SPEC_DIR_K) ** (dt * 60);
  const kp = 1 - (1 - SPEC_PRES_K) ** (dt * 60);
  // Every box is read before any host is written: interleaved, each read
  // after a write forced its own style and layout pass.
  const boxes = [];
  for (const el of specHosts) {
    if (!el.isConnected) {
      if (specState.has(el)) releaseSpec(el);
      specHosts.delete(el);
      continue;
    }
    if (specCandidate(el)) boxes.push(el, el.getBoundingClientRect());
  }
  let moving = false;
  for (let k = 0; k < boxes.length; k += 2) {
    if (stepSpec(boxes[k], boxes[k + 1], kd, kp) === 1) moving = true;
  }
  specRafId = moving ? requestAnimationFrame(integrateSpec) : 0;
}

/** A scroll or resize moves the boxes under a resting pointer: the loop
 *  stopped at rest has to look again. */
function onSpecLayout() {
  if (specIn || specState.size) ensureSpecLoop();
}

function ensureSpecLoop() {
  if (specRafId) return;
  specPrevT = performance.now();
  specRafId = requestAnimationFrame(integrateSpec);
}

/* ── press: you acted on this ────────────────────────────────────────────────
   Re-scope R2 (owner's pick A, 2026-09-25). A press lights the control from
   within; controls within PRESS_REACH catch the light on the edge that faces
   it. Three custom properties per lit host and nothing else — --slfx-press,
   --slfx-spill, --slfx-spill-a — and effects.css draws from them. The light
   rises fast and falls on a slower, light-like decay, each value on its own
   clock, so a release does not snap the room dark. Pointer and keyboard both
   press: Enter and Space on a focused control are the same act as a click. */
const PRESS_HOSTS = ".sl-fx-press";
/** How far a neighbour may sit from the pressed control and still catch it. */
const PRESS_REACH = 48;
/** Rise and fall time constants, in seconds: the press is quick to light and
 *  slower to fade; the spill lags it a little both ways. */
const PRESS_TAU_IN = 0.04;
const PRESS_TAU_OUT = 0.22;
const SPILL_TAU_IN = 0.05;
const SPILL_TAU_OUT = 0.3;
const PRESS_SETTLE = 0.004;

const pressState = new Map();
let pressedEl = null;
let pressRafId = 0;
let pressPrevT = 0;

function pressOf(el) {
  let s = pressState.get(el);
  if (!s) {
    s = { p: 0, s: 0, pt: 0, st: 0 };
    pressState.set(el, s);
  }
  return s;
}

/** Light the pressed control, and every press host within reach on the edge
 *  that faces it.
 *  @param {Element} el the control being pressed */
function pressStart(el) {
  if (el.matches(":disabled") || el.closest('[data-sl-fx="off"]')) return;
  pressedEl = el;
  pressOf(el).pt = 1;
  el.classList.add("is-press-lit");
  const r = el.getBoundingClientRect();
  const cx = r.left + r.width / 2;
  const cy = r.top + r.height / 2;
  for (const n of el.ownerDocument.querySelectorAll(PRESS_HOSTS)) {
    if (n === el || n.closest('[data-sl-fx="off"]')) continue;
    const nr = n.getBoundingClientRect();
    const gap = Math.max(nr.left - r.right, r.left - nr.right, nr.top - r.bottom, r.top - nr.bottom, 0);
    if (gap > PRESS_REACH) continue;
    pressOf(n).st = 1 - gap / PRESS_REACH;
    // The conic bearing of the neighbour's edge that faces the pressed
    // control: 12 o'clock is 0, clockwise, as on the specular edge.
    const bearing = (Math.atan2(cy - (nr.top + nr.height / 2), cx - (nr.left + nr.width / 2)) * 180) / Math.PI + 90;
    n.style.setProperty("--slfx-spill-a", `${bearing.toFixed(1)}deg`);
    n.classList.add("is-spill-lit");
  }
  ensurePressLoop();
}

function pressEnd() {
  if (!pressedEl) return;
  pressOf(pressedEl).pt = 0;
  for (const s of pressState.values()) s.st = 0;
  pressedEl = null;
  ensurePressLoop();
}

function onPressDown(ev) {
  const el = ev.target instanceof Element ? ev.target.closest(PRESS_HOSTS) : null;
  if (el) pressStart(el);
}

function onPressKey(ev) {
  if (ev.key !== "Enter" && ev.key !== " ") return;
  const el = ev.target instanceof Element ? ev.target.closest(PRESS_HOSTS) : null;
  if (!el) return;
  if (ev.type === "keydown" && !ev.repeat) pressStart(el);
  else if (ev.type === "keyup") pressEnd();
}

function releasePress(el) {
  pressState.delete(el);
  el.classList.remove("is-press-lit", "is-spill-lit");
  for (const k of ["--slfx-press", "--slfx-spill", "--slfx-spill-a"]) el.style.removeProperty(k);
  if (!el.getAttribute("style")) el.removeAttribute("style");
}

function integratePress(now) {
  const dt = Math.min((now - pressPrevT) / 1000 || 1 / 60, 1 / 20);
  pressPrevT = now;
  let lit = false;
  for (const [el, s] of pressState) {
    s.p += (s.pt - s.p) * (1 - Math.exp(-dt / (s.pt > s.p ? PRESS_TAU_IN : PRESS_TAU_OUT)));
    s.s += (s.st - s.s) * (1 - Math.exp(-dt / (s.st > s.s ? SPILL_TAU_IN : SPILL_TAU_OUT)));
    if (!el.isConnected || (s.pt === 0 && s.st === 0 && s.p < PRESS_SETTLE && s.s < PRESS_SETTLE)) {
      releasePress(el);
      continue;
    }
    lit = true;
    el.style.setProperty("--slfx-press", s.p.toFixed(3));
    el.style.setProperty("--slfx-spill", s.s.toFixed(3));
  }
  pressRafId = lit ? requestAnimationFrame(integratePress) : 0;
}

function ensurePressLoop() {
  if (pressRafId) return;
  pressPrevT = performance.now();
  pressRafId = requestAnimationFrame(integratePress);
}

/* ── glitch ──────────────────────────────────────────────────────────────────
   Re-roll the two ghost channels for this burst: each gets its own pulse
   duration and a negative phase delay (so they start mid-cycle and drift
   apart), plus a shared amplitude multiplier. */
function randomizeGhosts(el) {
  const dur = () => `${(0.11 + Math.random() * 0.12).toFixed(3)}s`;
  const phase = () => `-${(Math.random() * 0.22).toFixed(3)}s`;
  el.style.setProperty("--slfx-gda", dur());
  el.style.setProperty("--slfx-gdb", dur());
  el.style.setProperty("--slfx-gdj", dur());
  el.style.setProperty("--slfx-gpha", phase());
  el.style.setProperty("--slfx-gphb", phase());
  el.style.setProperty("--slfx-gmul", (0.75 + Math.random() * 0.7).toFixed(2));
  for (const s of el.querySelectorAll(".sl-fx-glitch__shard")) {
    const px = (5 + Math.random() * 13) * (Math.random() < 0.5 ? -1 : 1);
    s.style.setProperty("--slfx-sx", `${px.toFixed(1)}px`);
  }
}

/* ── aurora: the bloom's child ──────────────────────────────────────────────
   The bloom is a Gaussian blur of a thicker copy of the arc, and CSS applies
   a filter before a mask, so the masked band and the blur need two boxes:
   a child of the host, whose ::before is the band. The host's own two
   pseudo-elements stay what they are (::after the edge). Injected here, so
   a consumer who never calls mount() keeps the CSS-only ring and never sees
   an element it did not write; aria-hidden, and CSS makes it inert to the
   pointer and to selection. A linear aurora has no ring and gets none. */
/* ── the lens: removed 2026-09-11 ─────────────────────────────────────────────
   Four hundred lines went here: a plate rasterised from a body clone, one
   cropped canvas per pane, and an SVG displacement run over it. It is gone,
   and not because it was unfinished — because the route it serves was
   replaced.

   The owner chose the DECLARED BACKDROP: a surface says what lies behind a
   region and the pane bends that, which is how both implementations in the
   knowledge base work — glass-studio hands its shader images and videos,
   glass-vue is a demo over a backdrop, and neither has ever refracted a live
   document. That route needs no capture, no plate, and nothing to go stale.

   The lens's own record, kept because it cost a day to learn and would cost
   another to relearn:
     - the plate can never become a GPU texture. A foreignObject raster is
       marked origin-unclean for good, proven against a single variable: the
       same blob-URL SVG uploads WITHOUT foreignObject and throws SecurityError
       WITH it. So the lens could never have fed the shader this layer is
       heading toward.
     - its plate rendered every string in a fallback face, because a page
       cannot read @font-face rules out of a cross-origin stylesheet; fixed by
       vendoring the faces and inlining them as data URIs.
     - it removed glass elements from the clone, which took their boxes out of
       flow and drifted every crop below them by hundreds of pixels.
     - and after both fixes a filled pane still cropped an opaque black
       rectangle that nine measurements could not explain.
   docs/memory/the-lens-plate-misaligns-because-webfonts-do-not-load.md carries
   all of it. */

const GLOW_HOSTS = ".sl-fx-aurora:not(.sl-fx-aurora--linear)";
const GLASS_HOSTS = ".sl-fx-glass";

/** How far the glow reaches past its host, in CSS pixels. Read once from the
 *  host rather than hard-coded: --slfx-glow and --slfx-rim-out are the two
 *  properties the glow's own inset used before it became a sibling.
 *  @param {Element} el the aurora host
 *  @returns {number} the spread in pixels */
function glowSpread(el) {
  const cs = getComputedStyle(el);
  const px = (name) => parseFloat(cs.getPropertyValue(name)) || 0;
  return 1 + px("--slfx-rim-out") + px("--slfx-glow");
}

/** Put the glow where the lens can see it.
 *
 *  A child is painted as part of its element, AFTER backdrop-filter has
 *  sampled what is behind it — so a lensed pane cannot refract its own aurora,
 *  and the light lies on the glass rather than in it. As a sibling painted
 *  just before the host it is part of the backdrop, and the lens bends it:
 *  measured on lab/1r, it pools along the inner edges and squeezes into a band
 *  at the bottom.
 *
 *  The host's PARENT is the seat, not an injected wrapper. A wrapper around an
 *  element a consumer wrote is the more robust shape on paper and the more
 *  invasive one in practice — an inline-flex box around a button moves layout.
 *  The parent is already there; it only has to be positioned, which is one
 *  reversible declaration, and then the glow's box is a single rect
 *  subtraction rather than the viewport-versus-offset-parent arithmetic that
 *  made the reference implementation's panel jump on its first drag.
 *  @param {Element} el the aurora host
 *  @param {HTMLElement} glow the glow to place
 *  @returns {void} */
/** Everything the glow's ring is made of, and every one of them is declared on
 *  the AURORA HOST. As a child the glow inherited them; as a sibling it
 *  inherits from the host's parent instead, where none of them exist — so the
 *  ::before that paints the ring collapsed and the glow rendered nothing at
 *  all. Measured: one differing pixel in 404 096 between the glow shown and
 *  the glow display:none'd. Copied across explicitly rather than re-declared,
 *  so the values still come from the host and a component that re-parametrises
 *  its own aurora is still followed. */
const GLOW_VARS = [
  "--slfx-angle", "--slfx-arc", "--slfx-bloom-hi", "--slfx-bloom-lo",
  "--slfx-bloom-out", "--slfx-bloom-w", "--slfx-glow", "--slfx-rim-out",
  "--slfx-soft",
  // the band's inward reach: read on the glow's ::before, declared on the host
  "--slfx-bleed",
  // and the light pole itself: the beam is MADE of it, and a background of an
  // undefined var() is no background at all — the beam was lit, drifting,
  // rotated and invisible until this line existed
  "--slfx-glint",
  "--slfx-bloom-hi",
  // the pillars' palette root: accent, vivid tier, gamut-fitted
  "--slfx-accent-vivid",
];

function seatGlow(el, glow) {
  const parent = glow.parentElement;
  if (!parent) return;
  if (getComputedStyle(parent).position === "static") parent.style.position = "relative";
  const r = el.getBoundingClientRect();
  const p = parent.getBoundingClientRect();
  // Pillars stay inside the element: no spread, the box IS the host's box.
  // effects.css depends on exactly this — it excludes the pillars from the
  // clip written from --slfx-glow (which would eat a box that never had that
  // spread) and gives them their own at inset 0. One decision, two files.
  const spread = glow.hasAttribute("data-sl-fx-pillars") ? 0 : glowSpread(el);
  glow.style.left = `${r.left - p.left - spread}px`;
  glow.style.top = `${r.top - p.top - spread}px`;
  glow.style.width = `${r.width + spread * 2}px`;
  glow.style.height = `${r.height + spread * 2}px`;
  const hostStyle = getComputedStyle(el);
  glow.style.setProperty("--slfx-seat-radius", hostStyle.borderTopLeftRadius);
  for (const name of GLOW_VARS) {
    const v = hostStyle.getPropertyValue(name);
    if (v) glow.style.setProperty(name, v);
  }

  // HOW MANY CURTAINS, not how wide one is. The pattern was fitted at 104px on
  // a 320px card — three across the pane — and left at that length it does not
  // scale: a 1104px row showed nine ribs where the card it was measured on
  // showed three, and the owner read the difference as stripes rather than as
  // light. So the author binds a count and the width follows the host, which
  // is why this belongs here: seatGlow() is already what runs on every resize.
  //
  // effects.css carries 104px as its own fallback, so a page that never mounts
  // keeps exactly the pattern it has today.
  const count = parseFloat(hostStyle.getPropertyValue("--slfx-pillar-count")) || PILLAR_COUNT;
  if (r.width > 0) glow.style.setProperty("--slfx-pillar-pitch", `${r.width / count}px`);

  // And a phase of its own. Every animation starts when its rule begins to
  // match, so a surface that lights all its cards at once starts all of them
  // on the same instant and they drift in formation — one pattern printed as
  // many times as there are panes. The offset is derived from the host's
  // ordinal rather than rolled, so it is stable across a re-render: a curtain
  // that jumped to a new phase on every navigation would flicker for nothing.
  // The golden ratio is the usual way to spread a small integer over [0, 1)
  // without two neighbours landing close together.
  glow.style.setProperty("--slfx-pillar-phase", String((glowOrdinal(el) * PHI) % 1));
}

/** The golden ratio's fractional part, for spreading ordinals over [0, 1). */
const PHI = 0.618033988749895;

/** How many curtains hang across a pane when the author binds no count. Three
 *  is what the pattern was fitted at, on the card it was fitted on. */
const PILLAR_COUNT = 3;

/** A stable small integer per host, assigned in the order hosts are first
 *  seated. Read off the id seatGlow's caller already gives them, so two calls
 *  for the same host return the same number and a resize does not reshuffle
 *  the page.
 *  @param {Element} el the aurora host
 *  @returns {number} */
function glowOrdinal(el) {
  const id = el.dataset.slFxId;
  if (!id) return 0;
  if (!ordinals.has(id)) ordinals.set(id, ordinals.size);
  return ordinals.get(id);
}
const ordinals = new Map();

function injectGlow(el) {
  const parent = el.parentElement;
  // Without a parent there is nowhere to seat it, and a host that is its own
  // root keeps the old shape rather than losing its aurora entirely.
  const seat = parent || el;
  for (const child of seat.children) {
    if (child.classList.contains("sl-fx-aurora__glow") && child.dataset.slFxFor === el.dataset.slFxId) return;
  }
  const glow = document.createElement("span");
  glow.className = "sl-fx-aurora__glow";
  glow.setAttribute("aria-hidden", "true");
  if (!parent) {
    el.appendChild(glow);
    return;
  }
  el.dataset.slFxId = el.dataset.slFxId || `g${glowSerial++}`;
  glow.dataset.slFxFor = el.dataset.slFxId;
  glow.setAttribute("data-sl-fx-seated", "");
  // --beam: the glow paints a beam through the pane instead of a ring around
  // it. Carried as an attribute on the glow, since the CSS that draws it
  // cannot see the host's class from a previous sibling.
  if (el.classList.contains("sl-fx-aurora--beam")) glow.setAttribute("data-sl-fx-beam", "");
  // PILLARS ARE THE AURORA'S LIGHT BY DEFAULT (2026-09-12). Curtains clipped
  // to the host's own box, in the accent and its complement, sitting where the
  // lens can bend them. The reason is a measurement, not a preference: the
  // conic wash follows the edge it hugs, so displacing it maps it onto very
  // nearly itself — 0.93% of pixels moved, against 5.00% for the curtains
  // under the same lens. A light a lens cannot be seen to bend is not light in
  // glass. The rotation is --sweep, opt-in; --beam is the line through the
  // pane; --rim's light lies OUTSIDE a filled control, where curtains clipped
  // to the box would never show. --pillars stays honoured so existing markup
  // keeps its meaning.
  // --linear is not in this list because a linear aurora never reaches here:
  // GLOW_HOSTS excludes it, so it has no seated glow to give curtains to.
  const OWN_LIGHT = ["sweep", "beam", "rim"];
  const wantsPillars = el.classList.contains("sl-fx-aurora--pillars")
    || OWN_LIGHT.every((mode) => !el.classList.contains(`sl-fx-aurora--${mode}`));
  if (wantsPillars) glow.setAttribute("data-sl-fx-pillars", "");
  parent.insertBefore(glow, el);
  seatGlow(el, glow);
  // Again once the page has settled. A ResizeObserver fires when a box CHANGES
  // size, not when it moves, so a host pushed sideways by a webfont arriving
  // above it leaves its glow a couple of pixels behind — measured at up to 2px
  // on the showcase before this second pass.
  document.fonts?.ready.then(() => seatGlow(el, glow));
  if (typeof ResizeObserver === "function") {
    const watch = new ResizeObserver(() => seatGlow(el, glow));
    watch.observe(el);
    watch.observe(parent);
    glowWatches.set(el, { watch, glow });
  }
}

/** @type {number} pairs a seated glow with the host it belongs to */
let glowSerial = 0;
/** Each seated glow's observer, so a host that leaves takes both with it. */
const glowWatches = new Map();

/* ── the glass anatomy ──────────────────────────────────────────────────────
   glass.css paints the material in three stacked layers — frost, tint, rim —
   and they have to be elements: a host has two pseudo-elements and the aurora
   already owns ::after. They are injected rather than written by the consumer
   for the same reason the aurora's glow is: a surface asks for glass with one
   class, and forgetting a span is not a failure mode a design system should
   have. The layers sit at negative z-index, so the element's own label paints
   over them and nothing wraps the consumer's content. */
const GLASS_LAYERS = ["backdrop", "tint", "specular", "shine"];

function injectGlass(el) {
  // glass.css:76 declares border-radius: var(--slg-radius, var(--sl-radius-md))
  // on .sl-glass, the class classList.add below is about to stamp on el — so
  // unless --slg-radius is already bound, the component's OWN radius (.sl-btn
  // at 10px, say) is about to be overwritten by the family default the moment
  // it joins the anatomy: catalog.md's canonical-strings table measured
  // exactly that on .sl-btn, 10px → 6px. L6: "a surface stays free to
  // re-parametrise its own radius without touching the effect" — the glass
  // layer keeps that contract by reading the host's OWN computed radius
  // before it becomes a glass host, and writing it back as the property
  // glass.css already resolves through, so the layer's own declaration
  // resolves back to what the component declared.
  //
  // Read BEFORE classList.add, not after: once .sl-glass is on the element,
  // getComputedStyle(el) already reflects glass.css's own fallback — the
  // very value this exists to preserve — so reading here is the only reading
  // that still sees the component's own declaration.
  //
  // Two cases need no rebind and get none, because both already compute a
  // non-empty --slg-radius at this point: glass-text sets it to 0
  // deliberately, and the round icon button's 50% binding
  // (.sl-icon-btn--round.sl-fx-glass) is keyed off .sl-fx-glass, already in
  // the markup before injectGlass ever runs — neither waits on .sl-glass.
  const cs = getComputedStyle(el);
  // getComputedStyle cannot tell "declared no radius, relying on the glass
  // family default" from "declared border-radius: 0" — both resolve to 0px.
  // The pin exists to stop the layer overriding a radius the COMPONENT
  // chose; a zero computed radius carries no such choice, so the safe
  // reading is the one that shipped before this pin existed — leave the
  // family default (--sl-radius-md) in place. A component that genuinely
  // wants square corners under glass says so with --slg-radius: 0 directly,
  // the same escape hatch the GLASS TEXT declension already uses
  // (effects.css, the GLASS TEXT section) to flatten its own pane.
  const corners = [cs.borderTopLeftRadius, cs.borderTopRightRadius, cs.borderBottomRightRadius, cs.borderBottomLeftRadius];
  const isZero = corners.every((c) => parseFloat(c) === 0);
  if (!cs.getPropertyValue("--slg-radius").trim() && !isZero) {
    // All four corners, not just the top-left one, or a shorthand radius
    // like `10px 10px 0 0` flattens to a single value. Horizontal and
    // vertical radii are read apart from the start, because a corner can
    // itself be elliptical (`10px / 20px`) — a uniform ELLIPTICAL radius
    // still needs the h/v form: collapsing to `corners[0]` unmodified would
    // write `10px 20px` straight into --slg-radius, which in the four-value
    // shorthand reads as TL/BR = 10px, TR/BL = 20px, a DIFFERENT geometry
    // than one ellipse repeated on every corner.
    const h = corners.map((c) => c.split(" ")[0]);
    const v = corners.map((c) => c.split(" ")[1] ?? c.split(" ")[0]);
    const flat = h.every((x) => x === h[0]) && v.every((x) => x === h[0]);
    const ownRadius = flat ? h[0] : v.some((val, i) => val !== h[i]) ? `${h.join(" ")} / ${v.join(" ")}` : h.join(" ");
    el.style.setProperty("--slg-radius", ownRadius);
  }
  // The host joins the anatomy too, not just its layers. glass.css declares the
  // stacking context, the radius and the derived light pole on .sl-glass, and
  // an element carrying only .sl-fx-glass got the layers with none of that —
  // which read as a material that had lost its highlight colour. One class is
  // what makes .sl-fx-glass and .sl-glass the same road rather than two.
  el.classList.add("sl-glass");
  for (const child of el.children) {
    if (child.classList.contains("sl-glass__backdrop")) return;
  }
  // Prepended in reverse, so the three land ahead of any content in DOM order
  // while keeping their own order among themselves. Among negative z-indexes
  // the declared value orders them anyway; DOM order only decides ties.
  for (const name of [...GLASS_LAYERS].reverse()) {
    const layer = document.createElement("span");
    layer.className = `sl-glass__${name}`;
    layer.setAttribute("aria-hidden", "true");
    el.prepend(layer);
  }
}

/** The lenses currently bound, so mount() can be called twice without giving
 *  an element two filters, and so a theme change can take them all away. */
const lensed = new Map();

/** High contrast, and ONLY high contrast.
 *
 *  Not suppressed(): that also refuses under prefers-reduced-motion, and a lens
 *  is not motion — it is a static property of a surface, it does not animate,
 *  and it does not redraw on scroll. A visitor who asked for less movement did
 *  not ask for less material. High contrast is different in kind: it exists so
 *  that what is behind a surface does not interfere with what is on it, and a
 *  lens deforming the text behind a button is exactly that interference.
 *  @returns {boolean} whether a lens may not run */
function lensRefused() {
  if (typeof document === "undefined") return true;
  return document.documentElement.getAttribute("data-sl-theme") === "high-contrast";
}

/** Give every glass host a real refraction of whatever is behind it.
 *
 *  Dynamically imported, like the GPU tier was, so a consumer who never mounts
 *  pays nothing for it — and so a browser that cannot do it degrades to the CSS
 *  material in glass.css with no branch here.
 *  @param {ParentNode} root the subtree to arm
 *  @returns {Promise<void>} */
async function armLenses(root) {
  if (lensRefused()) return;
  let lens;
  try { ({ lens } = await import("./glass-lens.js")); } catch { return; }
  const hosts = [];
  if (root instanceof Element && root.matches(GLASS_HOSTS)) hosts.push(root);
  hosts.push(...root.querySelectorAll(GLASS_HOSTS));
  for (const el of hosts) {
    if (lensed.has(el) || el.closest('[data-sl-fx="off"]')) continue;
    const handle = lens(el);
    if (handle) lensed.set(el, handle);
  }
}

/** The live light (re-scope R3): the aurora's run and busy readings as light
 *  with body. Dynamically imported, so a surface with nothing working pays
 *  nothing for the module; it follows hosts and attribute changes itself.
 *  Behind the motion guard: it is motion, and the CSS arc is its fallback.
 *  @param {ParentNode} root
 *  @returns {Promise<void>} */
let liveArmed = false;
async function armLiveLight(root) {
  if (liveArmed) return;
  liveArmed = true;
  let liveLight;
  try { ({ liveLight } = await import("./live-light.js")); } catch { return; }
  liveLight(root);
}

let halftoneArmed = false;
/** Every canvas[data-sl-halftone] drawn in dots (halftone.js). Dynamically
 *  imported, like the goo and the live light; it follows canvases added
 *  later and owns its own reduced-motion rule (a settled frame, not nothing).
 *  @param {ParentNode} root
 *  @returns {Promise<void>} */
async function armHalftone(root) {
  if (halftoneArmed) return;
  halftoneArmed = true;
  let halftone;
  try { ({ halftone } = await import("./halftone.js")); } catch { return; }
  halftone(root);
}

/** The goo fields currently bound, for the same two reasons as `lensed`:
 *  mount() may be called twice, and a container may leave the document. */
const gooFields = new Map();
const GOO_HOSTS = ".sl-fx-goo";

/** Give every declared container its one shared gooey field.
 *
 *  Dynamically imported, like the lens and the GPU tier: a consumer whose
 *  surface has no field pays nothing for the module existing. Tier 0 — the
 *  members laid out and unmerged — is what a browser sees if this never runs.
 *  @param {ParentNode} root the subtree to arm
 *  @returns {Promise<void>} */
async function armGooFields(root) {
  let gooField;
  try { ({ gooField } = await import("./goo-field.js")); } catch { return; }
  const hosts = [];
  if (root instanceof Element && root.matches(GOO_HOSTS)) hosts.push(root);
  hosts.push(...root.querySelectorAll(GOO_HOSTS));
  for (const el of hosts) {
    if (gooFields.has(el)) continue;
    const handle = gooField(el);
    if (handle) gooFields.set(el, handle);
  }
}

/** Whether the goo has been armed on this document already. Its own flag, not
 *  `armed`: the goo is armed before Tier 1's motion guard and outlives its
 *  refusal, so the two are not the same question. */
let gooArmed = false;

/** Arm the goo, and keep it armed.
 *
 *  Separate from the rest of `mount()` because of the motion preference. The
 *  goo ANIMATES NOTHING — it is laid out, not played, and the spec's own
 *  reduced-motion clause is "nothing animates; static material survives". So
 *  it refuses on high contrast (where the point is that nothing behind a
 *  surface may interfere with what is on it) and runs under
 *  prefers-reduced-motion, on the same reasoning `lensRefused()` carries: a
 *  visitor who asked for less movement did not ask for less material. Its CSS
 *  is not gated on `data-sl-fx="on"` either, so nothing about it depends on
 *  Tier 1 having armed.
 *
 *  Two observers, both owned here so there is one place that keeps a field
 *  live: containers arriving later get a field, and a theme change repaints
 *  every field's colours.
 *  @param {ParentNode} root the subtree to arm
 *  @returns {boolean} whether the goo is armed on this document */
function mountGoo(root) {
  if (typeof window === "undefined") return false;
  if (document.documentElement.getAttribute("data-sl-theme") === "high-contrast") return false;
  if (gooArmed) return true;
  gooArmed = true;
  armGooFields(root);
  new MutationObserver((records) => {
    for (const r of records) for (const n of r.addedNodes) if (n instanceof Element) armGooFields(n);
  }).observe(root, { childList: true, subtree: true });
  new MutationObserver(repaintGooFields).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-sl-theme", "style"],
  });
  return true;
}

/** The hosts whose glass is cut to their own letterforms, guarded the way
 *  `lensed` and `gooFields` are: mount() may be called twice, and a host may
 *  leave the document. */
const glassTexts = new Map();
const GLASS_TEXT_HOSTS = ".sl-fx-glass--text";

/** Cut every declared host's glass to its own text.
 *
 *  Dynamically imported, like the lens and the goo: a surface with no glass
 *  text pays nothing for the module existing. There is no Tier 0 for this one
 *  — an uncut host is a rectangle of glass with an invisible label in it — so
 *  the modifier's CSS keeps the host transparent and lets its paired glow
 *  carry the text until the mask arrives.
 *  @param {ParentNode} root the subtree to arm
 *  @returns {Promise<void>} */
async function armGlassText(root) {
  if (typeof window === "undefined") return;
  let glassText;
  try { ({ glassText } = await import("./glass-text.js")); } catch { return; }
  const hosts = [];
  if (root instanceof Element && root.matches(GLASS_TEXT_HOSTS)) hosts.push(root);
  hosts.push(...root.querySelectorAll(GLASS_TEXT_HOSTS));
  for (const el of hosts) {
    if (glassTexts.has(el)) continue;
    const handle = glassText(el);
    if (handle) glassTexts.set(el, handle);
  }
}

/** Re-read every field's colours after a theme change.
 *
 *  A field is static once laid out, so nothing else would ever repaint it: a
 *  theme switch changes no box, fires no resize and moves no node. Its members'
 *  fills went on being the previous theme's, and on the light theme that put a
 *  ghost button's dark label on a substance still painted in the dark theme's
 *  raised surface — the label vanished. The observer that already watches
 *  data-sl-theme for the GPU tier's uniforms is the right place: one attribute,
 *  one read, every consumer of a token value updated from it.
 *  @returns {void} */
function repaintGooFields() {
  for (const handle of gooFields.values()) handle.measure(true);
}

/** Take every lens away, leaving the CSS material that glass.css paints.
 *  @returns {void} */
function dropLenses() {
  for (const handle of lensed.values()) handle.destroy();
  lensed.clear();
}

/** Follow the theme for as long as the page lives: a visitor can switch to
 *  high contrast — or away from it — long after mount() ran, and a kill switch
 *  that only reads its condition once is not a kill switch.
 *  @param {ParentNode} root the subtree mount() armed
 *  @returns {void} */
function watchLensTheme(root) {
  if (typeof MutationObserver !== "function") return;
  new MutationObserver(() => {
    if (lensRefused()) dropLenses();
    else armLenses(root);
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-sl-theme"] });
}

/** Release everything held for a host that has left the document: its lens
 *  (an <svg> in <body> with two PNG maps), its glass-text cut, its goo field
 *  (window listeners and observers), its seated glow and that glow's
 *  observer, its near-viewport watch and its specular state. Held strongly
 *  until now, so a long-lived app that re-renders glass leaked all of it.
 *  A host moved within one task is still connected here and keeps it all; one
 *  that comes back later is armed again as an added node.
 *  @returns {void} */
function pruneDetached() {
  for (const held of [lensed, glassTexts, gooFields]) {
    for (const [el, handle] of held) if (!el.isConnected) { handle.destroy(); held.delete(el); }
  }
  for (const [el, { watch, glow }] of glowWatches) {
    if (el.isConnected) continue;
    watch.disconnect();
    glow.remove();
    glowWatches.delete(el);
  }
  pruneDithers();
  prunePixels();
  for (const el of nearHosts) if (!el.isConnected) { nearObserver?.unobserve(el); nearHosts.delete(el); }
  for (const el of specHosts) {
    if (el.isConnected) continue;
    if (specState.has(el)) releaseSpec(el);
    specHosts.delete(el);
  }
}

/* ── pixel controls: the check, radio and switch change on a 3px lattice ─────
   components.css draws the final frame of all three with no script: a pixel
   tick, a pixel plus, a solid fill, a knob on steps(). This section only
   animates the way there. On a delegated `change` it builds, once and lazily,
   one clipped layer in the box or track: an opaque span in the control's OFF
   ground (CSS has already flipped to the final state, so the ground has to be
   repainted for the fill to be seen growing) holding a canvas that draws the
   fill cells (wipeFrame from the centre, or from the knob), then the tick cell
   by cell, or the radio's dot as one cell and then a plus. The cell lists and
   the timing maths are dither-core.js's, the same lists components.css is
   pinned to, so the last canvas frame is the CSS frame cell for cell; the
   layer is then hidden (.is-live off) and CSS shows through.
   The switch knob is CSS (steps(N) on --sl-motion-switch); the fill's origin
   follows it by running the same step maths on the same duration.
   Colours are never held here: the layer's `color` is the fill, its
   border-top-color the mark, read when a motion starts. Tier 1 only, and one
   shared rAF loop that sleeps when nothing moves. No outline, no blend, no
   opacity, no filter. */
const PIXEL_INPUTS = ".sl-check__input, .sl-radio__input, .sl-switch__input";
const PIXEL_BOXES = ".sl-check__box, .sl-radio__box, .sl-switch__track";
const PIXEL_CLASS = "sl-fx-pixel";
const PIXEL_DPR_CAP = 2;
/** Fallback durations in ms when the --sl-motion-* tokens do not resolve. */
const PIXEL_MS = { check: 180, radio: 150, switch: 260 };

const pixels = new Map();
const pixeling = new Set();
let pixelFrame = 0;
let pixelLast = 0;

const pixelEase = (p) => 1 - (1 - p) ** 3;

/** The control a change input paints, or null. The box or track is the
 *  input's next sibling in every kit control.
 *  @param {Element} input a check, radio or switch input */
function pixelHost(input) {
  const host = input.nextElementSibling;
  return host && host.matches(PIXEL_BOXES) ? host : null;
}

/** Build a control's layer once: the span and its canvas.
 *  @param {Element} host a check box, radio box or switch track
 *  @param {"check"|"radio"|"switch"} kind which control */
function ensurePixel(host, kind) {
  let d = pixels.get(host);
  if (d) return d;
  const layer = document.createElement("span");
  layer.className = PIXEL_CLASS;
  layer.setAttribute("aria-hidden", "true");
  const canvas = document.createElement("canvas");
  layer.append(canvas);
  host.prepend(layer);
  const grid = kind === "switch" ? { cols: SWITCH_CELLS.cols, rows: SWITCH_CELLS.rows } : { cols: CHECK_BOX_CELLS, rows: CHECK_BOX_CELLS };
  const cellPx = kind === "switch" ? CELL_PX : CONTROL_CELL_PX;
  d = { host, kind, layer, canvas, ctx: canvas.getContext("2d"), grid, cellPx, p: 0, dir: 0, ms: PIXEL_MS[kind], fill: "", on: "", dpr: 0, knob: 0, keep: false, settle: false };
  pixels.set(host, d);
  return d;
}

/** Size the canvas to whole cells at the capped device ratio. Assigning width
 *  clears it, so only when the ratio changed.
 *  @param {object} d a control's pixel state */
function sizePixel(d) {
  const dpr = Math.min(PIXEL_DPR_CAP, window.devicePixelRatio || 1);
  if (dpr === d.dpr) return;
  d.dpr = dpr;
  d.canvas.width = Math.round(d.grid.cols * d.cellPx * dpr);
  d.canvas.height = Math.round(d.grid.rows * d.cellPx * dpr);
}

/** Fill the given cells in the canvas's current fillStyle.
 *  @param {object} d a control's pixel state
 *  @param {Iterable<readonly [number, number]>} cells [column, row] pairs
 *  @param {number} [inset] cells added to both coordinates (a mark's margin in its box) */
function putPixelCells(d, cells, inset = 0) {
  const cell = d.cellPx * d.dpr;
  d.ctx.beginPath();
  for (const [ci, cj] of cells) {
    const i = ci + inset;
    const j = cj + inset;
    const x = Math.round(i * cell);
    const y = Math.round(j * cell);
    d.ctx.rect(x, y, Math.round((i + 1) * cell) - x, Math.round((j + 1) * cell) - y);
  }
  d.ctx.fill();
}

/** The covered cells of a row-major wipe frame as [column, row] pairs.
 *  @param {Uint8Array} frame wipeFrame output
 *  @param {number} cols grid width */
function coveredCells(frame, cols) {
  const out = [];
  for (let k = 0; k < frame.length; k++) if (frame[k]) out.push([k % cols, Math.floor(k / cols)]);
  return out;
}

/** Paint the control at progress `d.p` (0 = off, 1 = on) over its off ground.
 *  @param {object} d a control's pixel state */
function drawPixel(d) {
  const { ctx, canvas, grid, kind, p, dir } = d;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "transparent";
  ctx.fillStyle = d.fill;
  if (kind === "radio") {
    const f = dotFrame(p);
    if (f < 1) return;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "transparent";
    ctx.fillStyle = d.on;
    putPixelCells(d, DOT_FRAMES[f - 1], CHECK_INSET);
    return;
  }
  let origin = { x: grid.cols / 2, y: grid.rows / 2 };
  if (kind === "switch") {
    // The knob moves one way per direction: a reversal holds it where it is
    // and then carries it back, never jumping a step.
    const to = dir >= 0 ? knobCells(p, 0, 1) : knobCells(1 - p, 1, 0);
    d.knob = dir >= 0 ? Math.max(d.knob, to) : Math.min(d.knob, to);
    d.host.style.setProperty("--slfx-knob-x", `${d.knob * CELL_PX}px`);
    origin = { x: SWITCH_CELLS.inset + d.knob + SWITCH_CELLS.knob / 2, y: grid.rows / 2 };
  }
  const t = d.keep ? 1 : pixelEase(p);
  if (t >= 1) ctx.fillRect(0, 0, canvas.width, canvas.height);
  else if (t > 0) putPixelCells(d, coveredCells(wipeFrame(grid, origin, t), grid.cols));
  if (kind === "check") {
    const n = tickCount(p);
    if (n < 1) return;
    ctx.fillStyle = "transparent";
    ctx.fillStyle = d.on;
    putPixelCells(d, TICK_CELLS.slice(0, n), CHECK_INSET);
  }
}

/** One frame of every moving control; each leaves the loop at an end, where
 *  its layer is hidden so the CSS frame shows.
 *  @param {number} now the frame's timestamp */
function pixelStep(now) {
  pixelFrame = 0;
  const dt = Math.max(0, Math.min(64, now - pixelLast));
  pixelLast = now;
  for (const d of pixeling) {
    d.p = Math.min(1, Math.max(0, d.p + (d.dir * dt) / d.ms));
    drawPixel(d);
    if (d.p !== (d.dir > 0 ? 1 : 0)) d.settle = false;
    else if (!d.settle) d.settle = true; // the last frame is rendered once before the layer goes: the knob's var must be seen at its seat, or CSS starts a transition from the frame before
    else {
      pixeling.delete(d);
      d.dir = 0;
      d.keep = false;
      d.settle = false;
      d.host.style.removeProperty("--slfx-knob-x");
      d.layer.classList.remove("is-live");
    }
  }
  if (pixeling.size) pixelFrame = requestAnimationFrame(pixelStep);
}

/** Start a control moving toward on or off, from wherever it stands: from
 *  where a live motion has got to, else from the state it was painted in, which
 *  is the opposite of where it is going unless `mixed` (it was a filled
 *  indeterminate box: the fill stays, only the mark changes).
 *  @param {Element} host the box or track
 *  @param {boolean} on the state it is going to
 *  @param {boolean} [mixed] it was painted as the filled mixed state */
function movePixel(host, on, mixed = false) {
  const kind = host.matches(".sl-check__box") ? "check" : host.matches(".sl-radio__box") ? "radio" : "switch";
  const d = ensurePixel(host, kind);
  const cs = getComputedStyle(host);
  d.ms = motionMs(cs, `--sl-motion-${kind}`, PIXEL_MS[kind]);
  const ls = getComputedStyle(d.layer);
  d.fill = ls.color;
  d.on = ls.borderTopColor;
  if (!pixeling.has(d)) {
    d.p = mixed ? TICK_START : on ? 0 : 1;
    d.keep = mixed && on;
    d.knob = on ? 0 : SWITCH_CELLS.steps;
  }
  sizePixel(d);
  d.dir = on ? 1 : -1;
  d.layer.classList.add("is-live");
  drawPixel(d);
  pixeling.add(d);
  if (!pixelFrame) {
    pixelLast = performance.now();
    pixelFrame = requestAnimationFrame(pixelStep);
  }
}

/** The radios of `input`'s group: the same name in the same form, a formless
 *  group included (and a radio tied to a form by its form= attribute).
 *  @param {HTMLInputElement} input a radio
 *  @returns {HTMLInputElement[]} the group */
function radioGroup(input) {
  const pool = input.form ? input.form.elements : document.querySelectorAll(".sl-radio__input");
  return [...pool].filter((r) => r.type === "radio" && r.name === input.name && r.form === input.form);
}

/** What was painted just before a user's change: per form (or document) and
 *  radio name, the checked radio; per input, whether a check was mixed. A
 *  `change` fires only on the new radio, and by then the old one is already
 *  unchecked, so the old one has to be written down first. It is written
 *  again on every pointerdown and keydown, in the capture phase, which both
 *  come right before the `change` they cause, so a programmatic change or a
 *  form.reset() in between can never leave it stale. */
const radioSeen = new WeakMap();
const wasMixed = new WeakMap();

/** Write down the painted state of the control a pointer or key is about to change.
 *  @param {Event} ev a capture-phase pointerdown or keydown */
function seedPixel(ev) {
  const t = ev.target instanceof Element ? ev.target : null;
  const input = t?.matches?.(PIXEL_INPUTS) ? t : t?.closest?.(".sl-check, .sl-radio, .sl-switch")?.querySelector(PIXEL_INPUTS);
  if (!input) return;
  if (input.matches(".sl-check__input")) wasMixed.set(input, input.indeterminate);
  if (!input.matches(".sl-radio__input") || !input.name) return;
  const scope = input.form ?? document;
  let rec = radioSeen.get(scope);
  if (!rec) radioSeen.set(scope, (rec = new Map()));
  rec.set(input.name, radioGroup(input).find((r) => r.checked) ?? null);
}

/** A delegated change on a check, radio or switch input: animate it, and in a
 *  radio group animate the one it replaced too, if the record written before
 *  this change says it was on. Reduced motion, high contrast and an opt-out
 *  region start nothing; the CSS frame stands.
 *  @param {Event} ev a delegated change */
function onPixelChange(ev) {
  const input = ev.target instanceof Element ? ev.target : null;
  if (!input || !input.matches(PIXEL_INPUTS) || input.disabled) return;
  const rec = input.matches(".sl-radio__input") && input.name ? radioSeen.get(input.form ?? document) : null;
  const prev = rec?.get(input.name);
  rec?.delete(input.name);
  const mixed = wasMixed.get(input) === true;
  wasMixed.delete(input);
  if (suppressed() || window.matchMedia(REDUCE).matches || input.closest('[data-sl-fx="off"]')) return;
  const host = pixelHost(input);
  if (!host) return;
  movePixel(host, input.checked, mixed);
  if (prev && prev !== input && prev.isConnected && !prev.checked && prev.form === input.form) {
    const old = pixelHost(prev);
    if (old) movePixel(old, false);
  }
}

/** Release what a control that left the document held.
 *  @returns {void} */
function prunePixels() {
  for (const [host, d] of pixels) {
    if (host.isConnected) continue;
    pixeling.delete(d);
    pixels.delete(host);
  }
}
/* ── end pixel controls ── */

/* ── dither: the hover is an ordered-dither wipe ─────────────────────────────
   A button or icon button answers the pointer with a fill that grows through
   an 8x8 Bayer threshold from the point the pointer crossed the edge, and
   drains toward the point it left (dither-core.js owns the maths). The fill is
   drawn as CELL_PX squares on a canvas inside one injected span that sits over
   the glass tint, at z-index -2 after it in DOM order (and over the frost):
   the glass shine, live light, the press light, the ring and the label all
   paint over it. The span rounds with its host and
   clips; the canvas draws fill cells only, never an outline, so the control's
   own border stays the edge and a round button keeps a smooth circle.
   The layer is built lazily, on the first pointer entry into a host, and one
   shared rAF loop runs only while a wipe is moving. Colour is never held here:
   the fill is --slfx-dither-fill, read through the layer's own `color`, and the
   label flips through .is-dithered (effects.css) once the wipe is half done.
   Tier 1 only: under reduced motion mount() never arms this, and the control's
   declared CSS hover is what stands. */
const DITHER_HOSTS = ".sl-btn, .sl-icon-btn";
const DITHER_CLASS = "sl-fx-dither";
/** The label flips once this much of the wipe has covered the control. */
const DITHER_FLIP = 0.5;
/** Device pixels per CSS pixel are capped here, as live-light.js does. */
const DITHER_DPR_CAP = 2;
/** Fallback durations, in ms, when the motion tokens do not resolve. */
const DITHER_MS = { in: 260, out: 320 };

/** Controls with a layer, and the wipes in flight: host -> its state. */
const dithers = new Map();
const dithering = new Set();
let ditherFrame = 0;
let ditherLast = 0;

/** A motion token's duration in ms, or the fallback when it is not a time.
 *  @param {CSSStyleDeclaration} cs the host's computed style
 *  @param {string} name the custom property
 *  @param {number} fallback ms */
function motionMs(cs, name, fallback) {
  const v = cs.getPropertyValue(name).trim();
  const n = parseFloat(v);
  if (!Number.isFinite(n)) return fallback;
  return v.endsWith("ms") ? n : v.endsWith("s") ? n * 1000 : fallback;
}

/** The ease a wipe's progress runs through: fast first, long tail, both ways.
 *  @param {number} p linear progress 0..1 */
const ditherEase = (p) => 1 - (1 - p) ** 3;

/** The fill, read off the layer's own colour (--slfx-dither-fill).
 *  @param {object} d a control's dither state */
function readDitherFill(d) {
  d.fill = getComputedStyle(d.layer).color;
}

/** Set the canvas's fill. Resizing a canvas resets its state, so this runs on
 *  every draw. A value the canvas cannot parse leaves the transparent keyword
 *  it was reset to, so nothing paints rather than the previous colour.
 *  @param {object} d a control's dither state */
function applyDitherFill(d) {
  d.ctx.fillStyle = "transparent";
  d.ctx.fillStyle = d.fill;
}

let ditherThemeWatch = null;

/** A theme switch re-reads every live fill and repaints it, once, for the page.
 *  @returns {void} */
function watchDitherTheme() {
  if (ditherThemeWatch || typeof MutationObserver !== "function") return;
  ditherThemeWatch = new MutationObserver(() => {
    for (const d of dithers.values()) {
      if (d.p <= 0) continue;
      readDitherFill(d);
      drawDither(d);
    }
  });
  ditherThemeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ["data-sl-theme"] });
}

/** Build a control's layer, once: the span and its canvas, after the glass
 *  tint (else the backdrop) when there is one, so the fill is over the frost
 *  and the tint and under the shine.
 *  @param {Element} host a button or icon button
 *  @returns {object} the control's dither state */
function ensureDither(host) {
  let d = dithers.get(host);
  if (d) return d;
  const layer = document.createElement("span");
  layer.className = DITHER_CLASS;
  layer.setAttribute("aria-hidden", "true");
  const canvas = document.createElement("canvas");
  layer.append(canvas);
  const under = host.querySelector(":scope > .sl-glass__tint") ?? host.querySelector(":scope > .sl-glass__backdrop");
  if (under) under.after(layer);
  else host.prepend(layer);
  d = { host, layer, canvas, ctx: canvas.getContext("2d"), p: 0, dir: 0, origin: { x: 0, y: 0 }, grid: { cols: 0, rows: 0 }, fill: "", msIn: DITHER_MS.in, msOut: DITHER_MS.out, flipped: false, w: 0, h: 0, dpr: 1 };
  dithers.set(host, d);
  watchDitherTheme();
  return d;
}

/** Size the canvas to the host's padding box at the capped device ratio, and
 *  the grid to whole cells. Assigning width clears the canvas, so it only
 *  happens when something changed.
 *  @param {object} d a control's dither state */
function sizeDither(d) {
  const w = d.host.clientWidth;
  const h = d.host.clientHeight;
  const dpr = Math.min(DITHER_DPR_CAP, window.devicePixelRatio || 1);
  if (w === d.w && h === d.h && dpr === d.dpr) return;
  Object.assign(d, { w, h, dpr });
  d.canvas.width = Math.max(1, Math.round(w * dpr));
  d.canvas.height = Math.max(1, Math.round(h * dpr));
  d.grid = { cols: Math.ceil(w / CELL_PX), rows: Math.ceil(h / CELL_PX) };
}

/** Paint the wipe at its current progress: the covered cells as rects at
 *  device resolution, in the fill read when the wipe began. Toggles the label
 *  flip when the covered share crosses DITHER_FLIP.
 *  @param {object} d a control's dither state */
function drawDither(d) {
  const { ctx, canvas, grid, dpr } = d;
  const t = ditherEase(d.p);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  applyDitherFill(d);
  if (t >= 1) {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  } else if (t > 0) {
    const covered = wipeFrame(grid, d.origin, t);
    const cell = CELL_PX * dpr;
    ctx.beginPath();
    for (let j = 0; j < grid.rows; j++) {
      for (let i = 0; i < grid.cols; i++) {
        if (!covered[j * grid.cols + i]) continue;
        const x = Math.round(i * cell);
        const y = Math.round(j * cell);
        ctx.rect(x, y, Math.round((i + 1) * cell) - x, Math.round((j + 1) * cell) - y);
      }
    }
    ctx.fill();
  }
  const flipped = t >= DITHER_FLIP;
  if (flipped !== d.flipped) {
    d.flipped = flipped;
    d.host.classList.toggle("is-dithered", flipped);
  }
}

/** One frame of every moving wipe. A wipe leaves the loop at either end (full,
 *  or drained and cleared); the loop sleeps once none is left.
 *  @param {number} now the frame's timestamp */
function ditherStep(now) {
  ditherFrame = 0;
  const dt = Math.max(0, Math.min(64, now - ditherLast));
  ditherLast = now;
  for (const d of dithering) {
    sizeDither(d);
    d.p = Math.min(1, Math.max(0, d.p + (d.dir * dt) / (d.dir > 0 ? d.msIn : d.msOut)));
    drawDither(d);
    if (d.p === (d.dir > 0 ? 1 : 0)) {
      dithering.delete(d);
      d.dir = 0;
    }
  }
  if (dithering.size) ditherFrame = requestAnimationFrame(ditherStep);
}

function ditherWake() {
  if (ditherFrame) return;
  ditherLast = performance.now();
  ditherFrame = requestAnimationFrame(ditherStep);
}

/** A pointer crossing a control's edge. In: the layer is built if need be, the
 *  fill and durations are read, the wipe's origin is the crossing, and it runs
 *  forward from wherever the last one stopped. Out: the origin moves to where
 *  the pointer left and the wipe runs back. Touch has no hover to answer, and
 *  reduced motion and high contrast start no wipe: the declared hover stands.
 *  @param {PointerEvent} ev a delegated pointerover or pointerout */
function onDitherCross(ev) {
  if (ev.pointerType === "touch") return;
  const host = ev.target instanceof Element ? ev.target.closest(DITHER_HOSTS) : null;
  if (!host || (ev.relatedTarget instanceof Node && host.contains(ev.relatedTarget))) return;
  const entering = ev.type === "pointerover";
  // Only the way in is gated. A wipe already running always drains, even if
  // reduced motion came on mid-hover (effects.css hides the layer there). Not
  // even the motion derogation (mount's ignoreReducedMotion) lets one start:
  // that lets the glitch and the halftone move, and the wipe is neither.
  if (entering && (suppressed() || window.matchMedia(REDUCE).matches)) return;
  if (entering && host.matches(':disabled, [aria-disabled="true"]')) return;
  if (entering && host.closest('[data-sl-fx="off"]')) return;
  const d = entering ? ensureDither(host) : dithers.get(host);
  if (!d) return;
  const r = host.getBoundingClientRect();
  if (!r.width || !r.height) return;
  sizeDither(d);
  d.origin = { x: (ev.clientX - r.left) / CELL_PX, y: (ev.clientY - r.top) / CELL_PX };
  if (entering) {
    const cs = getComputedStyle(host);
    d.msIn = motionMs(cs, "--sl-motion-dither-in", DITHER_MS.in);
    d.msOut = motionMs(cs, "--sl-motion-dither-out", DITHER_MS.out);
    readDitherFill(d);
  }
  d.dir = entering ? 1 : -1;
  dithering.add(d);
  ditherWake();
}

/** Release what a control that left the document held: its state and its place
 *  in the loop. The layer goes with the host.
 *  @returns {void} */
function pruneDithers() {
  for (const [host, d] of dithers) {
    if (host.isConnected) continue;
    dithering.delete(d);
    dithers.delete(host);
  }
}

function injectGlassLayers(root) {
  if (root instanceof Element && root.matches(GLASS_HOSTS)) injectGlass(root);
  for (const el of root.querySelectorAll(GLASS_HOSTS)) injectGlass(el);
}

function injectGlows(root) {
  if (root instanceof Element && root.matches(GLOW_HOSTS)) injectGlow(root);
  for (const el of root.querySelectorAll(GLOW_HOSTS)) injectGlow(el);
}

/** The conditions under which no effect may run, re-read at call time rather
 *  than sampled at mount: a visitor can switch theme, or turn on reduced
 *  motion, long after the page loaded. */
/* ── near the viewport ───────────────────────────────────────────────────────
   An effect on an element nobody can see is work nobody asked for, and on a
   long page most elements are that. The expensive ones are the glass's
   backdrop-filter — a blur of whatever is behind it, recomputed as the page
   moves — and every running animation, which keeps ticking off-screen.

   So Tier 1 marks what is near enough to matter and effects.css stands the
   rest down. `near` rather than `visible`: the margin arms an element before
   it arrives, because a pane that blurs in as it crosses the fold is worse
   than one that was always blurred.

   SUBTRACTIVE, and that is the whole design. Tier 0 has no JavaScript to write
   the attribute, so the stand-down rules key on `[data-sl-fx="on"]` too — a
   page that never mounts keeps its full material, and only a page that armed
   Tier 1 gets the economy. The alternative (require the attribute to paint)
   would have turned the glass off entirely wherever this module is absent. */
const NEAR_MARGIN = "200px";
const NEAR_HOSTS = ".sl-fx-glass, .sl-fx-aurora";
let nearObserver = null;
/** What nearObserver watches: it holds each target strongly until unobserved. */
const nearHosts = new Set();

/** Watch an element and mark it while it is near the viewport.
 *  @param {ParentNode} root the subtree to watch
 *  @returns {void} */
function watchNear(root) {
  if (typeof IntersectionObserver !== "function") return;
  if (!nearObserver) {
    nearObserver = new IntersectionObserver(
      (entries) => {
        for (const e of entries) e.target.toggleAttribute("data-sl-fx-near", e.isIntersecting);
      },
      { rootMargin: NEAR_MARGIN },
    );
  }
  const hosts = [];
  if (root instanceof Element && root.matches(NEAR_HOSTS)) hosts.push(root);
  hosts.push(...root.querySelectorAll(NEAR_HOSTS));
  for (const el of hosts) {
    // FAIL OPEN. Marked near before it is observed, so the observer's job is
    // only ever to TAKE the mark away. Where the API is absent, or present and
    // never delivering — a background tab, a headless capture, a harness that
    // suppresses it — every host keeps its full material, which is the state
    // this layer promises is correct on its own. The economy is an
    // improvement on that state, never a precondition for it.
    el.toggleAttribute("data-sl-fx-near", true);
    nearObserver.observe(el);
    nearHosts.add(el);
  }
}

/** Whether this document has taken the reduced-motion derogation.
 *
 *  A product may have a documented reason to animate for a visitor reporting
 *  the preference. sondalab.eu is the case this exists for: Windows reports
 *  `reduce` by default under "adjust for best performance", and that site's
 *  owner ruled its glitch is brand rather than decoration. The derogation is
 *  the consumer's to take and to justify — mount() only takes it when asked,
 *  by a name a reader cannot mistake for anything else.
 *
 *  It reaches the GLITCH alone (effects.css keeps every other family's motion
 *  inside the media query, where nothing can overrule it), and it never
 *  reaches high contrast, which is a different requirement with a different
 *  reason: there the point is that what is behind a surface must not interfere
 *  with what is on it.
 *  @returns {boolean} */
function motionDerogated() {
  if (typeof document === "undefined") return false;
  return document.documentElement.getAttribute("data-sl-fx-motion") === "always";
}

function suppressed() {
  if (typeof document === "undefined") return true;
  if (document.documentElement.getAttribute("data-sl-theme") === "high-contrast") return true;
  return window.matchMedia(REDUCE).matches && !motionDerogated();
}

/* ── public API ──────────────────────────────────────────────────────────────*/

/**
 * Arm Tier 1: pointer-driven specular light, finite glitch bursts, animated aurora.
 * Refuses under reduced motion and under high contrast — Tier 0 is the default
 * state and stays correct on its own.
 *
 * `ignoreReducedMotion` is a derogation, not a convenience, and a consumer
 * takes it on their own evidence. It arms Tier 1 for a visitor reporting
 * `prefers-reduced-motion: reduce` and lets the GLITCH and the HALFTONE's gather-and-float animate for them; every
 * other family's motion stays inside the media query in effects.css, where
 * nothing can overrule it. It has no effect on high contrast, which refuses
 * regardless. The case it exists for: sondalab.eu, where Windows reports the
 * preference by default under "adjust for best performance" and the owner
 * ruled the glitch is brand rather than decoration.
 *
 * @param {Document|Element} root
 * @param {{ignoreReducedMotion?: boolean}} [options]
 * @returns {boolean} whether Tier 1 was armed
 */
export function mount(root = document, { ignoreReducedMotion = false, bursts = 1 } = {}) {
  if (armed) return true;
  if (typeof window === "undefined") return false;
  const html = document.documentElement;
  if (html.getAttribute("data-sl-theme") === "high-contrast") return false;
  // Written before anything reads it, and written by the caller's request
  // alone: effects.css keys the glitch's reduced-motion removal off this
  // attribute, so the CSS and the JavaScript decline together or not at all.
  if (ignoreReducedMotion) html.setAttribute("data-sl-fx-motion", "always");
  burstBudget = Math.max(1, Math.floor(bursts));
  // Before the motion guard, and deliberately: the goo animates nothing, so
  // refusing it for a motion preference would take away material rather than
  // movement. mountGoo() owns its own observers and its own armed flag.
  mountGoo(root);
  // The halftone goes with the goo, on the goo's argument: the picture is
  // content. Under reduced motion it draws its settled frame, not nothing.
  armHalftone(root);
  // The lens goes with it, and for the reason lensRefused() has argued in its
  // own comment since it was written: a lens is not motion. It is a static
  // property of a surface — it does not animate, and it does not redraw on
  // scroll — and a visitor who asked for less movement did not ask for less
  // material. The comment said so; the call site said otherwise, because it
  // sat after this guard. armLenses() still refuses high contrast on its own,
  // which is a different requirement: there the point is that what is behind a
  // surface must not interfere with what is on it.
  //
  // injectGlassLayers goes with it, and it is the larger half of the same
  // defect: those children ARE the glass material — glass.css puts the
  // backdrop-filter on .sl-glass__backdrop, split from the tint because
  // opacity on the filtering element throttles it — so a visitor reporting a
  // motion preference was getting no glass at all, not merely no lens. The
  // layers carry no animation of their own; what animates on a pane is in
  // effects.css, inside the media query, where it stays.
  //
  // Late arrivals need the material too, so the observer that gives it to them
  // is set up here rather than with Tier 1's: a page that refuses below still
  // has to dress a card rendered after mount().
  injectGlassLayers(root);
  armLenses(root);
  // And the letterform cut goes with them, for the third time in the same
  // argument: a mask is not motion. It is the shape the material has.
  armGlassText(root);
  watchLensTheme(root);
  new MutationObserver((records) => {
    let removed = false;
    for (const r of records) {
      removed ||= r.removedNodes.length > 0;
      for (const n of r.addedNodes) if (n instanceof Element) {
        injectGlassLayers(n);
        armLenses(n);
        armGlassText(n);
      }
    }
    if (removed) pruneDetached();
  }).observe(root, { childList: true, subtree: true });
  if (window.matchMedia(REDUCE).matches && !ignoreReducedMotion) return false;

  // Before data-sl-fx="on" goes up: the arc's sweep and its bloom's sweep
  // start on the same frame, so the two stay in phase. Elements that arrive
  // later get theirs as they arrive.
  injectGlows(root);
  watchNear(root);
  collectSpecHosts(root);
  armLiveLight(root);
  // Tier 1's own late arrivals: the glow is a moving layer and the viewport
  // gate only matters where something is running, so both belong on this side
  // of the guard. The material's observer is above.
  new MutationObserver((records) => {
    for (const r of records) for (const n of r.addedNodes) if (n instanceof Element) {
      injectGlows(n);
      watchNear(n);
      collectSpecHosts(n);
    }
    // Content arriving or leaving can move a box under a resting pointer
    // (a streamed card pushing the wall down): the specular loop looks again.
    onSpecLayout();
  }).observe(root, { childList: true, subtree: true });

  // One delegated listener for the whole subtree, not one per element.
  root.addEventListener("pointermove", onPointerMove, { passive: true });
  window.addEventListener("pointerout", onSpecOut, { passive: true });
  window.addEventListener("scroll", onSpecLayout, { capture: true, passive: true });
  window.addEventListener("resize", onSpecLayout, { passive: true });
  root.addEventListener("change", onPixelChange);
  root.addEventListener("pointerdown", seedPixel, { capture: true, passive: true });
  root.addEventListener("keydown", seedPixel, { capture: true, passive: true });
  root.addEventListener("pointerover", onDitherCross, { passive: true });
  root.addEventListener("pointerout", onDitherCross, { passive: true });
  root.addEventListener("pointerdown", onPressDown, { passive: true });
  window.addEventListener("pointerup", pressEnd, { passive: true });
  window.addEventListener("pointercancel", pressEnd, { passive: true });
  root.addEventListener("keydown", onPressKey);
  root.addEventListener("keyup", onPressKey);

  html.setAttribute("data-sl-fx", "on");
  armed = true;
  return true;
}

/**
 * One finite glitch burst of randomised length. A burst requested while another
 * runs is dropped, not queued: a queue produces a stutter train, which reads as
 * breakage rather than as signal. A detached element is refused.
 * @param {Element} el an element carrying .sl-fx-glitch, still in the document
 */
export function burst(el, { burstMin = 200, burstMax = 500 } = {}) {
  // isConnected: a caller may take the element out of the document between the
  // event and the burst, and a timer toggling a class on a detached node would
  // otherwise run for nothing. The guard is carried over from the prior art,
  // where it exists because of a real defect. blocked is read into a local
  // rather than called inline so the guard clause stays a single paren-free
  // condition — scripts/effects-js.test.mjs locates it by that shape.
  const blocked = suppressed();
  if (!armed || burstsRunning >= burstBudget || blocked || !el?.isConnected) return;
  burstsRunning++;
  randomizeGhosts(el);
  el.classList.add("is-glitching");
  setTimeout(() => {
    el.classList.remove("is-glitching");
    burstsRunning--;
  }, Math.min(CAP_MS, rand(burstMin, burstMax)));
}

/**
 * Burst, wait, burst again — an ambient glitch on one element, with a random
 * pause between each. The element re-resolving every so often is a different
 * statement from one that glitches once: it says the thing is live, not that
 * something just happened.
 *
 * Returns a stop function. Callers who never call it are not leaking a timer
 * for the life of the page: the loop checks `isConnected` and lets itself go
 * when the element leaves the document, which is the case that actually bites
 * (a scene field drops its fallback picture once its tube is up).
 *
 * @param {Element} el an element carrying .sl-fx-glitch
 * @param {{waitMin?: number, waitMax?: number, burstMin?: number, burstMax?: number}} [options]
 * @returns {() => void} stop
 */
export function autoBurst(el, { waitMin = 6000, waitMax = 11000, burstMin = 200, burstMax = 500 } = {}) {
  let timer = 0;
  let stopped = false;
  const loop = () => {
    timer = setTimeout(() => {
      if (stopped || !el?.isConnected) return;
      burst(el, { burstMin, burstMax });
      loop();
    }, rand(waitMin, waitMax));
  };
  loop();
  return () => { stopped = true; clearTimeout(timer); };
}

/**
 * While the pointer is over one element, glitch another in bursts separated by
 * random pauses; stop the moment it leaves.
 *
 * The two elements are separate on purpose: what the pointer is over is often
 * a whole row or a link, and what should re-resolve is the name inside it.
 *
 * This does NOT go through burst(): a hover is a direct answer to the pointer,
 * and dropping it because something ambient happens to be mid-burst would make
 * the surface feel unresponsive at exactly the moment the visitor asked it for
 * something. It is the one path allowed past the budget, and it is bounded by
 * the pointer being there.
 *
 * @param {Element} hoverEl what the pointer is over
 * @param {Element} targetEl what re-resolves — may be the same element
 * @param {{burstMin?: number, burstMax?: number, pauseMin?: number, pauseMax?: number}} [options]
 * @returns {() => void} stop, which also releases the listeners
 */
export function hoverBurst(hoverEl, targetEl, { burstMin = 200, burstMax = 500, pauseMin = 500, pauseMax = 1600 } = {}) {
  let active = false;
  const timers = [];
  const clear = () => { while (timers.length) clearTimeout(timers.pop()); };
  const cycle = () => {
    if (!active || suppressed() || !targetEl?.isConnected) return;
    randomizeGhosts(targetEl);
    targetEl.classList.add("is-glitching");
    timers.push(setTimeout(() => {
      targetEl.classList.remove("is-glitching");
      if (active) timers.push(setTimeout(cycle, rand(pauseMin, pauseMax)));
    }, Math.min(CAP_MS, rand(burstMin, burstMax))));
  };
  const enter = () => { if (active || !armed) return; active = true; cycle(); };
  const leave = () => { active = false; clear(); targetEl.classList.remove("is-glitching"); };
  hoverEl.addEventListener("pointerenter", enter, { passive: true });
  hoverEl.addEventListener("pointerleave", leave, { passive: true });
  return () => {
    leave();
    hoverEl.removeEventListener("pointerenter", enter);
    hoverEl.removeEventListener("pointerleave", leave);
  };
}

/**
 * Decode-scramble a short label: cycle random glyphs and lock the real
 * characters in left to right, terminal-readout style. Distinct from the ghost
 * glitch. Under high contrast or reduced motion the cycling is suppressed, but
 * the final value still lands immediately — dropping it would replace absent
 * motion with absent information, the same failure the aurora fix closes.
 * Reads its target from data-num so repeats stay correct.
 * @param {HTMLElement} el
 */
export function scramble(el, { duration = 520, tick = 45 } = {}) {
  if (!el || el.dataset.scrambling === "1") return;
  const final = el.dataset.num ?? el.textContent ?? "";
  if (suppressed()) {
    el.textContent = final;
    return;
  }
  const n = final.length;
  const glyphs = "0123456789#/\\<>";
  el.dataset.scrambling = "1";
  const start = performance.now();
  const iv = setInterval(() => {
    const t = performance.now() - start;
    if (t >= duration) {
      clearInterval(iv);
      el.textContent = final;
      delete el.dataset.scrambling;
      return;
    }
    const locked = Math.floor((t / duration) * n);
    let out = "";
    for (let i = 0; i < n; i++) out += i < locked ? final[i] : glyphs[(Math.random() * glyphs.length) | 0];
    el.textContent = out;
  }, tick);
}

/**
 * Grant the one ambient aurora. Most recent request wins; the previous element
 * falls back to its resting ring. Busy auroras do not go through here — they are
 * bound to real work and are exempt from the cap.
 * @param {Element} el
 */
export function grantLive(el) {
  if (liveEl && liveEl !== el) liveEl.removeAttribute("data-sl-fx-live");
  // liveEl only remembers a grant THIS function handed out. A page can also
  // write data-sl-fx-live="on" straight into its markup — a grant this
  // function never issued and so cannot know about through liveEl alone — so
  // the query is needed IN ADDITION: it finds every element currently
  // holding the attribute, wherever it came from, and revokes all but `el`.
  if (typeof document !== "undefined") {
    for (const prior of document.querySelectorAll('[data-sl-fx-live="on"]')) {
      // A grant inside a data-sl-fx="off" subtree is switched off, not
      // competing for the one-per-document budget — the ambient-budget fence
      // (scripts/effects-showcase.test.mjs, offRegionExtents) excludes
      // off-region grants from the count for exactly that reason. Revoking
      // it would be taking something from a page that was not asking.
      if (prior !== el && !prior.closest('[data-sl-fx="off"]')) prior.removeAttribute("data-sl-fx-live");
    }
  }
  liveEl = el;
  el.setAttribute("data-sl-fx-live", "on");
}

/** Revoke the ambient aurora from `el` if it currently holds it. */
export function revokeLive(el) {
  if (liveEl !== el) return;
  el.removeAttribute("data-sl-fx-live");
  liveEl = null;
}

/**
 * Tier 2 is retired (2026-09-24, re-scope R1). It drew the glass edge as a
 * WebGPU chrome that reflected a moving environment; the glass edge is now the
 * specular arc, drawn in CSS from what mount() writes, and there is nothing
 * left for a GPU to draw on it. The export stays because consumers call it
 * (sondalab-site does), and because the re-scope's Signature slice brings a
 * WebGPU tier back for a different job. Until then it resolves false, which is
 * the answer it always gave a browser without WebGPU — so every caller already
 * handles it.
 * @returns {Promise<boolean>} always false
 */
export async function mountGpu() {
  return false;
}
