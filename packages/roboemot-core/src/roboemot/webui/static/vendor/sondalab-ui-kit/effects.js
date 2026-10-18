/* @sondalab/ui-kit — effects.js
   Hand-authored Tier 1 driver for effects.css: the press light and its spill,
   the glass's specular edge, the
   liquid's silhouette, the glitch's timing and the aurora's one-per-document
   grant.
   Contract: docs/specs/2026-09-08-effects-layer-design.md

   Importing this module does nothing. mount() is explicit, which is what makes
   the IDE opt-in possible: a Theia surface that never calls it pays nothing.

   COLOUR: sl-audit reads CSS only, so a literal in this file would pass the gate
   while breaking the design-system contract. Every hue is read from a token with
   getComputedStyle and re-read on theme change. scripts/effects-js.test.mjs
   enforces the absence of literals. */

import { cornerPx, roundedRectPath, GOO_CONTRAST, GOO_INTERCEPT } from "./geom.js";
// A neutral (grey wash, ink label) has no hue to glow with: the ink pool then
// keeps the CSS fallback, the accent. The notation read lives in colour.js, and
// so does the paint-and-read that tells a light ground from a dark one.
import { isChromatic, toSrgb } from "./colour.js";

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

/* ── liquid: the silhouette goes gooey ──────────────────────────────────────
   The reference (libraries.dev/gooey, liquid-gooey) never filters the UI: a
   SILHOUETTE layer below the content carries the goo — one plain shape per
   item, run through blur + alpha-contrast so touching pieces merge with a
   rounded neck — and "your real DOM rides crisp on top". Its Metal entry does
   the same by default (metal-fx v2 `applyTo: "ring"`: the bend is a vector
   outline; "button content and glow stay rigid"). The displacement filter
   this replaced was a lens: it refracted the glyphs and re-rasterised the
   element, which is the text distortion and the pixelated edge the owner saw.

   So the liquid here is a shape, not a filter over pixels. effects.js keeps
   ONE hidden <svg class="sl-fx-liquid__skin"> and, while the pointer shapes
   an element, parents it into that element at z-index -2: above the host's
   own background and border (a negative z-index child paints over both, and
   under the content — the aurora's bloom already relies on this), below the
   text. The skin draws the host's silhouette — its rounded box, with the
   radii read from its own computed style — plus a DROP: a circle whose
   centre sits just inside the outline at the point nearest the pointer, so
   it protrudes by a few pixels. Box and drop go through the goo filter
   (Gaussian blur, then an alpha slope of 18 about 0.5 — the reference's own
   pair), which welds the drop to the box with a soft neck; the crisp box is
   drawn again unfiltered on top of that, so the goo can only ADD the swell
   and never rounds a corner the component declared. A second filter turns
   the merged alpha into paint: a flood of the host's own background inside
   it, and a band one border-width wide in the host's own border colour along
   its outline, so the ring bends with the swell. Both colours are the host's
   computed values, re-read every frame — hover and active states are the
   host's, and the skin follows them. Meanwhile the host's own border is
   hidden by a transparent border-image (effects.css), never by border-color:
   the computed border colour stays the truth for the skin to read.

   Nothing inside the element is ever filtered, the host's border-radius is
   never written, and the skin is removed the moment the spring settles. */

const NS = "http://www.w3.org/2000/svg";
/** What the pointer may shape. The gooey belongs to a ROUND silhouette:
 *  welded onto the long side of a text button the drop reads as a blister on
 *  a box, and the owner said so. The gate is an explicit modifier written in
 *  the markup — `--drop` for a round icon-only button (the full swell),
 *  and nothing else — never a measured radius: a
 *  geometric heuristic misfires the moment a component re-parametrises its
 *  own --sl-radius-*, and not breaking under that is a promise this layer
 *  has made twice. A bare .sl-fx-liquid is inert at this tier. */
const LIQUID_HOSTS = ".sl-fx-liquid.sl-fx-liquid--drop";
/** The goo: blur sigma and alpha slope. liquid-gooey's defaults are 6 and
 *  18; the sigma is smaller here because the swell is a few pixels and its
 *  neck has to stay in proportion — "molto leggero". The slope itself is
 *  geom.js's: one house goo, shared with the field family, whose weld
 *  arithmetic is derived from the same pair. */
const GOO_BLUR = 4;
/** The chrome's smooth-union radius (effects-shader.js, `smin`) that matches
 *  this goo's neck; passed to the GPU tier so the two silhouettes agree. */
let skin = null;
let skinParts = null;

function buildSkin() {
  if (skin) return;
  skin = document.createElementNS(NS, "svg");
  skin.setAttribute("class", "sl-fx-liquid__skin");
  skin.setAttribute("aria-hidden", "true");
  // sRGB on both filters: under the default linearRGB the alpha slope would
  // be applied to a gamma-decoded alpha, moving the goo's edge.
  const srgb = 'color-interpolation-filters="sRGB"';
  // Shapes are drawn in currentColor — only their ALPHA is used: the second
  // filter floods the merged silhouette with the host's own colours.
  skin.innerHTML = `
    <defs>
      <filter id="slfx-goo" filterUnits="userSpaceOnUse" ${srgb}>
        <feGaussianBlur in="SourceGraphic" stdDeviation="${GOO_BLUR}" />
        <feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 ${GOO_CONTRAST} ${GOO_INTERCEPT}" />
      </filter>
      <filter id="slfx-skin" filterUnits="userSpaceOnUse" ${srgb}>
        <feMorphology in="SourceAlpha" operator="erode" radius="1" result="inner" />
        <feComposite in="SourceAlpha" in2="inner" operator="out" result="band" />
        <feFlood result="fill-colour" />
        <feComposite in="fill-colour" in2="SourceAlpha" operator="in" result="fill" />
        <feFlood result="ring-colour" />
        <feComposite in="ring-colour" in2="band" operator="in" result="ring" />
        <feMerge><feMergeNode in="fill" /><feMergeNode in="ring" /></feMerge>
      </filter>
    </defs>
    <g filter="url(#slfx-skin)">
      <g filter="url(#slfx-goo)"><path fill="currentColor" /><circle fill="currentColor" r="0" /></g>
      <path fill="currentColor" />
    </g>`;
  const floods = skin.querySelectorAll("feFlood");
  const paths = skin.querySelectorAll("path");
  skinParts = {
    filters: skin.querySelectorAll("filter"),
    erode: skin.querySelector("feMorphology"),
    fill: floods[0],
    ring: floods[1],
    gooBox: paths[0],
    box: paths[1],
    drop: skin.querySelector("circle"),
  };
}

/** The point on the rounded box's outline nearest to p (p inside the box),
 *  with the outward normal there and the distance to it. On a corner arc the
 *  normal points away from the arc's centre; on a side it is the axis. */
function nearestEdge(px, py, { w, h, radii }) {
  const [tl, tr, br, bl] = radii;
  // Each corner: its arc centre, radius, and which way is "outside" it.
  const corners = [
    [tl, tl, tl, -1, -1], [w - tr, tr, tr, 1, -1], [w - br, h - br, br, 1, 1], [bl, h - bl, bl, -1, 1],
  ];
  for (const [cx, cy, r, sx, sy] of corners) {
    if (r <= 0) continue;
    const inX = sx < 0 ? px < cx : px > cx;
    const inY = sy < 0 ? py < cy : py > cy;
    if (inX && inY) {
      const dx = px - cx, dy = py - cy;
      const len = Math.hypot(dx, dy) || 1e-6;
      return { qx: cx + (dx / len) * r, qy: cy + (dy / len) * r, nx: dx / len, ny: dy / len, d: r - len };
    }
  }
  const sides = [
    { d: px, nx: -1, ny: 0, qx: 0, qy: py },
    { d: w - px, nx: 1, ny: 0, qx: w, qy: py },
    { d: py, nx: 0, ny: -1, qx: px, qy: 0 },
    { d: h - py, nx: 0, ny: 1, qx: px, qy: h },
  ];
  return sides.reduce((a, b) => (b.d < a.d ? b : a));
}

/* Critically underdamped: the swell overshoots and settles rather than easing
   to a stop. Stiffness and damping are tuned so a flick reads as elastic and
   a slow drag as viscous — the same spring, different input. */
const LIQUID_K = 170;
const LIQUID_DAMP = 14;
const LIQUID_SETTLE_EPS = 0.002;
/** The drop's radius from the element's own size, clamped so a keycap still
 *  swells visibly and a card does not grow a blister; how far it protrudes
 *  at full amplitude, as a share of that radius (≈3px on a 40px button);
 *  how far from an edge the pointer is felt, as a multiple of the radius —
 *  a Gaussian, so a card's edge answers the pointer approaching it and is
 *  still across the middle.
 *
 *  The owner called the swell "quasi inesistente" on round buttons. The cause
 *  was not the amplitude but cornerPx: --sl-radius-pill is 9999px, and without
 *  the clamp the drop was placed thousands of pixels away. With the outline
 *  correct, 0.24 was still slight and 0.5 read as a blob rather than a button
 *  giving; 0.34 gives 4.8px on a 2.5rem button — the shape yields, and stays a
 *  shape. Checked in Chrome at all three. */
const DROP_K = 0.35;
const DROP_MIN = 10;
const DROP_MAX = 16;
const BULGE_K = 0.34;
const REACH_K = 1.5;

let amp = 0, ampV = 0, target = 0;
let lagX = 0, lagY = 0;              // CSS px, within the live element's own box
let pointerX = 0, pointerY = 0;
let liquidLiveEl = null;
let liquidBox = null;                // { w, h, radii, border, drop, bulge, reach }
let liquidDrop = { x: 0, y: 0, r: 0 };
let liquidRafId = 0;
let liquidPrevT = 0;


function setLiquidLive(el) {
  if (liquidLiveEl === el) return;
  if (liquidLiveEl) {
    liquidLiveEl.classList.remove("is-liquid-live");
    skin?.remove();
  }
  liquidLiveEl = el;
  liquidBox = null;
  liquidDrop = { x: 0, y: 0, r: 0 };
  if (!el) return;
  buildSkin();
  const rect = el.getBoundingClientRect();
  const cs = getComputedStyle(el);
  const w = rect.width, h = rect.height;
  const radii = [cs.borderTopLeftRadius, cs.borderTopRightRadius, cs.borderBottomRightRadius, cs.borderBottomLeftRadius]
    .map((raw) => cornerPx(raw, w, h));
  const border = Math.max(parseFloat(cs.borderTopWidth) || 0, 0);
  const drop = Math.max(DROP_MIN, Math.min(DROP_MAX, Math.min(w, h) * DROP_K));
  const bulge = drop * BULGE_K;
  liquidBox = { w, h, radii, border, drop, bulge, reach: drop * REACH_K };

  const d = roundedRectPath(w, h, radii);
  skinParts.gooBox.setAttribute("d", d);
  skinParts.box.setAttribute("d", d);
  skinParts.erode.setAttribute("radius", String(border));
  // Both filter regions: the box plus room for the drop and the blur's tail.
  const pad = Math.ceil(GOO_BLUR * 3 + drop + border + 2);
  for (const f of skinParts.filters) {
    f.setAttribute("x", String(-pad));
    f.setAttribute("y", String(-pad));
    f.setAttribute("width", String(Math.ceil(w) + 2 * pad));
    f.setAttribute("height", String(Math.ceil(h) + 2 * pad));
  }
  // The skin's box is the host's BORDER box: an absolutely positioned child
  // resolves inset: 0 against the padding box, so it steps out by the
  // border's width (effects.css reads --slfx-bw).
  skin.style.setProperty("--slfx-bw", `${border}px`);
  el.appendChild(skin);
  el.classList.add("is-liquid-live");
}

function setDrop(name, value) {
  if (skinParts.drop.getAttribute(name) !== value) skinParts.drop.setAttribute(name, value);
}

/** Rebuild the silhouette for this frame: the drop at the outline point
 *  nearest the lagged pointer, protruding by the spring's amplitude scaled
 *  by how near the pointer is to that edge; the floods in the host's own
 *  colours as they are right now. */
function paintSkin() {
  const el = liquidLiveEl;
  const b = liquidBox;
  if (!el || !b || !skinParts) return;
  const px = Math.max(0, Math.min(b.w, lagX));
  const py = Math.max(0, Math.min(b.h, lagY));
  const e = nearestEdge(px, py, b);
  const near = Math.exp(-0.5 * (Math.max(e.d, 0) / b.reach) ** 2);
  const out = amp * b.bulge * near;
  // Below a quarter pixel the drop is entirely inside the box and adds
  // nothing; report radius 0 so the chrome does not weld an invisible circle.
  const visible = out > 0.25;
  liquidDrop = {
    x: e.qx + e.nx * (out - b.drop),
    y: e.qy + e.ny * (out - b.drop),
    r: visible ? b.drop : 0,
  };
  // Only what changed: an attribute written inside a filter dirties it and
  // re-rasterises the goo even when the value is the same.
  setDrop("cx", liquidDrop.x.toFixed(2));
  setDrop("cy", liquidDrop.y.toFixed(2));
  setDrop("r", visible ? b.drop.toFixed(2) : "0");

  const cs = getComputedStyle(el);
  if (skinParts.fill.style.floodColor !== cs.backgroundColor) skinParts.fill.style.floodColor = cs.backgroundColor;
  if (skinParts.ring.style.floodColor !== cs.borderTopColor) skinParts.ring.style.floodColor = cs.borderTopColor;
}

function integrateLiquid(now) {
  // A live host taken out of the document fires no pointerleave, so nothing
  // would ever clear `target` and the loop would run forever for nothing.
  // Same guard burst() carries, for the same reason: a timer or a loop
  // mutating something nothing can see.
  if (liquidLiveEl && !liquidLiveEl.isConnected) target = 0;
  const dt = Math.min((now - liquidPrevT) / 1000 || 0, 1 / 30);
  liquidPrevT = now;
  const a = LIQUID_K * (target - amp) - LIQUID_DAMP * ampV;
  ampV += a * dt;
  amp += ampV * dt;
  // The drop chases the pointer with its own lag, so it arrives after the
  // cursor rather than under it — that trail IS the mass.
  lagX += (pointerX - lagX) * Math.min(dt * 11, 1);
  lagY += (pointerY - lagY) * Math.min(dt * 11, 1);

  paintSkin();

  const settled = target === 0 && Math.abs(amp) < LIQUID_SETTLE_EPS && Math.abs(ampV) < LIQUID_SETTLE_EPS;
  if (settled) {
    liquidRafId = 0;
    amp = 0; ampV = 0;
    setLiquidLive(null);
    return; // idle page costs nothing: the loop does not reschedule itself
  }
  // At rest under a still pointer: nothing moves and no colour is in flight,
  // so the frame would repaint the same drop. The next pointermove, or a
  // transition starting on the host (a press), wakes it.
  const resting = target === 1 && Math.abs(amp - 1) < LIQUID_SETTLE_EPS && Math.abs(ampV) < LIQUID_SETTLE_EPS
    && Math.abs(pointerX - lagX) < 0.05 && Math.abs(pointerY - lagY) < 0.05
    && !liquidLiveEl?.getAnimations().length;
  if (resting) {
    liquidRafId = 0;
    return;
  }
  liquidRafId = requestAnimationFrame(integrateLiquid);
}

function ensureLiquidLoop() {
  if (liquidRafId) return;
  liquidPrevT = performance.now();
  liquidRafId = requestAnimationFrame(integrateLiquid);
}

function onPointerMove(ev) {
  if (!(ev.target instanceof Element)) return;
  onSpecPointer(ev);
  const el = ev.target.closest(LIQUID_HOSTS);
  if (!el || el.closest('[data-sl-fx="off"]')) return;
  const r = el.getBoundingClientRect();
  pointerX = ev.clientX - r.left;
  pointerY = ev.clientY - r.top;
  if (liquidLiveEl !== el) {
    // A fresh target starts its lag from its own entry point, not the
    // previous element's leftover coordinates.
    lagX = pointerX;
    lagY = pointerY;
    setLiquidLive(el);
  }
  target = 1;
  ensureLiquidLoop();
}

/** A resting drop follows its host's colours again once they start to move. */
function onLiquidTransition(ev) {
  if (liquidLiveEl && ev.target === liquidLiveEl) ensureLiquidLoop();
}

function onPointerLeave(ev) {
  if (!(ev.target instanceof Element)) return;
  const el = ev.target.closest(LIQUID_HOSTS);
  // pointerleave does not bubble, but the listener is on the root in the
  // capture phase, so it also fires for every CHILD the pointer leaves — an
  // icon inside the button. Only the host's own leave releases the host.
  if (!el || el !== liquidLiveEl || ev.target !== el) return;
  target = 0;
  ensureLiquidLoop(); // keep the loop running so the overshoot plays out
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
  for (const el of nearHosts) if (!el.isConnected) { nearObserver?.unobserve(el); nearHosts.delete(el); }
  for (const el of specHosts) {
    if (el.isConnected) continue;
    if (specState.has(el)) releaseSpec(el);
    specHosts.delete(el);
  }
}

/* ── ink: light pools under the pointer ──────────────────────────────────────
   effects.css paints a pool of light on one injected span per control; this
   file gives it the span, the four moving noise fields its rim is torn by, the
   trailing pointer, the pool's size, the control's colour and the ground the
   pool lies on. Ink owns no colour: the control's own hover applies as
   declared, and the pool is only light over it. Tier 1 only: without the span
   the plain fade stands. */
const INK_HOSTS = ".sl-btn, .sl-icon-btn";
const INK_FILTER = "sl-fx-ink-noise";
const SVG_NS = "http://www.w3.org/2000/svg";
/** Noise fields, dealt round-robin, so neighbouring pools wobble apart. */
const INK_SEEDS = [3, 11, 23, 41];
/** The share of the gap to the pointer the pool closes each frame: the lag
 *  that reads as viscosity. */
const INK_FOLLOW = 0.09;
/** How long the pool drains after the pointer leaves (effects.css's
 *  --sl-motion-ink-out fallback); the noise stays on until it has. */
const INK_DRAIN_MS = 520;
/** The pool's radius: 0.55 of the control's longer side, held to 26–72px so an
 *  icon button gets a drop and a wide button a pool, never a flood. */
const INK_SIZE = { share: 0.55, min: 26, max: 72 };
/** A ground at or above this relative luminance is light (oklch L ≈ 0.56). */
const INK_LIGHT_Y = 0.18;

let inkDealt = 0;

/** Give a control its ink layer, once, with its noise field. Prepended after
 *  the glass layers are in, so among the z-index -1 layers it paints first:
 *  above the tint, below the glass's shine and specular.
 *  @param {Element} el a button or icon button */
function injectInk(el) {
  if (el.querySelector(":scope > .sl-fx-ink")) return;
  const layer = document.createElement("span");
  layer.className = "sl-fx-ink";
  layer.setAttribute("aria-hidden", "true");
  layer.style.setProperty("--slfx-ink-noise", `url(#${INK_FILTER}-${inkDealt++ % INK_SEEDS.length})`);
  el.prepend(layer);
}

function injectInks(root) {
  if (root instanceof Element && root.matches(INK_HOSTS)) injectInk(root);
  for (const el of root.querySelectorAll(INK_HOSTS)) injectInk(el);
}

/** The noise fields every pool's rim is torn by, added to the document once.
 *  Each displaces the pool by up to 14px and slowly breathes its frequency,
 *  so the rim keeps moving like a liquid's edge; each seed tears differently.
 *  Applied only while a pool is up (effects.css, data-sl-ink-live). */
function ensureInkFilter() {
  if (document.getElementById(`${INK_FILTER}-0`)) return;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "0");
  svg.setAttribute("height", "0");
  svg.setAttribute("aria-hidden", "true");
  svg.style.position = "absolute";
  INK_SEEDS.forEach((seed, i) => {
    const filter = document.createElementNS(SVG_NS, "filter");
    filter.id = `${INK_FILTER}-${i}`;
    for (const [k, v] of [["x", "-30%"], ["y", "-30%"], ["width", "160%"], ["height", "160%"]]) filter.setAttribute(k, v);
    const noise = document.createElementNS(SVG_NS, "feTurbulence");
    for (const [k, v] of [["type", "fractalNoise"], ["baseFrequency", "0.035"], ["numOctaves", "2"], ["seed", String(seed)], ["result", "n"]]) noise.setAttribute(k, v);
    const breathe = document.createElementNS(SVG_NS, "animate");
    for (const [k, v] of [["attributeName", "baseFrequency"], ["dur", `${5 + i}s`], ["values", "0.03;0.045;0.03"], ["repeatCount", "indefinite"]]) breathe.setAttribute(k, v);
    noise.append(breathe);
    const shift = document.createElementNS(SVG_NS, "feDisplacementMap");
    for (const [k, v] of [["in", "SourceGraphic"], ["in2", "n"], ["scale", "14"], ["xChannelSelector", "R"], ["yChannelSelector", "G"]]) shift.setAttribute(k, v);
    filter.append(noise, shift);
    svg.append(filter);
  });
  document.body.append(svg);
}

/** A computed colour's alpha, 0–1: 0 for `transparent`, 1 when it has none.
 *  Computed colours come back in several notations, so the alpha is read as
 *  whatever follows a "/", or as the fourth comma-separated argument.
 *  @param {string} v a computed colour */
function alphaOf(v) {
  if (!v || v === "transparent") return 0;
  const a = v.match(/\/\s*([\d.]+%?)\s*\)$/)?.[1] ?? v.match(/^rgba\([^,]+,[^,]+,[^,]+,\s*([\d.]+)\)$/)?.[1];
  if (a === undefined) return 1;
  return a.endsWith("%") ? parseFloat(a) / 100 : parseFloat(a);
}

/** Whether a computed colour paints anything: not `transparent`, not alpha 0.
 *  @param {string} v a computed colour */
function paints(v) {
  return alphaOf(v) > 0;
}

/** The colour arriving on a control at the moment the pointer crosses it: the
 *  end of its own running background (or glass fill) transition when there is
 *  one, else its current fill, else its ink -- the first that has a hue. Ink
 *  tints its pool with this so a red control glows red; a neutral one has
 *  none and glows in the accent. `fill` says the colour is an opaque fill the
 *  pool lies on, rather than a wash or the label.
 *  @param {Element} host the button or icon button
 *  @returns {{ tint: string|null, fill: boolean }} */
function inkTint(host) {
  const found = (v, fill) => ({ tint: v, fill: fill && alphaOf(v) >= 0.5 });
  for (const a of host.getAnimations()) {
    const prop = a.transitionProperty;
    if (prop !== "background-color" && prop !== "--slfx-accent-t") continue;
    const frames = a.effect?.getKeyframes?.() ?? [];
    const to = frames[frames.length - 1];
    const v = to?.backgroundColor ?? to?.["--slfx-accent-t"];
    if (paints(v) && isChromatic(v)) return found(v, true);
  }
  const cs = getComputedStyle(host);
  for (const v of [cs.getPropertyValue("--slfx-accent-t").trim(), cs.backgroundColor]) {
    if (paints(v) && isChromatic(v)) return found(v, true);
  }
  if (paints(cs.color) && isChromatic(cs.color)) return found(cs.color, false);
  return { tint: null, fill: false };
}

/** Whether the first opaque ground under a control is light: its own fill if
 *  it has one, else the nearest ancestor's. Nothing opaque up the tree is the
 *  page's canvas, which is light unless a theme painted it.
 *  @param {Element} host the button or icon button
 *  @returns {"light"|"dark"} */
function inkGround(host) {
  for (let el = host; el; el = el.parentElement) {
    const bg = getComputedStyle(el).backgroundColor;
    if (alphaOf(bg) < 0.5) continue;
    const [r, g, b] = toSrgb(document, bg).map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b >= INK_LIGHT_Y ? "light" : "dark";
  }
  return "light";
}

/** Live pools, by ink layer: where each is (x, y), where the pointer is
 *  (tx, ty), whether the pointer is still on it, and its drain timer. */
const inkPools = new Map();
let inkFrame = 0;

/** One frame of trailing: every pool closes INK_FOLLOW of its gap to the
 *  pointer. The loop sleeps once every pool has caught up; a pool is only
 *  forgotten by its drain timer, so a pointer that comes back finds it. */
function inkStep() {
  inkFrame = 0;
  let moving = false;
  for (const [ink, p] of inkPools) {
    if (Math.abs(p.tx - p.x) + Math.abs(p.ty - p.y) < 0.5) continue;
    p.x += (p.tx - p.x) * INK_FOLLOW;
    p.y += (p.ty - p.y) * INK_FOLLOW;
    ink.style.setProperty("--slfx-ink-x", `${p.x.toFixed(1)}px`);
    ink.style.setProperty("--slfx-ink-y", `${p.y.toFixed(1)}px`);
    moving = true;
  }
  if (moving) inkFrame = requestAnimationFrame(inkStep);
}

function inkTrail() {
  if (!inkFrame) inkFrame = requestAnimationFrame(inkStep);
}

/** A pointer crossing a control's edge. On the way in, the pool is placed at
 *  the crossing, sized to the control, tinted and read against its ground, and
 *  goes live; on the way out it trails to where the pointer left and drains
 *  there. Moves inside the control are onInkMove's. Touch has no hover to
 *  answer, so it raises no pool.
 *  @param {PointerEvent} ev a delegated pointerover or pointerout */
function onInkCross(ev) {
  if (ev.pointerType === "touch") return;
  const host = ev.target instanceof Element ? ev.target.closest(INK_HOSTS) : null;
  if (!host || (ev.relatedTarget instanceof Node && host.contains(ev.relatedTarget))) return;
  const ink = host.querySelector(":scope > .sl-fx-ink");
  const r = host.getBoundingClientRect();
  if (!ink || !r.width || !r.height) return;
  const x = ev.clientX - r.left;
  const y = ev.clientY - r.top;
  let p = inkPools.get(ink);
  if (ev.type === "pointerout") {
    if (!p) return;
    Object.assign(p, { tx: x, ty: y, on: false });
    p.drain = setTimeout(() => {
      if (p.on) return;
      inkPools.delete(ink);
      ink.removeAttribute("data-sl-ink-live");
    }, INK_DRAIN_MS);
    inkTrail();
    return;
  }
  if (host.matches(':disabled, [aria-disabled="true"]')) return;
  const { tint, fill } = inkTint(host);
  if (tint) ink.style.setProperty("--slfx-ink-tint", tint);
  else ink.style.removeProperty("--slfx-ink-tint");
  ink.setAttribute("data-sl-ink-ground", fill ? "fill" : inkGround(host));
  const size = Math.min(INK_SIZE.max, Math.max(INK_SIZE.min, Math.max(r.width, r.height) * INK_SIZE.share));
  ink.style.setProperty("--slfx-ink-size", `${Math.round(size)}px`);
  if (!p) {
    p = { x, y };
    ink.style.setProperty("--slfx-ink-x", `${x.toFixed(1)}px`);
    ink.style.setProperty("--slfx-ink-y", `${y.toFixed(1)}px`);
    inkPools.set(ink, p);
  }
  clearTimeout(p.drain);
  Object.assign(p, { tx: x, ty: y, on: true });
  ink.setAttribute("data-sl-ink-live", "");
  inkTrail();
}

/** The pointer moving inside a control with a live pool: the pool's target
 *  follows it, and the trailing loop wakes if it had caught up.
 *  @param {PointerEvent} ev a delegated pointermove */
function onInkMove(ev) {
  if (!inkPools.size || !(ev.target instanceof Element)) return;
  const host = ev.target.closest(INK_HOSTS);
  const ink = host?.querySelector(":scope > .sl-fx-ink");
  const p = ink && inkPools.get(ink);
  if (!p || !p.on) return;
  const r = host.getBoundingClientRect();
  p.tx = ev.clientX - r.left;
  p.ty = ev.clientY - r.top;
  inkTrail();
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
 * Arm Tier 1: pointer-driven liquid, finite glitch bursts, animated aurora.
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
  ensureInkFilter();
  injectInks(root);
  watchNear(root);
  collectSpecHosts(root);
  armLiveLight(root);
  // Tier 1's own late arrivals: the glow is a moving layer and the viewport
  // gate only matters where something is running, so both belong on this side
  // of the guard. The material's observer is above.
  new MutationObserver((records) => {
    for (const r of records) for (const n of r.addedNodes) if (n instanceof Element) {
      injectGlows(n);
      injectInks(n);
      watchNear(n);
      collectSpecHosts(n);
    }
    // Content arriving or leaving can move a box under a resting pointer
    // (a streamed card pushing the wall down): the specular loop looks again.
    onSpecLayout();
  }).observe(root, { childList: true, subtree: true });

  // One delegated listener for the whole subtree, not one per element.
  root.addEventListener("pointermove", onPointerMove, { passive: true });
  root.addEventListener("pointerleave", onPointerLeave, { capture: true, passive: true });
  root.addEventListener("transitionrun", onLiquidTransition, { passive: true });
  window.addEventListener("pointerout", onSpecOut, { passive: true });
  window.addEventListener("scroll", onSpecLayout, { capture: true, passive: true });
  window.addEventListener("resize", onSpecLayout, { passive: true });
  root.addEventListener("pointerover", onInkCross, { passive: true });
  root.addEventListener("pointerout", onInkCross, { passive: true });
  root.addEventListener("pointermove", onInkMove, { passive: true });
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
