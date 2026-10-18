/* @sondalab/ui-kit — halftone.js
   The live half of the halftone: every canvas[data-sl-halftone] under a root
   becomes its picture in dots, gathering the first time it is 40% in view (or fills 40% of the viewport),
   then floating. The maths is halftone-core.js, the same an export prints.

     <canvas data-sl-halftone data-sl-src="/art/kinship.png"
             data-sl-grid="130" aria-hidden="true"></canvas>

   SIZE: the canvas takes its CSS width (effects.css gives an unsized one its
   container's width, square).

   COLOUR: the canvas's own inherited `color`, so a scene on an inversion band
   prints in the band's ink. Ink darker than mid-grey prints the picture's
   dark (a halftone on paper); light ink draws its light. A theme change that
   flips the polarity resamples; one that only changes the colour repaints.
   Followed on <html> and on every section under the root: a data-sl-theme or
   a class (an .sl-invert band) that changes after arming re-reads the ink of
   the scenes inside it, and again when a colour transition there ends.
   data-sl-ink="spectral" opts a canvas into the light theme's spectrum: on a
   theme that declares it the dots are filled with one gradient through
   --sl-spectrum-ink-0..4 instead; the polarity still follows `color`. Without
   the attribute the dots are single-ink on every theme (owner, 2026-10-02:
   no rainbow by default). "mono", the old opt-out, is now the default.

   MOTION: one frame loop for every scene, running only while one is within
   200px of the viewport and the page is visible. Under reduced motion each
   scene draws its settled frame, once, when seen, resized or re-themed; the
   picture is content, so it is shown, not refused. mount()'s
   ignoreReducedMotion writes data-sl-fx-motion="always", which brings the
   motion back. data-sl-fx-still on <html> holds every scene where it is and
   resumes it from there, as live-light.js does.

   OFF: a canvas under data-sl-fx="off" is never armed; under high contrast
   effects.css hides it and nothing is drawn. */

import { sampleDots, dotFrame, settledFrame, cover } from "./halftone-core.js";
import { toSrgb } from "./colour.js";

const SEL = "canvas[data-sl-halftone]";
const GRID = 96;
const MIN_GRID = 16;
const MAX_GRID = 256;
const REDUCE = "(prefers-reduced-motion: reduce)";
const STILL = "data-sl-fx-still";
const MOTION = "data-sl-fx-motion";

function gridOf(canvas) {
  const n = Math.round(Number(canvas.getAttribute("data-sl-grid")));
  return n > 0 ? Math.min(MAX_GRID, Math.max(MIN_GRID, n)) : GRID;
}

function seedOf(canvas) {
  const raw = canvas.getAttribute("data-sl-seed");
  return raw === null || !Number.isFinite(Number(raw)) ? undefined : Number(raw);
}

const INK_STOPS = 5;

/** The light theme's spectrum ink as read on the canvas, or null unless the
 * author opted in (data-sl-ink="spectral") and the theme declares it. */
function spectrumOf(canvas) {
  if (canvas.getAttribute("data-sl-ink") !== "spectral") return null;
  const cs = getComputedStyle(canvas);
  const stops = [];
  for (let i = 0; i < INK_STOPS; i++) {
    const v = cs.getPropertyValue(`--sl-spectrum-ink-${i}`).trim();
    if (!v) return null;
    stops.push(v);
  }
  return stops;
}

/** Ink darker than mid-grey prints the picture's dark. */
function darkInk([r, g, b]) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5;
}

/** Decode the picture, crop its centred square, keep its bytes at twice the grid. */
async function load(canvas) {
  const doc = canvas.ownerDocument;
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = canvas.getAttribute("data-sl-src");
  await img.decode();
  const grid = gridOf(canvas);
  const size = grid * 2;
  const off = doc.createElement("canvas");
  off.width = off.height = size;
  const g = off.getContext("2d", { willReadFrequently: true });
  const c = cover(img.naturalWidth, img.naturalHeight);
  g.drawImage(img, c.sx, c.sy, c.s, c.s, 0, 0, size, size);
  // Throws on a picture from another origin without CORS: the scene is dropped.
  const px = g.getImageData(0, 0, size, size).data;
  return { canvas, grid, px, seed: seedOf(canvas), dots: [], ink: "", spectrum: null, dark: false, t0: null, near: false, width: canvas.clientWidth };
}

/**
 * Brings every `canvas[data-sl-halftone]` under `root` to life and follows
 * canvases added later, theme changes, and the still flag. `ready` settles
 * once the canvases present now have loaded (a failed one is skipped).
 * @param {Document|Element} [root]
 * @returns {{stop(): void, ready: Promise<void>}}
 */
export function halftone(root = document) {
  const doc = root.ownerDocument || root;
  const html = doc.documentElement;
  const scenes = new Map();
  const pending = new Set();
  const reduce = globalThis.matchMedia?.(REDUCE) ?? { matches: false };
  const hasIO = typeof IntersectionObserver === "function";
  let raf = 0;
  let stillAt = null;
  let stopped = false;

  const moving = () => !reduce.matches || html.getAttribute(MOTION) === "always";
  const contrast = () => html.getAttribute("data-sl-theme") === "high-contrast";

  function paint(s) {
    s.ink = getComputedStyle(s.canvas).color;
    s.spectrum = spectrumOf(s.canvas);
    const dark = darkInk(toSrgb(doc, s.ink));
    if (dark !== s.dark || s.dots.length === 0) {
      s.dark = dark;
      s.dots = sampleDots(s.px, s.grid * 2, { grid: s.grid, invert: dark, seed: s.seed });
    }
  }

  function draw(s, now) {
    const w = s.width;
    if (!w) return false;
    const dpr = globalThis.devicePixelRatio || 1;
    const side = Math.round(w * dpr);
    if (s.canvas.width !== side) s.canvas.width = s.canvas.height = side;
    const g = s.canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, w);
    if (contrast()) return false;
    const still = !moving();
    const t = (now - s.t0) / 1000;
    if (s.spectrum) {
      // Prism: one gradient across the scene through the spectrum ink, so the
      // dots stay a single fill and the export stays single-ink.
      const grad = g.createLinearGradient(0, 0, w, w);
      s.spectrum.forEach((c, i) => grad.addColorStop(i / (s.spectrum.length - 1), c));
      g.fillStyle = grad;
    } else {
      g.fillStyle = s.ink;
    }
    g.beginPath();
    for (const d of s.dots) {
      const f = still ? settledFrame(d, w, s.grid) : dotFrame(d, t, w, s.grid);
      if (f.r <= 0) continue;
      g.moveTo(f.x + f.r, f.y);
      g.arc(f.x, f.y, f.r, 0, Math.PI * 2);
    }
    g.fill();
    return true;
  }

  function frame(now) {
    raf = 0;
    const hold = html.hasAttribute(STILL);
    // Held: every frame draws the instant it froze at. Released: each scene's
    // origin moves by the pause, so it picks up where it stopped.
    if (hold) stillAt ??= now;
    else if (stillAt !== null) {
      for (const s of scenes.values()) if (s.t0 !== null) s.t0 += now - stillAt;
      stillAt = null;
    }
    let drawn = 0;
    for (const s of scenes.values()) {
      if (!s.canvas.isConnected) { forget(s); continue; }
      if (s.near && s.t0 !== null && draw(s, hold ? stillAt : now)) drawn++;
    }
    if (drawn && moving() && !hold && doc.visibilityState !== "hidden") raf = requestAnimationFrame(frame);
  }

  function wake() {
    if (!raf && !stopped) raf = requestAnimationFrame(frame);
  }

  const near = hasIO && new IntersectionObserver((entries) => {
    for (const e of entries) { const s = scenes.get(e.target); if (s) s.near = e.isIntersecting; }
    wake();
  }, { rootMargin: "200px" });
  const seen = hasIO && new IntersectionObserver((entries) => {
    for (const e of entries) {
      const s = scenes.get(e.target);
      if (s && s.t0 === null && e.isIntersecting && (e.intersectionRatio >= 0.4 || (e.rootBounds && e.intersectionRect.height >= 0.4 * e.rootBounds.height))) {
        s.t0 = stillAt ?? performance.now();
      }
    }
    wake();
  }, { threshold: [0, 0.1, 0.2, 0.3, 0.4] });
  const size = typeof ResizeObserver === "function" && new ResizeObserver((entries) => {
    for (const e of entries) { const s = scenes.get(e.target); if (s) s.width = e.contentRect.width; }
    wake();
  });

  function adopt(s) {
    if (stopped) return;
    paint(s);
    scenes.set(s.canvas, s);
    if (hasIO) { near.observe(s.canvas); seen.observe(s.canvas); }
    else { s.near = true; s.t0 = stillAt ?? performance.now(); }
    if (size) size.observe(s.canvas);
    wake();
  }

  function forget(s) {
    scenes.delete(s.canvas);
    if (hasIO) { near.unobserve(s.canvas); seen.unobserve(s.canvas); }
    if (size) size.unobserve(s.canvas);
  }

  function scan(n) {
    const found = n.matches?.(SEL) ? [n] : [];
    found.push(...(n.querySelectorAll?.(SEL) ?? []));
    const loads = [];
    for (const c of found) {
      if (scenes.has(c) || pending.has(c) || c.closest('[data-sl-fx="off"]')) continue;
      pending.add(c);
      loads.push(load(c).then(adopt).finally(() => pending.delete(c)));
    }
    return Promise.allSettled(loads);
  }

  const ready = scan(root).then(() => {});

  /** Re-read the ink of every scene inside `el`, and draw them once. */
  function retint(el) {
    let hit = false;
    for (const s of scenes.values()) if (el.contains(s.canvas)) { paint(s); hit = true; }
    if (hit) wake();
  }

  const added = new MutationObserver((records) => {
    for (const r of records) {
      if (r.type === "attributes") { if (r.target !== html) retint(r.target); continue; }
      for (const n of r.addedNodes) if (n instanceof Element) scan(n);
    }
  });
  added.observe(root, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-sl-theme", "class", "data-sl-ink"] });

  const faded = (e) => { if (e.propertyName === "color") retint(e.target); };
  root.addEventListener("transitionend", faded);

  const theme = new MutationObserver((records) => {
    if (!records.every((r) => r.attributeName === STILL)) for (const s of scenes.values()) paint(s);
    wake();
  });
  theme.observe(html, { attributes: true, attributeFilter: ["data-sl-theme", STILL, MOTION] });

  doc.addEventListener("visibilitychange", wake);

  return {
    ready,
    stop() {
      stopped = true;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      if (hasIO) { near.disconnect(); seen.disconnect(); }
      if (size) size.disconnect();
      added.disconnect();
      theme.disconnect();
      doc.removeEventListener("visibilitychange", wake);
      root.removeEventListener("transitionend", faded);
    },
  };
}
