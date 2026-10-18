/* @sondalab/ui-kit — halftone-core.js
   A picture resampled into a grid of round dots whose size follows the light:
   sondalab-site's dot scenes (src/lib/dot-scene.ts, 2026-09-26), moved into
   the kit so a live canvas (halftone.js) and an exported asset (toSVG) draw
   the same dots. Pure maths over RGBA bytes: no DOM, runs in Node.

   The defaults are the site's code, not the site's spec, which drifted.
   Photos read as dots only when the subject is one iconic outline with
   near-binary contrast, at ≥ 320px with a grid near 130. */

const FLOOR = 0.03;

/** A small seeded PRNG (mulberry32): same seed, same sequence, on every engine. */
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Samples an RGBA picture (`size` px square) on a `grid`×`grid` lattice. The
 * picture's own 2nd..99th luminance percentile becomes the full range; the
 * outer half of the inscribed circle dissolves. `invert` prints the dark, for
 * ink on a light ground. A transparent pixel counts as ground in either case.
 * `seed` makes the edge, scatter, delays and phases reproducible.
 */
export function sampleDots(px, size, { grid = 96, invert = false, seed, floor = FLOOR } = {}) {
  const rand = seed === undefined ? Math.random : seeded(seed);
  const ground = invert ? 1 : 0;
  const lum0 = (k) => {
    const a = px[k + 3] / 255;
    return (a * (0.2126 * px[k] + 0.7152 * px[k + 1] + 0.0722 * px[k + 2])) / 255 + (1 - a) * ground;
  };
  const lum = invert ? (k) => (1 - lum0(k)) ** 1.8 : lum0;
  const count = size * size;
  const all = new Float64Array(count);
  for (let i = 0; i < count; i++) all[i] = lum(i * 4);
  all.sort();
  const lo = all[Math.floor(count * 0.02)];
  const hi = Math.max(lo + 0.05, all[Math.floor(count * 0.99)]);

  const dots = [];
  const cell = size / grid;
  for (let j = 0; j < grid; j++) {
    for (let i = 0; i < grid; i++) {
      const x = (i + 0.5) / grid;
      const y = (j + 0.5) / grid;
      const edge = Math.hypot(x - 0.5, y - 0.5) / 0.5;
      if (edge >= 1) continue;
      const k = (Math.floor((j + 0.5) * cell) * size + Math.floor((i + 0.5) * cell)) * 4;
      const lin = Math.min(1, Math.max(0, (lum(k) - lo) / (hi - lo)));
      const f = Math.min(1, (1 - edge) / 0.5);
      const fade = f * f * (3 - 2 * f);
      const L = lin * lin * (3 - 2 * lin) * fade;
      if (L < floor || rand() > fade ** 0.6) continue;
      // Scatter inside the circle, uniform by area, so the square never shows.
      const sr = 0.5 * Math.sqrt(rand());
      const sa = rand() * Math.PI * 2;
      dots.push({
        x,
        y,
        L,
        sx: 0.5 + sr * Math.cos(sa),
        sy: 0.5 + sr * Math.sin(sa),
        delay: rand() * 0.35 + Math.hypot(x - 0.5, y - 0.5) * 0.5,
        phase: rand() * Math.PI * 2,
      });
    }
  }
  return dots;
}

/**
 * Where dot `d` is `t` seconds after its scene began to gather, on a
 * `width`-px square with a `grid`-cell lattice: it eases from its scatter to
 * its place, then floats, its light breathing, a scan band crossing every
 * `scanPeriod` seconds. `motion` overrides the site's constants.
 */
export function dotFrame(d, t, width, grid, { assemble = 1.1, drift = 0.15, shimmer = 0.075, scanPeriod = 6 } = {}) {
  const cell = width / grid;
  const k = Math.min(1, Math.max(0, (t - d.delay) / assemble));
  const e = 1 - (1 - k) ** 3;
  const orbit = drift * cell * e;
  const x = (d.sx + (d.x - d.sx) * e) * width + orbit * Math.sin(t * 0.55 + d.phase);
  const y = (d.sy + (d.y - d.sy) * e) * width + orbit * Math.cos(t * 0.43 + d.phase * 1.7);
  const breath = 1 + shimmer * Math.sin(t * 1.6 + d.phase);
  const scan = ((t % scanPeriod) / scanPeriod) * 1.4 - 0.2;
  const band = Math.max(0, 1 - Math.abs(d.y - scan) * 14) * 0.5;
  const L = Math.min(1, d.L * breath + band * d.L);
  return { x, y, r: 0.56 * cell * Math.sqrt(L) * (0.4 + 0.6 * e) };
}

/**
 * The dot at rest: at its place, full size, no float, breath or scan band.
 * What reduced motion shows and what an export prints.
 */
export function settledFrame(d, width, grid) {
  return { x: d.x * width, y: d.y * width, r: 0.56 * (width / grid) * Math.sqrt(d.L) };
}

/**
 * The centred square of a `width`×`height` picture, as drawImage source
 * coordinates: a wide or tall picture is cropped, never squashed.
 */
export function cover(width, height) {
  const s = Math.min(width, height);
  return { sx: (width - s) / 2, sy: (height - s) / 2, s };
}

const escape = (v) => String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const round = (v) => Math.round(v * 100) / 100;

/**
 * The settled halftone as an SVG string, `size` px square: one circle per
 * dot, the fill set once (`currentColor` unless given, so an inlined SVG
 * takes the text colour), an optional full-size ground. For print and for
 * marketing assets; the motion is the runtime's, not the file's.
 */
export function toSVG(dots, { size, grid, color = "currentColor", background } = {}) {
  let circles = "";
  for (const d of dots) {
    const f = settledFrame(d, size, grid);
    circles += `<circle cx="${round(f.x)}" cy="${round(f.y)}" r="${round(f.r)}"/>`;
  }
  const ground = background ? `<rect width="${size}" height="${size}" fill="${escape(background)}"/>` : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">${ground}<g fill="${escape(color)}">${circles}</g></svg>`;
}
