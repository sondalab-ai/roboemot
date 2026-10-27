/* @sondalab/ui-kit — dither-core.js
   The Bayer ordered dither the pixel direction draws with: a wipe that
   spreads from a point (the button hover) and a ramp that thickens along a
   strip (the signal strip). Ported from the lab probe lab/pixel-directions.html
   (BAYER, ditherFx). Pure maths over a grid of cells: no DOM, no colour,
   runs in Node. The caller owns the canvas and the colours. */

/** The kit's pixel unit, in CSS px: one dither cell is CELL_PX × CELL_PX. */
export const CELL_PX = 3;

/** The 8×8 Bayer thresholds, each in (0,1), 64 distinct values, frozen. */
export const BAYER = (() => {
  let m = [[0]];
  for (let n = 1; n < 8; n *= 2) {
    const r = [];
    for (let y = 0; y < 2 * n; y++) {
      r[y] = [];
      for (let x = 0; x < 2 * n; x++) {
        r[y][x] = 4 * m[y % n][x % n] + [[0, 2], [3, 1]][Math.floor(y / n)][Math.floor(x / n)];
      }
    }
    m = r;
  }
  return Object.freeze(m.map((row) => Object.freeze(row.map((v) => (v + 0.5) / 64))));
})();

/**
 * The Bayer threshold at cell (i, j); the matrix tiles, so any integer
 * (including a negative one) is valid. Returns a value in (0,1).
 */
export function bayerAt(i, j) {
  return BAYER[((j % 8) + 8) % 8][((i % 8) + 8) % 8];
}

/**
 * Is cell (i, j) covered by a wipe at time `t`? `grid` is `{cols, rows}`,
 * `origin` is `{x, y}` in cell units (it may lie outside the grid), `t` runs
 * 0..1 and `spread` is how far the dithered edge reaches. t=0 covers no
 * cell, t=1 covers every cell.
 */
export function wipeCovered(i, j, origin, t, grid, spread = 0.7) {
  const maxD = farthest(grid, origin);
  return covered(i, j, origin, t, maxD, spread);
}

/**
 * The whole wipe frame: a `cols * rows` Uint8Array, row-major, 1 where the
 * cell is covered. Callers draw only the covered cells. Same arguments as
 * wipeCovered, without the cell.
 */
export function wipeFrame(grid, origin, t, spread = 0.7) {
  const { cols, rows } = grid;
  const out = new Uint8Array(cols * rows);
  const maxD = farthest(grid, origin);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) out[j * cols + i] = covered(i, j, origin, t, maxD, spread) ? 1 : 0;
  }
  return out;
}

/**
 * The signal-strip ramp: a `cols * rows` Uint8Array, row-major, 1 where the
 * cell is on. `level` (0..1) is how much of the strip is filled: the ramp
 * is stretched over the first `level` of the width and nothing lies beyond
 * it. Inside, density rises from none at the low end to full at the high end.
 * level 1 is the lab's ramp (on when i / cols > BAYER), level 0 is empty.
 */
export function rampFrame(grid, level = 1) {
  const { cols, rows } = grid;
  const out = new Uint8Array(cols * rows);
  if (!(level > 0)) return out;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const u = i / (cols * level);
      if (u < 1 && u > bayerAt(i, j)) out[j * cols + i] = 1;
    }
  }
  return out;
}

/** Distance from the origin to the farthest corner of the grid, in cells. */
function farthest({ cols, rows }, { x, y }) {
  return Math.hypot(Math.max(x, cols - x), Math.max(y, rows - y));
}

function covered(i, j, origin, t, maxD, spread) {
  const dn = maxD > 0 ? Math.hypot(i + 0.5 - origin.x, j + 0.5 - origin.y) / maxD : 0;
  return dn < t * (1 + spread) - bayerAt(i, j) * spread;
}

/* ── pixel controls (check, radio, switch) ──────────────────────────────────
   The cells the CSS draws as the final frame and the canvas plots on the way
   there. components.css places the tick, the dot and the dash with these exact
   offsets (scripts/pixel-controls.test.mjs pins the two to this one list), and
   effects.js draws the same cells, so the last canvas frame is the CSS frame. */

/** The check and radio marks are CHECK_CELLS x CHECK_CELLS cells, centred in a
 *  box of CHECK_BOX_CELLS per side: CHECK_INSET cells of margin all round. */
export const CHECK_CELLS = 6;
export const CHECK_INSET = 1;
export const CHECK_BOX_CELLS = CHECK_CELLS + 2 * CHECK_INSET;

/** The check and radio marks, and the fill that dithers in behind them, are
 *  CONTROL_CELL_PX squares, not CELL_PX: on 3px cells the 6 x 6 mark needs an
 *  18px box, which the owner found a tad large (2026-10-26). On 2px cells the
 *  same shapes measure 12px in a 16px box (the pre-pixel size). 2px is a whole
 *  number of device pixels at DPR 1, 1.5 and 2, so the cells stay crisp. The
 *  switch keeps CELL_PX. */
export const CONTROL_CELL_PX = 2;

const cells = (list) => Object.freeze(list.map((c) => Object.freeze(c)));

/** The tick, in plotting order, as [column, row]: a V, short arm down, long arm
 *  up. None sits in a corner cell: the layer clips at the box's radius and the
 *  CSS pixel does not. */
export const TICK_CELLS = cells([[0, 2], [1, 3], [2, 4], [3, 3], [4, 2], [5, 1]]);

/** The mixed state's dash, as [column, row]: 4 x 2 cells, centred on the 6 x 6 grid. */
export const DASH_CELLS = cells([[1, 2], [2, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3], [4, 3]]);

/** The radio dot in two frames, both centred on the even grid: the 2 x 2 core,
 *  then the core with a one-cell arm on each of its eight sides (12 cells, a
 *  "fat plus" with no corners, which reads round). A 5-cell plus cannot be
 *  centred on an even grid. */
export const DOT_FRAMES = Object.freeze([
  cells([[2, 2], [3, 2], [2, 3], [3, 3]]),
  cells([[2, 1], [3, 1], [1, 2], [2, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3], [4, 3], [2, 4], [3, 4]]),
]);

/** The switch track's padding box in cells (33 x 18px), the knob's width in
 *  cells (12px), its inset (3px) and its travel in steps: steps(5) in CSS. */
export const SWITCH_CELLS = Object.freeze({ cols: 11, rows: 6, knob: 4, inset: 1, steps: 5 });

/** Share of a check's motion that is the fill; the tick is plotted over the rest. */
export const TICK_START = 0.45;

/**
 * How many tick cells are plotted at progress `t` (0..1): none until the fill
 * is TICK_START in, then one more cell per equal slice, all of them at t=1.
 */
export function tickCount(t) {
  const n = Math.ceil(((t - TICK_START) / (1 - TICK_START)) * TICK_CELLS.length);
  return Math.min(TICK_CELLS.length, Math.max(0, n));
}

/** Which radio frame shows at progress `t`: 0 none, 1 one cell, 2 the plus. */
export function dotFrame(t) {
  return Math.min(2, Math.max(0, Math.round(t * 2)));
}

/**
 * The knob's offset in cells after `p` (0..1) of a steps(N) transition, from
 * `from` to `to` (0 or 1 for off and on): CSS steps(N, jump-end) holds each
 * value until its slice is done, so the position is floor(p * N) of N.
 */
export function knobCells(p, from, to) {
  const k = Math.floor(Math.min(1, Math.max(0, p)) * SWITCH_CELLS.steps);
  return from * SWITCH_CELLS.steps + (to - from) * k;
}
