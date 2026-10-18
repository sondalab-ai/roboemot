/* @sondalab/ui-kit — glass-lens.js
   A real refraction on an element that already exists.

   ADAPTED FROM shuding/liquid-glass, and the adaptation is deliberately thin:

     liquid-glass — https://github.com/shuding/liquid-glass
     MIT License · Copyright (c) 2025 Shu Ding
     Permission is hereby granted, free of charge, to any person obtaining a
     copy of this software and associated documentation files (the "Software"),
     to deal in the Software without restriction, including without limitation
     the rights to use, copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software, and to permit persons to whom the
     Software is furnished to do so, subject to the following conditions:
     The above copyright notice and this permission notice shall be included in
     all copies or substantial portions of the Software.
     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

   WHAT IS HIS, VERBATIM: smoothStep, the rounded-rect SDF, the fragment, and
   the map builder — the normalisation (`maxScale *= 0.5`), the `scale` derived
   from the measured maximum displacement, and the SDF's own half-extents and
   radius. Those are load-bearing: a hand rewrite of the same idea (lab/1p)
   produced a filter that applied but never displaced, at any scale.

   WHAT IS OURS: the target. His file builds one fixed, draggable panel and
   puts the filter on that; this binds the filter to an element the consumer
   already wrote. The difference matters twice over —
     - his panel sits OVER a button, so the button's label refracts with the
       backdrop. Filtering the element itself leaves its content untouched,
       because backdrop-filter reaches only what is behind an element. AC-25
       holds by construction rather than by rule.
     - his drag reads getBoundingClientRect() and writes style.left/top, which
       are different coordinate spaces the moment the container is re-parented;
       moving it makes it jump on the first pointer movement. Never taking his
       container means never meeting that.

   WHY NOT THE GPU. This slice spent a day on a WebGPU lens and a capture-plate
   before it, on the strength of a ruling that said backdrop-filter displacement
   was inert in Chrome 152. It is not: that measurement was of liqui.design's
   filter, not of the browser. This costs nothing per frame — the map is built
   once per size, backdrop-filter is continuous, and the browser reapplies it
   itself. No render loop, no canvas per frame, no device to lose. */

/** @type {number} distinguishes one element's filter from another's */
let serial = 0;

/* ── his, verbatim ─────────────────────────────────────────────────────────── */

function smoothStep(a, b, t) {
  t = Math.max(0, Math.min(1, (t - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function length(x, y) {
  return Math.sqrt(x * x + y * y);
}

function roundedRectSDF(x, y, width, height, radius) {
  const qx = Math.abs(x) - width + radius;
  const qy = Math.abs(y) - height + radius;
  return Math.min(Math.max(qx, qy), 0) + length(Math.max(qx, 0), Math.max(qy, 0)) - radius;
}

/** His default fragment: how far each point of the backdrop is pulled toward
 *  the middle. The numbers are his and are not guesses to re-derive.
 *  @param {{x: number, y: number}} uv a point in the element, 0..1
 *  @returns {{x: number, y: number}} where that point samples from */
function fragment(uv) {
  const ix = uv.x - 0.5;
  const iy = uv.y - 0.5;
  const distanceToEdge = roundedRectSDF(ix, iy, 0.3, 0.2, 0.6);
  const displacement = smoothStep(0.8, 0, distanceToEdge - 0.15);
  const scaled = smoothStep(0, 1, displacement);
  return { x: ix * scaled + 0.5, y: iy * scaled + 0.5 };
}

/* ── ours: bind it to an element ───────────────────────────────────────────── */

/** Build the displacement map for a box, as a data: URI.
 *
 *  A data: URI and not an element reference: `feImage href="#id"` is silently
 *  ignored in Chrome, which is why an earlier attempt at this rendered nothing
 *  at all. Only the inline form displaces.
 *  @param {number} w width in CSS pixels
 *  @param {number} h height in CSS pixels
 *  @returns {{href: string, scale: number}} the map and the scale it implies */
function buildMap(w, h) {
  const data = new Uint8ClampedArray(w * h * 4);
  const raw = [];
  let maxScale = 0;

  for (let i = 0; i < data.length; i += 4) {
    const x = (i / 4) % w;
    const y = Math.floor(i / 4 / w);
    const pos = fragment({ x: x / w, y: y / h });
    const dx = pos.x * w - x;
    const dy = pos.y * h - y;
    maxScale = Math.max(maxScale, Math.abs(dx), Math.abs(dy));
    raw.push(dx, dy);
  }
  maxScale *= 0.5;

  let index = 0;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = (raw[index++] / maxScale + 0.5) * 255;
    data[i + 1] = (raw[index++] / maxScale + 0.5) * 255;
    data[i + 2] = 0;
    data[i + 3] = 255;
  }

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").putImageData(new ImageData(data, w, h), 0, 0);
  return { href: canvas.toDataURL(), scale: maxScale };
}

/** Red-to-blue separation, in CSS pixels, per unit of `disperse`. At the
 *  fitted default of 18 that is the 4.32px fringe the prism was judged at on
 *  lab/1t and lab/1v. Exported so the figure lives in one place: refresh()
 *  and the guard that holds it to 4.32 must not each carry their own copy. */
export const PX_PER_DISPERSE = 0.24;

/** The three displacement scales for one bend, red bent least.
 *
 *  Split out of refresh() because it is the whole of the optical model and
 *  none of the DOM: a pure function can be held to its invariants by a test,
 *  and the invariant below was broken in the shipped tree for want of one.
 *
 *  @param {number} capped the bend the box allows, already capped
 *  @param {number} disperse the asked-for prism, as a pixel budget
 *  @returns {{r: number, g: number, b: number}} feDisplacementMap scales
 */
export function channelScales(capped, disperse) {
  // THE SPLIT CANNOT EXCEED TWICE THE BEND. It is symmetric about `capped`, so
  // a split wider than 2 * capped sends red PAST zero and out the other side —
  // and a negative scale on an feDisplacementMap does not fail, it displaces
  // the channel the opposite way. That is not the IOR triple at a lower
  // amplitude but a different claim about the glass: red leaning the wrong
  // side of every edge. Measured on the showcase before this bound existed,
  // at the aurora's disperse of 90: ten of sixteen aurora hosts had a negative
  // red, a 32px icon button worst at -5.61 against a green of 5.19.
  // Clamping the SPLIT rather than the asked-for disperse is what keeps the
  // bound invisible where the prism already fitted: on a panel with room for
  // the full 21.6px nothing here changes, and a small host gets the widest
  // prism its own box can hold instead of a broken one.
  const split = Math.min(disperse * PX_PER_DISPERSE, 2 * capped);
  return { r: capped - split / 2, g: capped, b: capped + split / 2 };
}

/** The convex bulge's default: magnification at the pane's centre, as a
 *  fraction. The owner's pick (re-scope R5, 2026-09-25) was the shipped lens
 *  "with a slightly convex distortion", looked at from 3% to 40% on
 *  lab/r5-convex.html, and chose 40%. Readable from CSS as --slg-lens-bulge. */
export const BULGE = 0.4;

/** Where the convex stage samples from, as a shift in CSS px, for a point
 *  (px, py) measured from the pane's centre in a box of half-size (hw, hh).
 *  The shift pulls toward the centre by `m * (1 - rho²)`, rho the point's
 *  elliptical radius: strongest in the middle, zero at the rim and beyond,
 *  so it meets the rim bend without doubling it. A pure function, so the
 *  shape can be held to its invariants by a test.
 *  @returns {[number, number]} */
export function convexShift(px, py, hw, hh, m) {
  const rho2 = Math.min(1, (px / hw) ** 2 + (py / hh) ** 2);
  const k = m * (1 - rho2);
  return [-px * k, -py * k];
}

/** The convex stage's map for a w x h box.
 *  @returns {{href: string, scale: number}} */
/** Maps depend on nothing but their arguments, and a toolbar of identical
 *  buttons asks for the same one each: two PNG encodes per host per resize.
 *  Shared by reference count, so a map lives exactly as long as some lens
 *  shows it: a window drag through a hundred sizes keeps none of them. */
const mapCache = new Map();

/** Take a map, building it if no lens holds it yet.
 *  @param {string} key what the map depends on
 *  @param {() => {href: string, scale: number}} build
 *  @returns {{href: string, scale: number}} */
function acquireMap(key, build) {
  let entry = mapCache.get(key);
  if (!entry) {
    entry = { map: build(), refs: 0 };
    mapCache.set(key, entry);
  }
  entry.refs++;
  return entry.map;
}

/** Give a map back; the last holder frees it.
 *  @param {string} key */
function releaseMap(key) {
  const entry = mapCache.get(key);
  if (entry && --entry.refs <= 0) mapCache.delete(key);
}

function buildConvexMap(w, h, m) {
  const data = new Uint8ClampedArray(w * h * 4);
  const raw = new Float32Array(w * h * 2);
  let max = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [dx, dy] = convexShift(x + 0.5 - w / 2, y + 0.5 - h / 2, w / 2, h / 2, m);
      raw[(y * w + x) * 2] = dx;
      raw[(y * w + x) * 2 + 1] = dy;
      max = Math.max(max, Math.abs(dx), Math.abs(dy));
    }
  }
  const norm = max || 1;
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = (raw[i * 2] / norm) * 127.5 + 127.5;
    data[i * 4 + 1] = (raw[i * 2 + 1] / norm) * 127.5 + 127.5;
    data[i * 4 + 3] = 255;
  }
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").putImageData(new ImageData(data, w, h), 0, 0);
  // feDisplacementMap moves by scale * (C - 0.5): a full channel swing is
  // `scale`, so twice the largest shift maps 0..255 onto -max..+max.
  return { href: canvas.toDataURL(), scale: max * 2 };
}

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Give an element a lens: what is behind it bends, what is inside it does not.
 *
 * @param {Element} el the element to filter. Its content is untouched. Its
 *   own background is cleared by glass.css's `.sl-glass.is-lensed` rule, not
 *   here — a stylesheet the consumer can read and override beats an inline
 *   style they cannot.
 * @param {{blur?: number, saturate?: number, brightness?: number, disperse?: number, reach?: number}} [look]
 *   the rest of the backdrop-filter chain, in his own proportions; `disperse`
 *   amplifies the IOR triple's ±0.02 — 18 puts a visible fringe on a button,
 *   0 is a lens with no glass in it; `reach` caps the bend in CSS pixels, so a
 *   card does not bend six times harder than a button just for being wider.
 *   `disperse` is readable from CSS as `--slg-lens-disperse`, and
 *   `reach` is also readable from CSS as `--slg-lens-reach` on the element,
 *   which wins when it is declared: how hard a surface bends is a property of
 *   that surface, and a stylesheet is where a surface says what it is. The
 *   aurora's own hosts raise it, because there the thing behind the pane is
 *   light with a shape, and a bend of six pixels on it is not visible.
 * @returns {{refresh(): void, destroy(): void}|null} null where the browser
 *   cannot do it, in which case the element keeps the material glass.css gives
 *   it and nothing else changes.
 */
export function lens(el, look = {}) {
  if (typeof document === "undefined") return null;
  // Feature-detected, not assumed: Safari supports backdrop-filter but not an
  // SVG filter reference in one, and must fall through to the CSS material.
  if (!CSS.supports("backdrop-filter", "url(#x)")) return null;

  // A LENS CANNOT SEE THROUGH A GLASS ANCESTOR, and this is the browser's rule,
  // not a choice. A glass host carries `isolation: isolate` so its own layers
  // can sit under the host's label, and it usually carries a lens of its own;
  // either one makes it a Backdrop Root, and a backdrop-filter inside a
  // Backdrop Root gets an EMPTY backdrop. The pane then renders whatever its
  // own material makes of nothing, which on the light theme is a flat grey
  // slab — the defect the owner reported on a dialog's buttons.
  //
  // Isolated by subtraction on that dialog, each step a capture: the button
  // stayed grey with the displacement at 0, with the cap at 5 instead of 18,
  // with the filter region grown 24px a side, at 44px tall instead of 30, and
  // with the ANCESTOR's own lens removed. It rendered correctly only when the
  // ancestor gave up its lens AND its isolation together — which is not
  // available, since the isolation is what keeps the material under the label.
  //
  // So the innermost lens is the one that cannot be served, and it refuses
  // here rather than painting grey. Everywhere else a button keeps its lens:
  // an earlier attempt at this defect took the lens off small panes outright,
  // and that was a design decision about liquid glass which was not mine to
  // take. This is the narrow case the platform forbids, and nothing more.
  if (el.parentElement?.closest(".sl-glass, .sl-fx-glass")) return null;

  // blur 1.5 and disperse 90: A LENS YOU CAN SEE, which is the owner's ruling
  // and is not what either number was before. The two fight, and measurement
  // is what settled them. The bend is capped at a sixth of the shorter side —
  // ~5.3px on a 32px field — so a frost of 5.9 erases exactly the detail the
  // bend moved. Measured when fields were still 37px (6.2px of bend, before
  // 0.27.0): toggling the displacement off a field changed the interior by at
  // most 10/255, and forcing the bend to 18 with the frost up changed nothing
  // visible at all. The 32px field bends less, so the frost wins by more. The frost is the limiting knob, not the bend.
  // At 1.5 the copy behind a pane is a legible ghost rather than a smear, and
  // that is the cost the ruling accepts to get refraction on a small host.
  //
  // 90 is sondalab.eu's prism, 21.6px of red-to-blue, and it is safe to make
  // it the default ONLY because channelScales() bounds the split: a box with
  // no room gets the widest prism it can hold instead of a red channel
  // displaced backwards. Before that bound this default would have broken ten
  // of sixteen aurora hosts.
  const { blur = 2.5, saturate = 1.1, brightness = 1.05, disperse = 90, reach = 6 } = look;
  const id = `sl-lens-${serial++}`;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "0");
  svg.setAttribute("height", "0");
  svg.setAttribute("aria-hidden", "true");
  svg.style.cssText = "position:absolute;width:0;height:0;pointer-events:none";

  const filter = document.createElementNS(SVG_NS, "filter");
  filter.setAttribute("id", `${id}_filter`);
  filter.setAttribute("filterUnits", "userSpaceOnUse");
  filter.setAttribute("color-interpolation-filters", "sRGB");
  filter.setAttribute("x", "0");
  filter.setAttribute("y", "0");

  const feImage = document.createElementNS(SVG_NS, "feImage");
  feImage.setAttribute("id", `${id}_map`);
  feImage.setAttribute("result", "map");

  /* ── DISPERSION: the prism ─────────────────────────────────────────────────
     One displacement moves every colour by the same amount, and that is a
     lens with no glass in it. Glass bends red less than blue, so a white light
     crossing an edge opens into a spectrum — the fringe the owner means when
     he says "prism". So: split the backdrop into its three channels, displace
     each with the SAME map at a DIFFERENT scale, and add them back.

     The ratio is glass-studio's IOR triple, N_R 0.98 / N_G 1.00 / N_B 1.02 —
     but amplified. On a 96x40 button the map's own scale is four or five
     pixels, and ±2% of that is a tenth of a pixel: physically faithful and
     invisible. `disperse` is the amplification, and it is a parameter rather
     than a constant because the right amount depends on how large the pane
     is, the way glass-studio's own "dispersion" knob does. */
  const channel = (name, r, g, b) => {
    const m = document.createElementNS(SVG_NS, "feColorMatrix");
    m.setAttribute("in", "SourceGraphic");
    m.setAttribute("type", "matrix");
    m.setAttribute("values", `${r} 0 0 0 0  0 ${g} 0 0 0  0 0 ${b} 0 0  0 0 0 1 0`);
    m.setAttribute("result", name);
    return m;
  };
  const displace = (input, result) => {
    const d = document.createElementNS(SVG_NS, "feDisplacementMap");
    d.setAttribute("in", input);
    d.setAttribute("in2", "map");
    d.setAttribute("xChannelSelector", "R");
    d.setAttribute("yChannelSelector", "G");
    d.setAttribute("result", result);
    return d;
  };
  const add = (a, b, result) => {
    const c = document.createElementNS(SVG_NS, "feComposite");
    c.setAttribute("in", a);
    c.setAttribute("in2", b);
    c.setAttribute("operator", "arithmetic");
    c.setAttribute("k1", "0"); c.setAttribute("k2", "1"); c.setAttribute("k3", "1"); c.setAttribute("k4", "0");
    c.setAttribute("result", result);
    return c;
  };
  const dispR = displace("chR", "dR");
  const dispG = displace("chG", "dG");
  const dispB = displace("chB", "dB");

  /* ── THE BULGE: a slab that is convex ───────────────────────────────────
     A second, independent stage on the prism's output: the pane magnifies
     what is behind it, most at the centre and not at all at the rim (see
     convexShift). Independent so the rim bend and the prism keep exactly the
     numbers they were fitted at; only this stage's strength is new. */
  const feConvex = document.createElementNS(SVG_NS, "feImage");
  feConvex.setAttribute("result", "convexMap");
  const dispConvex = document.createElementNS(SVG_NS, "feDisplacementMap");
  dispConvex.setAttribute("in", "out");
  dispConvex.setAttribute("in2", "convexMap");
  dispConvex.setAttribute("xChannelSelector", "R");
  dispConvex.setAttribute("yChannelSelector", "G");

  filter.append(
    feImage,
    channel("chR", 1, 0, 0), dispR,
    channel("chG", 0, 1, 0), dispG,
    channel("chB", 0, 0, 1), dispB,
    add("dR", "dG", "dRG"),
    add("dRG", "dB", "out"),
    feConvex, dispConvex,
  );
  const defs = document.createElementNS(SVG_NS, "defs");
  defs.append(filter);
  svg.append(defs);
  document.body.append(svg);

  let last = "";
  /** The two maps this lens holds, released when it takes others or dies. */
  let held = [];

  /** Rebuild the map for the element's current size. Cheap to call: it returns
   *  immediately unless the box actually changed, and the box only changes on
   *  layout, never on scroll — an element and its backdrop scroll together, so
   *  there is nothing new behind it to bend.
   *  @returns {void} */
  function refresh() {
    const r = el.getBoundingClientRect();
    const w = Math.round(r.width);
    const h = Math.round(r.height);
    if (!w || !h) return;
    const key = `${w}x${h}`;
    if (key === last) return;
    last = key;

    const keys = [`p${w}x${h}`];
    const { href, scale } = acquireMap(keys[0], () => buildMap(w, h));
    // One style read for the three custom properties below.
    const cs = getComputedStyle(el);
    filter.setAttribute("width", String(w));
    filter.setAttribute("height", String(h));
    feImage.setAttribute("width", String(w));
    feImage.setAttribute("height", String(h));
    feImage.setAttribute("href", href);
    // THE SCALE IS CAPPED. His map's displacement is a fraction of the box, so
    // it grows with the element: about 5px on a 96px button and about 30px on
    // a 500px card. A prism on a button is a wave on a card — the showcase's
    // live card bent the letters behind it as if under water, and nothing
    // about the light could be judged through that. Glass does not bend
    // more because the pane is larger; a real slab bends at its edge by a
    // fixed few pixels whatever its width. `reach` is that cap, in pixels.
    // Declared on the element it belongs to, read every time the box changes.
    // Nothing is cached in a variable here: a number held in two places is the
    // first thing to drift, which is the same rule the GPU tier's thickness,
    // tint and fresnel already follow.
    const asked = parseFloat(cs.getPropertyValue("--slg-lens-reach"));
    // AND THE BOX CAPS IT TOO, at a sixth of its shorter side. A slab bends at
    // its edge by a fixed few pixels, which is what `reach` says; but "a few"
    // has to stay few RELATIVE TO THE PANE, and the aurora's hosts ask for
    // 18px. On a 494 x 139 card that is a tenth of the height and reads as
    // glass. On a 55 x 30 button it is most of the box: the filter samples far
    // outside its own region, comes back transparent, and the pane renders a
    // flat GREY slab — which is what the owner saw on the light theme, with a
    // 96 x 40 button shredding the text behind it for the same reason.
    // A sixth of the shorter side gives the card its full 18, the 40px button
    // 6.7 and the 30px one 5, all close to the material's own fitted 6. The
    // lens stays bound on every pane: taking it off a button was a design
    // decision, it was not mine to take, and it is not this defect's fix.
    const asking = Number.isFinite(asked) ? asked : reach;
    const cap = Math.min(asking, Math.min(w, h) / 6);
    const capped = Math.min(scale, cap);
    // THE COLOUR SPLIT IS A PIXEL BUDGET, not a fraction of the bend. The model
    // is still glass-studio's IOR triple — N_R 0.98 / N_G 1.00 / N_B 1.02, red
    // bent least — but a fraction of the bend is only the right shape while the
    // bend itself is physical, and ours is CAPPED for looks. At the material's
    // 6px cap the old form gave a red-to-blue separation of 6 * 0.02 * 18 * 2 =
    // 4.3px, which is the fringe the prism was fitted at; at the aurora's 18px
    // cap the same expression gave 13px, and 13px of channel separation on a
    // line of text behind a pane is not a prism, it is a broken render — the
    // owner saw it as one. So `disperse` is read as that separation in PIXELS
    // and the cap only decides how far the light bends, not how far apart its
    // colours land.
    // 0.24: at disperse 18 that is the 4.3px separation the prism was fitted
    // at on lab/1t and lab/1v, when the cap was 6 and the split was a fraction
    // of it. The first version of this line used 0.12 and halved the prism.
    // AND THE SPLIT IS READABLE FROM CSS, the way `reach` above is. It was the
     // one number of the five with no hook, so a surface that wanted a wider
     // prism had no way to ask for one — the option exists and nothing passes
     // it. Read here rather than cached, for the reason `reach` gives: a number
     // held in two places is the first thing to drift.
     //
     // The default does not move. The figure this was fitted at answers a
     // question about a pane with TEXT behind it, where the note above records
     // 13px of channel separation reading as a broken render rather than as a
     // prism; a surface laid over a picture or a bed has no such reader, and
     // that is the surface that should be raising it.
     const askedSplit = parseFloat(cs.getPropertyValue("--slg-lens-disperse"));
     // The arithmetic, and the bound on it, are channelScales() above — a pure
     // function so the invariant it holds can be executed by a guard rather
     // than read off a comment.
     const bend = channelScales(capped, Number.isFinite(askedSplit) ? askedSplit : disperse);
    dispR.setAttribute("scale", String(bend.r));
    dispG.setAttribute("scale", String(bend.g));
    dispB.setAttribute("scale", String(bend.b));

    // The bulge, read from CSS like reach and disperse; 0 turns it off.
    const askedBulge = parseFloat(cs.getPropertyValue("--slg-lens-bulge"));
    const bulge = Number.isFinite(askedBulge) ? Math.max(0, askedBulge) : BULGE;
    keys.push(`c${w}x${h}x${bulge}`);
    const convex = acquireMap(keys[1], () => buildConvexMap(w, h, bulge));
    // Acquired before the old ones go, so a lens re-taking its own map never
    // frees and rebuilds it.
    for (const k of held) releaseMap(k);
    held = keys;
    feConvex.setAttribute("width", String(w));
    feConvex.setAttribute("height", String(h));
    feConvex.setAttribute("href", convex.href);
    dispConvex.setAttribute("scale", String(bulge > 0 ? convex.scale : 0));
  }

  refresh();
  const watch = typeof ResizeObserver === "function" ? new ResizeObserver(refresh) : null;
  watch?.observe(el);

  el.classList.add("is-lensed");
  // The chain's numbers are var() fallbacks, not literals, so a surface can
  // ANIMATE them — a lens whose frost breathes is the difference between glass
  // and a photograph of glass. glass.css registers both with @property so a
  // transition or keyframe interpolates them instead of snapping.
  el.style.backdropFilter =
    `url(#${id}_filter) blur(var(--slg-lens-blur, ${blur}px)) ` +
    `saturate(var(--slg-lens-sat, ${saturate})) brightness(${brightness}) ` +
    // The calmer backdrop (glass.css, --slg-lens-contrast): last, so it softens
    // what the lens and the frost already made of the page behind.
    `contrast(var(--slg-lens-contrast, 0.72))`;

  return {
    refresh,
    destroy() {
      watch?.disconnect();
      for (const k of held) releaseMap(k);
      held = [];
      svg.remove();
      el.classList.remove("is-lensed");
      el.style.removeProperty("backdrop-filter");
      if (!el.getAttribute("style")) el.removeAttribute("style");
    },
  };
}
