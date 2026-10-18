/* @sondalab/ui-kit — goo-field.js
   The shared gooey field: several elements read as ONE SUBSTANCE.

   Contract: docs/specs/2026-09-08-effects-layer-design.md

   WHY THIS IS NOT sl-fx-liquid. The liquid parents one <svg> INTO its host,
   aligned to that host's border box, and deforms it under the pointer — "you
   are recognised". It is structurally incapable of merging with a neighbour,
   because two silhouettes can only weld if they are in the SAME filter
   coordinate space. So this family inverts the ownership: a declared
   CONTAINER owns one field, every member contributes its silhouette to it in
   the container's coordinates, and one goo filter runs over the group. The
   meaning is different too — not "you are recognised" but "these belong to
   one another" — which is why it is a family and not a looser gate on the
   liquid.

   WHAT IS UNCHANGED. The goo is on a SILHOUETTE LAYER BELOW THE CONTENT, the
   way libraries.dev/gooey does it and the way the spec has required twice:
   the real DOM rides crisp on top and is never filtered. A member contributes
   its rounded box, dilated by a few pixels so the substance peeks out as a
   rim; its text, its border and its own background are untouched.

   COLOUR. Each member's silhouette is painted in that member's own computed
   background colour, so hover and active states stay the component's. The goo
   chain blurs premultiplied RGBA and then steepens ONLY the alpha, so where
   two members weld their colours blend across the neck on their own — no GPU,
   no gradient, no second pass. A member with no fill of its own (a bare wrapper,
   anything whose glass paints on a pseudo-element) falls back to the field's
   declared --slfx-goo-fill.
   sl-audit reads CSS only, so there is no colour literal in this file:
   every value is read from a computed style or from a --slfx-* binding. */

import {
  cornerPx, roundedRectPathAt, fieldOrigin, memberShape,
  weldPlan, weldedMembers, filterRegion, GOO_CONTRAST, GOO_INTERCEPT,
} from "./geom.js";

const NS = "http://www.w3.org/2000/svg";

/** What contributes a silhouette. A member is MARKED IN THE MARKUP, never
 *  inferred from "every child of the container": the same ruling that made the
 *  liquid's --drop a class rather than a measured radius. A heuristic — every
 *  child, or every child above some size — misfires the moment a consumer
 *  wraps two members in a layout div, and the author loses the ability to keep
 *  one control out of the substance. */
const MEMBERS = ".sl-fx-goo__member";

/** What may never be a member, however it is marked.
 *
 *  A form control and a link paint their own chrome — a border, a focus ring,
 *  a hover fill, and on a replaced control a box the UA owns outright — and
 *  they carry it through layers this family does not own: a glass pane's
 *  hairline lives on `::after` and in the children `glass.css` injects, so a
 *  welded button went on drawing the outline the substance was supposed to
 *  replace, while a welded `<div>` next to it lost its edge entirely. The two
 *  did not look alike, and the one that looked wrong was always the control.
 *
 *  The owner's ruling (2026-09-12) is to remove the case rather than paper
 *  over it: the goo is for standard elements, and a control keeps its own
 *  affordance. Excluding by TAG is not the heuristic the liquid's `--drop`
 *  gate exists to avoid — a tag name is a stable semantic fact, where a
 *  measured `border-radius` is a value a component may re-parametrise at any
 *  time. A `div[role="button"][tabindex="0"]` is deliberately still eligible:
 *  it paints nothing of its own, so the substance can be its whole surface,
 *  and the drag board depends on it. */
const EXCLUDED = "button, input, select, textarea, a[href]";

/** A field whose members move on their own. Opt-in, because it is the one
 *  thing this family does that a laid-out page does not already imply, and
 *  because it costs a frame loop for as long as it is on screen.
 *
 *  The margin is the one watchNear() uses: a field that starts drifting only
 *  once it is strictly on screen starts from a standing position in view. */
const DRIFT = ".sl-fx-goo--drift";
const DRIFT_MARGIN = "200px";
const REDUCE = "(prefers-reduced-motion: reduce)";

/** Conservative defaults, so a container that opts in without binding anything
 *  renders the most restrained version of the effect rather than a broken one.
 *  effects.css declares both; these exist for a caller that reaches for the
 *  module directly. */
const REACH_FALLBACK = 3;
const WELD_FALLBACK = 14;
const SOFT_FALLBACK = 5;

let fieldSeq = 0;

/** Write an attribute only when it would change.
 *
 *  Not a micro-optimisation. Setting an attribute on an element inside an SVG
 *  filter marks the filter dirty and the browser re-rasterises the whole filter
 *  region — a Gaussian blur over 0.1 to 0.6 megapixels at device-pixel-ratio 2,
 *  measured across the showcase's own fields. The interaction loop re-derives
 *  every value each frame because a member may have moved, but on a resting
 *  hover none of them differ, and that re-raster is paid for nothing.
 *
 *  The JavaScript was never the cost here, which is why this is the shape the
 *  fix takes: measure() itself is 0.138 ms per call on a three-member field.
 *  @param {Element} el
 *  @param {string} name
 *  @param {string} value
 *  @returns {void} */
function setAttr(el, name, value) {
  if (el.getAttribute(name) !== value) el.setAttribute(name, value);
}

/** Tell a member which member it is, so the drift can give each one its own
 *  phase and its own period.
 *
 *  Written by the field rather than authored, unlike the glitch's shards: the
 *  order that matters is COLLECTION order — what querySelectorAll returns
 *  across the whole subtree — and CSS cannot name it. `:nth-child` counts
 *  against a member's own parent, so the moment a consumer wraps two members
 *  in a layout div every one of them lands in the same bucket and the field
 *  marches in lockstep.
 *  @param {Element} el the member
 *  @param {number} i its index in collection order */
function stampIndex(el, i) {
  const v = String(i);
  if (el.style.getPropertyValue("--slfx-i") !== v) el.style.setProperty("--slfx-i", v);
  // Neighbours swing against each other. Parity of the collection index, not
  // :nth-child — wrapped members would all land in one bucket (see above).
  const dir = i % 2 ? "reverse" : "normal";
  if (el.style.getPropertyValue("--slfx-dir") !== dir) el.style.setProperty("--slfx-dir", dir);
}

/** Read a length binding off the container, in CSS pixels.
 *  @param {CSSStyleDeclaration} cs the container's computed style
 *  @param {string} name the custom property
 *  @param {number} fallback used when the property is unset or unparseable
 *  @returns {number} */
function lengthVar(cs, name, fallback) {
  const v = parseFloat(cs.getPropertyValue(name));
  return Number.isFinite(v) ? v : fallback;
}

/** Whether a computed colour has zero alpha, without naming a colour syntax.
 *  getComputedStyle returns either a comma form ending in the alpha or a
 *  slash form; both are matched by position rather than by function name, so
 *  this keeps working as Chrome moves its serialisation to color().
 *  @param {string} c a computed colour string
 *  @returns {boolean} */
function isClear(c) {
  const m = /\/\s*([\d.]+)%?\s*\)\s*$/.exec(c) || /,\s*([\d.]+)\s*\)\s*$/.exec(c);
  return m ? parseFloat(m[1]) === 0 : false;
}

/** The same colour with its alpha slot removed, by string surgery rather than
 *  by naming a colour syntax: the trailing `/ a` of the modern form and the
 *  trailing `, a` of the legacy one are the only two shapes an alpha takes in
 *  a computed value, and dropping either leaves a function the parser still
 *  accepts (rgb() and rgba() take three components or four).
 *
 *  The silhouette must be OPAQUE. The goo's alpha slope maps everything below
 *  alpha 0.5 to nothing, so a member whose own fill is translucent contributes
 *  a silhouette the filter then erases — the substance thinning or vanishing
 *  under one member and not its neighbour, which reads as a rendering fault
 *  rather than as a material. (Measured 2026-09-12: the kit's own glass
 *  classes report a FULLY transparent background-color — they paint on a
 *  pseudo-element — so a glass member takes the field's --slfx-goo-fill
 *  through isClear() and never reaches here. This guards the partial alphas a
 *  consumer's own component brings.) Translucency belongs to the member's
 *  crisp box riding on top; the substance underneath is one material, and a
 *  material is not see-through in places.
 *  @param {string} c a computed colour string
 *  @returns {string} the same colour at full alpha */
function opaque(c) {
  const slash = c.replace(/\s*\/\s*[\d.]+%?\s*\)\s*$/, ")");
  if (slash !== c) return slash;
  // Only a FOURTH comma component is an alpha. Stripping the last one
  // unconditionally turned an already-opaque rgb(15, 21, 26) into
  // rgb(15, 21) — every silhouette losing its blue channel.
  return c.replace(/^([^(]+\([^,()]+,[^,()]+,[^,()]+),\s*[\d.]+%?\s*\)\s*$/, "$1)");
}

/** The four painted radii of an element, clockwise from top-left.
 *  @param {CSSStyleDeclaration} cs the element's computed style
 *  @param {number} w its border-box width
 *  @param {number} h its border-box height
 *  @returns {number[]} */
function radiiOf(cs, w, h) {
  return [
    cornerPx(cs.borderTopLeftRadius, w, h),
    cornerPx(cs.borderTopRightRadius, w, h),
    cornerPx(cs.borderBottomRightRadius, w, h),
    cornerPx(cs.borderBottomLeftRadius, w, h),
  ];
}

/**
 * Give a container one shared gooey field: its marked members stop reading as
 * separate boxes and start reading as one substance, welding where they come
 * within the declared gap of each other.
 *
 * Nothing is written to a member. The field is an injected first child that
 * takes no pointer, holds no text, and is removed entirely by destroy().
 *
 * @param {Element} container an element carrying .sl-fx-goo
 * @returns {{measure: () => void, destroy: () => void} | null} a handle, or
 *   null if the container is not eligible (already has a field, or sits under
 *   the subtree opt-out)
 */
export function gooField(container) {
  if (!(container instanceof Element) || !container.isConnected) return null;
  if (container.querySelector(":scope > .sl-fx-goo__field")) return null;
  if (container.closest('[data-sl-fx="off"]')) return null;

  const id = `slfx-goo-${++fieldSeq}`;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "sl-fx-goo__field");
  svg.setAttribute("aria-hidden", "true");
  // sRGB, not the SVG default of linearRGB: the alpha slope would otherwise be
  // applied after a gamma decode that also shifts every blended colour in the
  // neck away from what the CSS around it paints.
  svg.innerHTML = `
    <defs>
      <filter id="${id}" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
        <feGaussianBlur in="SourceGraphic" stdDeviation="0" />
        <feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 ${GOO_CONTRAST} ${GOO_INTERCEPT}" />
      </filter>
    </defs>
    <g filter="url(#${id})"></g>`;
  const filter = svg.querySelector("filter");
  const blur = svg.querySelector("feGaussianBlur");
  const group = svg.querySelector("g");
  const defs = svg.querySelector("defs");
  container.prepend(svg);

  let members = [];
  let paths = [];
  let lastShapes = [];
  let alive = true;
  // The CSS animation is gated on prefers-reduced-motion in effects.css, and
  // this is the same gate on the JS half. It has to be read here and not left
  // to mount(): mountGoo() runs BEFORE mount()'s motion guard, because the goo
  // animates nothing and refusing it for a motion preference would take away
  // material rather than movement. The drift is the exception — it is movement
  // — so it carries its own refusal. Without this the loop would run forever
  // re-deriving boxes a suppressed animation never moves: invisible, and a
  // forced layout every frame.
  const drifts = container.matches(DRIFT)
    && !(typeof matchMedia === "function" && matchMedia(REDUCE).matches);

  /** The members this field will draw: marked in the markup, minus the ones a
   *  field may never hold. A skipped element is left completely alone — no
   *  class, no silhouette — the way a bare `.sl-fx-liquid` is inert.
   *  @returns {Element[]} */
  function collect() {
    return [...container.querySelectorAll(MEMBERS)].filter((el) => !el.matches(EXCLUDED));
  }

  /** Give every member back its own edge. The class is the only thing this
   *  family writes onto an element it did not create, and a field that is torn
   *  down has to leave no trace of it. */
  function unweld() {
    for (const el of collect()) el.classList.remove("is-goo-welded");
  }

  /** Match the pool of <path> elements to the members: TWO each, the same
   *  outline drawn twice.
   *
   *  The first is the SUBSTANCE — the field's declared fill, opaque — and the
   *  second is the member's own FACE, painted exactly as the component painted
   *  it, alpha included. That is compositing, and it is why the pair exists:
   *  the goo's alpha slope erases everything below 0.5, so a member the
   *  component tinted rather than filled has to stand on something. Forcing
   *  its own colour opaque instead was tried and is worse than the disease —
   *  `sl-tag` is the accent at low alpha, and at full alpha it is no longer a
   *  tint but the accent itself, so a row of quiet tags rendered as three
   *  saturated slabs. A translucent face over the substance keeps the colour
   *  the component chose and gives the filter an opaque silhouette to cut. */
  function resize(n) {
    while (paths.length < n * 2) {
      const p = document.createElementNS(NS, "path");
      group.append(p);
      paths.push(p);
    }
    while (paths.length > n * 2) paths.pop().remove();
  }

  /** An image member's paint is its CONTENT, not a background: an <img> has no
   *  fill of its own, and falling back to --slfx-goo-fill would drop a picture
   *  into the substance as a flat rectangle. So its silhouette is filled with
   *  the picture itself, through a pattern locked to that silhouette's box. The
   *  goo then blurs the picture like any other paint, which is what lets an
   *  image weld to a neighbour with its own colours running into the neck.
   *
   *  `slice` rather than `meet`: the silhouette is the element's box dilated by
   *  reach, and a picture letterboxed inside it would show the substance's
   *  fallback in the bars.
   *  @param {HTMLImageElement} img the member
   *  @param {{x: number, y: number, w: number, h: number}} shape its silhouette
   *  @param {number} i the member's index, which names its pattern
   *  @returns {string} a `fill` value */
  function imageFill(img, shape, i) {
    const patternId = `${id}-img-${i}`;
    let pattern = defs.querySelector(`#${patternId}`);
    if (!pattern) {
      pattern = document.createElementNS(NS, "pattern");
      pattern.setAttribute("id", patternId);
      pattern.setAttribute("patternUnits", "userSpaceOnUse");
      pattern.append(document.createElementNS(NS, "image"));
      defs.append(pattern);
    }
    const picture = pattern.querySelector("image");
    picture.setAttribute("preserveAspectRatio", "xMidYMid slice");
    // The tile is placed in user space; the picture is placed in the TILE's own
    // space, which starts at the tile's origin. Giving the picture the same
    // absolute x/y as the pattern offsets it twice, so every tile shows the
    // wrapped-around part of the image — the picture still looks present, and
    // its edges, which are the only part the weld blends, are the wrong ones.
    setAttr(pattern, "x", String(shape.x));
    setAttr(pattern, "y", String(shape.y));
    setAttr(picture, "x", "0");
    setAttr(picture, "y", "0");
    for (const [k, v] of [["width", shape.w], ["height", shape.h]]) {
      setAttr(pattern, k, String(v));
      setAttr(picture, k, String(v));
    }
    // href matters more than the rest: re-setting it to the same value does not
    // merely dirty the filter, it restarts the load and the decode in Chrome. A
    // 220x132 PNG decoded per frame blocked the main thread for seconds.
    setAttr(picture, "href", img.currentSrc || img.src);
    return `url(#${patternId})`;
  }

  /** Write each member's substance and face. Geometry is untouched, so this is
   *  what a hover or an active state costs.
   *  @param {string} fallback the field's declared substance */
  function paint(fallback, restore) {
    const ground = opaque(fallback);
    for (let i = 0; i < members.length; i++) {
      const el = members[i];
      setAttr(paths[i * 2], "fill", ground);
      const face = paths[i * 2 + 1];
      // A member that came into this frame already welded has handed its
      // background to the substance, so it reports no colour of its own. Its
      // face already holds the colour it had when it welded — the frame it
      // welded on read it, because the class goes on at the END of measure().
      // Reading it again would write the fallback over a coloured chip.
      if (!restore && el.classList.contains("is-goo-welded")) continue;
      if (el instanceof HTMLImageElement && lastShapes[i]) {
        setAttr(face, "fill", imageFill(el, lastShapes[i], i));
        continue;
      }
      const own = getComputedStyle(el).backgroundColor;
      // A member with no fill of its own is the substance and nothing else;
      // painting the face too would only double a colour already there.
      setAttr(face, "fill", isClear(own) ? ground : own);
    }
  }

  /** Re-read the layout and rebuild every silhouette. Called on mount, on a
   *  resize of the container or any member, and whenever members are added or
   *  removed. Not called per frame: a laid-out field is static. */
  function measure(restore = false) {
    if (!alive) return;
    members = collect();
    resize(members.length);
    if (!members.length) return;

    // Hand every member its own paint back before reading it, when the caller
    // asked for a true read: a welded member has given its background to the
    // substance (effects.css), so it reports a clear background-color for as
    // long as the class is on it.
    //
    // NOT on every frame. Doing it unconditionally was the first version and
    // it made hover feel broken: .sl-btn transitions `background`, so removing
    // and re-adding the class each frame restarted a 120ms transition forever,
    // and the button's own hover and leave never arrived. The frame loop
    // therefore leaves welded members alone — their face already holds the
    // colour they had when they welded, read on the frame they welded, because
    // the class goes on at the END of this function. A theme change, a resize
    // or a new member calls measure(true) and gets the true read.
    if (restore) unweld();

    const cs = getComputedStyle(container);
    const fallback = cs.getPropertyValue("--slfx-goo-fill").trim() || cs.backgroundColor;
    // Dilation preserves the shape and blur melts it, so the weld distance is
    // spent on the first and only capped amounts of the second. See weldPlan.
    const { sigma, reach } = weldPlan(
      lengthVar(cs, "--slfx-goo-weld", WELD_FALLBACK),
      lengthVar(cs, "--slfx-goo-reach", REACH_FALLBACK),
      lengthVar(cs, "--slfx-goo-soft", SOFT_FALLBACK),
    );

    const origin = fieldOrigin(
      container.getBoundingClientRect(),
      parseFloat(cs.borderLeftWidth) || 0,
      parseFloat(cs.borderTopWidth) || 0,
    );

    const shapes = members.map((el) => {
      const rect = el.getBoundingClientRect();
      const radii = radiiOf(getComputedStyle(el), rect.width, rect.height);
      return memberShape(rect, origin, radii, reach);
    });

    lastShapes = shapes;
    if (drifts) for (let i = 0; i < members.length; i++) stampIndex(members[i], i);
    // A member that has joined a mass lets the mass carry the edge. Its own
    // border would otherwise draw a line straight across the merged shape —
    // three chips dragged together read as three cards stacked on a blob
    // rather than as one substance, which is the opposite of what this family
    // claims. A member standing on its own keeps its border: it is still a
    // button, and the substance behind it is a rim, not a replacement.
    const welded = weldedMembers(shapes, sigma);
    for (let i = 0; i < shapes.length; i++) {
      const s = shapes[i];
      const d = roundedRectPathAt(s.x, s.y, s.w, s.h, s.radii);
      setAttr(paths[i * 2], "d", d);
      setAttr(paths[i * 2 + 1], "d", d);
    }
    setAttr(blur, "stdDeviation", sigma.toFixed(3));

    const region = filterRegion(shapes, sigma);
    setAttr(filter, "x", region.x.toFixed(2));
    setAttr(filter, "y", region.y.toFixed(2));
    setAttr(filter, "width", region.width.toFixed(2));
    setAttr(filter, "height", region.height.toFixed(2));

    paint(fallback, restore);

    // Last, once every colour has been read: the welded members hand their
    // border and their background back to the substance.
    for (let i = 0; i < members.length; i++) members[i].classList.toggle("is-goo-welded", welded[i]);
  }

  // The substance is static once laid out, so there is no animation frame at
  // rest. The one thing that does change without a layout change is a member's
  // own colour — a hover or an active state, usually through a transition, and
  // a rim still painted in the resting colour reads as a misprint. So colours
  // are re-read per frame WHILE THE POINTER IS IN THE CONTAINER, and not
  // otherwise. Geometry is not touched by this loop.
  let raf = 0;
  const tick = () => {
    if (!alive) return;
    // measure(), not paint(): a MOVE is the one change nothing reports. A
    // ResizeObserver sees a box change size, never change place; a transform,
    // an inset write or a reflow above the field mutates no childList and
    // fires no resize. Re-reading only the colours was enough for a hover and
    // wrong for everything else — a dragged chip left its silhouette behind,
    // and so did one nudged with the arrow keys, which fires no pointer event
    // at all. "These are one substance" is a claim about where they are, so
    // the loop re-derives where they are.
    measure();
    raf = requestAnimationFrame(tick);
  };
  // A pointer is DOWN somewhere in this field. Not a mode — the loop does the
  // same work either way — but a reason to keep it running: a drag that uses
  // setPointerCapture retargets boundary events to the captured element, so
  // the container sees a pointerleave the instant the drag begins, and a loop
  // that stopped there left the silhouette at the position the chip started
  // from while the chip itself went on moving.
  let held = false;
  // A drifting field that is on screen never comes to rest, so every reason to
  // stop — the pointer leaving, a drag ending, focus going elsewhere — stops
  // nothing. Leaving the stage does.
  let onStage = false;
  const start = () => { if (!raf) raf = requestAnimationFrame(tick); };
  const halt = () => { cancelAnimationFrame(raf); raf = 0; };
  const stop = () => { if (!onStage) halt(); };
  const onDown = () => { held = true; start(); };
  const onUp = () => {
    // The release is caught on the window, so every click on the page reached
    // every field and re-measured and unwelded it. Only a field that saw the
    // press has a position to land.
    if (!held) return;
    held = false;
    measure(true);                                // land the released position
    if (!container.matches(":hover")) stop();     // the pointer left during the drag
  };
  const onBlur = () => { if (!held) stop(); };
  const onLeave = (e) => {
    // Delegated: pointerleave fires for every descendant as the pointer moves
    // between members. Only the container's own leave ends the loop — the same
    // guard the liquid and the pane both carry — and not while a drag is live.
    if (e.target !== container || held) return;
    stop();
  };
  // Focus as well as the pointer. Nothing in components.css changes a
  // background on :focus-visible today — the shared focus ring is an outline —
  // but a consumer's component may, and a member whose fill changed with no
  // pointer event is the staleness that made a light-theme label vanish,
  // arriving through the keyboard instead of through the theme. focusin and
  // focusout bubble, so one pair on the container covers every member.
  // Pointer OR focus: a member can be moved with the arrow keys as readily as
  // with a drag, and a keyboard move fires no pointer event. Both start the
  // same loop, and it runs only while one of them is inside the container.
  container.addEventListener("pointerenter", start, { passive: true });
  container.addEventListener("pointerleave", onLeave, { capture: true, passive: true });
  container.addEventListener("focusin", start, { passive: true });
  container.addEventListener("focusout", onBlur, { passive: true });
  // The press is caught on the container; the release can happen anywhere —
  // outside the field, or outside the window — so it is caught on the window.
  container.addEventListener("pointerdown", onDown, { passive: true });
  window.addEventListener("pointerup", onUp, { passive: true });
  window.addEventListener("pointercancel", onUp, { passive: true });

  const ro = typeof ResizeObserver === "function" ? new ResizeObserver(() => measure(true)) : null;
  // measure() adds and removes <path> elements, which are themselves childList
  // mutations inside the container — so a callback that did not exclude the
  // field's own subtree would re-enter itself forever. The field is the one
  // subtree its own observer must not see.
  const mo = typeof MutationObserver === "function"
    ? new MutationObserver((records) => {
        if (!records.some((r) => r.target !== svg && !svg.contains(r.target))) return;
        observeMembers();
        measure(true);
        awaitImages();
      })
    : null;

  /** Watch every member's box, not only the container's: a member can change
   *  size on its own (a button whose label loads, an image that decodes) with
   *  the container's own box never moving. */
  function observeMembers() {
    if (!ro) return;
    ro.disconnect();
    ro.observe(container);
    for (const el of collect()) ro.observe(el);
  }

  /** Re-measure once every image member has decoded.
   *
   *  An <img> whose bytes have not arrived contributes an empty pattern, and a
   *  CSS-sized <img> never changes its box when they do — so the ResizeObserver
   *  would never fire and the hole would be permanent. decode() is the event
   *  that actually matters here, and a failed one is simply skipped: that
   *  member keeps the substance's fallback fill, which is the right reading of
   *  a picture that did not load. */
  function awaitImages() {
    const pending = members
      .filter((el) => el instanceof HTMLImageElement && !el.complete)
      .map((el) => el.decode().catch(() => {}));
    if (pending.length) Promise.all(pending).then(() => measure(true));
  }

  observeMembers();
  mo?.observe(container, { childList: true, subtree: true });
  measure(true);
  awaitImages();
  container.classList.add("is-goo-live");

  /** The drift's own gate: the loop runs while the field is on screen.
   *
   *  FAILS OPEN. The stage is taken to be occupied before the observer is
   *  asked, so an environment that never delivers an entry — and there is one,
   *  the automation tab this layer was debugged in — drifts rather than sits
   *  still. The failure that matters here is a field frozen for a visitor who
   *  can see it; a loop running for one who cannot is waste, and waste is the
   *  cheaper of the two to be wrong about. */
  let io = null;
  if (drifts) {
    onStage = true;
    start();
    if (typeof IntersectionObserver === "function") {
      io = new IntersectionObserver((entries) => {
        onStage = entries[entries.length - 1].isIntersecting;
        if (onStage) start();
        else if (!held) halt();
      }, { rootMargin: DRIFT_MARGIN });
      io.observe(container);
    }
  }

  return {
    measure,
    destroy() {
      alive = false;
      onStage = false;
      cancelAnimationFrame(raf);
      ro?.disconnect();
      mo?.disconnect();
      io?.disconnect();
      container.removeEventListener("pointerenter", start);
      container.removeEventListener("pointerleave", onLeave, { capture: true });
      container.removeEventListener("focusin", start);
      container.removeEventListener("focusout", onBlur);
      container.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      container.classList.remove("is-goo-live");
      for (const el of collect()) { el.style.removeProperty("--slfx-i"); el.style.removeProperty("--slfx-dir"); }
      unweld();
      svg.remove();
    },
  };
}
