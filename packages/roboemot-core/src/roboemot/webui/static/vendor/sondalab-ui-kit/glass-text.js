/* @sondalab/ui-kit — glass-text.js
 *
 * Cut a glass pane to its own letterforms, so the TEXT is the glass rather
 * than a box containing it.
 *
 * A mask clips a `backdrop-filter`'s output, which is the whole trick: mask a
 * pane to its glyphs and what is left is glass in the shape of the words. The
 * host keeps every knob the material already has — the blur, the tint, the
 * lens — and loses only the box it used to be.
 *
 * The mask has to be an SVG `<text>` living IN the document. A data URI has no
 * access to the page's webfonts and silently falls back to a system face,
 * which is a different shape from the one the element measured itself from;
 * the mask then sits beside the glyphs instead of on them.
 *
 * The geometry is measured, never declared. `maskContentUnits="objectBoundingBox"`
 * was tried and is a trap: an SVG `font-size` in bounding-box units scales by
 * the NORMALISED DIAGONAL, not by the height, so on a box that is not square
 * the glyphs come out stretched and oversized and the mask renders as a slab.
 * `userSpaceOnUse` with real pixels has no such surprise, and the pixels come
 * from the element itself — so the face, the size and the tracking can change
 * without this module knowing anything about them.
 *
 * ONE LINE. The mask is a single `<text>` run and SVG text does not wrap: a
 * host that wraps onto a second line gets its first line cut and the rest
 * invisible. Give the host `width: max-content` (the modifier does) and keep
 * the string short enough to mean something at display size.
 */

/** The `<defs>` every mask lives in. Created on first use and owned here, the
 *  way effects.js owns its one shared light canvas: a consumer cannot be asked
 *  to hand-author an empty `<svg>` before the module will work. */
let defs = null;

/** Monotonic, so two hosts with the same text still get two masks. */
let serial = 0;

/** A canvas is the only thing that reports the REAL ascent of this face at
 *  this size. A guessed ratio (0.7 em, cap height, whatever) is wrong per
 *  face and the figure sits visibly high or low in its own box. */
let ruler = null;

function defsHost() {
  if (defs?.isConnected) return defs;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("width", "0");
  svg.setAttribute("height", "0");
  // Out of flow and out of the way: it paints nothing and must not be able to
  // give the page a scrollbar.
  svg.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;pointer-events:none";
  defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  svg.append(defs);
  document.body.append(svg);
  return defs;
}

/** Re-cut one host's mask to the text and the box it has right now.
 *  @param {HTMLElement} el the glass host
 *  @param {string} id the mask's id, stable for the life of the host */
function cut(el, id) {
  const box = el.getBoundingClientRect();
  if (!box.width || !box.height) return;
  const cs = getComputedStyle(el);
  const text = el.textContent ?? "";

  ruler ??= document.createElement("canvas").getContext("2d");
  if (!ruler) return;
  ruler.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  const m = ruler.measureText(text);
  const ascent = m.actualBoundingBoxAscent;
  const descent = m.actualBoundingBoxDescent;

  const host = defsHost();
  let mask = host.querySelector(`#${CSS.escape(id)}`);
  if (!mask) {
    mask = document.createElementNS("http://www.w3.org/2000/svg", "mask");
    mask.id = id;
    mask.setAttribute("maskUnits", "userSpaceOnUse");
    mask.setAttribute("maskContentUnits", "userSpaceOnUse");
    mask.append(document.createElementNS("http://www.w3.org/2000/svg", "text"));
    host.append(mask);
  }
  mask.setAttribute("x", "0");
  mask.setAttribute("y", "0");
  mask.setAttribute("width", String(box.width));
  mask.setAttribute("height", String(box.height));

  const t = mask.firstElementChild;
  t.textContent = text;
  t.setAttribute("fill", "white");
  // CENTRED BY THE RENDERER, not by a measured width. The canvas's advance
  // ignores `letter-spacing`, so centring on it is right for a numeral at
  // -0.02em and visibly wrong for a label tracked at 0.2em. `text-anchor:
  // middle` makes SVG lay the run out with the real advance and centre it,
  // and the host is `width: max-content`, so its box IS the run plus the
  // symmetric padding below — the two agree for any tracking.
  t.setAttribute("text-anchor", "middle");
  t.setAttribute("x", String(box.width / 2));
  // Vertically: centre the ink, then drop to the baseline. Measured ink, not
  // the line box — `line-height` on a display figure is usually well under 1.
  t.setAttribute("y", String((box.height - (ascent + descent)) / 2 + ascent));
  for (const [k, v] of [
    ["font-family", cs.fontFamily],
    ["font-style", cs.fontStyle],
    ["font-weight", cs.fontWeight],
    ["font-size", cs.fontSize],
    ["letter-spacing", cs.letterSpacing],
  ]) t.setAttribute(k, v);

  el.style.setProperty("-webkit-mask-image", `url(#${id})`);
  el.style.setProperty("mask-image", `url(#${id})`);
}

/** The type the glow has to borrow so its box is the host's box.
 *
 *  Not a convenience. The glow is a bare span the module created, so it
 *  inherits the SURFACE's body type, not the host's display face: left alone
 *  it laid a 14px halo under an 88px figure — the light in one place and the
 *  glass in another. Copied inline rather than asked of the consumer, because
 *  a surface that has to restate the face on a second element has two places
 *  to keep in step and no way to be told when they drift. */
const BORROWED = [
  "font-family", "font-size", "font-weight", "font-style", "font-stretch",
  "letter-spacing", "line-height", "text-transform", "word-spacing",
  "padding-top", "padding-right", "padding-bottom", "padding-left",
];

/** The unmasked copy that carries the light. It is a SIBLING, seated before
 *  the host the way the aurora's glow is: the host is masked to the glyphs, so
 *  a glow inside it would be masked away with everything else. Created here
 *  rather than asked of the consumer, because the two copies must carry the
 *  same string and a consumer who writes both by hand has two places to keep
 *  in step — the site's first version let the glass decode while the halo
 *  stayed still, which reads as the text changing BEHIND its own light.
 *
 *  WHERE it goes is still the surface's: the pair has to coincide, and only the
 *  surface knows whether that means absolute placement in a slot, a grid cell
 *  or a stack. The module guarantees the two boxes are the same SIZE and carry
 *  the same string; putting them in the same place is one rule in the sheet.
 *  @param {HTMLElement} el @returns {HTMLElement} */
function seatGlow(el) {
  const prev = el.previousElementSibling;
  if (prev?.classList.contains("sl-fx-glass__glow")) return prev;
  const glow = document.createElement("span");
  glow.className = "sl-fx-glass__glow";
  glow.setAttribute("aria-hidden", "true");
  el.before(glow);
  return glow;
}

/** Cut one host's glass to its own letterforms.
 *
 *  @param {Element} el a `.sl-fx-glass.sl-fx-glass--text` host
 *  @returns {{ cut(): void, destroy(): void } | null} null if it is refused */
export function glassText(el) {
  if (!(el instanceof HTMLElement)) return null;
  if (el.closest('[data-sl-fx="off"]')) return null;

  const id = `sl-glass-text-${++serial}`;
  const glow = seatGlow(el);
  const recut = () => {
    if (glow.textContent !== el.textContent) glow.textContent = el.textContent;
    const cs = getComputedStyle(el);
    for (const prop of BORROWED) glow.style.setProperty(prop, cs.getPropertyValue(prop));
    cut(el, id);
  };

  const ro = typeof ResizeObserver === "function" ? new ResizeObserver(recut) : null;
  ro?.observe(el);
  // characterData + subtree, because a glitch scramble rewrites a text node in
  // place rather than replacing the element's child list.
  const mo = typeof MutationObserver === "function" ? new MutationObserver(recut) : null;
  mo?.observe(el, { characterData: true, childList: true, subtree: true });

  recut();
  // The face has to have ARRIVED, or the ascent belongs to the fallback font
  // and the figure sits wrong until something else happens to re-cut it.
  document.fonts?.ready.then(recut);

  return {
    cut: recut,
    destroy() {
      ro?.disconnect();
      mo?.disconnect();
      defs?.querySelector(`#${CSS.escape(id)}`)?.remove();
      el.style.removeProperty("-webkit-mask-image");
      el.style.removeProperty("mask-image");
      if (!el.getAttribute("style")) el.removeAttribute("style");
      glow.remove();
    },
  };
}
