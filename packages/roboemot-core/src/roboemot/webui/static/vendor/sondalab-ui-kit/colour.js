/* @sondalab/ui-kit — colour.js (internal, not an export)
   The kit's one way to turn a CSS colour into sRGB numbers, shared by
   live-light.js and halftone.js. A computed colour is not always sRGB bytes:
   a colour mix or a relative colour serialises in Oklab or as a colour
   function, and reading its numbers as bytes gave black (spexr, 0.15.0). So
   it is painted on a one-pixel 2D canvas, which converts any CSS colour to
   sRGB, and the pixel is read back. One canvas per document. isChromatic()
   below is the exception, and says why; effects.js uses it for ink. */

const swatches = new WeakMap();

/**
 * Any CSS colour string, as sRGB 0–1, painted and read back. `[0, 0, 0]`
 * where a 2D context is unavailable.
 * @param {Document} doc @param {string} css
 * @returns {[number, number, number]}
 */
export function toSrgb(doc, css) {
  let g = swatches.get(doc);
  if (g === undefined) {
    g = doc.createElement("canvas").getContext("2d", { willReadFrequently: true });
    swatches.set(doc, g);
  }
  if (!g) return [0, 0, 0];
  g.clearRect(0, 0, 1, 1);
  g.fillStyle = css;
  g.fillRect(0, 0, 1, 1);
  const [r, gr, b] = g.getImageData(0, 0, 1, 1).data;
  return [r / 255, gr / 255, b / 255];
}

/**
 * A role variable's resolved sRGB, read through a probe element on `host`,
 * so a band or a product override that remaps the role is followed.
 * @param {Element} host @param {string} prop e.g. "--slc-accent"
 * @returns {[number, number, number]}
 */
export function roleColour(host, prop) {
  const doc = host.ownerDocument;
  const probe = doc.createElement("span");
  probe.style.cssText = `position:absolute;visibility:hidden;color:var(${prop})`;
  host.append(probe);
  const css = getComputedStyle(probe).color;
  probe.remove();
  return toSrgb(doc, css);
}

/* The one place the kit READS a colour string rather than painting it. A
   paint-and-read swatch cannot answer this question: it needs a document, and
   it flattens a wide-gamut colour into sRGB bytes where its chroma is lost.
   The question is only "is there a hue here", so a tolerant read of the
   notation suffices -- and anything this does not recognise answers no, which
   lets the caller fall back to the accent rather than shimmer in a guess. */

/** The spaces of color() whose three channels are red, green and blue, 0-1.
 *  xyz is not one: D65 white is 0.95 1 1.09, a spread a grey would fail. */
const RGB_SPACES = new Set(["srgb", "srgb-linear", "display-p3", "a98-rgb", "prophoto-rgb", "rec2020"]);
/** A channel: a plain, signed, fractional or exponent number, or `none`. */
const CHANNEL = /-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?|none/gi;

/**
 * Whether a computed CSS colour carries a hue rather than a neutral grey.
 * Reads the notation the browser serialised: chroma for oklch/lch, the a-b
 * distance for oklab/lab, the red-green-blue spread for rgb() and the RGB
 * spaces of color(). `none` is 0. Anything else (a keyword, xyz, hsl) is false.
 * @param {string} css a computed colour, e.g. getComputedStyle(el).color
 * @returns {boolean}
 */
export function isChromatic(css) {
  const m = /^\s*([a-z]+)\(\s*(.*)\)\s*$/i.exec(String(css));
  if (!m) return false;
  const fn = m[1].toLowerCase();
  let body = m[2];
  let space = "";
  if (fn === "color") {
    const id = /^([a-z][a-z0-9-]*)\s+/i.exec(body);
    if (!id) return false;
    space = id[1].toLowerCase();
    body = body.slice(id[0].length);
  }
  const n = (body.match(CHANNEL) ?? []).slice(0, 3).map((x) => (x.toLowerCase() === "none" ? 0 : Number(x)));
  if (n.length < 3 || n.some((x) => !Number.isFinite(x))) return false;
  const spread = () => Math.max(...n) - Math.min(...n);
  switch (fn) {
    case "oklch": return n[1] > 0.04;
    case "lch":   return n[1] > 4;
    case "oklab": return Math.hypot(n[1], n[2]) > 0.04;
    case "lab":   return Math.hypot(n[1], n[2]) > 4;
    case "rgb":
    case "rgba":  return spread() > 24;
    case "color": return RGB_SPACES.has(space) && spread() > 0.094;
    default:      return false;
  }
}
