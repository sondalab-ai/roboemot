"""XMP sidecar parser for the catalogue import (SPEC 6, Catalogue import (XMP); SPEC 12).

Reads what common photo tools write: MWG Regions (``mwg-rs:RegionList`` with ``Name`` and
``Type``), and hierarchical keywords (``lr:hierarchicalSubject``, for example
``Animals|Cat|Mia``; ``dc:subject`` entries that contain ``|`` count as hierarchical too).
Bare keywords never create a record: only a region name or a path under a known root does.

The input is untrusted: it is size-capped before parsing and parsed with ``defusedxml``
(DTDs and entities are refused, so neither external entities nor expansion bombs run).
Names go through a caller-supplied validator; whatever fails is reported, never raised.

Both RDF serialisations are read (properties as attributes of ``rdf:Description`` or as child
elements). Region areas follow MWG 2.0: ``stArea:x``/``y`` are the *centre* of the region,
``w``/``h`` its size, all normalised; any other unit is skipped.

Mapping: Face regions and ``People`` paths give ``person`` items (name only, never a region
kept: nothing is ever derived from a person's photo); Pet regions and ``Animals`` paths give
``animal``; ``Objects`` paths give ``object``. Focus and BarCode regions are skipped.
Categories are normalised to the detector's lower-case labels through an explicit alias table;
an unmapped category is reported as a conflict and the item is not imported (no guessing).
Items are deduplicated by name (case-insensitive), kind and category; the same name and kind
with different categories is a conflict and neither is imported.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from xml.etree.ElementTree import Element, ParseError

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

ItemKind = Literal["person", "animal", "object"]

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
MWG = "http://www.metadataworkinggroup.com/schemas/regions/"
ST_AREA = "http://ns.adobe.com/xap/1.0/sType/Area#"
ST_DIM = "http://ns.adobe.com/xap/1.0/sType/Dimensions#"
DC = "http://purl.org/dc/elements/1.1/"
LR = "http://ns.adobe.com/lightroom/1.0/"

PATH_SEP = "|"
PERSON_ROOTS = frozenset({"people", "person", "persone", "persona"})
ANIMAL_ROOTS = frozenset({"animals", "animal", "pets", "pet", "animali"})
OBJECT_ROOTS = frozenset({"objects", "object", "things", "oggetti"})
REGION_KINDS: dict[str, ItemKind] = {"face": "person", "pet": "animal"}
NORMALIZED = "normalized"
MAX_NAME_CHARS = 64

# Detector labels (COCO class names, lower case) that can be imported as a category.
_LABELS = (
    "bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie "
    "suitcase frisbee skis snowboard ball kite bat glove skateboard surfboard racket bottle "
    "glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot pizza donut "
    "cake chair couch plant bed table toilet tv laptop mouse remote keyboard phone microwave "
    "oven toaster sink refrigerator book clock vase scissors teddy hairdryer toothbrush"
).split()
CATEGORY_ALIASES: dict[str, str] = {
    **{label: label for label in _LABELS},
    "cats": "cat",
    "dogs": "dog",
    "birds": "bird",
    "horses": "horse",
    "mug": "cup",
    "sofa": "couch",
    "potted plant": "plant",
    "tv monitor": "tv",
    "teddy bear": "teddy",
    "cell phone": "phone",
    "gatto": "cat",
    "gatta": "cat",
    "cane": "dog",
    "uccello": "bird",
    "cavallo": "horse",
    "tazza": "cup",
    "bicchiere": "glass",
    "bottiglia": "bottle",
    "libro": "book",
    "pianta": "plant",
    "divano": "couch",
    "sedia": "chair",
    "orologio": "clock",
    "vaso": "vase",
}


class XmpError(ValueError):
    """The sidecar cannot be read; ``code`` is a stable machine-readable reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Region:
    """A tagged area in normalised image coordinates; ``cx``/``cy`` are the centre."""

    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True)
class CatalogueItem:
    """One individual the sidecar names. ``regions`` stay empty for people."""

    name: str
    kind: ItemKind
    category: str = ""
    regions: tuple[Region, ...] = ()

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.name.casefold(), self.kind, self.category)


@dataclass(frozen=True)
class Conflict:
    """Something the owner must fix in the tags; the item is not imported."""

    name: str
    code: str


@dataclass
class ParsedXmp:
    """The result of ``parse_xmp``: items, conflicts and counts of what was left out.

    ``applied_dims`` is the pixel size the regions were drawn on (``mwg-rs:AppliedToDimensions``)
    when the sidecar says so. ``skipped`` maps a reason code to a count.
    """

    items: list[CatalogueItem] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    applied_dims: tuple[int, int] | None = None
    truncated: bool = False

    def skip(self, code: str) -> None:
        self.skipped[code] = self.skipped.get(code, 0) + 1


def default_validator(text: str) -> str:
    """Strip, NFC-normalise, refuse control characters and over-long names."""
    name = unicodedata.normalize("NFC", text).strip()
    if not name or len(name) > MAX_NAME_CHARS or not name.isprintable():
        raise ValueError("invalid name")
    return name


def _q(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _text(node: Element) -> str | None:
    value = (node.text or "").strip()
    return value or None


def _prop(nodes: list[Element], ns: str, local: str) -> str | None:
    """A simple property as an attribute of any node, or as the text of a child element."""
    tag = _q(ns, local)
    for node in nodes:
        if tag in node.attrib:
            return node.attrib[tag].strip() or None
    for node in nodes:
        for child in node:
            if child.tag == tag:
                return _text(child)
    return None


def _holders(li: Element) -> list[Element]:
    """The ``li`` itself and the ``rdf:Description`` it wraps (both serialisations)."""
    return [li, *(c for c in li if c.tag == _q(RDF, "Description"))]


def _area(holders: list[Element]) -> Region | None:
    tag = _q(MWG, "Area")
    for holder in holders:
        for child in holder:
            if child.tag != tag:
                continue
            nodes = [child, *(c for c in child if c.tag == _q(RDF, "Description"))]
            vals = [_prop(nodes, ST_AREA, key) for key in ("x", "y", "w", "h")]
            unit = _prop(nodes, ST_AREA, "unit")
            if unit is not None and unit != NORMALIZED:
                return None
            try:
                cx, cy, w, h = (float(v) for v in vals)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            finite = all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))
            return Region(cx, cy, w, h) if finite and w > 0 and h > 0 else None
    return None


def _applied_dims(root: Element) -> tuple[int, int] | None:
    for node in root.iter(_q(MWG, "AppliedToDimensions")):
        nodes = [node, *(c for c in node if c.tag == _q(RDF, "Description"))]
        try:
            w, h = (int(float(_prop(nodes, ST_DIM, k) or "")) for k in ("w", "h"))
        except ValueError:
            return None
        unit = _prop(nodes, ST_DIM, "unit")
        return (w, h) if w > 0 and h > 0 and unit in (None, "pixel") else None
    return None


def _bag_items(root: Element, ns: str, local: str) -> list[str]:
    out: list[str] = []
    for node in root.iter(_q(ns, local)):
        for li in node.iter(_q(RDF, "li")):
            value = _text(li)
            if value:
                out.append(value)
    return out


def _category(parts: list[str]) -> str | None:
    """The detector label for the path between root and name; "" when there is none."""
    middle = parts[1:-1]
    if not middle:
        return ""
    for part in middle:
        label = CATEGORY_ALIASES.get(part.strip().casefold())
        if label:
            return label
    return None


def parse_xmp(
    data: bytes | str,
    *,
    max_bytes: int,
    max_items: int,
    max_regions: int,
    validate_name: Callable[[str], str] = default_validator,
) -> ParsedXmp:
    """Parse one sidecar into catalogue items; raises ``XmpError`` when it cannot be read."""
    raw = data.encode() if isinstance(data, str) else data
    if len(raw) > max_bytes:
        raise XmpError("too_large")
    try:
        root = SafeET.fromstring(raw, forbid_dtd=True)
    except DefusedXmlException as exc:
        raise XmpError("forbidden_xml") from exc
    except (ParseError, ValueError, UnicodeError, RecursionError) as exc:
        raise XmpError("not_xml") from exc
    out = ParsedXmp(applied_dims=_applied_dims(root))
    raws: list[CatalogueItem] = []
    _read_regions(root, out, raws, max_regions, validate_name)
    _read_paths(root, out, raws, validate_name)
    _merge(raws, out, max_items)
    return out


def _read_regions(
    root: Element,
    out: ParsedXmp,
    raws: list[CatalogueItem],
    max_regions: int,
    validate_name: Callable[[str], str],
) -> None:
    seen = 0
    for listing in root.iter(_q(MWG, "RegionList")):
        for li in listing.iter(_q(RDF, "li")):
            holders = _holders(li)
            if _prop(holders, MWG, "Name") is None and _prop(holders, MWG, "Type") is None:
                continue
            seen += 1
            if seen > max_regions:
                out.truncated = True
                return
            kind = REGION_KINDS.get((_prop(holders, MWG, "Type") or "").casefold())
            raw_name = _prop(holders, MWG, "Name")
            if kind is None:
                out.skip("region_type")
            elif raw_name is None:
                out.skip("region_unnamed")
            else:
                _add_region(out, raws, kind, raw_name, _area(holders), validate_name)


def _add_region(
    out: ParsedXmp,
    raws: list[CatalogueItem],
    kind: ItemKind,
    raw_name: str,
    area: Region | None,
    validate_name: Callable[[str], str],
) -> None:
    try:
        name = validate_name(raw_name)
    except ValueError:
        out.skip("name_invalid")
        return
    if kind == "person":
        raws.append(CatalogueItem(name, "person"))
        return
    if area is None:
        out.skip("region_area")
    raws.append(CatalogueItem(name, kind, "", () if area is None else (area,)))


def _read_paths(
    root: Element,
    out: ParsedXmp,
    raws: list[CatalogueItem],
    validate_name: Callable[[str], str],
) -> None:
    paths = _bag_items(root, LR, "hierarchicalSubject")
    paths += [p for p in _bag_items(root, DC, "subject") if PATH_SEP in p]
    for path in dict.fromkeys(paths):
        parts = [p.strip() for p in path.split(PATH_SEP)]
        root_name = parts[0].casefold()
        kind: ItemKind | None = (
            "person"
            if root_name in PERSON_ROOTS
            else "animal"
            if root_name in ANIMAL_ROOTS
            else "object"
            if root_name in OBJECT_ROOTS
            else None
        )
        if kind is None or len(parts) < 2 or not all(parts):
            out.skip("path_ignored")
            continue
        try:
            name = validate_name(parts[-1])
        except ValueError:
            out.skip("name_invalid")
            continue
        category = _category(parts) if kind != "person" else ""
        if category is None:
            out.conflicts.append(Conflict(name, "unknown_category"))
            continue
        raws.append(CatalogueItem(name, kind, category))


def _merge(raws: list[CatalogueItem], out: ParsedXmp, max_items: int) -> None:
    """Group by (name, kind); fill empty categories; drop and report category conflicts."""
    groups: dict[tuple[str, ItemKind], list[CatalogueItem]] = {}
    for item in raws:
        groups.setdefault((item.name.casefold(), item.kind), []).append(item)
    for members in groups.values():
        categories = sorted({m.category for m in members if m.category})
        if len(categories) > 1:
            out.conflicts.append(Conflict(members[0].name, "category_conflict"))
            continue
        regions = tuple(r for m in members for r in m.regions)
        item = CatalogueItem(
            members[0].name, members[0].kind, categories[0] if categories else "", regions
        )
        if len(out.items) >= max_items:
            out.truncated = True
            break
        out.items.append(item)
