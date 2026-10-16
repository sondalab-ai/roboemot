"""XMP catalogue parser (SPEC 6, Catalogue import; SPEC 12): formats, mapping, hostile input."""

from __future__ import annotations

import pytest
from roboemot.registry.xmp import XmpError, parse_xmp

LIMITS = {"max_bytes": 65536, "max_items": 50, "max_regions": 20}

HEAD = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/" '
    'xmlns:stArea="http://ns.adobe.com/xap/1.0/sType/Area#" '
    'xmlns:stDim="http://ns.adobe.com/xap/1.0/sType/Dimensions#" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:lr="http://ns.adobe.com/lightroom/1.0/">'
)
TAIL = "</rdf:RDF></x:xmpmeta>"

ATTR_FORM = (
    HEAD
    + """<rdf:Description><mwg-rs:Regions rdf:parseType="Resource">
<mwg-rs:AppliedToDimensions stDim:w="4000" stDim:h="3000" stDim:unit="pixel"/>
<mwg-rs:RegionList><rdf:Bag>
<rdf:li><rdf:Description mwg-rs:Name="Mia" mwg-rs:Type="Pet">
  <mwg-rs:Area stArea:x="0.5" stArea:y="0.4" stArea:w="0.2" stArea:h="0.3" stArea:unit="normalized"/>
</rdf:Description></rdf:li>
<rdf:li><rdf:Description mwg-rs:Name="Anna" mwg-rs:Type="Face">
  <mwg-rs:Area stArea:x="0.1" stArea:y="0.1" stArea:w="0.1" stArea:h="0.1" stArea:unit="normalized"/>
</rdf:Description></rdf:li>
<rdf:li><rdf:Description mwg-rs:Name="Code" mwg-rs:Type="BarCode"/></rdf:li>
</rdf:Bag></mwg-rs:RegionList></mwg-rs:Regions></rdf:Description>"""
    + TAIL
)

ELEMENT_FORM = (
    HEAD
    + """<rdf:Description><mwg-rs:Regions><rdf:Description><mwg-rs:RegionList><rdf:Bag>
<rdf:li rdf:parseType="Resource"><mwg-rs:Name>Rex</mwg-rs:Name><mwg-rs:Type>Pet</mwg-rs:Type>
<mwg-rs:Area><rdf:Description><stArea:x>0.25</stArea:x><stArea:y>0.5</stArea:y>
<stArea:w>0.1</stArea:w><stArea:h>0.2</stArea:h><stArea:unit>normalized</stArea:unit>
</rdf:Description></mwg-rs:Area></rdf:li>
</rdf:Bag></mwg-rs:RegionList></rdf:Description></mwg-rs:Regions></rdf:Description>"""
    + TAIL
)


def subjects(*paths, flat=()):
    lis = "".join(f"<rdf:li>{p}</rdf:li>" for p in paths)
    dc = "".join(f"<rdf:li>{p}</rdf:li>" for p in flat)
    return (
        HEAD
        + f"<rdf:Description><lr:hierarchicalSubject><rdf:Bag>{lis}</rdf:Bag>"
        + f"</lr:hierarchicalSubject><dc:subject><rdf:Bag>{dc}</rdf:Bag></dc:subject>"
        + "</rdf:Description>"
        + TAIL
    )


def parse(text, **kw):
    return parse_xmp(text, **{**LIMITS, **kw})


def test_attribute_form_regions_and_types():
    out = parse(ATTR_FORM)
    by = {i.name: i for i in out.items}
    assert by["Mia"].kind == "animal" and by["Anna"].kind == "person"
    assert out.applied_dims == (4000, 3000)
    assert out.skipped == {"region_type": 1}
    region = by["Mia"].regions[0]
    # x/y are the centre of the area (MWG), not its top-left corner.
    assert (region.cx, region.cy, region.w, region.h) == (0.5, 0.4, 0.2, 0.3)
    assert by["Anna"].regions == ()


def test_element_form_regions():
    (item,) = parse(ELEMENT_FORM).items
    assert (item.name, item.kind) == ("Rex", "animal")
    assert (item.regions[0].cx, item.regions[0].h) == (0.25, 0.2)


def test_hierarchical_subjects_map_kind_and_category():
    out = parse(subjects("Animals|Cat|Mia", "Objects|Mug|Blue mug", "People|Anna", "Places|Rome"))
    got = {(i.name, i.kind, i.category) for i in out.items}
    assert got == {("Mia", "animal", "cat"), ("Blue mug", "object", "cup"), ("Anna", "person", "")}
    assert out.skipped == {"path_ignored": 1}


def test_flat_keywords_create_nothing_but_pipe_ones_count():
    assert parse(subjects(flat=["holiday", "cat"])).items == []
    out = parse(subjects(flat=["Animals|Dog|Rex"]))
    assert [(i.name, i.category) for i in out.items] == [("Rex", "dog")]


def test_region_and_path_are_deduplicated_and_category_filled():
    xml = (
        HEAD
        + "<rdf:Description><mwg-rs:RegionList><rdf:Bag><rdf:li><rdf:Description "
        + 'mwg-rs:Name="mia" mwg-rs:Type="Pet"/></rdf:li></rdf:Bag></mwg-rs:RegionList>'
        + "<lr:hierarchicalSubject><rdf:Bag><rdf:li>Animals|Cat|Mia</rdf:li></rdf:Bag>"
        + "</lr:hierarchicalSubject></rdf:Description>"
        + TAIL
    )
    out = parse(xml)
    assert len(out.items) == 1 and out.items[0].category == "cat"


def test_conflicting_categories_and_unknown_category_are_reported_not_imported():
    out = parse(subjects("Animals|Cat|Mia", "Animals|Dog|Mia", "Animali|Criceto|Bob"))
    assert out.items == []
    assert {(c.name, c.code) for c in out.conflicts} == {
        ("Mia", "category_conflict"),
        ("Bob", "unknown_category"),
    }


def test_italian_roots_and_aliases():
    out = parse(subjects("Animali|Gatto|Micio", "Persone|Anna"))
    assert {(i.name, i.kind, i.category) for i in out.items} == {
        ("Micio", "animal", "cat"),
        ("Anna", "person", ""),
    }


def test_bad_names_are_skipped_and_counted():
    out = parse(subjects("Animals|Cat|", "Animals|Cat|Mia​x", "Animals|Cat|" + "x" * 80))
    assert out.items == [] and out.skipped.get("name_invalid", 0) + out.skipped.get(
        "path_ignored", 0
    ) == 3


def test_custom_name_validator():
    def only_short(text):
        if len(text) > 3:
            raise ValueError
        return text

    out = parse(subjects("Animals|Cat|Mia", "Animals|Cat|Alexander"), validate_name=only_short)
    assert [i.name for i in out.items] == ["Mia"]


def test_non_normalized_and_out_of_range_areas_are_dropped():
    pixel = ATTR_FORM.replace('stArea:unit="normalized"', 'stArea:unit="pixel"')
    assert parse(pixel).items[0].regions == ()
    bad = ATTR_FORM.replace('stArea:w="0.2"', 'stArea:w="7"')
    assert parse(bad).items[0].regions == ()


def test_limits_truncate():
    many = subjects(*[f"Animals|Cat|C{i}" for i in range(10)])
    out = parse(many, max_items=3)
    assert len(out.items) == 3 and out.truncated
    assert parse(ATTR_FORM, max_regions=1).truncated


def test_huge_file_rejected_before_parsing():
    with pytest.raises(XmpError) as exc:
        parse(b"<a>" + b"x" * 70000 + b"</a>")
    assert exc.value.code == "too_large"


@pytest.mark.parametrize("payload", [b"not xml", b"", b"<a><b></a>", b"\xff\xfe\x00"])
def test_malformed(payload):
    with pytest.raises(XmpError) as exc:
        parse(payload)
    assert exc.value.code == "not_xml"


def test_xxe_is_refused(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    xml = (
        f'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY e SYSTEM "file://{secret}">]>'
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>&e;</x:xmpmeta>"
    )
    with pytest.raises(XmpError) as exc:
        parse(xml)
    assert exc.value.code == "forbidden_xml"


def test_billion_laughs_is_refused():
    xml = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY a "aaaaaaaaaa">'
        '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
        '<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">]><lolz>&c;</lolz>'
    )
    with pytest.raises(XmpError) as exc:
        parse(xml)
    assert exc.value.code == "forbidden_xml"


def test_internal_dtd_without_entities_is_refused_too():
    with pytest.raises(XmpError) as exc:
        parse("<!DOCTYPE r [<!ELEMENT r ANY>]><r/>")
    assert exc.value.code == "forbidden_xml"
