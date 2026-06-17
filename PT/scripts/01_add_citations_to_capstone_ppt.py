from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "PT" / "스포츠과학과 캡스톤디자인 발표 19101207 김태현.pptx"
OUTPUT = ROOT / "PT" / "스포츠과학과 캡스톤디자인 발표 19101207 김태현_cited.pptx"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

SLIDE_W = 12_192_000
SLIDE_H = 6_858_000


MARKER_REPLACEMENTS: dict[int, list[tuple[str, str]]] = {
    2: [
        ("250,000", "250,000 [1]"),
        (" surgeries in the United States", " injuries in the United States"),
        ("25%", "25% [2]"),
        ("50%", "50% [2,3]"),
    ],
    3: [("Current", "Current [7]")],
    4: [("Standard", "Standard [7]"), ("Clinical ", "Clinical [7] ")],
    5: [
        ("IMU", "IMU [4]"),
        ("6,000–12,000", "6,000–12,000 [5]"),
        ("continuous", "continuous [2,4]"),
    ],
    6: [
        ("Single-speed", "Single-speed [4]"),
        ("scalar", "scalar [3,6]"),
        ("High-cost", "High-cost [2,7]"),
    ],
    10: [("Data", "Data [4]")],
    11: [("Biomechanical metrics", "Biomechanical metrics [3]")],
    12: [
        ("Peak / min / ROM / ", "Peak / min / ROM / [3] "),
        ("Limb Symmetry Index", "Limb Symmetry Index [3]"),
    ],
    13: [
        ("Waveform, SPM1D", "Waveform, SPM1D [6]"),
        ("Effect size", "Effect size [6]"),
    ],
    15: [
        (
            "SHAP was used to interpret the model’s predictions and improve explainability, which is crucial in medical applications.",
            "SHAP was used to interpret the model’s predictions and improve explainability, which is crucial in medical applications. [8]",
        )
    ],
    18: [("Why ML?", "Why ML? [8]")],
}

FOOTNOTES: dict[int, list[str]] = {
    2: [
        "[1] Gao & Zheng report ~250,000 ACL injuries and ~130,000 ACLR procedures/year in the US.",
        "[2] Tan et al. report >20% reinjury and 50-80% knee OA rates after ACL injury.",
        "[3] Hart et al. report post-traumatic knee OA in >50% within 10-20 years after ACLR.",
    ],
    3: [
        "[7] Conventional kinetic assessment requires complex, expensive systems and is limited to laboratory settings.",
    ],
    4: [
        "[7] Laboratory-based systems remain the conventional reference for kinetic assessment but limit routine clinical access.",
    ],
    5: [
        "[4] COMPWALK-ACL describes portable, cost-effective IMU gait assessment outside the lab across slow/normal/fast speeds.",
        "[5] Post-ACLR daily-step tertiles ranged from 3,326-12,680 steps/day in Lisee et al.",
    ],
    6: [
        "[4] Speed strongly modulates joint kinematics, motivating multi-speed assessment.",
        "[6] Waveform analysis evaluates the entire stance phase instead of only discrete peak outcomes.",
    ],
    10: [
        "[4] COMPWALK-ACL collected overground gait at self-selected slow, normal, and fast speeds.",
    ],
    11: [
        "[3] ACLR gait reviews focus on knee kinematics and joint moments during level walking.",
    ],
    12: [
        "[3] Peak, ROM, and side-to-side variables are common discrete gait metrics in ACLR literature.",
    ],
    13: [
        "[6] Bilateral waveform analysis used functional models across stance and reported significant waveform regions/effects.",
    ],
    15: [
        "[8] Kokkotis et al. used SHAP-based explainability to rank gait biomechanical parameters in ACL injury classification.",
    ],
    18: [
        "[8] Explainable ML can identify biomechanical parameters that contribute to ACL injury classification.",
    ],
}

REFERENCES = [
    "[1] Gao, B., & Zheng, N. (2014). Progressive Changes in Walking Kinematics and Kinetics After Anterior Cruciate Ligament Injury and Reconstruction: A Review and Meta-Analysis.",
    "[2] Tan, T., Gatti, A. A., Fan, B., et al. (2022). Towards Out-of-Lab Anterior Cruciate Ligament Injury Prevention and Rehabilitation Assessment: A Review of Portable Sensing Approaches.",
    "[3] Hart, H. F., Culvenor, A. G., Collins, N. J., et al. (2016). Knee kinematics and joint moments during gait following anterior cruciate ligament reconstruction: A systematic review and meta-analysis.",
    "[4] Yona, T., Peskin, B., & Fischer, A. (2026). The COMPWALK-ACL: A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients.",
    "[5] Lisee, C., Davis-Wilson, H., Evans-Pickett, A., et al. (2022). Linking Gait Biomechanics and Daily Steps Post ACL-Reconstruction.",
    "[6] Buttner, C., Lisee, C., Bjornsen, E., et al. (2025). Bilateral waveform analysis of gait biomechanics presurgery to 12 months following ACL reconstruction compared to controls.",
    "[7] Krishnakumar, S., van Beijnum, B.-J. F., Baten, C. T. M., et al. (2024). Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation: A Systematic Review.",
    "[8] Kokkotis, C., Moustakidis, S., Tsatalas, T., et al. (2022). Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury.",
]


def parse_xml(data: bytes) -> etree._Element:
    return etree.fromstring(data)


def serialize_xml(root: etree._Element) -> bytes:
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def max_shape_id(slide: etree._Element) -> int:
    ids = []
    for node in slide.xpath(".//p:cNvPr", namespaces=NS):
        val = node.get("id")
        if val and val.isdigit():
            ids.append(int(val))
    return max(ids or [1])


def text_shape(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, paragraphs: list[str],
               size: int = 900, color: str = "505050", bold_first: bool = False) -> etree._Element:
    sp = etree.Element(qn("p", "sp"))
    nv_sp_pr = etree.SubElement(sp, qn("p", "nvSpPr"))
    etree.SubElement(nv_sp_pr, qn("p", "cNvPr"), id=str(shape_id), name=name)
    etree.SubElement(nv_sp_pr, qn("p", "cNvSpPr"))
    etree.SubElement(nv_sp_pr, qn("p", "nvPr"))

    sp_pr = etree.SubElement(sp, qn("p", "spPr"))
    xfrm = etree.SubElement(sp_pr, qn("a", "xfrm"))
    etree.SubElement(xfrm, qn("a", "off"), x=str(x), y=str(y))
    etree.SubElement(xfrm, qn("a", "ext"), cx=str(cx), cy=str(cy))
    geom = etree.SubElement(sp_pr, qn("a", "prstGeom"), prst="rect")
    etree.SubElement(geom, qn("a", "avLst"))
    etree.SubElement(sp_pr, qn("a", "noFill"))
    ln = etree.SubElement(sp_pr, qn("a", "ln"))
    etree.SubElement(ln, qn("a", "noFill"))

    tx_body = etree.SubElement(sp, qn("p", "txBody"))
    body_pr = etree.SubElement(tx_body, qn("a", "bodyPr"), wrap="square", lIns="0", tIns="0", rIns="0", bIns="0")
    etree.SubElement(body_pr, qn("a", "spAutoFit"))
    etree.SubElement(tx_body, qn("a", "lstStyle"))

    for idx, para in enumerate(paragraphs):
        p = etree.SubElement(tx_body, qn("a", "p"))
        r = etree.SubElement(p, qn("a", "r"))
        attrs = {"lang": "en-US", "sz": str(size)}
        if bold_first and idx == 0:
            attrs["b"] = "1"
        r_pr = etree.SubElement(r, qn("a", "rPr"), **attrs)
        fill = etree.SubElement(r_pr, qn("a", "solidFill"))
        etree.SubElement(fill, qn("a", "srgbClr"), val=color)
        etree.SubElement(r_pr, qn("a", "latin"), typeface="Aptos")
        etree.SubElement(r_pr, qn("a", "ea"), typeface="Aptos")
        t = etree.SubElement(r, qn("a", "t"))
        t.text = para
        etree.SubElement(p, qn("a", "endParaRPr"), lang="en-US", sz=str(size))
    return sp


def add_footnote(slide: etree._Element, notes: list[str]) -> None:
    sp_tree = slide.find(".//p:spTree", namespaces=NS)
    if sp_tree is None:
        raise RuntimeError("slide has no spTree")
    shape = text_shape(
        max_shape_id(slide) + 1,
        "Citation footnotes",
        410_000,
        6_155_000,
        11_350_000,
        520_000,
        notes,
        size=760,
        color="5B5B5B",
    )
    sp_tree.append(shape)


def apply_markers(slide: etree._Element, replacements: list[tuple[str, str]]) -> None:
    used: set[str] = set()
    for old, new in replacements:
        for t in slide.xpath(".//a:t", namespaces=NS):
            if t.text == old and old not in used:
                t.text = new
                used.add(old)
                break
        if old not in used:
            for t in slide.xpath(".//a:t", namespaces=NS):
                if t.text and old in t.text and old not in used:
                    t.text = t.text.replace(old, new, 1)
                    used.add(old)
                    break


def blank_slide(paragraphs: list[str]) -> etree._Element:
    sld = etree.Element(qn("p", "sld"), nsmap={"a": NS["a"], "r": NS["r"], "p": NS["p"]})
    c_sld = etree.SubElement(sld, qn("p", "cSld"))
    bg = etree.SubElement(c_sld, qn("p", "bg"))
    bg_pr = etree.SubElement(bg, qn("p", "bgPr"))
    solid = etree.SubElement(bg_pr, qn("a", "solidFill"))
    etree.SubElement(solid, qn("a", "srgbClr"), val="FFFFFF")
    etree.SubElement(bg_pr, qn("a", "effectLst"))
    sp_tree = etree.SubElement(c_sld, qn("p", "spTree"))
    nv_grp = etree.SubElement(sp_tree, qn("p", "nvGrpSpPr"))
    etree.SubElement(nv_grp, qn("p", "cNvPr"), id="1", name="")
    etree.SubElement(nv_grp, qn("p", "cNvGrpSpPr"))
    etree.SubElement(nv_grp, qn("p", "nvPr"))
    grp_pr = etree.SubElement(sp_tree, qn("p", "grpSpPr"))
    xfrm = etree.SubElement(grp_pr, qn("a", "xfrm"))
    etree.SubElement(xfrm, qn("a", "off"), x="0", y="0")
    etree.SubElement(xfrm, qn("a", "ext"), cx="0", cy="0")
    etree.SubElement(xfrm, qn("a", "chOff"), x="0", y="0")
    etree.SubElement(xfrm, qn("a", "chExt"), cx="0", cy="0")

    title = text_shape(2, "References title", 600_000, 390_000, 10_900_000, 650_000, ["References"], size=3600, color="111111", bold_first=True)
    body = text_shape(3, "References list", 760_000, 1_250_000, 10_900_000, 5_120_000, paragraphs, size=1320, color="222222")
    sp_tree.extend([title, body])
    etree.SubElement(sld, qn("p", "clrMapOvr")).append(etree.Element(qn("a", "masterClrMapping")))
    return sld


def add_content_type(content_types: etree._Element, slide_no: int) -> None:
    part_name = f"/ppt/slides/slide{slide_no}.xml"
    existing = content_types.xpath(f".//ct:Override[@PartName='{part_name}']", namespaces=NS)
    if existing:
        return
    etree.SubElement(
        content_types,
        qn("ct", "Override"),
        PartName=part_name,
        ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
    )


def add_presentation_slide(presentation: etree._Element, rels: etree._Element, slide_no: int) -> None:
    sld_id_lst = presentation.find(".//p:sldIdLst", namespaces=NS)
    if sld_id_lst is None:
        raise RuntimeError("presentation has no sldIdLst")

    used_rids = {rel.get("Id") for rel in rels.xpath(".//rel:Relationship", namespaces=NS)}
    max_rid = max(int(r[3:]) for r in used_rids if r and re.fullmatch(r"rId\d+", r))
    rid = f"rId{max_rid + 1}"

    max_sid = max(int(n.get("id")) for n in sld_id_lst.xpath(".//p:sldId", namespaces=NS))
    etree.SubElement(sld_id_lst, qn("p", "sldId"), id=str(max_sid + 1), **{qn("r", "id"): rid})
    etree.SubElement(
        rels,
        qn("rel", "Relationship"),
        Id=rid,
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
        Target=f"slides/slide{slide_no}.xml",
    )


def update_app_slide_count(app_xml: etree._Element, count: int) -> None:
    for tag in ("Slides", "Notes"):
        node = app_xml.find(f".//{{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}}{tag}")
        if node is not None and tag == "Slides":
            node.text = str(count)


def slide_rels(layout_target: str = "../slideLayouts/slideLayout2.xml") -> bytes:
    rels = etree.Element(qn("rel", "Relationships"), nsmap={None: NS["rel"]})
    etree.SubElement(
        rels,
        qn("rel", "Relationship"),
        Id="rId1",
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
        Target=layout_target,
    )
    return serialize_xml(rels)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    with zipfile.ZipFile(SOURCE, "r") as zin:
        entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    for slide_no, replacements in MARKER_REPLACEMENTS.items():
        name = f"ppt/slides/slide{slide_no}.xml"
        slide = parse_xml(entries[name])
        apply_markers(slide, replacements)
        if slide_no in FOOTNOTES:
            add_footnote(slide, FOOTNOTES[slide_no])
        entries[name] = serialize_xml(slide)

    ref_chunks = [REFERENCES[:4], REFERENCES[4:]]
    first_ref_slide = 19
    for offset, refs in enumerate(ref_chunks):
        slide_no = first_ref_slide + offset
        entries[f"ppt/slides/slide{slide_no}.xml"] = serialize_xml(blank_slide(refs))
        entries[f"ppt/slides/_rels/slide{slide_no}.xml.rels"] = slide_rels()

    content_types = parse_xml(entries["[Content_Types].xml"])
    presentation = parse_xml(entries["ppt/presentation.xml"])
    presentation_rels = parse_xml(entries["ppt/_rels/presentation.xml.rels"])

    for slide_no in (19, 20):
        add_content_type(content_types, slide_no)
        add_presentation_slide(presentation, presentation_rels, slide_no)

    entries["[Content_Types].xml"] = serialize_xml(content_types)
    entries["ppt/presentation.xml"] = serialize_xml(presentation)
    entries["ppt/_rels/presentation.xml.rels"] = serialize_xml(presentation_rels)

    if "docProps/app.xml" in entries:
        app = parse_xml(entries["docProps/app.xml"])
        update_app_slide_count(app, 20)
        entries["docProps/app.xml"] = serialize_xml(app)

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        with zipfile.ZipFile(SOURCE, "r") as zin:
            for info in zin.infolist():
                if info.filename in entries:
                    zout.writestr(info, entries.pop(info.filename))
                else:
                    zout.writestr(info, zin.read(info.filename))
        for name, data in sorted(entries.items()):
            zout.writestr(name, data)

    print(f"created: {OUTPUT}")


if __name__ == "__main__":
    main()
