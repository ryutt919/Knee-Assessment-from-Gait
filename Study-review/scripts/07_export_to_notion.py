"""Transform Study-review markdown outputs into Notion-flavored Markdown.

Produces a JSON manifest (one entry per paper) for `notion-create-pages`,
plus a Notion `<table>` block for `mds/06_paper_index_table.md`. Content is
NOT uploaded here -- this script only stages the transformed text so it can
be inspected before any Notion API calls are made.

Usage:
    python3 Study-review/scripts/07_export_to_notion.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Study-review/
PAPERS_REVISED = ROOT / "mds" / "papers_revised"
NEW_CATEGORY_DIR = ROOT / "mds" / "papers" / "07_composite_kinematic_kinetic_scoring_indices"
INDEX_TABLE_MD = ROOT / "mds" / "06_paper_index_table.md"
OUT_DIR = ROOT / "logs" / "notion_export"

CATEGORY_LABELS = {
    "01_acl_gait_biomechanics_studies": "ACL 보행 생체역학 연구",
    "02_acl_gait_reviews_meta_analyses": "ACL 보행 리뷰·메타분석",
    "03_wearable_imu_and_portable_sensing": "웨어러블 IMU·휴대용 센싱",
    "04_machine_learning_and_deep_learning": "머신러닝·딥러닝",
    "05_return_to_sport_and_functional_tests": "스포츠 복귀·기능 테스트",
    "06_general_gait_and_other_knee_conditions": "일반 보행·기타 무릎 질환",
    "07_composite_kinematic_kinetic_scoring_indices": "복합 운동학/운동역학 점수화 지표",
}

RESERVED_CHARS = "\\*~`$[]<>{}|^"

# ---------------------------------------------------------------------------
# Reserved-character escaping with protection for intentional Notion markup
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")


def _protect_and_escape(text: str) -> str:
    """Escape Notion-reserved characters while preserving intentional markup.

    Preserves: **bold** spans, *italic* spans (already converted from the
    `_(...)_ ` locator pattern before this runs), and real markdown links
    `[text](url)`. Everything else that is a reserved character is escaped
    with a backslash so it renders literally.
    """
    protected: list[str] = []

    def stash(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    # Order matters: protect the longest/most specific patterns first.
    text = re.sub(r"\[[^\[\]\n]+\]\([^()\n]+\)", stash, text)  # [text](url)
    text = re.sub(r"\*\*[^*\n]+\*\*", stash, text)  # **bold**
    text = re.sub(r"\*[^*\n]+\*", stash, text)  # *italic*
    text = re.sub(r"<br>", stash, text)  # intentional inline line break

    escaped_chars = set(RESERVED_CHARS)
    out = []
    for ch in text:
        out.append("\\" + ch if ch in escaped_chars else ch)
    text = "".join(out)

    def restore(match: re.Match) -> str:
        return protected[int(match.group(1))]

    return _PLACEHOLDER_RE.sub(restore, text)


_LOCATOR_ITALIC_RE = re.compile(r"_(\(.*?\))_")


def convert_locator_italic(text: str) -> str:
    """`_(근거: PAGE 1, Abstract)_` -> `*(근거: PAGE 1, Abstract)*`.

    Uses a non-greedy dot (not `[^)]*`) because some locator/fact-check
    parentheticals contain their own nested `(...)`, e.g.
    `_(사실검증 ... 본문 인용 (Zeni and Higginson, 2009)에 대응...)_`.
    """
    return _LOCATOR_ITALIC_RE.sub(r"*\1*", text)


# ---------------------------------------------------------------------------
# Line classification for wrapped-quote merging and blockquote handling
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^(\s*)-\s+(.*)$")
_HEADING_RE = re.compile(r"^#{1,6}\s")

# A `- > ...` bullet-quote opener always starts a run (see
# `_flatten_blockquote_runs` for the bare `>` start condition). Once a run
# has started, every subsequent `>`-prefixed line belongs to it regardless
# of content -- this is what lets wrapped citation text like
# `> clinicbiomech.2009...` stay attached to its run.
_QUOTE_RUN_START_RE = re.compile(r"^(\s*)-\s+>\s?(.*)$")
_QUOTE_RUN_CONT_RE = re.compile(r"^>\s?(.*)$")


def _join_quote_segments(segments: list[str]) -> str:
    """Undo PDF word-wrap within each labeled part, join parts with <br>.

    `segments` is one raw line's content per element, in source order. A
    blank segment marks a paragraph break (e.g. between AS-IS/TO-BE/사실검증
    labels) and starts a new group; non-blank segments within a group are
    word-wrap continuations and get joined with a plain space.
    """
    groups: list[list[str]] = [[]]
    for seg in segments:
        seg = seg.strip()
        if seg == "":
            if groups[-1]:
                groups.append([])
        else:
            groups[-1].append(seg)
    groups = [g for g in groups if g]
    return "<br>".join(" ".join(g) for g in groups)


def _flatten_blockquote_runs(lines: list[str]) -> list[str]:
    """Rewrite `>`-prefixed runs into Notion-safe single-line blocks.

    Notion quote blocks can't contain raw newlines mid-quote (each `>` line
    would otherwise become its own separate quote block), so every
    contiguous run of `>` lines is collapsed into one line first, with
    `<br>` separating logical parts (blank `>` lines mark a part boundary)
    and plain spaces re-joining PDF word-wrap continuations within a part.

    A `>` line only *starts* a run if the previous output line is a
    boundary (blank / bullet / heading / start of file) or a `- > ...`
    bullet-quote opener. This is what distinguishes a genuine blockquote
    (e.g. `- > **[AS-IS]** ...` fact-check notes, or a standalone citation
    quote) from a wrapped raw quote whose continuation line just happens to
    start with a literal `>` (e.g. `R2 >0.95, MAPE<5%)`), which is left
    alone here and merged as an ordinary continuation later.

    Runs that were attached to a bullet (`- > ...`) become a plain bullet,
    since they're really list content (e.g. fact-check annotations). Bare
    top-level `>` runs become a real Notion multi-line quote block.
    """
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _QUOTE_RUN_START_RE.match(line)
        prev = out[-1] if out else ""
        prev_is_boundary = (
            not out
            or prev.strip() == ""
            or bool(_BULLET_RE.match(prev))
            or bool(_HEADING_RE.match(prev))
        )
        bare_m = _QUOTE_RUN_CONT_RE.match(line) if (not m and prev_is_boundary) else None
        if not (m or bare_m):
            out.append(line)
            i += 1
            continue
        if m:
            indent, first, as_bullet = m.group(1), m.group(2), True
        else:
            indent, first, as_bullet = "", bare_m.group(1), False
        segments = [first]
        i += 1
        while i < n:
            nxt = lines[i]
            m2 = _QUOTE_RUN_CONT_RE.match(nxt)
            if m2:
                segments.append(m2.group(1))
                i += 1
                continue
            # Bare PDF word-wrap continuation with no '>' marker at all --
            # only the *first* line of a wrapped sentence inside the source
            # got a '> ' prefix, so these must still be absorbed into the
            # same run (see the AS-IS citation-wrap case in the docstring).
            if nxt.strip() != "" and not _BULLET_RE.match(nxt) and not _HEADING_RE.match(nxt):
                segments.append(nxt)
                i += 1
                continue
            break
        joined = _join_quote_segments(segments)
        out.append(f"{indent}- {joined}" if as_bullet else f"> {joined}")
    return out


def _merge_wrapped_lines(lines: list[str]) -> list[str]:
    """Join PDF word-wrap continuation lines back into their source line.

    Must run after `_flatten_blockquote_runs`. A continuation line is a
    non-blank line that is not itself a bullet or heading, immediately
    following a bullet's text line. These occur inside raw quote extracts
    (`근거 원문: "..."`) where the original PDF text wrapped at ~70 chars;
    the newline is not semantically meaningful and must become a space.
    """
    merged: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        is_bullet = bool(_BULLET_RE.match(line))
        is_heading = bool(_HEADING_RE.match(line))
        is_blank = line.strip() == ""
        if not (is_bullet or is_heading or is_blank) and merged:
            merged[-1] = merged[-1].rstrip() + " " + line.strip()
        else:
            merged.append(line)
    return merged


def _reindent_bullets(lines: list[str]) -> list[str]:
    """Convert 2-space markdown sub-bullet indentation to one Notion tab."""
    out = []
    for line in lines:
        m = _BULLET_RE.match(line)
        if m and len(m.group(1)) >= 2:
            out.append("\t- " + m.group(2))
        else:
            out.append(line)
    return out


def transform_body(body: str) -> str:
    lines = body.split("\n")
    lines = _flatten_blockquote_runs(lines)
    lines = _merge_wrapped_lines(lines)
    lines = _reindent_bullets(lines)

    out_lines = []
    for line in lines:
        # A real Notion quote marker ("> ") must survive escaping untouched --
        # only the content after it should be escaped.
        prefix = ""
        content = line
        qm = re.match(r"^> ?", line)
        if qm:
            prefix, content = qm.group(0), line[qm.end():]
        content = convert_locator_italic(content)
        content = _protect_and_escape(content)
        out_lines.append(prefix + content)
    # Collapse 3+ consecutive blank lines to keep output compact.
    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Per-paper parsing
# ---------------------------------------------------------------------------


@dataclass
class PaperRecord:
    title: str
    category_key: str
    authors: str = ""
    year: str = ""
    journal: str = ""
    doi_url: str = ""
    number: int = 0
    body_markdown: str = ""
    source_path: str = ""


def _extract_bib_field(text: str, label: str) -> str:
    m = re.search(rf"^- {re.escape(label)}:\s*(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_revised_paper(path: Path, category_key: str, number: int) -> PaperRecord:
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    assert lines[0].startswith("# "), f"{path} missing H1 title"
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:]).lstrip("\n")

    authors = _extract_bib_field(raw, "저자")
    year = _extract_bib_field(raw, "연도")
    journal = _extract_bib_field(raw, "저널")
    doi = _extract_bib_field(raw, "DOI")
    doi_url = f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi

    return PaperRecord(
        title=title,
        category_key=category_key,
        authors=authors,
        year=year,
        journal=journal,
        doi_url=doi_url,
        number=number,
        body_markdown=transform_body(body),
        source_path=str(path.relative_to(ROOT)),
    )


_CITATION_LINE_RE = re.compile(
    r"^>\s*(?P<authors>[^(]+)\((?P<year>\d{4})\)\.\s*\*(?P<journal>[^*]+)\*,\s*"
    r".*?DOI:\s*(?P<doi>10\.\S+?)\.\s*(?:PMID.*)?$"
)


def parse_new_category_paper(path: Path, category_key: str, number: int) -> PaperRecord:
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    assert lines[0].startswith("# "), f"{path} missing H1 title"
    title = lines[0][2:].strip()

    authors = year = journal = doi = ""
    for line in lines[1:6]:
        m = _CITATION_LINE_RE.match(line.strip())
        if m:
            authors = m.group("authors").strip().rstrip(".").rstrip(",")
            year = m.group("year")
            journal = m.group("journal").strip()
            doi = m.group("doi").rstrip(".")
            break
    doi_url = f"https://doi.org/{doi}" if doi else ""

    body = "\n".join(lines[1:]).lstrip("\n")
    return PaperRecord(
        title=title,
        category_key=category_key,
        authors=authors,
        year=year,
        journal=journal,
        doi_url=doi_url,
        number=number,
        body_markdown=transform_body(body),
        source_path=str(path.relative_to(ROOT)),
    )


def collect_papers() -> list[PaperRecord]:
    records: list[PaperRecord] = []
    number = 1
    for category_dir in sorted(PAPERS_REVISED.iterdir()):
        if not category_dir.is_dir():
            continue
        category_key = category_dir.name
        if category_key == NEW_CATEGORY_DIR.name:
            # This folder is a stray duplicate of mds/papers/07_.../ (copied by
            # an external sync, not produced by the 01/02 pipeline) and also
            # contains the non-paper synthesis doc. The real 3 papers for this
            # category are parsed from NEW_CATEGORY_DIR below instead.
            continue
        for md_path in sorted(category_dir.glob("*.md")):
            records.append(parse_revised_paper(md_path, category_key, number))
            number += 1

    category_key = NEW_CATEGORY_DIR.name
    for md_path in sorted(NEW_CATEGORY_DIR.glob("0[1-3]_*.md")):
        records.append(parse_new_category_paper(md_path, category_key, number))
        number += 1

    return records


# ---------------------------------------------------------------------------
# 06_paper_index_table.md -> Notion <table> block
# ---------------------------------------------------------------------------


def build_notion_table(md_path: Path) -> str:
    raw = md_path.read_text(encoding="utf-8")
    lines = [l for l in raw.split("\n") if l.strip().startswith("|")]
    assert len(lines) >= 3, "expected header + separator + >=1 data row"

    def split_row(line: str) -> list[str]:
        cells = line.strip().strip("|").split("|")
        return [c.strip() for c in cells]

    header = split_row(lines[0])
    data_rows = [split_row(l) for l in lines[2:]]

    out = ['<table fit-page-width="true" header-row="true">']
    out.append("<tr>")
    for cell in header:
        out.append(f"<td>{cell}</td>")
    out.append("</tr>")
    for row in data_rows:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{cell}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = collect_papers()
    manifest = [
        {
            "title": r.title,
            "category_key": r.category_key,
            "category_label": CATEGORY_LABELS[r.category_key],
            "authors": r.authors,
            "year": r.year,
            "journal": r.journal,
            "doi_url": r.doi_url,
            "number": r.number,
            "body_markdown": r.body_markdown,
            "source_path": r.source_path,
        }
        for r in records
    ]
    manifest_path = OUT_DIR / "papers_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Per-category staging .md files for human review.
    by_category: dict[str, list[PaperRecord]] = {}
    for r in records:
        by_category.setdefault(r.category_key, []).append(r)
    for category_key, recs in by_category.items():
        staging_path = OUT_DIR / f"{category_key}.md"
        chunks = []
        for r in recs:
            chunks.append(f"# [{r.number}] {r.title}\n\n")
            chunks.append(f"(저자: {r.authors} | 연도: {r.year} | 저널: {r.journal} | DOI: {r.doi_url})\n\n")
            chunks.append(r.body_markdown)
            chunks.append("\n\n---\n\n")
        staging_path.write_text("".join(chunks), encoding="utf-8")

    table_block = build_notion_table(INDEX_TABLE_MD)
    (OUT_DIR / "06_table_notion.md").write_text(table_block, encoding="utf-8")

    print(f"Parsed {len(records)} papers -> {manifest_path}")
    print(f"Staging files written under {OUT_DIR}")
    print(f"Notion table block written to {OUT_DIR / '06_table_notion.md'} ({table_block.count('<tr>')} rows incl. header)")


if __name__ == "__main__":
    main()
