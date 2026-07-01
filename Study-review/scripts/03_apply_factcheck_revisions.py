#!/usr/bin/env python3
"""Build AS-IS/TO-BE revised copies of the 45 paper summaries from the 02 fact-check findings.

Reads 02_review_paper_summaries.py's review_manifest.json (findings + suggested
corrections) and the original summary MD files (untouched). For each finding,
locates the flagged sentence inside the original MD and replaces that exact
span with an inline [AS-IS]/[TO-BE] block, leaving everything else byte-identical.
Originals in mds/papers/ are never modified; output goes to
mds/papers_revised/{source category}/.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
REVIEW_MANIFEST_PATH = TASK_ROOT / "logs" / "review" / "review_manifest.json"
OUTPUT_DIR = TASK_ROOT / "mds" / "papers_revised"
AGGREGATE_PATH = TASK_ROOT / "mds" / "04_all_papers_revised.md"

from path_utils import categorized_output_path


def comparison_normalized_with_map(text: str) -> tuple[str, list[int]]:
    """Same normalization as 02's comparison_normalized(), but keeps a map from
    each normalized character back to its index in the original (unnormalized) text,
    so a normalized-substring match can be translated into an exact original-text span."""
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = re.sub(r"--- page \d+ ---\s*\d*", "", folded)
    folded = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", folded)
    # folded is derived from text via NFKC+lower+regex drops; map by re-scanning
    # text directly instead of through `folded`, since NFKC can change length.
    norm_chars: list[str] = []
    index_map: list[int] = []
    skip_until = -1
    page_marker = re.compile(r"--- page \d+ ---\s*\d*", re.I)
    i = 0
    lowered = text.lower()
    while i < len(text):
        if i > skip_until:
            m = page_marker.match(lowered, i)
            if m:
                skip_until = m.end() - 1
                i = m.end()
                continue
        ch = unicodedata.normalize("NFKC", text[i]).lower()
        if ch.isalnum():
            norm_chars.append(ch)
            index_map.append(i)
        i += 1
    return "".join(norm_chars), index_map


def comparison_normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"--- page \d+ ---\s*\d*", "", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return "".join(ch for ch in text if ch.isalnum())


def find_span(original: str, quoted: str) -> tuple[int, int] | None:
    exact = original.find(quoted)
    if exact != -1:
        return exact, exact + len(quoted)
    norm_text, idx_map = comparison_normalized_with_map(original)
    norm_quote = comparison_normalized(quoted)
    if not norm_quote:
        return None
    pos = norm_text.find(norm_quote)
    if pos == -1:
        return None
    start = idx_map[pos]
    end = idx_map[pos + len(norm_quote) - 1] + 1
    return start, end


def revision_block(finding: dict) -> str:
    return (
        f"\n> **[AS-IS]** {finding['quoted_summary_text']}\n"
        f">\n"
        f"> **[TO-BE]** {finding['suggested_correction']}\n"
        f">\n"
        f"> _(사실검증 — {finding['issue_type']}/{finding['severity']}: {finding['explanation']})_\n"
    )


def apply_revisions(original_text: str, findings: list[dict]) -> tuple[str, list[dict]]:
    spans = []
    unmatched = []
    for finding in findings:
        span = find_span(original_text, finding["quoted_summary_text"])
        if span is None:
            unmatched.append(finding)
            continue
        spans.append((span[0], span[1], finding))
    spans.sort(key=lambda s: s[0], reverse=True)
    text = original_text
    used: list[tuple[int, int]] = []
    for start, end, finding in spans:
        if any(start < u_end and end > u_start for u_start, u_end in used):
            unmatched.append(finding)
            continue
        text = text[:start] + revision_block(finding).strip("\n") + text[end:]
        used.append((start, end))
    return text, unmatched


def aggregate_revised(revised_paths: list[Path]) -> None:
    parts = ["# 전체 논문 요약 수정본 (AS-IS/TO-BE 사실검증 반영)"]
    for path in revised_paths:
        parts.append(path.read_text(encoding="utf-8").rstrip())
    AGGREGATE_PATH.write_text("\n\n---\n\n".join(parts).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    manifest = json.loads(REVIEW_MANIFEST_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_findings = 0
    total_unmatched = 0
    papers_with_unmatched = []
    revised_paths: list[Path] = []
    for key, record in sorted(manifest["papers"].items(), key=lambda kv: kv[1].get("number", 0)):
        if record.get("status") != "success":
            continue
        data = record["data"]
        original_path = Path(data["source_summary_path"])
        original_text = original_path.read_text(encoding="utf-8")
        findings = data.get("findings", [])
        total_findings += len(findings)
        revised_text, unmatched = apply_revisions(original_text, findings)
        if unmatched:
            total_unmatched += len(unmatched)
            papers_with_unmatched.append((record["number"], data["title"], len(unmatched)))
            revised_text = revised_text.rstrip("\n") + "\n\n## 자동 위치 매칭 실패 항목 (수동 확인 필요)\n\n"
            for finding in unmatched:
                revised_text += (
                    f"- **[AS-IS]** {finding['quoted_summary_text']}\n"
                    f"  **[TO-BE]** {finding['suggested_correction']}\n"
                    f"  _(사실검증 — {finding['issue_type']}/{finding['severity']}: {finding['explanation']})_\n"
                )
        pdf = Path(key)
        out_path = categorized_output_path(TASK_ROOT, "papers_revised", pdf, f"{original_path.stem}_revised.md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(revised_text, encoding="utf-8")
        revised_paths.append(out_path)
        print(f"[{record['number']:02d}] {len(findings)-len([f for f in unmatched])}/{len(findings)} 반영 -> {out_path.name}")
    aggregate_revised(revised_paths)
    print(f"\n총 findings {total_findings}건 중 위치 자동매칭 실패 {total_unmatched}건")
    for number, title, count in papers_with_unmatched:
        print(f"  - [{number:02d}] {title[:60]} : {count}건 수동 확인 필요")
    print(f"통합본: {AGGREGATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
