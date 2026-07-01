#!/usr/bin/env python3
"""Build an AS-IS/TO-BE revised copy of mds/02_referenceable_claims.md.

02_referenceable_claims.md (01's render_claims() aggregate) only ever reflects
the original, pre-fact-check referenceable_claims. The 02 fact-check pass found
34 issues across 21 papers specifically in that section, and those corrections
currently only live inline inside mds/papers_revised/*.md. This script re-renders
each paper's claims block from the original data (same line format render_claims()
uses) and applies the same AS-IS/TO-BE substitution 03 already validated, so the
citation-ready claims document also reflects the fact-check fixes. The original
mds/02_referenceable_claims.md is never modified.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_MANIFEST_PATH = TASK_ROOT / "logs" / "manifest.json"
REVIEW_MANIFEST_PATH = TASK_ROOT / "logs" / "review" / "review_manifest.json"
AGGREGATE_PATH = TASK_ROOT / "mds" / "05_referenceable_claims_revised.md"

_spec = importlib.util.spec_from_file_location(
    "apply_factcheck_revisions", TASK_ROOT / "scripts" / "03_apply_factcheck_revisions.py"
)
revisions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(revisions)


def render_claims(data: dict[str, Any]) -> str:
    """Mirrors 01_analyze_reference_papers.py's render_claims() line-for-line."""
    lines = [
        f"## {data['title']}", "",
        f"- 분석 대상 논문 인용표기: {data['paper_bibliography']}",
        f"- 분석 provider: {data['provider']}", "",
    ]
    for i, claim in enumerate(data["referenceable_claims"], 1):
        lines += [
            f"### {i}. {claim['topic']}", "", f"- 원문 발췌: “{claim['original_quote']}”",
            f"- 한국어 번역: {claim['korean_translation']}", f"- 원문 위치: {claim['locator']}",
            f"- 원문 내 인용표기: {claim['in_text_citation']}", f"- 해당 선행문헌: {claim['cited_reference']}",
            f"- 활용 맥락과 주의: {claim['usage_note']}", "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    analysis_manifest = json.loads(ANALYSIS_MANIFEST_PATH.read_text(encoding="utf-8"))
    review_manifest = json.loads(REVIEW_MANIFEST_PATH.read_text(encoding="utf-8"))
    review_by_pdf = {k: v for k, v in review_manifest["papers"].items() if v.get("status") == "success"}

    parts = ["# 레퍼런스 활용 내용 모음 (사실검증 반영)"]
    total_findings = 0
    total_unmatched = 0
    papers_with_unmatched = []
    for key, record in sorted(analysis_manifest["papers"].items(), key=lambda kv: review_by_pdf.get(kv[0], {}).get("number", 0)):
        if record.get("status") != "success" or key not in review_by_pdf:
            continue
        data = record["data"]
        number = review_by_pdf[key]["number"]
        findings = [f for f in review_by_pdf[key]["data"].get("findings", []) if f["section"] == "referenceable_claims"]
        total_findings += len(findings)
        text = render_claims(data)
        revised_text, unmatched = revisions.apply_revisions(text, findings)
        if unmatched:
            total_unmatched += len(unmatched)
            papers_with_unmatched.append((number, data["title"], len(unmatched)))
            revised_text = revised_text.rstrip("\n") + "\n\n#### 자동 위치 매칭 실패 항목 (수동 확인 필요)\n\n"
            for finding in unmatched:
                revised_text += (
                    f"- **[AS-IS]** {finding['quoted_summary_text']}\n"
                    f"  **[TO-BE]** {finding['suggested_correction']}\n"
                    f"  _(사실검증 — {finding['issue_type']}/{finding['severity']}: {finding['explanation']})_\n"
                )
        parts.append(revised_text.rstrip())
        print(f"[{number:02d}] {len(findings)-len(unmatched)}/{len(findings)} 반영 -> {data['title'][:60]}")

    AGGREGATE_PATH.write_text("\n\n---\n\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"\n총 referenceable_claims findings {total_findings}건 중 위치 자동매칭 실패 {total_unmatched}건")
    for number, title, count in papers_with_unmatched:
        print(f"  - [{number:02d}] {title[:60]} : {count}건 수동 확인 필요")
    print(f"통합본: {AGGREGATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
