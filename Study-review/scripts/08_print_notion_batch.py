"""Print a notion-create-pages `pages` JSON array for a batch of paper numbers.

Usage:
    python3 Study-review/scripts/08_print_notion_batch.py 2 3 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "logs" / "notion_export" / "papers_manifest.json"


def build_page(r: dict) -> dict:
    props = {
        "논문 제목": r["title"],
        "카테고리": r["category_label"],
        "번호": r["number"],
        "저자": r["authors"],
        "연도": r["year"],
        "저널": r["journal"],
    }
    if r["doi_url"]:
        props["DOI/URL"] = r["doi_url"]
    return {"properties": props, "content": r["body_markdown"]}


def main() -> None:
    numbers = {int(x) for x in sys.argv[1:]}
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = [r for r in records if r["number"] in numbers]
    selected.sort(key=lambda r: r["number"])
    missing = numbers - {r["number"] for r in selected}
    if missing:
        print(f"WARNING: numbers not found: {sorted(missing)}", file=sys.stderr)
    pages = [build_page(r) for r in selected]
    print(json.dumps(pages, ensure_ascii=False))


if __name__ == "__main__":
    main()
