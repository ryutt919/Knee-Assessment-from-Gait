#!/usr/bin/env python3
"""Move existing per-paper Study-review MD outputs into ref_papers categories."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from path_utils import categorized_output_path, category_for_pdf


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


analyzer = load_module("study_review_analyzer", SCRIPTS_DIR / "01_analyze_reference_papers.py")
reviewer = load_module("study_review_reviewer", SCRIPTS_DIR / "02_review_paper_summaries.py")
reviser = load_module("study_review_reviser", SCRIPTS_DIR / "03_apply_factcheck_revisions.py")
claims_rebuilder = load_module("study_review_claims_rebuilder", SCRIPTS_DIR / "04_rebuild_referenceable_claims.py")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp = Path(fh.name)
    tmp.replace(path)


def move_one(src: Path, dst: Path, dry_run: bool) -> str:
    if src.resolve() == dst.resolve():
        return "already"
    if src.exists():
        if dst.exists():
            if analyzer.sha256(src) != analyzer.sha256(dst):
                raise FileExistsError(f"destination exists with different content: {dst}")
            if not dry_run:
                src.unlink()
            return "deduped"
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
        return "moved"
    if dst.exists():
        return "already"
    return "missing"


def expected_for(pdf: Path, number: int) -> dict[str, Path]:
    safe = analyzer.safe_name(pdf)
    return {
        "paper": categorized_output_path(TASK_ROOT, "papers", pdf, f"{number:02d}_{safe}.md"),
        "review": categorized_output_path(TASK_ROOT, "reviews", pdf, f"{number:02d}_{safe}_review.md"),
        "revised": categorized_output_path(TASK_ROOT, "papers_revised", pdf, f"{number:02d}_{safe}_revised.md"),
    }


def flat_for(pdf: Path, number: int) -> dict[str, Path]:
    safe = analyzer.safe_name(pdf)
    return {
        "paper": TASK_ROOT / "mds" / "papers" / f"{number:02d}_{safe}.md",
        "review": TASK_ROOT / "mds" / "reviews" / f"{number:02d}_{safe}_review.md",
        "revised": TASK_ROOT / "mds" / "papers_revised" / f"{number:02d}_{safe}_revised.md",
    }


def patch_analysis_manifest(pdfs: list[Path], global_index: dict[str, int], dry_run: bool) -> dict[str, Any] | None:
    path = TASK_ROOT / "logs" / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["input_dir"] = str(analyzer.DEFAULT_INPUT.resolve())
    manifest["inventory"] = [str(p.resolve()) for p in pdfs]
    for key, record in manifest.get("papers", {}).items():
        pdf = Path(key)
        number = global_index.get(str(pdf.resolve()), record.get("number"))
        if not number:
            continue
        expected = expected_for(pdf, number)["paper"]
        record["number"] = number
        record["category"] = category_for_pdf(pdf)
        if record.get("status") == "success":
            record["output_path"] = str(expected)
            if expected.exists():
                record["output_hash"] = analyzer.sha256(expected)
    if not dry_run:
        atomic_json(path, manifest)
    return manifest


def patch_review_manifest(pdfs: list[Path], global_index: dict[str, int], dry_run: bool) -> dict[str, Any] | None:
    path = TASK_ROOT / "logs" / "review" / "review_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["inventory"] = [str(p.resolve()) for p in pdfs]
    for key, record in manifest.get("papers", {}).items():
        pdf = Path(key)
        number = global_index.get(str(pdf.resolve()), record.get("number"))
        if not number:
            continue
        expected = expected_for(pdf, number)
        record["number"] = number
        record["category"] = category_for_pdf(pdf)
        if record.get("status") == "success":
            record["review_path"] = str(expected["review"])
            if expected["review"].exists():
                record["output_hash"] = analyzer.sha256(expected["review"])
            data = record.get("data", {})
            data["source_pdf"] = str(pdf)
            data["source_summary_path"] = str(expected["paper"])
            record["data"] = data
    if not dry_run:
        atomic_json(path, manifest)
    return manifest


def print_counts(pdfs: list[Path]) -> None:
    categories: dict[str, int] = {}
    for pdf in pdfs:
        category = category_for_pdf(pdf)
        categories[category] = categories.get(category, 0) + 1
    print("카테고리별 PDF 수:")
    for category, count in sorted(categories.items()):
        print(f"- {category}: {count}")


def count_flat_outputs() -> dict[str, int]:
    counts = {}
    for kind in ("papers", "reviews", "papers_revised"):
        folder = TASK_ROOT / "mds" / kind
        counts[kind] = len(list(folder.glob("*.md"))) if folder.exists() else 0
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pdfs = analyzer.discover_pdfs(analyzer.DEFAULT_INPUT)
    global_index = {str(pdf.resolve()): i for i, pdf in enumerate(pdfs, 1)}
    stats = {"moved": 0, "already": 0, "deduped": 0, "missing": 0}

    print(f"대상 PDF: {len(pdfs)}편")
    print_counts(pdfs)
    print("이동 전 flat MD 수:", count_flat_outputs())

    for pdf in pdfs:
        number = global_index[str(pdf.resolve())]
        flat = flat_for(pdf, number)
        expected = expected_for(pdf, number)
        for key in ("paper", "review", "revised"):
            result = move_one(flat[key], expected[key], args.dry_run)
            stats[result] += 1

    print("정리 결과:", stats)
    if args.dry_run:
        print("dry-run: 파일과 manifest를 변경하지 않았습니다.")
        return 0

    analysis_manifest = patch_analysis_manifest(pdfs, global_index, dry_run=False)
    review_manifest = patch_review_manifest(pdfs, global_index, dry_run=False)
    if analysis_manifest:
        analyzer.aggregate(analysis_manifest["papers"], TASK_ROOT)
    if review_manifest:
        reviewer.aggregate_reviews(review_manifest["papers"], TASK_ROOT)
    revised_paths = [expected_for(pdf, global_index[str(pdf.resolve())])["revised"] for pdf in pdfs]
    reviser.aggregate_revised([path for path in revised_paths if path.exists()])
    claims_rebuilder.main()

    print("이동 후 flat MD 수:", count_flat_outputs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
