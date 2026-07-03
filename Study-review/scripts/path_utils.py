"""Shared input/output path helpers for the Study-review pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

CATEGORY_RE = re.compile(r"^0[1-7]_")
UNCATEGORIZED_CATEGORY = "00_uncategorized"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdfs_from_input_manifest(manifest_path: Path) -> list[Path]:
    """Load the exact PDF batch declared by a JSON input manifest.

    The canonical shape is ``{"papers": [{"canonical_pdf_path": ..., "sha256": ...}]}``.
    ``final_pdf_path``, ``pdf_path`` and ``path`` are accepted as aliases to keep batch manifests
    portable between the categorization and analysis steps.
    """
    manifest_path = manifest_path.resolve()
    payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("papers") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("input manifest의 papers는 배열이어야 합니다")

    pdfs: list[Path] = []
    seen: set[Path] = set()
    for index, entry in enumerate(entries, 1):
        if isinstance(entry, str):
            raw_path, expected_hash = entry, None
        elif isinstance(entry, dict):
            raw_path = next(
                (entry.get(key) for key in ("canonical_pdf_path", "final_pdf_path", "pdf_path", "path") if entry.get(key)),
                None,
            )
            expected_hash = entry.get("sha256") or entry.get("pdf_hash")
        else:
            raise ValueError(f"input manifest papers[{index}] 형식이 올바르지 않습니다")
        if not raw_path:
            raise ValueError(f"input manifest papers[{index}]에 PDF 경로가 없습니다")
        pdf = Path(raw_path).expanduser()
        if not pdf.is_absolute():
            pdf = manifest_path.parent / pdf
        pdf = pdf.resolve()
        if pdf.suffix.lower() != ".pdf" or not pdf.is_file():
            raise FileNotFoundError(f"input manifest PDF를 찾을 수 없습니다: {pdf}")
        if expected_hash and _sha256(pdf).lower() != str(expected_hash).lower():
            raise ValueError(f"input manifest SHA-256 불일치: {pdf}")
        if pdf not in seen:
            seen.add(pdf)
            pdfs.append(pdf)
    return pdfs


def category_for_pdf(pdf: Path) -> str:
    """Return the existing docs/ref_papers category folder for a PDF.

    The default corpus stores PDFs directly below folders such as
    ``01_acl_gait_biomechanics_studies``. For custom input folders, walk up the
    parent chain and use the first matching category-like folder if one exists.
    """
    for parent in (pdf.parent, *pdf.parents):
        if CATEGORY_RE.match(parent.name):
            return parent.name
    return UNCATEGORIZED_CATEGORY


def categorized_output_path(output_root: Path, kind: str, pdf: Path, filename: str) -> Path:
    """Build ``mds/{kind}/{category}/{filename}`` under the output root."""
    return output_root / "mds" / kind / category_for_pdf(pdf) / filename
