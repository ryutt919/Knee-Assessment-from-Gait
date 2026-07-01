"""Shared output-path helpers for the Study-review pipeline."""

from __future__ import annotations

import re
from pathlib import Path

CATEGORY_RE = re.compile(r"^0[1-6]_")
UNCATEGORIZED_CATEGORY = "00_uncategorized"


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
