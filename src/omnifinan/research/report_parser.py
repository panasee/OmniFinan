"""Financial report parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedReport:
    text: str
    source: str


def parse_pdf_report(file_path: str | Path) -> ParsedReport:
    """Parse PDF report text with optional pypdf dependency."""
    path = Path(file_path)
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("pypdf is required for PDF parsing. Install with `pip install pypdf`.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return ParsedReport(text="\n".join(pages).strip(), source=str(path))
