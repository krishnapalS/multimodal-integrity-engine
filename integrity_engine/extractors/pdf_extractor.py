"""PDF extraction via PyMuPDF with page-level provenance.

Robustness contract (brief scenario 3): zero-byte, truncated, or otherwise
unreadable PDFs must degrade to a flagged document, never crash the run.
Repeated header/footer lines are stripped so page furniture does not pollute
the embedding index.
"""
import os
from collections import Counter

import re

import pymupdf


from integrity_engine.config import Config
from integrity_engine.models import Atom, ExtractedDocument

_CFG = Config()


def _normalize_line(line: str) -> str:
    # Digits are dropped so furniture like "Confidential - page 3" matches
    # the same line on every page regardless of the page number.
    return re.sub(r"\d+", "", " ".join(line.split())).strip().lower()


def _strip_headers_footers(pages_lines: list) -> list:
    """Remove lines repeated at the top/bottom of most pages."""
    n_pages = len(pages_lines)
    if n_pages < 2:
        return pages_lines
    counts = Counter()
    for lines in pages_lines:
        candidates = set()
        if lines:
            candidates.add(lines[0])
            candidates.add(lines[-1])
        if len(lines) > 2:
            candidates.add(lines[1])
            candidates.add(lines[-2])
        for cand in candidates:
            if cand:
                counts[cand] += 1
    threshold = max(2, int(round(n_pages * _CFG.header_footer_min_page_fraction)))
    boiler = {line for line, c in counts.items() if c >= threshold and len(line) > 3}
    cleaned = []
    for lines in pages_lines:
        kept = [ln for ln in lines if ln not in boiler]
        cleaned.append(kept)
    return cleaned


def extract_pdf(path: str) -> ExtractedDocument:
    doc_id = os.path.basename(path)
    doc = ExtractedDocument(doc_id=doc_id, path=path, fmt="pdf")

    if os.path.getsize(path) == 0:
        doc.status = "empty"
        doc.warnings.append("file is 0 bytes; nothing to extract")
        return doc

    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        doc.status = "corrupted"
        doc.warnings.append(f"PyMuPDF could not open file: {exc}")
        return doc

    try:
        pages_lines = []
        for page in pdf:
            raw = page.get_text("text") or ""
            lines = [_normalize_line(ln) for ln in raw.splitlines()]
            pages_lines.append([ln for ln in lines if ln])

        for page_no, page in enumerate(pdf, start=1):
            # Block mode preserves paragraph-ish structure so shared
            # template paragraphs become their own atoms (and chunks).
            for block in page.get_text("blocks"):
                block_text = " ".join(" ".join(block[4].splitlines()).split())
                if block_text:
                    doc.atoms.append(Atom(text=block_text, page=page_no))
            # Apply the header/footer stripper per page: rebuild the page's
            # line list to identify repeated furniture, then drop atoms
            # whose text is exactly a furniture line.
        furniture = set()
        cleaned_pages = _strip_headers_footers(pages_lines)
        for raw_lines, kept_lines in zip(pages_lines, cleaned_pages):
            furniture |= (set(raw_lines) - set(kept_lines))
        if furniture:
            doc.atoms = [a for a in doc.atoms
                         if _normalize_line(a.text) not in furniture]
    except Exception as exc:
        doc.warnings.append(f"partial extraction: {exc}")

    n_pages = len(pdf)
    pdf.close()

    if not doc.atoms:
        # Scanned/image-only PDFs land here: valid file, no text layer.
        doc.status = "empty"
        doc.warnings.append(
            f"no extractable text across {n_pages} page(s); "
            "possibly scanned/image-only PDF"
        )
    return doc
