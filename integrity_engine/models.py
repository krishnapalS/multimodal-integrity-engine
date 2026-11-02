"""Core data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Atom:
    """Smallest addressable unit of extracted text.

    An atom knows where it came from: a PDF page, a DOCX paragraph, or an
    MP4 timestamp range. Chunks are built from atoms and inherit the span
    of atoms they cover, which is what makes the final report traceable.
    """
    text: str
    page: Optional[int] = None            # 1-based page number (PDF)
    paragraph: Optional[int] = None       # 1-based paragraph index (DOCX)
    ts_start: Optional[float] = None      # seconds (MP4)
    ts_end: Optional[float] = None        # seconds (MP4)


@dataclass
class ExtractedDocument:
    doc_id: str                            # file name, e.g. "submission_042.pdf"
    path: str
    fmt: str                               # "pdf" | "docx" | "mp4"
    status: str = "ok"                     # ok | corrupted | empty | no_speech | unsupported
    atoms: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str                          # "{doc_id}#c{n:03d}"
    doc_id: str
    text: str
    word_count: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    para_start: Optional[int] = None
    para_end: Optional[int] = None
    ts_start: Optional[float] = None
    ts_end: Optional[float] = None
    is_boilerplate: bool = False

    def location_label(self) -> str:
        """Human-readable provenance for the audit report."""
        if self.ts_start is not None:
            return f"{_fmt_ts(self.ts_start)} - {_fmt_ts(self.ts_end)}"
        if self.page_start is not None:
            if self.page_end and self.page_end != self.page_start:
                return f"pages {self.page_start}-{self.page_end}"
            return f"page {self.page_start}"
        if self.para_start is not None:
            if self.para_end and self.para_end != self.para_start:
                return f"paragraphs {self.para_start}-{self.para_end}"
            return f"paragraph {self.para_start}"
        return "unknown location"


def _fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return "00:00:00"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
