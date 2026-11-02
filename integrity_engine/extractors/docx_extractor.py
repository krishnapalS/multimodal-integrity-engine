"""DOCX extraction via python-docx with paragraph-level provenance.

DOCX has no fixed pagination (it reflows per renderer), so location metadata
uses 1-based paragraph indices; the assumption is documented in README.md.
"""
import os

import docx

from integrity_engine.models import Atom, ExtractedDocument


def extract_docx(path: str) -> ExtractedDocument:
    doc_id = os.path.basename(path)
    doc = ExtractedDocument(doc_id=doc_id, path=path, fmt="docx")

    if os.path.getsize(path) == 0:
        doc.status = "empty"
        doc.warnings.append("file is 0 bytes; nothing to extract")
        return doc

    try:
        document = docx.Document(path)
    except Exception as exc:
        doc.status = "corrupted"
        doc.warnings.append(f"python-docx could not open file: {exc}")
        return doc

    para_idx = 0
    try:
        for para in document.paragraphs:
            text = " ".join(para.text.split())
            if not text:
                continue
            para_idx += 1
            doc.atoms.append(Atom(text=text, paragraph=para_idx))
        # Table cell text often carries real content in case submissions.
        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(
                    " ".join(cell.text.split()) for cell in row.cells if cell.text.strip()
                )
                if row_text.strip(" |"):
                    para_idx += 1
                    doc.atoms.append(Atom(text=row_text, paragraph=para_idx))
    except Exception as exc:
        doc.warnings.append(f"partial extraction: {exc}")

    if not doc.atoms:
        doc.status = "empty"
        doc.warnings.append("no extractable text in document body or tables")
    return doc
