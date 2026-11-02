"""Format-specific extractors, dispatched by file extension."""
from integrity_engine.models import ExtractedDocument


def extract(path: str) -> ExtractedDocument:
    """Route a file to the right extractor. Never raises: any failure is
    captured as a corrupted/unsupported document with a warning so one bad
    file can never crash the batch (brief scenario 3)."""
    import os
    from integrity_engine.extractors import docx_extractor, mp4_extractor, pdf_extractor

    ext = os.path.splitext(path)[1].lower()
    doc_id = os.path.basename(path)
    try:
        if ext == ".pdf":
            return pdf_extractor.extract_pdf(path)
        if ext == ".docx":
            return docx_extractor.extract_docx(path)
        if ext == ".mp4":
            return mp4_extractor.extract_mp4(path)
        return ExtractedDocument(doc_id=doc_id, path=path, fmt=ext.lstrip("."),
                                 status="unsupported",
                                 warnings=[f"unsupported file type '{ext}'"])
    except Exception as exc:  # last-resort guard; extractors also guard internally
        return ExtractedDocument(doc_id=doc_id, path=path, fmt=ext.lstrip("."),
                                 status="corrupted",
                                 warnings=[f"unexpected extractor error: {exc}"])
