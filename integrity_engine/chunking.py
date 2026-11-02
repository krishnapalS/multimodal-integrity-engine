"""Atom -> chunk segmentation.

Chunks target ~300 words (roughly 200-400 tokens as the brief requests)
with a small overlap so sentences split across a boundary still match.
Each chunk records the full span of atoms it covers, so page numbers and
video timestamps survive into the final audit report.
"""
from integrity_engine.config import Config
from integrity_engine.models import Atom, Chunk, ExtractedDocument

_CFG = Config()


def _words(text: str) -> list:
    return text.split()


def chunk_document(doc: ExtractedDocument) -> list:
    """Split an extracted document into overlapping word-window chunks."""
    if not doc.atoms:
        return []

    # Flatten atoms into (word, atom_index) pairs so overlap can cross
    # atom boundaries while provenance stays exact.
    words = []
    atom_of = []
    for idx, atom in enumerate(doc.atoms):
        for w in _words(atom.text):
            words.append(w)
            atom_of.append(idx)

    # Boundaries where a chunk may close: after any atom that leaves the
    # chunk at least half full. This keeps self-contained units (such as a
    # shared template paragraph) inside their own chunk so the boilerplate
    # filter can isolate them (brief scenario 4).
    break_min = 80   # close at an atom boundary once the chunk holds >= this
    first_word_of_atom = {}
    for wi, ai in enumerate(atom_of):
        first_word_of_atom.setdefault(ai, wi)

    chunks = []
    n = len(words)
    start = 0
    serial = 0
    while start < n:
        end = min(n, start + _CFG.chunk_target_words)
        if end < n:
            # Close at the first atom boundary that leaves the chunk at
            # least `break_min` words long; otherwise hard-cut at target.
            for ai in range(len(doc.atoms)):
                fw = first_word_of_atom.get(ai)
                if fw is None or fw <= start:
                    continue
                if start + break_min <= fw <= end:
                    end = fw
                    break
        # Merge a tiny tail into the previous chunk instead of emitting a
        # fragment that would match nothing meaningful.
        if chunks and (n - end) < _CFG.min_chunk_words:
            end = n
        first_atom = doc.atoms[atom_of[start]]
        last_atom = doc.atoms[atom_of[end - 1]]
        text = " ".join(words[start:end])
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#c{serial:03d}",
            doc_id=doc.doc_id,
            text=text,
            word_count=end - start,
            page_start=first_atom.page, page_end=last_atom.page,
            para_start=first_atom.paragraph, para_end=last_atom.paragraph,
            ts_start=first_atom.ts_start, ts_end=last_atom.ts_end,
        ))
        serial += 1
        if end >= n:
            break
        start = max(start + 1, end - _CFG.chunk_overlap_words)
    return chunks


def chunk_corpus(documents: list) -> list:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


def sentence_windows(doc: ExtractedDocument, window_sentences: int = 3,
                     stride: int = 2) -> list:
    """Fine-grained windows for citation-grade external matching.

    Coarse chunks retrieve; these 3-sentence windows (stride 2) localize the
    exact copied passage so reports quote the offending sentence, not a
    200-word neighbourhood. Each window keeps atom provenance (page /
    paragraph / timestamp) for the report.
    """
    import re
    sentences = []  # (sentence_text, atom)
    for atom in doc.atoms:
        for sent in re.split(r"(?<=[.!?])\s+", atom.text):
            sent = sent.strip()
            if len(sent.split()) >= 4:
                sentences.append((sent, atom))
    windows = []
    for i in range(0, max(1, len(sentences) - window_sentences + 1), stride):
        group = sentences[i:i + window_sentences]
        if not group:
            break
        first_atom, last_atom = group[0][1], group[-1][1]
        windows.append({
            "text": " ".join(g[0] for g in group),
            "page_start": first_atom.page, "page_end": last_atom.page,
            "para_start": first_atom.paragraph, "para_end": last_atom.paragraph,
            "ts_start": first_atom.ts_start, "ts_end": last_atom.ts_end,
        })
    return windows
