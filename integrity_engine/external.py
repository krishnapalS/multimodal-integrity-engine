"""External/web reference similarity against a local reference knowledge base.

The brief explicitly permits simulating web search with a local reference
KB (Wikipedia-style articles). In production this module is the swap point
for a real search API; the matching mechanics stay identical.
"""
import os

from integrity_engine.chunking import chunk_document
from integrity_engine.models import ExtractedDocument, Atom


def load_reference_kb(kb_dir: str) -> list:
    """Load each .txt file in kb_dir as a reference 'document' and chunk it."""
    docs = []
    if not os.path.isdir(kb_dir):
        return docs
    for name in sorted(os.listdir(kb_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(kb_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        doc = ExtractedDocument(
            doc_id=f"reference:{name}",
            path=path,
            fmt="reference",
            atoms=[Atom(text=para, paragraph=i + 1)
                   for i, para in enumerate(p.strip() for p in text.split("\n\n"))
                   if para.strip()],
        )
        docs.append(doc)
    # Reference entries are chunked at paragraph granularity: a copied
    # passage then dominates its chunk instead of being diluted across a
    # whole article, which keeps citation matching sensitive.
    from integrity_engine.models import Chunk
    chunks = []
    for doc in docs:
        for i, atom in enumerate(doc.atoms):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#c{i:03d}",
                doc_id=doc.doc_id,
                text=atom.text,
                word_count=len(atom.text.split()),
                para_start=atom.paragraph, para_end=atom.paragraph,
            ))
    return chunks
