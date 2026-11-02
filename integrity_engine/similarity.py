"""Peer-to-peer and external-reference similarity.

Scoring model (documented in README.md):
- Every chunk gets its best-matching chunk from *other* submissions
  (self-matches excluded by doc_id) and separately from the reference KB.
- A chunk counts as "matched" when its best cosine clears the threshold.
- A document's similarity percentage is its matched-chunk coverage:
  matched chunks / non-boilerplate chunks x 100. Coverage is deliberately
  conservative: one lucky sentence cannot brand a whole document, and a
  fully copied document cannot hide behind one original paragraph.
- Boilerplate: chunks near-duplicated across >= N different submissions
  (default 3) are common template/institutional text (brief scenario 4).
  They are excluded from scoring and reported separately so reviewers can
  still see them.
"""
from collections import defaultdict

import numpy as np

from integrity_engine.config import Config
from integrity_engine.indexer import VectorIndex

_CFG = Config()


def compute_template_mask(chunks: list, shingle_size: int = 8,
                          df_threshold: int = 3) -> list:
    """Document-frequency boilerplate suppression (brief scenario 4).

    Build the set of word 8-grams (shingles) that appear in >= df_threshold
    distinct submissions - this is institutional template text (case prompts,
    shared instructions, standard citations). Return, per chunk, a masked
    word list with template-covered words removed. Masking happens BEFORE
    embedding, so shared templates can neither create false matches nor
    inflate similarity; verbatim copying between two submissions keeps
    df = 2 and is never masked.
    """
    shingle_docs = defaultdict(set)

    def shingles(words):
        norm = [w.lower().strip(".,;:!?()[]\"'") for w in words]
        return [tuple(norm[i:i + shingle_size])
                for i in range(len(norm) - shingle_size + 1)]

    per_chunk = []
    for chunk in chunks:
        words = chunk.text.split()
        sh = shingles(words)
        per_chunk.append((words, sh))
        for s in set(sh):
            shingle_docs[s].add(chunk.doc_id)

    template_shingles = {s for s, docs in shingle_docs.items()
                         if len(docs) >= df_threshold}

    masked_all = []
    for chunk, (words, sh) in zip(chunks, per_chunk):
        covered = [False] * len(words)
        for i, s in enumerate(sh):
            if s in template_shingles:
                for j in range(i, min(i + shingle_size, len(words))):
                    covered[j] = True
        masked_all.append([w for w, c in zip(words, covered) if not c])
    return masked_all


def mark_boilerplate(chunks: list, vectors: np.ndarray,
                     similarity: float = None, min_docs: int = None) -> int:
    """Flag chunks that recur near-verbatim across many submissions."""
    similarity = similarity or _CFG.boilerplate_similarity
    min_docs = min_docs or _CFG.boilerplate_min_docs
    n = len(chunks)
    if n == 0:
        return 0
    sims = vectors @ vectors.T  # normalized vectors => cosine matrix
    flagged = 0
    for i in range(n):
        docs_hit = {
            chunks[j].doc_id
            for j in np.where(sims[i] >= similarity)[0]
            if chunks[j].doc_id != chunks[i].doc_id
        }
        if len(docs_hit) >= (min_docs - 1):  # +1 for its own document
            chunks[i].is_boilerplate = True
            flagged += 1
    return flagged


def best_matches(query_vectors: np.ndarray, query_chunks: list,
                 index: VectorIndex, k: int, exclude_own_doc: bool,
                 threshold: float) -> list:
    """For each query chunk, find its best match above threshold in `index`.

    Returns a list parallel to query_chunks: None or a dict with the matched
    record and cosine score.
    """
    sims, idxs = index.search(query_vectors, k)
    results = []
    for qi in range(len(query_chunks)):
        best = None
        for sim, idx in zip(sims[qi], idxs[qi]):
            if idx < 0:
                continue
            rec = index.records[idx]
            if exclude_own_doc and rec["doc_id"] == query_chunks[qi].doc_id:
                continue  # self-match exclusion (brief task 2)
            if sim < threshold:
                break  # search results are sorted descending
            best = {"record": rec, "similarity": float(sim)}
            break
        results.append(best)
    return results


def coverage_score(chunk_matches: list, chunks: list) -> float:
    """Matched-chunk coverage percentage, excluding boilerplate chunks."""
    eligible = [i for i, c in enumerate(chunks) if not c.is_boilerplate]
    if not eligible:
        return 0.0
    matched = sum(1 for i in eligible if chunk_matches[i] is not None)
    return round(100.0 * matched / len(eligible), 1)


def top_matches(chunk_matches: list, chunks: list, limit: int) -> list:
    """Flatten chunk-level evidence into ranked report rows."""
    rows = []
    for match, chunk in zip(chunk_matches, chunks):
        if match is None or chunk.is_boilerplate:
            continue
        rec = match["record"]
        rows.append({
            "matched_file": rec["doc_id"],
            "similarity_score": round(match["similarity"], 3),
            "query_passage": _clip(chunk.text),
            "matched_passage": _clip(rec["text"]),
            "query_location": chunk.location_label(),
            "matched_location": rec["location"],
        })
    rows.sort(key=lambda r: r["similarity_score"], reverse=True)
    # De-duplicate near-identical evidence rows from overlapping chunks:
    # keep the highest-scoring row per (matched_file, ~same query span).
    seen = set()
    deduped = []
    for row in rows:
        key = (row["matched_file"], row["query_location"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def aggregate_pair_similarity(chunk_matches: list, chunks: list) -> dict:
    """Best chunk similarity per matched document, for the summary matrix."""
    best_per_doc = defaultdict(float)
    for match, chunk in zip(chunk_matches, chunks):
        if match is None or chunk.is_boilerplate:
            continue
        other = match["record"]["doc_id"]
        best_per_doc[other] = max(best_per_doc[other], match["similarity"])
    return dict(best_per_doc)


def _clip(text: str, words: int = 60) -> str:
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:words]) + " ..."
