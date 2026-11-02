#!/usr/bin/env python3
"""Multimodal Submission Integrity & Plagiarism Detection Engine.

Usage:
    python run_integrity_check.py --input_dir ./sample_submissions \
        --output_dir ./reports

Stages (heavy ML models load one at a time, then unload, so the pipeline
runs on commodity CPU hardware):
  1. Extract text from PDF / DOCX / MP4 with full provenance metadata
  2. Chunk with overlap, embed, and index in FAISS
  3. Peer-to-peer, external-reference, and AI-content signals
  4. Explainable JSON + Markdown audit reports and a batch summary
"""
import argparse
import json
import logging
import os
import sys
import time

from integrity_engine import ai_detection, report as reporting, scoring, similarity
from integrity_engine.chunking import chunk_corpus
from integrity_engine.config import Config
from integrity_engine.embeddings import Embedder
from integrity_engine.external import load_reference_kb
from integrity_engine.extractors import extract
from integrity_engine.indexer import VectorIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("integrity")
CFG = Config()


def stage_extract(input_dir: str) -> list:
    from integrity_engine.extractors import mp4_extractor

    names = sorted(os.listdir(input_dir))
    paths = [os.path.join(input_dir, n) for n in names
             if os.path.isfile(os.path.join(input_dir, n))
             and n.lower().endswith(CFG.supported_extensions)]
    skipped = [n for n in names
               if os.path.isfile(os.path.join(input_dir, n))
               and not n.lower().endswith(CFG.supported_extensions)]
    for n in skipped:
        log.info("skipping non-submission file: %s", n)

    whisper_model = None
    if any(p.lower().endswith(".mp4") for p in paths):
        log.info("loading whisper model '%s' for MP4 transcription", CFG.whisper_model)
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(CFG.whisper_model, device="cpu",
                                     compute_type=CFG.whisper_compute_type)

    documents = []
    for path in paths:
        if path.lower().endswith(".mp4") and whisper_model is not None:
            doc = mp4_extractor.extract_mp4(path, whisper_model)
        else:
            doc = extract(path)
        documents.append(doc)
        level = logging.INFO if doc.status == "ok" else logging.WARNING
        log.log(level, "extracted %-28s status=%-10s atoms=%d %s",
                doc.doc_id, doc.status, len(doc.atoms),
                "; ".join(doc.warnings))

    del whisper_model  # free ASR memory before the embedding stage
    return documents


def _window_location(win: dict) -> str:
    if win.get("ts_start") is not None:
        from integrity_engine.models import _fmt_ts
        return f"{_fmt_ts(win['ts_start'])} - {_fmt_ts(win['ts_end'])}"
    if win.get("page_start") is not None:
        if win["page_end"] and win["page_end"] != win["page_start"]:
            return f"pages {win['page_start']}-{win['page_end']}"
        return f"page {win['page_start']}"
    if win.get("para_start") is not None:
        if win["para_end"] and win["para_end"] != win["para_start"]:
            return f"paragraphs {win['para_start']}-{win['para_end']}"
        return f"paragraph {win['para_start']}"
    return "unknown location"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_dir", required=True,
                        help="folder of candidate submissions (pdf/docx/mp4)")
    parser.add_argument("--output_dir", required=True,
                        help="folder to write JSON/Markdown audit reports")
    parser.add_argument("--reference_dir", default="./reference_kb",
                        help="local reference knowledge base (default ./reference_kb)")
    args = parser.parse_args()
    t0 = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("STAGE 1/4 extraction <- %s", args.input_dir)
    documents = stage_extract(args.input_dir)
    ok_docs = [d for d in documents if d.status == "ok"]
    flagged = [d for d in documents if d.status != "ok"]

    log.info("STAGE 2/4 chunking, embedding, indexing")
    chunks = chunk_corpus(ok_docs)
    ref_chunks = load_reference_kb(args.reference_dir)
    log.info("%d submission chunks, %d reference chunks", len(chunks), len(ref_chunks))

    # Suppress institutional template text before embedding (scenario 4).
    masked = similarity.compute_template_mask(chunks) if chunks else []
    for chunk, kept_words in zip(chunks, masked):
        if len(kept_words) < CFG.min_chunk_words:
            chunk.is_boilerplate = True
    embed_texts = [
        (" ".join(kept) if not c.is_boilerplate else c.text)
        for c, kept in zip(chunks, masked)
    ]

    embedder = Embedder(CFG.embedding_model, CFG.embedding_batch_size)
    vectors = embedder.encode(embed_texts) if chunks else None
    ref_vectors = embedder.encode([c.text for c in ref_chunks]) if ref_chunks else None
    # Fine sentence windows per document for citation-grade external matching.
    from integrity_engine.chunking import sentence_windows
    doc_windows = {doc.doc_id: sentence_windows(doc) for doc in ok_docs}
    window_vectors = {
        doc_id: (embedder.encode([w["text"] for w in wins]) if wins else None)
        for doc_id, wins in doc_windows.items()
    }
    embedder.unload()

    peer_index = VectorIndex(vectors.shape[1]) if chunks else None
    if chunks:
        peer_index.add(vectors, [
            {"doc_id": c.doc_id, "text": c.text, "location": c.location_label()}
            for c in chunks
        ])
    ref_index = VectorIndex(ref_vectors.shape[1]) if ref_chunks else None
    if ref_chunks:
        ref_index.add(ref_vectors, [
            {"doc_id": c.doc_id, "text": c.text, "location": c.location_label()}
            for c in ref_chunks
        ])

    n_boiler = sum(1 for c in chunks if c.is_boilerplate)
    log.info("template suppression excluded %d chunk(s)", n_boiler)

    log.info("STAGE 3/4 similarity + AI-content signals")
    by_doc = {}
    for i, c in enumerate(chunks):
        by_doc.setdefault(c.doc_id, []).append(i)

    # One detector subprocess for the whole corpus (model load ~0.9GB RSS).
    ai_results = ai_detection.score_corpus({
        doc.doc_id: " ".join(chunks[i].text for i in by_doc.get(doc.doc_id, []))
        for doc in ok_docs
    })

    pairwise = {}
    doc_reports = []
    for doc in documents:
        if doc.status != "ok":
            doc_reports.append({
                "document_id": doc.doc_id,
                "total_integrity_risk": "UNABLE TO PROCESS",
                "status": doc.status,
                "scores": {"peer_similarity_percentage": 0.0,
                           "external_similarity_percentage": 0.0,
                           "ai_generated_probability": 0.0},
                "peer_matches": [], "multimodal_matches": [],
                "external_matches": [], "boilerplate_excluded": [],
                "ai_analysis": {}, "explanations":
                    "The file could not be processed; see warnings. It was "
                    "skipped without affecting the rest of the batch.",
                "warnings": doc.warnings,
            })
            continue

        idxs = by_doc.get(doc.doc_id, [])
        doc_chunks = [chunks[i] for i in idxs]
        doc_vectors = vectors[idxs]

        peer_best = similarity.best_matches(
            doc_vectors, doc_chunks, peer_index, CFG.neighbor_search_k,
            exclude_own_doc=True, threshold=CFG.peer_match_threshold)
        peer_pct = similarity.coverage_score(peer_best, doc_chunks)

        # External matching at sentence-window granularity: precise quoted
        # passages for the reviewer (brief task 4).
        ext_rows = []
        ext_pct = 0.0
        wins = doc_windows.get(doc.doc_id, [])
        wv = window_vectors.get(doc.doc_id)
        if wins and ref_index is not None and wv is not None:
            sims, idxs = ref_index.search(wv, 3)
            matched = 0
            for wi, win in enumerate(wins):
                best = None
                for sim, ridx in zip(sims[wi], idxs[wi]):
                    if ridx < 0 or sim < CFG.external_match_threshold:
                        break
                    best = (float(sim), ref_index.records[ridx])
                    break
                if best:
                    matched += 1
                    sim_val, rec = best
                    ext_rows.append({
                        "matched_file": rec["doc_id"],
                        "similarity_score": round(sim_val, 3),
                        "query_passage": win["text"],
                        "matched_passage": similarity._clip(rec["text"]),
                        "query_location": _window_location(win),
                        "matched_location": rec["location"],
                    })
            ext_pct = round(100.0 * matched / len(wins), 1)
            ext_rows.sort(key=lambda r: r["similarity_score"], reverse=True)
            ext_rows = ext_rows[:CFG.top_matches_per_doc]

        peer_rows = similarity.top_matches(peer_best, doc_chunks,
                                           CFG.top_matches_per_doc)
        # Split cross-modal evidence out for reviewer visibility.
        mm_rows = [r for r in peer_rows
                   if r["matched_file"].lower().endswith(".mp4") != doc.doc_id.lower().endswith(".mp4")]
        peer_rows = [r for r in peer_rows if r not in mm_rows]

        full_text = " ".join(c.text for c in doc_chunks)
        ai = ai_results.get(doc.doc_id) or ai_detection.score_document(doc.doc_id, full_text)
        ai_pct = ai["ai_generated_probability"]

        for other, sim in similarity.aggregate_pair_similarity(peer_best, doc_chunks).items():
            key = tuple(sorted((doc.doc_id, other)))
            pairwise[key] = max(pairwise.get(key, 0.0), round(sim, 3))

        boiler = [{"location": c.location_label(),
                   "excerpt": " ".join(c.text.split()[:40]) + " ..."}
                  for c in doc_chunks if c.is_boilerplate]

        risk = scoring.risk_band(peer_pct, ext_pct, ai_pct)
        doc_reports.append({
            "document_id": doc.doc_id,
            "total_integrity_risk": risk,
            "status": "ok",
            "scores": {
                "peer_similarity_percentage": peer_pct,
                "external_similarity_percentage": ext_pct,
                "ai_generated_probability": ai_pct,
            },
            "peer_matches": peer_rows,
            "multimodal_matches": mm_rows,
            "external_matches": ext_rows,
            "boilerplate_excluded": boiler,
            "ai_analysis": ai,
            "explanations": scoring.explain(
                doc.doc_id, peer_pct, ext_pct, ai_pct,
                peer_rows, ext_rows, mm_rows, len(boiler), ai["method"]),
            "warnings": doc.warnings,
        })

    log.info("STAGE 4/4 writing reports -> %s", args.output_dir)
    run_summary = {
        "documents_processed": len(documents),
        "documents_ok": len(ok_docs),
        "documents_flagged": len(flagged),
        "chunks_indexed": len(chunks),
        "boilerplate_chunks": n_boiler,
        "elapsed_seconds": round(time.time() - t0, 1),
        "documents": [
            {"document_id": r["document_id"], "risk": r["total_integrity_risk"],
             "peer": r["scores"]["peer_similarity_percentage"],
             "external": r["scores"]["external_similarity_percentage"],
             "ai": r["scores"]["ai_generated_probability"],
             "status": r["status"]}
            for r in doc_reports
        ],
    }
    for rep in doc_reports:
        md = reporting.render_document_markdown(rep)
        reporting.write_document_reports(rep, md, args.output_dir)
    matrix_rows = [{"doc_a": a, "doc_b": b, "max_similarity": s}
                   for (a, b), s in pairwise.items()]
    reporting.write_summary(run_summary, matrix_rows, args.output_dir)

    log.info("done in %.1fs: %d documents, %d flagged",
             time.time() - t0, len(documents), len(flagged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
