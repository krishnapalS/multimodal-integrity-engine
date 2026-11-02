#!/usr/bin/env python3
"""Evaluate pipeline reports against the ground-truth manifest.

Reads ./reports/summary.json + per-document JSONs and the manifest, then
prints pair-detection precision/recall/confusion, per-signal checks, and
edge-case handling results. The numbers in EVALUATION.md come from here.
"""
import itertools
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PEER_THRESHOLD = 0.75   # matches config.peer_match_threshold
AI_THRESHOLD = 70.0     # matches the MEDIUM AI band boundary
EXTERNAL_THRESHOLD = 25.0


def main() -> int:
    manifest = json.load(open(os.path.join(ROOT, "sample_submissions", "dataset_manifest.json")))
    summary = json.load(open(os.path.join(ROOT, "reports", "summary.json")))
    reports = {r["document_id"]: r for r in summary["run"]["documents"]}

    gt_pairs = {tuple(sorted(p[:2])) for p in manifest["ground_truth_pairs"]["plagiarized_pairs"]}
    pairwise = {(r["doc_a"], r["b"] if False else r["doc_b"]): r["max_similarity"]
                for r in summary["pairwise"]}

    processable = [d for d, r in reports.items() if r["status"] == "ok"]
    tp = fp = fn = tn = 0
    fp_list, fn_list = [], []
    for a, b in itertools.combinations(sorted(processable), 2):
        sim = pairwise.get((a, b), pairwise.get((b, a), 0.0))
        predicted = sim >= PEER_THRESHOLD
        actual = (a, b) in gt_pairs
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
            fp_list.append((a, b, sim))
        elif not predicted and actual:
            fn += 1
            fn_list.append((a, b, sim))
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("PAIR-DETECTION CONFUSION MATRIX (threshold cosine %.2f)" % PEER_THRESHOLD)
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    if fp_list:
        print("  false positives:", fp_list)
    if fn_list:
        print("  false negatives:", fn_list)

    print("\nPER-SIGNAL CHECKS")
    ai_doc = manifest["ground_truth_pairs"]["ai_generated"][0]
    ai_score = reports[ai_doc]["ai"]
    others_ai = {d: r["ai"] for d, r in reports.items()
                 if r["status"] == "ok" and d != ai_doc}
    max_other_ai = max(others_ai.values())
    print(f"  AI doc {ai_doc}: {ai_score}% (threshold {AI_THRESHOLD}) "
          f"-> {'DETECTED' if ai_score >= AI_THRESHOLD else 'MISSED'}")
    print(f"  max AI score among human originals: {max_other_ai}% "
          f"-> {'CLEAN' if max_other_ai < AI_THRESHOLD else 'FALSE POSITIVE'}")

    ext_doc = manifest["ground_truth_pairs"]["external_copies"][0][0]
    ext_score = reports[ext_doc]["external"]
    others_ext = {d: r["external"] for d, r in reports.items()
                  if r["status"] == "ok" and d != ext_doc}
    max_other_ext = max(others_ext.values())
    print(f"  external-copy doc {ext_doc}: {ext_score}% "
          f"-> {'DETECTED' if ext_score >= EXTERNAL_THRESHOLD else 'MISSED'}")
    print(f"  max external score elsewhere: {max_other_ext}% "
          f"-> {'CLEAN' if max_other_ext < EXTERNAL_THRESHOLD else 'FALSE POSITIVE'}")

    print("\nEDGE-CASE HANDLING")
    for doc_id, meta in manifest["files"].items():
        if "unprocessable_no_crash" in meta["expected_flags"]:
            r = reports.get(doc_id)
            print(f"  {doc_id}: status={r['status']} (expected non-ok, no crash) "
                  f"-> {'OK' if r['status'] != 'ok' else 'UNEXPECTEDLY PROCESSED'}")

    template_docs = manifest["ground_truth_pairs"]["benign_shared_template"]["documents"]
    flagged = [d for d in template_docs
               if d not in {p for pair in gt_pairs for p in pair}
               and reports[d]["peer"] > 0.0]
    print(f"  benign template docs wrongly peer-flagged: {flagged or 'none'}")

    print("\nSUMMARY")
    print(f"  documents={summary['run']['documents_processed']} "
          f"chunks={summary['run']['chunks_indexed']} "
          f"template_excluded={summary['run']['boilerplate_chunks']} "
          f"elapsed={summary['run']['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
