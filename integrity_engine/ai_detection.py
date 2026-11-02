"""AI-generated content estimation.

Primary signal: the open-source `Hello-SimpleAI/chatgpt-detector-roberta`
classifier (RoBERTa trained on the HC3 corpus of ChatGPT answers vs human
answers), run in an isolated subprocess because its peak RSS is ~0.9GB and
this pipeline targets commodity hardware. Detector selection rationale -
the GPT-2-era `roberta-base-openai-detector` was evaluated first and
rejected: it fires on nearly all fluent modern prose (~100% AI on human
originals in our corpus), which is documented in EVALUATION.md.

Secondary signal (always computed, used as fallback and corroboration):
stylometric heuristics - burstiness (sentence-length variance), type-token
ratio, and mean sentence length. LLM prose is characteristically uniform
(low burstiness) with smooth vocabulary distribution. The heuristic score
is a calibrated, documented approximation, not a black box.

Final probability: the detector's word-weighted mean when available;
otherwise the heuristic score (the report records which method ran).
"""
import json
import math
import os
import re
import subprocess
import sys

from integrity_engine.config import Config

_CFG = Config()
_WORKER = os.path.join(os.path.dirname(__file__), "ai_detect_worker.py")


def _windows(text: str, window_words: int) -> list:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + window_words])
            for i in range(0, len(words), window_words)][: _CFG.ai_max_windows_per_doc]


def _heuristic_score(text: str) -> dict:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(s.split()) for s in sentences] or [0]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    burstiness = math.sqrt(variance)
    words = re.findall(r"[a-z']+", text.lower())
    ttr = len(set(words)) / max(1, len(words))

    # Calibration notes (documented in README): human academic/business
    # writing typically shows burstiness ~8-14; LLM output clusters ~4-8.
    # Map burstiness 4..12 -> AI likelihood 0.9..0.1 (clamped).
    burst_score = min(1.0, max(0.0, (12.0 - burstiness) / 8.0))
    # Very uniform vocabulary richness also tilts toward machine text.
    ttr_score = 1.0 - min(1.0, max(0.0, (ttr - 0.35) / 0.35))
    score = round(100.0 * (0.7 * burst_score + 0.3 * ttr_score), 1)
    return {
        "heuristic_ai_probability": score,
        "burstiness": round(burstiness, 2),
        "type_token_ratio": round(ttr, 3),
        "mean_sentence_words": round(mean_len, 1),
    }


def _detector_scores(windows: list) -> list:
    """Run the HF classifier in a subprocess; return None if unavailable.

    Accepts a flat list of windows from *many* documents: the worker loads
    the ~0.9GB model once per batch run, not once per document.
    """
    if not windows:
        return []
    payload = json.dumps({"model": _CFG.ai_detector_model, "texts": windows})
    env = dict(os.environ)
    try:
        proc = subprocess.run(
            [sys.executable, _WORKER],
            input=payload, capture_output=True, text=True,
            timeout=1800, env=env,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout.strip().splitlines()[-1])["scores"]
    except Exception:
        return None


def score_document(doc_id: str, text: str) -> dict:
    """Return the AI-likelihood analysis block for one document."""
    windows = _windows(text, _CFG.ai_window_words)
    if not windows:
        return {"ai_generated_probability": 0.0, "method": "insufficient_text",
                "details": {}}

    heur = _heuristic_score(text)
    detector = _detector_scores(windows) if _CFG.ai_subprocess else None

    if detector is not None and len(detector) == len(windows):
        weights = [len(w.split()) for w in windows]
        total = sum(weights)
        prob = sum(s * w for s, w in zip(detector, weights)) / total
        return {
            "ai_generated_probability": round(prob * 100.0, 1),
            "method": "chatgpt-detector-roberta (HC3-trained RoBERTa, open-source)",
            "details": {
                "windows_scored": len(windows),
                "corroborating_heuristics": heur,
            },
        }
    return {
        "ai_generated_probability": heur["heuristic_ai_probability"],
        "method": "stylometric heuristics (detector unavailable; documented fallback)",
        "details": heur,
    }


def score_corpus(doc_texts: dict) -> dict:
    """Score many documents with a single detector subprocess launch.

    Returns {doc_id: analysis_block}; documents with no usable text get an
    insufficient_text result.
    """
    plan = []          # (doc_id, window_text)
    per_doc = {}       # doc_id -> (windows, weights, heuristics)
    for doc_id, text in doc_texts.items():
        windows = _windows(text, _CFG.ai_window_words)
        heur = _heuristic_score(text) if windows else None
        per_doc[doc_id] = (windows, heur)
        plan.extend(windows)

    flat = _detector_scores(plan) if (plan and _CFG.ai_enabled) else None
    if flat is not None and len(flat) != len(plan):
        flat = None

    results = {}
    cursor = 0
    for doc_id, (windows, heur) in per_doc.items():
        if not windows:
            results[doc_id] = {"ai_generated_probability": 0.0,
                               "method": "insufficient_text", "details": {}}
            continue
        if flat is not None:
            scores = flat[cursor:cursor + len(windows)]
            cursor += len(windows)
            weights = [len(w.split()) for w in windows]
            prob = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            results[doc_id] = {
                "ai_generated_probability": round(prob * 100.0, 1),
                "method": "chatgpt-detector-roberta (HC3-trained RoBERTa, open-source)",
                "details": {"windows_scored": len(windows),
                            "corroborating_heuristics": heur},
            }
        else:
            results[doc_id] = {
                "ai_generated_probability": heur["heuristic_ai_probability"],
                "method": "stylometric heuristics (detector unavailable; documented fallback)",
                "details": heur,
            }
    return results
