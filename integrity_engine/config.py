"""Central configuration for the integrity engine.

All thresholds and model choices live here so the pipeline is tunable
without touching business logic. Every value is documented in README.md.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Chunking -------------------------------------------------------
    chunk_target_words: int = 220      # ~200-400 tokens per the brief
    chunk_overlap_words: int = 50      # slight overlap to preserve context
    min_chunk_words: int = 25          # tail chunks smaller than this merge back

    # --- Embedding / indexing ------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 16

    # --- Similarity ------------------------------------------------------
    peer_match_threshold: float = 0.75      # cosine >= this counts as a match
    external_match_threshold: float = 0.72  # slightly looser: cross-source wording
    boilerplate_similarity: float = 0.85    # near-duplicate cutoff
    boilerplate_min_docs: int = 3           # in >= N docs => common template
    top_matches_per_doc: int = 8            # max evidence rows per report section
    neighbor_search_k: int = 12             # FAISS neighbours inspected per chunk

    # --- AI-content detection -------------------------------------------
    ai_detector_model: str = "Hello-SimpleAI/chatgpt-detector-roberta"
    ai_window_words: int = 350              # score windows of this size
    ai_max_windows_per_doc: int = 12        # cap cost on long docs
    ai_subprocess: bool = True              # isolate the ~0.9GB detector RSS
    ai_enabled: bool = os.environ.get("AI_DETECTION", "on") != "off"

    # --- Risk bands (percentages) ---------------------------------------
    risk_high_peer: float = 50.0
    risk_high_external: float = 50.0
    risk_high_ai: float = 90.0
    risk_medium_peer: float = 25.0
    risk_medium_external: float = 25.0
    risk_medium_ai: float = 70.0

    # --- Extraction ------------------------------------------------------
    header_footer_min_page_fraction: float = 0.6  # line on >=60% pages => strip
    whisper_model: str = "tiny"
    whisper_compute_type: str = "int8"
    supported_extensions: tuple = field(default=(".pdf", ".docx", ".mp4"))
