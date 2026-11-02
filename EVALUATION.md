# Evaluation

All numbers below are produced by `python tools/evaluate.py` against
`reports/` from the checked-in sample corpus (13 files, ground truth in
`sample_submissions/dataset_manifest.json`). Nothing here is hand-written.

## 1. Pair detection vs ground truth

A document pair is predicted plagiarized when its maximum chunk-level cosine
(over template-suppressed embeddings) is >= 0.75. Ground truth contains 3
plagiarized pairs and 52 benign pairs among the 11 processable documents:

| | Predicted copied | Predicted clean |
| --- | --- | --- |
| **Actually copied** | TP = 3 | FN = 0 |
| **Actually clean** | FP = 0 | TN = 52 |

**Precision 1.000, Recall 1.000, F1 1.000** on this corpus.

Detected pairs:

| Pair | Relationship | Max chunk cosine |
| --- | --- | --- |
| submission_003_drones.pdf x submission_007_drones_copy.docx | exact copy across formats | 1.000 |
| submission_001_greencharge.pdf x submission_006_greencharge_reworded.docx | paraphrase | 0.870 |
| submission_001_greencharge.pdf x submission_011_pitch.mp4 | cross-modal (spoken reading) | 0.843 |

## 2. Per-signal checks

| Signal | Ground truth | Result |
| --- | --- | --- |
| AI content | submission_008 (LLM-styled essay) | **99.9%** AI probability -> HIGH |
| AI false positives | 10 human-written originals | max **0.5%** -> clean |
| External copy | submission_004 (2 verbatim reference passages) | **50.0%** external coverage -> HIGH, passages quoted at cosine 1.0 with page + reference-paragraph citations |
| External false positives | all other documents | max **0.0%** -> clean |
| Benign shared template | 5 documents share a ~115-word case prompt | 5 template chunks suppressed; **no** document peer-flagged for it |
| Corrupted input | zero-byte PDF | status `empty`, warning recorded, batch unaffected |
| Silent input | MP4 without audio stream | status `no_speech`, warning recorded, batch unaffected |

End-to-end batch time: **26.5s** on a 1-core / 1 GB container
(ASR ~7s, embedding ~4s, AI detector subprocess ~11s, rest I/O and matching).

## 3. Failure analysis (honest, with resolutions)

### 3.1 The first-choice AI detector failed catastrophically, and was replaced
`roberta-base-openai-detector` (the detector the brief's FAQ gestures at)
scored **every** document in the corpus - human originals included - at
99-100% AI probability. It was trained on GPT-2 outputs and simply fires on
fluent modern prose. Shipping it would have made the AI signal worthless
(100% false-positive rate on human text). It was replaced with
`Hello-SimpleAI/chatgpt-detector-roberta` (trained on the HC3 corpus of
real ChatGPT answers), which separates the corpus perfectly (99.9% vs <=0.5%).
**Residual limitation:** a student who lightly edits LLM output, or uses a
model stylistically different from 2023-era ChatGPT, may score low; the
stylometric heuristics in each report's `ai_analysis.details` are the second
opinion, and production would add a perplexity signal from a small causal LM.

### 3.2 ASR transcription errors cap cross-modal similarity
whisper-tiny mistranscribes some TTS speech ("purchasing" -> "per chasing",
"GreenCharge" -> "green charge"), so the cross-modal match scores 0.843
instead of ~1.0 for what is verbatim content. The threshold has headroom,
but genuinely accented or noisy speech would degrade this further.
**Resolution path:** whisper-base or small on GPU workers (see scaling
brief), plus ASR confidence-aware chunk weighting.

### 3.3 Chunk dilution hid verbatim external copying (found and fixed during development)
An early iteration matched whole ~240-word chunks against whole reference
articles: a 60-word verbatim passage inside a 240-word chunk scored only
~0.62 and slipped under threshold. This motivated the sentence-window
(3-sentence, stride 2) external matcher, which now catches the passage at
cosine 1.0. **Residual limitation:** a copied passage paraphrased at the
sentence level (not just chunk-diluted) scores lower; the 0.72 external
threshold plus coverage percentage keeps the signal honest rather than
hidden.

### 3.4 Paraphrase recall is deliberately not 100%
The paraphrased document's coverage is 66.7%, not 100%: its most aggressively
reworded sections fall under the 0.75 chunk threshold. Raising recall by
lowering the threshold to ~0.70 was tested in development and immediately
produced false matches between *original* documents that merely discuss the
same case study (topical similarity ~0.72-0.76). The current operating point
trades some paraphrase recall for near-zero false positives; the pairwise
matrix in `summary.md` exposes the borderline region to reviewers.

### 3.5 Template masking could theoretically hide a copied prompt
Because any 8-gram shared by >=3 documents is masked, a student copying a
*third* student's answer to a shared prompt gets the prompt portion ignored
(correct) - but if three students collude with identical full answers, the
shared text itself approaches template status. **Resolution path:** mask only
shingles that also appear in a pre-registered template list or that recur
across *competitions* (persistent KB), not within a single cohort; flagged
as future work rather than silently shipped.

## 4. Trade-off analysis

| Choice | Selected | Alternatives | Reasoning |
| --- | --- | --- | --- |
| Embedding model | all-MiniLM-L6-v2 (384-d, ~90MB) | all-mpnet-base-v2 (768-d, ~420MB), multilingual-e5 | MiniLM is the brief's suggested class, runs realtime on commodity CPU, and its paraphrase retrieval quality is sufficient for the observed margins (0.84-1.0 on true pairs vs <=0.76 topical noise). mpnet buys ~2-3 points of retrieval quality for 4x RAM and latency - wrong trade for a screening tool reviewers re-verify anyway |
| Vector index | FAISS IndexFlatIP (exact) | HNSW, IVF-PQ | At 100k vectors exact search is milliseconds; approximate indexes add recall loss and tuning for no measurable gain until ~1M+ vectors. Migration path in README scaling brief |
| ASR | whisper-tiny int8 (faster-whisper) | whisper-base/small, Vosk | tiny fits the 1 GB container and handles the 10-60s clips the brief scopes; known transcription-quality ceiling documented in 3.2 |
| AI detection | chatgpt-detector-roberta + stylometric corroboration | openai detector (rejected, see 3.1), commercial APIs (out of scope/paid) | Best open-source separation on the corpus; heuristics keep the signal interpretable |
| Similarity metric | coverage of matched chunks | mean similarity, max similarity | Coverage is robust to both single-sentence luck and full-document copying, and maps directly to "how much of this submission is non-original" |
| Template handling | shingle document-frequency masking pre-embedding | post-hoc chunk exclusion only | Post-hoc exclusion leaves template-contaminated *overlap chunks* in play (they caused real false positives in development: original docs sharing a case prompt scored 0.83-0.86); masking removes the template's influence from the embedding itself |

## 5. Reproduce

```bash
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
python tools/evaluate.py
```
