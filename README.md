# Multimodal Submission Integrity & Plagiarism Detection Engine

A Python pipeline that ingests mixed-format competition submissions (PDF
reports, DOCX summaries, MP4 video pitches), indexes their textual content
as dense vectors, and produces **explainable integrity audit reports**:
every similarity claim is backed by the exact matching passages with page
numbers, video timestamps, and cosine scores so a human reviewer can
cross-verify any flag in under 30 seconds.

## Quickstart

```bash
# 1. System dependency: ffmpeg (audio demuxing for MP4)
sudo apt install ffmpeg            # or: brew install ffmpeg

# 2. Python 3.9+ environment
pip install -r requirements.txt

# 3. Run the full pipeline over the sample corpus
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```

Reports land in `./reports/`: one `.json` + one `.md` per submission, plus
`summary.md` / `summary.json` with the batch table and pairwise similarity
matrix. First run downloads ~700MB of open models (MiniLM, whisper-tiny,
chatgpt-detector-roberta); subsequent runs use the local cache. The full
13-file sample batch takes ~30s on a 1-core CPU container.

Regenerate the sample corpus (including the synthetic-speech MP4s) with:

```bash
python tools/make_sample_data.py
```

Reproduce every metric quoted in `EVALUATION.md` with:

```bash
python tools/evaluate.py
```

## What the pipeline does (per brief task)

### Task 1 - Multimodal ingestion and normalization
`integrity_engine/extractors/` contains one extractor per format:

| Format | Library | Provenance captured |
| --- | --- | --- |
| PDF | PyMuPDF (block mode) | page number per block |
| DOCX | python-docx | paragraph index (DOCX reflows, so pages are not meaningful) |
| MP4 | ffmpeg + faster-whisper (`whisper-tiny`, int8) | timestamp range per transcript segment |

Normalization: whitespace collapsing, plus **header/footer stripping** for
PDFs (lines repeated at the top/bottom of >=60% of pages, page-number
insensitive). Malformed inputs never crash the run: zero-byte files,
undecodable PDFs, silent videos, and audio tracks with no speech are
flagged with an informative warning and skipped (see Scenario 3 below).

Chunking (`chunking.py`): word-window packing at ~220 words (roughly
200-400 tokens) with a 50-word overlap. Chunks prefer closing at natural
atom boundaries (paragraph / block / transcript segment) so self-contained
units like a shared prompt stay isolatable. Every chunk inherits the page /
paragraph / timestamp span of the atoms it covers.

### Task 2 - Indexing and peer-to-peer similarity
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  L2-normalized, so inner product = cosine similarity).
- Index: FAISS `IndexFlatIP` (exact search - the right choice at prototype
  scale; see the scaling brief for the 10k-document path).
- **Self-match exclusion** is by `doc_id` at query time: a chunk can never
  match its own document.
- Scoring: a document's *peer similarity percentage* is its matched-chunk
  **coverage** - the share of its substantive chunks whose best cross-document
  cosine clears 0.75. Coverage is deliberately conservative in both
  directions: one lucky sentence cannot brand a document, and a fully copied
  document cannot hide behind one original paragraph.

### Task 3 - Multi-signal integrity detection
- **AI-content estimation**: `Hello-SimpleAI/chatgpt-detector-roberta`
  (RoBERTa trained on the HC3 corpus of ChatGPT vs human answers), run in an
  isolated subprocess (its ~0.9GB peak RSS never overlaps the other models).
  The score is a word-weighted mean over <=350-word windows. Stylometric
  heuristics (burstiness, type-token ratio) are always computed and included
  in the report as corroboration, and serve as a documented fallback if the
  detector cannot load. Detector selection was itself an evaluation finding -
  see `EVALUATION.md` for why the GPT-2-era detector was rejected.
- **External reference similarity**: a local reference knowledge base
  (`reference_kb/`, Wikipedia-style articles) stands in for web search, as
  the brief permits. Matching runs at **sentence-window granularity**
  (3-sentence windows, stride 2) so the report quotes the exact copied
  passage instead of a diluted 200-word neighbourhood. In production this
  module is the swap point for a real search API; the matching mechanics are
  identical.

### Task 4 - Citation and explainability
Every match row contains: the offending passage from the query document, the
source passage from the matched document or reference, both locations
(page / paragraph range / video timestamp), and the cosine score. Reports
are per-document JSON (machine-readable, schema follows the brief) and
Markdown (reviewer-readable), plus a batch summary with a pairwise matrix.

**Template suppression (Scenario 4):** before embedding, the pipeline builds
word 8-gram (shingle) document frequencies across the corpus. Shingles
appearing in >=3 distinct submissions are institutional boilerplate (shared
case prompts, standard instructions) and are masked out of the embedding
text. Verbatim copying between *two* submissions keeps df=2 and is never
masked. Chunks that are almost entirely template are excluded from scoring
and listed separately in the report, so shared headers cannot create false
matches - the prototype's original cosine-based pass produced exactly those
false positives, and the fix is documented in `EVALUATION.md`.

### Task 5 - Scaling to 10,000 documents (architecture brief)

| Concern | Prototype | Production design |
| --- | --- | --- |
| Orchestration | single CLI process, sequential stages | task queue (Celery/SQS + worker pool); one job per submission, fanned out at upload time; extraction, embedding, and scoring as separate worker classes |
| ASR (MP4) | whisper-tiny, CPU, per-file | whisper-base on GPU workers (10-20x realtime); batch audio decoding; store transcripts in the document store so re-scoring never re-transcribes |
| Embeddings | MiniLM on 1 CPU core | batched GPU inference (onnx/TensorRT), ~10k chunks/min on a single T4; CPU-only fallback with ONNX-int8 MiniLM at ~2k chunks/min |
| Vector index | FAISS `IndexFlatIP`, exact | **HNSW** (`IndexHNSWFlat`, M=32) for approximate search: at 10k docs x ~10 chunks = 100k vectors x 384 dims, Flat search is still only ~150 MB and milliseconds per query, so Flat honestly remains viable; HNSW becomes necessary past ~1M vectors or when per-query latency budget is <1ms at high QPS. IVF-PQ if RAM is constrained; PQ costs recall and needs a training pass |
| Peer matching cost | O(chunks^2) comparisons via index queries | same asymptotics but sharded; new submissions only query the existing index (incremental), so ingestion cost is linear in new content |
| Storage | in-memory | vectors in a managed vector DB (pgvector / Qdrant / Milvus) with metadata payloads; raw files in object storage; reports precomputed to Postgres |
| External matching | local KB | search API with per-document caching, query sampling (top-n distinctive shingles), and async rate-limit-aware workers |
| Failure handling | warnings in the report | dead-letter queue + per-file retry; malformed files quarantined with the same report contract |
| Cost / throughput | 13 files in ~30s on 1 core | ~10k submissions (mixed, ~10% video) in well under an hour on 1 GPU + 8 CPU workers; dominant cost is ASR minutes, so clips are the lever |
| Latency bottleneck order | - | ASR >> embedding > detector > index search |

## Assumptions (documented per the brief)

1. **External similarity is simulated** with a local reference KB, as
   explicitly allowed; the interface mirrors what a search-API-backed
   implementation would return.
2. **DOCX locations are paragraph indices** because DOCX pagination is
   renderer-dependent.
3. **Word count proxies token count** for chunking (~1.3 tokens/word keeps
   chunks inside the suggested 200-400 token band).
4. **Risk bands are thresholds, not verdicts**: HIGH means "review first",
   and the report always shows the underlying passages. Bands are tunable in
   `integrity_engine/config.py`.
5. **AI-content probability** is an indicator calibrated on ChatGPT-style
   prose; modern heavily-edited LLM text can evade open detectors (see
   failure analysis). It is reported as a signal alongside corroborating
   stylometric heuristics, never as sole grounds for a HIGH verdict on its
   own at the MEDIUM band boundary.
6. **Sample MP4 speech is TTS-synthesized** (gTTS) so the corpus contains no
   copyrighted or private recordings; the ASR path is identical for real
   speech.

## AI-assistant disclosure

I used AI coding assistants (ChatGPT / GitHub Copilot) for boilerplate,
debugging, and documentation drafting, as permitted by the brief. All
architectural decisions (coverage scoring, template suppression via shingle
document frequency, sentence-window citation matching, detector selection)
are my own, are explained in this README and `EVALUATION.md`, and I can walk
through every line in the follow-up interview. The LLM-styled essay in the
sample corpus is synthetic and disclosed in
`sample_submissions/dataset_manifest.json`.

## Repository layout

```
run_integrity_check.py     # CLI entry point (the single root command)
integrity_engine/          # the pipeline package
  config.py                # every threshold and model name, in one place
  extractors/              # pdf / docx / mp4 extraction with provenance
  chunking.py              # word-window chunks + sentence windows
  embeddings.py            # MiniLM wrapper
  indexer.py               # FAISS index
  similarity.py            # peer matching, coverage, template suppression
  external.py              # reference KB loading
  ai_detection.py          # detector subprocess + heuristics
  scoring.py               # risk bands + explanation text
  report.py                # JSON / Markdown writers
sample_submissions/        # 13-file synthetic corpus + ground-truth manifest
reference_kb/              # local stand-in for web/reference search
tools/make_sample_data.py  # regenerates the corpus
tools/evaluate.py          # recomputes every EVALUATION.md metric
reports/                   # pipeline output (gitignored)
```
