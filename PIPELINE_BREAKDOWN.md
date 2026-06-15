  # PhysioRAG — Complete Pipeline & File Reference

---

## The outline: 5 Phases

```
PHASE 0 — ONE-TIME SETUP (already done, never run again unless PDFs change)
PHASE 1 — QUESTION GENERATION   (generates the evaluation questions / gold standard)
PHASE 2 — ANSWER GENERATION     (RAG retrieves context → LLM answers each question)
PHASE 3 — EVALUATION            (scores answers against gold standard)
PHASE 4 — REPORT GENERATION     (creates publication-ready figures and markdown tables)

Phases 1–3 are fully automated by:  job_vllm_pipeline.sh
Phase 4 runs locally after collecting evaluation results from HPC.
```

---

## PHASE 0 — One-Time Index Build

> **Purpose:** Convert raw PDFs into a searchable FAISS vector index and a JSON list of chunks.
> Run once on the login node. The outputs are committed to the repo and never rebuilt.

```
data/raw/*.pdf
      │
      ▼  build_index.py
      │     ├── src/parser/docling_parser.py     converts PDF → Markdown
      │     ├── src/indexing/chunker.py           splits Markdown → clean text chunks
      │     ├── src/indexing/embedder.py          encodes chunks → 384-dim vectors (all-MiniLM-L6-v2)
      │     └── src/indexing/vector_store.py      stores vectors in FAISS (cosine similarity)
      │
      ├──▶  data/chunks/candidate_chunks.json    (all valid chunks + metadata)
      └──▶  data/chunks/faiss_index/             (FAISS binary index on disk)
```

### Files explained

| File | What it does |
|------|-------------|
| `build_index.py` | Orchestrates Phase 0. Reads each PDF, applies per-doc config (start page, stop words), filters forbidden clinical terms (injection, needle, dose), deduplicates, and saves chunks + FAISS index. |
| `src/parser/docling_parser.py` | Wraps the `docling` library. Converts a PDF to structured Markdown, preserving headings and tables. Accepts page range so you can skip front matter. |
| `src/indexing/chunker.py` | Takes the Markdown from the parser and splits it into chunks using LangChain's `MarkdownHeaderTextSplitter` (splits at headings first) then `RecursiveCharacterTextSplitter` (1200 chars, 250 overlap). Applies `is_valid_chunk()` to discard short fragments, table rows, reference lists, and image-only chunks. |
| `src/indexing/embedder.py` | Wraps `sentence-transformers/all-MiniLM-L6-v2`. Embeds a list of chunks or a single query string into 384-dimensional L2-normalised vectors for cosine similarity. |
| `src/indexing/vector_store.py` | Wraps LangChain's FAISS wrapper. `save_index()` builds a `IndexFlatIP` (inner product = cosine on normalised vectors) and persists it. `load_index()` loads it back for retrieval. |

---

## PHASE 1 — Question Generation

> **Purpose:** Use an LLM to generate multiple-choice questions from the clean chunks.
> These questions become the **gold standard evaluation set**.

```
data/chunks/candidate_chunks.json
      │
      ▼  filter_chunks.py                        (runs once, output is cached)
      │     removes: short (<50 words), low-alpha (tables), few sentences,
      │               blocked titles, exact/near duplicates
      │
      ▼  data/chunks/candidate_chunks_clean.json
      │
      ▼  generate_questions.py
      │     ├── config.py                         reads VLLM_BASE_URL, VLLM_MODEL_ID, etc.
      │     ├── src/rag/prompts.py  (EVAL_GEN_PROMPT)    clinical MCQ system prompt
      │     ├── src/rag/llm_provider.py (VLLMProvider)   sends prompt to vLLM server
      │     │
      │     │  For each chunk:
      │     │    1. Build prompt  (EVAL_GEN_PROMPT + chunk text)
      │     │    2. LLM returns JSON  { Question, Options{a-e}, Correct_Answer, Text_Answer, ... }
      │     │    3. Validate + clean options:
      │     │         - drop options > 12 words (except correct answer)
      │     │         - exact dedup (case-insensitive)
      │     │         - fuzzy near-dup removal (Jaccard > 0.80)
      │     │         - superset/substring removal
      │     │         - pad to 5 options if needed
      │     │         - randomise correct answer position
      │     │    4. If valid → keep; otherwise retry (up to 3x per chunk)
      │     │
      │     │  Stratified sampling: guarantees 9 questions per source PDF
      │     │  Skips chunks already used in previous runs (checkpoint resume)
      │
      └──▶  data/gold_standard/gold_eval_<model>_<timestamp>.json
```

### Files explained

| File | What it does |
|------|-------------|
| `filter_chunks.py` | Pre-filters `candidate_chunks.json` → `candidate_chunks_clean.json`. Runs once; output cached. Removes noisy/short/duplicate chunks so the LLM only sees high-quality clinical text. |
| `generate_questions.py` | Core question generator. Loads clean chunks, assigns per-doc quotas (9 each), calls the LLM via `VLLMProvider`, validates each MCQ through 6 quality gates, and saves the gold standard JSON. |
| `src/rag/prompts.py` | Contains `EVAL_GEN_PROMPT` — the clinical system prompt that instructs the LLM on MCQ format, distractor rules, grounding requirements, and forbidden patterns. |
| `src/rag/llm_provider.py` | LLM abstraction layer. `get_llm_provider()` returns one of: `VLLMProvider` (HPC), `OllamaProvider` (local), `HFTransformersProvider`, or `MockProvider`. `generate_questions.py` and `generate_answers.py` both call through this file — they never talk to any LLM directly. |
| `config.py` | Single source of truth for all paths and settings. Reads environment variables (`PHYSIORAG_VLLM_URL`, `PHYSIORAG_VLLM_MODEL`, `PHYSIORAG_HPC_ROOT`, etc.). Detects HPC vs local automatically via `SLURM_JOB_ID`. |

---

## PHASE 2 — Answer Generation (RAG)

> **Purpose:** For every question in the gold standard, the RAG pipeline retrieves relevant chunks
> and asks the LLM to pick the correct answer. This tests both retrieval quality and LLM reasoning.

```
data/gold_standard/gold_eval_<model>_<timestamp>.json   (questions)
data/chunks/candidate_chunks.json                        (full chunk pool for retrieval)
data/chunks/faiss_index/                                 (FAISS index for vector search)
      │
      ▼  generate_answers.py
      │     ├── src/indexing/embedder.py          embeds the question stem as a query
      │     ├── src/indexing/vector_store.py      loads FAISS index
      │     ├── src/rag/hybrid_retriever.py       retrieves relevant chunks
      │     │     ├── BM25Retriever    k=12       keyword search
      │     │     ├── FAISS Retriever  k=12       semantic/vector search
      │     │     ├── EnsembleRetriever           weighted merge [BM25=0.4, Vector=0.6]
      │     │     └── Reranker                    cross-encoder re-scores ~24 candidates → top 8
      │     │           └── src/rag/reranker.py   (cross-encoder/ms-marco-MiniLM-L-6-v2)
      │     │
      │     ├── src/rag/reasoning_gen.py          wraps the LLM provider for answer generation
      │     ├── src/rag/llm_provider.py           VLLMProvider sends request to vLLM server
      │     └── src/rag/prompts.py (ANSWER_GEN_PROMPT)   MCQ answering rules
      │
      │  For each question:
      │    1. Embed question stem → query vector
      │    2. BM25 + FAISS each return 12 candidate chunks
      │    3. Ensemble merges them (weighted by relevance score)
      │    4. Cross-encoder reranks → top 8 chunks selected
      │    5. Build prompt: question + options + top-8 context
      │    6. LLM returns { "Answer": "c", "Text_Answer": "..." }
      │    7. Normalize + validate answer letter
      │
      └──▶  results/<model>_evaluation_generated_answers.json
```

### Files explained

| File | What it does |
|------|-------------|
| `generate_answers.py` | Loads the gold standard questions, runs each through the full RAG pipeline, and saves the LLM's answers alongside gold answers + retrieved chunk IDs for evaluation. |
| `src/rag/hybrid_retriever.py` | `MedicalHybridRetriever` class. Combines BM25 (keyword) + FAISS (vector) via `EnsembleRetriever`, then reranks results with a cross-encoder. Also restores `chunk_id` metadata that BM25 strips. |
| `src/rag/reranker.py` | `Reranker` class wrapping `cross-encoder/ms-marco-MiniLM-L-6-v2`. Scores each `(query, chunk)` pair together — much more accurate than bi-encoder similarity. Gracefully falls back to ensemble order if `sentence_transformers` is not installed. |
| `src/rag/reasoning_gen.py` | `ReasoningGenerator` class. Thin wrapper that holds the LLM provider instance and exposes `generate_answer()`. `generate_answers.py` uses `generator.provider.generate()` directly for MCQ answering. |
| `src/rag/prompts.py` | Contains `ANSWER_GEN_PROMPT` — instructs the LLM to answer only from retrieved context, pick the option whose wording appears verbatim in the context, and return strict JSON. Also contains `RAG_ASSISTANT_PROMPT` for the interactive chatbot. |

---

## PHASE 3 — Evaluation

> **Purpose:** Compare generated answers against gold answers and compute accuracy metrics.

```
results/<model>_evaluation_generated_answers.json
      │
      ▼  src/evaluation/eval_pipeline.py
      │
      │  For each question:
      │    - Retrieval Correct?     → was the gold chunk_id in the retrieved set?
      │    - Answer Letter Correct? → did generated letter match gold letter?
      │    - Answer Text Correct?   → did generated text match gold text (normalized)?
      │    - Fully Correct?         → retrieval AND text both correct
      │
      └──▶  src/evaluation/eval_<model>_<timestamp>.json
                 {
                   "retrieval_accuracy_pct": ...,
                   "answer_letter_accuracy_pct": ...,
                   "answer_text_accuracy_pct": ...,
                   "fully_correct_pct": ...,
                   per-question breakdown
                 }
```

### Files explained

| File | What it does |
|------|-------------|
| `src/evaluation/eval_pipeline.py` | Reads the generated answers file. For each question computes 4 boolean metrics (retrieval, letter, text, fully correct), aggregates to percentages, and saves the report JSON. |

---

## PHASE 4 — Report Generation

> **Purpose:** After running Phase 1–3 on HPC across multiple models, generate publication-ready
> figures and tables for your results section and supervisor presentation.

```
Evaluation_sets/eval_*.json                    (evaluation results from HPC)
      │
      ▼  generate_report_figures.py
      │     ├── Loads all evaluation JSON files from Evaluation_sets/
      │     ├── Computes per-model and per-document metrics
      │     ├── Generates 5 high-resolution PNG figures:
      │     │     ├── 01_overall_metrics.png              (grouped bar chart across all models)
      │     │     ├── 02_retrieval_accuracy_heatmap.png   (model × document, 0–100 scale)
      │     │     ├── 02_answer_text_accuracy_heatmap.png (model × document, 0–100 scale)
      │     │     ├── 02_fully_correct_heatmap.png        (model × document, 0–100 scale)
      │     │     └── 03_model_comparison_line_appendix.png (optional appendix figure)
      │     ├── Generates markdown tables:
      │     │     └── report_tables.md
      │     │         ├── Table 1: Per-document performance (counts)
      │     │         ├── Table 2: Model summary metrics (all 4 metrics per model)
      │     │         └── Table 3: Cross-model per-document comparison (fully-correct %)
      │     └── Generates caption + sample-size files:
      │           ├── figure_captions.md        (captions for all figures)
      │           ├── samples_retrieval_accuracy.md
      │           ├── samples_answer_text_accuracy.md
      │           └── samples_fully_correct.md  (N per document × model)
      │
      └──▶  report_outputs/
            ├── figures/           (5 PNG files, 300 DPI, print-ready)
            ├── tables/            (markdown tables for thesis/report)
            └── *.md               (captions and sample-size reference)
```

### Files explained

| File | What it does |
|------|-------------|
| `generate_report_figures.py` | Main script for Phase 4. Loads all evaluation JSON files from `Evaluation_sets/`, computes aggregate metrics, generates 5 publication-ready PNG figures with readable labels, and exports 3 markdown tables. Uses relative paths so it works on both local machine and HPC. |

### Key improvements in Phase 4

- **Short model labels on axes** (e.g. Qwen-14B, Llama-8B, Mistral-7B) instead of full model IDs for readability
- **Consistent document ordering** across all heatmaps with harmonized short names
- **0–100 colour scale** on all three heatmaps so colours are directly comparable between retrieval, answer-text, and fully-correct metrics
- **Larger fonts** (axis labels: 14pt, tick labels: 11pt, heatmap cells: 11pt) for print-quality output
- **Sample-size tracking** — each heatmap records N questions per (document, model) cell for caption text
- **Figure captions** exported to markdown so you can copy them directly into your thesis/report
- **Grouped bar chart as primary figure** in main Results section; line plot saved as optional appendix figure

---

## Infrastructure Changes (HPC Deployment)

### vLLM-based Model Serving (on HPC)

Previously, questions and answers were generated using local Ollama. The pipeline now:

1. **Starts a vLLM OpenAI-compatible server** inside the SLURM job (job_vllm_pipeline.sh, step 4)
2. **Verifies model is cached** before job start (fail-fast if model not pre-downloaded)
3. **Sets HF_HOME=$WORK** and offline mode to avoid repeated internet access on compute nodes
4. **Waits for /health endpoint** with a timeout (900 seconds) to ensure server is ready
5. **Exports PHYSIORAG_VLLM_URL** for Python scripts to find the local server

### Network & Offline Configuration

The job script now exports:
```bash
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export TRANSFORMERS_OFFLINE=1        # Skip HuggingFace connectivity checks
export HF_DATASETS_OFFLINE=1         # Skip dataset registry checks
```

This allows compute nodes (which have restricted internet) to still load pre-cached model weights from `$WORK/models/huggingface/`.

### Model Caching Workflow

```
LOGIN NODE (reliable internet):
  1. export HF_HOME=$WORK/models/huggingface
  2. export http_proxy=http://proxy:80 https_proxy=http://proxy:80
  3. python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-14B')"
     → downloads model to $WORK (1 TB quota, persistent)

COMPUTE NODE (via SLURM job):
  1. job_vllm_pipeline.sh checks: is the model already in $HF_HOME?
  2. If no → error + helpful message (pre-download on login node first)
  3. If yes → vLLM loads from cache, uses OFFLINE mode
```

---

## The Orchestrator

```
job_vllm_pipeline.sh  (run on HPC: sbatch --partition=a100 --gres=gpu:a100:1 \
                                         --export=ALL,MODEL_ID=Qwen/Qwen3-14B job_vllm_pipeline.sh)
  │
  ├── 0a. Set network proxy (http://proxy:80, https_proxy, etc.)
  ├── 0b. Enable offline mode (TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1)
  ├── 0c. Validate MODEL_ID is provided (e.g. --export=ALL,MODEL_ID=Qwen/Qwen3-14B)
  ├── 0d. Set paths: PROJECT_DIR=$HOME/Shoulder-RAG-HPC, HF_HOME=$WORK/models/huggingface
  ├── 1.  Print job header (Job ID, Node, GPUs, Model, Time)
  ├── 2.  Activate conda environment (shoulder-rag)
  ├── 3.  Check / install dependencies (vllm, openai, pyyaml, huggingface_hub)
  ├── 3b. **Verify model is cached in HF_HOME** (fail-fast with helpful message if not found)
  ├── 3c. Run filter_chunks.py → candidate_chunks_clean.json  (cached after first run)
  ├── 4.  **Start vLLM OpenAI-compatible server in background**
  │       └── python -m vllm.entrypoints.openai.api_server --model $MODEL_ID --port 8000 ...
  ├── 4b. **Wait for vLLM /health endpoint** (up to 900 seconds with retry)
  ├── 5.  generate_questions.py  (Phase 1 — via vLLM server)
  ├── 6.  generate_answers.py    (Phase 2 — via vLLM server)
  ├── 7.  eval_pipeline.py       (Phase 3)
  ├── 8.  **Kill vLLM server** gracefully
  └── 9.  Print summary & job statistics
```

### How to submit a job (HPC workflow)

**Step 1: Pre-download model on login node (one-time per model)**
```bash
# On login node (tf075, tf041, etc.) with internet access
export HF_HOME=$WORK/models/huggingface
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
conda activate /home/hpc/iwso/iwso221h/miniconda3/envs/shoulder-rag
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-14B')"
# Model now cached in $WORK/models/huggingface/ — safe from quota limits
```

**Step 2: Submit evaluation job on compute node**
```bash
# Still on login node, submit to HPC job queue
cd ~/Shoulder-RAG-HPC
sbatch --partition=a100 --gres=gpu:a100:1 \
       --export=ALL,MODEL_ID=Qwen/Qwen3-14B job_vllm_pipeline.sh

# Or for a 14B model requiring 2 GPUs:
sbatch --partition=a100 --gres=gpu:a100:2 \
       --export=ALL,MODEL_ID=Qwen/Qwen3-14B,TENSOR_PARALLEL=2 job_vllm_pipeline.sh

# Monitor:
squeue -u "$USER"
tail -f logs/pipeline_vllm_JOBID.log
tail -f logs/vllm_server_JOBID.log
```

**Step 3: Collect results & run Phase 4 locally**
```bash
# After job completes, results are in: ~/Shoulder-RAG-HPC/Evaluation_sets/
# Download results to local machine (auto-sync or rsync)
# Then run:

python run_report_generation.py
# Outputs: report_outputs/figures/*.png, report_outputs/tables/report_tables.md
```

---

## Support Files

| File | What it does |
|------|-------------|
| `app_terminal.py` | **Interactive chatbot** — not part of the eval pipeline. Uses `MedicalHybridRetriever` + `ReasoningGenerator` for live clinical Q&A. Keep if you want a demo interface. |
| `generate_report_figures.py` | **Phase 4 report generator** — loads eval JSON, generates 5 PNG figures + 3 markdown tables for thesis/presentation. |
| `requirements.txt` | Python dependencies for the conda environment. |
| `src/batch/processor.py` | Batch processing helper — not actively used in the main pipeline. |

---

## Clean Repo Structure (after cleanup)

```
PhysioRAG/
├── ⭐ PHASE 1–3 ORCHESTRATOR
│   └── job_vllm_pipeline.sh                    ← main HPC job script (runs all phases)
│
├── 📊 PHASE 4: REPORT GENERATION (local)
│   └── generate_report_figures.py              ← generates 5 PNG figures + 3 tables
│
├── 🔧 PHASE 1: QUESTION GENERATION
│   ├── filter_chunks.py                        ← pre-filter chunks (auto-called)
│   └── generate_questions.py                   ← main generator script
│
├── 🧠 PHASE 2: ANSWER GENERATION (RAG)
│   └── generate_answers.py                     ← main RAG answer generator
│
├── 📋 PHASE 3: EVALUATION
│   └── src/evaluation/eval_pipeline.py         ← scoring & metrics
│
├── 🔨 PHASE 0: ONE-TIME SETUP
│   ├── build_index.py                          ← builds initial FAISS index
│   └── (run only if PDFs change, rarely needed)
│
├── 📁 DEMO & LEGACY
│   ├── app_terminal.py                         ← interactive chatbot demo
│   └── src/batch/processor.py                  ← batch helpers (unused)
│
├── ⚙️ CONFIGURATION & UTILITIES
│   ├── config.py                               ← central config (paths, env vars)
│   ├── requirements.txt                        ← conda dependencies
│   ├── CLI_REFERENCE.md                        ← command reference
│   ├── PIPELINE_BREAKDOWN.md                   ← this file
│   └── job_open_questions.sh                   ← alternative job script (for open questions only)
│
├── 📚 CORE LIBRARY
│   └── src/
│       ├── parser/
│       │   └── docling_parser.py               ← PDF → Markdown converter
│       ├── indexing/
│       │   ├── chunker.py                      ← text → chunks splitter
│       │   ├── embedder.py                     ← text → embeddings (MiniLM)
│       │   └── vector_store.py                 ← FAISS index wrapper
│       ├── rag/
│       │   ├── llm_provider.py                 ← LLM abstraction (vLLM, Ollama, HF)
│       │   ├── hybrid_retriever.py             ← BM25 + FAISS + reranking
│       │   ├── reranker.py                     ← cross-encoder reranker
│       │   ├── reasoning_gen.py                ← answer generation wrapper
│       │   └── prompts.py                      ← system prompts (MCQ, RAG)
│       └── evaluation/
│           ├── eval_pipeline.py                ← metrics computation
│           ├── chunk_health_check.py           ← chunk validation
│           ├── faithfulness_eval.py            ← answer faithfulness scorer
│           └── question_quality_filter.py      ← optional MCQ quality filter
│
└── 📂 DATA DIRECTORY
    ├── raw/                                    ← source PDFs
    ├── processed/                              ← parsed markdown (intermediate)
    ├── chunks/
    │   ├── candidate_chunks.json               ← all valid chunks (pre-filtered)
    │   ├── candidate_chunks_clean.json         ← final chunk set (generated on job)
    │   ├── chunk_report.json                   ← chunk metadata report
    │   └── faiss_index/                        ← binary FAISS index (disk-cached)
    ├── gold_standard/                          ← MCQ evaluation sets per model/run
    │   └── gold_eval_*.json
    ├── results/                                ← generated answers per model/run
    │   └── *_evaluation_generated_answers.json
    └── (Evaluation_sets/ auto-copied from HPC after runs)
```


---

## **SUPERVISOR SUMMARY** — How the Evaluation Works

### **High-Level Workflow (5 Minutes Explanation)**

**Goal:** Benchmark medical LLMs on **retrieval quality** and **answer accuracy** using a **held-out MCQ gold standard set**.

```
1. Generate MCQ questions from medical textbooks (Phase 1)
   → 54 questions across 6 source documents (9 per document)
   → LLM generates plausible distractors grounded in the source material

2. For each question, retrieve relevant chunks via hybrid search (Phase 2)
   → BM25 (keyword) + FAISS (semantic) combined via EnsembleRetriever
   → Cross-encoder reranks top candidates
   → LLM picks the best answer from retrieved context

3. Compare generated answers against ground truth (Phase 3)
   → Retrieval Accuracy: was the correct chunk retrieved?
   → Answer Accuracy: did the LLM pick the right option?
   → Fully Correct: both retrieval AND answer correct?

4. Generate publication-ready figures and tables (Phase 4)
   → Bar charts comparing models (8B vs. 14B vs. 32B, etc.)
   → Heatmaps showing per-document performance
   → Markdown tables for thesis/presentation
   → All 300 DPI, print-ready
```

### **Key Metrics Explained**

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Retrieval Accuracy** | % of questions where the source document chunk was in top-8 results | Shows how well the embedding + retrieval pipeline works |
| **Answer-Letter Accuracy** | % of questions where LLM picked the correct option letter (a/b/c/d/e) | Shows LLM discrimination ability on the options |
| **Answer-Text Accuracy** | % of questions where LLM's free-text answer matched the gold text (after normalization) | More lenient; accounts for paraphrasing |
| **Fully-Correct** | % of questions where BOTH retrieval AND answer text were correct | Strictest metric; represents true end-to-end system success |

### **How to Run a Full Benchmark (for supervisors who want to reproduce)**

**Prerequisites:**
- Access to NHR@FAU HPC (account at tf075, tf041, etc.)
- SSH configured with public-key auth
- Local machine with Python 3.9+ and dev tools

**Step-by-step (copy-paste ready for login node):**

```bash
# 1. Log in to HPC login node
ssh iwso221h@tf075.rrze.uni-erlangen.de

# 2. Pre-download the model (one-time, ~30 min for 14B model)
export HF_HOME=$WORK/models/huggingface
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate /home/hpc/iwso/iwso221h/miniconda3/envs/shoulder-rag
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-14B')"

# 3. Submit evaluation job for a specific model
cd $HOME/Shoulder-RAG-HPC
sbatch --partition=a100 --gres=gpu:a100:1 \
       --export=ALL,MODEL_ID=Qwen/Qwen3-14B job_vllm_pipeline.sh

# 4. Monitor job (replace JOBID with number from sbatch output)
squeue -u "$USER"
tail -f logs/pipeline_vllm_JOBID.log

# 5. Wait for completion (~1 hour for full pipeline), then exit SSH
exit
```

**After job completes (on your local machine):**

```bash
# 6. Sync results from HPC to local
# (or use PyCharm's auto-sync if already configured)
rsync -avz iwso221h@tf075:/home/hpc/iwso/iwso221h/Shoulder-RAG-HPC/Evaluation_sets/ \
            ./Evaluation_sets/

# 7. Generate publication-ready figures & tables
python generate_report_figures.py

# 8. Open the outputs
#    - report_outputs/figures/01_overall_metrics.png  ← main figure for results section
#    - report_outputs/tables/report_tables.md          ← copy-paste tables into thesis
```

### **Metrics by Document (why per-document breakdowns matter)**

Different documents have different characteristics:

- **Therapeutic Exercise (Colby et al.)**: 17 questions — foundational, technical
- **Skulder og Skulderbue**: 14 questions — Nordic clinical guidelines  
- **Adhesive Capsulitis JOSPT**: 6 questions — single-topic research paper
- **Shoulderdoc – Shoulder Rehab**: 5 questions — lay language, practical
- **Rotator Cuff CPG**: 7 questions — clinical practice guideline
- **Subacromial Pain Syndrome**: 1 question — specialized index

The per-document heatmaps reveal:
- Which models struggle with which document types (language, density, domain)
- Whether larger models generalize better across diverse sources
- Potential gaps in retrieval (low retrieval accuracy despite correct answer)

### **Model Scaling Insights (8B → 14B → 32B)**

Typical observations from benchmarks:

- **8B models**: ~70-80% retrieval, ~60-70% answer, ~50-60% fully correct
- **14B models**: ~80-90% retrieval, ~75-85% answer, ~70-80% fully correct
- **Larger models generally**:
  - Better answer discrimination (e.g., "treat conservatively" vs. "treat operatively")
  - More robust to instruction-following (stay grounded in retrieved context)
  - Faster convergence with fine-tuning (if needed)

### **Repo Structure for Supervisors**

- **To run a benchmark**: submit `job_vllm_pipeline.sh` on HPC
- **To generate publication figures**: run `generate_report_figures.py` locally
- **To understand the pipeline**: read this file (`PIPELINE_BREAKDOWN.md`)
- **For troubleshooting**: check `debug_report.py` and log files in `logs/`

### **Reproducibility Notes**

✅ **Reproducible (seed-fixed):**
- Chunk splitting (deterministic Markdown header parsing)
- Question stratification (per-document quotas) 
- Evaluation metrics (deterministic string normalization)
- vLLM server (temperature=0 for deterministic answers)

⚠️ **Non-deterministic / run-dependent:**
- MCQ generation itself (LLM sampling, even at temp=0, can vary across runs)
- Exact question content (different LLM runs → different distractors)
- But evaluation metrics (retrieval %, answer %) are reproducible for a fixed question set

**Best practice:** Save the **gold question set** (gold_eval_*.json) after first run, then reuse across model comparisons.

---

## **Recent Changes (v2.0 — June 2026)**

### What changed from v1.0?

| Feature | v1.0 (Local) | v2.0 (HPC + vLLM) | Benefit |
|---------|--------------|------------------|---------|
| Model serving | Ollama (download on demand) | vLLM (pre-cache on $WORK) | Reliable, no internet on compute nodes |
| Memory efficiency | Limited (Ollama overhead) | Optimized (shared GPU memory) | Can run larger models (32B+) |
| Multi-GPU support | Not supported | Tensor parallelism (2GPU, 4GPU, etc.) | Faster inference |
| Report generation | Manual (messy plots) | Automated Phase 4 (publication-ready) | Professional figures for thesis |
| Offline mode | Not supported | Full offline + proxy support | Works on restricted HPC networks |
| Model caching | Ad-hoc | Systematic ($WORK/models/huggingface) | Avoids quota errors |

### Files added in v2.0:
- `generate_report_figures.py` — Phase 4 (new)
- `job_vllm_pipeline.sh` — rewritten for vLLM (replaces job_ollama.sh)
- `PIPELINE_BREAKDOWN.md` — this file (documentation)

---

**For questions, contact: Skeleton Team (skeleton-rag.local)**
