# PhysioRAG — Medical LLM Evaluation Pipeline

Automated benchmark for testing LLMs on medical Q&A using retrieval-augmented generation (RAG).

**Key Features:**
- ✅ Multi-model evaluation across 8B/14B/32B parameter scales
- ✅ Hybrid retrieval (BM25 + FAISS semantic search with cross-encoder reranking)
- ✅ Automatic MCQ generation from medical documents
- ✅ Publication-ready figures and tables (300 DPI PNG, markdown tables)
- ✅ HPC-optimized with vLLM + SLURM job automation
- ✅ Reproducible evaluation metrics

---

## Quick Start (5 minutes)

### 1. **Prerequisites**
```bash
# Local machine
- Python 3.9+
- Conda environment: shoulder-rag
- SSH access to NHR@FAU HPC (tf075, tf041, etc.)
```

### 2. **Pre-download model on HPC login node (one-time)**
```bash
ssh iwso221h@tf075.rrze.uni-erlangen.de

export HF_HOME=$WORK/models/huggingface
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
conda activate /home/hpc/iwso/iwso221h/miniconda3/envs/shoulder-rag

python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-14B')"
```

### 3. **Submit evaluation job**
```bash
cd ~/Shoulder-RAG-HPC

sbatch --partition=a100 --gres=gpu:a100:1 \
       --export=ALL,MODEL_ID=Qwen/Qwen3-14B job_vllm_pipeline.sh
```

### 4. **Monitor job**
```bash
squeue -u "$USER"
tail -f logs/pipeline_vllm_JOBID.log
```

### 5. **Generate report locally (after job completes)**
```bash
# Sync results from HPC
rsync -avz iwso221h@tf075:~/Shoulder-RAG-HPC/Evaluation_sets/ ./Evaluation_sets/

# Generate figures & tables
python generate_report_figures.py

# Outputs → report_outputs/figures/*.png + report_outputs/tables/report_tables.md
```

---

## The Pipeline (5 Phases)

```
PHASE 0: Index Build           → PDF → FAISS index + chunks (one-time)
PHASE 1: Question Generation   → LLM generates 54 MCQs from chunks
PHASE 2: Answer Generation     → RAG retrieves context + LLM answers
PHASE 3: Evaluation            → Score answers, compute metrics
PHASE 4: Report Generation     → Publication-ready figures (local)
```

**Phases 1–3 run automatically on HPC via `job_vllm_pipeline.sh`**  
**Phase 4 runs locally after collecting results**

---

## Key Files

| File | Purpose |
|------|---------|
| `job_vllm_pipeline.sh` | **Main HPC job script** — orchestrates Phases 1–3 |
| `generate_questions.py` | Phase 1: LLM generates MCQs |
| `generate_answers.py` | Phase 2: RAG pipeline answers questions |
| `src/evaluation/eval_pipeline.py` | Phase 3: Computes metrics |
| `generate_report_figures.py` | Phase 4: Creates publication figures (local) |
| `PIPELINE_BREAKDOWN.md` | **Full technical documentation** (read this for details) |

---

## Supported Models

**8B Parameter:**
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-8B`
- `mistralai/Mistral-7B-Instruct-v0.3`

**14B Parameter:**
- `Qwen/Qwen3-14B`

**32B Parameter:**
- `meta-llama/Llama-2-34B` (requires 2× GPU, set `TENSOR_PARALLEL=2`)

---

## Evaluation Metrics

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Retrieval Accuracy** | % of questions where correct chunk in top-8 | Embedding + search quality |
| **Answer-Letter Accuracy** | % of questions where LLM picked correct option | LLM discrimination ability |
| **Answer-Text Accuracy** | % of questions where LLM's answer matched gold text | More lenient (accounts for paraphrasing) |
| **Fully-Correct** | % of questions with BOTH retrieval AND answer correct | End-to-end system success |

---

## Example Results

```
Model: Qwen-14B (54 questions across 6 medical documents)
├── Retrieval Accuracy:    87.5%
├── Answer-Letter Accuracy: 82.1%
├── Answer-Text Accuracy:   79.6%
└── Fully-Correct Rate:     74.1%

Per-Document:
├── Therapeutic Exercise:    90% fully correct (11/12 questions)
├── Adhesive Capsulitis:     75% fully correct (3/4 questions)
└── Rotator Cuff CPG:        60% fully correct (3/5 questions)
```

---

## Advanced Usage

### Generate only questions (no answers)
```bash
sbatch --partition=a100 --gres=gpu:a100:1 \
       --export=ALL,MODEL_ID=Qwen/Qwen3-14B,QUESTION_COUNT=100 job_vllm_pipeline.sh
```

### Use 2 GPUs for larger model
```bash
sbatch --partition=a100 --gres=gpu:a100:2 \
       --export=ALL,MODEL_ID=Qwen/Qwen3-14B,TENSOR_PARALLEL=2 job_vllm_pipeline.sh
```

### Consolidate multiple model results into best evaluation set
```bash
python src/evaluation/question_quality_filter.py \
    --input_dir Evaluation_sets/ \
    --output data/gold_standard/eval_combined_clean.json \
    --target_per_doc 9
```
This filters out low-quality questions and creates a robust benchmark by selecting the best questions across all models.

---

## Outputs

### Phase 4 Report (Local)
```
report_outputs/
├── figures/
│   ├── 01_overall_metrics.png              (main bar chart)
│   ├── 02_retrieval_accuracy_heatmap.png
│   ├── 02_answer_text_accuracy_heatmap.png
│   ├── 02_fully_correct_heatmap.png
│   └── 03_model_comparison_line_appendix.png
└── tables/
    └── report_tables.md                    (3 markdown tables for thesis)
```

**All figures:** 300 DPI PNG, print-ready, with readable axis labels and value annotations

---

## Troubleshooting

### Model not cached
```
ERROR: Model 'Qwen/Qwen3-14B' is NOT cached in HF_HOME
→ Run pre-download step on LOGIN NODE first (see Quick Start, step 2)
```

### vLLM server timeout
```
ERROR: vLLM server did not start within 900s
→ Check logs/vllm_server_JOBID.log
→ Try increasing --time in job script (default: 8 hours)
```

### No eval files found
```
ERROR: No evaluation files found in Evaluation_sets/
→ Sync results from HPC: rsync -avz iwso221h@tf075:~/Shoulder-RAG-HPC/Evaluation_sets/ .
```

---

## Documentation

- **[PIPELINE_BREAKDOWN.md](PIPELINE_BREAKDOWN.md)** — Full technical reference (phases, file structure, infrastructure)
- **[CLI_REFERENCE.md](archive/CLI_REFERENCE.md)** — Command-line options reference

---

## Project Structure

```
PhysioRAG/
├── job_vllm_pipeline.sh              ← Main HPC orchestrator
├── generate_questions.py             ← Phase 1
├── generate_answers.py               ← Phase 2
├── generate_report_figures.py        ← Phase 4
├── src/
│   ├── evaluation/
│   │   ├── eval_pipeline.py         ← Phase 3
│   │   └── question_quality_filter.py ← Quality filtering
│   ├── rag/                          ← RAG components
│   ├── indexing/                     ← Embeddings + FAISS
│   └── parser/                       ← PDF parsing
└── data/
    ├── raw/                          ← Source PDFs
    ├── chunks/                       ← Processed chunks + index
    └── gold_standard/                ← Generated questions
```

## Support

- **Supervisor questions:** See PIPELINE_BREAKDOWN.md → Supervisor Summary section
- **Technical issues:** Check logs/ directory and debug_report.py
- **HPC documentation:** https://hpc.fau.de/

---




