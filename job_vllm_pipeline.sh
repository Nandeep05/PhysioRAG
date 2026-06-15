#!/bin/bash
# =============================================================================
# SLURM Job: Full PhysioRAG pipeline with vLLM
#   1. Start vLLM server
#   2. Generate evaluation questions
#   3. Generate answers via RAG
#   4. Run evaluation and print report
# =============================================================================
# Submit (always specify partition, GPU type, and MODEL_ID):
#
#   sbatch --partition=a100 --gres=gpu:a100:1 \
#          --export=ALL,MODEL_ID=Qwen/Qwen3-8B job_vllm_pipeline.sh
#
#   sbatch --partition=a100 --gres=gpu:a100:1 \
#          --export=ALL,MODEL_ID=meta-llama/Llama-3.1-8B-Instruct job_vllm_pipeline.sh
#
#   sbatch --partition=a100 --gres=gpu:a100:2 \
#          --export=ALL,MODEL_ID=Qwen/Qwen3-14B,TENSOR_PARALLEL=2 job_vllm_pipeline.sh
#
# Monitor (replace JOBID with the number printed after sbatch, NO spaces):
#   squeue -u "$USER"
#   tail -f logs/pipeline_vllm_JOBID.log
#   tail -f logs/vllm_server_JOBID.log
#
# Cancel:
#   scancel JOBID
# =============================================================================

#SBATCH --job-name=physiorag-pipeline
# NOTE: --partition and --gres are NOT set here on purpose.
# Always specify them on the command line at submit time.
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/pipeline_vllm_%j.log
#SBATCH --error=logs/pipeline_vllm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nandeep.somashekar@fau.de

set -euo pipefail

# ===========================================================================
# 0a. Network proxy  (required on NHR@FAU compute nodes for internet access)
# ===========================================================================
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

# Use cached model weights — skip HuggingFace connectivity checks at startup.
# The model must already be downloaded to HF_HOME before submitting this job.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ===========================================================================
# 0b. Validate MODEL_ID is provided
# ===========================================================================
if [ -z "${MODEL_ID:-}" ]; then
    echo "ERROR: MODEL_ID not set."
    echo "       Submit with:  sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-8B ..."
    exit 1
fi

# ===========================================================================
# 0c. Configuration  (all overridable via --export at sbatch time)
# ===========================================================================
PROJECT_DIR="${PROJECT_DIR:-$HOME/Shoulder-RAG-HPC}"
CONDA_ENV="${CONDA_ENV:-/home/hpc/iwso/iwso221h/miniconda3/envs/shoulder-rag}"

export PHYSIORAG_HPC_ROOT="$PROJECT_DIR"

# Store HuggingFace models on $WORK (1 TB quota) — NOT $HOME (105 GB quota)
export HF_HOME="${WORK}/models/huggingface"
mkdir -p "$HF_HOME"

TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
QUESTION_COUNT="${QUESTION_COUNT:-54}"   # 9 per document × 6 documents

GOLD_DIR="${PROJECT_DIR}/data/gold_standard"
RESULTS_DIR="${PROJECT_DIR}/results"
EVAL_DIR="${PROJECT_DIR}/Evaluation_sets"

VLLM_PORT=8000
VLLM_URL="http://localhost:${VLLM_PORT}"
MODEL_SHORT="$(basename $MODEL_ID)"


# ===========================================================================
# 1.  Print job header
# ===========================================================================
echo "============================================================"
echo "  PhysioRAG – Full vLLM Pipeline"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  GPUs        : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Model       : $MODEL_ID"
echo "  TP size     : $TENSOR_PARALLEL"
echo "  Questions   : $QUESTION_COUNT"
echo "  HF_HOME     : $HF_HOME"
echo "  Project dir : $PROJECT_DIR"
echo "  Started     : $(date)"
echo "============================================================"

# ===========================================================================
# 2.  Activate conda environment
# ===========================================================================
# Always use your own conda installation/env to avoid mixed module/user-site packages.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found; cannot activate $CONDA_ENV"
    exit 1
fi

conda activate "$CONDA_ENV"

# Prevent imports from ~/.local/lib/pythonX.Y/site-packages (caused the missing yaml issue).
export PYTHONNOUSERSITE=1

python --version
echo "Conda env  : $CONDA_DEFAULT_ENV"
echo "Python exe : $(which python)"

cd "$PROJECT_DIR"
mkdir -p logs "$GOLD_DIR" "$RESULTS_DIR" "$EVAL_DIR"
# ===========================================================================
# 3.  Install vLLM / openai if missing
# ===========================================================================
echo ""
echo ">>> [Step 0] Checking dependencies..."
if ! python -c "import vllm" 2>/dev/null; then
    echo "    Installing vLLM (first run – this may take ~5 min)..."
    python -m pip install vllm openai pyyaml huggingface_hub --quiet
    echo "    vLLM installed."
else
    echo "    vLLM OK."
fi

if ! python -c "import openai" 2>/dev/null; then
    python -m pip install openai --quiet
fi

if ! python -c "import yaml" 2>/dev/null; then
    echo "    Installing missing PyYAML..."
    python -m pip install pyyaml --quiet
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "    Installing/repairing huggingface_hub..."
    python -m pip install huggingface_hub --quiet
fi

# ===========================================================================
# 3b. Verify model is already cached (compute nodes have unreliable internet)
# ===========================================================================
echo ""
echo ">>> [Pre-check] Verifying model is cached in HF_HOME=$HF_HOME ..."
python - <<'PYCHECK'
import os, sys
from huggingface_hub import try_to_load_from_cache
model_id = os.environ["MODEL_ID"]
cfg = try_to_load_from_cache(model_id, "config.json")
if not isinstance(cfg, str) or not os.path.isfile(cfg):
    hf_home = os.environ.get("HF_HOME", "(unset)")
    print(f"ERROR: Model '{model_id}' is NOT cached in HF_HOME={hf_home}")
    print("       Pre-download from the LOGIN NODE first:")
    print(f"         export HF_HOME=$WORK/models/huggingface")
    print(f"         export http_proxy=http://proxy:80 https_proxy=http://proxy:80")
    print(f"         conda activate /home/hpc/iwso/iwso221h/miniconda3/envs/shoulder-rag")
    print(f'         python -c "from huggingface_hub import snapshot_download; snapshot_download(\'{model_id}\')"')
    sys.exit(1)
print(f"    Model cached OK: {cfg}")
PYCHECK
echo "    Model cache check passed."

# ===========================================================================
# 3c. Pre-filter chunks  (runs once; output is cached for all future jobs)
#     Produces candidate_chunks_clean.json from candidate_chunks.json
#     Conda is active here so python uses the correct environment.
# ===========================================================================
CLEAN_CHUNKS="${PROJECT_DIR}/data/chunks/candidate_chunks_clean.json"
echo ""
if [ ! -f "$CLEAN_CHUNKS" ]; then
    echo ">>> [Pre-step] Building clean chunk set (first run only)..."
    python filter_chunks.py \
        --input  data/chunks/candidate_chunks.json \
        --output "$CLEAN_CHUNKS"
    echo "    Saved: $CLEAN_CHUNKS"
else
    echo ">>> [Pre-step] Clean chunks already cached — skipping filter."
    echo "    Using: $CLEAN_CHUNKS"
fi

# ===========================================================================
# 4.  Start vLLM OpenAI-compatible server
# ===========================================================================
echo ""
echo ">>> [Step 1] Starting vLLM server  (model: $MODEL_ID)"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$MODEL_SHORT" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    > "logs/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
echo "    PID: $VLLM_PID  |  log: logs/vllm_server_${SLURM_JOB_ID}.log"

# Wait for /health to respond
echo "    Waiting for vLLM to be ready..."
MAX_WAIT=900
WAITED=0
until curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s."
        echo "       Check: logs/vllm_server_${SLURM_JOB_ID}.log"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "    ... ${WAITED}s ..."
done
echo "    vLLM is ready! (${WAITED}s)"

# Export URLs for python scripts
export PHYSIORAG_VLLM_URL="$VLLM_URL"
export PHYSIORAG_VLLM_MODEL="$MODEL_SHORT"

# ===========================================================================
# 5.  Generate evaluation questions
# ===========================================================================
echo ""
echo ">>> [Step 2] Generating evaluation questions..."
python generate_questions.py \
    --provider vllm \
    --model    "$MODEL_SHORT" \
    --count    "$QUESTION_COUNT" \
    --input    data/chunks/candidate_chunks_clean.json \
    --output   "$GOLD_DIR"

# Pick the file we just created (newest in gold_dir)
QUESTIONS_FILE=$(ls -t "${GOLD_DIR}/"gold_eval_*.json | head -1)
echo "    Questions saved to: $QUESTIONS_FILE"

# ===========================================================================
# 6.  Generate answers via RAG
# ===========================================================================
echo ""
echo ">>> [Step 3] Generating RAG answers..."
python generate_answers.py \
    --provider vllm \
    --model    "$MODEL_SHORT" \
    --input    "$QUESTIONS_FILE" \
    --output   "$RESULTS_DIR"

# Pick the answers file we just created
ANSWERS_FILE=$(ls -t "${RESULTS_DIR}/"*_evaluation_generated_answers.json | head -1)
echo "    Answers saved to: $ANSWERS_FILE"

# ===========================================================================
# 7.  Evaluate
# ===========================================================================
echo ""
echo ">>> [Step 4] Running evaluation pipeline..."

# Build a timestamped output filename for this model run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_OUT="${EVAL_DIR}/eval_${MODEL_SHORT}_${TIMESTAMP}.json"

python src/evaluation/eval_pipeline.py \
    --predictions "$ANSWERS_FILE" \
    --output      "$EVAL_OUT"

echo "    Evaluation report: $EVAL_OUT"

# ===========================================================================
# 8.  Shutdown vLLM server
# ===========================================================================
echo ""
echo ">>> [Step 5] Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "    Done."

# ===========================================================================
# 9.  Summary
# ===========================================================================
echo ""
echo "============================================================"
echo "  ✅  Full pipeline COMPLETE"
echo "  Model       : $MODEL_ID"
echo "  Questions   : $QUESTIONS_FILE"
echo "  Answers     : $ANSWERS_FILE"
echo "  Eval report : $EVAL_OUT"
echo "  Finished    : $(date)"
echo "============================================================"

