cat > ~/Shoulder-RAG-HPC/job_open_questions.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=physiorag-open-questions
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/open_questions_%j.log
#SBATCH --error=logs/open_questions_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nandeep.somashekar@fau.de

set -euo pipefail

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PROJECT_DIR="${PROJECT_DIR:-$HOME/Shoulder-RAG-HPC}"
CONDA_ENV="${CONDA_ENV:-/home/hpc/iwso/iwso221h/miniconda3/envs/shoulder-rag}"
export HF_HOME="${WORK}/models/huggingface"
MODEL_SHORT="$(basename $MODEL_ID)"
VLLM_PORT=8000
VLLM_URL="http://localhost:${VLLM_PORT}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
RESULTS_DIR="${PROJECT_DIR}/results"

echo "============================================================"
echo "  PhysioRAG – Open Questions Answer Generation"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Model   : $MODEL_ID"
echo "  Input   : $INPUT_FILE"
echo "  Started : $(date)"
echo "============================================================"

# Activate conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
cd "$PROJECT_DIR"
mkdir -p logs "$RESULTS_DIR"

# Start vLLM server
echo ">>> Starting vLLM server..."
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

# Wait for server
echo ">>> Waiting for vLLM to be ready..."
MAX_WAIT=900
WAITED=0
until curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM did not start in time"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "    ...${WAITED}s..."
done
echo ">>> vLLM ready! (${WAITED}s)"

export PHYSIORAG_VLLM_URL="$VLLM_URL"
export PHYSIORAG_VLLM_MODEL="$MODEL_SHORT"

# Generate answers for open questions — skip question generation entirely
echo ">>> Generating answers for open questions..."
python generate_open_answers.py \
    --provider vllm \
    --model    "$MODEL_SHORT" \
    --input    "$INPUT_FILE" \
    --output   "$RESULTS_DIR"

ANSWERS_FILE=$(ls -t "${RESULTS_DIR}/"*_evaluation_generated_answers.json | head -1)
echo ">>> Answers saved to: $ANSWERS_FILE"

# Stop vLLM
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "============================================================"
echo "  COMPLETE"
echo "  Answers : $ANSWERS_FILE"
echo "  Finished: $(date)"
echo "============================================================"
EOF