# PhysioRAG - CLI Commands Reference

Quick reference for all new CLI commands and options.

---

## 1. Generate Questions

### Basic Usage
```bash
python generate_questions.py
```
Generates 50 questions with default model (qwen3:8b).

### With Model Selection
```bash
python generate_questions.py --model qwen2.5:7b --count 100
```

### With Custom Output
```bash
python generate_questions.py --count 50 --output ~/my_results/questions
```

### With Mock Provider (No Ollama Needed)
```bash
python generate_questions.py --count 10 --provider mock
```

### Full Command Example
```bash
python generate_questions.py \
    --count 50 \
    --model qwen3:8b \
    --output results/questions_v1 \
    --input data/chunks/candidate_chunks.json
```

### Help & Options
```bash
python generate_questions.py --help

Options:
  -c, --count COUNT              Target number of questions (default: 50)
  -m, --model MODEL              LLM model to use (default: qwen3:8b)
  -o, --output PATH              Output directory for results
  -i, --input PATH               Path to candidate chunks JSON file
  --provider {ollama,mock}       LLM provider to use (default: ollama)
  --no-skip-existing             Don't skip chunks that have been used
```

---

## 2. Generate Answers

### Basic Usage
```bash
python generate_answers.py --input questions.json
```
Generates answers for the questions using default model.

### With Model Selection
```bash
python generate_answers.py --input questions.json --model qwen2.5:7b
```

### Resume from Checkpoint
```bash
python generate_answers.py --input questions.json --resume
```
Continues from last checkpoint if previous run was interrupted.

### Custom Batch Size
```bash
python generate_answers.py --input questions.json --batch-size 5
```

### Full Command Example
```bash
python generate_answers.py \
    --input results/questions/gold_eval_qwen3_8b.json \
    --model qwen3:8b \
    --output results/answers_v1 \
    --embed-model all-MiniLM-L6-v2 \
    --batch-size 10 \
    --resume
```

### Help & Options
```bash
python generate_answers.py --help

Options:
  -i, --input PATH               Path to evaluation questions JSON file (REQUIRED)
  -m, --model MODEL              LLM model to use (default: qwen3:8b)
  -o, --output PATH              Output directory for results
  --embed-model MODEL            Embedding model to use (default: all-MiniLM-L6-v2)
  --batch-size SIZE              Batch size for processing (default: 10)
  --provider {ollama,mock}       LLM provider to use (default: ollama)
  --resume                       Resume from checkpoint if available
```

---

## 3. Synchronize to HPC

### Using Configuration File
```bash
python sync_to_hpc.py --config hpc_sync.json
```

### Direct Parameters
```bash
python sync_to_hpc.py --host hpc.uni.edu --user myusername --path /home/myusername/physiorag
```

### Dry Run (Preview Changes)
```bash
python sync_to_hpc.py --config hpc_sync.json --dry-run
```

### Code Only (No Data)
```bash
python sync_to_hpc.py --config hpc_sync.json --code-only
```

### Pull Results from HPC
```bash
python sync_to_hpc.py --config hpc_sync.json --pull-results
```

### Full Command Example
```bash
python sync_to_hpc.py \
    --host hpc.university.edu \
    --user john_doe \
    --path /home/john_doe/physiorag \
    --dry-run
```

### Help & Options
```bash
python sync_to_hpc.py --help

Options:
  --host HOST                    HPC server hostname
  -u, --user USER                Username on HPC
  -p, --path PATH                Remote project path
  --config FILE                  Load configuration from JSON file
  --dry-run                      Preview changes without making them
  --code-only                    Only sync code files, not data
  --pull-results                 Pull results from HPC to local
```

---

## 4. Test Local Setup

### Full Test Suite
```bash
python test_local_setup.py
```

Runs comprehensive tests including:
- File structure validation
- Python imports
- Configuration loading
- CLI interface
- Ollama availability
- Basic functionality

---

## Environment Variables

Control behavior via environment variables (alternative to CLI):

```bash
# Model selection
export PHYSIORAG_LLM_MODEL="qwen2.5:7b"
export PHYSIORAG_EMBED_MODEL="all-MiniLM-L6-v2"

# Ollama configuration
export PHYSIORAG_OLLAMA_URL="http://localhost:11434"

# Environment
export PHYSIORAG_ENV="hpc"  # or "local"

# Batch processing
export PHYSIORAG_BATCH_SIZE=10
export PHYSIORAG_MAX_WORKERS=4
export PHYSIORAG_CHECKPOINT_INTERVAL=50

# Logging
export PHYSIORAG_LOG_LEVEL="INFO"
export PHYSIORAG_DEBUG="false"

# Paths (HPC only)
export PHYSIORAG_HPC_ROOT="/scratch/username/physiorag"
```

---

## Common Workflows

### Workflow 1: Quick Local Test
```bash
# Test with mock provider (no Ollama needed)
python generate_questions.py --count 5 --provider mock
```

### Workflow 2: Generate with Different Models
```bash
# Model 1
python generate_questions.py --count 50 --model qwen3:8b --output results/model1_q
python generate_answers.py --input results/model1_q/gold_eval_*.json --model qwen3:8b --output results/model1_a

# Model 2
python generate_questions.py --count 50 --model qwen2.5:7b --output results/model2_q
python generate_answers.py --input results/model2_q/gold_eval_*.json --model qwen2.5:7b --output results/model2_a

# Compare results
ls -la results/model1_a/ results/model2_a/
```

### Workflow 3: Deploy to HPC
```bash
# 1. Test locally
python test_local_setup.py

# 2. Sync to HPC
python sync_to_hpc.py --config hpc_sync.json

# 3. Setup HPC (on HPC node)
ssh username@hpc.server
bash ~/physiorag/setup_hpc.sh
cd ~/physiorag && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
ollama pull qwen3:8b

# 4. Submit jobs
sbatch job_generate_questions.sh
sbatch job_generate_answers.sh

# 5. Monitor
squeue -u $USER
tail -f logs/questions_*.log

# 6. Get results (from local machine)
python sync_to_hpc.py --config hpc_sync.json --pull-results
```

### Workflow 4: Resume Interrupted Job
```bash
# If a job was interrupted, continue from checkpoint
python generate_answers.py --input questions.json --resume
```

### Workflow 5: Use Custom Batch Size
```bash
# For better GPU memory usage
export PHYSIORAG_BATCH_SIZE=5
python generate_answers.py --input questions.json

# Or via CLI
python generate_answers.py --input questions.json --batch-size 5
```

---

## SLURM Job Commands

### Submit Jobs
```bash
# From HPC node
sbatch job_generate_questions.sh
sbatch job_generate_answers.sh
```

### Check Status
```bash
# View all your jobs
squeue -u $USER

# View specific job details
sinfo
squeue --job=123456
```

### Monitor Job
```bash
# Watch logs in real-time
tail -f ~/physiorag/logs/questions_*.log

# Check GPU usage
nvidia-smi
```

### Cancel Jobs
```bash
# Cancel specific job
scancel 123456

# Cancel all your jobs
scancel -u $USER
```

---

## Tips & Tricks

### Speed Up Generation
```bash
# Increase batch size (if GPU memory allows)
python generate_answers.py --input questions.json --batch-size 20
```

### Save Progress Frequently
```bash
# Checkpoint every 10 items (instead of default 50)
export PHYSIORAG_CHECKPOINT_INTERVAL=10
python generate_answers.py --input questions.json
```

### Debug Issues
```bash
# Enable debug logging
export PHYSIORAG_LOG_LEVEL="DEBUG"
export PHYSIORAG_DEBUG="true"
python generate_questions.py --count 5 --provider mock
```

### Test Sync Before Deploying
```bash
# Preview what will be synced
python sync_to_hpc.py --config hpc_sync.json --dry-run

# Only sync code (no data)
python sync_to_hpc.py --config hpc_sync.json --code-only
```

### Run Tests
```bash
# Full local validation
python test_local_setup.py
```

---

## Troubleshooting Commands

### Check Configuration
```bash
python -c "from config import *; print(f'Model: {OLLAMA_LLM_MODEL}'); print(f'Batch: {BATCH_SIZE}')"
```

### Verify Imports
```bash
python -c "from src.rag.llm_provider import *; from src.batch.processor import *; print('✓ OK')"
```

### Check File Paths
```bash
python -c "from config import CANDIDATE_CHUNKS_PATH; import os; print(f'Chunks file: {CANDIDATE_CHUNKS_PATH}'); print(f'Exists: {os.path.exists(CANDIDATE_CHUNKS_PATH)}')"
```

### Verify Ollama
```bash
ollama --version
ollama list
ollama show qwen3:8b
```

### Check SSH Access
```bash
# Test connection
ssh -v username@hpc.server "echo Connection OK"
```

---

## Command Examples by Use Case

### For Research: Testing Multiple Models
```bash
for model in "qwen3:8b" "qwen2.5:7b" "mistral:7b"; do
    python generate_questions.py --count 30 --model $model --output results/q_$model
    python generate_answers.py --input results/q_$model/gold_eval_*.json --model $model --output results/a_$model
done
```

### For Production: Batch Processing
```bash
python generate_questions.py --count 500 --model qwen3:8b --output results/prod_questions
python generate_answers.py --input results/prod_questions/gold_eval_*.json --resume --batch-size 20
```

### For Validation: Mock Testing
```bash
python generate_questions.py --count 100 --provider mock --output results/mock_test
```

### For HPC Scaling: Parallel Jobs
```bash
# Job 1: Questions
sbatch job_generate_questions.sh

# Job 2: Wait for Job 1, then answers
# Edit job_generate_answers.sh to add:
# #SBATCH --dependency=afterok:JOBID

sbatch job_generate_answers.sh
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Generate 50 questions | `python generate_questions.py` |
| Generate 100 questions | `python generate_questions.py --count 100` |
| With model qwen2.5:7b | `--model qwen2.5:7b` |
| Generate answers | `python generate_answers.py --input questions.json` |
| Resume from checkpoint | `--resume` |
| Custom batch size | `--batch-size 5` |
| Test without Ollama | `--provider mock` |
| Sync to HPC | `python sync_to_hpc.py --config hpc_sync.json` |
| Dry run (preview) | `--dry-run` |
| Pull results from HPC | `--pull-results` |
| Test setup | `python test_local_setup.py` |
| Show help | `--help` |

---

**For more details**, see:
- HPC_DEPLOYMENT_GUIDE.md
- IMPLEMENTATION_CHECKLIST.md
- README_HPC_CHANGES.md

