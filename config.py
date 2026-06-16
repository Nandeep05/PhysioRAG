import os
import json

# ============================================================
# ENVIRONMENT DETECTION & PATH SETUP
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENVIRONMENT = os.getenv("PHYSIORAG_ENV", "local")  # local, hpc, docker

# Detect if running on HPC by checking common HPC environment variables
IS_HPC = bool(os.getenv("SLURM_JOB_ID") or os.getenv("PBS_JOBID"))

# ============================================================
# PATH CONFIGURATION
# ============================================================
if ENVIRONMENT == "hpc" or IS_HPC:
    # HPC paths (typically scratch or project directories)
    HPC_ROOT = os.getenv("PHYSIORAG_HPC_ROOT", os.path.expanduser("~/physiorag"))
    DATA_DIR = os.path.join(HPC_ROOT, "data")
    RESULTS_DIR = os.path.join(HPC_ROOT, "results")
else:
    # Local paths
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
FAISS_INDEX_PATH = os.path.join(CHUNKS_DIR, "faiss_index")
CANDIDATE_CHUNKS_PATH = os.path.join(CHUNKS_DIR, "candidate_chunks_clean.json")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# ============================================================
# MODEL CONFIGURATION
# ============================================================
# Defaults - can be overridden via environment variables or CLI
EMBED_MODEL_NAME = os.getenv("PHYSIORAG_EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("PHYSIORAG_LLM_PROVIDER", "ollama")
OLLAMA_LLM_MODEL = os.getenv("PHYSIORAG_LLM_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("PHYSIORAG_OLLAMA_URL", "http://localhost:11434")

# vLLM server (started inside a SLURM job via job_vllm_*.sh)
# Override with:  export PHYSIORAG_VLLM_URL="http://localhost:8000"
VLLM_BASE_URL = os.getenv("PHYSIORAG_VLLM_URL", "http://localhost:8000")
# HuggingFace model ID used when provider is "vllm" or "hf"
VLLM_MODEL_ID = os.getenv("PHYSIORAG_VLLM_MODEL", "Qwen/Qwen3-14B")

HF_LLM_MODEL = os.getenv("PHYSIORAG_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Supported models for easy switching
SUPPORTED_MODELS = {
    "qwen3:8b": {"type": "ollama", "parameters": "8b", "local_memory_gb": 8},
    "qwen2.5:7b": {"type": "ollama", "parameters": "7b", "local_memory_gb": 7},
    "mistral:7b": {"type": "ollama", "parameters": "7b", "local_memory_gb": 7},
    "llama2:7b": {"type": "ollama", "parameters": "7b", "local_memory_gb": 7},
    "neural-chat:7b": {"type": "ollama", "parameters": "7b", "local_memory_gb": 7},
}

# ============================================================
# RAG HYPERPARAMETERS
# ============================================================
CHUNK_SIZE = 1200      # Increased for better semantic coherence (~300 tokens)
CHUNK_OVERLAP = 250    # Slightly larger overlap
TOP_K_RETRIEVAL = 4
HYBRID_ALPHA = 0.5

# ============================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================
BATCH_SIZE = int(os.getenv("PHYSIORAG_BATCH_SIZE", "10"))
MAX_WORKERS = int(os.getenv("PHYSIORAG_MAX_WORKERS", "4"))
CHECKPOINT_INTERVAL = int(os.getenv("PHYSIORAG_CHECKPOINT_INTERVAL", "50"))

# ============================================================
# LOGGING & DEBUGGING
# ============================================================
LOG_LEVEL = os.getenv("PHYSIORAG_LOG_LEVEL", "INFO")
DEBUG_MODE = os.getenv("PHYSIORAG_DEBUG", "false").lower() == "true"

# ============================================================
# CONFIG FILE LOADING (Optional - for advanced users)
# ============================================================
CONFIG_FILE = os.getenv("PHYSIORAG_CONFIG_FILE")

def load_config_file(config_path: str) -> dict:
    """Load additional config from JSON file if provided."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

# Apply config file overrides if present
if CONFIG_FILE:
    _config = load_config_file(CONFIG_FILE)
    for key, value in _config.items():
        if key.isupper():  # Only override uppercase constants
            globals()[key] = value

