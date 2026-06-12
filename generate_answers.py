"""
Generate answers for evaluation questions using the PhysioRAG pipeline.

Supports:
- Multiple LLM models via CLI
- Batch processing with checkpointing
- Environment-aware paths
- Custom output directories
"""

import json
import re
import logging
import argparse
from pathlib import Path
from src.rag.hybrid_retriever import MedicalHybridRetriever
from src.indexing.vector_store import MedicalVectorStore
from src.indexing.embedder import MedicalEmbedder
from src.rag.reasoning_gen import ReasoningGenerator
from langchain_core.documents import Document
from src.rag.prompts import ANSWER_GEN_PROMPT
from config import OLLAMA_LLM_MODEL, OLLAMA_BASE_URL, HF_LLM_MODEL, VLLM_MODEL_ID, VLLM_BASE_URL, LLM_PROVIDER, CANDIDATE_CHUNKS_PATH, RESULTS_DIR, LOG_LEVEL

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_model_tag(model_name: str) -> str:
    """Convert model names to filesystem-safe tags for output/checkpoint filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)


def load_pipeline_components(embed_model: str = None):
    """Load embedding, vector store, and retriever components."""
    logger.info("Loading pipeline components...")
    embedder = MedicalEmbedder(embed_model or "all-MiniLM-L6-v2")
    store = MedicalVectorStore()
    vectorstore = store.load_index(embedder)

    # Load candidate chunks
    with open(CANDIDATE_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    formatted_docs = [
        Document(
            page_content=c["chunk_text"],
            metadata={
                **c.get("metadata", {}),
                "chunk_id": c.get("chunk_id"),
                "document": c.get("document"),
                "section_title": c.get("section_title"),
                "page_number": c.get("page_number")
            }
        )
        for c in chunks
    ]

    retriever = MedicalHybridRetriever(vectorstore, formatted_docs)
    logger.info(f"Loaded {len(formatted_docs)} chunks into retriever")
    return retriever, embedder


def load_evaluation_questions(questions_file: str):
    """Load evaluation questions from JSON file."""
    questions_path = Path(questions_file)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} evaluation questions")
    return questions


def format_question_for_prompt(item: dict) -> str:
    """Build a readable MCQ string: Stem + options."""
    stem = item["Question"]
    options = item["Options"]
    options_str = "\n".join([f"{letter}. {text}" for letter, text in sorted(options.items())])
    return f"{stem}\n{options_str}"


def process_single_question(
    question_item: dict,
    retriever: MedicalHybridRetriever,
    generator: ReasoningGenerator
) -> dict:
    """Process a single question and generate answer."""
    question_stem = question_item["Question"]
    options_dict = question_item["Options"]
    gold_answer_letter = question_item["Correct_Answer"].strip().lower()
    gold_text_answer = question_item["Text_Answer"].strip()
    gold_chunk_id = question_item.get("chunk_id")

    # Retrieve context
    context_docs = retriever.get_relevant_documents(question_stem)
    MAX_CONTEXT_CHUNKS = 8
    context_docs_capped = context_docs[:MAX_CONTEXT_CHUNKS]

    context_docs_info = [
        {
            "text": doc.page_content,
            "chunk_id": doc.metadata.get("chunk_id"),
            "document": doc.metadata.get("document", "Unknown"),
            "section_title": doc.metadata.get("section_title", "General")
        }
        for doc in context_docs_capped
    ]

    context_text = "\n\n---\n\n".join([d["text"] for d in context_docs_info])
    options_display = "\n".join([f"{letter}. {text}" for letter, text in sorted(options_dict.items())])

    # Build prompt
    prompt = f"""{ANSWER_GEN_PROMPT}

Use ONLY the retrieved context below. Choose the option that matches the context precisely.

QUESTION:
{question_stem}

OPTIONS:
{options_display}

RETRIEVED CONTEXT:
{context_text}

Return JSON ONLY:
{{
    "Answer": "letter",
    "Text_Answer": "exact text"
}}
"""

    # Generate answer
    try:
        raw_output = generator.provider.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Deterministic for MCQs
        )
    except Exception as e:
        logger.error(f"Error generating answer for chunk {gold_chunk_id}: {e}")
        raw_output = "{}"

    # Parse JSON output
    try:
        if "```json" in raw_output:
            json_str = raw_output.split("```json")[-1].split("```")[0].strip()
        elif "```" in raw_output:
            json_str = raw_output.split("```")[1].strip()
        else:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            json_str = raw_output[start:end]
        answer_json = json.loads(json_str)
    except Exception:
        answer_json = {"Answer": None, "Text_Answer": None}

    # Normalize answer
    if "Text Answer" in answer_json:
        answer_json["Text_Answer"] = answer_json.pop("Text Answer")

    raw_letter = str(answer_json.get("Answer", "")).lower()
    normalized_letter = re.sub(r'[^a-e]', '', raw_letter)

    if normalized_letter in options_dict:
        answer_json["Answer"] = normalized_letter
        answer_json["Text_Answer"] = options_dict[normalized_letter]
    else:
        answer_json["Answer"] = None
        answer_json["Text_Answer"] = None

    return {
        "question_stem": question_stem,
        "options": options_dict,
        "generated_answer": {
            "Answer": answer_json["Answer"],
            "Text_Answer": answer_json["Text_Answer"]
        },
        "gold_answer": {
            "Answer": gold_answer_letter,
            "Text_Answer": gold_text_answer,
            "chunk_id": gold_chunk_id,
            "document": question_item.get("Reference", "Unknown"),
        },
        "context_docs": context_docs_info
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers for evaluation questions using PhysioRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate answers with default model
  python generate_answers.py --input data/gold_standard/questions.json

  # Use specific model
  python generate_answers.py --input data/gold_standard/questions.json --model qwen2.5:7b

  # Resume from checkpoint
  python generate_answers.py --input data/gold_standard/questions.json --resume

  # Custom output directory
  python generate_answers.py --input data/gold_standard/questions.json --output /path/to/results
        """
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to evaluation questions JSON file"
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help=(
            "LLM model to use. "
            f"Defaults to {OLLAMA_LLM_MODEL} for ollama, "
            f"{VLLM_MODEL_ID} for vllm, "
            f"and {HF_LLM_MODEL} for hf."
        )
    )

    parser.add_argument(
        "--embed-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model to use (default: all-MiniLM-L6-v2)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=RESULTS_DIR,
        help=f"Output directory for results (default: {RESULTS_DIR})"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_PROVIDER,
        choices=["ollama", "vllm", "hf", "mock"],
        help=f"LLM provider to use (default: {LLM_PROVIDER})"
    )

    args = parser.parse_args()

    if not args.model:
        if args.provider == "hf":
            args.model = HF_LLM_MODEL
        elif args.provider == "vllm":
            args.model = VLLM_MODEL_ID
        else:
            args.model = OLLAMA_LLM_MODEL

    logger.info(f"Starting answer generation with arguments: {vars(args)}")

    try:
        # Load pipeline
        retriever, embedder = load_pipeline_components(args.embed_model)

        _url_map = {"ollama": OLLAMA_BASE_URL, "vllm": VLLM_BASE_URL}
        generator = ReasoningGenerator(
            model_name=args.model,
            provider_type=args.provider,
            base_url=_url_map.get(args.provider),
        )

        # Load questions
        questions = load_evaluation_questions(args.input)

        logger.info(f"Using model: {args.model}, Provider: {args.provider}")

        if args.resume:
            logger.warning("--resume is currently ignored in the rich evaluation output mode.")

        # Process each question while preserving gold/context fields for evaluation.
        results = []
        for idx, q in enumerate(questions, 1):
            logger.info(f"[{idx}/{len(questions)}] Processing question")
            results.append(process_single_question(q, retriever, generator))

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_tag = safe_model_tag(args.model)
        output_path = output_dir / f"{model_tag}_evaluation_generated_answers.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Completed! Results saved to: {output_path}")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
