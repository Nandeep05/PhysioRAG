"""
generate_red_flag_questions.py

Generate red-flag MCQ questions + answers from pre-filtered chunks.

Input:
  Evaluation_sets/intermediate/candidate_chunks_redflags.json

Output:
  Evaluation_sets/final/red_flag_questions.json

Schema matches existing red_flag_questions.json:
{
  "summary": {
    "totalquestions": int,
    "category": "red_flag",
    "note": "..."
  },
  "questions": [
    {
      "question_id": "RF_01",
      "category": "red_flag",
      "question": "...",
      "options": {
        "a": "...",
        "b": "...",
        "c": "...",
        "d": "...",
        "e": "..."
      },
      "correct_answer": "b",
      "text_answer": "...",
      "source_document": "..."
    }
  ]
}
"""

import json
import argparse
import logging
from pathlib import Path

from config import VLLM_BASE_URL, LOG_LEVEL  # matches your other scripts
from src.rag.reasoning_gen import ReasoningGenerator  # same as generate_open_answers.py

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


MCQ_PROMPT_TEMPLATE = """
You are a physiotherapist and clinical educator.

Using ONLY the clinical text below, write ONE multiple-choice question
specifically targeting RED FLAG recognition in a shoulder pain context.

Rules:
1. The question must focus on identifying serious pathology / red flags,
   such as fracture, dislocation, neurological deficit, systemic disease,
   or other urgent referral situations.
2. Create exactly 5 answer options (A, B, C, D, E).
3. Exactly ONE option must be clearly the best / correct answer.
4. The other options should be plausible but incorrect (distractors).
5. Do not mention the source document name.
6. Aim for the level of a practicing physiotherapist.

Return the result as valid JSON in the following format ONLY:
{
  "stem": "<question text>",
  "options": ["<A>", "<B>", "<C>", "<D>", "<E>"],
  "correct_index": 0
}

CLINICAL TEXT:
----------------
{context}
----------------
"""


def load_redflag_chunks(path: Path) -> list[dict]:
    """Load pre-filtered red-flag chunks."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"]
    return data


def generate_mcq_for_chunk(chunk: dict, generator: ReasoningGenerator) -> dict | None:
    """Generate one red-flag MCQ from a single chunk."""
    context = chunk.get("chunk_text") or chunk.get("chunktext", "")
    if not context.strip():
        logger.warning("Chunk %s has empty text, skipping", chunk.get("chunk_id"))
        return None

    prompt = MCQ_PROMPT_TEMPLATE.format(context=context)

    try:
        raw = generator.provider.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as e:
        logger.error("LLM error for chunk %s: %s", chunk.get("chunk_id"), e)
        return None

    try:
        mcq = json.loads(raw)
    except Exception as e:
        logger.error("JSON parse error for chunk %s: %s", chunk.get("chunk_id"), e)
        return None

    if not isinstance(mcq, dict):
        logger.error("MCQ output not a dict for chunk %s", chunk.get("chunk_id"))
        return None

    stem = mcq.get("stem", "").strip()
    options = mcq.get("options", [])
    correct_index = mcq.get("correct_index")

    if not stem or not isinstance(options, list):
        logger.error("Invalid MCQ structure for chunk %s", chunk.get("chunk_id"))
        return None

    if len(options) != 5:
        logger.error(
            "Expected 5 options for chunk %s, got %d",
            chunk.get("chunk_id"),
            len(options),
        )
        return None

    if not isinstance(correct_index, int) or not (0 <= correct_index < 5):
        logger.error("Invalid correct_index for chunk %s", chunk.get("chunk_id"))
        return None

    # Map list → dict with keys a–e
    option_keys = ["a", "b", "c", "d", "e"]
    options_dict = {k: v for k, v in zip(option_keys, options)}
    correct_key = option_keys[correct_index]
    text_answer = options_dict[correct_key]

    return {
        "question_id": f"RF_{int(chunk.get('chunk_id', 0)):02d}",
        "category": "red_flag",
        "question": stem,
        "options": options_dict,
        "correct_answer": correct_key,
        "text_answer": text_answer,
        "source_document": chunk.get("document"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate red-flag MCQ questions from pre-filtered chunks"
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to candidate_chunks_redflags.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output red_flag_questions.json",
    )
    parser.add_argument(
        "--model",
        default="Qwen3-14B",
        help="Base model name (default: Qwen3-14B)",
    )
    parser.add_argument(
        "--provider",
        default="vllm",
        choices=["vllm", "ollama", "hf"],
        help="LLM provider type (default: vllm)",
    )
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    output_path = Path(args.output)

    logger.info("Loading red-flag chunks from %s", chunks_path)
    chunks = load_redflag_chunks(chunks_path)
    logger.info("Loaded %d red-flag chunks", len(chunks))

    logger.info("Initialising ReasoningGenerator...")
    generator = ReasoningGenerator(
        model_name=args.model,
        provider_type=args.provider,
        base_url=VLLM_BASE_URL,
    )

    questions = []
    for idx, chunk in enumerate(chunks, start=1):
        cid = chunk.get("chunk_id")
        logger.info(
            "[%d/%d] Generating MCQ from chunk_id=%s (doc=%s)",
            idx,
            len(chunks),
            cid,
            chunk.get("document"),
        )
        mcq = generate_mcq_for_chunk(chunk, generator)
        if mcq:
            questions.append(mcq)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "summary": {
            "totalquestions": len(questions),
            "category": "red_flag",
            "note": (
                "Questions initially generated from pre-filtered red-flag chunks "
                "using LLM assistance and then manually curated by physiotherapists."
            ),
        },
        "questions": questions,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d red-flag questions to %s", len(questions), output_path)


if __name__ == "__main__":
    main()