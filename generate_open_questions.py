"""
generate_open_questions.py

Generate open-ended clinical questions with LLM assistance and
save them as a JSON file under Evaluation_sets/final.

Input (seed):
  Evaluation_sets/final/open_ended_seed_physio.json

Output:
  Evaluation_sets/final/open_ended_questions.json

Schema matches existing open_ended_questions.json:
{
  "metadata": {
    "total": 14,
    "source": "Collected from physiotherapist — cleaned and expanded",
    "type": "open_ended",
    "categories": [...]
  },
  "questions": [
    {
      "question_id": "OQ_01",
      "category": "...",
      "question": "...",
      "source": "physio_collected" or "llm_generated"
    }
  ]
}
"""

import json
import argparse
import logging
from pathlib import Path

from config import VLLM_BASE_URL, LOG_LEVEL
from src.rag.reasoning_gen import ReasoningGenerator

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_seed_questions(seed_path: Path) -> list[dict]:
    """Load physio-supplied seed questions from JSON."""
    with seed_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    return data.get("questions", [])


def build_generation_prompt(seed_questions: list[dict]) -> str:
    """Build a prompt asking the LLM to propose similar open-ended questions."""
    lines = [
        "You are a physiotherapist and clinical educator.",
        "You will generate open-ended clinical questions about shoulder pain.",
        "",
        "Below are example questions provided by a human physiotherapist.",
        "They illustrate the desired style and level of detail.",
        "",
        "SEED EXAMPLES:",
    ]

    for q in seed_questions:
        qid = q.get("question_id", "")
        cat = q.get("category", "General")
        text = q.get("question", "")
        lines.append(f"- [{qid} / {cat}] {text}")

    lines.extend(
        [
            "",
            "TASK:",
            "Based on these seed examples, propose 10 new open-ended questions.",
            "Each question should:",
            "1. Be a single free-text question (no multiple choice).",
            "2. Be clinically realistic and relevant to shoulder assessment,",
            "   management, exercise progression, or red-flag screening.",
            "3. Avoid mentioning specific document titles or guideline names.",
            "",
            "Return the questions as a JSON list with objects of the form:",
            "{",
            '  "category": "<category>",',
            '  "question": "<question text>"',
            "}",
            "",
            "ONLY output valid JSON (no extra commentary).",
        ]
    )

    return "\n".join(lines)


def generate_questions_with_llm(
    seed_questions: list[dict],
    model_name: str,
    provider_type: str,
) -> list[dict]:
    """Use ReasoningGenerator to generate additional open-ended questions."""
    logger.info(
        "Initialising ReasoningGenerator: model=%s provider=%s",
        model_name,
        provider_type,
    )

    generator = ReasoningGenerator(
        model_name=model_name,
        provider_type=provider_type,
        base_url=VLLM_BASE_URL,
    )

    prompt = build_generation_prompt(seed_questions)
    logger.info("Sending prompt to LLM...")
    raw_json = generator.provider.generate(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    generated = json.loads(raw_json)
    if isinstance(generated, list):
        return generated
    logger.warning("LLM output was not a list; wrapping into list.")
    return [generated]


def main():
    parser = argparse.ArgumentParser(
        description="Generate open-ended questions with LLM assistance"
    )
    parser.add_argument(
        "--seed",
        required=True,
        help="Path to physio seed questions JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for final open-ended questions",
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

    seed_path = Path(args.seed)
    output_path = Path(args.output)

    logger.info("Loading seed questions from %s", seed_path)
    seed_questions = load_seed_questions(seed_path)
    logger.info("Loaded %d seed questions", len(seed_questions))

    generated_questions = generate_questions_with_llm(
        seed_questions,
        model_name=args.model,
        provider_type=args.provider,
    )

    final_questions = []
    next_id = 1

    # Keep seed questions
    for q in seed_questions:
        qid = q.get("question_id") or f"OQ_{next_id:02d}"
        cat = q.get("category", "General")
        text = q.get("question", "")
        final_questions.append(
            {
                "question_id": qid,
                "category": cat,
                "question": text,
                "source": q.get("source", "physio_collected"),
            }
        )
        next_id += 1

    # Add generated questions
    for q in generated_questions:
        cat = q.get("category", "General")
        text = q.get("question", "")
        qid = f"OQ_{next_id:02d}"
        final_questions.append(
            {
                "question_id": qid,
                "category": cat,
                "question": text,
                "source": "llm_generated",
            }
        )
        next_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_categories = sorted({q.get("category", "General") for q in final_questions})

    output_data = {
        "metadata": {
            "total": len(final_questions),
            "source": (
                "Collected from physiotherapist — cleaned and expanded with "
                "LLM assistance"
            ),
            "type": "open_ended",
            "categories": all_categories,
        },
        "questions": final_questions,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d open-ended questions to %s", len(final_questions), output_path)


if __name__ == "__main__":
    main()