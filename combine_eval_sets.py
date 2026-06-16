import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_eval_file(path: Path) -> list[dict]:
    """Load an eval JSON file and return its per-question entries."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "perquestionresults" in data:
            return data["perquestionresults"]
        if isinstance(data.get("results"), list):
            return data["results"]

    if isinstance(data, list):
        return data

    logger.warning("Unrecognized structure in %s; returning empty list", path)
    return []


def select_questions(
    all_questions: list[dict],
    target_total: int | None = None,
) -> list[dict]:
    """
    Optionally downsample to a target total, balancing roughly per document.
    If target_total is None, all questions are kept.
    """
    if target_total is None or target_total >= len(all_questions):
        return all_questions

    by_doc = defaultdict(list)
    for q in all_questions:
        doc = q.get("document", "Unknown")
        by_doc[doc].append(q)

    selected = []
    docs = list(by_doc.keys())
    idx_per_doc = {doc: 0 for doc in docs}

    while len(selected) < target_total:
        made_progress = False
        for doc in docs:
            doc_list = by_doc[doc]
            i = idx_per_doc[doc]
            if i < len(doc_list):
                selected.append(doc_list[i])
                idx_per_doc[doc] += 1
                made_progress = True
                if len(selected) >= target_total:
                    break
        if not made_progress:
            break

    return selected


def make_summary(
    questions: list[dict],
    source_files: list[str],
    target_per_document: str = "variable",
    note: str = "Combined evaluation set prepared for manual review.",
) -> dict:
    docs = sorted({q.get("document", "Unknown") for q in questions})

    return {
        "totalquestions": len(questions),
        "target_per_document": target_per_document,
        "documents": len(docs),
        "sources": sorted(set(source_files)),
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple evaluation JSON files into a single set"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of eval JSON files to combine",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output combined JSON file",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Optional target total number of questions (default: keep all)",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.output)

    logger.info("Combining %d eval files", len(input_paths))

    all_questions = []
    source_files = []

    for path in input_paths:
        logger.info("Loading %s", path)
        questions = load_eval_file(path)
        logger.info("  Found %d questions", len(questions))

        for q in questions:
            q = dict(q)
            q["source_file"] = path.name
            all_questions.append(q)
        source_files.append(path.name)

    logger.info("Total questions before selection: %d", len(all_questions))

    selected = select_questions(all_questions, target_total=args.target_total)
    logger.info("Selected %d questions for combined set", len(selected))

    summary = make_summary(
        selected,
        source_files=source_files,
        target_per_document="variable",
        note=f"Combined evaluation set prepared for manual review ({len(selected)} questions).",
    )

    output_data = {
        "summary": summary,
        "perquestionresults": selected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved combined evaluation set to %s", output_path)


if __name__ == "__main__":
    main()