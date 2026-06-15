import argparse
import json
import re
from pathlib import Path


def norm(s):
    """Normalize whitespace and case for text comparison."""
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def pct(count, total):
    return round(count / total * 100, 2) if total > 0 else 0.0


def evaluate_results(results):
    evaluation_results = []

    total_questions = len(results)
    retrieval_correct_count = 0
    answer_letter_correct_count = 0
    answer_text_correct_count = 0
    fully_correct_count = 0

    for item in results:
        question_stem = item.get("question_stem") or item.get("question", "")
        options = item.get("options", {})

        gold = item.get("gold_answer", {})
        gold_answer_letter = norm(gold.get("Answer") or "")
        gold_text_answer = (gold.get("Text_Answer") or gold.get("Text Answer") or "").strip()
        gold_chunk_id = gold.get("chunk_id")
        gold_document = gold.get("document", "Unknown")

        gen = item.get("generated_answer") or {}
        generated_letter = norm(gen.get("Answer") or "")
        generated_text = (gen.get("Text_Answer") or gen.get("Text Answer") or "").strip()

        context_docs = item.get("context_docs", [])
        all_retrieved_chunk_ids = [
            doc.get("chunk_id")
            for doc in context_docs
            if doc.get("chunk_id") is not None
        ]
        gold_chunk_ids = [gold_chunk_id] if gold_chunk_id is not None else []

        retrieval_correct = any(cid in gold_chunk_ids for cid in all_retrieved_chunk_ids)
        if retrieval_correct:
            retrieval_correct_count += 1

        gen_letter_clean = generated_letter.replace(".", "").strip()
        gold_letter_clean = gold_answer_letter.replace(".", "").strip()
        answer_letter_correct = (gen_letter_clean == gold_letter_clean) and bool(gen_letter_clean)
        if answer_letter_correct:
            answer_letter_correct_count += 1

        answer_text_correct = bool(generated_text) and (norm(generated_text) == norm(gold_text_answer))
        if answer_text_correct:
            answer_text_correct_count += 1

        fully_correct = retrieval_correct and answer_text_correct
        if fully_correct:
            fully_correct_count += 1

        evaluation_results.append({
            "question": question_stem,
            "options": options,
            "gold_answer": {
                "Answer": gold_letter_clean,
                "Text_Answer": gold_text_answer,
                "chunk_id": gold_chunk_id,
                "document": gold_document,
            },
            "generated_answer": {
                "Answer": gen_letter_clean,
                "Text_Answer": generated_text,
            },
            "all_retrieved_chunk_ids": all_retrieved_chunk_ids,
            "gold_chunk_ids": gold_chunk_ids,
            "retrieval_correct": retrieval_correct,
            "answer_letter_correct": answer_letter_correct,
            "answer_text_correct": answer_text_correct,
            "fully_correct": fully_correct,
        })

    none_answers = [r for r in evaluation_results if not r["generated_answer"]["Answer"]]
    attempted = total_questions - len(none_answers)

    summary = {
        "total_questions": total_questions,
        "failed_to_answer": len(none_answers),
        "retrieval_accuracy_pct": pct(retrieval_correct_count, total_questions),
        "answer_letter_accuracy_pct": pct(answer_letter_correct_count, total_questions),
        "answer_text_accuracy_pct": pct(answer_text_correct_count, total_questions),
        "fully_correct_pct": pct(fully_correct_count, total_questions),
        "attempted_answer_accuracy_pct": pct(answer_text_correct_count, attempted) if attempted else 0.0,
        "counts": {
            "retrieval_correct": retrieval_correct_count,
            "answer_letter_correct": answer_letter_correct_count,
            "answer_text_correct": answer_text_correct_count,
            "fully_correct": fully_correct_count,
        },
    }

    doc_breakdown = {}
    for item in evaluation_results:
        doc = item["gold_answer"].get("document", "Unknown")
        if doc not in doc_breakdown:
            doc_breakdown[doc] = {
                "total": 0,
                "retrieval_correct": 0,
                "answer_text_correct": 0,
                "fully_correct": 0,
            }
        doc_breakdown[doc]["total"] += 1
        if item["retrieval_correct"]:
            doc_breakdown[doc]["retrieval_correct"] += 1
        if item["answer_text_correct"]:
            doc_breakdown[doc]["answer_text_correct"] += 1
        if item["fully_correct"]:
            doc_breakdown[doc]["fully_correct"] += 1

    return summary, doc_breakdown, evaluation_results


def print_summary(summary, doc_breakdown):
    total_questions = summary["total_questions"]
    counts = summary["counts"]
    attempted = total_questions - summary["failed_to_answer"]

    print("\n" + "=" * 62)
    print("EVALUATION SUMMARY")
    print("=" * 62)
    print(f"  Total questions:              {total_questions}")
    print(f"  Failed to answer (None):      {summary['failed_to_answer']}")
    print(f"  Retrieval accuracy:           {summary['retrieval_accuracy_pct']}%  ({counts['retrieval_correct']}/{total_questions})")
    print(f"  Answer letter accuracy:       {summary['answer_letter_accuracy_pct']}%  ({counts['answer_letter_correct']}/{total_questions})")
    print(f"  Answer text accuracy:         {summary['answer_text_accuracy_pct']}%  ({counts['answer_text_correct']}/{total_questions})")
    print(f"  Attempted answer accuracy:    {summary['attempted_answer_accuracy_pct']}%  ({counts['answer_text_correct']}/{attempted})")
    print(f"  Fully correct (R + A):        {summary['fully_correct_pct']}%  ({counts['fully_correct']}/{total_questions})")

    print("\nPer-Document Breakdown:")
    for doc, stats in doc_breakdown.items():
        r = pct(stats["retrieval_correct"], stats["total"])
        a = pct(stats["answer_text_correct"], stats["total"])
        f = pct(stats["fully_correct"], stats["total"])
        print(f"  {doc[:52]:<52} R:{r:5.1f}%  A:{a:5.1f}%  Full:{f:5.1f}%  (n={stats['total']})")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhysioRAG generated answers.")
    parser.add_argument("--predictions", required=True, help="Path to evaluation_generated_answers.json")
    parser.add_argument(
        "--output",
        default="Evaluation_sets/eval_default.json",
        help="Output path for evaluation JSON",
    )
    args = parser.parse_args()

    generated_file = Path(args.predictions)
    if not generated_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {generated_file}")

    with open(generated_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    summary, doc_breakdown, evaluation_results = evaluate_results(results)
    print_summary(summary, doc_breakdown)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_document_breakdown": doc_breakdown,
                "per_question_results": evaluation_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
