import json
from fuzzywuzzy import fuzz
from pathlib import Path

# -----------------------------
# 1️⃣ Load Evaluation Data
# -----------------------------
generated_answers_file = Path(
    r"D:\College\FAU_Notes\4th_sem\Graph_RAG_Project\PhysioRAG_pipeline\src\evaluation\evaluation_generated_answers.json"
)

gold_standard_file = Path(
    r"D:\College\FAU_Notes\4th_sem\Graph_RAG_Project\PhysioRAG_pipeline\data\gold_standard\factual_eval_qwen_20260220_031604.json"
)

with open(generated_answers_file, "r", encoding="utf-8") as f:
    generated_answers = json.load(f)

with open(gold_standard_file, "r", encoding="utf-8") as f:
    gold_standard = json.load(f)

# -----------------------------
# 2️⃣ Text Normalization Function
# -----------------------------
def normalize(text):
    """
    Normalize text by:
    - Lowercasing
    - Removing extra whitespace
    """
    return " ".join(text.strip().lower().split())


# -----------------------------
# 3️⃣ Faithfulness Evaluation Function
# -----------------------------
def evaluate_faithfulness(generated_answers, gold_standard, threshold=75):
    """
    Compare generated answers with gold standard answers using
    improved fuzzy matching suitable for RAG outputs.
    """

    results = []
    faithful_count = 0

    for generated, gold in zip(generated_answers, gold_standard):
        question = generated["question"]
        generated_answer = generated["generated_answer"]
        gold_answer = gold["Text Answer"]

        gen_norm = normalize(generated_answer)
        gold_norm = normalize(gold_answer)

        # Use more robust fuzzy comparison
        match_score = max(
            fuzz.token_set_ratio(gen_norm, gold_norm),
            fuzz.partial_ratio(gen_norm, gold_norm),
            fuzz.token_sort_ratio(gen_norm, gold_norm)
        )

        is_faithful = match_score >= threshold

        if is_faithful:
            faithful_count += 1

        results.append({
            "question": question,
            "generated_answer": generated_answer,
            "gold_answer": gold_answer,
            "match_score": match_score,
            "is_faithful": is_faithful
        })

    total = len(results)
    accuracy = (faithful_count / total) * 100 if total > 0 else 0

    print("\n📊 Faithfulness Evaluation Summary")
    print(f"Total Questions: {total}")
    print(f"Faithful Answers: {faithful_count}")
    print(f"Faithfulness Accuracy: {accuracy:.2f}%\n")

    return results


# -----------------------------
# 4️⃣ Run Faithfulness Evaluation
# -----------------------------
faithfulness_results = evaluate_faithfulness(
    generated_answers,
    gold_standard,
    threshold=75  # Adjust if needed
)

# -----------------------------
# 5️⃣ Save Results
# -----------------------------
output_file = Path(
    r"D:\College\FAU_Notes\4th_sem\Graph_RAG_Project\PhysioRAG_pipeline\src\evaluation\faithfulness_evaluation_results.json"
)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(faithfulness_results, f, indent=2, ensure_ascii=False)

print(f"✅ Faithfulness evaluation results saved to: {output_file}")
