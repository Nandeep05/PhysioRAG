import json
import argparse
from pathlib import Path
from collections import defaultdict

# ── Phrases that immediately disqualify a question ──────────────────
BAD_PHRASES = [
    # Direct source-leaking phrases
    "according to the provided context",
    "according to the provided text",
    "according to the provided passage",
    "according to the context",
    "according to the study",
    "according to the text",
    "according to the document",
    "according to the pdf",
    "according to the guideline",
    "according to the jospt",
    "according to the cpg",
    "according to the above text",
    "according to the above",
    "as mentioned in",
    "as described in the",
    "as stated in",
    "as outlined in",    # Dosage and metric-specific phrases
    "mg per",
    "mg/kg",
    "ml per",
    "units per",
    "dose of",
    "dosage of",
    "mg dose",
    "injection of",
    "injected with",
    "administered at",
    "prescribed at",
    "as indicated in",
    "as per the",
    "in the provided",
    "in the context",
    "in the passage",
    "in the given context",
    "in the given text",
    "the passage states",
    "the text mentions",
    "the provided text",
    "the provided passage",
    "based on the context",
    "based on the provided",
    "based on the passage",
    # Vague study references
    "colleagues' study",
    "colleagues study",
    "et al study",
    "the above study",
    "in the above",
    "in the following",
    "mentioned above",
    # Document name leakage
    ".pdf",
    ".PDF",
    # Vague pointers that make no sense standalone
    "this condition",
    "this method",
    "this technique",
    "this approach",
    "this lesion",
    "this treatment",
    "the lesion",
    "the method",
]

# ── Generic stems that are too vague without specific clinical detail ─
GENERIC_STEMS = [
    "which intervention is recommended for patients",
    "what is the recommended treatment for",
    "which treatment is used for",
    "which technique was used",
    "which method is applied",
    "which method was used",
]

# ── Target questions per document ────────────────────────────────────
TARGET_PER_DOC = 9


# ────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────

def check_bad_phrase(text: str):
    """Returns (is_bad, reason) if a banned phrase is found."""
    text_lower = text.lower()
    for phrase in BAD_PHRASES:
        if phrase.lower() in text_lower:
            return True, f"Contains banned phrase: '{phrase}'"
    return False, ""


def check_generic_stem(text: str):
    """Flags overly generic question stems lacking clinical specificity."""
    text_lower = text.lower()
    for stem in GENERIC_STEMS:
        if stem in text_lower:
            word_count = len(text.split())
            if word_count < 14:
                return True, f"Generic stem + too short ({word_count} words): '{stem}'"
    return False, ""


def check_gold_answer(q: dict):
    """
    Flags questions where the gold answer is too vague (< 2 words).
    Handles both formats:
      NEW: gold_answer = {"Answer": "d", "Text_Answer": "some phrase", ...}
      OLD: goldanswer  = "Answer c, TextAnswer some phrase"
    """
    # NEW format
    gold_new = q.get("gold_answer", None)
    if isinstance(gold_new, dict):
        text_answer = gold_new.get("Text_Answer", "")
        if text_answer and len(text_answer.split()) < 2:
            return True, f"Gold answer too vague (single word): '{text_answer}'"
        return False, ""

    # OLD format
    gold_old = q.get("goldanswer", "")
    if isinstance(gold_old, dict):
        text_answer = gold_old.get("TextAnswer", "")
    elif "TextAnswer" in str(gold_old):
        parts = str(gold_old).split("TextAnswer")
        text_answer = parts[-1].strip() if len(parts) > 1 else ""
    else:
        text_answer = str(gold_old)

    if text_answer and len(text_answer.split()) < 2:
        return True, f"Gold answer too vague (single word): '{text_answer}'"
    return False, ""


def check_duplicate_options(q: dict):
    """
    Flags questions where two or more options are identical.
    Handles both formats:
      NEW: options = {"a": "...", "b": "...", ...}
      OLD: options = ["opt1", "opt2", ...]
    """
    options = q.get("options", None)
    if not options:
        return False, ""

    if isinstance(options, dict):
        option_values = list(options.values())
    elif isinstance(options, list):
        option_values = options
    else:
        return False, ""

    seen = set()
    for opt in option_values:
        opt_clean = str(opt).strip().lower()
        if opt_clean in seen:
            return True, f"Duplicate option detected: '{opt}'"
        seen.add(opt_clean)
    return False, ""

def check_clinical_specifics(q: dict):
    """
    Flags questions that rely on dosage values, injection amounts,
    or specific metrics that vary by patient — these make poor
    standalone evaluation questions.
    """
    import re

    question_text = q.get("question", "")

    # Check answer options too — not just the question stem
    options = q.get("options", {})
    if isinstance(options, dict):
        all_text = question_text + " " + " ".join(options.values())
    elif isinstance(options, list):
        all_text = question_text + " " + " ".join(options)
    else:
        all_text = question_text

    all_text_lower = all_text.lower()

    # Pattern 1 — numeric dosage values (e.g. "40 mg", "10 ml", "500 units")
    dosage_pattern = re.compile(
        r'\b\d+\s*(mg|ml|mcg|iu|units?|mmol|cc|g/kg|mg/kg|mg/ml)\b',
        re.IGNORECASE
    )
    if dosage_pattern.search(all_text):
        return True, "Contains specific dosage value (varies by patient)"

    # Pattern 2 — specific injection/drug names with quantities
    injection_pattern = re.compile(
        r'\b(cortisone|corticosteroid|triamcinolone|methylprednisolone|'
        r'lidocaine|bupivacaine|hyaluronic acid|platelet.rich plasma|prp)\s+'
        r'(injection|dose|dosage|administered|prescribed)',
        re.IGNORECASE
    )
    if injection_pattern.search(all_text_lower):
        return True, "Contains specific injection/drug dosage reference"

    # Pattern 3 — gold answer is purely a number or dosage
    gold = q.get("gold_answer", {})
    if isinstance(gold, dict):
        text_answer = gold.get("Text_Answer", "")
    else:
        text_answer = str(q.get("goldanswer", ""))

    # If gold answer is just a number like "40" or "2 weeks" or "500 mg"
    pure_number = re.compile(r'^\d+(\.\d+)?\s*(mg|ml|weeks?|months?|days?|units?)?$')
    if pure_number.match(text_answer.strip().lower()):
        return True, f"Gold answer is a bare numeric/dosage value: '{text_answer}'"

    return False, ""



def get_document(q: dict) -> str:
    """
    Extract document name from question dict.
    Handles both formats:
      NEW: gold_answer.document
      OLD: top-level document key
    """
    if "document" in q:
        return q["document"]
    gold = q.get("gold_answer", {})
    if isinstance(gold, dict):
        return gold.get("document", "unknown")
    return "unknown"


def get_chunk_id(q: dict):
    """
    Extract chunk ID from question dict.2
    Handles both formats:
      NEW: gold_answer.chunk_id
      OLD: top-level chunkid key
    """
    if "chunkid" in q:
        return q["chunkid"]
    gold = q.get("gold_answer", {})
    if isinstance(gold, dict):
        return gold.get("chunk_id", "unknown")
    return "unknown"


def get_fully_correct(q: dict) -> bool:
    """
    Extract fully_correct flag from question dict.
    Handles both old (fullycorrect) and new (fully_correct) key names.
    """
    if "fully_correct" in q:
        return bool(q["fully_correct"])
    if "fullycorrect" in q:
        return bool(q["fullycorrect"])
    return False


def get_answer_length(q: dict) -> int:
    """
    Get word count of gold answer text — used for sorting.
    Longer answers are more specific and preferred.
    """
    gold = q.get("gold_answer", {})
    if isinstance(gold, dict):
        return len(gold.get("Text_Answer", "").split())
    gold_old = str(q.get("goldanswer", ""))
    if "TextAnswer" in gold_old:
        text = gold_old.split("TextAnswer")[-1].strip()
        return len(text.split())
    return 0


def detect_structure(data):
    """
    Detect the JSON structure and return (questions_list, structure_label).
    Handles all known formats.
    """
    if isinstance(data, list):
        return data, "list"
    if isinstance(data, dict):
        if "perquestionresults" in data:
            return data["perquestionresults"], "eval_report_old"
        if "per_question_results" in data:
            return data["per_question_results"], "eval_report_new"
        if "questions" in data:
            return data["questions"], "questions_key"
        if "question" in data and "gold_answer" in data:
            return [data], "single_question"
    return None, None


def is_good_question(q: dict):
    question_text = q.get("question", "")

    # Check 1 — bad phrases in question stem
    bad, reason = check_bad_phrase(question_text)
    if bad:
        return False, reason

    # Check 2 — overly generic stem
    generic, reason = check_generic_stem(question_text)
    if generic:
        return False, reason

    # Check 3 — question too short
    if len(question_text.split()) < 8:
        return False, f"Question too short ({len(question_text.split())} words)"

    # Check 4 — gold answer too vague
    vague, reason = check_gold_answer(q)
    if vague:
        return False, reason

    # Check 5 — duplicate answer options
    dup, reason = check_duplicate_options(q)
    if dup:
        return False, reason

    # Check 6 — dosage values and patient-specific metrics    ← ADD THIS
    dosage, reason = check_clinical_specifics(q)
    if dosage:
        return False, reason

    return True, ""



# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Filter low-quality questions from eval JSON sets"
    )
    parser.add_argument(
        "--input_dir",
        default="/home/hpc/iwso/iwso221h/Shoulder-RAG-HPC/src/evaluation",
        help="Directory containing eval JSON files"
    )
    parser.add_argument(
        "--output",
        default="/home/hpc/iwso/iwso221h/Shoulder-RAG-HPC/src/evaluation/eval_combined_clean.json",
        help="Output combined filtered JSON file"
    )
    parser.add_argument(
        "--target_per_doc",
        type=int,
        default=TARGET_PER_DOC,
        help="Maximum questions to keep per document (default: 9)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # ── Find all eval JSON files ───────────────────────────────────
    eval_files = list(input_dir.glob("eval_*.json"))
    eval_files = [f for f in eval_files if "_clean" not in f.name]

    if not eval_files:
        print(f"No eval_*.json files found in {input_dir}")
        exit(1)

    print(f"\nFound {len(eval_files)} eval files:")
    for f in eval_files:
        print(f"  - {f.name}")

    # ── Loop through all files and collect good questions ──────────
    all_good = []
    all_rejected = []
    seen_questions = set()

    for eval_file in sorted(eval_files):
        print(f"\nProcessing: {eval_file.name}")

        # Load JSON safely
        try:
            with open(eval_file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Skipping — invalid JSON: {e}")
            continue

        # Detect structure
        questions, structure = detect_structure(data)
        if questions is None:
            print(f"  Skipping — unrecognised structure")
            continue

        print(f"  Structure: {structure} | {len(questions)} questions")

        file_good = 0
        file_rejected = 0

        for q in questions:
            # Normalise document field to top level
            if "document" not in q:
                q["document"] = get_document(q)

            ok, reason = is_good_question(q)

            if not ok:
                all_rejected.append({
                    "source_file": eval_file.name,
                    "question": q.get("question", "")[:120],
                    "reason": reason,
                    "chunkid": get_chunk_id(q),
                    "document": get_document(q),
                })
                file_rejected += 1
                continue

            # Deduplicate across files
            q_text = q.get("question", "").strip().lower()
            if q_text in seen_questions:
                all_rejected.append({
                    "source_file": eval_file.name,
                    "question": q.get("question", "")[:120],
                    "reason": "Duplicate question (already in combined set)",
                    "chunkid": get_chunk_id(q),
                    "document": get_document(q),
                })
                file_rejected += 1
                continue

            seen_questions.add(q_text)
            q["source_file"] = eval_file.name
            all_good.append(q)
            file_good += 1

        print(f"  Kept: {file_good}  |  Rejected: {file_rejected}")

    # ── Filter report before capping ──────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  FILTER REPORT (before per-document cap)")
    print(f"{'=' * 60}")
    print(f"  Files processed : {len(eval_files)}")
    print(f"  Total questions : {len(all_good) + len(all_rejected)}")
    print(f"  Passed filter   : {len(all_good)}")
    print(f"  Rejected        : {len(all_rejected)}")
    print(f"{'=' * 60}")

    if all_rejected:
        print(f"\n  Rejected Questions:")
        for i, r in enumerate(all_rejected, 1):
            print(f"\n  {i}. [{r['reason']}]")
            print(f"     file={r['source_file']} | chunk={r['chunkid']}")
            print(f"     Q: {r['question']}...")

    # ── Apply per-document cap ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  APPLYING PER-DOCUMENT CAP ({args.target_per_doc} per document)")
    print(f"{'=' * 60}")

    by_doc = defaultdict(list)
    for q in all_good:
        by_doc[get_document(q)].append(q)

    final_questions = []

    for doc, qs in sorted(by_doc.items()):
        # Sort preference:
        # 1. Fully correct first (RAG answered correctly)
        # 2. Longer gold answer (more specific question)
        qs_sorted = sorted(
            qs,
            key=lambda q: (not get_fully_correct(q), -get_answer_length(q))
        )
        selected = qs_sorted[:args.target_per_doc]
        final_questions.extend(selected)
        print(f"  {doc[:52]:<52} {len(qs):>3} available → {len(selected):>2} selected")

    # ── Final summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  FINAL COMBINED SET")
    print(f"{'=' * 60}")
    print(f"  Total questions : {len(final_questions)}")
    print(f"  Documents       : {len(by_doc)}")

    doc_final_counts = defaultdict(int)
    for q in final_questions:
        doc_final_counts[get_document(q)] += 1

    print(f"\n  Per-Document Breakdown (final set):")
    for doc, count in sorted(doc_final_counts.items()):
