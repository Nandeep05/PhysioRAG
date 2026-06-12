"""
Generate gold standard evaluation questions using the PhysioRAG pipeline.

Supports:
- Multiple LLM models via CLI
- Customizable number of questions
- Stratified sampling per document
- Environment-aware paths
- Checkpoint resume capability
"""

import json
import re
import pandas as pd
import datetime
import os
import random
import logging
import argparse
from src.rag.prompts import EVAL_GEN_PROMPT
from src.rag.llm_provider import get_llm_provider
from config import (
    OLLAMA_LLM_MODEL,
    OLLAMA_BASE_URL,
    HF_LLM_MODEL,
    VLLM_MODEL_ID,
    VLLM_BASE_URL,
    LLM_PROVIDER,
    CANDIDATE_CHUNKS_PATH,
    RESULTS_DIR,
    LOG_LEVEL,
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CLINICALLY STABLE FACTUAL QUESTION STYLES ---
FACTUAL_PATIENT_SAMPLES = [
    "Which nerve root is most frequently compressed in cervical disc protrusion causing shoulder pain?",
    "Which tendon is commonly involved in rotator cuff tendinopathy?",
    "Which stage of adhesive capsulitis is characterized by progressive stiffness?",
    "Which anatomical structure is affected in subacromial impingement syndrome?",
    "Which symptom pattern is typical of nerve root compression in the cervical spine?",
    "Which clinical feature distinguishes referred shoulder pain from primary shoulder pathology?",
    "Which structure limits pain spread in a posterocentral cervical disc lesion?",
    "Which muscle is responsible for initiating shoulder abduction?",
    "Which classification is used to describe stages of adhesive capsulitis?",
    "Which pathological process underlies cervical radiculopathy?"
]

MAX_OPTION_WORDS = 12

# Minimum questions required per source document for balanced evaluation
# 9 × 6 documents = 54 = QUESTION_COUNT default in job_vllm_pipeline.sh
MIN_QUESTIONS_PER_DOC = 9

# Maximum retries per chunk when question generation/validation fails
MAX_RETRIES_PER_SLOT = 3

# Similarity threshold: options whose normalized texts are >80% overlapping are considered duplicates
FUZZY_DEDUP_THRESHOLD = 0.80


def safe_model_tag(model_name: str) -> str:
    """Convert model names to filesystem-safe tags for output filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase for comparison purposes."""
    return re.sub(r'\s+', ' ', text.strip()).lower()


def _token_set(text: str) -> set:
    """Return set of lowercase alpha-numeric tokens for fuzzy comparison."""
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def _fuzzy_similar(a: str, b: str) -> float:
    """Token-set Jaccard similarity between two option strings."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_substring_of(short: str, long: str) -> bool:
    """Check if normalised 'short' is contained inside normalised 'long'."""
    ns, nl = normalize_text(short), normalize_text(long)
    if ns == nl:
        return False  # exact match is not a substring issue
    return ns in nl or nl in ns


def extract_options(question_text: str) -> list:
    """
    Robustly extracts options from a question string.
    Handles formats: 'a. ', 'a) ', 'A. ', '(a) '
    Returns a list of option strings (text only, no prefix).
    """
    options = []
    for line in question_text.split("\n")[1:]:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^[\(\[]?[a-eA-E][\.\)\]]\s*(.+)', line)
        if match:
            options.append(match.group(1).strip())
    return options


def rebuild_question(stem: str, options: list, correct_answer_text: str) -> tuple:
    """
    Builds Options dict {a-e: text} and randomizes correct answer position.

    Quality gates applied in order:
    1. Drop options over MAX_OPTION_WORDS (keep correct answer regardless).
    2. Exact case-insensitive deduplication.
    3. Fuzzy near-duplicate removal (token-set Jaccard > threshold).
    4. Superset/substring removal: if option A contains option B, drop the longer one
       (unless one of them is the correct answer).
    5. Pad with clinically-neutral filler *only as last resort* (never placeholder text).

    Returns: stem (str), options_dict (dict), answer_letter (str)
            or (None, None, None) if we cannot build 5 valid distinct options.
    """
    correct_norm = normalize_text(correct_answer_text)

    # --- Step 1: Drop options that are too long (keep correct answer always) ---
    filtered = []
    for o in options:
        if normalize_text(o) == correct_norm:
            filtered.append(o)
        elif len(o.split()) <= MAX_OPTION_WORDS:
            filtered.append(o)

    # --- Step 2: Exact case-insensitive deduplication ---
    seen_norm = set()
    deduped = []
    for o in filtered:
        norm = normalize_text(o)
        if norm not in seen_norm:
            seen_norm.add(norm)
            deduped.append(o)

    # --- Step 3: Fuzzy near-duplicate removal ---
    # Keep the first occurrence; drop later options that are too similar to any kept one.
    fuzzy_clean = []
    for o in deduped:
        is_dup = False
        for kept in fuzzy_clean:
            if _fuzzy_similar(o, kept) >= FUZZY_DEDUP_THRESHOLD:
                # Always keep the correct answer
                if normalize_text(o) == correct_norm:
                    # Drop the *kept* one instead and replace with correct
                    if normalize_text(kept) != correct_norm:
                        fuzzy_clean.remove(kept)
                        break
                else:
                    is_dup = True
                    break
        if not is_dup:
            fuzzy_clean.append(o)

    # --- Step 4: Superset / substring removal ---
    # If option A text contains option B text, drop the longer one (keep shorter, more concise).
    # Exception: always keep the correct answer.
    clean = list(fuzzy_clean)
    to_remove = set()
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            if _is_substring_of(clean[i], clean[j]):
                # One contains the other — drop the longer one (unless it is the correct answer)
                long_idx = i if len(clean[i]) > len(clean[j]) else j
                short_idx = j if long_idx == i else i
                if normalize_text(clean[long_idx]) == correct_norm:
                    to_remove.add(short_idx)  # keep the correct answer even if longer
                else:
                    to_remove.add(long_idx)

    clean = [o for idx, o in enumerate(clean) if idx not in to_remove]

    options = clean

    # Ensure correct answer is present
    if not any(normalize_text(o) == correct_norm for o in options):
        options = [correct_answer_text] + options

    # --- Step 5: Pad to 5 if needed (use neutral clinical filler, NOT placeholder text) ---
    _neutral_fillers = [
        "Not applicable in this clinical context",
        "Insufficient evidence to determine",
        "None of the listed options",
        "All of the above equally",
        "Cannot be determined from available data",
        "No specific recommendation available",
        "Clinical judgement required",
    ]
    filler_idx = 0
    while len(options) < 5 and filler_idx < len(_neutral_fillers):
        candidate = _neutral_fillers[filler_idx]
        filler_idx += 1
        # Ensure filler is not a near-dup of existing options
        if not any(_fuzzy_similar(candidate, o) >= FUZZY_DEDUP_THRESHOLD for o in options):
            options.append(candidate)

    # Absolute fallback (should never be needed, but prevents crash)
    while len(options) < 5:
        options.append(f"Option {len(options) + 1}")

    options = options[:5]

    # --- Step 6: Final validation — reject if options still contain near-duplicates ---
    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            if _fuzzy_similar(options[i], options[j]) >= FUZZY_DEDUP_THRESHOLD:
                logger.warning("Options still have near-duplicate after cleanup — rejecting question.")
                return None, None, None
            if _is_substring_of(options[i], options[j]):
                logger.warning("Options still have superset/subset after cleanup — rejecting question.")
                return None, None, None

    # --- Step 7: Randomize correct answer position ---
    correct_idx = next(i for i, o in enumerate(options) if normalize_text(o) == correct_norm)
    new_position = random.randint(0, 4)
    options[correct_idx], options[new_position] = options[new_position], options[correct_idx]

    answer_letter = chr(97 + next(i for i, o in enumerate(options) if normalize_text(o) == correct_norm))
    options_dict = {chr(97 + i): opt for i, opt in enumerate(options)}

    return stem.strip(), options_dict, answer_letter


def generate_factual_question(chunk: dict, llm_provider, max_option_words: int = MAX_OPTION_WORDS):
    """Generate a single factual question from a chunk."""
    chunk_text = chunk["chunk_text"]
    section_title = chunk.get("section_title", "Clinical Section")
    document_name = chunk.get("document", "Unknown Document")
    page_number = chunk.get("page_number", "N/A")
    chunk_id = chunk.get("chunk_id", "Unknown")

    style_samples = random.sample(FACTUAL_PATIENT_SAMPLES, 3)
    samples_str = "\n".join([f"- {s}" for s in style_samples])

    prompt = f"""{EVAL_GEN_PROMPT}

You are generating a GOLD STANDARD evaluation MCQ for a RAG system focused on shoulder pain and related disorders.

The question stem will be used as the SEARCH QUERY to retrieve the relevant passage.
Therefore the stem MUST contain specific, distinctive terms (condition name, structure, or phrase from the text)
so that a retriever can find this exact chunk. Generic stems lead to wrong or missing retrieval and lower accuracy.

REQUIRED — Question stem MUST:
- Explicitly name the condition, structure, or disorder (e.g. adhesive capsulitis, rotator cuff tendinopathy, subacromial bursitis, SAP).
- If the text refers to a specific study or author (e.g. Vermeulen et al, Johnson et al), include that in the stem so the question is both standalone and retrievable.
- Be fully standalone — no vague references.

FORBIDDEN in the question stem (never use these without naming the condition/context):
- "the study" / "used in the study" / "in the study" without naming the condition or authors
- "the lesion" / "site of the lesion" / "the lesion site" without naming the condition (e.g. subacromial bursitis)
- "this condition" / "this text" / "this method" / "this technique"
- "Which method is applied..." or "Which technique was used..." without specifying for which condition or in which context

Other rules:
- Be clinically meaningful and strictly answerable from the provided text.
- Be based on stable clinical concepts (anatomy, pathophysiology, symptom patterns, classification, mechanisms).
- NOT rely on measurements, numerical thresholds, percentages, durations, frequency, or dosage.
- NOT be patient-specific. NOT use NOT or EXCEPT phrasing (positive questions only).

DISTRACTOR RULES (STRICTLY ENFORCE):
- All 5 options must be the SAME semantic type (all structures, all muscles, all phases, etc.)
- Each option must be a SHORT clinical term or phrase — maximum {max_option_words} words.
- Do NOT copy full sentences from the context as options.
- Do NOT create two options that differ only in capitalization, articles (a/the), or minor rewording.
- Do NOT create an option that is a SUBSET or SUPERSET of another option.
  BAD example: option a = "lower trapezius activity" AND option d = "Lower trapezius" (one contains the other).
  BAD example: option b = "adults with shoulder pain" AND option d = "Participants with shoulder pain" (same meaning).
- Do NOT embed the correct answer in the question stem.
- Do NOT use placeholder text or "None of the above".
- Each option must be clearly DISTINCT from every other option — a domain expert should never be confused about which of two options to pick because they mean the same thing.

Use these example question styles as guidance:
{samples_str}

SECTION TITLE:
{section_title}

SOURCE DOCUMENT:
{document_name}

CLINICAL TEXT:
{chunk_text}

Return ONLY valid JSON using EXACTLY this schema:

{{
    "Section_Type": "Anatomy/Assessment/Treatment/Pathophysiology/Clinical Features",
    "Question": "Question stem text only (no options here)",
    "Options": {{
        "a": "Short option text",
        "b": "Short option text",
        "c": "Short option text",
        "d": "Short option text",
        "e": "Short option text"
    }},
    "Correct_Answer": "c",
    "Text_Answer": "Exact short phrase copied from context",
    "Reference": "{document_name}",
    "Page": "{page_number}",
    "Context": "Exact snippet used",
    "Complexity": "basic/intermediate/advanced"
}}

JSON OUTPUT:
"""

    try:
        response = llm_provider.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        raw_content = response

        # Robustly extract JSON
        if "```json" in raw_content:
            json_str = raw_content.split("```json")[-1].split("```")[0].strip()
        elif "```" in raw_content:
            json_str = raw_content.split("```")[1].strip()
        else:
            start_idx = raw_content.find("{")
            end_idx = raw_content.rfind("}") + 1
            json_str = raw_content[start_idx:end_idx]

        mcq = json.loads(json_str)

        # Normalize field names
        if "Text Answer" in mcq and "Text_Answer" not in mcq:
            mcq["Text_Answer"] = mcq.pop("Text Answer")
        if "Answer" in mcq and "Correct_Answer" not in mcq:
            mcq["Correct_Answer"] = mcq.pop("Answer")

        correct_answer_text = mcq.get("Text_Answer", "").strip()
        if not correct_answer_text:
            logger.warning(f"Empty Text_Answer for chunk {chunk_id}, skipping.")
            return None

        # Handle both output formats
        if "Options" in mcq and isinstance(mcq["Options"], dict) and len(mcq["Options"]) >= 3:
            options_list = [mcq["Options"].get(chr(97 + i), "") for i in range(5)]
            options_list = [o for o in options_list if o]
            stem = mcq["Question"].strip()
        else:
            lines = mcq.get("Question", "").strip().splitlines()
            stem = lines[0].strip() if lines else "Question stem missing"
            options_list = extract_options(mcq.get("Question", ""))

        if len(options_list) < 3:
            logger.warning(f"Too few options ({len(options_list)}) for chunk {chunk_id}, skipping.")
            return None

        # Rebuild with all fixes applied (fuzzy dedup, superset removal, etc.)
        stem, options_dict, answer_letter = rebuild_question(stem, options_list, correct_answer_text)

        if stem is None:
            logger.warning(f"Option quality check failed for chunk {chunk_id}, skipping.")
            return None

        # Final validations
        if normalize_text(correct_answer_text) not in normalize_text(chunk_text):
            logger.warning(f"Grounding fail for chunk {chunk_id}")
            return None

        if "NOT" in stem or "EXCEPT" in stem.upper():
            logger.warning(f"NOT/EXCEPT question detected for chunk {chunk_id}, skipping.")
            return None

        stem_lower = stem.lower()
        forbidden_phrases = [
            "used in the study",
            "in the study?",
            "site of the lesion",
            "the lesion?",
            "which method is applied",
        ]
        if any(phrase in stem_lower for phrase in forbidden_phrases):
            logger.warning(f"Vague stem for chunk {chunk_id}, skipping.")
            return None

        complexity = mcq.get("Complexity", "basic") or "basic"

        result = {
            "chunk_id": chunk_id,
            "Section_Type": mcq.get("Section_Type", "Clinical Features"),
            "Question": stem,
            "Options": options_dict,
            "Correct_Answer": answer_letter,
            "Text_Answer": correct_answer_text,
            "Reference": mcq.get("Reference", document_name),
            "Page": mcq.get("Page", page_number),
            "Context": mcq.get("Context", ""),
            "Complexity": complexity
        }

        return result

    except Exception as e:
        logger.error(f"Error generating question for {document_name} (chunk {chunk_id}): {e}")
        return None


def generate_factual_gold_standard(
    candidate_file: str = CANDIDATE_CHUNKS_PATH,
    output_dir: str = None,
    target_questions: int = 50,
    model_name: str = OLLAMA_LLM_MODEL,
    provider_type: str = LLM_PROVIDER,
    skip_existing: bool = True
):
    """
    Generate gold standard questions from candidate chunks.

    Key guarantees:
    - At least MIN_QUESTIONS_PER_DOC questions per source document.
    - If a chunk fails validation, retry with another chunk from the same document
      (up to MAX_RETRIES_PER_SLOT times).
    - Final count targets exactly `target_questions`.
    """
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "gold_standard")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(candidate_file):
        logger.error(f"Could not find {candidate_file}. Ensure indexing has run.")
        raise FileNotFoundError(f"Candidate chunks file not found: {candidate_file}")

    with open(candidate_file, "r", encoding="utf-8") as f:
        candidate_chunks = json.load(f)

    # Filter non-clinical sections
    excluded_keywords = ["reference", "bibliography", "chapter contents", "source"]
    candidate_chunks = [
        c for c in candidate_chunks
        if not any(kw in c.get("section_title", "").lower() for kw in excluded_keywords)
    ]

    # Skip already-generated chunk_ids (only if requested)
    existing_chunk_ids = set()
    if skip_existing:
        for fname in os.listdir(output_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                        existing = json.load(f)
                        for q in existing:
                            if "chunk_id" in q:
                                existing_chunk_ids.add(str(q["chunk_id"]))
                except Exception:
                    pass

        if existing_chunk_ids:
            before = len(candidate_chunks)
            candidate_chunks = [
                c for c in candidate_chunks
                if str(c.get("chunk_id", "")) not in existing_chunk_ids
            ]
            skipped = before - len(candidate_chunks)
            if skipped:
                logger.info(f"⏭️  Skipped {skipped} already-generated chunks.")

    if not candidate_chunks:
        logger.info("All chunks have already been used. No new questions to generate.")
        return

    df = pd.DataFrame(candidate_chunks)
    total_chunks = len(df)
    unique_docs = df["document"].unique().tolist()
    doc_counts = df["document"].value_counts()

    logger.info(f"🎯 Stratified Sampling | Available: {total_chunks} | Target: {target_questions} | Documents: {len(unique_docs)}")

    # ------------------------------------------------------------------
    # Quota logic: guarantee at least MIN_QUESTIONS_PER_DOC per document,
    # then distribute remaining quota proportionally.
    # ------------------------------------------------------------------
    per_doc_quota = {}
    guaranteed_total = 0

    for doc in unique_docs:
        available = doc_counts[doc]
        guarantee = min(MIN_QUESTIONS_PER_DOC, available)
        per_doc_quota[doc] = guarantee
        guaranteed_total += guarantee

    remaining = max(0, target_questions - guaranteed_total)

    if remaining > 0:
        # Distribute remaining quota proportionally by chunk count
        docs_with_headroom = {
            doc: doc_counts[doc] - per_doc_quota[doc]
            for doc in unique_docs
            if doc_counts[doc] - per_doc_quota[doc] > 0
        }
        total_headroom = sum(docs_with_headroom.values())

        if total_headroom > 0:
            for doc, headroom in docs_with_headroom.items():
                extra = min(headroom, round(headroom / total_headroom * remaining))
                per_doc_quota[doc] += extra

    # Final adjustment to exactly hit target
    current_total = sum(per_doc_quota.values())
    diff = target_questions - current_total
    if diff > 0:
        for doc in doc_counts.index:
            if diff == 0:
                break
            headroom = doc_counts[doc] - per_doc_quota[doc]
            add = min(diff, headroom)
            per_doc_quota[doc] += add
            diff -= add

    logger.info(f"   Per-document quota: {per_doc_quota}")
    total_targeted = sum(per_doc_quota.values())
    logger.info(f"   Total targeted: {total_targeted} questions")

    # ------------------------------------------------------------------
    # Build per-document chunk pools (shuffled for random sampling)
    # ------------------------------------------------------------------
    doc_pools = {}
    for doc in unique_docs:
        pool = df[df["document"] == doc].to_dict("records")
        random.shuffle(pool)
        doc_pools[doc] = pool

    # ------------------------------------------------------------------
    # Initialize LLM provider
    # ------------------------------------------------------------------
    _url_map = {"ollama": OLLAMA_BASE_URL, "vllm": VLLM_BASE_URL}
    llm_provider = get_llm_provider(
        model_name=model_name,
        provider_type=provider_type,
        base_url=_url_map.get(provider_type),
    )

    # ------------------------------------------------------------------
    # Generate questions with retry logic
    # ------------------------------------------------------------------
    final_dataset = []
    total_attempts = 0
    total_slots = total_targeted
    generated_chunk_ids = set()  # prevent duplicate chunk usage within this run

    for doc, quota in per_doc_quota.items():
        pool = [c for c in doc_pools[doc] if str(c.get("chunk_id", "")) not in generated_chunk_ids]
        pool_idx = 0
        doc_generated = 0

        while doc_generated < quota and pool_idx < len(pool):
            chunk = pool[pool_idx]
            chunk_id = str(chunk.get("chunk_id", ""))
            pool_idx += 1

            # Skip if we somehow already used this chunk
            if chunk_id in generated_chunk_ids:
                continue

            total_attempts += 1
            slot_num = len(final_dataset) + 1
            logger.info(f"[{slot_num}/{total_slots}] {doc} | {chunk.get('section_title', '')[:60]}")

            retries = 0
            success = False
            while retries < MAX_RETRIES_PER_SLOT and not success:
                mcq = generate_factual_question(chunk, llm_provider)
                if mcq:
                    final_dataset.append(mcq)
                    generated_chunk_ids.add(chunk_id)
                    doc_generated += 1
                    success = True
                else:
                    retries += 1
                    if retries < MAX_RETRIES_PER_SLOT:
                        logger.info(f"   ↩️  Retry {retries}/{MAX_RETRIES_PER_SLOT} for chunk {chunk_id}")

            if not success:
                logger.info(f"   ⏭️  Chunk {chunk_id} failed after {MAX_RETRIES_PER_SLOT} attempts, trying next chunk")

        if doc_generated < quota:
            logger.warning(f"⚠️  {doc}: only generated {doc_generated}/{quota} questions (pool exhausted)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    model_tag = safe_model_tag(model_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"gold_eval_{model_tag}_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)

    rate = len(final_dataset) / total_attempts * 100 if total_attempts else 0
    logger.info(f"✅ {len(final_dataset)}/{total_targeted} MCQs generated ({rate:.1f}% success rate)")
    logger.info(f"📁 Saved to: {output_path}")

    # Per-document summary
    doc_summary = {}
    for q in final_dataset:
        doc = q.get("Reference", "Unknown")
        doc_summary[doc] = doc_summary.get(doc, 0) + 1
    logger.info(f"📊 Per-document: {doc_summary}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate gold standard evaluation questions from candidate chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 questions with default model
  python generate_questions.py --count 50

  # Use specific model
  python generate_questions.py --count 50 --model qwen2.5:7b

  # Custom output directory
  python generate_questions.py --count 50 --output /path/to/results

  # Don't skip existing questions
  python generate_questions.py --count 50 --no-skip-existing
        """
    )

    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=50,
        help="Target number of questions to generate (default: 50)"
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
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=CANDIDATE_CHUNKS_PATH,
        help="Path to candidate chunks JSON file"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_PROVIDER,
        choices=["ollama", "vllm", "hf", "mock"],
        help=f"LLM provider to use (default: {LLM_PROVIDER})"
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip chunks that have already been used"
    )

    args = parser.parse_args()

    if not args.model:
        if args.provider == "hf":
            args.model = HF_LLM_MODEL
        elif args.provider == "vllm":
            args.model = VLLM_MODEL_ID
        else:
            args.model = OLLAMA_LLM_MODEL

    logger.info(f"Starting question generation with arguments: {vars(args)}")

    try:
        output_path = generate_factual_gold_standard(
            candidate_file=args.input,
            output_dir=args.output,
            target_questions=args.count,
            model_name=args.model,
            provider_type=args.provider,
            skip_existing=not args.no_skip_existing
        )
        logger.info(f"Successfully generated questions: {output_path}")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
