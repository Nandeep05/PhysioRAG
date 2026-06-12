import json
import re
import hashlib
import argparse
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MIN_WORDS = 50                # minimum word count for a usable chunk
NEAR_DUP_THRESHOLD = 0.85     # Jaccard similarity above which a chunk is a near-dup
MIN_ALPHA_RATIO = 0.55        # fraction of chars that must be letters or spaces
MIN_SENTENCES = 2             # minimum sentence-ending punctuation marks (.  !  ?)

# Section titles that contain any of these strings are dropped
BLOCKED_TITLE_KEYWORDS = {
    "reference", "bibliography", "chapter contents", "source",
    "index", "table of contents", "contents", "acknowledgement",
    "acknowledgment", "appendix", "figures", "tables",
}
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Collapse whitespace and lowercase."""
    return re.sub(r"\s+", " ", text.strip().lower())


def token_set(text: str) -> set:
    """Return set of lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity between two strings."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def alpha_ratio(text: str) -> float:
    """Fraction of characters that are alphabetic or whitespace."""
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha() or c.isspace()) / len(text)


def sentence_count(text: str) -> int:
    """Count sentence-ending punctuation marks."""
    return len(re.findall(r"[.!?]", text))


def chunk_hash(text: str) -> str:
    """MD5 hash of the normalised text."""
    return hashlib.md5(normalize(text).encode()).hexdigest()


def is_blocked_title(title: str) -> bool:
    """Return True if the section title contains any blocked keyword."""
    t = title.strip().lower()
    return any(kw in t for kw in BLOCKED_TITLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Filter candidate_chunks.json → candidate_chunks_clean.json"
    )
    parser.add_argument(
        "--input",
        default="data/chunks/candidate_chunks.json",
        help="Path to raw candidate chunks (default: data/chunks/candidate_chunks.json)",
    )
    parser.add_argument(
        "--output",
        default="data/chunks/candidate_chunks_clean.json",
        help="Path to write filtered chunks (default: data/chunks/candidate_chunks_clean.json)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=MIN_WORDS,
        help=f"Minimum word count to keep a chunk (default: {MIN_WORDS})",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=NEAR_DUP_THRESHOLD,
        help=f"Jaccard threshold for near-dup removal (default: {NEAR_DUP_THRESHOLD})",
    )
    args = parser.parse_args()

    # --- Load ---
    with open(args.input, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Chunk Pre-Filter")
    print(f"  Input : {args.input}  ({len(chunks)} chunks)")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    removed: dict[str, int] = defaultdict(int)
    seen_hashes: set[str] = set()
    after_basic: list[dict] = []

    # -----------------------------------------------------------------------
    # Pass 1: per-chunk scalar filters
    # -----------------------------------------------------------------------
    for c in chunks:
        text  = c.get("chunk_text", "")
        title = c.get("section_title", "")
        wc    = c.get("word_count") or len(text.split())

        # 1. Blocked title
        if is_blocked_title(title):
            removed["blocked_title"] += 1
            continue

        # 2. Too short
        if wc < args.min_words:
            removed["too_short"] += 1
            continue

        # 3. Low alpha-ratio  (table / figure / list-heavy)
        if alpha_ratio(text) < MIN_ALPHA_RATIO:
            removed["low_alpha_ratio"] += 1
            continue

        # 4. Too few sentences  (pure bullet lists or headings)
        if sentence_count(text) < MIN_SENTENCES:
            removed["too_few_sentences"] += 1
            continue

        # 5. Exact duplicate
        h = chunk_hash(text)
        if h in seen_hashes:
            removed["exact_duplicate"] += 1
            continue
        seen_hashes.add(h)

        after_basic.append(c)

    print(f"After scalar filters: {len(after_basic)} chunks remain")
    print(f"  (removed {len(chunks) - len(after_basic)} so far)\n")

    # -----------------------------------------------------------------------
    # Pass 2: near-duplicate removal  (within each document for speed)
    # -----------------------------------------------------------------------
    by_doc: dict[str, list] = defaultdict(list)
    for c in after_basic:
        by_doc[c.get("document", "")].append(c)

    final: list[dict] = []
    for doc, doc_chunks in by_doc.items():
        kept: list[dict] = []
        for c in doc_chunks:
            text = c.get("chunk_text", "")
            is_near_dup = any(
                jaccard(text, k.get("chunk_text", "")) >= args.near_dup_threshold
                for k in kept
            )
            if is_near_dup:
                removed["near_duplicate"] += 1
            else:
                kept.append(c)
        final.extend(kept)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(f"{'─'*50}")
    print(f"Removal summary:")
    for reason, count in sorted(removed.items(), key=lambda x: -x[1]):
        print(f"  {reason:<25s}: {count}")
    print(f"{'─'*50}")
    print(f"Total removed  : {len(chunks) - len(final)}")
    print(f"Clean chunks   : {len(final)}  (out of {len(chunks)} original)")
    print(f"Retention rate : {len(final)/len(chunks)*100:.1f}%\n")

    doc_counts = Counter(c.get("document", "") for c in final)
    print("Clean chunks per document:")
    for doc, count in doc_counts.most_common():
        print(f"  {count:4d}  {doc}")

    # Quality distribution of kept chunks
    kept_lengths = sorted(len(c.get("chunk_text", "")) for c in final)
    if kept_lengths:
        print(f"\nKept chunk text-length distribution:")
        print(f"  min:    {kept_lengths[0]}")
        print(f"  median: {kept_lengths[len(kept_lengths)//2]}")
        print(f"  max:    {kept_lengths[-1]}")
        print(f"  300–600 chars  : {sum(1 for l in kept_lengths if 300 <= l < 600)}")
        print(f"  600–1200 chars : {sum(1 for l in kept_lengths if 600 <= l < 1200)}")
        print(f"  >1200 chars    : {sum(1 for l in kept_lengths if l >= 1200)}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Saved {len(final)} clean chunks → {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

