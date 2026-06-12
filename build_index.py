import os
import re
import json
from statistics import mean
import matplotlib.pyplot as plt

from src.parser.docling_parser import MedicalParser
from src.indexing.chunker import chunk_medical_data
from src.indexing.embedder import MedicalEmbedder
from src.indexing.vector_store import MedicalVectorStore

# --------------------------------------------------
# 🛠️ 1. Configuration for Filtering and Cleaning
# --------------------------------------------------
DOC_CONFIGS = {
    "Shoulderdoc - Shoulder Rehab Book.pdf": {
        "start_page": 3,
        "stop_word": "References"
    },
    "Adhesive_capsulitis_JOSPT.pdf": {
        "start_page": 1,
        "exclude_last_2nd": True,
        "stop_word": "REFERENCES"
    },
    "Therapeutic_exercise_Foundations_and_techniques_by_Colby_Lynn_Allen-572-650.pdf": {
        "start_page": 1,
        "max_page": 71
    },
    "Subacromial pain syndrome.pdf": {
        "stop_word": "REFERENCES"
    },
    "Rotator cuff tendinopathy CPG.pdf": {
        "stop_word": "REFERENCES"
    },
    # ── NEW ──────────────────────────────────────────────────────
    "Atraumatic-Shoulder-Instability.pdf": {
        "start_page": 1,
        "max_page": 8,          # pages 1-8 only, exclude references
        "stop_word": "References"
    }
}

# --------------------------------------------------
# 🛠️ 2. Forbidden Term Filters (Medical Administration)
# --------------------------------------------------
FORBIDDEN_TERMS = ["injection", "needle", "administer", "intra-articular", "intra articular"]
FORBIDDEN_WHOLE_WORD = ["mg", "ml", "dose"]


def contains_forbidden_content(text: str) -> bool:
    """
    Returns True if the chunk contains medical administration terms.
    Uses substring match for long unambiguous terms, whole-word regex
    for short terms to avoid false positives (e.g. 'ml' in 'small').
    """
    text_lower = text.lower()

    for term in FORBIDDEN_TERMS:
        if term in text_lower:
            return True

    for term in FORBIDDEN_WHOLE_WORD:
        if re.search(rf'\b{re.escape(term)}\b', text_lower):
            return True

    return False


def is_reference_list(text: str) -> bool:
    """
    Detects bibliography/reference list chunks that slipped through
    stop_word truncation because they appear under clinical section headers.

    Handles multiple citation formats:
      - Numbered:  "68. Guyatt GH, Oxman AD..."
      - Author:    "Smith J, Jones A. Title. 2001"
      - Bullet:    "- Author A. Title. Journal. 2001"
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False

    citation_lines = []
    for line in lines:
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', line))
        is_numbered = bool(re.match(r'^\d+[\.\)]\s+[A-Z]', line))
        is_author_list = bool(re.match(r'^[A-Z][a-z]+\s+[A-Z]{1,3}[,\.]', line))
        is_bullet_citation = line.startswith('-') and has_year

        if (is_numbered and has_year) or is_author_list or is_bullet_citation:
            citation_lines.append(line)

    return len(citation_lines) / len(lines) > 0.35


def clean_and_truncate(text: str, stop_word: str = None) -> str:
    """
    Cuts off text at the References section and filters out noisy posture labels.
    """
    if not text:
        return ""

    if stop_word:
        pattern = re.compile(rf'\n{re.escape(stop_word)}', re.IGNORECASE)
        text = pattern.split(text)[0]

    lines = text.split('\n')
    blacklist = {'good', 'poor', 'xii', 'xiii', 'xiv'}

    cleaned_lines = [
        line for line in lines
        if line.strip().lower() not in blacklist and len(line.strip()) > 30
    ]
    return "\n".join(cleaned_lines)


def trim_last_n_pages(md_content: str, n: int = 2) -> str:
    """
    Removes the last N pages using Docling's form-feed page separator.
    """
    pages = md_content.split("\f")
    if len(pages) > n:
        trimmed = "\f".join(pages[:-n])
        print(f"   ✂️  Excluded last {n} pages ({len(pages)} → {len(pages) - n} pages)")
        return trimmed
    else:
        print(f"   ⚠️  Document has only {len(pages)} pages; skipping exclude_last_2nd trim.")
        return md_content


def run_indexing_pipeline():
    """
    End-to-End Indexing Pipeline:
    PDF → Markdown → Clean → Header-Aware Chunking → Embedding → FAISS
    """
    parser = MedicalParser()
    embedder = MedicalEmbedder()
    vector_store = MedicalVectorStore()

    raw_data_path = "data/raw"
    all_chunks = []

    print("🚀 Starting Medical RAG Indexing Pipeline...")

    for file in os.listdir(raw_data_path):
        if not file.endswith(".pdf"):
            continue

        print(f"\n📄 Processing: {file}")
        pdf_path = os.path.join(raw_data_path, file)
        config = DOC_CONFIGS.get(file, {})

        start_page = config.get("start_page", None)
        max_page = config.get("max_page", None)

        print(f"   📑 Page range: start={start_page}, end={max_page}")

        # A. Convert PDF → Markdown
        md_content = parser.convert_to_markdown(
            pdf_path,
            start_page=start_page,
            end_page=max_page
        )

        # B. Trim last 2 pages if needed
        if config.get("exclude_last_2nd"):
            md_content = trim_last_n_pages(md_content, n=2)

        # C. Truncate at References and clean noise
        cleaned_md = clean_and_truncate(md_content, config.get("stop_word"))

        # D. Header-aware chunking
        chunks = chunk_medical_data(cleaned_md)

        # E. Metadata enrichment & filtering
        skipped_section = 0
        skipped_references = 0
        skipped_forbidden = 0

        for chunk in chunks:
            chunk.metadata["source"] = file

            header_levels = ["Header_1", "Header_2", "Header_3"]
            section = next(
                (chunk.metadata.get(h) for h in header_levels if h in chunk.metadata),
                "General"
            )
            chunk.metadata["section"] = section

            # Skip by section header name
            if any(x in section.lower() for x in ["reference", "bibliography", "chapter contents"]):
                skipped_section += 1
                continue

            # Skip reference list content under clinical headers
            if is_reference_list(chunk.page_content):
                skipped_references += 1
                continue

            # Skip medical administration content
            if contains_forbidden_content(chunk.page_content):
                skipped_forbidden += 1
                continue

            chunk.metadata["word_count"] = len(chunk.page_content.split())
            chunk.metadata["char_count"] = len(chunk.page_content)

            all_chunks.append(chunk)

        accepted = len(chunks) - skipped_section - skipped_references - skipped_forbidden
        print(f"   🚫 Skipped (bad section header): {skipped_section}")
        print(f"   🚫 Skipped (reference list):     {skipped_references}")
        print(f"   🚫 Skipped (forbidden terms):    {skipped_forbidden}")
        print(f"   ✅ Accepted: {accepted}")

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------
    total_chunks = len(all_chunks)
    if total_chunks == 0:
        raise ValueError("❌ No valid chunks created. Check PDF content and filtering logic.")

    print(f"\n✂️  Total clean chunks created: {total_chunks}")
    print(f"📊 Avg word count: {mean([c.metadata['word_count'] for c in all_chunks]):.1f}")

    # --------------------------------------------------
    # Generate Embeddings
    # --------------------------------------------------
    print("\n🧠 Generating normalized embeddings (MiniLM)...")
    embeddings = embedder.embed_chunks(all_chunks)

    # --------------------------------------------------
    # Save FAISS Index
    # --------------------------------------------------
    vector_store.save_index(all_chunks, embeddings, embedder)

    # --------------------------------------------------
    # Save Candidate JSON Audit
    # --------------------------------------------------
    candidate_file = os.path.join("data", "chunks", "candidate_chunks.json")
    os.makedirs(os.path.dirname(candidate_file), exist_ok=True)

    chunks_to_save = [
        {
            "chunk_id": i,
            "document": chunk.metadata.get("source", "unknown"),
            "section_title": chunk.metadata.get("section", "General"),
            "chunk_text": chunk.page_content,
            "word_count": chunk.metadata.get("word_count"),
            "char_count": chunk.metadata.get("char_count"),
            "metadata": chunk.metadata
        }
        for i, chunk in enumerate(all_chunks)
    ]

    with open(candidate_file, "w", encoding="utf-8") as f:
        json.dump(chunks_to_save, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved {len(chunks_to_save)} clean chunks to {candidate_file}")

    # --------------------------------------------------
    # Visualize
    # --------------------------------------------------
    word_counts = [c.metadata["word_count"] for c in all_chunks]
    char_counts = [c.metadata["char_count"] for c in all_chunks]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(word_counts, bins=20, color="#4C72B0")
    plt.title("Word Count Distribution")
    plt.xlabel("Words per Chunk")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(char_counts, bins=20, color="#55A868")
    plt.title("Char Count Distribution")
    plt.xlabel("Characters per Chunk")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_indexing_pipeline()