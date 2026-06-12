from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import re


def is_reference_list(text: str) -> bool:
    """
    Detects bibliography/reference list chunks.
    Handles both bullet-style and plain citation formats:
      - "- Author A. Title. Journal. 2001;..."
      - "Author A, Author B. Title. Journal. 2001;..."
      - "68. Guyatt GH, Oxman AD..."
    A chunk is flagged if more than 35% of non-empty lines look like citations.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False

    citation_lines = []
    for line in lines:
        # Contains a 4-digit year (1900-2029)
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', line))
        # Looks like a numbered reference: "68. Author..."
        is_numbered = bool(re.match(r'^\d+[\.\)]\s+[A-Z]', line))
        # Looks like an author list: "Smith J, Jones A." or "Smith JA, Jones B."
        is_author_list = bool(re.match(r'^[A-Z][a-z]+\s+[A-Z]{1,3}[,\.]', line))
        # Starts with a dash and has a year
        is_bullet_citation = line.startswith('-') and has_year

        if (is_numbered and has_year) or is_author_list or is_bullet_citation:
            citation_lines.append(line)

    return len(citation_lines) / len(lines) > 0.35


def is_valid_chunk(text: str) -> bool:
    """
    Filters out low-quality chunks:
    - Pure image marker chunks
    - Table of contents
    - Markdown table rows and dividers
    - Reference/bibliography lists
    - Very small fragments under 40 words
    - Header-only sections
    """
    # --- Filter 1: Reject pure-image chunks ---
    text_without_images = text.replace("", "").strip()
    if not text_without_images:
        return False

    # --- Filter 2: Reject table of contents ---
    if "CHAPTER CONTENTS" in text:
        return False

    # --- Filter 3: Reject table divider lines (---|--- patterns) ---
    # These are markdown table separator rows that add zero value
    stripped = text_without_images.strip()
    if re.match(r'^[\|\-\s]+$', stripped):
        return False

    # --- Filter 4: Reject all table chunks regardless of content ---
    # Tables from PDFs rarely chunk well — they lose structure and become noise.
    # Clinical tables (exercise parameters) are better described in surrounding prose.
    if text.count("|") > 3:
        return False

    # --- Filter 5: Reject reference/bibliography list chunks ---
    if is_reference_list(text):
        return False

    # --- Filter 6: Reject very short chunks (applied AFTER table checks) ---
    word_count = len(text_without_images.split())
    if word_count < 40:
        return False

    # --- Filter 7: Reject header-only chunks ---
    if text_without_images.strip().startswith("#") and word_count < 20:
        return False

    return True


def chunk_medical_data(markdown_text: str) -> list:
    """
    Two-Stage Chunking Strategy:
    1. Split by Markdown Headers to maintain clinical context.
    2. Sub-split large sections into model-compatible sizes.
    3. Filter garbage chunks.

    Args:
        markdown_text (str): Cleaned markdown content from the parser.

    Returns:
        list: List of clean LangChain Document chunks with metadata.
    """
    # ---------- STAGE 1: HEADER SPLIT ----------
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    header_splits = md_splitter.split_text(markdown_text)

    # ---------- STAGE 2: SEMANTIC SPLIT ----------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? "],
    )
    final_chunks = text_splitter.split_documents(header_splits)

    # ---------- STAGE 3: FILTER GARBAGE ----------
    clean_chunks = [
        chunk for chunk in final_chunks
        if is_valid_chunk(chunk.page_content)
    ]

    print(f"📦 Total chunks before filtering: {len(final_chunks)}")
    print(f"🧹 Clean chunks after filtering: {len(clean_chunks)}")

    return clean_chunks