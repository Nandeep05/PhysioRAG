import json
import os
from config import CANDIDATE_CHUNKS_PATH

def verify_json_chunks():

    if not os.path.exists(CANDIDATE_CHUNKS_PATH):
        print(f"❌ File not found at the expected location:")
        print(f"📍 {CANDIDATE_CHUNKS_PATH}")
        print("\n💡 Tip: Make sure you ran 'build_index.py' successfully first.")
        return

    # Open using the absolute path with UTF-8 encoding
    with open(CANDIDATE_CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"📊 --- Chunk Verification Report ---")
    print(f"Total Chunks Found: {len(chunks)}")

    if len(chunks) == 0:
        print("⚠️ Warning: JSON file is empty!")
        return

    # Previewing content
    print("\n🔍 Previewing first 3 chunks:")
    for i in range(min(3, len(chunks))):
        c = chunks[i]
        print(f"\n--- Chunk {c['chunk_id']} ---")
        print(f"Source: {c.get('filename', 'Unknown')}")
        print(f"Section: {c.get('section_type', 'N/A')}")
        print(f"Length: {len(c['text'])} characters")
        print(f"Snippet: {c['text'][:200].replace('\n', ' ')}...")

    # Verification checks
    empty_chunks = [c for c in chunks if not c['text'].strip()]
    if empty_chunks:
        print(f"\n⚠️ Found {len(empty_chunks)} empty chunks!")
    else:
        print("\n✅ All chunks contain valid text and are properly indexed.")

if __name__ == "__main__":
    verify_json_chunks()