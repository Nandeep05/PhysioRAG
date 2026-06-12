import json
from collections import Counter

with open(r"D:\College\FAU_Notes\4th_sem\Graph_RAG_Project\PhysioRAG_pipeline_old\data\chunks\candidate_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Basic stats ---
word_counts = [c["word_count"] for c in chunks]
print(f"Total chunks: {len(chunks)}")
print(f"Avg words: {sum(word_counts)/len(word_counts):.1f}")
print(f"Min words: {min(word_counts)}")
print(f"Max words: {max(word_counts)}")

# --- Chunks per document ---
print("\n📄 Chunks per document:")
doc_counts = Counter(c["document"] for c in chunks)
for doc, count in doc_counts.items():
    print(f"  {doc}: {count}")

# --- Chunks per section ---
print("\n📂 Top 15 sections:")
section_counts = Counter(c["section_title"] for c in chunks)
for section, count in section_counts.most_common(15):
    print(f"  {section}: {count}")

# --- Spot check: print 5 random chunks ---
import random
print("\n🔍 Random chunk samples:")
for chunk in random.sample(chunks, 5):
    print(f"\n  Source: {chunk['document']}")
    print(f"  Section: {chunk['section_title']}")
    print(f"  Words: {chunk['word_count']}")
    print(f"  Text preview: {chunk['chunk_text'][:300]}")
    print("  " + "-"*60)

# --- Flag suspicious chunks ---
print("\n⚠️  Potentially bad chunks (under 40 words):")
short_chunks = [c for c in chunks if c["word_count"] < 40]
print(f"  Count: {len(short_chunks)}")
for c in short_chunks[:5]:
    print(f"  [{c['word_count']} words] {c['chunk_text'][:150]}")

# --- Check for OCR noise ---
print("\n🔡 Chunks with possible OCR noise (many single chars):")
noisy = [c for c in chunks if len([w for w in c["chunk_text"].split() if len(w) == 1]) > 10]
print(f"  Count: {len(noisy)}")
for c in noisy[:3]:
    print(f"  {c['chunk_text'][:200]}")



