import json
from pathlib import Path
from src.rag.hybrid_retriever import MedicalHybridRetriever
from src.indexing.vector_store import MedicalVectorStore
from src.indexing.embedder import MedicalEmbedder
from langchain_core.documents import Document

# -----------------------------
# 1️⃣ Load Pipeline
# -----------------------------
embedder = MedicalEmbedder()
store = MedicalVectorStore()
vectorstore = store.load_index(embedder)

# Load candidate chunks
candidate_chunks_file = "data/chunks/candidate_chunks.json"
with open(candidate_chunks_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

formatted_docs = [
    Document(page_content=c['chunk_text'], metadata=c.get('metadata', {}))
    for c in chunks
]

retriever = MedicalHybridRetriever(vectorstore, formatted_docs)

# -----------------------------
# 2️⃣ Load Evaluation Questions
# -----------------------------
evaluation_file = Path(
    r"D:\College\FAU_Notes\4th_sem\Graph_RAG_Project\PhysioRAG_pipeline\data\gold_standard\factual_eval_qwen_20260220_001054.json")
with open(evaluation_file, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)


# -----------------------------
# 3️⃣ Validate Chunking
# -----------------------------
def validate_chunking_accuracy(eval_questions, retriever, top_k=3):
    total_questions = len(eval_questions)
    correct_retrievals = 0

    for item in eval_questions:
        question = item["Question"]
        gold_answer = item["Text Answer"]
        context_docs = retriever.get_relevant_documents(question, top_k=top_k)

        found_answer_in_retrieved = any(
            gold_answer.lower() in doc.page_content.lower() for doc in context_docs
        )

        if found_answer_in_retrieved:
            correct_retrievals += 1

    coverage_percentage = (correct_retrievals / total_questions) * 100
    return coverage_percentage


coverage = validate_chunking_accuracy(eval_questions, retriever, top_k=3)
print(f"✅ Chunking coverage: {coverage:.2f}% of questions have their answer in top 3 retrieved chunks.")
