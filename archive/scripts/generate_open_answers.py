# generate_open_answers.py
"""
Generate free-text RAG answers for open-ended questions.
Separate from generate_answers.py which is MCQ-only.
"""

import json
import logging
import argparse
from pathlib import Path
from src.rag.hybrid_retriever import MedicalHybridRetriever
from src.indexing.vector_store import MedicalVectorStore
from src.indexing.embedder import MedicalEmbedder
from langchain_core.documents import Document
from config import (
    VLLM_BASE_URL, LLM_PROVIDER, CANDIDATE_CHUNKS_PATH,
    RESULTS_DIR, LOG_LEVEL
)

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OPEN_ANSWER_PROMPT = """
You are a Clinical Decision Support Assistant specialising in 
shoulder pain management and rehabilitation.

Answer the following patient question using ONLY the retrieved 
clinical context below. 

RULES:
1. Base your answer ONLY on the provided context
2. If the context does not contain enough information to answer,
   say: "The available cli4nical guidelines do not specify this."
3. Be concise and clinically accurate
4. Do NOT use bullet points — write in clear paragraph form
5. Do NOT mention the source documents by name

PATIENT QUESTION:
{question}

RETRIEVED CLINICAL CONTEXT:
{context}

Provide a clear, evidence-based answer:
"""


def load_pipeline_components(embed_model: str = "all-MiniLM-L6-v2"):
    """Load embedding, vector store and retriever."""
    logger.info("Loading pipeline components...")
    embedder = MedicalEmbedder(embed_model)
    store = MedicalVectorStore()
    vectorstore = store.load_index(embedder)

    with open(CANDIDATE_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    formatted_docs = [
        Document(
            page_content=c.get("chunk_text") or c.get("chunktext", ""),
            metadata={
                **c.get("metadata", {}),
                "chunk_id": c.get("chunk_id") or c.get("chunkid"),
                "document": c.get("document"),
            }
        )
        for c in chunks
    ]

    retriever = MedicalHybridRetriever(vectorstore, formatted_docs)
    logger.info(f"Loaded {len(formatted_docs)} chunks into retriever")
    return retriever


def process_open_question(
    question_item: dict,
    retriever: MedicalHybridRetriever,
    generator
) -> dict:
    """Process a single open-ended question."""

    # Handle both "question" and "Question" key names
    question_text = (
        question_item.get("question")
        or question_item.get("Question", "")
    )

    # Retrieve relevant chunks
    context_docs = retriever.get_relevant_documents(question_text)
    context_docs = context_docs[:8]

    context_docs_info = [
        {
            "text": doc.page_content,
            "chunk_id": doc.metadata.get("chunk_id"),
            "document": doc.metadata.get("document", "Unknown"),
        }
        for doc in context_docs
    ]

    context_text = "\n\n---\n\n".join(
        [d["text"] for d in context_docs_info]
    )

    # Build prompt
    prompt = OPEN_ANSWER_PROMPT.format(
        question=question_text,
        context=context_text
    )

    # Generate free-text answer
    try:
        raw_answer = generator.provider.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Slight variation for open answers
        )
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raw_answer = "Error generating answer."

    return {
        "question_id": question_item.get("question_id", "unknown"),
        "category": question_item.get("category", "General"),
        "question": question_text,
        "generated_answer": raw_answer.strip(),
        "retrieved_chunks": context_docs_info,
        "retrieved_chunk_ids": [
            d["chunk_id"] for d in context_docs_info
        ],
        "source_documents": list(set(
            d["document"] for d in context_docs_info
        )),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAG answers for open-ended questions"
    )
    parser.add_argument("--input", required=True,
                        help="Path to open questions JSON file")
    parser.add_argument("--output", default=RESULTS_DIR,
                        help="Output directory")
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default="vllm",
                        choices=["ollama", "vllm", "hf"])
    parser.add_argument("--embed-model",
                        default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    logger.info(f"Starting open question answering: {vars(args)}")

    # Load components
    retriever = load_pipeline_components(args.embed_model)

    from src.rag.reasoning_gen import ReasoningGenerator
    generator = ReasoningGenerator(
        model_name=args.model or "Qwen3-14B",
        provider_type=args.provider,
        base_url=VLLM_BASE_URL,
    )

    # Load questions
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict with "questions" key
    if isinstance(data, list):
        questions = data
    else:
        questions = data.get("questions", [])

    logger.info(f"Processing {len(questions)} open questions...")

    results = []
    for idx, q in enumerate(questions, 1):
        logger.info(
            f"[{idx}/{len(questions)}] "
            f"{q.get('question_id','?')} — "
            f"{q.get('question','')[:60]}..."
        )
        result = process_open_question(q, retriever, generator)
        results.append(result)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "open_questions_answered.json"

    output_data = {
        "metadata": {
            "total_questions": len(results),
            "model": args.model,
            "provider": args.provider,
            "input_file": args.input,
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(results)} answers → {output_path}")


if __name__ == "__main__":
    main()