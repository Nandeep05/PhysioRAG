from src.indexing.embedder import MedicalEmbedder
from src.indexing.vector_store import MedicalVectorStore
from src.rag.hybrid_retriever import MedicalHybridRetriever
from src.rag.reasoning_gen import ReasoningGenerator
import json
from langchain_core.documents import Document


def interactive_chat():
    # 1. Setup
    embedder = MedicalEmbedder()
    store = MedicalVectorStore()
    vectorstore = store.load_index(embedder)

    # Load chunks for BM25
    with open("data/chunks/candidate_chunks.json", "r") as f:
        chunks = json.load(f)
    # Convert your list of dictionaries into a list of LangChain Documents
    formatted_docs = []
    for chunk in chunks:
        # Safely get metadata, providing defaults if keys are missing
        meta = chunk.get('metadata', {})
        source_info = meta.get('source', 'Unlabeled Document')  # Better than 'Unknown'
        page_info = meta.get('page', 'N/A')

        doc = Document(
            page_content=chunk.get('text', ''),
            metadata={
                "source": source_info,
                "page": page_info
            }
        )
        formatted_docs.append(doc)

    # Now pass the formatted documents to your retriever
    retriever = MedicalHybridRetriever(vectorstore, formatted_docs)
    generator = ReasoningGenerator()

    print("\n--- Shoulder Pain Evidence-Based Assistant ---")
    print("Type 'exit' to quit. Type 'debug' to toggle source visibility.\n")

    while True:
        query = input("Patient Query: ")
        if query.lower() == 'exit': break

        # 2. RETRIEVAL STEP
        context_docs = retriever.get_relevant_documents(query)
        # Modern LangChain standard
        # context_docs = retriever.invoke(query)

        print("\n🔍 [DEBUG: RETRIEVED CHUNKS]")
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown')
            # This helps you check if the right PDF/Page was found
            print(f"{i + 1}. Source: {source} | Content: {doc.page_content[:150]}...")

        # 3. GENERATION STEP
        answer = generator.generate_answer(query, context_docs)
        print(f"\n🤖 ASSISTANT:\n{answer}\n")


if __name__ == "__main__":
    interactive_chat()