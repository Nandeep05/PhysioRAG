from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from typing import List


def _load_reranker(use_reranker: bool, top_k: int):
    """Lazy-load reranker only when enabled."""
    if not use_reranker:
        return None
    try:
        from src.rag.reranker import Reranker
        r = Reranker()
        return (r, top_k)
    except Exception as e:
        print(f"⚠️  Reranker disabled ({e}). Using ensemble order only.")
        return None


class MedicalHybridRetriever:
    def __init__(self, vectorstore, chunks, use_reranker: bool = True, rerank_top_k: int = 8):
        """
        Args:
            vectorstore: Loaded FAISS vectorstore
            chunks: List of LangChain Document objects with metadata
            use_reranker: If True, rerank retrieved docs with a cross-encoder for better accuracy.
            rerank_top_k: Number of docs to keep after reranking (only if use_reranker=True).
        """
        # BM25 + Vector: fetch more candidates so reranker has a larger pool to promote gold into top_k
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 12

        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 12}
        )

        # Ensemble — weighted combination
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.4, 0.6]
        )

        # Optional reranker: reorders ensemble results so the best chunk is first (top_k passed to answer model)
        self._reranker, self._rerank_top_k = _load_reranker(use_reranker, rerank_top_k) or (None, rerank_top_k)

        # Build a chunk_id → metadata lookup so BM25 results
        # can have their chunk_id restored after retrieval.
        # BM25Retriever strips metadata during indexing, so chunk_ids
        # come back as None unless we re-attach them by content match.
        self._content_to_metadata = {}
        for doc in chunks:
            # Use first 200 chars as key — unique enough for lookup
            key = doc.page_content[:200].strip()
            self._content_to_metadata[key] = doc.metadata

    def _restore_metadata(self, docs: List[Document]) -> List[Document]:
        """
        Re-attaches chunk_id and other metadata to docs that lost it
        during BM25 retrieval.
        """
        restored = []
        for doc in docs:
            if doc.metadata.get("chunk_id") is None:
                key = doc.page_content[:200].strip()
                if key in self._content_to_metadata:
                    doc.metadata.update(self._content_to_metadata[key])
            restored.append(doc)
        return restored

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieval with optional reranking. Returns docs with chunk_ids intact.
        """
        docs = self.ensemble_retriever.invoke(query)

        if not docs:
            print(f"⚠️  WARNING: Retriever returned 0 docs for: '{query[:60]}'")
            return []

        # Restore chunk_ids lost by BM25
        docs = self._restore_metadata(docs)

        # Rerank for better precision (puts most relevant chunk first)
        if self._reranker is not None and docs:
            docs = self._reranker.rerank(query, docs, top_k=self._rerank_top_k)

        return docs