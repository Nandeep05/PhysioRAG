"""
Reranker for retrieval: scores query–document pairs and reorders by relevance.
Uses a cross-encoder (sentence-transformers) for better precision than bi-encoder retrieval alone.
"""
from typing import List
from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    _HAS_CROSS_ENCODER = False


class Reranker:
    """Rerank retrieved documents by query–document relevance using a cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if not _HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence_transformers is required for Reranker. "
                "Install with: pip install sentence-transformers"
            )
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 8,
    ) -> List[Document]:
        """
        Score each (query, doc) pair and return docs sorted by score descending.
        Keeps at most top_k documents.
        """
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        indexed = list(zip(scores, docs))
        indexed.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in indexed[:top_k]]


def rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int = 8,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Document]:
    """
    Standalone helper: rerank a list of documents by relevance to the query.
    Returns top_k documents. If sentence_transformers is not available, returns docs unchanged.
    """
    if not docs:
        return []
    if not _HAS_CROSS_ENCODER:
        return docs[:top_k]
    r = Reranker(model_name=model_name)
    return r.rerank(query, docs, top_k=top_k)
