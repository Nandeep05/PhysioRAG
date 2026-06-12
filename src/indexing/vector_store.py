import faiss
import os
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


class MedicalVectorStore:
    def __init__(self, index_path="data/chunks/faiss_index", dimension=384):
        self.index_path = index_path
        self.dimension = dimension

    def save_index(self, chunks, embeddings, embed_model):

        # Use Inner Product for cosine similarity
        index = faiss.IndexFlatIP(self.dimension)

        vector_store = FAISS(
            embedding_function=embed_model.embed_query,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )

        vector_store.add_embeddings(
            text_embeddings=zip([c.page_content for c in chunks], embeddings),
            metadatas=[c.metadata for c in chunks]
        )

        vector_store.save_local(self.index_path)

        print(f"✅ Cosine-based FAISS index saved to {self.index_path}")

    def load_index(self, embed_model):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("Index not found. Please run build_index.py first.")

        return FAISS.load_local(
            self.index_path,
            embed_model.embed_query,
            allow_dangerous_deserialization=True
        )
