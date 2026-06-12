from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class MedicalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        return embeddings

    def embed_query(self, query):
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        faiss.normalize_L2(embedding)

        return embedding[0]
