from pathlib import Path
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DB_DIR = Path(__file__).parent.parent / "db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_collection(name="fundamentos_c")
    return _collection


def retrieve(query: str, n_results: int = 5) -> list[dict]:
    """Búsqueda híbrida: semántica + BM25."""
    model = _get_model()
    collection = _get_collection()

    embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    embedding = np.asarray(embedding, dtype=np.float32)

    semantic_results = collection.query(
        query_embeddings=embedding.tolist(),
        n_results=n_results * 2,
        include=["documents", "metadatas", "distances"]
    )

    docs = semantic_results["documents"][0]
    metas = semantic_results["metadatas"][0]
    distances = semantic_results["distances"][0]

    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.lower().split())

    semantic_scores = np.array([1 - d for d in distances])
    bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
    semantic_norm = semantic_scores / (semantic_scores.max() + 1e-9)
    combined = 0.5 * semantic_norm + 0.5 * bm25_norm

    ranked_indices = np.argsort(combined)[::-1][:n_results]

    return [
        {
            "text": docs[i],
            "source": metas[i]["source"],
            "distance": distances[i],
            "score": float(combined[i]),
        }
        for i in ranked_indices
    ]


if __name__ == "__main__":
    query = "¿Cómo funciona el bucle for en C?"
    chunks = retrieve(query)
    for chunk in chunks:
        print(f"\n--- {chunk['source']} (score: {chunk['score']:.4f}) ---")
        print(chunk["text"][:300])
