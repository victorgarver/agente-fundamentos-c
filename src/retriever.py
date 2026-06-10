from pathlib import Path
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Rutas
DB_DIR = Path("db")

# Modelo de embeddings
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# Configuración ChromaDB local
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name="fundamentos_c")

def retrieve(query: str, n_results: int = 5) -> list[dict]:
    """Búsqueda híbrida: semántica + BM25."""

    # 1. Búsqueda semántica
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

    # 2. Búsqueda BM25 sobre los candidatos semánticos
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.lower().split())

    # 3. Combinar puntuaciones (semántica normalizada + BM25 normalizada)
    semantic_scores = np.array([1 - d for d in distances])
    bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
    semantic_norm = semantic_scores / (semantic_scores.max() + 1e-9)
    combined = 0.5 * semantic_norm + 0.5 * bm25_norm

    # 4. Ordenar por puntuación combinada
    ranked_indices = np.argsort(combined)[::-1][:n_results]

    chunks = []
    for i in ranked_indices:
        chunks.append({
            "text": docs[i],
            "source": metas[i]["source"],
            "distance": distances[i],
            "score": float(combined[i])
        })
    return chunks

if __name__ == "__main__":
    query = "¿Cómo funciona el bucle for en C?"
    chunks = retrieve(query)
    for chunk in chunks:
        print(f"\n--- {chunk['source']} (score: {chunk['score']:.4f}) ---")
        print(chunk["text"][:300])
