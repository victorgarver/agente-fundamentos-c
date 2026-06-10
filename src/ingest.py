from pathlib import Path
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# Rutas
DOCS_DIR = Path("docs")
DB_DIR = Path("db")

# Modelo de embeddings
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# Configuración ChromaDB local
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(name="fundamentos_c", metadata={"hnsw:space": "cosine"})

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_documents():
    txt_files = list(DOCS_DIR.glob("*.txt"))
    print(f"Encontrados {len(txt_files)} documentos")

    for filepath in txt_files:
        print(f"Procesando: {filepath.name}")
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
                ids=[f"{filepath.stem}_{i}"],
                metadatas=[{"source": filepath.name}]
            )

    print("✓ Ingestión completada")

if __name__ == "__main__":
    ingest_documents()
