# Agente Fundamentos C

RAG local sobre apuntes de programación en C.

## Estructura
- `docs/` — apuntes en .txt
- `src/ingest.py` — carga los documentos en ChromaDB
- `src/retriever.py` — busca fragmentos relevantes
- `src/agent.py` — genera la respuesta con Ollama
- `db/` — base de datos vectorial local (no se sube a GitHub)

## Uso
1. Añade tus apuntes .txt en `docs/`
2. Ejecuta `uv run python src/ingest.py`
3. Ejecuta `uv run python main.py`

## Modelos
- LLM: qwen2.5:4b (Ollama)
- Embeddings: nomic-embed-text (Ollama)
