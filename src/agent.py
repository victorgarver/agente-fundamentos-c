import ollama
from retriever import retrieve

MODEL = "gemma4:latest"

def build_prompt(query: str, context_chunks: list[dict], historial: list[dict]) -> str:
    context = "\n\n".join([
        f"[{chunk['source']}]\n{chunk['text']}"
        for chunk in context_chunks
    ])

    historial_text = ""
    if historial:
        historial_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in historial
        ])
        historial_text = f"\nHistorial de conversación:\n{historial_text}\n"

    return f"""Eres un profesor experto en programación en C.
Responde de forma clara y precisa basándote únicamente en el contexto proporcionado.
Si la información no está en el contexto, dilo explícitamente.
{historial_text}
Contexto de los apuntes:
{context}

Pregunta actual: {query}
Respuesta:"""

def chat():
    historial = []
    print("Agente Fundamentos C — escribe 'salir' para terminar\n")

    while True:
        query = input("Tú: ").strip()

        if query.lower() == "salir":
            print("Hasta luego!")
            break

        if not query:
            continue

        chunks = retrieve(query, n_results=3)

        prompt = build_prompt(query, chunks, historial)

        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"]

        historial.append({"role": "user", "content": query})
        historial.append({"role": "assistant", "content": answer})

        print(f"\nAgente: {answer}")
        sources = list(set([chunk["source"] for chunk in chunks]))
        print(f"\nFuentes: {', '.join(sources)}\n")

if __name__ == "__main__":
    chat()
