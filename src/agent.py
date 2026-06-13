import ollama
from .retriever import retrieve
from .modes import MAX_HISTORY, detect_mode, build_messages

MODEL = "gemma4:latest"


def chat():
    history = []
    print("Agente Fundamentos C — escribe 'salir' para terminar")
    print("Comandos: /guiado, /solución, /corregir\n")

    while True:
        query = input("Tú: ").strip()

        if query.lower() == "salir":
            print("Hasta luego!")
            break

        if not query:
            continue

        mode, clean_query = detect_mode(query)

        if not clean_query:
            print("Por favor escribe el enunciado después del comando.\n")
            continue

        chunks = retrieve(clean_query, n_results=3)
        messages = build_messages(mode, clean_query, chunks, history)

        try:
            response = ollama.chat(model=MODEL, messages=messages)
            answer = response["message"]["content"]
        except Exception as e:
            print(f"\nError al conectar con Ollama: {e}\n")
            continue

        history.append({"role": "user", "content": clean_query})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

        print(f"\nAgente: {answer}")
        sources = list(set(chunk["source"] for chunk in chunks))
        print(f"\nFuentes: {', '.join(sources)}\n")
