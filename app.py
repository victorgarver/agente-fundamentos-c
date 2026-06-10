import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
import ollama
from retriever import retrieve

MODEL = "gemma4:latest"

st.set_page_config(
    page_title="Agente Fundamentos C",
    page_icon="🖥️",
    layout="centered"
)

st.title("🖥️ Agente Fundamentos de Programación en C")
st.caption("Haz preguntas sobre los apuntes del curso")

# Inicializar historial en session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if query := st.chat_input("Escribe tu pregunta..."):

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Recuperar contexto
    chunks = retrieve(query, n_results=3)

    # Construir contexto
    context = "\n\n".join([
        f"[{chunk['source']}]\n{chunk['text']}"
        for chunk in chunks
    ])

    # Construir historial para el prompt
    historial_text = ""
    if len(st.session_state.messages) > 1:
        historial_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.messages[:-1]
        ])
        historial_text = f"\nHistorial de conversación:\n{historial_text}\n"

    prompt = f"""Eres un profesor experto en programación en C.
Responde de forma clara y precisa basándote únicamente en el contexto proporcionado.
Si la información no está en el contexto, dilo explícitamente.
{historial_text}
Contexto de los apuntes:
{context}

Pregunta actual: {query}
Respuesta:"""

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"]
            st.markdown(answer)

            # Mostrar fuentes
            sources = list(set([chunk["source"] for chunk in chunks]))
            with st.expander("📚 Fuentes"):
                for source in sources:
                    st.write(f"- {source}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
