import streamlit as st
import ollama
from src.retriever import retrieve
from src.modes import detect_mode, build_messages

MODEL = "gemma4:latest"

st.set_page_config(
    page_title="Agente Fundamentos C",
    page_icon="🖥️",
    layout="centered"
)

st.title("🖥️ Agente Fundamentos de Programación en C")
st.caption("Asistente para ejercicios y teoría del curso")

with st.expander("📖 Comandos disponibles"):
    st.markdown("""
    - **Sin comando** — pregunta teórica sobre los apuntes
    - **/guiado** — el agente te guía paso a paso sin darte la solución
    - **/solución** — genera pseudocódigo + código C completo
    - **/corregir** — pega tu código y el agente lo revisa
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

mode_labels = {
    "teorico": "💬 Pregunta teórica",
    "guiado": "🧭 Modo guiado",
    "solucion": "✅ Modo solución",
    "corregir": "🔍 Modo corrección",
}

if query := st.chat_input("Escribe tu pregunta o usa /guiado, /solución, /corregir..."):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    mode, clean_query = detect_mode(query)

    if not clean_query:
        with st.chat_message("assistant"):
            st.warning("Por favor escribe el enunciado después del comando.")
    else:
        chunks = retrieve(clean_query, n_results=4)
        history = st.session_state.messages[:-1]
        messages = build_messages(mode, clean_query, chunks, history)

        with st.chat_message("assistant"):
            st.caption(mode_labels[mode])
            with st.spinner("Pensando..."):
                try:
                    response = ollama.chat(model=MODEL, messages=messages)
                    answer = response["message"]["content"]
                except Exception as e:
                    st.error(f"Error al conectar con Ollama: {e}")
                    st.session_state.messages.pop()
                    st.stop()

                st.markdown(answer)

                sources = list(set(chunk["source"] for chunk in chunks))
                with st.expander("📚 Fuentes"):
                    for source in sources:
                        st.write(f"- {source}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
