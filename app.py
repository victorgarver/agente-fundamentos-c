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

def detect_mode(query: str) -> tuple[str, str]:
    """Detecta el modo y devuelve (modo, query_limpia)."""
    query = query.strip()
    if query.lower().startswith("/guiado"):
        return "guiado", query[7:].strip()
    elif query.lower().startswith("/solución") or query.lower().startswith("/solucion"):
        return "solucion", query[9:].strip()
    elif query.lower().startswith("/corregir"):
        return "corregir", query[9:].strip()
    return "teorico", query

def build_prompt(mode: str, query: str, context: str, historial_text: str) -> str:
    base = f"""Eres un profesor experto en programación en C.
IMPORTANTE: Usa ÚNICAMENTE el contenido de los apuntes proporcionados.
No añadas conocimiento propio. Si algo no está en los apuntes, dilo explícitamente.
{historial_text}
Apuntes relevantes:
{context}

"""
    if mode == "teorico":
        return base + f"Pregunta: {query}\nRespuesta:"

    elif mode == "guiado":
        return base + f"""El alumno quiere resolver este ejercicio con tu ayuda paso a paso: {query}

No des la solución completa. Guía al alumno con preguntas y pistas basadas en los apuntes.
Empieza identificando qué conceptos de los apuntes son necesarios para resolver el ejercicio.
Primera pista:"""

    elif mode == "solucion":
        return base + f"""Resuelve este ejercicio: {query}

Proporciona:
1. Pseudocódigo siguiendo el estilo de los apuntes
2. Código en C siguiendo el estilo de los apuntes

Usa solo construcciones y funciones que aparezcan en los apuntes.
Solución:"""

    elif mode == "corregir":
        return base + f"""El alumno ha enviado esta solución para corregir:

{query}

Revisa la solución basándote únicamente en los apuntes. Indica:
1. Si es correcta o tiene errores
2. Qué errores tiene (si los hay) y en qué apunte se explica la forma correcta
3. Sugerencias de mejora basadas en los apuntes

Corrección:"""

    return base + f"Pregunta: {query}\nRespuesta:"

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

        context = "\n\n".join([
            f"[{chunk['source']}]\n{chunk['text']}"
            for chunk in chunks
        ])

        historial_text = ""
        if len(st.session_state.messages) > 1:
            historial_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in st.session_state.messages[:-1]
            ])
            historial_text = f"\nHistorial de conversación:\n{historial_text}\n"

        prompt = build_prompt(mode, clean_query, context, historial_text)

        mode_labels = {
            "teorico": "💬 Pregunta teórica",
            "guiado": "🧭 Modo guiado",
            "solucion": "✅ Modo solución",
            "corregir": "🔍 Modo corrección"
        }

        with st.chat_message("assistant"):
            st.caption(mode_labels[mode])
            with st.spinner("Pensando..."):
                response = ollama.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response["message"]["content"]
                st.markdown(answer)

                sources = list(set([chunk["source"] for chunk in chunks]))
                with st.expander("📚 Fuentes"):
                    for source in sources:
                        st.write(f"- {source}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
