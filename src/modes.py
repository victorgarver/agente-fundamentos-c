MAX_HISTORY = 10

_PREFIXES = [
    ("/solución", "solucion"),
    ("/solucion", "solucion"),
    ("/guiado", "guiado"),
    ("/corregir", "corregir"),
]


def detect_mode(query: str) -> tuple[str, str]:
    lowered = query.strip().lower()
    for prefix, mode in _PREFIXES:
        if lowered.startswith(prefix):
            return mode, query.strip()[len(prefix):].strip()
    return "teorico", query.strip()


def _system_prompt() -> str:
    return (
        "Eres un profesor experto en programación en C.\n"
        "IMPORTANTE: Usa ÚNICAMENTE el contenido de los apuntes proporcionados.\n"
        "No añadas conocimiento propio. Si algo no está en los apuntes, dilo explícitamente."
    )


def _user_content(mode: str, query: str, context: str) -> str:
    base = f"Apuntes relevantes:\n{context}\n\n"
    if mode == "guiado":
        return (
            base
            + f"El alumno quiere resolver este ejercicio con tu ayuda paso a paso: {query}\n\n"
            "No des la solución completa. Guía al alumno con preguntas y pistas basadas en los apuntes.\n"
            "Empieza identificando qué conceptos de los apuntes son necesarios para resolver el ejercicio.\n"
            "Primera pista:"
        )
    if mode == "solucion":
        return (
            base
            + f"Resuelve este ejercicio: {query}\n\n"
            "Proporciona:\n"
            "1. Pseudocódigo siguiendo el estilo de los apuntes\n"
            "2. Código en C siguiendo el estilo de los apuntes\n\n"
            "Usa solo construcciones y funciones que aparezcan en los apuntes."
        )
    if mode == "corregir":
        return (
            base
            + f"El alumno ha enviado esta solución para corregir:\n\n{query}\n\n"
            "Revisa la solución basándote únicamente en los apuntes. Indica:\n"
            "1. Si es correcta o tiene errores\n"
            "2. Qué errores tiene (si los hay) y en qué apunte se explica la forma correcta\n"
            "3. Sugerencias de mejora basadas en los apuntes"
        )
    return base + f"Pregunta: {query}"


def build_messages(mode: str, query: str, context_chunks: list[dict], history: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{chunk['source']}]\n{chunk['text']}" for chunk in context_chunks
    )
    messages = [{"role": "system", "content": _system_prompt()}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": _user_content(mode, query, context)})
    return messages
