import sys
from pathlib import Path

# Añadir src/ al path para que Python encuentre los módulos
sys.path.append(str(Path(__file__).parent / "src"))

from agent import chat

if __name__ == "__main__":
    chat()
