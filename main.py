"""
main.py – Entrypoint
======================
Datenfluss (inkrementell):

    ┌──────────────────────────────────────────────────────────────────┐
    │  data/ vs. processed_files.json (Hash-Tracker) – Mirror          │
    │      → Cleanup: im Tracker, aber nicht mehr auf Disk → löschen   │
    │      → Delta: neue oder geänderte Dateien                        │
    │                                                                  │
    │  DB fehlt        → Vollständiger Erstaufbau aller Dateien        │
    │  Delta leer      → Chroma.load() (kein Embedding-API-Call)       │
    │  Delta nicht leer→ Chroma.load() + add_documents(nur Delta)      │
    └──────────────────────────────────────────────────────────────────┘
         │
         ▼  retriever (Top-k via Kosinus-Ähnlichkeit)
         │  format_docs()   → context: str
         │  answer_chain    → LLM
         ▼
    answer: str

Dieses Modul ist ausschließlich für Orchestrierung zuständig.
Keine Business-Logik – nur Initialisierung, Verkabelung und Start.
"""

from dotenv import load_dotenv

load_dotenv()

from src.config import MODEL_NAME, TOP_K_DOCUMENTS  # noqa: E402
from src.utils.logger import get_logger, setup_logging  # noqa: E402

setup_logging()
log = get_logger(__name__)


def main() -> None:
    """Orchestriert den Programmstart: Logging → Retriever → Chat."""
    # Imports hier, damit setup_logging() + load_dotenv() zuerst laufen
    from src.cli import run_interactive_chat
    from src.core.vectorstore import get_or_create_retriever

    try:
        log.info("\nModell: %s", MODEL_NAME)
        log.info("%s", "=" * 70)
        log.info("  RAG Demo – Retrieval-Augmented Generation (Persistent)")
        log.info("%s\n", "=" * 70)

        retriever = get_or_create_retriever(k=TOP_K_DOCUMENTS)
        run_interactive_chat(retriever)
    except KeyboardInterrupt:
        log.info("\n[👋] Abgebrochen (Ctrl+C). Auf Wiedersehen!")


if __name__ == "__main__":
    main()