"""
cli.py – Interaktive Chat-UI
==============================
Verantwortlich für:
  - _log_answer_and_sources(): Formatierte Terminal-Ausgabe von Antwort + Quellen.
  - run_interactive_chat():    REPL-Schleife (Read–Eval–Print-Loop) für den RAG-Chat.

Dieses Modul hat keine Kenntnis von Chroma oder der Ingestion-Pipeline;
es arbeitet ausschließlich mit dem Retriever und der Answer-Chain.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from src.core.rag import ask, build_answer_chain
from src.utils.logger import get_logger

log = get_logger(__name__)

_EXIT_COMMANDS: frozenset[str] = frozenset({"exit", "quit", "q"})


def _log_answer_and_sources(answer: str, relevant_docs: list[Document]) -> None:
    """
    Gibt Antwort und Quellenübersicht formatiert im Terminal aus.

    Args:
        answer:        Generierte Antwort des LLM.
        relevant_docs: Abgerufene Dokument-Chunks mit Metadaten.
    """
    log.info("[↑] Antwort:\n%s\n%s\n%s", "-" * 80, answer, "-" * 80)
    log.info("[🔍] Verwendete Quellen:")
    for d_idx, doc in enumerate(relevant_docs, start=1):
        source = Path(doc.metadata.get("source", "Unbekannt")).name
        loc = doc.metadata.get("row", doc.metadata.get("page", "?"))
        preview = (
            doc.page_content.split("\n")[1]
            if "\n" in doc.page_content
            else doc.page_content[:50]
        )
        log.info(
            "    - [Treffer %d] %s (Zeile/Seite %s) | %s",
            d_idx,
            source,
            loc,
            preview,
        )


def run_interactive_chat(retriever: VectorStoreRetriever) -> None:
    """
    Interaktive RAG-Session: Benutzer stellt Fragen, bis ``exit``/``quit``/``q`` oder Ctrl+C.

    - Leere Eingaben werden übersprungen.
    - Fehler bei einzelnen Anfragen brechen die Session nicht ab.
    - ``EOFError`` (z. B. piped input) beendet die Session sauber.

    Args:
        retriever: Konfigurierter VectorStore-Retriever (aus :mod:`src.core.vectorstore`).
    """
    answer_chain: Runnable = build_answer_chain()
    log.info("Interaktiver Modus – 'exit', 'quit' oder 'q' zum Beenden.\n")

    try:
        while True:
            try:
                raw = input("\nDeine Frage (oder 'exit' zum Beenden): ")
            except EOFError:
                log.info("\n[👋] Eingabe beendet. Auf Wiedersehen!")
                break

            question = raw.strip()
            if question.lower() in _EXIT_COMMANDS:
                log.info("[👋] Auf Wiedersehen!")
                break
            if not question:
                continue

            log.info("%s", "-" * 80)
            log.info("[?] %s", question)
            log.info("[⏳] Suche relevante Dokumente und generiere Antwort …\n")

            try:
                answer, relevant_docs = ask(question, retriever, answer_chain)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("[✗] Fehler bei der Anfrage: %s", exc)
                continue

            _log_answer_and_sources(answer, relevant_docs)

    except KeyboardInterrupt:
        log.info("\n[👋] Abgebrochen (Ctrl+C). Auf Wiedersehen!")