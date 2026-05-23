"""
core/rag.py – RAG-Pipeline
===========================
Verantwortlich für:
  - format_docs():       Konvertiert Dokument-Liste in einen Kontext-String für den Prompt.
  - build_answer_chain(): LCEL-Factory: Prompt → LLM → StrOutputParser.
  - ask():               Vollständige RAG-Anfrage (Retrieve → Format → Generate).
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from src.config import MODEL_NAME
from src.utils.logger import get_logger

log = get_logger(__name__)


def format_docs(docs: list[Document]) -> str:
    """
    Wandelt eine Liste von Dokumenten in einen einzigen Kontext-String um.

    Warum: Der Prompt erwartet ``{context}`` als String, kein Python-Objekt.
    Leerzeilen als Trenner helfen dem LLM, Quellen auseinanderzuhalten.

    Args:
        docs: Liste der abgerufenen Dokument-Chunks.

    Returns:
        Kontext-String, bereit für den Prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_answer_chain() -> Runnable:
    """
    Erstellt die LCEL-Pipeline: Prompt → LLM → String-Parser.

    Warum als Factory: Hält die Chain lokal und verhindert versteckte
    globale Zustände, die beim Testen schwer zu mocken sind.

    Returns:
        Ausführbare LCEL-Chain, die ``{"question": ..., "context": ...}``
        als Input erwartet und einen String zurückgibt.
    """
    system_prompt = (
        "Du bist ein intelligenter und akademischer Studien-Assistent.\n"
        "Deine Aufgabe ist es, Fachbegriffe, Konzepte und Fragen basierend auf "
        "den Vorlesungsskripten des Nutzers zu erklären.\n"
        "Nutze AUSSCHLIESSLICH den folgenden Kontext, um die Frage zu beantworten. "
        "Wenn du die Antwort im Kontext absolut nicht findest, antworte mit: "
        "'Diese Information ist in den aktuellen Skripten nicht enthalten.'\n\n"
        "Kontext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    return prompt | llm | StrOutputParser()


def ask(
    question: str,
    retriever: VectorStoreRetriever,
    answer_chain: Runnable,
) -> tuple[str, list[Document]]:
    """
    Führt eine vollständige RAG-Anfrage durch.

    Ablauf: Frage → Retriever → relevante Docs → Kontext-String → LLM → Antwort.

    Args:
        question:     Die Nutzerfrage als String.
        retriever:    Konfigurierter VectorStore-Retriever.
        answer_chain: LCEL-Chain aus :func:`build_answer_chain`.

    Returns:
        Tupel ``(answer, relevant_docs)``:
          - *answer*:        Generierte Antwort als String.
          - *relevant_docs*: Abgerufene Dokument-Chunks (für Quellenangaben).
    """
    relevant_docs: list[Document] = retriever.invoke(question)
    context: str = format_docs(relevant_docs)
    answer: str = answer_chain.invoke({"question": question, "context": context})
    return answer, relevant_docs