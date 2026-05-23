"""
RAG – Retrieval-Augmented Generation (Persistente Vektordatenbank)
==================================================================

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
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import hashlib
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)  # Chroma-Interna stumm schalten

# ── Konfiguration ──────────────────────────────────────────────────────────────
MODEL_NAME       = "gpt-4o-mini"
TOP_K_DOCUMENTS  = 5
VECTOR_DB_PATH   = Path("./vector_db")          # Persistenz-Ordner für Chroma
CHROMA_SENTINEL  = VECTOR_DB_PATH / "chroma.sqlite3"  # Existenz-Marker der DB
PROCESSED_FILES_PATH = VECTOR_DB_PATH / "processed_files.json"

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Factory: Suffix → LangChain-Loader-Klasse (Open/Closed Principle).
# Neues Format? Nur hier einen Eintrag ergänzen – load_documents bleibt unverändert.
LOADER_BY_SUFFIX: dict[str, type] = {
    ".csv": CSVLoader,
    ".pdf": PyPDFLoader,
}


def discover_data_paths(data_dir: Path = DATA_DIR) -> list[Path]:
    """Sammelt alle unterstützten Dateien in data_dir (Reihenfolge stabil via sort)."""
    if not data_dir.is_dir():
        log.warning("Datenverzeichnis nicht gefunden: %s", data_dir)
        return []
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in LOADER_BY_SUFFIX
    )


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    """
    Wandelt Dokumente in einen einzigen Kontext-String um.

    Warum: Der Prompt erwartet {context} als String, kein Python-Objekt.
    Leerzeilen als Trenner helfen dem LLM, Quellen auseinanderzuhalten.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_answer_chain() -> Runnable:
    """
    Erstellt die LCEL-Pipeline: Prompt → LLM → String-Parser.

    Warum als Factory: Hält die Chain lokal und verhindert versteckte
    globale Zustände, die beim Testen schwer zu mocken sind.
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


# ── Tracker (processed_files.json) ─────────────────────────────────────────────

def _tracker_key(path: Path) -> str:
    """Eindeutiger Schlüssel im Tracker (Dateiname unter data/)."""
    return path.name


def file_content_hash(path: Path) -> str:
    """SHA-256 über Dateiinhalt – erkennt geänderte Dateien mit gleichem Namen."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_processed_registry() -> dict[str, str]:
    """Lädt filename → content_hash aus processed_files.json."""
    if not PROCESSED_FILES_PATH.is_file():
        return {}
    try:
        data = json.loads(PROCESSED_FILES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("[⚠] Tracker beschädigt (%s) – wird neu aufgebaut.", exc)
    return {}


def save_processed_registry(registry: dict[str, str]) -> None:
    """Persistiert den Tracker neben der Chroma-DB."""
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILES_PATH.write_text(
        json.dumps(registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_registry_for_paths(paths: list[Path]) -> dict[str, str]:
    """Erzeugt Einträge filename → Hash für die übergebenen Pfade."""
    return {_tracker_key(p): file_content_hash(p) for p in paths}


def compute_ingest_delta(
    all_paths: list[Path],
    registry: dict[str, str],
) -> list[Path]:
    """
    Delta: Dateien, die noch nicht im Tracker sind oder deren Hash sich geändert hat.
    """
    delta: list[Path] = []
    for path in all_paths:
        key = _tracker_key(path)
        if registry.get(key) != file_content_hash(path):
            delta.append(path)
    return delta


def merge_into_registry(
    registry: dict[str, str],
    paths: list[Path],
) -> dict[str, str]:
    """Fügt erfolgreich verarbeitete Dateien in den Tracker ein."""
    updated = dict(registry)
    updated.update(build_registry_for_paths(paths))
    return updated


def compute_deleted_files(
    registry: dict[str, str],
    all_paths: list[Path],
) -> list[str]:
    """
    Dateinamen im Tracker, die im Dateisystem (data/) nicht mehr existieren.

    Ermöglicht Mirror-Sync: DB + Tracker spiegeln den Inhalt von data/.
    """
    present_keys = {_tracker_key(p) for p in all_paths}
    return sorted(key for key in registry if key not in present_keys)


def _paths_for_tracker_keys(keys: list[str]) -> list[Path]:
    """Rekonstruiert absolute Pfade für Chroma-Löschung (Datei muss nicht mehr existieren)."""
    return [(DATA_DIR / key).resolve() for key in keys]


def remove_keys_from_registry(
    registry: dict[str, str],
    keys: list[str],
) -> dict[str, str]:
    """Entfernt Einträge aus dem Tracker (ohne zu persistieren)."""
    keys_set = set(keys)
    return {k: v for k, v in registry.items() if k not in keys_set}


def _cleanup_deleted_files(
    store: Chroma,
    registry: dict[str, str],
    all_paths: list[Path],
) -> dict[str, str]:
    """
    Entfernt verwaiste Vektoren und Tracker-Einträge für physisch gelöschte Dateien.

    Läuft ohne Embedding-API-Calls.
    """
    deleted_keys = compute_deleted_files(registry, all_paths)
    if not deleted_keys:
        return registry

    n = len(deleted_keys)
    log.info(
        "[🗑️] %d gelöschte Datei(en) erkannt. Entferne verwaiste Vektoren aus der Datenbank …",
        n,
    )
    for key in deleted_keys:
        log.info("      - %s", key)

    _delete_documents_for_paths(store, _paths_for_tracker_keys(deleted_keys))
    updated = remove_keys_from_registry(registry, deleted_keys)
    save_processed_registry(updated)
    log.info("[✓] %d Eintrag/Einträge aus Tracker entfernt.", n)
    return updated


# ── Chroma-Hilfen ──────────────────────────────────────────────────────────────

def _db_exists() -> bool:
    """
    Prüft, ob Chroma bereits eine befüllte DB auf Disk hat.

    Warum CHROMA_SENTINEL statt nur VECTOR_DB_PATH.exists():
    Chroma legt den Ordner manchmal leer an – die sqlite3-Datei ist
    der zuverlässige Indikator für eine vollständig initialisierte DB.
    """
    return CHROMA_SENTINEL.exists()


def _open_chroma_store(embeddings: OpenAIEmbeddings) -> Chroma:
    """Öffnet den persistenten Chroma-Store (ohne Embedding-API-Calls)."""
    return Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings,
    )


def _vector_count(store: Chroma) -> int:
    return store._collection.count()


def _source_paths_in_store(store: Chroma) -> set[str]:
    """Alle in Chroma gespeicherten source-Pfade (absolut, wie vom Loader gesetzt)."""
    result = store._collection.get(include=["metadatas"])
    sources: set[str] = set()
    for meta in result.get("metadatas") or []:
        if meta and meta.get("source"):
            sources.add(str(meta["source"]))
    return sources


def _delete_documents_for_paths(store: Chroma, paths: list[Path]) -> None:
    """Entfernt Vektoren einer Quelldatei vor Re-Ingest (z. B. nach Dateiänderung)."""
    for path in paths:
        resolved = str(path.resolve())
        store._collection.delete(where={"source": resolved})


def reconcile_registry_from_chroma(
    store: Chroma,
    all_paths: list[Path],
) -> dict[str, str]:
    """
    Fallback: Tracker fehlt, DB existiert → bekannte Quellen aus Chroma + Hashes von Disk.

    Verhindert Voll-Re-Ingest nach Löschen nur von processed_files.json.
    """
    indexed_sources = _source_paths_in_store(store)
    registry: dict[str, str] = {}
    for path in all_paths:
        if str(path.resolve()) in indexed_sources:
            registry[_tracker_key(path)] = file_content_hash(path)
    if registry:
        save_processed_registry(registry)
        log.info(
            "[↻] Tracker aus Chroma-Metadaten rekonstruiert (%d Dateien).",
            len(registry),
        )
    return registry


# ── Pipeline-Schritte ──────────────────────────────────────────────────────────

def load_documents(paths: list[Path]) -> list[Document]:
    """
    Lädt Dateien über die Loader-Factory (Suffix → Loader-Klasse).

    Unterstützt: .csv (CSVLoader, eine Zeile = ein Document),
                 .pdf (PyPDFLoader, mode=page → eine Seite = ein Document).
    Unbekannte Endungen werden übersprungen; defekte Dateien werden geloggt, nicht abgebrochen.
    """
    documents: list[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        loader_cls = LOADER_BY_SUFFIX.get(suffix)
        if loader_cls is None:
            log.warning("    [✗] Nicht unterstützt: %s – übersprungen.", path.name)
            continue
        try:
            documents.extend(loader_cls(str(path)).load())
            log.info("    [✓] Geladen: %s (%s)", path.name, loader_cls.__name__)
        except FileNotFoundError:
            log.warning("    [✗] Datei nicht gefunden: %s – übersprungen.", path)
        except Exception as exc:  # noqa: BLE001
            log.error("    [✗] Fehler beim Laden von '%s': %s", path, exc)
    return documents


def _bootstrap_store(
    embeddings: OpenAIEmbeddings,
    paths: list[Path],
) -> Chroma:
    """Erstaufbau: alle Dateien laden, embedden, Tracker schreiben."""
    log.info("[🆕] Keine Vektordatenbank gefunden. Erstelle neue DB …")
    log.info("[1/3] Lade %d Datei(en) aus %s:", len(paths), DATA_DIR)

    documents = load_documents(paths)
    if not documents:
        raise RuntimeError(
            "Keine Dokumente geladen – DB kann nicht erstellt werden. "
            "Bitte prüfe, ob unterstützte Dateien (.csv, .pdf) in data/ liegen."
        )
    log.info("      %d Dokument-Chunks insgesamt geladen.", len(documents))

    log.info("[2/3] Berechne Embeddings und baue Vektorindex auf …")
    log.info("      (Dieser Schritt ruft die OpenAI Embeddings API auf.)")
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
    )

    save_processed_registry(build_registry_for_paths(paths))
    log.info("[3/3] Datenbank persistent gespeichert unter: %s", VECTOR_DB_PATH)
    log.info("[✓] %d Vektoren indexiert.\n", _vector_count(store))
    return store


def _ingest_delta(
    store: Chroma,
    embeddings: OpenAIEmbeddings,
    delta_paths: list[Path],
    registry: dict[str, str],
) -> Chroma:
    """Inkrementell: geänderte Quellen entfernen, neue Chunks embedden und hinzufügen."""
    changed_keys = {
        _tracker_key(p)
        for p in delta_paths
        if _tracker_key(p) in registry
    }
    if changed_keys:
        changed_paths = [p for p in delta_paths if _tracker_key(p) in changed_keys]
        log.info(
            "[↻] %d geänderte Datei(en) – entferne alte Vektoren vor Re-Ingest.",
            len(changed_paths),
        )
        _delete_documents_for_paths(store, changed_paths)

    n = len(delta_paths)
    log.info("[+] %d neue Datei(en) gefunden. Füge hinzu …", n)
    log.info("[1/2] Lade Dokumente:")
    documents = load_documents(delta_paths)
    if not documents:
        log.warning("[⚠] Keine Dokumente aus Delta geladen – Tracker unverändert.")
        return store

    log.info("      %d Dokument-Chunks zum Indexieren.", len(documents))
    log.info("[2/2] Berechne Embeddings (OpenAI API) und füge zum Store hinzu …")
    store.add_documents(documents)

    save_processed_registry(merge_into_registry(registry, delta_paths))
    log.info("[✓] Delta ingestiert. %d Vektoren gesamt.\n", _vector_count(store))
    return store


def initialize_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    """
    Entscheidet: Erstaufbau, Mirror-Cleanup, inkrementelles Delta oder reines Laden.

    Ablauf bei bestehender DB: Tracker laden → verwaiste Dateien bereinigen → Delta ingest.
    """
    all_paths = discover_data_paths()

    if not _db_exists():
        if not all_paths:
            raise RuntimeError(
                "Keine Vektordatenbank und keine Daten unter data/. "
                "Lege .csv- oder .pdf-Dateien in data/ ab."
            )
        return _bootstrap_store(embeddings, all_paths)

    log.info("[📂] Bestehende Vektordatenbank: %s", VECTOR_DB_PATH)
    store = _open_chroma_store(embeddings)
    registry = load_processed_registry()
    if not registry:
        registry = reconcile_registry_from_chroma(store, all_paths)

    registry = _cleanup_deleted_files(store, registry, all_paths)

    delta_paths = compute_ingest_delta(all_paths, registry)
    if not delta_paths:
        log.info("[↩] Keine neuen Dateien gefunden. Lade DB …")
        log.info("[✓] %d Vektoren verfügbar (ohne Embedding-API).\n", _vector_count(store))
        return store

    return _ingest_delta(store, embeddings, delta_paths, registry)


def get_or_create_retriever(k: int) -> VectorStoreRetriever:
    """
    Gibt einen Retriever zurück – mit inkrementeller Ingestion:

      • Keine DB     → Vollständiger Erstaufbau
      • Gelöscht     → Vektoren + Tracker-Einträge entfernen (Mirror)
      • Delta leer   → Nur Chroma.load() (kein Embedding-API-Call)
      • Delta > 0    → load + add_documents(nur neue/geänderte Dateien)
    """
    embeddings = OpenAIEmbeddings()
    store = initialize_vector_store(embeddings)
    return store.as_retriever(search_kwargs={"k": k})


def ask(
    question: str,
    retriever: VectorStoreRetriever,
    answer_chain: Runnable,
) -> tuple[str, list[Document]]:
    """
    Führt eine vollständige RAG-Anfrage durch.

    Ablauf: Frage → Retriever → relevante Docs → Kontext-String → LLM → Antwort
    """
    relevant_docs: list[Document] = retriever.invoke(question)
    context: str = format_docs(relevant_docs)
    answer: str = answer_chain.invoke({"question": question, "context": context})
    return answer, relevant_docs


_EXIT_COMMANDS = frozenset({"exit", "quit", "q"})


def _log_answer_and_sources(answer: str, relevant_docs: list[Document]) -> None:
    """Gibt Antwort und Quellenübersicht formatiert im Terminal aus."""
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
    Interaktive RAG-Session: Benutzer stellt Fragen, bis exit/quit/q oder Ctrl+C.

    Leere Eingaben werden übersprungen. Fehler bei einzelnen Anfragen brechen
    die Session nicht ab.
    """
    answer_chain = build_answer_chain()
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


# ── Einstiegspunkt ─────────────────────────────────────────────────────────────

def main() -> None:
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
