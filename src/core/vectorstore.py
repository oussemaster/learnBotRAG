"""
core/vectorstore.py – ChromaDB-Management
==========================================
Verantwortlich für:
  - Chroma öffnen / Bootstrap (Erstaufbau).
  - Inkrementelles Delta ingestieren.
  - Interne Chroma-Hilfen (Vektor-Count, Source-Abfrage, Löschung).
  - initialize_vector_store(): Entscheidungslogik DB-Aufbau vs. Delta vs. reines Laden.
  - get_or_create_retriever(): Öffentliche Fassade für main.py.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from src.config import (
    CHROMA_SENTINEL,
    DATA_DIR,
    TOP_K_DOCUMENTS,
    VECTOR_DB_PATH,
    CHROMA_BATCH_SIZE,
)
from src.core.loaders import discover_data_paths, load_documents, split_documents
from src.core.tracker import (
    build_registry_for_paths,
    cleanup_deleted_files,
    compute_ingest_delta,
    load_processed_registry,
    merge_into_registry,
    save_processed_registry,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Interne Chroma-Hilfen ──────────────────────────────────────────────────────


def _db_exists() -> bool:
    """
    Prüft, ob Chroma bereits eine befüllte DB auf Disk hat.

    Warum CHROMA_SENTINEL statt nur VECTOR_DB_PATH.exists():
    Chroma legt den Ordner manchmal leer an – die sqlite3-Datei ist
    der zuverlässige Indikator für eine vollständig initialisierte DB.
    """
    return CHROMA_SENTINEL.exists()


def _open_chroma_store(embeddings: Embeddings) -> Chroma:
    """Öffnet den persistenten Chroma-Store (ohne Embedding-API-Calls)."""
    return Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings,
    )


def _vector_count(store: Chroma) -> int:
    """Gibt die Anzahl der indexierten Vektoren zurück."""
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
    """
    Entfernt Vektoren einer Quelldatei vor Re-Ingest (z. B. nach Dateiänderung).

    Wird als Callback an tracker.cleanup_deleted_files übergeben.

    Args:
        store: Geöffneter Chroma-Store.
        paths: Pfade, deren Vektoren gelöscht werden sollen.
    """
    for path in paths:
        resolved = str(path.resolve())
        store._collection.delete(where={"source": resolved})


def _add_documents_in_batches(store: Chroma, documents: list[Document]) -> None:
    """
    Fügt Dokumente in Chroma in kleinen Stapeln (Batches) hinzu.

    Warum Batch-Add statt Einzel-Add? Reduziert Speicher- und API-Last,
    besonders bei großen Dokumenten oder vielen Chunks.

    Args:
        store:     Geöffneter Chroma-Store.
        documents: Liste der Dokumente, die hinzugefügt werden sollen.
    """
    total_documents = len(documents)
    total_batches = total_documents + CHROMA_BATCH_SIZE - 1
    for i in range(0, total_documents, CHROMA_BATCH_SIZE):
        batch = documents[i : i + CHROMA_BATCH_SIZE]
        store.add_documents(batch)
        log.info(
            "      - Batch %d/%d  (%d Chunks) indexiert.",
            (i // CHROMA_BATCH_SIZE) + 1,
            total_batches,
            len(batch),
        )


# ── Tracker-Fallback ───────────────────────────────────────────────────────────


def reconcile_registry_from_chroma(
    store: Chroma,
    all_paths: list[Path],
) -> dict[str, str]:
    """
    Fallback: Tracker fehlt, DB existiert → bekannte Quellen aus Chroma + Hashes von Disk.

    Verhindert Voll-Re-Ingest nach Löschen nur von processed_files.json.

    Args:
        store:     Geöffneter Chroma-Store.
        all_paths: Alle aktuell auf Disk gefundenen Dateipfade.

    Returns:
        Rekonstruiertes Registry-Dict (bereits persistiert).
    """
    from src.core.tracker import _tracker_key  # lokaler Import verhindert Zirkularität

    indexed_sources = _source_paths_in_store(store)
    registry: dict[str, str] = {}
    for path in all_paths:
        if str(path.resolve()) in indexed_sources:
            from src.core.tracker import file_content_hash

            registry[_tracker_key(path)] = file_content_hash(path)
    if registry:
        save_processed_registry(registry)
        log.info(
            "[↻] Tracker aus Chroma-Metadaten rekonstruiert (%d Dateien).",
            len(registry),
        )
    return registry


# ── Pipeline-Schritte ──────────────────────────────────────────────────────────


def _bootstrap_store(
    embeddings: Embeddings,  # noqa: ARG001  (Signatur-Konsistenz)
    paths: list[Path],
) -> Chroma:
    """
    Erstaufbau: alle Dateien laden, embedden und Tracker schreiben.

    Args:
        embeddings: OpenAI-Embeddings-Instanz.
        paths:      Alle zu indexierenden Dateipfade.

    Returns:
        Fertig befüllter und persistierter Chroma-Store.

    Raises:
        RuntimeError: Wenn keine Dokumente geladen werden konnten.
    """
    log.info("[🆕] Keine Vektordatenbank gefunden. Erstelle neue DB …")
    log.info("[1/3] Lade %d Datei(en) aus %s:", len(paths), DATA_DIR)

    documents = load_documents(paths)
    if not documents:
        raise RuntimeError(
            "Keine Dokumente geladen – DB kann nicht erstellt werden. "
            "Bitte prüfe, ob unterstützte Dateien (.csv, .pdf) in data/ liegen."
        )
    documents = split_documents(documents)
    log.info("      %d Dokument-Chunks insgesamt geladen.", len(documents))

    log.info("[2/3] Berechne Embeddings und baue Vektorindex auf …")
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    ## 1. Eine leere Datenbank erstellen
    store = Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings,
    )

    # 2. Dokumente in kleinen Stapeln (Batches) hinzufügen, um Speicher- und API-Last zu reduzieren
    _add_documents_in_batches(store, documents)

    save_processed_registry(build_registry_for_paths(paths))
    log.info("[3/3] Datenbank persistent gespeichert unter: %s", VECTOR_DB_PATH)
    log.info("[✓] %d Vektoren indexiert.\n", _vector_count(store))
    return store


def _ingest_delta(
    store: Chroma,
    embeddings: Embeddings,
    delta_paths: list[Path],
    registry: dict[str, str],
) -> Chroma:
    """
    Inkrementell: geänderte Quellen entfernen, neue Chunks embedden und hinzufügen.

    Args:
        store:       Geöffneter Chroma-Store.
        embeddings:  Nicht direkt genutzt; vorhanden für Konsistenz der Signatur.
        delta_paths: Pfade der neuen / geänderten Dateien.
        registry:    Aktueller Tracker-Stand.

    Returns:
        Aktualisierter Chroma-Store (in-place mutiert, zur Klarheit zurückgegeben).
    """
    from src.core.tracker import _tracker_key  # lokaler Import verhindert Zirkularität

    changed_keys = {_tracker_key(p) for p in delta_paths if _tracker_key(p) in registry}
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

    documents = split_documents(documents)
    log.info("      %d Dokument-Chunks zum Indexieren.", len(documents))
    log.info("[2/2] Berechne Embeddings und füge zum Store hinzu …")

    _add_documents_in_batches(store, documents)

    save_processed_registry(merge_into_registry(registry, delta_paths))
    log.info("[✓] Delta ingestiert. %d Vektoren gesamt.\n", _vector_count(store))
    return store


# ── Öffentliche API ────────────────────────────────────────────────────────────


def initialize_vector_store(embeddings: Embeddings) -> Chroma:
    """
    Entscheidet: Erstaufbau, Mirror-Cleanup, inkrementelles Delta oder reines Laden.

    Ablauf bei bestehender DB:
      1. Tracker laden.
      2. Verwaiste Dateien bereinigen (Mirror-Sync).
      3. Delta ingestieren (falls nötig).

    Args:
        embeddings: Embeddings-Instanz.

    Returns:
        Einsatzbereiter Chroma-Store.

    Raises:
        RuntimeError: Wenn keine DB und keine Daten vorhanden sind.
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

    # Mirror-Cleanup: Callback kapselt die Chroma-spezifische Lösch-Operation
    registry = cleanup_deleted_files(
        registry=registry,
        all_paths=all_paths,
        delete_vectors_fn=lambda paths: _delete_documents_for_paths(store, paths),
    )

    delta_paths = compute_ingest_delta(all_paths, registry)
    if not delta_paths:
        log.info("[↩] Keine neuen Dateien gefunden. Lade DB …")
        log.info(
            "[✓] %d Vektoren verfügbar (ohne Embedding-API).\n", _vector_count(store)
        )
        return store

    return _ingest_delta(store, embeddings, delta_paths, registry)


def get_or_create_retriever(k: int = TOP_K_DOCUMENTS) -> VectorStoreRetriever:
    """
    Gibt einen Retriever zurück – mit inkrementeller Ingestion:

      - Keine DB     → Vollständiger Erstaufbau
      - Gelöscht     → Vektoren + Tracker-Einträge entfernen (Mirror)
      - Delta leer   → Nur Chroma.load() (kein Embedding-API-Call)
      - Delta > 0    → load + add_documents(nur neue/geänderte Dateien)

    Args:
        k: Anzahl der zurückgegebenen Dokumente pro Anfrage (Top-k).

    Returns:
        Konfigurierter :class:`~langchain_core.vectorstores.VectorStoreRetriever`.
    """

    from src.core.providers import get_embeddings

    embeddings: Embeddings = get_embeddings()
    store = initialize_vector_store(embeddings)
    return store.as_retriever(search_kwargs={"k": k})
