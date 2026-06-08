"""
core/loaders.py – Loader-Factory und Dokument-Einlese-Logik
=============================================================
Verantwortlich für:
  - LOADER_BY_SUFFIX: Erweiterbare Factory (Open/Closed Principle).
    Neues Format? Nur hier einen Eintrag ergänzen.
  - discover_data_paths(): Findet alle unterstützten Dateien in data/.
  - load_documents():      Liest eine Liste von Dateipfaden über die Factory ein.
"""

from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Loader-Factory ─────────────────────────────────────────────────────────────
# Suffix → LangChain-Loader-Klasse.
# Neues Format? Nur hier einen Eintrag ergänzen – load_documents bleibt unverändert.
LOADER_BY_SUFFIX: dict[str, type] = {
    ".csv": CSVLoader,
    ".pdf": PyPDFLoader,
}


def discover_data_paths(data_dir: Path = DATA_DIR) -> list[Path]:
    """
    Sammelt alle unterstützten Dateien in *data_dir* (Reihenfolge stabil via sort).

    Args:
        data_dir: Verzeichnis, das rekursiv durchsucht wird.
                  Standard: DATA_DIR aus config.py.

    Returns:
        Sortierte Liste der gefundenen Dateipfade.
    """
    if not data_dir.is_dir():
        log.warning("Datenverzeichnis nicht gefunden: %s", data_dir)
        return []
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in LOADER_BY_SUFFIX
    )


def load_documents(paths: list[Path]) -> list[Document]:
    """
    Loads files using the Loader Factory (Suffix -> Loader Class).

    Supported Formats:
        - `.csv` -> CSVLoader (one row = one document)
        - `.pdf` -> PyPDFLoader (mode=page -> one page = one document)

    Unknown extensions will be skipped; corrupted files will be logged
    without interrupting the entire ingestion pipeline.
    """
    raw_documents: list[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        loader_cls = LOADER_BY_SUFFIX.get(suffix)
        if loader_cls is None:
            log.warning("    [✗] Nicht unterstützt: %s – übersprungen.", path.name)
            continue
        try:
            raw_documents.extend(loader_cls(str(path)).load())
            log.info("    [✓] Geladen: %s (%s)", path.name, loader_cls.__name__)
        except FileNotFoundError:
            log.warning("    [✗] Datei nicht gefunden: %s – übersprungen.", path)
        except Exception as exc:  # noqa: BLE001
            log.error("    [✗] Fehler beim Laden von '%s': %s", path, exc)
    if not raw_documents:
        return raw_documents

    log.info("    [✂️] Zerschneide Dokumente in kleinere Chunks ...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )

    split_docs = text_splitter.split_documents(raw_documents)
    return split_docs
