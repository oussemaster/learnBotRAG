"""
core/tracker.py – Hash-Tracking, Delta-Berechnung und Cleanup-Logik
=====================================================================
Verantwortlich für:
  - processed_files.json lesen/schreiben (Registry).
  - Delta berechnen: neue oder geänderte Dateien seit letztem Ingest.
  - Mirror-Cleanup: Einträge entfernen, deren Quelldatei gelöscht wurde.

Die eigentliche Chroma-Lösch-Operation (_delete_documents_for_paths) wird von
vectorstore.py geliefert und hier als Callback entgegengenommen, damit tracker.py
keine direkte Abhängigkeit auf Chroma hat (Dependency Inversion).
"""

import hashlib
import json
from pathlib import Path
from typing import Callable

from src.config import DATA_DIR, PROCESSED_FILES_PATH, VECTOR_DB_PATH
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Interne Hilfsfunktionen ────────────────────────────────────────────────────

def _tracker_key(path: Path) -> str:
    """Eindeutiger Schlüssel im Tracker (Dateiname unter data/)."""
    return path.name


# ── Hashing ────────────────────────────────────────────────────────────────────

def file_content_hash(path: Path) -> str:
    """
    Berechnet den SHA-256-Hash des Dateiinhalts.

    Erkennt geänderte Dateien mit gleichem Namen zuverlässig.

    Args:
        path: Zu hashende Datei.

    Returns:
        Hex-Digest des SHA-256-Hashes.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Registry I/O ───────────────────────────────────────────────────────────────

def load_processed_registry() -> dict[str, str]:
    """
    Lädt ``filename → content_hash`` aus *processed_files.json*.

    Returns:
        Dict mit den bekannten Dateien und ihren Hashes,
        oder ein leeres Dict wenn der Tracker fehlt oder beschädigt ist.
    """
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
    """
    Persistiert den Tracker als JSON neben der Chroma-DB.

    Args:
        registry: Aktueller Stand des ``filename → hash``-Mappings.
    """
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILES_PATH.write_text(
        json.dumps(registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ── Registry-Operationen ───────────────────────────────────────────────────────

def build_registry_for_paths(paths: list[Path]) -> dict[str, str]:
    """
    Erzeugt ``filename → hash``-Einträge für die übergebenen Pfade.

    Args:
        paths: Dateipfade, für die Einträge erstellt werden sollen.

    Returns:
        Neues Registry-Dict (ohne Seiteneffekte).
    """
    return {_tracker_key(p): file_content_hash(p) for p in paths}


def merge_into_registry(
    registry: dict[str, str],
    paths: list[Path],
) -> dict[str, str]:
    """
    Fügt erfolgreich verarbeitete Dateien in den Tracker ein (immutable).

    Args:
        registry: Bestehender Tracker-Stand.
        paths:    Neu verarbeitete Dateipfade.

    Returns:
        Aktualisiertes Registry-Dict.
    """
    updated = dict(registry)
    updated.update(build_registry_for_paths(paths))
    return updated


def remove_keys_from_registry(
    registry: dict[str, str],
    keys: list[str],
) -> dict[str, str]:
    """
    Entfernt Einträge aus dem Tracker (ohne zu persistieren).

    Args:
        registry: Bestehender Tracker-Stand.
        keys:     Zu entfernende Dateinamen-Schlüssel.

    Returns:
        Bereinigtes Registry-Dict.
    """
    keys_set = set(keys)
    return {k: v for k, v in registry.items() if k not in keys_set}


# ── Delta-Berechnung ───────────────────────────────────────────────────────────

def compute_ingest_delta(
    all_paths: list[Path],
    registry: dict[str, str],
) -> list[Path]:
    """
    Berechnet das Ingest-Delta: Dateien, die neu sind oder deren Hash sich geändert hat.

    Args:
        all_paths: Alle aktuell auf Disk gefundenen Dateipfade.
        registry:  Bekannter Stand aus dem Tracker.

    Returns:
        Liste der Pfade, die (erneut) indexiert werden müssen.
    """
    delta: list[Path] = []
    for path in all_paths:
        key = _tracker_key(path)
        if registry.get(key) != file_content_hash(path):
            delta.append(path)
    return delta


def compute_deleted_files(
    registry: dict[str, str],
    all_paths: list[Path],
) -> list[str]:
    """
    Findet Dateinamen im Tracker, die im Dateisystem (data/) nicht mehr existieren.

    Ermöglicht Mirror-Sync: DB + Tracker spiegeln den Inhalt von data/.

    Args:
        registry:  Aktueller Tracker-Stand.
        all_paths: Alle aktuell auf Disk gefundenen Dateipfade.

    Returns:
        Sortierte Liste verwaister Tracker-Schlüssel.
    """
    present_keys = {_tracker_key(p) for p in all_paths}
    return sorted(key for key in registry if key not in present_keys)


def paths_for_tracker_keys(keys: list[str]) -> list[Path]:
    """
    Rekonstruiert absolute Pfade aus Tracker-Schlüsseln.

    Wird für die Chroma-Löschung benötigt – die Datei muss nicht mehr existieren.

    Args:
        keys: Dateinamen-Schlüssel aus dem Tracker.

    Returns:
        Liste absolut aufgelöster Pfade unter DATA_DIR.
    """
    return [(DATA_DIR / key).resolve() for key in keys]


# ── Mirror-Cleanup ─────────────────────────────────────────────────────────────

def cleanup_deleted_files(
    registry: dict[str, str],
    all_paths: list[Path],
    delete_vectors_fn: Callable[[list[Path]], None],
) -> dict[str, str]:
    """
    Entfernt verwaiste Vektoren und Tracker-Einträge für physisch gelöschte Dateien.

    Läuft ohne Embedding-API-Calls.

    Die eigentliche Vektoren-Löschung wird über *delete_vectors_fn* delegiert,
    damit dieser Modul keine direkte Chroma-Abhängigkeit hat.

    Args:
        registry:          Aktueller Tracker-Stand.
        all_paths:         Alle aktuell auf Disk gefundenen Dateipfade.
        delete_vectors_fn: Callback, der eine Liste von Pfaden aus Chroma entfernt.
                           Signatur: ``(paths: list[Path]) -> None``

    Returns:
        Bereinigter Tracker-Stand (bereits persistiert).
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

    delete_vectors_fn(paths_for_tracker_keys(deleted_keys))
    updated = remove_keys_from_registry(registry, deleted_keys)
    save_processed_registry(updated)
    log.info("[✓] %d Eintrag/Einträge aus Tracker entfernt.", n)
    return updated