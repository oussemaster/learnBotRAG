"""
config.py – Zentrale Konfiguration
===================================
Alle Konstanten, Pfade und Modell-Namen an einem Ort.
Andere Module importieren ausschließlich von hier – keine hartkodierten Werte.
"""

from pathlib import Path

# ── Modell ─────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "gpt-4o-mini"
TOP_K_DOCUMENTS: int = 5

# ── Verzeichnisse & Pfade ──────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.parent          # Projekt-Root
DATA_DIR: Path = BASE_DIR / "data"                     # Quelldokumente

VECTOR_DB_PATH: Path = BASE_DIR / "vector_db"         # Chroma-Persistenz-Ordner
CHROMA_SENTINEL: Path = VECTOR_DB_PATH / "chroma.sqlite3"   # Indikator: DB existiert
PROCESSED_FILES_PATH: Path = VECTOR_DB_PATH / "processed_files.json"  # Hash-Tracker