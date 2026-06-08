"""
config.py – Zentrale Konfiguration
===================================
Alle Konstanten, Pfade und Modell-Namen an einem Ort.
Andere Module importieren ausschließlich von hier – keine hartkodierten Werte.
"""

from pathlib import Path

# ── Modell ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = "openai"  # Wähle "openai" oder "ollama"

OPENAI_MODEL_NAME: str = "gpt-4o-mini"         # Für OpenAI
LOCAL_MODEL_NAME: str = "llama3.1"      # Für Ollama
LOCAL_EMBEDDING_NAME: str = "nomic-embed-text"

TOP_K_DOCUMENTS: int = 5

# ── Verzeichnisse & Pfade ──────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.parent          # Projekt-Root
DATA_DIR: Path = BASE_DIR / "data"                     # Quelldokumente

VECTOR_DB_PATH: Path = BASE_DIR / f"vector_db_{LLM_PROVIDER}"         # Chroma-Persistenz-Ordner
CHROMA_SENTINEL: Path = VECTOR_DB_PATH / "chroma.sqlite3"   # Indikator: DB existiert
PROCESSED_FILES_PATH: Path = VECTOR_DB_PATH / "processed_files.json"  # Hash-Tracker