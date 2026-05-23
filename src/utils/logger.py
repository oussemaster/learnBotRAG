"""
utils/logger.py – Logging-Konfiguration
========================================
Zentraler Ort für alle Logging-Einstellungen.

Importiere `get_logger(__name__)` in jedem Modul statt `logging.getLogger(__name__)`,
um einheitliches Format und Unterdrückung von Drittanbieter-Warnungen sicherzustellen.
"""

import logging
import warnings


def setup_logging() -> None:
    """
    Konfiguriert Root-Logger und unterdrückt störende Drittanbieter-Ausgaben.

    Muss einmalig beim Programmstart aufgerufen werden (in main.py).
    Mehrfachaufrufe sind idempotent (basicConfig greift nur beim ersten Mal).
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Drittanbieter-Logger auf WARNING setzen, um die Ausgabe sauber zu halten
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Gibt einen benannten Logger zurück.

    Verwendung in jedem Modul::

        from src.utils.logger import get_logger
        log = get_logger(__name__)
    """
    return logging.getLogger(name)