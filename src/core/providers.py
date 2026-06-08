"""
core/providers.py – LLM- und Embeddings-Factory
=================================================
Zentraler Ort für alle Provider-Registrierungen.

Neuen Provider hinzufügen? Nur hier zwei Dict-Einträge ergänzen.
"""

from typing import Callable
import httpx

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.config import (
    LLM_PROVIDER,
    LOCAL_EMBEDDING_NAME,
    LOCAL_MODEL_NAME,
    OPENAI_MODEL_NAME,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Registry: provider-name → nullary factory (lazy — importiert erst bei Aufruf)
_LLM_REGISTRY: dict[str, Callable[[], BaseChatModel]] = {}
_EMBEDDINGS_REGISTRY: dict[str, Callable[[], Embeddings]] = {}

OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama_available() -> None:
    """Wirft einen klaren Fehler, wenn der Ollama-Server nicht erreichbar ist."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        response.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Ollama-Server nicht erreichbar unter {OLLAMA_BASE_URL}. "
            "Bitte starte Ollama im Terminal mit: `ollama serve`"
        ) from None


def _register_openai() -> None:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    _LLM_REGISTRY["openai"] = lambda: ChatOpenAI(
        model=OPENAI_MODEL_NAME, temperature=0.0
    )
    _EMBEDDINGS_REGISTRY["openai"] = lambda: OpenAIEmbeddings()


def _register_ollama() -> None:
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    _LLM_REGISTRY["ollama"] = lambda: ChatOllama(
        model=LOCAL_MODEL_NAME, temperature=0.0
    )
    _EMBEDDINGS_REGISTRY["ollama"] = lambda: OllamaEmbeddings(
        model=LOCAL_EMBEDDING_NAME
    )


# Registrierungen beim Modulimport ausführen
_register_openai()
_register_ollama()


def get_llm(provider: str = LLM_PROVIDER) -> BaseChatModel:
    """Gibt das LLM für den konfigurierten Provider zurück."""
    if provider not in _LLM_REGISTRY:
        raise ValueError(
            f"Unbekannter LLM_PROVIDER: '{provider}'. "
            f"Gültige Werte: {sorted(_LLM_REGISTRY)}"
        )
    if provider == "ollama":
        check_ollama_available()
    log.info("[⚙️] LLM-Provider: %s", provider)
    return _LLM_REGISTRY[provider]()


def get_embeddings(provider: str = LLM_PROVIDER) -> Embeddings:
    """Gibt die Embeddings-Instanz für den konfigurierten Provider zurück."""
    if provider not in _EMBEDDINGS_REGISTRY:
        raise ValueError(
            f"Unbekannter LLM_PROVIDER: '{provider}'. "
            f"Gültige Werte: {sorted(_EMBEDDINGS_REGISTRY)}"
        )
    if provider == "ollama":
        check_ollama_available()
    log.info("[⚙️] Embeddings-Provider: %s", provider)
    return _EMBEDDINGS_REGISTRY[provider]()
