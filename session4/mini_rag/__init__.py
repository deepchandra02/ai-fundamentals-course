"""mini_rag — a tiny, readable RAG pipeline built in Session 4."""

from .ingest import ingest
from .rag import ask

__all__ = ["ingest", "ask"]
