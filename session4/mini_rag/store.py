"""Wraps ChromaDB — our local vector database."""

from pathlib import Path

import chromadb

DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "mini_rag"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_collection(reset: bool = False):
    """Return the collection. If `reset=True`, delete it first (used by ingest)."""
    client = get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet — fine
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_chunks(
    collection,
    ids: list[str],
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search(collection, query_embedding: list[float], top_k: int = 3) -> list[dict]:
    """Return the top_k most similar chunks as a list of dicts."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "source": meta.get("source", "?"), "distance": dist})
    return hits
