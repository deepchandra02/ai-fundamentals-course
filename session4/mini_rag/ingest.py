"""Walks a corpus directory, chunks every file, embeds chunks, stores them."""

from pathlib import Path

from . import chunk, embed, store

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def iter_corpus_files(corpus_dir: str | Path):
    """Yield every supported file under corpus_dir, recursively.

    Skips the `corpus/README.md` placeholder so the demo file doesn't pollute the index.
    """
    root = Path(corpus_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.parent == root and path.name.lower() == "readme.md":
            continue
        yield path


def ingest(corpus_dir: str | Path = "corpus", chunk_size: int = 500, overlap: int = 50) -> int:
    """Index every supported file in `corpus_dir`. Returns the number of chunks indexed."""
    root = Path(corpus_dir).resolve()
    files = list(iter_corpus_files(root))
    if not files:
        print(f"No .md or .txt files found in {root}. Drop some in and try again.")
        return 0

    collection = store.get_collection(reset=True)

    all_ids: list[str] = []
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        relative = str(path.relative_to(root))
        for i, piece in enumerate(chunk.chunk_text(text, size=chunk_size, overlap=overlap)):
            all_ids.append(f"{relative}::chunk-{i}")
            all_chunks.append(piece)
            all_metadatas.append({"source": relative})

    if not all_chunks:
        print("Found files but they were empty. Nothing to index.")
        return 0

    print(f"Embedding {len(all_chunks)} chunks from {len(files)} file(s)...")
    embeddings = embed.embed_texts(all_chunks)
    store.add_chunks(collection, all_ids, all_chunks, embeddings, all_metadatas)
    print(f"Indexed {len(all_chunks)} chunks. Ready to ask questions!")
    return len(all_chunks)
