"""Splits long text into overlapping chunks of (roughly) fixed size."""


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks of `size` characters with `overlap` shared characters.

    We chunk on characters here for simplicity. Real systems often chunk on tokens
    or sentences — but this is enough to see the idea (and matches the slides).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    step = size - overlap
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks
