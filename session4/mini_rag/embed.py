"""Turn text into vectors using the Azure OpenAI embeddings API."""

from . import config


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input string."""
    if not texts:
        return []
    client = config.get_client()
    response = client.embeddings.create(
        model=config.EMBED_DEPLOYMENT,
        input=texts,
    )
    return [item.embedding for item in response.data]
