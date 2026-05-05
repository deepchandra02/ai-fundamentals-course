"""The RAG glue: retrieve relevant chunks, then ask the LLM."""

from . import config, embed, store

PROMPT_TEMPLATE = """\
Using ONLY this context, answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


def ask(question: str, top_k: int = 3) -> str:
    """Answer `question` using whatever has been ingested into the vector store."""
    collection = store.get_collection()
    if collection.count() == 0:
        return "The index is empty. Run `python main.py ingest` first."

    [query_embedding] = embed.embed_texts([question])
    hits = store.search(collection, query_embedding, top_k=top_k)

    context_blocks = [f"[{h['source']}]\n{h['text']}" for h in hits]
    prompt = PROMPT_TEMPLATE.format(
        context="\n\n---\n\n".join(context_blocks),
        question=question,
    )

    client = config.get_client()
    response = client.chat.completions.create(
        model=config.CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers strictly from the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content
