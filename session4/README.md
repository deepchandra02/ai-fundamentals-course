# Session 4 — Hands-On GenAI: Build Your Own Mini RAG 🛠️

This is the hands-on companion to Session 3. Where Session 3 built the **vocabulary** (tokens, embeddings, RAG, etc.), Session 4 builds the **code**.

## What's here

| File | Purpose |
| --- | --- |
| `session4.ipynb` | The teaching notebook — walks through every concept from Session 3 in runnable code. **Open this first.** |
| `mini_rag/` | A small Python package — the same code from the notebook, organized into modules so you can extend it. |
| `main.py` | CLI entry point: `python main.py ingest` and `python main.py ask "..."`. |
| `corpus/` | Drop your own `.md` / `.txt` files (or folders) here. The RAG indexes whatever's in this folder. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for your Azure OpenAI credentials — copy to `.env` and fill in. |

## Setup (one time)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy the env template and fill in your Azure OpenAI keys
cp .env.example .env
# then edit .env

# 3. Open the notebook
jupyter notebook session4.ipynb
```

## Try the project from the CLI

```bash
# Drop a file into corpus/, then:
python main.py ingest
python main.py ask "What is our refund policy?"
```

## Module map (for tinkering)

- **`config.py`** — loads your `.env`, builds one `AzureOpenAI` client.
- **`chunk.py`** — splits long text into overlapping chunks.
- **`embed.py`** — calls Azure OpenAI to turn text into vectors.
- **`store.py`** — wraps ChromaDB (the vector database).
- **`ingest.py`** — walks `corpus/`, chunks every file, embeds, stores.
- **`rag.py`** — `ask(question)`: retrieves relevant chunks and asks the LLM.

Each file is intentionally short and boring. Open them, change them, see what happens.
