"""Mini RAG CLI.

Usage:
    python main.py ingest                  # index everything in ./corpus
    python main.py ask "your question"     # ask a question
"""

import argparse
import sys

from mini_rag import ask, ingest


def main() -> int:
    parser = argparse.ArgumentParser(description="Mini RAG over a folder of .md/.txt files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index every .md/.txt file under the corpus directory.")
    p_ingest.add_argument("--corpus", default="corpus", help="Path to the corpus folder (default: ./corpus)")
    p_ingest.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters (default: 500)")
    p_ingest.add_argument("--overlap", type=int, default=50, help="Overlap between chunks in characters (default: 50)")

    p_ask = sub.add_parser("ask", help="Ask a question against the indexed corpus.")
    p_ask.add_argument("question", help="The question to ask, in quotes.")
    p_ask.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(corpus_dir=args.corpus, chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.command == "ask":
        answer = ask(args.question, top_k=args.top_k)
        print(answer)

    return 0


if __name__ == "__main__":
    sys.exit(main())
