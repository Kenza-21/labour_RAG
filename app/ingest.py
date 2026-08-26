"""CLI that runs pipeline 1 (ingestion): PDF/text -> chunks -> vectors -> store."""

import argparse

from app.chunking import chunk_text
from app.embeddings import embed_texts
from app.store import get_store


def read_source(source: str) -> str:
    if source.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(source)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(source, encoding="utf-8") as f:
        return f.read()


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def ingest(source: str, reset: bool = False, batch_size: int = 100) -> int:
    raw = read_source(source)
    chunks = chunk_text(raw)
    print(f"{len(chunks)} chunks produced from {source}")

    store = get_store()
    store.setup(reset=reset)

    for batch in batched(chunks, batch_size):
        vectors = embed_texts([c.content for c in batch])
        rows = [{"article": c.article, "content": c.content} for c in batch]
        store.add(rows, vectors)

    total = store.count()
    print(f"{total} chunks stored")
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest a document into the vector store.")
    parser.add_argument("--source", required=True, help="Path to a .pdf or .txt file")
    parser.add_argument("--reset", action="store_true", help="Wipe the store before ingesting")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    ingest(args.source, reset=args.reset, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
