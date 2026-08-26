"""Text -> retrieval units. Pure logic: no network calls, no external state."""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    article: str | None = None


PAGE_NOISE_RE = re.compile(r"^[ \t]*(?:page\s*\d+|\d{1,4})[ \t]*$", re.IGNORECASE)
HYPHEN_BREAK_RE = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")
ARTICLE_RE = re.compile(r"^[ \t]*article\s+(\d+|premier)\b", re.IGNORECASE | re.MULTILINE)


def clean_text(raw: str) -> str:
    text = raw.replace(" ", " ")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [line for line in text.split("\n") if not PAGE_NOISE_RE.match(line)]
    return "\n".join(lines).strip()


def chunk_by_article(text: str) -> list[Chunk]:
    matches = list(ARTICLE_RE.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.append(Chunk(content=body, article=m.group(1)))
    return chunks


def chunk_fixed(text: str, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + size, len(text))
        window = text[cursor:end]
        for sep in ("\n\n", ". ", "\n", " "):
            pivot = window.rfind(sep)
            if pivot > size // 2:
                end = cursor + pivot + len(sep)
                break
        chunks.append(Chunk(content=text[cursor:end].strip()))
        cursor = end - overlap
        if end >= len(text):
            break
    return chunks


def chunk_text(raw: str) -> list[Chunk]:
    """Entry point used by ingest.py: clean, try article split, fall back to fixed windows."""
    cleaned = clean_text(raw)
    chunks = chunk_by_article(cleaned)
    if chunks:
        return chunks
    return chunk_fixed(cleaned)
