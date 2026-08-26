"""Text -> vectors. Fake (offline), OpenAI, or HuggingFace (local, no API key)."""

import hashlib
import math
import re

from app.config import settings


def _fake_embed(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.md5(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _embed_texts_fake(texts: list[str]) -> list[list[float]]:
    return [_fake_embed(t, settings.embedding_dim) for t in texts]


def _embed_texts_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI  # imported lazily: not installed in the fake-only path

    client = OpenAI(api_key=settings.openai_api_key)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 100):
        batch = texts[start:start + 100]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    return vectors


_hf_model = None  # loaded once, reused across calls -- loading it is slow, calling it is fast


def _get_hf_model():
    global _hf_model
    if _hf_model is None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        _hf_model = SentenceTransformer(settings.embedding_model)
    return _hf_model


def _embed_texts_huggingface(texts: list[str]) -> list[list[float]]:
    model = _get_hf_model()
    vectors = model.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
    return [[float(x) for x in v] for v in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if settings.embedding_provider == "fake":
        return _embed_texts_fake(texts)
    if settings.embedding_provider == "huggingface":
        return _embed_texts_huggingface(texts)
    return _embed_texts_openai(texts)


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
