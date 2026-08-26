"""Where vectors live, and how you search them. Two backends behind one interface."""

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import settings


@dataclass
class SearchHit:
    id: int
    article: str | None
    content: str
    distance: float


class VectorStore(ABC):
    @abstractmethod
    def setup(self, reset: bool = False) -> None: ...

    @abstractmethod
    def add(self, rows: list[dict], vectors: list[list[float]]) -> int: ...

    @abstractmethod
    def search(self, vector: list[float], k: int) -> list[SearchHit]: ...

    @abstractmethod
    def count(self) -> int: ...

    def close(self) -> None:
        pass


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 2.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - similarity


class JsonStore(VectorStore):
    """Backend for tests and offline development: no server, one file on disk."""

    def __init__(self, path: str):
        self.path = path
        self._rows: list[dict] = []

    def setup(self, reset: bool = False) -> None:
        if reset and os.path.exists(self.path):
            os.remove(self.path)
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self._rows = json.load(f)
        else:
            self._rows = []
            self._flush()

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._rows, f)

    def add(self, rows: list[dict], vectors: list[list[float]]) -> int:
        for row, vector in zip(rows, vectors):
            next_id = len(self._rows) + 1
            self._rows.append({
                "id": next_id,
                "article": row.get("article"),
                "content": row["content"],
                "embedding": vector,
            })
        self._flush()
        return len(rows)

    def search(self, vector: list[float], k: int) -> list[SearchHit]:
        scored = [
            SearchHit(
                id=row["id"],
                article=row["article"],
                content=row["content"],
                distance=cosine_distance(vector, row["embedding"]),
            )
            for row in self._rows
        ]
        scored.sort(key=lambda hit: hit.distance)
        return scored[:k]

    def count(self) -> int:
        return len(self._rows)


class PgVectorStore(VectorStore):
    """Backend for production: real Postgres + the pgvector extension."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._conn = None

    def _connect(self):
        import psycopg  # imported lazily: not installed unless you use this backend

        if self._conn is None:
            self._conn = psycopg.connect(self.database_url, autocommit=True)
        return self._conn

    def setup(self, reset: bool = False) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            if reset:
                cur.execute("DROP TABLE IF EXISTS chunks;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    article TEXT,
                    content TEXT NOT NULL,
                    embedding vector({settings.embedding_dim}) NOT NULL
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops);
            """)

    def add(self, rows: list[dict], vectors: list[list[float]]) -> int:
        conn = self._connect()
        with conn.cursor() as cur:
            for row, vector in zip(rows, vectors):
                cur.execute(
                    "INSERT INTO chunks (article, content, embedding) VALUES (%s, %s, %s::vector)",
                    (row.get("article"), row["content"], _vector_literal(vector)),
                )
        return len(rows)

    def search(self, vector: list[float], k: int) -> list[SearchHit]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, article, content, embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (_vector_literal(vector), k),
            )
            return [
                SearchHit(id=r[0], article=r[1], content=r[2], distance=float(r[3]))
                for r in cur.fetchall()
            ]

    def count(self) -> int:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks;")
            return cur.fetchone()[0]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vector) + "]"


def get_store() -> VectorStore:
    if settings.store_backend == "pgvector":
        return PgVectorStore(settings.database_url)
    return JsonStore(os.getenv("JSON_STORE_PATH", "data/store.json"))
