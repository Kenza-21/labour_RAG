"""API request/response contracts. Pydantic validates every request before it reaches rag.py."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)


class Source(BaseModel):
    article: str | None
    excerpt: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    abstained: bool
    tool_calls: list[dict] = []


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)


class SearchResponse(BaseModel):
    hits: list[Source]


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
