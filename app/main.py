"""FastAPI wiring: turns rag.py into HTTP endpoints."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import rag
from app.schemas import AskRequest, AskResponse, HealthResponse, SearchRequest, SearchResponse, Source
from app.store import get_store

_STATIC_DIR = Path(__file__).parent / "static"

_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store
    _store = get_store()
    _store.setup(reset=False)
    yield
    _store.close()


app = FastAPI(title="Assistant Code du Travail marocain", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    result = rag.answer(payload.question, store=_store, k=payload.top_k)
    return AskResponse(**result)


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest):
    hits = rag.retrieve(payload.question, payload.top_k or 5, _store)
    return SearchResponse(
        hits=[Source(article=h.article, excerpt=h.content[:300], distance=h.distance) for h in hits]
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", chunks_indexed=_store.count())


# Interface de chat (montée en dernier : /ask, /search, /health gardent la
# priorité). Sert app/static/index.html sur "/".
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")
