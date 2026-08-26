"""Glues chunking/embeddings/store/tools/llm together: retrieve, guard, prompt, generate."""

from app.config import settings
from app.embeddings import embed_query
from app.llm import complete
from app.store import SearchHit, VectorStore

ABSTENTION_MESSAGE = (
    "Je ne dispose pas d'assez d'information dans le Code du travail pour repondre "
    "a cette question avec certitude."
)

SYSTEM_PROMPT = f"""Tu es un assistant specialise dans le Code du Travail marocain.
1. Reponds UNIQUEMENT a partir des extraits fournis dans le CONTEXTE.
2. Cite le numero d'article entre parentheses apres chaque affirmation.
3. N'invente JAMAIS un numero d'article, un chiffre ou une regle.
4. Si le CONTEXTE ne contient pas la reponse, ecris exactement : "{ABSTENTION_MESSAGE}"
"""


def retrieve(question: str, k: int, store: VectorStore) -> list[SearchHit]:
    vector = embed_query(question)
    return store.search(vector, k)


def build_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(f"[Article {h.article}]\n{h.content}" for h in hits)


def build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}"},
    ]


def answer(question: str, store: VectorStore, k: int | None = None) -> dict:
    k = k or settings.top_k
    hits = retrieve(question, k, store)

    if not hits or hits[0].distance > settings.max_distance:
        return {"answer": ABSTENTION_MESSAGE, "sources": [], "abstained": True, "tool_calls": []}

    context = build_context(hits)
    messages = build_messages(question, context)
    text, tool_trace = complete(messages, use_tools=True)

    return {
        "answer": text,
        "sources": [
            {"article": h.article, "excerpt": h.content[:300], "distance": h.distance}
            for h in hits
        ],
        "abstained": False,
        "tool_calls": tool_trace,
    }
