"""Mesure la qualite de la RECHERCHE seule (pas de la generation): hit-rate@k et MRR."""

import json
import sys
from pathlib import Path

from app.config import settings
from app.embeddings import embed_query
from app.store import VectorStore, get_store

QUESTIONS_PATH = Path(__file__).parent / "questions.json"

OUT_OF_SCOPE_QUESTIONS = [
    "Quelle est la recette du tajine aux olives ?",
    "Quelle est la capitale du Bresil ?",
    "Comment reparer une voiture ?",
]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(questions: list[dict], store: VectorStore, k: int) -> dict:
    hits = 0
    reciprocal_ranks = []

    for item in questions:
        expected = set(item["expected_articles"])
        results = store.search(embed_query(item["question"]), k=k)
        got_articles = [r.article for r in results]

        hit = any(article in got_articles for article in expected)
        hits += int(hit)

        rank = next((i + 1 for i, a in enumerate(got_articles) if a in expected), 0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    n = len(questions)
    return {
        "hit_rate_at_k": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "n_questions": n,
        "k": k,
    }


def evaluate_abstention(out_of_scope_questions: list[str], store: VectorStore, max_distance: float) -> dict:
    correct = 0
    for q in out_of_scope_questions:
        results = store.search(embed_query(q), k=1)
        abstained = not results or results[0].distance > max_distance
        correct += int(abstained)

    n = len(out_of_scope_questions)
    return {"abstention_rate": correct / n, "n_questions": n}


def main():
    questions = load_questions()
    store = get_store()
    store.setup(reset=False)

    if store.count() == 0:
        print("Le store est vide -- lance d'abord: python -m app.ingest --source <fichier> --reset")
        sys.exit(1)

    retrieval = evaluate_retrieval(questions, store, k=settings.top_k)
    abstention = evaluate_abstention(OUT_OF_SCOPE_QUESTIONS, store, settings.max_distance)

    print(f"Chunks indexes           : {store.count()}")
    print(f"Questions evaluees        : {retrieval['n_questions']}")
    print(f"Hit-rate@{retrieval['k']}                : {retrieval['hit_rate_at_k']:.1%}")
    print(f"MRR                       : {retrieval['mrr']:.3f}")
    print(f"Taux d'abstention (hors-sujet) : {abstention['abstention_rate']:.1%} ({abstention['n_questions']} questions)")


if __name__ == "__main__":
    main()
