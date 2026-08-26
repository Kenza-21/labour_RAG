from app.embeddings import embed_query
from app.store import JsonStore
from eval.run_eval import evaluate_abstention, evaluate_retrieval


def make_store(tmp_path):
    store = JsonStore(str(tmp_path / "store.json"))
    store.setup(reset=True)
    contents = {
        "231": "Le salarie a droit a un conge annuel paye chaque annee",
        "184": "La duree normale du travail est fixee a quarante quatre heures par semaine",
    }
    store.add(
        rows=[{"article": a, "content": c} for a, c in contents.items()],
        vectors=[embed_query(c) for c in contents.values()],
    )
    return store


def test_evaluate_retrieval_perfect_match_scores_one(tmp_path):
    store = make_store(tmp_path)
    questions = [
        {"question": "Le salarie a droit a un conge annuel paye chaque annee", "expected_articles": ["231"]},
        {"question": "La duree normale du travail est fixee a quarante quatre heures par semaine", "expected_articles": ["184"]},
    ]
    result = evaluate_retrieval(questions, store, k=2)
    assert result["hit_rate_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert result["n_questions"] == 2


def test_evaluate_retrieval_miss_scores_zero(tmp_path):
    store = make_store(tmp_path)
    questions = [{"question": "un texte totalement sans rapport avec le contenu stocke", "expected_articles": ["999"]}]
    result = evaluate_retrieval(questions, store, k=2)
    assert result["hit_rate_at_k"] == 0.0
    assert result["mrr"] == 0.0


def test_evaluate_retrieval_second_place_gives_half_mrr(tmp_path):
    store = make_store(tmp_path)
    # cette question ressemble surtout a l'article 184, mais 231 doit rester dans le top-2
    questions = [{
        "question": "duree normale du travail quarante quatre heures semaine",
        "expected_articles": ["231"],
    }]
    result = evaluate_retrieval(questions, store, k=2)
    assert result["hit_rate_at_k"] == 1.0  # present dans le top-2
    assert result["mrr"] == 0.5  # mais en 2e position, pas en 1ere


def test_evaluate_abstention_all_out_of_scope_detected(tmp_path):
    store = make_store(tmp_path)
    out_of_scope = ["texte completement etranger au corpus stocke ici"]
    result = evaluate_abstention(out_of_scope, store, max_distance=0.3)
    assert result["abstention_rate"] == 1.0
