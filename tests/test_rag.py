from app.embeddings import embed_query
from app.rag import ABSTENTION_MESSAGE, SYSTEM_PROMPT, answer
from app.store import JsonStore


def make_store(tmp_path):
    store = JsonStore(str(tmp_path / "store.json"))
    store.setup(reset=True)
    content = "Le salarie a droit a des conges payes chaque annee"
    store.add(
        rows=[{"article": "231", "content": content}],
        vectors=[embed_query(content)],
    )
    return store


def test_in_scope_question_returns_sources_and_does_not_abstain(tmp_path):
    store = make_store(tmp_path)
    # Le fake embedder ne comprend que les mots exacts (pas le sens), donc on
    # reprend volontairement du vocabulaire proche du chunk stocke. Avec le vrai
    # modele OpenAI, une reformulation plus libre fonctionnerait aussi.
    result = answer("Est-ce que le salarie a droit a des conges payes ?", store=store)

    assert result["abstained"] is False
    assert len(result["sources"]) > 0
    assert result["sources"][0]["article"] == "231"


def test_out_of_scope_question_abstains_with_no_sources(tmp_path):
    store = make_store(tmp_path)
    result = answer("Quelle est la recette du tajine aux olives ?", store=store)

    assert result["abstained"] is True
    assert result["sources"] == []
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["tool_calls"] == []


def test_system_prompt_contains_grounding_rules():
    assert "UNIQUEMENT" in SYSTEM_PROMPT
    assert "N'invente JAMAIS" in SYSTEM_PROMPT
    assert ABSTENTION_MESSAGE in SYSTEM_PROMPT


def test_empty_store_always_abstains(tmp_path):
    store = JsonStore(str(tmp_path / "empty.json"))
    store.setup(reset=True)
    result = answer("N'importe quelle question", store=store)
    assert result["abstained"] is True
    assert result["sources"] == []
