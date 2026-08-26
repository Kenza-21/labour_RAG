from app.config import settings
from app.embeddings import embed_query, embed_texts


def test_same_input_gives_same_output_deterministic():
    v1 = embed_query("Combien de jours de conge annuel ?")
    v2 = embed_query("Combien de jours de conge annuel ?")
    assert v1 == v2


def test_vector_length_matches_embedding_dim():
    v = embed_query("test")
    assert len(v) == settings.embedding_dim


def test_order_preserved_across_a_batch():
    texts = ["premier texte", "deuxieme texte", "troisieme texte"]
    vectors = embed_texts(texts)
    # chaque vecteur individuel doit correspondre a son texte, peu importe l'ordre du batch
    assert vectors[0] == embed_query(texts[0])
    assert vectors[1] == embed_query(texts[1])
    assert vectors[2] == embed_query(texts[2])


def test_empty_list_returns_empty_list():
    assert embed_texts([]) == []


def test_similar_texts_are_closer_than_unrelated_texts():
    def cosine_sim(a, b):
        return sum(x * y for x, y in zip(a, b))

    v_conges_1 = embed_query("Le salarie a droit a des conges payes")
    v_conges_2 = embed_query("Le salarie beneficie de conges payes")
    v_unrelated = embed_query("La recette du tajine aux olives")

    sim_related = cosine_sim(v_conges_1, v_conges_2)
    sim_unrelated = cosine_sim(v_conges_1, v_unrelated)
    assert sim_related > sim_unrelated
