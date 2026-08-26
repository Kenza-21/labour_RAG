import math

from app.store import JsonStore, cosine_distance


def test_cosine_distance_orthogonal_vectors():
    # deux vecteurs perpendiculaires: aucune relation -> distance = 1.0
    assert math.isclose(cosine_distance([1, 0], [0, 1]), 1.0, abs_tol=1e-9)


def test_cosine_distance_opposite_vectors():
    # vecteurs opposes -> distance maximale = 2.0
    assert math.isclose(cosine_distance([1, 0], [-1, 0]), 2.0, abs_tol=1e-9)


def test_cosine_distance_identical_vectors():
    # meme vecteur -> distance = 0.0
    assert math.isclose(cosine_distance([3, 4], [3, 4]), 0.0, abs_tol=1e-9)


def test_json_store_insert_and_search_roundtrip(tmp_path):
    path = str(tmp_path / "store.json")
    store = JsonStore(path)
    store.setup(reset=True)

    store.add(
        rows=[{"article": "1", "content": "premier article"}],
        vectors=[[1.0, 0.0, 0.0]],
    )

    hits = store.search([1.0, 0.0, 0.0], k=5)
    assert len(hits) == 1
    assert hits[0].article == "1"
    assert math.isclose(hits[0].distance, 0.0, abs_tol=1e-9)


def test_json_store_results_sorted_ascending_by_distance(tmp_path):
    path = str(tmp_path / "store.json")
    store = JsonStore(path)
    store.setup(reset=True)

    store.add(
        rows=[
            {"article": "far", "content": "loin"},
            {"article": "close", "content": "proche"},
            {"article": "mid", "content": "moyen"},
        ],
        vectors=[
            [-1.0, 0.0],   # oppose a la query -> distance 2.0
            [1.0, 0.0],    # identique a la query -> distance 0.0
            [0.0, 1.0],    # orthogonal a la query -> distance 1.0
        ],
    )

    hits = store.search([1.0, 0.0], k=3)
    assert [h.article for h in hits] == ["close", "mid", "far"]


def test_json_store_count():
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    try:
        store = JsonStore(path)
        store.setup(reset=True)
        assert store.count() == 0
        store.add(rows=[{"article": "1", "content": "x"}], vectors=[[1.0, 0.0]])
        assert store.count() == 1
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_json_store_persists_across_instances(tmp_path):
    path = str(tmp_path / "store.json")

    store1 = JsonStore(path)
    store1.setup(reset=True)
    store1.add(rows=[{"article": "1", "content": "x"}], vectors=[[1.0, 0.0]])

    store2 = JsonStore(path)
    store2.setup(reset=False)
    assert store2.count() == 1
