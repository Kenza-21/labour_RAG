from app.ingest import ingest
from app.store import JsonStore


def test_ingest_txt_file_stores_all_chunks(tmp_path, monkeypatch):
    source = tmp_path / "code.txt"
    source.write_text(
        "Article 1\nPremiere regle.\nArticle 2\nDeuxieme regle.\nArticle 3\nTroisieme regle.",
        encoding="utf-8",
    )
    store_path = str(tmp_path / "store.json")
    monkeypatch.setenv("JSON_STORE_PATH", store_path)

    total = ingest(str(source), reset=True)

    assert total == 3
    store = JsonStore(store_path)
    store.setup(reset=False)
    assert store.count() == 3


def test_ingest_respects_batch_size(tmp_path, monkeypatch):
    articles = "".join(f"Article {i}\nContenu {i}.\n" for i in range(1, 8))
    source = tmp_path / "code.txt"
    source.write_text(articles, encoding="utf-8")
    store_path = str(tmp_path / "store.json")
    monkeypatch.setenv("JSON_STORE_PATH", store_path)

    total = ingest(str(source), reset=True, batch_size=3)

    assert total == 7


def test_ingest_reset_wipes_previous_data(tmp_path, monkeypatch):
    store_path = str(tmp_path / "store.json")
    monkeypatch.setenv("JSON_STORE_PATH", store_path)

    source1 = tmp_path / "first.txt"
    source1.write_text("Article 1\nPremiere version.", encoding="utf-8")
    ingest(str(source1), reset=True)

    source2 = tmp_path / "second.txt"
    source2.write_text("Article 9\nAutre document.", encoding="utf-8")
    total = ingest(str(source2), reset=True)

    assert total == 1  # l'ancien contenu a bien ete efface, pas accumule
