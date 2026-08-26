from fastapi.testclient import TestClient

from app.embeddings import embed_query
from app.main import app
from app.store import JsonStore


def make_client(tmp_path, monkeypatch):
    store_path = str(tmp_path / "store.json")
    monkeypatch.setenv("JSON_STORE_PATH", store_path)

    seed = JsonStore(store_path)
    seed.setup(reset=True)
    content = "Le salarie a droit a des conges payes chaque annee"
    seed.add(rows=[{"article": "231", "content": content}], vectors=[embed_query(content)])

    return TestClient(app)  # trigger lifespan via `with`


def test_health_reports_chunk_count(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "chunks_indexed": 1}


def test_ask_in_scope_returns_sources(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.post("/ask", json={"question": "Est-ce que le salarie a droit a des conges payes ?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["abstained"] is False
        assert len(body["sources"]) > 0


def test_ask_out_of_scope_abstains(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.post("/ask", json={"question": "Quelle est la recette du tajine aux olives ?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["abstained"] is True
        assert body["sources"] == []


def test_ask_rejects_too_short_question(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.post("/ask", json={"question": "ok"})
        assert resp.status_code == 422


def test_ask_rejects_missing_question(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.post("/ask", json={})
        assert resp.status_code == 422


def test_search_returns_hits(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.post("/search", json={"question": "conges payes"})
        assert resp.status_code == 200
        assert len(resp.json()["hits"]) > 0
