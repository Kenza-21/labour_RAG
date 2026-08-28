# Assistant Code du Travail — RAG

Un assistant qui répond à des questions sur le **Code du travail marocain**, uniquement à
partir du texte officiel — jamais de sa mémoire d'entraînement — avec sources citées et un
refus structurel de répondre quand l'information n'est pas trouvée dans le corpus.

Construit de zéro (chunking, embeddings, recherche vectorielle, garde-fou d'abstention,
tool-calling) pour comprendre chaque couche d'un système RAG, pas juste l'assembler.

---

## Ce que ça fait

- Découpe le Code du travail officiel (589 articles) en unités de recherche
- Transforme chaque article en vecteur sémantique (embeddings)
- Pour chaque question : cherche les articles les plus pertinents, **refuse de répondre**
  si rien d'assez proche n'est trouvé (pas d'appel au modèle dans ce cas)
- Génère une réponse **uniquement** à partir des articles retrouvés, avec citation obligatoire
- Peut appeler un outil de calcul déterministe (ex: jours de congés) plutôt que de calculer
  "de tête"
- Renvoie toujours les sources utilisées, pour que la réponse soit vérifiable

---

## Architecture

```
INGESTION (une fois)
  PDF officiel --chunking--> articles --embeddings--> vecteurs --> store (JSON ou Postgres)

QUERY (a chaque question)
  question --embeddings--> vecteur --> recherche des k plus proches
    --> distance trop grande ? --> ABSTENTION (le modele n'est jamais appele)
    --> sinon --> contexte + question --> modele --> reponse + sources
```

Détails complets (décisions, bugs trouvés, résultats d'éval) dans [`BUILD_LOG.md`](BUILD_LOG.md).

---

## Stack

| Composant | Options disponibles | Utilisé dans ce projet |
|---|---|---|
| API | FastAPI | ✅ |
| Embeddings | `fake` (test) · HuggingFace (local, gratuit) · OpenAI | HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Génération | `fake` (test) · Ollama (local, gratuit) · Claude (Anthropic) · OpenAI | Ollama (`phi3`) |
| Stockage vecteurs | JSON (fichier) · pgvector (Postgres) | JSON |

100% gratuit et local par défaut — aucune clé API requise pour faire tourner le projet.

---

## Installation

```bash
git clone https://github.com/TON-USERNAME/labour-rag.git
cd labour-rag

python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/Mac
```

Pour la génération locale, installe [Ollama](https://ollama.com) et récupère un modèle :
```bash
ollama pull phi3
```

---

## Utilisation

### 1. Ingérer un document

```bash
EMBEDDING_PROVIDER=huggingface python -m app.ingest --source data/code_travail.pdf --reset
```

### 2. Lancer le serveur

```bash
EMBEDDING_PROVIDER=huggingface CHAT_PROVIDER=ollama python -m uvicorn app.main:app --port 8000
```

### 3. Interroger l'assistant

- **Chat web** : <http://127.0.0.1:8000/> (question → réponse citée + panneau des sources avec scores de similarité et seuil d'abstention)
- API interactive : <http://127.0.0.1:8000/docs>

Ou en ligne de commande :
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Combien de jours de conge annuel un salarie a-t-il ?"}'
```

Réponse :
```json
{
  "answer": "Selon l'Article 231, le salarie a droit a un conge annuel paye...",
  "sources": [{"article": "231", "excerpt": "...", "distance": 0.28}],
  "abstained": false,
  "tool_calls": []
}
```

---

## Configuration

Toutes les variables d'environnement (aucune obligatoire, valeurs par défaut sûres) :

| Variable | Valeurs | Défaut |
|---|---|---|
| `EMBEDDING_PROVIDER` | `fake` \| `huggingface` \| `openai` | `fake` |
| `CHAT_PROVIDER` | `fake` \| `ollama` \| `anthropic` \| `openai` | `fake` |
| `STORE_BACKEND` | `json` \| `pgvector` | `json` |
| `TOP_K` | nombre de chunks retrouvés | `5` |
| `MAX_DISTANCE` | seuil d'abstention (recalibrer si tu changes de modèle d'embedding) | `0.5` |
| `ANTHROPIC_API_KEY` | clé Anthropic (console.anthropic.com) | — |
| `OPENAI_API_KEY` | clé OpenAI | — |
| `DATABASE_URL` | connexion Postgres (si `STORE_BACKEND=pgvector`) | — |

---

## Tests

```bash
./.venv/Scripts/python -m pytest -v
```

55 tests, 100% en mode `fake`/`json` — aucune clé API, aucun serveur externe, aucun coût.

Évaluation de la qualité de recherche (hit-rate@k, MRR) :
```bash
EMBEDDING_PROVIDER=huggingface python -m eval.run_eval
```

---

## Structure du projet

```
app/
├── config.py       # reglages, une seule source de verite
├── chunking.py     # texte -> chunks (decoupage par article)
├── embeddings.py   # texte -> vecteurs (fake/huggingface/openai)
├── store.py        # stockage + recherche vectorielle (JSON/pgvector)
├── tools.py        # fonctions deterministes appelables par le modele
├── llm.py          # boucle de tool-calling (fake/ollama/anthropic/openai)
├── rag.py          # orchestration: recherche -> garde-fou -> generation
├── schemas.py       # contrats API (Pydantic)
├── main.py          # serveur FastAPI
└── ingest.py         # CLI d'ingestion
eval/
├── questions.json    # jeu de questions/reponses connues
└── run_eval.py         # hit-rate@k, MRR, taux d'abstention
tests/                   # un fichier de test par module ci-dessus
data/
└── code_travail.pdf      # texte officiel (Bulletin Officiel n5210)
```

---

## Limites connues

- Le modèle local (`phi3`) peut mal recopier un chiffre pourtant présent dans le contexte
  fourni — voir `BUILD_LOG.md` pour un exemple réel documenté
- `phi3` ne supporte pas le tool-calling (dégradation automatique, calcul de congés
  indisponible avec ce modèle)
- Backend `pgvector` écrit mais jamais testé en conditions réelles (pas de Postgres/Docker
  disponible lors du développement)

Détails complets, décisions, et bugs réels trouvés pendant le développement : [`BUILD_LOG.md`](BUILD_LOG.md).

---

## Source du corpus

Le Code du travail marocain (Loi n°65-99), texte publié au Bulletin Officiel n°5210 du
6 mai 2004, republié par l'[OIT/ILO](https://webapps.ilo.org/static/english/inwork/cb-policy-guide/moroccolabourcode2004.pdf).
