# Build Log — Assistant Code du Travail marocain (RAG)

Journal de construction du projet, écrit du point de vue de l'ingénieur IA qui l'a construit.
Couvre : l'architecture, l'ordre de build, les décisions prises et pourquoi, les bugs réels
trouvés en cours de route, et l'état actuel du système.

---

## 1. Vue d'ensemble

**Objectif :** un assistant qui répond à des questions sur le Code du travail marocain,
uniquement à partir du texte officiel (jamais de sa mémoire d'entraînement), avec sources
citées et un refus structurel de répondre quand l'information n'est pas trouvée.

**Stack finale :**

| Composant | Choix | Pourquoi |
|---|---|---|
| API | FastAPI | endpoints `/ask`, `/search`, `/health` |
| Embeddings | HuggingFace local (`sentence-transformers`) | gratuit, hors-ligne, pas de clé API |
| Modèle d'embedding | `paraphrase-multilingual-MiniLM-L12-v2` | corpus en français — un modèle anglais-only séparait mal les synonymes (voir §4) |
| Génération | Ollama local (`phi3`) | gratuit, hors-ligne, pas de clé API |
| Store vecteurs | fichier JSON (`store.py::JsonStore`) | pas de Docker/Postgres disponible sur cette machine |
| Chat cloud (optionnel, prêt mais pas testé) | Claude (Anthropic) via `anthropic` SDK | code écrit et branché, en attente d'une clé API |

**Coût total du projet à ce stade : 0€.** Tout tourne en local (embeddings + génération),
zéro clé API payante utilisée.

---

## 2. Architecture — les deux pipelines

```
INGESTION (une fois)
  PDF officiel (589 articles)
    -> chunking.py   : decoupe par "Article N" (regex)
    -> embeddings.py : chunk -> vecteur 384 dimensions (HuggingFace local)
    -> store.py       : vecteurs stockes dans data/store.json

QUERY (a chaque question)
  question utilisateur
    -> embeddings.py : question -> vecteur (MEME modele que l'ingestion)
    -> store.py       : recherche des k plus proches (distance cosinus)
    -> rag.py          : garde-fou d'abstention (distance > seuil -> refus, LLM jamais appele)
    -> llm.py            : generation de la reponse (Ollama/phi3, ou Claude si configure)
    -> reponse + sources retournees ensemble
```

---

## 3. Ordre de build (10 fichiers, bottom-up)

Chaque fichier a été construit et testé isolément avant d'être branché au suivant — jamais
écrit `main.py` avant que `chunking.py` ait ses propres tests verts.

1. **`config.py`** — un seul objet `Settings` (dataclass immuable), lu une fois au démarrage.
   Toutes les autres couches lisent `settings.xxx`, jamais `os.getenv()` directement.
2. **`chunking.py`** — texte brut → chunks. Nettoyage (mots coupés par un tiret, numéros de
   page), découpage par `"Article N"` (regex), filet de sécurité en fenêtres glissantes si
   aucun article n'est trouvé.
3. **`embeddings.py`** — texte → vecteur. Trois fournisseurs derrière une même fonction
   `embed_texts()` : `fake` (hash de mots, hors-ligne, pour les tests), `huggingface` (modèle
   local), `openai` (API, code prêt mais non utilisé dans ce projet).
4. **`store.py`** — interface abstraite `VectorStore` avec deux implémentations : `JsonStore`
   (fichier, utilisé ici) et `PgVectorStore` (Postgres+pgvector, code écrit mais jamais testé
   faute de Docker sur cette machine).
5. **`tools.py`** — une fonction Python déterministe (`calculer_conges_annuels`) que le modèle
   peut appeler au lieu de calculer lui-même. Deux formats de spec : `TOOL_SPECS` (OpenAI/Ollama)
   et `CLAUDE_TOOL_SPECS` (Anthropic — schéma différent : `input_schema` au lieu de `parameters`).
6. **`llm.py`** — boucle de tool-calling. Quatre fournisseurs : `fake`, `openai`, `ollama`
   (réutilise le SDK `openai` pointé sur `localhost:11434/v1` — Ollama est compatible OpenAI),
   `anthropic` (SDK et format de tools différents, boucle manuelle séparée).
7. **`rag.py`** — assemble tout : `retrieve()` → garde-fou d'abstention → `build_context()` →
   `complete()` → réponse avec sources toujours attachées.
8. **`schemas.py`** — contrats Pydantic (`AskRequest`, `AskResponse`, etc.), validation
   automatique des requêtes HTTP.
9. **`main.py`** — FastAPI, endpoints `/ask`, `/search`, `/health`, connexion au store une
   seule fois au démarrage (`lifespan`).
10. **`ingest.py`** — CLI qui exécute le pipeline 1 (`python -m app.ingest --source ... --reset`).

Plus `eval/run_eval.py` — mesure la qualité de la recherche (hit-rate@k, MRR), séparément
de la génération.

---

## 4. Décisions clés et pourquoi

### 4.1 Séparer `embedding_provider` et `chat_provider`
Le guide de départ suppose un seul `PROVIDER` pour tout. Cassé dès qu'on a voulu Claude (pas
d'API d'embeddings chez Anthropic) + HuggingFace (pas de génération de texte de qualité en
local sans gros modèle). `config.py` a deux réglages indépendants, chacun avec ses propres
valeurs par défaut calculées dynamiquement (modèle, dimension du vecteur, etc.).

### 4.2 Le modèle d'embedding par défaut a changé une fois testé en vrai
Premier choix (`all-MiniLM-L6-v2`, anglais) : séparation synonymes vs hors-sujet faible
(0.49 vs 0.55 — écart quasi nul). Testé `paraphrase-multilingual-MiniLM-L12-v2` à la place :
écart net (0.61 vs 0.99). Le corpus est en français — un modèle anglais-only était le mauvais
choix par défaut. **Leçon : toujours mesurer avant de figer un choix de modèle, ne jamais
supposer qu'un modèle "standard" convient à une langue non-anglaise sans vérifier.**

### 4.3 `EMBEDDING_DIM` doit suivre le modèle, pas rester une constante
Bug classique du guide (§5.2/§7) : mélanger deux modèles d'embedding donne des résultats
silencieusement faux (pas d'erreur). `config.py` calcule `embedding_dim` par défaut à partir
du fournisseur choisi (1536 pour OpenAI, 384 pour HuggingFace/MiniLM), pour rendre ce bug
impossible à l'usage normal plutôt que juste documenté.

### 4.4 `MAX_DISTANCE` recalibré sur le vrai corpus
Valeur d'exemple du guide (0.65) calibrée pour des embeddings OpenAI. Mesuré sur le vrai
corpus (589 articles) avec nos embeddings HuggingFace :
```
in-scope      : 0.241 - 0.329
out-of-scope  : 0.677 - 0.840
```
0.65 était trop près du bord bas des questions hors-sujet (0.677) — marge de sécurité faible.
Seuil recalé à **0.5**, au milieu de l'écart observé. **Leçon : le guide le dit explicitement
et c'est vrai en pratique — ce seuil doit être remesuré à chaque changement de modèle
d'embedding ou de corpus, jamais copié tel quel d'un exemple.**

### 4.5 Ollama = SDK OpenAI pointé ailleurs
Ollama expose une API compatible OpenAI sur `localhost:11434/v1`. Plutôt que d'écrire un
client HTTP séparé, `_complete_ollama` réutilise le SDK `openai` avec juste un `base_url`
différent et une clé factice. La boucle de tool-calling (`_run_openai_compatible_loop`) est
donc partagée entre `_complete_openai` et `_complete_ollama` — même code, client différent.

### 4.6 Dégradation propre quand le modèle local ne supporte pas les tools
`phi3` (déjà installé localement) ne supporte pas le function-calling — erreur `400 does not
support tools` à l'appel. `_create_completion()` intercepte cette erreur précise et redemande
sans outils automatiquement, plutôt que de planter. Le calcul de congés (`tools.py`) devient
simplement indisponible avec ce modèle — dégradation silencieuse mais propre, pas un crash.

---

## 5. Bugs et découvertes réelles en cours de route

Ce ne sont pas des bugs de démonstration — ce sont des choses réellement trouvées en testant
contre le vrai document officiel.

### 5.1 Hallucination numérique malgré un contexte correct
Question réelle posée en bout en bout (HuggingFace + Ollama/phi3) :
*"Quelle est la durée légale hebdomadaire du travail ?"*

Le texte réellement retrouvé (Article 184, distance 0.266) dit **"44 heures par semaine"**.
Le modèle a répondu **"40 heures par semaine"** — un chiffre différent, alors que le bon
chiffre était dans son contexte. Aucune ligne de code n'a causé cette erreur ; c'est une
limite du petit modèle local (phi3, 3.8B paramètres) à recopier fidèlement un nombre précis.

**Pourquoi ce n'est pas un défaut de conception :** c'est justement pour détecter ce genre
d'erreur que `sources` est toujours retourné avec la réponse — sans ça, l'erreur serait
invisible. Un modèle plus capable (Claude, GPT-4o) réduirait ce risque sans l'éliminer à 100%.

### 5.2 Un exemple du guide de départ était faux pour le vrai texte de loi
`tools.py` (suivant l'exemple du guide) supposait que l'Article 78 traite du paiement du
salaire. Vérifié contre le vrai PDF officiel : l'Article 78 concerne en réalité des **amendes
pour non-respect du préavis** — rien à voir. Le vrai article sur l'intervalle de paiement du
salaire est l'**Article 363**. Trouvé en cherchant le mot-clé "intervalle" dans le corpus réel
plutôt qu'en faisant confiance à l'exemple du guide. **Leçon : même un exemple dans une
documentation de référence doit être vérifié contre la source primaire avant d'être utilisé
en production, surtout en droit.**

### 5.3 Instabilité de classement sur des distances très resserrées
Pour une question donnée, deux exécutions successives de la même recherche (même question,
même store, même modèle) ont donné des top-5 différents quand les distances candidates
étaient très proches les unes des autres (écart de 0.05 entre le 1er et le 5e résultat).
Cause probable : légères variations en virgule flottante liées au threading CPU dans
`sentence-transformers`. **Leçon : ne jamais tirer une conclusion de qualité de recherche
sur une seule exécution quand les distances sont serrées — c'est aussi pourquoi
`eval/run_eval.py` existe, pour mesurer sur un ensemble de questions plutôt qu'à l'œil sur un
cas isolé.**

---

## 6. Résultats de l'évaluation (`eval/run_eval.py`) sur le vrai corpus

```
Chunks indexes                  : 589
Questions evaluees               : 6
Hit-rate@5                       : 100.0%
MRR                              : 0.567
Taux d'abstention (hors-sujet)   : 100.0% (3 questions)
```

MRR à 0.567 (et non 1.0) est honnête : plusieurs bonnes réponses sont trouvées en 2e position
plutôt qu'en 1ère (voir §5.3), et une question (Article 363) n'est retrouvée qu'en 5e position
sur un des runs. Le jeu de questions (`eval/questions.json`) inclut volontairement ce cas
difficile plutôt que de ne garder que des questions faciles — un score gonflé artificiellement
n'aurait aucune valeur diagnostique.

---

## 7. Couverture de tests

**55 tests, tous verts**, répartis par fichier :

| Fichier testé | Nb tests | Ce qui est couvert |
|---|---|---|
| `config.py` | 6 | defaults, override par env var, calcul des defaults conditionnels (HF/Anthropic), immutabilité |
| `chunking.py` | 6 | nettoyage, découpage par article, filet de sécurité, cas limites |
| `embeddings.py` | 5 | déterminisme, dimension, ordre préservé, similarité sémantique |
| `store.py` | 7 | distance cosinus (cas connus), insert/search JSON, tri, persistance |
| `tools.py` | 7 | tous les cas limites du calcul de congés, dispatch par nom |
| `llm.py` | 2 | mode fake, indépendance du flag `use_tools` |
| `rag.py` | 4 | garde-fou d'abstention (in-scope vs out-of-scope), zéro source si abstention |
| `schemas.py` | 5 | validation Pydantic, rejets 422 |
| `main.py` | 6 | tous les endpoints via `TestClient`, bout-en-bout HTTP |
| `ingest.py` | 3 | CLI complet, respect du batch-size, reset |
| `eval/run_eval.py` | 4 | hit-rate, MRR (y compris rang 2 = MRR 0.5), abstention |

Tous les tests tournent en mode `fake`/`json` — zéro appel réseau, zéro clé API, zéro
dépendance à Docker. C'est ce qui rend la CI possible sans secrets (§6 du guide de départ).

---

## 8. Comment lancer le projet

```bash
# Environnement
cd labour-rag
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt

# Tests (rapides, gratuits, hors-ligne)
./.venv/Scripts/python -m pytest -v

# Ingestion reelle (deja faite, fichier data/store.json present)
EMBEDDING_PROVIDER=huggingface python -m app.ingest --source data/code_travail.pdf --reset

# Evaluation de la recherche
EMBEDDING_PROVIDER=huggingface python -m eval.run_eval

# Lancer le serveur (recherche + generation reelles, 100% gratuit/local)
EMBEDDING_PROVIDER=huggingface CHAT_PROVIDER=ollama python -m uvicorn app.main:app --port 8000

# Exemple de requete
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Combien de jours de conge annuel un salarie a-t-il ?"}'
```

**Variables d'environnement clés :**

| Variable | Valeurs possibles | Defaut |
|---|---|---|
| `EMBEDDING_PROVIDER` | `fake` \| `huggingface` \| `openai` | `fake` |
| `CHAT_PROVIDER` | `fake` \| `ollama` \| `anthropic` \| `openai` | `fake` |
| `STORE_BACKEND` | `json` \| `pgvector` | `json` |
| `ANTHROPIC_API_KEY` | ta cle (console.anthropic.com) | vide |
| `MAX_DISTANCE` | seuil d'abstention | `0.5` |

---

## 9. Ce qui reste (pas encore fait)

- **Tester Claude (Anthropic) en vrai** — code écrit et branché (`llm.py::_complete_claude`),
  jamais exécuté faute de clé API. Comparaison intéressante à faire avec le bug §5.1 (phi3
  invente 40 au lieu de 44) une fois la clé disponible.
- **Backend `pgvector`/Postgres réel** — code écrit (`store.py::PgVectorStore`), jamais testé
  faute de Docker installé sur cette machine.
- **Docker Compose** — pas commencé.
- **CI** (GitHub Actions ou équivalent) — pas commencé, mais déjà rendu possible par le fait
  que toute la suite de tests tourne en mode `fake`/`json`, zéro secret nécessaire.
- **`.env` support** — `config.py` lit seulement les variables d'environnement système
  actuellement, pas de fichier `.env` chargé automatiquement.

---

## 10. Ce qu'un recruteur devrait retenir de ce projet

- **Architecture en couches testées indépendamment** : chaque fichier a ses propres tests
  avant d'être branché au suivant — pas un script monolithique.
- **Garde-fou d'abstention structurel**, pas juste une instruction de prompt : la génération
  n'a physiquement pas lieu si rien d'assez pertinent n'est trouvé.
- **Choix de stack justifiés par des mesures, pas par défaut** : le modèle d'embedding a
  changé après un test chiffré (§4.2), le seuil d'abstention a été recalibré sur le vrai
  corpus (§4.4), un exemple du guide de départ a été corrigé après vérification contre la
  source primaire (§5.2).
- **Conscience explicite des limites** : la hallucination numérique (§5.1) et l'instabilité
  de classement (§5.3) sont documentées, pas cachées — savoir *pourquoi* un système RAG peut
  encore se tromper malgré tous les garde-fous est aussi important que de les construire.
- **Zéro coût, zéro secret pour développer et tester** : toute la chaîne (embeddings +
  génération) tourne en local, ce qui rend le projet reproductible par n'importe qui sans
  budget ni compte cloud.
