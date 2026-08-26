"""Single source of truth for every setting the app reads from the environment."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    embedding_provider: str
    chat_provider: str
    openai_api_key: str
    anthropic_api_key: str
    ollama_base_url: str
    embedding_model: str
    chat_model: str
    embedding_dim: int
    store_backend: str
    database_url: str
    top_k: int
    max_distance: float

    @classmethod
    def from_env(cls) -> "Settings":
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "fake")
        chat_provider = os.getenv("CHAT_PROVIDER", "fake")

        default_embedding_model = (
            # multilingue: le corpus est en francais, un modele anglais-only
            # separe mal les synonymes francais (teste: gain net de 0.49/0.55 -> 0.61/0.99)
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            if embedding_provider == "huggingface"
            else "text-embedding-3-small"
        )
        default_embedding_dim = "384" if embedding_provider == "huggingface" else "1536"
        default_chat_model = {
            "anthropic": "claude-opus-5",
            "ollama": "phi3",
        }.get(chat_provider, "gpt-4o-mini")

        return cls(
            embedding_provider=embedding_provider,
            chat_provider=chat_provider,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            embedding_model=os.getenv("EMBEDDING_MODEL", default_embedding_model),
            chat_model=os.getenv("CHAT_MODEL", default_chat_model),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", default_embedding_dim)),
            store_backend=os.getenv("STORE_BACKEND", "json"),
            database_url=os.getenv("DATABASE_URL", ""),
            top_k=int(os.getenv("TOP_K", "5")),
            # Recalibre pour nos embeddings HuggingFace (paraphrase-multilingual-MiniLM):
            # in-scope mesure a 0.24-0.33, out-of-scope a 0.68-0.84 sur le vrai corpus.
            # 0.65 (valeur d'exemple du guide, calibree pour OpenAI) etait trop juste ici.
            max_distance=float(os.getenv("MAX_DISTANCE", "0.5")),
        )


settings = Settings.from_env()
