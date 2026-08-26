from app import config


def test_defaults_are_sane():
    s = config.Settings.from_env()
    assert s.embedding_provider == "fake"
    assert s.chat_provider == "fake"
    assert s.store_backend == "json"
    assert s.embedding_dim == 1536
    assert s.top_k == 5
    assert 0 < s.max_distance < 2


def test_env_override_changes_value(monkeypatch):
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("CHAT_PROVIDER", "anthropic")
    s = config.Settings.from_env()
    assert s.top_k == 9
    assert s.chat_provider == "anthropic"


def test_huggingface_provider_switches_default_model_and_dim(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    s = config.Settings.from_env()
    assert s.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert s.embedding_dim == 384


def test_anthropic_provider_switches_default_chat_model(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "anthropic")
    s = config.Settings.from_env()
    assert s.chat_model == "claude-opus-5"


def test_explicit_env_overrides_the_computed_default(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("EMBEDDING_DIM", "768")  # un autre modele HF, par ex.
    s = config.Settings.from_env()
    assert s.embedding_dim == 768


def test_module_level_settings_is_frozen():
    try:
        config.settings.top_k = 999
        assert False, "expected an error mutating a frozen dataclass"
    except Exception:
        pass
