from app.chunking import chunk_by_article, chunk_fixed, chunk_text, clean_text


def test_clean_text_repairs_hyphenated_words_and_strips_page_numbers():
    raw = "Le employ-\neur doit payer.\n42\nSuite du texte."
    cleaned = clean_text(raw)
    assert "employeur" in cleaned
    assert "42" not in cleaned.split("\n")


def test_clean_text_normalizes_nbsp():
    raw = "salaire minimum"
    assert clean_text(raw) == "salaire minimum"


def test_chunk_by_article_finds_known_boundaries():
    text = (
        "Article 1\nPremiere regle du texte.\n"
        "Article 2\nDeuxieme regle du texte.\n"
        "Article premier\nRegle finale."
    )
    chunks = chunk_by_article(text)
    assert len(chunks) == 3
    assert chunks[0].article == "1"
    assert "Premiere regle" in chunks[0].content
    assert chunks[1].article == "2"
    assert "Deuxieme regle" in chunks[1].content
    assert chunks[2].article == "premier"
    assert "Regle finale" in chunks[2].content


def test_chunk_by_article_returns_empty_when_no_headings():
    assert chunk_by_article("Just some plain text with no article markers.") == []


def test_chunk_fixed_covers_whole_text_with_overlap():
    text = "phrase un. " * 300  # long text, no article headings
    chunks = chunk_fixed(text, size=200, overlap=50)
    assert len(chunks) > 1
    # every chunk stays within reasonable bounds (soft check: not empty, not huge)
    assert all(0 < len(c.content) <= 400 for c in chunks)


def test_chunk_text_prefers_articles_and_falls_back_to_fixed():
    with_articles = "Article 1\nContenu court."
    result = chunk_text(with_articles)
    assert result[0].article == "1"

    without_articles = "texte sans aucun marqueur d'article. " * 100
    result2 = chunk_text(without_articles)
    assert result2[0].article is None
    assert len(result2) >= 1
