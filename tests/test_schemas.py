import pytest
from pydantic import ValidationError

from app.schemas import AskRequest, AskResponse, Source


def test_ask_request_accepts_valid_question():
    req = AskRequest(question="Combien de jours de conge ?")
    assert req.top_k is None


def test_ask_request_rejects_too_short_question():
    with pytest.raises(ValidationError):
        AskRequest(question="ok")


def test_ask_request_rejects_missing_question():
    with pytest.raises(ValidationError):
        AskRequest()


def test_ask_request_rejects_top_k_out_of_range():
    with pytest.raises(ValidationError):
        AskRequest(question="Une vraie question ?", top_k=50)


def test_ask_response_shape():
    resp = AskResponse(
        answer="Vous avez droit a 18 jours (Article 231).",
        sources=[Source(article="231", excerpt="Tout salarie...", distance=0.12)],
        abstained=False,
    )
    assert resp.tool_calls == []
