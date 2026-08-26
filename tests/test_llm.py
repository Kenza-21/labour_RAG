from app.llm import complete


def test_fake_provider_returns_text_and_no_tool_calls():
    messages = [{"role": "user", "content": "Combien de jours de conge ?"}]
    text, tool_trace = complete(messages, use_tools=True)
    assert isinstance(text, str)
    assert len(text) > 0
    assert tool_trace == []


def test_fake_provider_ignores_use_tools_flag():
    messages = [{"role": "user", "content": "test"}]
    text_with_tools, trace1 = complete(messages, use_tools=True)
    text_without_tools, trace2 = complete(messages, use_tools=False)
    assert trace1 == trace2 == []
