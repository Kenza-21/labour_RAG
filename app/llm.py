"""Chat completion + tool-calling loop. Fake (offline), OpenAI, or Anthropic (Claude)."""

import json

from app.config import settings
from app.tools import CLAUDE_TOOL_SPECS, TOOL_SPECS, run_tool

MAX_TOOL_ROUNDS = 3
CLAUDE_MAX_TOKENS = 1024  # reponses courtes attendues (quelques phrases): pas besoin de plus


def complete(messages: list[dict], use_tools: bool = False) -> tuple[str, list[dict]]:
    if settings.chat_provider == "fake":
        return _complete_fake(messages)
    if settings.chat_provider == "anthropic":
        return _complete_claude(messages, use_tools)
    if settings.chat_provider == "ollama":
        return _complete_ollama(messages, use_tools)
    return _complete_openai(messages, use_tools)


def _complete_fake(messages: list[dict]) -> tuple[str, list[dict]]:
    # Aucun appel reseau: renvoie un texte fixe pour que rag.py soit testable hors ligne.
    return "Reponse generee (mode fake, aucun appel API).", []


def _create_completion(client, messages: list[dict], use_tools: bool):
    """Envoie la requete; si le modele ne supporte pas le tool-calling, redemande sans outils."""
    import openai as openai_sdk

    try:
        return client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            tools=TOOL_SPECS if use_tools else None,
            tool_choice="auto" if use_tools else None,
        ), use_tools
    except openai_sdk.BadRequestError as e:
        if use_tools and "does not support tools" in str(e):
            # modele local sans function-calling (ex: phi3 sous Ollama): on degrade proprement
            return client.chat.completions.create(model=settings.chat_model, messages=messages), False
        raise


def _run_openai_compatible_loop(client, messages: list[dict], use_tools: bool) -> tuple[str, list[dict]]:
    """Boucle partagee par OpenAI et Ollama: les deux exposent la meme API (le SDK openai)."""
    tool_trace: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response, use_tools = _create_completion(client, messages, use_tools)
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content, tool_trace

        messages.append({"role": "assistant", "tool_calls": message.tool_calls})
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = run_tool(call.function.name, args)
            tool_trace.append({"name": call.function.name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    # budget d'outils epuise: on redemande une derniere fois, sans outils, pour forcer une reponse finale
    response = client.chat.completions.create(model=settings.chat_model, messages=messages)
    return response.choices[0].message.content, tool_trace


def _complete_openai(messages: list[dict], use_tools: bool) -> tuple[str, list[dict]]:
    from openai import OpenAI  # imported lazily: not installed in the fake-only path

    client = OpenAI(api_key=settings.openai_api_key)
    return _run_openai_compatible_loop(client, messages, use_tools)


def _complete_ollama(messages: list[dict], use_tools: bool) -> tuple[str, list[dict]]:
    from openai import OpenAI  # Ollama expose une API compatible OpenAI, meme SDK

    client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")  # cle factice, non verifiee
    return _run_openai_compatible_loop(client, messages, use_tools)


def _split_system_message(messages: list[dict]) -> tuple[str, list[dict]]:
    """Claude prend le system prompt a part (parametre `system`), pas dans `messages`."""
    system_text = ""
    rest = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            rest.append(m)
    return system_text, rest


def _complete_claude(messages: list[dict], use_tools: bool) -> tuple[str, list[dict]]:
    import anthropic  # imported lazily: not installed unless you use this provider

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool_trace: list[dict] = []
    system_text, claude_messages = _split_system_message(messages)

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=settings.chat_model,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system_text,
            messages=claude_messages,
            tools=CLAUDE_TOOL_SPECS if use_tools else [],
        )

        if response.stop_reason != "tool_use":
            text = next((b.text for b in response.content if b.type == "text"), "")
            return text, tool_trace

        claude_messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_trace.append({"name": block.name, "args": block.input, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        claude_messages.append({"role": "user", "content": tool_results})

    # budget d'outils epuise: on redemande une derniere fois, sans outils, pour forcer une reponse finale
    response = client.messages.create(
        model=settings.chat_model,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=system_text,
        messages=claude_messages,
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text, tool_trace
