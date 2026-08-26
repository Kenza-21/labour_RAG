"""Deterministic functions the model can call. Plain Python, fully tested, no LLM involved."""


def calculer_conges_annuels(mois_travailles: float, annees_anciennete: float = 0) -> dict:
    if mois_travailles < 6:
        return {"eligible": False, "jours_ouvrables": 0.0, "basis": "Article 231"}
    base = mois_travailles * 1.5                      # 1.5 jours / mois (Art. 231)
    bonus = (annees_anciennete // 5) * 1.5             # bonus par tranche de 5 ans (Art. 232)
    total = min(base + bonus, 30.0)                    # plafond legal (Art. 232)
    return {"eligible": True, "jours_ouvrables": total, "basis": "Articles 231 et 232"}


TOOL_SPECS = [{
    "type": "function",
    "function": {
        "name": "calculer_conges_annuels",
        "description": "Calcule le nombre de jours de conge annuel paye.",
        "parameters": {
            "type": "object",
            "properties": {
                "mois_travailles": {"type": "number"},
                "annees_anciennete": {"type": "number"},
            },
            "required": ["mois_travailles"],
        },
    },
}]

CLAUDE_TOOL_SPECS = [{
    "name": "calculer_conges_annuels",
    "description": "Calcule le nombre de jours de conge annuel paye.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mois_travailles": {"type": "number"},
            "annees_anciennete": {"type": "number"},
        },
        "required": ["mois_travailles"],
    },
}]

_REGISTRY = {
    "calculer_conges_annuels": calculer_conges_annuels,
}


def run_tool(name: str, args: dict) -> dict:
    if name not in _REGISTRY:
        return {"error": f"unknown tool: {name}"}
    return _REGISTRY[name](**args)
