from app.tools import calculer_conges_annuels, run_tool


def test_ineligible_under_six_months():
    result = calculer_conges_annuels(mois_travailles=3)
    assert result["eligible"] is False
    assert result["jours_ouvrables"] == 0.0


def test_eligible_basic_case_no_seniority():
    result = calculer_conges_annuels(mois_travailles=12, annees_anciennete=0)
    assert result["eligible"] is True
    assert result["jours_ouvrables"] == 18.0  # 12 * 1.5
    assert result["basis"] == "Articles 231 et 232"


def test_seniority_bonus_applied_per_five_years():
    result = calculer_conges_annuels(mois_travailles=12, annees_anciennete=10)
    # base = 18, bonus = (10 // 5) * 1.5 = 3.0 -> total 21
    assert result["jours_ouvrables"] == 21.0


def test_capped_at_thirty_days():
    result = calculer_conges_annuels(mois_travailles=24, annees_anciennete=30)
    assert result["jours_ouvrables"] == 30.0


def test_exactly_six_months_is_eligible():
    result = calculer_conges_annuels(mois_travailles=6)
    assert result["eligible"] is True


def test_run_tool_dispatches_by_name():
    result = run_tool("calculer_conges_annuels", {"mois_travailles": 12})
    assert result["eligible"] is True


def test_run_tool_unknown_name_returns_error():
    result = run_tool("fonction_inexistante", {})
    assert "error" in result
