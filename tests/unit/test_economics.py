import pytest

from signalforge.economics import (
    Assumptions,
    evaluate_opportunity,
    monte_carlo,
    portfolio_exposure,
    risk_adjusted_priority,
    tornado,
)

OPP = dict(cluster_id=1, acv_sum_unique=1_000_000.0, avg_severity=3.0, effort_weeks=5.0,
           account_acv={"A": 600_000.0, "B": 400_000.0})


def test_formulas_by_hand():
    a = Assumptions(base_exposure_fraction=0.3, recovery_probability=0.4, eng_cost_per_person_week=4000,
                    infra_cost_annual=2000, support_cost_annual=3000)
    e = evaluate_opportunity(OPP, a)
    assert e["potential_revenue_exposure"] == pytest.approx(300_000)
    assert e["expected_recoverable_value"] == pytest.approx(120_000)
    assert e["total_investment"] == pytest.approx(25_000)
    assert e["net_opportunity_value"] == pytest.approx(95_000)
    assert e["roi"] == pytest.approx(3.8)
    assert e["payback_months"] == pytest.approx(25_000 / 10_000)
    assert risk_adjusted_priority(e, 0.5) == pytest.approx(95_000 * 0.5 / 5)


def test_invalid_assumptions_rejected():
    with pytest.raises(ValueError):
        evaluate_opportunity(OPP, Assumptions(recovery_probability=1.2))
    with pytest.raises(ValueError):
        evaluate_opportunity(OPP, Assumptions(eng_cost_per_person_week=-1))


def test_zero_recovery_has_no_payback_and_no_crash():
    e = evaluate_opportunity(OPP, Assumptions(recovery_probability=0.0))
    assert e["payback_months"] is None and e["net_opportunity_value"] < 0


def test_missing_effort_uses_default():
    o = {**OPP, "effort_weeks": None}
    assert evaluate_opportunity(o, Assumptions(default_effort_weeks=6))["effort_weeks"] == 6


def test_portfolio_exposure_does_not_double_count():
    a = Assumptions()
    e = evaluate_opportunity(OPP, a)
    total = portfolio_exposure([OPP, {**OPP, "cluster_id": 2}], [e, e])
    assert total == pytest.approx(e["potential_revenue_exposure"])  # same accounts counted once


def test_monte_carlo_reproducible_and_ordered():
    a = Assumptions()
    m1, m2 = monte_carlo(OPP, a), monte_carlo(OPP, a)
    assert m1 == m2 and m1["p10"] <= m1["p50"] <= m1["p90"] and 0 <= m1["prob_positive"] <= 1


def test_tornado_direction():
    t = tornado(OPP, Assumptions()).set_index("driver")
    assert t.loc["Recovery probability", "delta_high"] > 0 > t.loc["Recovery probability", "delta_low"]
    assert t.loc["Eng cost / person-week", "delta_high"] < 0
