import pandas as pd
import pytest
from tests.fixtures_data import demo

from signalforge.optimizer import Constraints, alternative, greedy_baseline, optimise
from signalforge.prioritization import FRAMEWORKS, Weights, compare_frameworks, score, weight_sensitivity


@pytest.fixture(scope="module")
def ranked():
    return demo()[3]


@pytest.mark.parametrize("fw", list(FRAMEWORKS))
def test_every_framework_scores_every_row(ranked, fw):
    s = score(ranked, fw)
    assert len(s) == len(ranked) and s.notna().all()


def test_framework_comparison_has_rank_spread(ranked):
    c = compare_frameworks(ranked)
    assert set(FRAMEWORKS) <= set(c.columns) and (c.rank_spread >= 0).all()


def test_weight_sensitivity_identifies_decision_sensitive_weights(ranked):
    ws = weight_sensitivity(ranked, Weights())
    assert len(ws) == 8 and ws.decision_sensitive.dtype == bool


def test_zero_effort_weight_ignores_effort(ranked):
    w = Weights(effort=0.0)
    a = score(ranked.assign(effort_weeks=ranked.effort_weeks * 10), "Custom weighted", w)
    b = score(ranked, "Custom weighted", w)
    assert (a.round(9) == b.round(9)).all()


def test_capacity_is_never_exceeded(ranked):
    for cap in (0, 3, 10, 25, 1000):
        r = optimise(ranked, Constraints(capacity_weeks=cap))
        assert r.effort_used <= cap + 1e-9


def test_optimiser_at_least_matches_greedy(ranked):
    for cap in (5, 10, 15, 20, 30):
        c = Constraints(capacity_weeks=cap)
        assert optimise(ranked, c).total_value >= greedy_baseline(ranked, c) - 1e-6


def test_mandatory_dependency_and_confidence_threshold(ranked):
    ids = ranked.cluster_id.tolist()
    low = ranked.sort_values("priority").cluster_id.iloc[0]
    c = Constraints(capacity_weeks=1000, mandatory=[low], dependencies=[(ids[0], ids[1])], min_confidence=0.0)
    r = optimise(ranked, c)
    assert low in r.selected.cluster_id.values
    if ids[0] in r.selected.cluster_id.values:
        assert ids[1] in r.selected.cluster_id.values
    hi_conf = Constraints(capacity_weeks=1000, min_confidence=0.99)
    assert optimise(ranked, hi_conf).selected.empty


def test_infeasible_mandatory_reported(ranked):
    big = ranked.sort_values("effort_weeks").cluster_id.iloc[-1]
    r = optimise(ranked, Constraints(capacity_weeks=0.1, mandatory=[big]))
    assert r.status == "infeasible" and r.infeasible_reason


def test_alternative_excludes_primary_top_item(ranked):
    c = Constraints(capacity_weeks=20)
    p = optimise(ranked, c)
    alt = alternative(ranked, c, p)
    top = p.selected.sort_values("net_opportunity_value").cluster_id.iloc[-1]
    assert top not in alt.selected.cluster_id.values and alt.total_value <= p.total_value


def test_empty_roadmap():
    assert optimise(pd.DataFrame(), Constraints()).status == "empty"
