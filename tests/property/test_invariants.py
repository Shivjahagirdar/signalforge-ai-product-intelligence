"""Property-based (Hypothesis) and metamorphic tests of financial and ML invariants."""
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.fixtures_data import demo

from signalforge.audit import Overrides
from signalforge.economics import Assumptions, evaluate_opportunity, resolve_acv, unique_account_acv
from signalforge.intelligence import cluster_feedback, evaluate_clustering
from signalforge.pipeline import build_opportunities, rank_opportunities

OPP = st.fixed_dictionaries({"cluster_id": st.just(1), "acv_sum_unique": st.floats(0, 5e6), "avg_severity": st.floats(1, 5),
                             "effort_weeks": st.floats(0.5, 50)})


@given(acvs=st.lists(st.floats(1, 1e6), min_size=1, max_size=20), dup=st.integers(1, 6))
def test_duplicate_records_never_multiply_unique_account_acv(acvs, dup):
    df = pd.DataFrame({"account_id": [f"a{i}" for i in range(len(acvs))] * dup, "annual_contract_value": acvs * dup,
                       "customer_segment": "SMB"})
    u = unique_account_acv(resolve_acv(df))
    assert np.isclose(u["acv"].sum(), sum(acvs))


@given(o=OPP, extra=st.floats(0.1, 40))
def test_more_engineering_cost_never_increases_roi(o, extra):
    a = Assumptions()
    base = evaluate_opportunity(o, a)
    worse = evaluate_opportunity({**o, "effort_weeks": o["effort_weeks"] + extra}, a)
    assert worse["roi"] <= base["roi"] + 1e-12 and worse["net_opportunity_value"] <= base["net_opportunity_value"] + 1e-9


@given(o=OPP, p=st.floats(0, 1), f=st.floats(0, 1))
def test_probability_bounds_and_outputs_finite(o, p, f):
    e = evaluate_opportunity(o, Assumptions(recovery_probability=p, base_exposure_fraction=f))
    assert 0 <= e["exposure_fraction"] <= 1
    assert e["expected_recoverable_value"] <= e["potential_revenue_exposure"] + 1e-6
    assert np.isfinite(e["net_opportunity_value"])


@given(st.floats(-1e6, -0.01))
def test_negative_inputs_rejected(v):
    import pytest

    with pytest.raises(ValueError):
        Assumptions(eng_cost_per_person_week=v).validated()
    with pytest.raises(ValueError):
        Assumptions(assumed_acv_per_account=v).validated()


def test_missing_acv_never_silently_becomes_zero():
    df = pd.DataFrame({"account_id": ["a", "b"], "annual_contract_value": [np.nan, np.nan], "customer_segment": "Unknown"})
    r = resolve_acv(df, impute=True)
    assert r["acv_resolved"].isna().all() and (r["acv_source"] == "missing").all()
    r2 = resolve_acv(df, impute=True, assumed=100.0)
    assert (r2["acv_source"] == "user_assumption").all()


@settings(deadline=None, max_examples=15)
@given(rec=st.floats(0.05, 0.95))
def test_scenario_change_updates_dependent_values_and_keeps_overrides(rec):
    clustered, a, opps, ranked = demo()
    cid = int(ranked.cluster_id.iloc[-1])
    ov = Overrides(rank={cid: {"rank": 1, "rationale": "exec"}}, rename={cid: "Renamed"})
    a2 = Assumptions(recovery_probability=rec)
    r = rank_opportunities(build_opportunities(clustered, a2, ov), a2, ov)
    assert r.iloc[0]["cluster_id"] == cid and r.iloc[0]["label"] == "Renamed"
    assert np.isclose(r["expected_recoverable_value"].sum() / r["potential_revenue_exposure"].sum(), rec)


# ---------------- metamorphic ML tests ----------------
def _mini():
    clustered, *_ = demo()
    return clustered.groupby("ground_truth_issue", group_keys=False).head(25)[["ticket_id", "feedback_text", "ground_truth_issue"]].reset_index(drop=True)


def _partition(df):
    c = cluster_feedback(df)
    return evaluate_clustering(c["ground_truth_issue"], c["cluster_id"]).adjusted_rand_index


def test_record_order_does_not_change_partition_quality():
    d = _mini()
    assert abs(_partition(d) - _partition(d.sample(frac=1, random_state=3).reset_index(drop=True))) < 0.1


def test_irrelevant_punctuation_does_not_change_partition_quality():
    d = _mini()
    noisy = d.assign(feedback_text=d.feedback_text + " !!! ...")
    assert abs(_partition(d) - _partition(noisy)) < 0.1


def test_duplicate_templates_collapse_not_split():
    d = pd.DataFrame({"ticket_id": [str(i) for i in range(30)],
                      "feedback_text": ["Export to PDF fails with a timeout"] * 15 + ["Mobile app crashes when opening settings"] * 15})
    assert cluster_feedback(d)["cluster_id"].nunique() == 2
