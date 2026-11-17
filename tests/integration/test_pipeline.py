import pytest

from signalforge.audit import AuditStore, Overrides
from signalforge.economics import Assumptions, monte_carlo
from signalforge.exports import generate_memo, generate_prd
from signalforge.ingestion import validate
from signalforge.intelligence import cluster_feedback, evaluate_clustering, near_duplicates
from signalforge.pipeline import build_opportunities, explain_outranks, rank_opportunities, scenario_ranks
from signalforge.synthetic import generate


@pytest.fixture(scope="module")
def clustered():
    return cluster_feedback(validate(generate()).df)


def test_clustering_quality_vs_ground_truth(clustered):
    m = evaluate_clustering(clustered["ground_truth_issue"], clustered["cluster_id"])
    assert m.adjusted_rand_index > 0.9 and m.pairwise_precision > 0.95 and m.coverage > 0.95


def test_clustering_is_deterministic():
    d = validate(generate()).df.head(300)
    assert cluster_feedback(d)["cluster_id"].tolist() == cluster_feedback(d)["cluster_id"].tolist()


def test_demo_scenario_integration_fix_outranks_cosmetic(clustered):
    a = Assumptions()
    r = rank_opportunities(build_opportunities(clustered, a), a)
    truth = clustered.groupby("cluster_id")["ground_truth_issue"].agg(lambda s: s.mode().iloc[0])
    integ = r[r.cluster_id.map(truth) == "integration_sync"].iloc[0]
    dark = r[r.cluster_id.map(truth) == "dark_mode"].iloc[0]
    assert integ.final_rank < dark.final_rank
    assert integ.unique_accounts < dark.unique_accounts and integ.acv_sum_unique > dark.acv_sum_unique
    assert integ.baseline_rank > dark.baseline_rank or integ.model_rank < dark.model_rank
    assert "outranks" in explain_outranks(integ, dark)


def test_rankings_reproducible_and_change_with_assumptions(clustered):
    a = Assumptions()
    o = build_opportunities(clustered, a)
    assert rank_opportunities(o, a).label.tolist() == rank_opportunities(o, a).label.tolist()
    harsh = Assumptions(recovery_probability=0.01, eng_cost_per_person_week=40_000)
    assert rank_opportunities(o, harsh).net_opportunity_value.max() < rank_opportunities(o, a).net_opportunity_value.max()
    assert set(scenario_ranks(o, a).columns) >= {"bear", "base", "bull"}


def test_overrides_rename_irrelevant_and_rank(clustered):
    a = Assumptions()
    o = build_opportunities(clustered, a)
    base = rank_opportunities(o, a)
    last = int(base.iloc[-1].cluster_id)
    ov = Overrides(rename={last: "Renamed"}, rank={last: {"rank": 1, "rationale": "strategic"}})
    r = rank_opportunities(build_opportunities(clustered, a, ov), a, ov)
    assert r.iloc[0].cluster_id == last and r.iloc[0].label == "Renamed" and r.iloc[0].override_rationale == "strategic"
    drop = o[0]["ticket_ids"][:2]
    o2 = build_opportunities(clustered, a, Overrides(irrelevant=drop))
    assert sum(x["tickets"] for x in o2) == sum(x["tickets"] for x in o) - 2


def test_merge_combines_clusters(clustered):
    a = Assumptions()
    o = build_opportunities(clustered, a)
    c1, c2 = o[0]["cluster_id"], o[1]["cluster_id"]
    m = build_opportunities(clustered, a, Overrides(merge=[[c1, c2]]))
    assert len(m) == len(o) - 1


def test_prd_and_memo_traceable(clustered):
    a = Assumptions()
    opps = build_opportunities(clustered, a)
    r = rank_opportunities(opps, a)
    top = r.iloc[0]
    opp = next(x for x in opps if x["cluster_id"] == top.cluster_id)
    prd = generate_prd(top, opp, a, monte_carlo(opp, a))
    assert opp["ticket_ids"][0] in prd and "Problem statement" in prd and "not verified churn" in prd
    assert "VALIDATE" in generate_memo(r, a, 1.0)


def test_near_duplicates_and_audit_store(clustered):
    assert list(near_duplicates(clustered.head(100)).columns) == ["ticket_a", "ticket_b", "similarity"]
    s = AuditStore()
    s.save_overrides(Overrides(rename={3: "x"}))
    s.log("test", "hello")
    assert s.load_overrides().rename == {3: "x"} and s.events()[0][1] == "test"
    s.reset()
    assert s.load_overrides().rename == {} and len(s.events()) == 1
