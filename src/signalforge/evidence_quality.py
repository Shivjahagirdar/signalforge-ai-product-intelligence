"""Evidence-quality engine: is a recommendation backed by enough, fresh, unbiased, non-contradictory evidence?

Each check is deterministic and explainable; the score is the weighted share of checks passed. "Missing" lists exactly
which inputs a PM must supply before committing capacity (e.g. a real engineering estimate, observed revenue).
"""
from __future__ import annotations

import pandas as pd

CHECKS = [  # key, weight, label
    ("volume", 0.15, "At least 10 supporting records"),
    ("breadth", 0.15, "At least 5 distinct accounts"),
    ("concentration", 0.15, "No single account contributes >40% of records (sample bias)"),
    ("revenue_observed", 0.15, "Revenue is observed, not imputed or assumed"),
    ("effort_estimate", 0.15, "Engineering effort is estimated from data, not a default"),
    ("freshness", 0.10, "Recent evidence (record in the last 90 days of the dataset)"),
    ("contradictions", 0.10, "Contradicting records are under 15% of evidence"),
    ("multi_source", 0.05, "Evidence from more than one source/channel"),
]


def assess(opp: dict, clustered: pd.DataFrame) -> dict:
    g = clustered[clustered["ticket_id"].isin(opp["ticket_ids"])]
    n = max(len(g), 1)
    top_share = float(g["account_id"].value_counts(normalize=True).iloc[0]) if len(g) else 1.0
    dates = pd.to_datetime(clustered["created_at"], errors="coerce", utc=True)
    gd = pd.to_datetime(g["created_at"], errors="coerce", utc=True)
    if dates.notna().any() and gd.notna().any():
        fresh = bool(gd.max() >= dates.max() - pd.Timedelta(days=90))
        age_days = int((dates.max() - gd.max()).days)
    else:
        fresh, age_days = False, None
    contra = len(opp.get("contradictions", [])) / n
    sources = g["source"].nunique() if "source" in g else 1
    observed = opp.get("accounts_observed", 0)
    total_acc = max(opp["unique_accounts"], 1)
    results = {
        "volume": opp["tickets"] >= 10,
        "breadth": opp["unique_accounts"] >= 5,
        "concentration": top_share <= 0.40,
        "revenue_observed": observed / total_acc >= 0.8,
        "effort_estimate": opp.get("effort_weeks") is not None,
        "freshness": fresh,
        "contradictions": contra < 0.15,
        "multi_source": sources > 1,
    }
    score = sum(w for k, w, _ in CHECKS if results[k])
    missing = [lab for k, _, lab in CHECKS if not results[k]]
    if score >= 0.75:
        grade = "strong"
    elif score >= 0.5:
        grade = "moderate"
    else:
        grade = "weak"
    return {"score": round(score, 2), "grade": grade, "checks": results, "missing": missing,
            "top_account_share": round(top_share, 2), "contradiction_share": round(contra, 2),
            "latest_record_age_days": age_days, "sources": int(sources),
            "next_step": ("Evidence is strong enough to commit, pending approval." if grade == "strong" else
                          "Validate before committing capacity: " + "; ".join(missing[:3]) + ".")}


def table(opps: list[dict], clustered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for o in opps:
        q = assess(o, clustered)
        rows.append({"cluster_id": o["cluster_id"], "theme": o["label"], "evidence_score": q["score"], "grade": q["grade"],
                     "missing": "; ".join(q["missing"]) or "-"})
    return pd.DataFrame(rows)
