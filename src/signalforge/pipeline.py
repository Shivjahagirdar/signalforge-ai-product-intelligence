"""Opportunity building, ranking, scenarios and explanations (domain logic, no UI)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .audit import Overrides
from .economics import (
    Assumptions,
    evaluate_opportunity,
    portfolio_exposure,
    resolve_acv,
    risk_adjusted_priority,
    unique_account_acv,
)


def confidence(cohesion: float, unique_accounts: int, tickets: int) -> float:
    """Evidence-quality heuristic in [0,1] (cohesion + breadth + volume). NOT a calibrated probability."""
    return float(np.clip(0.5 * min(max(cohesion, 0), 1) + 0.3 * min(1, unique_accounts / 5) + 0.2 * min(1, tickets / 20), 0, 1))


def build_opportunities(clustered: pd.DataFrame, a: Assumptions, ov: Overrides | None = None) -> list[dict]:
    ov = ov or Overrides()
    d = clustered[~clustered["ticket_id"].isin(ov.irrelevant)].copy()
    remap = {c: grp[0] for grp in ov.merge for c in grp}
    d["cluster_id"] = d["cluster_id"].map(lambda c: remap.get(c, c))
    d = resolve_acv(d, a.impute_missing_acv)
    opps = []
    for cid, g in d.groupby("cluster_id"):
        if len(g) < 3:  # outliers stay as unclustered evidence, not opportunities
            continue
        acc = unique_account_acv(g)
        acc_valid = acc.dropna(subset=["acv"])
        eff = g["estimated_engineering_effort"].median()
        label = ov.rename.get(int(cid), g["cluster_label"].iloc[0])
        opps.append(dict(
            cluster_id=int(cid), label=label, tickets=len(g), ticket_ids=g["ticket_id"].tolist(),
            unique_accounts=int(acc["account_id"].nunique()), acv_sum_unique=float(acc_valid["acv"].sum()),
            account_acv=dict(zip(acc_valid["account_id"], acc_valid["acv"])),
            accounts_imputed=int((acc["source"] == "imputed_segment_median").sum()),
            accounts_excluded=int(acc["acv"].isna().sum()),
            avg_severity=float(g["severity"].mean()), total_frequency=int(g["frequency"].sum()),
            segments=g.groupby("customer_segment")["account_id"].nunique().to_dict(),
            effort_weeks=None if pd.isna(eff) else float(eff), cohesion=float(g["cohesion"].mean()),
            product_area=g["product_area"].mode().iloc[0],
            confidence=confidence(float(g["cohesion"].mean()), acc["account_id"].nunique(), len(g)),
            renamed=int(cid) in ov.rename))
    return opps


def rank_opportunities(opps: list[dict], a: Assumptions, ov: Overrides | None = None) -> pd.DataFrame:
    ov = ov or Overrides()
    rows = []
    for o in opps:
        e = evaluate_opportunity(o, a)
        rice = o["unique_accounts"] * (o["avg_severity"] / 5 * 3) * o["confidence"] / max(e["effort_weeks"], 0.5)
        rows.append({**{k: v for k, v in o.items() if k not in ("ticket_ids", "account_acv", "segments")}, **e,
                     "priority": risk_adjusted_priority(e, o["confidence"]), "rice": rice,
                     "baseline_score": o["avg_severity"] * o["total_frequency"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["priority", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    df["model_rank"] = np.arange(1, len(df) + 1)
    df["baseline_rank"] = df["baseline_score"].rank(ascending=False, method="first").astype(int)
    df["rice_rank"] = df["rice"].rank(ascending=False, method="first").astype(int)
    order = df["cluster_id"].tolist()
    for cid, spec in sorted(ov.rank.items(), key=lambda kv: kv[1]["rank"]):
        if cid in order:
            order.remove(cid)
            order.insert(max(0, min(len(order), spec["rank"] - 1)), cid)
    pos = {c: i + 1 for i, c in enumerate(order)}
    df["final_rank"] = df["cluster_id"].map(pos)
    df["override_rationale"] = df["cluster_id"].map(lambda c: ov.rank.get(c, {}).get("rationale", ""))
    return df.sort_values("final_rank").reset_index(drop=True)


def scenario_assumptions(a: Assumptions) -> dict[str, Assumptions]:
    return {
        "bear": replace(a, recovery_probability=min(1, a.recovery_probability * 0.6),
                        base_exposure_fraction=min(1, a.base_exposure_fraction * 0.7),
                        eng_cost_per_person_week=a.eng_cost_per_person_week * 1.3),
        "base": a,
        "bull": replace(a, recovery_probability=min(1, a.recovery_probability * 1.3),
                        base_exposure_fraction=min(1, a.base_exposure_fraction * 1.2),
                        eng_cost_per_person_week=a.eng_cost_per_person_week * 0.9),
    }


def scenario_ranks(opps: list[dict], a: Assumptions) -> pd.DataFrame:
    out = {}
    for name, sa in scenario_assumptions(a).items():
        r = rank_opportunities(opps, sa)
        out[name] = r.set_index("cluster_id")["model_rank"]
    t = pd.DataFrame(out)
    t["label"] = {o["cluster_id"]: o["label"] for o in opps}
    return t.sort_values("base")


def explain_outranks(hi: pd.Series, lo: pd.Series) -> str:
    parts = []
    if hi["net_opportunity_value"] > lo["net_opportunity_value"]:
        parts.append(f"higher modeled net value (${hi['net_opportunity_value']:,.0f} vs ${lo['net_opportunity_value']:,.0f})")
    if hi["effort_weeks"] < lo["effort_weeks"]:
        parts.append(f"lower effort ({hi['effort_weeks']:.1f} vs {lo['effort_weeks']:.1f} person-weeks)")
    if hi["confidence"] > lo["confidence"]:
        parts.append(f"stronger evidence quality ({hi['confidence']:.2f} vs {lo['confidence']:.2f})")
    if hi["acv_sum_unique"] > lo["acv_sum_unique"]:
        parts.append(f"more unique-account ACV (${hi['acv_sum_unique']:,.0f} vs ${lo['acv_sum_unique']:,.0f}) "
                     f"from {hi['unique_accounts']} vs {lo['unique_accounts']} accounts")
    why = "; ".join(parts) or "a higher combined risk-adjusted priority score"
    return f"'{hi['label']}' outranks '{lo['label']}' on: {why}."


def portfolio_total(opps: list[dict], a: Assumptions) -> float:
    return portfolio_exposure(opps, [evaluate_opportunity(o, a) for o in opps])
