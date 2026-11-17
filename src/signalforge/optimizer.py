"""Constraint-aware roadmap optimiser (0/1 integer program via scipy.optimize.milp, HiGHS).

maximise   sum(value_i x_i)
subject to sum(effort_i x_i)        <= capacity_weeks
           sum(investment_i x_i)    <= budget (optional)
           x_child <= x_parent      (dependencies)
           x_i = 1                  (mandatory: compliance / committed work)
           x_i = 0                  (confidence below threshold, or excluded)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class Constraints:
    capacity_weeks: float = 40.0
    budget_usd: float | None = None
    min_confidence: float = 0.0
    mandatory: list = field(default_factory=list)  # cluster_ids
    excluded: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)  # [(child, parent)]: child requires parent
    value_column: str = "net_opportunity_value"


@dataclass
class RoadmapResult:
    selected: pd.DataFrame
    deferred: pd.DataFrame
    status: str
    total_value: float
    effort_used: float
    utilization: float
    binding: list[str]
    infeasible_reason: str | None = None


def optimise(ranked: pd.DataFrame, c: Constraints) -> RoadmapResult:
    r = ranked.reset_index(drop=True)
    n = len(r)
    if n == 0:
        return RoadmapResult(r, r, "empty", 0.0, 0.0, 0.0, [])
    ids = r["cluster_id"].tolist()
    pos = {cid: i for i, cid in enumerate(ids)}
    value = r[c.value_column].astype(float).to_numpy()
    effort = r["effort_weeks"].astype(float).to_numpy()
    invest = r["total_investment"].astype(float).to_numpy()
    lo, hi = np.zeros(n), np.ones(n)
    for cid in c.excluded:
        if cid in pos:
            hi[pos[cid]] = 0
    low_conf = r["confidence"].to_numpy() < c.min_confidence
    hi[low_conf] = 0
    for cid in c.mandatory:
        if cid in pos:
            lo[pos[cid]] = 1
            hi[pos[cid]] = 1
    if (lo > hi).any():
        return RoadmapResult(r.iloc[0:0], r, "infeasible", 0.0, 0.0, 0.0, [], "A mandatory item is also excluded.")
    rows, lbs, ubs = [effort], [-np.inf], [c.capacity_weeks]
    if c.budget_usd is not None:
        rows.append(invest); lbs.append(-np.inf); ubs.append(c.budget_usd)
    for child, parent in c.dependencies:
        if child in pos and parent in pos:
            v = np.zeros(n); v[pos[child]] = 1; v[pos[parent]] = -1
            rows.append(v); lbs.append(-np.inf); ubs.append(0)
    res = milp(c=-value, constraints=LinearConstraint(np.vstack(rows), lbs, ubs), integrality=np.ones(n), bounds=Bounds(lo, hi))
    if not res.success:
        reason = "Mandatory work exceeds capacity or budget." if c.mandatory else res.message
        return RoadmapResult(r.iloc[0:0], r, "infeasible", 0.0, 0.0, 0.0, [], reason)
    x = np.round(res.x).astype(bool)
    sel, dfr = r[x].copy(), r[~x].copy()
    used = float(effort[x].sum())
    binding = []
    if c.capacity_weeks - used < max(effort[~x].min() if (~x).any() else np.inf, 0.5) and (~x).any():
        binding.append("engineering capacity")
    if c.budget_usd is not None and (~x).any() and c.budget_usd - invest[x].sum() < invest[~x].min():
        binding.append("budget")
    if low_conf.any():
        binding.append(f"confidence threshold ({int(low_conf.sum())} item(s) blocked)")
    if any(d[0] in pos for d in c.dependencies):
        binding.append("dependencies")
    dfr["deferral_reason"] = np.where(low_conf[~x], "below confidence threshold",
                                      np.where(r[~x]["cluster_id"].isin(c.excluded), "excluded by user",
                                               np.where(value[~x] <= 0, "negative modeled value", "does not fit capacity/budget")))
    return RoadmapResult(sel, dfr, "optimal", float(value[x].sum()), used,
                         used / c.capacity_weeks if c.capacity_weeks > 0 else 0.0, binding)


def greedy_baseline(ranked: pd.DataFrame, c: Constraints) -> float:
    """Value of 'sort by score and fill until full' - shows what optimisation adds over a ranked list."""
    total, used = 0.0, 0.0
    for r in ranked.sort_values("priority", ascending=False).itertuples():
        v = getattr(r, c.value_column)
        if v > 0 and r.confidence >= c.min_confidence and used + r.effort_weeks <= c.capacity_weeks:
            total += v; used += r.effort_weeks
    return total


def alternative(ranked: pd.DataFrame, c: Constraints, primary: RoadmapResult) -> RoadmapResult:
    """Best roadmap that does NOT contain the primary plan's highest-value item - a genuine alternative."""
    if primary.selected.empty:
        return primary
    top = primary.selected.sort_values(c.value_column, ascending=False)["cluster_id"].iloc[0]
    if top in c.mandatory:
        return primary
    c2 = Constraints(**{**c.__dict__, "excluded": list(c.excluded) + [top]})
    return optimise(ranked, c2)
