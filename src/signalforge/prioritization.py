"""Selectable prioritisation frameworks with visible formulas, normalisation and sensitivity analysis.

Correlated-input guard: in 'custom weighted' scoring, revenue impact and customer reach are both driven by the set of
affected accounts. We therefore use *revenue exposure* and *non-revenue reach* (accounts with no ACV evidence) as the
two impact inputs, so the same accounts are not rewarded twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FRAMEWORKS: dict[str, str] = {
    "Risk-adjusted ROI": "net_value x evidence_confidence x strategic_alignment / effort_weeks",
    "RICE": "(Reach=unique accounts) x (Impact=severity/5 x 3) x Confidence / Effort",
    "ICE": "Impact(1-10) x Confidence(1-10) x Ease(1-10); Impact = severity x 2, Ease = 10 / (1 + effort/2)",
    "WSJF": "(business value + time criticality + risk reduction) / job size; each 1-10 rank-scaled",
    "Cost of Delay": "monthly recoverable value forgone ($/month) / effort_weeks (CD3)",
    "Value vs Effort": "expected recoverable value / total investment",
    "Custom weighted": "sum(weight_i x minmax(input_i)); inputs: revenue exposure, non-revenue reach, severity, confidence, "
                       "urgency (trend), alignment, -effort, -risk",
}


@dataclass
class Weights:
    revenue: float = 0.30
    reach: float = 0.10
    severity: float = 0.15
    confidence: float = 0.15
    urgency: float = 0.10
    alignment: float = 0.10
    effort: float = 0.15
    risk: float = 0.05
    segment_importance: dict = field(default_factory=lambda: {"Enterprise": 1.0, "Mid-Market": 1.0, "SMB": 1.0, "Unknown": 1.0})


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float).fillna(0.0)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)


def _scale10(s: pd.Series) -> pd.Series:
    return 1 + 9 * _minmax(s)


def _urgency(trend_status: pd.Series) -> pd.Series:
    return trend_status.map({"emerging": 1.0, "rising": 0.8, "stable": 0.4, "falling": 0.1}).fillna(0.4)


def score(ranked: pd.DataFrame, framework: str, w: Weights | None = None) -> pd.Series:
    """Return a score per row (higher = do sooner). `ranked` is the output of rank_opportunities."""
    w = w or Weights()
    r = ranked
    eff = r["effort_weeks"].clip(lower=0.5)
    if framework == "Risk-adjusted ROI":
        return r["priority"]
    if framework == "RICE":
        return r["unique_accounts"] * (r["avg_severity"] / 5 * 3) * r["confidence"] / eff
    if framework == "ICE":
        return (r["avg_severity"] * 2) * (r["confidence"] * 10) * (10 / (1 + eff / 2))
    if framework == "WSJF":
        trend_s = r.get("trend_status", pd.Series("stable", index=r.index))
        bv = _scale10(r["expected_recoverable_value"])
        tc = 1 + 9 * _urgency(trend_s)
        rr = _scale10(r["avg_severity"] * r["unique_accounts"])
        return (bv + tc + rr) / _scale10(eff)
    if framework == "Cost of Delay":
        return r["cost_of_delay_per_month"] / eff
    if framework == "Value vs Effort":
        return r["expected_recoverable_value"] / r["total_investment"].replace(0, np.nan)
    if framework == "Custom weighted":
        seg_mult = r.get("dominant_segment", pd.Series("Unknown", index=r.index)).map(w.segment_importance).fillna(1.0)
        non_rev = (r["unique_accounts"] - r.get("accounts_observed", 0) - r.get("accounts_imputed", 0) - r.get("accounts_assumed", 0)).clip(lower=0)
        risk = 1 - r["confidence"]
        trend_s = r.get("trend_status", pd.Series("stable", index=r.index))
        base = (w.revenue * _minmax(r["potential_revenue_exposure"]) + w.reach * _minmax(non_rev)
                + w.severity * _minmax(r["avg_severity"]) + w.confidence * r["confidence"] + w.urgency * _urgency(trend_s)
                + w.alignment * _minmax(r["alignment"]) - w.effort * _minmax(eff) - w.risk * risk)
        return base * seg_mult
    raise ValueError(f"Unknown framework {framework}")


def compare_frameworks(ranked: pd.DataFrame, w: Weights | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"label": ranked["label"], "cluster_id": ranked["cluster_id"]})
    for f in FRAMEWORKS:
        out[f] = score(ranked, f, w).rank(ascending=False, method="first").astype(int)
    rank_cols = list(FRAMEWORKS)
    out["rank_spread"] = out[rank_cols].max(axis=1) - out[rank_cols].min(axis=1)
    return out.sort_values("Risk-adjusted ROI").reset_index(drop=True)


def weight_sensitivity(ranked: pd.DataFrame, w: Weights, swing: float = 0.5, top_k: int = 3) -> pd.DataFrame:
    """For each weight, scale it by (1 +/- swing) and report whether the top-k set changes.
    Tells executives which assumptions actually move the recommendation."""
    base_top = set(ranked.assign(s=score(ranked, "Custom weighted", w)).nlargest(top_k, "s")["cluster_id"])
    rows = []
    for name in ["revenue", "reach", "severity", "confidence", "urgency", "alignment", "effort", "risk"]:
        changed = []
        for m in (1 - swing, 1 + swing):
            w2 = Weights(**{**w.__dict__, name: getattr(w, name) * m})
            top = set(ranked.assign(s=score(ranked, "Custom weighted", w2)).nlargest(top_k, "s")["cluster_id"])
            changed.append(len(top ^ base_top) // 2)
        rows.append(dict(weight=name, value=getattr(w, name), top_k_changes_low=changed[0], top_k_changes_high=changed[1],
                         decision_sensitive=max(changed) > 0))
    return pd.DataFrame(rows)
