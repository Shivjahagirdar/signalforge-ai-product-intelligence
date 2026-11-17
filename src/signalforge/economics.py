"""Revenue-at-Risk financial model. All money in USD; ACV treated as annual contract value (not ARR)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    base_exposure_fraction: float = 0.30  # share of an affected account's ACV considered exposed (scenario, not churn)
    recovery_probability: float = 0.40  # P(fix recovers/protects exposed revenue)
    eng_cost_per_person_week: float = 4000.0  # USD / person-week
    infra_cost_annual: float = 2000.0  # incremental USD / year
    support_cost_annual: float = 3000.0  # incremental USD / year
    benefit_horizon_months: int = 12
    default_effort_weeks: float = 4.0
    impute_missing_acv: bool = True
    assumed_acv_per_account: float | None = None  # explicit USER ASSUMPTION for data with no revenue (e.g. public reviews)
    strategic_alignment: dict = field(default_factory=dict)  # cluster_id -> 0.5..1.5, default 1.0

    def validated(self) -> Assumptions:
        if not (0 <= self.base_exposure_fraction <= 1) or not (0 <= self.recovery_probability <= 1):
            raise ValueError("Fractions and probabilities must be within [0, 1].")
        if min(self.eng_cost_per_person_week, self.infra_cost_annual, self.support_cost_annual) < 0:
            raise ValueError("Costs must be non-negative.")
        if self.assumed_acv_per_account is not None and self.assumed_acv_per_account < 0:
            raise ValueError("Assumed ACV must be non-negative.")
        if self.benefit_horizon_months <= 0 or self.default_effort_weeks <= 0:
            raise ValueError("Horizon and default effort must be positive.")
        return self


def resolve_acv(df: pd.DataFrame, impute: bool = True, assumed: float | None = None) -> pd.DataFrame:
    """Add acv_resolved + acv_source (observed | imputed_segment_median | user_assumption | missing).

    Missing ACV never silently becomes zero: it stays NaN ('missing') unless imputed or explicitly assumed."""
    d = df.copy()
    acv = pd.to_numeric(d["annual_contract_value"], errors="coerce")
    seg_med = acv.groupby(d["customer_segment"]).transform("median")
    use_med = impute & seg_med.notna()
    d["acv_source"] = np.where(acv.notna(), "observed", np.where(use_med, "imputed_segment_median",
                               "user_assumption" if assumed is not None else "missing"))
    d["acv_resolved"] = acv.fillna(seg_med if impute else np.nan)
    if assumed is not None:
        d["acv_resolved"] = d["acv_resolved"].fillna(float(assumed))
    return d


def unique_account_acv(d: pd.DataFrame) -> pd.DataFrame:
    """One row per account (max ACV) so repeated tickets never inflate exposure."""
    g = d.groupby("account_id").agg(acv=("acv_resolved", "max"), segment=("customer_segment", "first"),
                                    source=("acv_source", "first"))
    return g.reset_index()


def severity_multiplier(avg_severity: float) -> float:
    return float(np.clip(avg_severity / 3.0, 0.2, 1.67))


def evaluate_opportunity(opp: dict, a: Assumptions) -> dict:
    """Compute exposure, recoverable value, investment, net value, ROI, payback, cost of delay."""
    a = a.validated()
    frac = float(np.clip(a.base_exposure_fraction * severity_multiplier(opp["avg_severity"]), 0, 1))
    exposure = opp["acv_sum_unique"] * frac
    recoverable = exposure * a.recovery_probability
    eff = opp["effort_weeks"] if opp["effort_weeks"] and opp["effort_weeks"] > 0 else a.default_effort_weeks
    eng = eff * a.eng_cost_per_person_week
    opex = a.infra_cost_annual + a.support_cost_annual
    invest = eng + opex
    net = recoverable - invest
    monthly = recoverable / a.benefit_horizon_months
    return dict(
        exposure_fraction=frac, potential_revenue_exposure=exposure, expected_recoverable_value=recoverable,
        engineering_cost=eng, operating_cost=opex, total_investment=invest, net_opportunity_value=net,
        roi=(net / invest) if invest > 0 else None,
        payback_months=(invest / monthly) if monthly > 0 else None,
        cost_of_delay_per_month=monthly, effort_weeks=eff,
        alignment=float(np.clip(a.strategic_alignment.get(opp["cluster_id"], 1.0), 0.5, 1.5)),
    )


def risk_adjusted_priority(econ: dict, confidence: float) -> float:
    """Net value x evidence confidence x strategic alignment / effort (person-weeks)."""
    return econ["net_opportunity_value"] * confidence * econ["alignment"] / max(econ["effort_weeks"], 0.5)


def portfolio_exposure(opps: list[dict], econs: list[dict]) -> float:
    """Total exposure without double counting: each account counts once at its max exposure fraction."""
    best: dict[str, float] = {}
    for o, e in zip(opps, econs):
        for acc, acv in o["account_acv"].items():
            best[acc] = max(best.get(acc, 0.0), acv * e["exposure_fraction"])
    return float(sum(best.values()))


def monte_carlo(opp: dict, a: Assumptions, n: int = 5000, seed: int = 7) -> dict:
    """Model-implied net value distribution conditional on assumptions (not an empirical probability)."""
    rng = np.random.default_rng(seed)
    base = evaluate_opportunity(opp, a)
    p = a.recovery_probability
    rec = rng.triangular(0.5 * p, p, min(1.0, 1.5 * p) or 1e-9, n)
    frac = np.clip(rng.triangular(0.7, 1.0, 1.3, n) * base["exposure_fraction"], 0, 1)
    adopt = rng.triangular(0.6, 0.85, 1.0, n)
    eff = rng.triangular(0.8, 1.0, 2.0, n) * base["effort_weeks"]
    opex = rng.triangular(0.8, 1.0, 1.5, n) * base["operating_cost"]
    net = opp["acv_sum_unique"] * frac * rec * adopt - eff * a.eng_cost_per_person_week - opex
    p10, p50, p90 = np.percentile(net, [10, 50, 90])
    return dict(p10=float(p10), p50=float(p50), p90=float(p90), prob_positive=float((net > 0).mean()), n=n)


def tornado(opp: dict, a: Assumptions, swing: float = 0.3) -> pd.DataFrame:
    """One-at-a-time +/- swing sensitivity of net opportunity value."""
    base = evaluate_opportunity(opp, a)["net_opportunity_value"]
    rows = []
    for name, field_name in [("Recovery probability", "recovery_probability"), ("Exposure fraction", "base_exposure_fraction"),
                             ("Eng cost / person-week", "eng_cost_per_person_week"), ("Infra cost", "infra_cost_annual"),
                             ("Support cost", "support_cost_annual")]:
        v = getattr(a, field_name)
        lo, hi = (evaluate_opportunity(opp, replace(a, **{field_name: min(1.0, v * m) if v <= 1 else v * m}))["net_opportunity_value"]
                  for m in (1 - swing, 1 + swing))
        rows.append((name, lo - base, hi - base))
    return pd.DataFrame(rows, columns=["driver", "delta_low", "delta_high"])
