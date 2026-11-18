"""Closed loop: opportunity -> decision (human approval) -> engineering ticket + PRD -> release -> measured outcome ->
forecast-vs-actual retrospective. Orchestrates the store, economics, evidence quality and telemetry; all numbers come
from deterministic functions, never from an LLM.
"""
from __future__ import annotations

import re

import pandas as pd

from .economics import Assumptions
from .evidence_quality import assess
from .exports import generate_prd
from .store import Decision, Store
from .telemetry import evaluate_release


def feature_key(label: str) -> str:
    k = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")
    return k[:40] or "feature"


def propose(store: Store, ws: str, row: pd.Series, opp: dict, clustered: pd.DataFrame, a: Assumptions, owner: str,
            rationale: str, target_adoption: float | None = None, actor: str = "user") -> int:
    q = assess(opp, clustered)
    d = Decision(
        title=f"Invest in: {row['label']}"[:160], cluster_label=str(row["label"]), owner=owner.strip()[:80],
        rationale=rationale.strip()[:2000],
        assumptions={k: v for k, v in a.__dict__.items() if isinstance(v, int | float | bool) or v is None},
        evidence_ids=list(opp["ticket_ids"][:50]),
        metrics={"net_value": float(row["net_opportunity_value"]), "roi": None if pd.isna(row.get("roi")) else float(row["roi"]),
                 "effort_weeks": float(row["effort_weeks"]), "unique_accounts": int(row["unique_accounts"]),
                 "evidence_score": q["score"], "evidence_grade": q["grade"], "missing_evidence": q["missing"],
                 "rank": int(row["final_rank"])},
        feature_key=feature_key(row["label"]),
        target={"metric": "feature_adoption", "target": target_adoption},
    )
    return store.create_decision(ws, d, actor)


def ticket_markdown(d: Decision, opp: dict) -> tuple[str, str]:
    title = f"[SignalForge] {d.cluster_label}"[:120]
    ev = "\n".join(f"- {t}" for t in d.evidence_ids[:10])
    miss = "\n".join(f"- [ ] {m}" for m in d.metrics.get("missing_evidence", [])) or "- none"
    tgt = d.target.get("target")
    body = f"""## Problem
{opp['unique_accounts']} accounts report recurring issues in **{opp['product_area']}** (avg severity {opp['avg_severity']:.1f}/5,
{opp['tickets']} records). Decision #{d.id} approved by {d.approved_by or d.owner}.

## Why now
Modeled net value ${d.metrics['net_value']:,.0f} (assumption-based), effort {d.metrics['effort_weeks']:.1f} person-weeks,
evidence grade **{d.metrics['evidence_grade']}** ({d.metrics['evidence_score']:.0%}).

## Acceptance criteria
- [ ] Root cause for the top 3 representative records identified and fixed
- [ ] Regression test reproducing each fixed failure
- [ ] `feature_used` event with `feature="{d.feature_key}"` instrumented (telemetry schema v1)
- [ ] Release recorded in SignalForge so the outcome can be measured
- Success metric: feature adoption{f' >= {tgt:.0%}' if tgt else ' increases vs 28-day pre-release baseline'}; guardrail: 4-week retention does not drop >5 pts

## Evidence (source record IDs)
{ev}

## Open evidence gaps to close during discovery
{miss}
"""
    return title, body


def create_ticket(store: Store, ws: str, did: int, opp: dict, row: pd.Series, a: Assumptions, actor: str = "user") -> int:
    d = store.get_decision(ws, did)
    title, body = ticket_markdown(d, opp)
    return store.add_ticket(ws, did, title, body, generate_prd(row, opp, a), actor)


def measure(store: Store, ws: str, did: int, events: pd.DataFrame, data_label: str, feature: str | None = None,
            actor: str = "user") -> dict:
    d = store.get_decision(ws, did)
    rel = store.releases(ws, did)
    if not rel:
        raise ValueError("Record a release before measuring.")
    res = evaluate_release(events, feature or d.feature_key, rel[0]["released_at"], target=d.target.get("target"))
    res["forecast"] = retrospective(d, res)
    if res["verdict"] != "too_early":
        store.add_outcome(ws, did, res, data_label, actor)
    return res


def retrospective(d: Decision, res: dict) -> dict:
    """'We recommended this. Here's what shipped and whether the predicted impact materialised.'"""
    tgt = d.target.get("target")
    lines = [f"Recommended at rank #{d.metrics.get('rank')} with evidence grade {d.metrics.get('evidence_grade')} "
             f"and modeled net value ${d.metrics.get('net_value', 0):,.0f}."]
    if res.get("verdict") == "too_early":
        return {"summary": lines[0] + " Not enough post-release data yet.", "error": None}
    post, pre = res["adoption_post"], res["adoption_pre"]
    lines.append(f"Adoption moved {pre:.0%} -> {post:.0%} (lift {res['lift']:+.1%}, 90% CI {res['lift_ci90'][0]:+.1%} to {res['lift_ci90'][1]:+.1%}).")
    err = None
    if tgt:
        err = post - tgt
        lines.append(f"Target was {tgt:.0%}: {'met' if err >= 0 else 'missed'} by {abs(err):.1%}.")
    if not res["guardrail_ok"]:
        lines.append("Retention guardrail breached - investigate before scaling.")
    learn = {"worked": "Keep the assumptions that drove this rank; reuse the evidence pattern.",
             "improved_below_target": "Direction right, magnitude over-forecast: lower the recovery-probability assumption for similar themes.",
             "regressed": "Forecast was wrong: revisit severity weighting and check for contradictory evidence that was ignored.",
             "inconclusive": "Effect too small to detect: extend the window or run an A/B test before more investment.",
             "guardrail_breached": "Gain came at a retention cost: add retention as a hard constraint next time.",
             "insufficient_data": "Instrument the feature event; no adoption signal was found."}
    lines.append("Next time: " + learn.get(res["verdict"], ""))
    return {"summary": " ".join(lines), "error": err}
