"""Shared fixtures built once per test session (clustering is the expensive step)."""
from functools import lru_cache

from signalforge.economics import Assumptions
from signalforge.ingestion import validate
from signalforge.intelligence import cluster_feedback
from signalforge.optimizer import Constraints
from signalforge.pipeline import build_opportunities, rank_opportunities
from signalforge.synthetic import generate


@lru_cache(maxsize=1)
def demo():
    clustered = cluster_feedback(validate(generate()).df, 0.9)
    a = Assumptions()
    opps = build_opportunities(clustered, a)
    return clustered, a, opps, rank_opportunities(opps, a)


def copilot_state(capacity=20.0):
    from signalforge.copilot import CopilotState

    clustered, a, opps, ranked = demo()
    return CopilotState(ranked, {o["cluster_id"]: o for o in opps}, a, Constraints(capacity_weeks=capacity))
