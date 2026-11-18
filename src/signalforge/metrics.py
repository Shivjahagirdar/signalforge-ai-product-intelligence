"""In-session product metrics, computed from the audit trail. Measured per browser session, never stored or shared.

These are *instrumentation*, not outcomes: they show whether the product is being used as designed (time to first
insight, how often humans correct the machine, whether SIGNAL and exports get used). Human-review agreement with the
model needs real reviewers and is not claimed anywhere.
"""
from __future__ import annotations

CORRECTION_EVENTS = {"rename", "merge", "reject", "split", "irrelevant", "rank_override"}


def session_metrics(events: list[tuple], n_themes: int, seconds_to_first_insight: float | None) -> dict:
    kinds = [e[1] for e in events]
    corrections = sum(k in CORRECTION_EVENTS for k in kinds)
    return {
        "seconds_to_first_insight": None if seconds_to_first_insight is None else round(float(seconds_to_first_insight), 1),
        "corrections": corrections,
        "correction_rate": round(corrections / n_themes, 3) if n_themes else None,  # human edits per theme shown
        "signal_questions": kinds.count("copilot"),
        "exports": kinds.count("export"),
        "audit_events": len(kinds),
    }
