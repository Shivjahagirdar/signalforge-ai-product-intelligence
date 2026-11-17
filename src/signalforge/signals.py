"""Signal intelligence: transparent rule-based tagging, volume-normalised trend detection, cluster explanations.

Rules are deliberately simple and inspectable (keyword patterns), so every tag can be traced to the words that
triggered it. Nothing here is a churn *prediction*: 'churn_language' only means the text mentions leaving/cancelling.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIGNAL_RULES: dict[str, str] = {
    "reliability": r"\b(crash\w*|outage|down(time)?|fail\w*|broken|error\w*|bug\w*|freez\w*|hang\w*|lost data|data loss)\b",
    "performance": r"\b(slow\w*|lag\w*|latency|timeout\w*|takes forever|performance|loading)\b",
    "integration": r"\b(integrat\w*|api|webhook\w*|sync\w*|salesforce|slack|jira|sso|okta|zapier|export|import)\b",
    "onboarding": r"\b(onboard\w*|setup|set up|getting started|tutorial|confus\w*|learning curve|first time)\b",
    "usability": r"\b(ui|ux|hard to (find|use)|unintuitive|clunky|cluttered|can'?t find|difficult to)\b",
    "pricing": r"\b(pric\w*|expensive|cost\w*|billing|invoice\w*|refund\w*|charge\w*|subscription|plan)\b",
    "feature_request": r"\b(would (be (nice|great|useful)|love)|we need|please add|feature request|wish|need(s)? (a|an|the)? ?option|support for|ability to)\b",
    "competitor_mention": r"\b(competitor|alternative|switch(ing)? to|moved to|better than|compared to)\b",
    "churn_language": r"\b(cancel\w*|uninstall\w*|leav(e|ing)|churn\w*|switch(ing)? to|not renew\w*|deleting|delete (my )?account)\b",
    "security_privacy": r"\b(security|privacy|permission\w*|password|2fa|mfa|gdpr|leak\w*|tracking)\b",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in SIGNAL_RULES.items()}


def tag_signals(text: str) -> list[str]:
    return [k for k, rx in _COMPILED.items() if rx.search(text or "")]


def add_signal_tags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["signals"] = d["feedback_text"].map(tag_signals)
    return d


def cluster_signal_profile(g: pd.DataFrame) -> dict[str, float]:
    c = Counter(s for tags in g["signals"] for s in tags)
    return {k: round(v / len(g), 3) for k, v in c.most_common()}


def representative_records(g: pd.DataFrame, k: int = 3) -> list[dict]:
    """Records closest to the cluster's TF-IDF centroid: shows WHY records were grouped."""
    texts = g["feedback_text"].astype(str).tolist()
    if len(texts) <= k:
        return [dict(ticket_id=t, text=x, similarity=1.0) for t, x in zip(g["ticket_id"], texts)]
    try:
        x = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    except ValueError:
        return [dict(ticket_id=t, text=x, similarity=0.0) for t, x in zip(g["ticket_id"][:k], texts[:k])]
    cen = np.asarray(x.mean(axis=0))
    sim = cosine_similarity(x, cen).ravel()
    idx = np.argsort(-sim)[:k]
    return [dict(ticket_id=str(g["ticket_id"].iloc[i]), text=texts[i], similarity=round(float(sim[i]), 3)) for i in idx]


def shared_terms(g: pd.DataFrame, top: int = 6) -> list[tuple[str, float]]:
    """Terms present in the largest share of the cluster's records (the grouping rationale)."""
    c = Counter(w for t in g["feedback_text"] for w in set(re.findall(r"[a-z]{4,}", str(t).lower())))
    stop = set(TfidfVectorizer(stop_words="english").get_stop_words())
    return [(w, round(n / len(g), 2)) for w, n in c.most_common(60) if w not in stop][:top]


def contradictory_evidence(g: pd.DataFrame) -> list[str]:
    """Records in the cluster that read as praise / 'works for me' - evidence that cuts against the theme."""
    rx = re.compile(r"\b(works (fine|great|well)|no (issue|problem)s?|love (it|this)|perfect|excellent)\b", re.I)
    return [str(t) for t, x in zip(g["ticket_id"], g["feedback_text"]) if rx.search(str(x))][:10]


def trend(g: pd.DataFrame, all_df: pd.DataFrame, periods: int = 2) -> dict:
    """Share-of-volume trend: cluster share in the latest window vs. the prior window.

    Normalising by total volume avoids calling a theme 'rising' just because overall feedback grew."""
    dates = pd.to_datetime(all_df["created_at"], errors="coerce")
    if dates.notna().sum() < 20:
        return dict(status="insufficient_dates", share_now=None, share_prev=None, lift=None)
    cut2 = dates.max()
    span = (cut2 - dates.min()) / periods
    cut1 = cut2 - span
    cut0 = cut1 - span
    gd = pd.to_datetime(g["created_at"], errors="coerce")
    now_all = ((dates > cut1) & (dates <= cut2)).sum()
    prev_all = ((dates > cut0) & (dates <= cut1)).sum()
    now = ((gd > cut1) & (gd <= cut2)).sum()
    prev = ((gd > cut0) & (gd <= cut1)).sum()
    if now_all == 0 or prev_all == 0:
        return dict(status="insufficient_dates", share_now=None, share_prev=None, lift=None)
    s_now, s_prev = now / now_all, prev / prev_all
    lift = (s_now + 1e-9) / (s_prev + 1e-9)
    status = "emerging" if prev <= 1 and now >= 5 else "rising" if lift >= 1.5 and now >= 3 else "falling" if lift <= 0.67 else "stable"
    return dict(status=status, share_now=round(float(s_now), 4), share_prev=round(float(s_prev), 4), lift=round(float(lift), 2),
                n_now=int(now), n_prev=int(prev))


def high_severity_low_frequency(g: pd.DataFrame) -> bool:
    return bool(g["severity"].mean() >= 4 and len(g) < 8)


def is_actionable(g: pd.DataFrame, profile: dict) -> bool:
    """Short, signal-free chatter ('thanks for the call', 'great product') is not a product problem.
    Kept visible as evidence, but not ranked as an investment."""
    words = g["feedback_text"].astype(str).str.split().str.len().median()
    return bool(profile) or words >= 8
