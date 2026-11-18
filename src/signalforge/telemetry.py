"""Product telemetry: documented event schema, metric semantics, a reproducible SIMULATED sample, and
release-to-outcome evaluation.

Behavioural telemetry is never inferred from feedback text. Either the user uploads events that match ``EVENT_COLUMNS``
(labelled "your telemetry"), or the app generates the seeded sample below (labelled "SIMULATED"; the injected effect is
a known parameter so the evaluator can be checked against ground truth).

Metric definitions (single source of truth - every page and SIGNAL use these functions):
* active account (window): account with >=1 event in the window.
* feature adoption (window): share of active accounts with >=1 ``feature_used`` event for the feature in the window.
* activation: share of accounts whose first ``key_action`` happens within 7 days of ``signup``.
* 4-week retention (guardrail): share of accounts active in the 28 days before the release that are also active in the
  28 days after it.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["event_id", "ts", "account_id", "user_id", "event_name", "feature"]
EVENT_NAMES = {"signup", "session_start", "key_action", "feature_used"}
WINDOW_DAYS = 28
MAX_EVENTS = 500_000


class TelemetryError(ValueError):
    pass


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise TelemetryError(f"Telemetry is missing columns: {', '.join(missing)}. Required: {', '.join(EVENT_COLUMNS)}.")
    if len(df) > MAX_EVENTS:
        raise TelemetryError(f"More than {MAX_EVENTS:,} events; sample or aggregate first.")
    out = df[EVENT_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)
    bad = out["ts"].isna().sum()
    out = out.dropna(subset=["ts"])
    unknown = ~out["event_name"].isin(EVENT_NAMES)
    if unknown.all():
        raise TelemetryError(f"No recognised event_name values. Use: {', '.join(sorted(EVENT_NAMES))}.")
    out = out[~unknown]
    out = out.drop_duplicates("event_id")
    out["feature"] = out["feature"].fillna("").astype(str)
    out.attrs["dropped_bad_ts"] = int(bad)
    out.attrs["dropped_unknown_event"] = int(unknown.sum())
    return out.sort_values("ts").reset_index(drop=True)


def read_events_csv(raw: bytes) -> pd.DataFrame:
    return validate_events(pd.read_csv(io.BytesIO(raw)))


def simulate_events(features: list[str], release_feature: str | None, release_date: str, true_lift: float = 0.12,
                    accounts: int = 160, days: int = 120, seed: int = 7) -> pd.DataFrame:
    """SIMULATED events. ``true_lift`` is the absolute increase in weekly use probability of ``release_feature`` after
    ``release_date`` (ground truth for the evaluator)."""
    rng = np.random.default_rng(seed)
    rel = pd.Timestamp(release_date, tz="UTC")
    start = rel - pd.Timedelta(days=days // 2)
    rows = []
    base_p = {f: rng.uniform(0.08, 0.3) for f in features}
    for a in range(accounts):
        acc = f"acct-{a:04d}"
        signup = start - pd.Timedelta(days=int(rng.integers(0, 60)))
        users = [f"{acc}-u{u}" for u in range(int(rng.integers(1, 7)))]
        rows.append((signup, acc, users[0], "signup", ""))
        if rng.random() < 0.7:
            rows.append((signup + pd.Timedelta(days=float(rng.uniform(0, 10))), acc, users[0], "key_action", ""))
        churn_day = days if rng.random() > 0.12 else int(rng.integers(10, days))
        engagement = rng.uniform(0.4, 1.0)
        for w in range(days // 7):
            if w * 7 > churn_day:
                break
            day = start + pd.Timedelta(days=w * 7)
            if rng.random() > engagement:
                continue
            u = users[int(rng.integers(len(users)))]
            rows.append((day + pd.Timedelta(hours=float(rng.uniform(0, 160))), acc, u, "session_start", ""))
            for f in features:
                p = base_p[f] * (0.6 + engagement)
                if f == release_feature and day >= rel:
                    p += true_lift
                if rng.random() < min(p, 0.95):
                    rows.append((day + pd.Timedelta(hours=float(rng.uniform(0, 160))), acc, u, "feature_used", f))
    df = pd.DataFrame(rows, columns=["ts", "account_id", "user_id", "event_name", "feature"])
    df.insert(0, "event_id", [f"ev-{i:07d}" for i in range(len(df))])
    df = validate_events(df)
    df.attrs["simulated"] = True
    df.attrs["true_lift"] = true_lift
    return df


def _window(df: pd.DataFrame, end: pd.Timestamp, days: int) -> pd.DataFrame:
    return df[(df["ts"] >= end - pd.Timedelta(days=days)) & (df["ts"] < end)]


def adoption(df: pd.DataFrame, feature: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, int]:
    w = df[(df["ts"] >= start) & (df["ts"] < end)]
    active = w["account_id"].unique()
    if len(active) == 0:
        return float("nan"), 0
    users = w[(w["event_name"] == "feature_used") & (w["feature"] == feature)]["account_id"].nunique()
    return users / len(active), len(active)


def activation_rate(df: pd.DataFrame, within_days: int = 7) -> float:
    s = df[df.event_name == "signup"].groupby("account_id")["ts"].min()
    k = df[df.event_name == "key_action"].groupby("account_id")["ts"].min()
    j = pd.concat([s.rename("s"), k.rename("k")], axis=1).dropna(subset=["s"])
    if j.empty:
        return float("nan")
    return float(((j["k"] - j["s"]) <= pd.Timedelta(days=within_days)).mean())


def retention(df: pd.DataFrame, at: pd.Timestamp, days: int = WINDOW_DAYS) -> float:
    before = set(_window(df, at, days)["account_id"])
    after = set(df[(df.ts >= at) & (df.ts < at + pd.Timedelta(days=days))]["account_id"])
    return len(before & after) / len(before) if before else float("nan")


def feature_catalog(df: pd.DataFrame) -> list[str]:
    return sorted(f for f in df.loc[df.event_name == "feature_used", "feature"].unique() if f)


def evaluate_release(df: pd.DataFrame, feature: str, release_date: str, baseline: float | None = None, target: float | None = None,
                     days: int = WINDOW_DAYS, boots: int = 400, seed: int = 3, guardrail_drop: float = 0.05) -> dict:
    """Pre/post comparison of feature adoption with an account-level bootstrap CI and a retention guardrail.
    This is an observational before/after estimate, not a causal experiment; seasonality and mix shifts are not removed."""
    rel = pd.Timestamp(release_date, tz="UTC")
    if df.empty:
        raise TelemetryError("No telemetry loaded.")
    if df["ts"].max() < rel + pd.Timedelta(days=7):
        return {"verdict": "too_early", "explanation": "Less than 7 days of post-release telemetry; check back later.",
                "feature": feature}
    pre_s, post_e = rel - pd.Timedelta(days=days), rel + pd.Timedelta(days=days)
    pre, n_pre = adoption(df, feature, pre_s, rel)
    post, n_post = adoption(df, feature, rel, post_e)
    rng = np.random.default_rng(seed)

    def per_account(start, end):
        w = df[(df.ts >= start) & (df.ts < end)]
        act = pd.Series(1, index=w["account_id"].unique())
        used = set(w[(w.event_name == "feature_used") & (w.feature == feature)]["account_id"])
        return act.index.map(lambda a: a in used).to_numpy(dtype=float)

    a_pre, a_post = per_account(pre_s, rel), per_account(rel, post_e)
    diffs = []
    if len(a_pre) and len(a_post):
        for _ in range(boots):
            diffs.append(rng.choice(a_post, len(a_post)).mean() - rng.choice(a_pre, len(a_pre)).mean())
    lo, hi = (np.percentile(diffs, [5, 95]) if diffs else (np.nan, np.nan))
    lift = post - pre
    base = pre if baseline is None else baseline
    ret = retention(df, rel, days)
    ret_prev = retention(df, rel - pd.Timedelta(days=days), days)
    guard_ok = not (np.isfinite(ret) and np.isfinite(ret_prev) and ret < ret_prev - guardrail_drop)
    if not np.isfinite(lift):
        verdict = "insufficient_data"
    elif lo > 0 and (target is None or post >= target) and guard_ok:
        verdict = "worked"
    elif lo > 0 and guard_ok:
        verdict = "improved_below_target"
    elif hi < 0:
        verdict = "regressed"
    elif not guard_ok:
        verdict = "guardrail_breached"
    else:
        verdict = "inconclusive"
    return {"feature": feature, "release_date": str(rel.date()), "window_days": days,
            "adoption_pre": float(pre), "adoption_post": float(post), "lift": float(lift),
            "lift_ci90": [float(lo), float(hi)], "accounts_pre": n_pre, "accounts_post": n_post,
            "baseline": float(base) if base is not None and np.isfinite(base) else None, "target": target,
            "retention_post": float(ret), "retention_prior": float(ret_prev), "guardrail_ok": bool(guard_ok),
            "verdict": verdict, "method": "pre/post adoption, account bootstrap 90% CI, retention guardrail (observational)"}
