"""Deterministic synthetic enterprise feedback generator (NOT real customer data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

ISSUES: dict[str, dict] = {
    "integration_sync": dict(
        area="Integrations", sev=(4, 5), effort=3.0, segment="Enterprise", n_accounts=12, per=(3, 6),
        templates=[
            "Salesforce sync fails silently and our opportunity records stop updating",
            "The CRM connector drops records during nightly sync, data mismatch between systems",
            "Integration with Salesforce throws a token refresh error and the sync job halts",
            "CRM sync is stuck, field mappings reset after the connector reauthorizes",
            "Our connector to Salesforce duplicates accounts after the sync retries",
        ]),
    "dark_mode": dict(
        area="UI", sev=(1, 2), effort=6.0, segment="SMB", n_accounts=55, per=(3, 6),
        templates=[
            "Please add a dark mode theme, the bright interface hurts my eyes at night",
            "Would love a dark theme option for the dashboard, white background is too bright",
            "Feature request: dark mode and nicer color themes for the web interface",
            "Can you offer a night theme? The light interface is tiring in the evening",
            "Dark appearance setting please, current colors feel dated and too bright",
        ]),
    "report_export": dict(
        area="Reporting", sev=(2, 4), effort=5.0, segment="Mixed", n_accounts=30, per=(2, 5),
        templates=[
            "Exporting large reports to CSV times out after a few minutes",
            "Scheduled report export is very slow and the download never completes",
            "Report export fails for big date ranges, spinner runs forever",
            "PDF report generation is slow and sometimes returns an empty file",
        ]),
    "api_rate_limit": dict(
        area="API", sev=(3, 4), effort=4.0, segment="Mixed", n_accounts=22, per=(2, 5),
        templates=[
            "API rate limits are too low for our batch jobs and we get 429 errors",
            "We keep hitting HTTP 429 throttling on the REST API during peak hours",
            "Need higher API quota, requests are throttled and our pipeline backs off",
            "Rate limiting on the public API breaks our automated ingestion scripts",
        ]),
    "onboarding": dict(
        area="Onboarding", sev=(2, 3), effort=5.0, segment="SMB", n_accounts=40, per=(2, 4),
        templates=[
            "Setup wizard is confusing and new users get lost during onboarding",
            "Onboarding tutorial is unclear, our team could not finish initial setup",
            "New hires struggle with the getting started guide and the first-time walkthrough",
            "Initial configuration steps are hard to follow, we needed support to onboard",
        ]),
    "audit_logs": dict(
        area="Security", sev=(3, 5), effort=10.0, segment="Enterprise", n_accounts=10, per=(3, 5),
        templates=[
            "We need audit logs of admin actions for our compliance review",
            "Missing audit trail for user permission changes, auditors require exportable logs",
            "Compliance team requires immutable audit logging and retention controls",
            "SOC 2 audit asks for admin activity logs which the product does not provide",
        ]),
    "mobile_crash": dict(
        area="Mobile", sev=(3, 5), effort=4.0, segment="Mixed", n_accounts=25, per=(2, 4),
        templates=[
            "The iOS app crashes on launch after the latest update",
            "Mobile application keeps crashing when opening the notifications screen",
            "Android app freezes and closes unexpectedly when uploading photos",
            "App crash on startup for several of our field staff phones",
        ]),
}
ANCHORS = {
    "integration_sync": ["salesforce", "crm", "sync", "connector"],
    "dark_mode": ["dark", "theme", "bright", "interface"],
    "report_export": ["report", "export", "slow", "download"],
    "api_rate_limit": ["api", "rate", "limit", "throttled"],
    "onboarding": ["onboarding", "setup", "tutorial", "confusing"],
    "audit_logs": ["audit", "logs", "compliance", "admin"],
    "mobile_crash": ["app", "crash", "mobile", "startup"],
}
PREFIX = ["", "", "Hi team,", "Urgent:", "FYI -", "Following up:", "Quick note:"]
SUFFIX = ["", "", "Please advise.", "This is blocking our rollout.", "Thanks.", "Any ETA?", "Happy to share details."]


def _accounts(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(25):
        rows.append((f"ACC-E{i:03d}", "Enterprise", int(rng.integers(150_000, 600_000))))
    for i in range(45):
        rows.append((f"ACC-M{i:03d}", "Mid-Market", int(rng.integers(40_000, 120_000))))
    for i in range(80):
        rows.append((f"ACC-S{i:03d}", "SMB", int(rng.integers(3_000, 25_000))))
    return pd.DataFrame(rows, columns=["account_id", "customer_segment", "annual_contract_value"])


def generate(n_target: int = 1000, seed: int = SEED, with_defects: bool = True) -> pd.DataFrame:
    """Return a feedback DataFrame incl. hidden `ground_truth_issue` label."""
    rng = np.random.default_rng(seed)
    acc = _accounts(rng)
    pairs: list[tuple[str, str]] = []
    for key, spec in ISSUES.items():
        pool = acc if spec["segment"] == "Mixed" else acc[acc.customer_segment == spec["segment"]]
        chosen = rng.choice(pool.account_id.to_numpy(), size=spec["n_accounts"], replace=False)
        for a in chosen:
            for _ in range(int(rng.integers(spec["per"][0], spec["per"][1] + 1))):
                pairs.append((a, key))
    while len(pairs) < n_target - 20:
        pairs.append(pairs[int(rng.integers(0, len(pairs)))])
    pairs = [pairs[i] for i in rng.permutation(len(pairs))][: n_target - 20]
    acc_idx = acc.set_index("account_id")
    rows = []
    for n, (a, key) in enumerate(pairs):
        s = ISSUES[key]
        a1, a2 = rng.choice(ANCHORS[key], size=2, replace=False)
        text = " ".join(
            t for t in [PREFIX[rng.integers(len(PREFIX))], s["templates"][rng.integers(len(s["templates"]))] + ".",
                        f"Affects our {a1} and {a2} workflow.", SUFFIX[rng.integers(len(SUFFIX))]] if t)
        rows.append(dict(
            ticket_id=f"T-{n:05d}", account_id=a, customer_segment=acc_idx.loc[a, "customer_segment"],
            annual_contract_value=int(acc_idx.loc[a, "annual_contract_value"]), product_area=s["area"],
            feedback_text=text, severity=int(rng.integers(s["sev"][0], s["sev"][1] + 1)),
            frequency=int(rng.integers(1, 5)),
            created_at=(pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))).date().isoformat(),
            status=str(rng.choice(["open", "open", "triaged", "in_progress"])),
            estimated_engineering_effort=round(float(s["effort"] * rng.uniform(0.8, 1.25)), 1),
            ground_truth_issue=key))
    vague = ["Great product overall", "Not sure how to describe this", "Thanks for the quick call yesterday",
             "Pricing question for next renewal", "General comment about the roadmap"]
    for k in range(20):
        a = acc.account_id.iloc[int(rng.integers(0, len(acc)))]
        rows.append(dict(
            ticket_id=f"T-N{k:04d}", account_id=a, customer_segment=acc_idx.loc[a, "customer_segment"],
            annual_contract_value=int(acc_idx.loc[a, "annual_contract_value"]), product_area="General",
            feedback_text=vague[k % len(vague)] + f" ({k})", severity=1, frequency=1,
            created_at="2026-06-01", status="open", estimated_engineering_effort=1.0, ground_truth_issue="noise"))
    df = pd.DataFrame(rows)
    if with_defects:
        i = rng.choice(len(df), size=12, replace=False)
        df.loc[i, "annual_contract_value"] = np.nan  # recoverable from same-account tickets
        df.loc[rng.choice(len(df), 8, replace=False), "severity"] = np.nan
        dup = df.sample(25, random_state=seed).copy()  # exact duplicate re-submissions
        dup["ticket_id"] = [f"T-D{k:04d}" for k in range(len(dup))]
        df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
