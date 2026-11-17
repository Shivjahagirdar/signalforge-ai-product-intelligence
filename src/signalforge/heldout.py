"""Held-out paraphrase set for evaluation: wording NOT used by the training templates in synthetic.py,
including typos, abbreviations and terse/verbose styles. Used only for evaluation and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

HELDOUT: dict[str, list[str]] = {
    "integration_sync": [
        "our Salesforce data hasnt synced since monday and reps are working from stale records",
        "contacts pushed from your platform never show up in SFDC anymore",
        "the HubSpot/CRM connection keeps disconnecting and we have to re-authenticate every few days",
        "records are out of sync between the CRM and your app, sales ops is furious",
        "integration job fails w/ 401 after the token expires, nothing gets updated",
        "bi-directional sync overwrote our opportunity amounts in the CRM",
    ],
    "dark_mode": [
        "can we get a night mode? staring at the white screen late at night is painful",
        "Would be great if the UI had a darker color scheme option",
        "pls add dark theme!! my eyes hurt",
        "the app looks very bright and dated, a dark skin would be a nice touch",
        "any plans for a low-light display setting?",
        "requesting theming support, esp. dark colors, for the web console",
    ],
    "report_export": [
        "downloading the quarterly numbers as a spreadsheet takes forever and then errors out",
        "the analytics dashboard export hangs at 90% for large datasets",
        "generating the monthly PDF summary is painfully slow, often times out",
        "exports of more than 50k rows just spin and never finish",
        "scheduled email of the usage report arrives blank or very late",
        "data extract to xlsx fails with a timeout error on big accounts",
    ],
    "api_rate_limit": [
        "we're getting throttled constantly when our integration calls the endpoints in bulk",
        "429 Too Many Requests whenever we run our nightly batch against your REST interface",
        "request quota is far too restrictive for a customer our size",
        "your API cuts us off after a few hundred calls per minute, please raise the cap",
        "bulk loading via the public endpoints gets blocked by throttling after seconds",
        "need a higher requests-per-minute allowance for our automation",
    ],
    "onboarding": [
        "new teammates cant figure out how to get started, the first-run experience is baffling",
        "the welcome flow skips steps and leaves our admins lost",
        "took us three weeks to get configured because the getting-started docs are unclear",
        "first login was overwhelming, no guidance about what to set up first",
        "our trainees keep asking support how to do initial setup, it isnt intuitive",
        "implementation checklist is confusing for first time admins",
    ],
    "audit_logs": [
        "security review needs a record of who changed permissions and when",
        "we cannot export an activity history of admin actions for our auditors",
        "ISO / SOC2 evidence request: tamper-proof history of configuration changes",
        "no way to see which user deleted a record, we need an access trail",
        "compliance officer wants retention of user activity events for 7 years",
        "please provide an event log of privileged operations",
    ],
    "mobile_crash": [
        "iPhone version quits unexpectedly right after opening",
        "the android build force-closes whenever I attach a picture",
        "mobile client is unstable - freezes then dies on the home screen",
        "since the update our drivers' phones crash the moment they log in",
        "tablet app keeps shutting down when syncing offline data",
        "app closes by itself after the splash screen on several devices",
    ],
}


def generate_heldout(n_per_issue: int = 40, seed: int = 123) -> pd.DataFrame:
    """Held-out records (labels in `ground_truth_issue`) with random typos/case noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, texts in HELDOUT.items():
        for i in range(n_per_issue):
            t = texts[int(rng.integers(len(texts)))]
            if rng.random() < 0.3 and len(t) > 10:  # typo: drop a char
                j = int(rng.integers(3, len(t) - 2))
                t = t[:j] + t[j + 1:]
            if rng.random() < 0.15:
                t = t.upper()
            rows.append({"ticket_id": f"H-{key[:3]}-{i:03d}", "feedback_text": t, "ground_truth_issue": key})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
