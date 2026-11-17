# Prioritisation framework
- Exposure fraction = base fraction x clip(avg severity / 3, 0.2, 1.67)  *(assumption; severity scales exposure)*
- Potential Revenue Exposure = sum of **unique-account** ACV x exposure fraction (scenario, not churn)
- Expected Recoverable Value = Exposure x recovery probability
- Total Investment = effort (person-weeks) x cost/week + annual infra + annual support
- Net Opportunity Value = Recoverable - Total Investment; ROI = Net / Investment
- Risk-adjusted priority = Net x evidence confidence x strategic alignment / effort weeks
- Evidence confidence = 0.5 x cluster cohesion + 0.3 x min(1, accounts/5) + 0.2 x min(1, tickets/20) - a heuristic, **not** calibrated.
- Portfolio exposure counts each account once at its maximum fraction (no double counting).
Alternatives shown side by side: RICE and severity x frequency baseline.
