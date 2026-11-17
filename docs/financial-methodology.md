# Financial & prioritisation methodology

Every value is labelled as one of **Observed fact · User assumption · Model estimate · Unknown**.

```
unique ACV   = Σ over UNIQUE accounts of max(ACV)        (duplicates can't inflate revenue - property-tested)
ACV source   = observed | imputed_segment_median | user_assumption | missing   (missing is never 0)
exposure     = unique ACV × clip(base_exposure_fraction × severity/3, 0, 1)
recoverable  = exposure × recovery_probability
investment   = effort_weeks × eng_cost_per_week + infra + support
net value    = recoverable − investment            ROI = net / investment
payback      = investment / (recoverable/12) months  cost of delay = recoverable/12 per month
portfolio    = Σ accounts once at their max exposure fraction across themes
priority     = net × evidence_confidence × strategic_alignment / effort
```
Datasets with no revenue (public reviews, GitHub issues) show a red "NO REVENUE DATA" badge; $ values then come from an
explicit "assumed value per affected account" input.

**Frameworks** (Roadmap Optimizer): Risk-adjusted ROI, RICE, ICE, WSJF, Cost of Delay (CD3), Value vs Effort, Custom
weighted. Formulas are displayed in-app. Custom weighting uses *revenue exposure* and *non-revenue reach* as separate
inputs so the same accounts aren't rewarded twice. Weight sensitivity reports which weights change the top-3 at ±50%.

**Optimiser**: 0/1 integer program (SciPy `milp`/HiGHS) maximising modeled value subject to capacity, optional budget,
dependencies, mandatory items and a confidence floor; reports binding constraints, deferred items with reasons, an
alternative plan without the top-value item, and the sort-and-fill baseline for comparison.

**Uncertainty**: tornado (one-at-a-time ±30%) and seeded Monte Carlo over triangular assumption ranges. Outputs are
model-implied, conditional on assumptions - never empirical probabilities of churn.
