# PRD - SignalForge AI (MVP)
**Status:** MVP built; commercial claims are hypotheses. All demo data is synthetic.

## Problem
B2B SaaS product teams hold fragmented feedback (tickets, calls, CRM notes) but prioritise by volume and loudness. Duplicate complaints inflate priority, and revenue linkage is manual. A small integration defect hitting few high-ACV accounts is routinely outranked by a popular cosmetic request from many low-ACV accounts.

## Target customer
Senior/Principal PMs and Product Ops at B2B SaaS companies (buyer: VP Product/CPO). See `customer-personas.md`.

## Jobs-to-be-Done
When I must prioritise competing investments from fragmented signals, help me quantify the most important opportunities so I can make defensible roadmap decisions without weeks of manual analysis.

## Solution (MVP journey)
Upload CSV -> validate -> cluster themes -> unique-account revenue exposure -> effort/ROI -> risk-adjusted rank -> explain -> export PRD/memo.

## Functional requirements and acceptance criteria
| ID | Requirement | Acceptance (verified by test) |
|---|---|---|
| F1 | CSV ingestion with schema validation, size/type limits | `tests/unit/test_ingestion.py` |
| F2 | Duplicate handling (exact removed, near-duplicates flagged) | ingestion + `test_near_duplicates_and_audit_store` |
| F3 | Semantic clustering with TF-IDF fallback, ground-truth benchmark | ARI > 0.9 on synthetic (`test_clustering_quality_vs_ground_truth`) |
| F4 | Revenue-at-Risk model, unique-account ACV, no double counting | `test_economics.py` |
| F5 | Risk-adjusted ranking, RICE and severity x frequency baseline | `test_rankings_reproducible_and_change_with_assumptions` |
| F6 | Human overrides (rename, merge, exclude evidence, rank override w/ rationale), persisted + audited | `test_overrides_*`, `test_merge_*` |
| F7 | Scenario lab: bear/base/bull, tornado, Monte Carlo | economics tests |
| F8 | PRD + executive memo export, every claim traceable to tickets/assumptions | `test_prd_and_memo_traceable` |

## Scope / non-goals (deferred)
Salesforce/Zendesk integrations, multi-tenant auth/SSO, streaming, billing, churn prediction, autonomous roadmap execution. Deferred to keep time-to-first-insight under one upload.

## Success metrics (hypotheses, unmeasured)
North Star: evidence-backed product decisions per active product team per month. Supporting: time to first insight, % recommendations with ticket evidence (100% by construction), ranking stability across scenarios, PRD creation time.

## Risks
Exposure is scenario math, not churn prediction; TF-IDF clusters on real, messy text are unproven; willingness to pay unvalidated.
