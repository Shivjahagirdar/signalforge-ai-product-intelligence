# BUILD_STATUS (2026-11-17)

**Status: PARTIALLY COMPLETE.** Core product, tests, CI, Docker and public deployment are done and verified. A few
acceptance criteria are only partly met (listed below).

## Verified, with evidence
| Item | Evidence |
|---|---|
| Live public app | https://signalforge-decision-os.streamlit.app (Streamlit Community Cloud, Python 3.12) |
| Live journey | Real browser on the live URL: demo load → Roadmap capacity 8 pw recalculated (2 items, 90% used) → override with rationale → audit trail entry → SIGNAL "Challenge this roadmap." grounded answer → switched to real F-Droid reviews (1,500 rows, 27 themes, NO-REVENUE badge) |
| Session isolation (live) | Second independent session: default capacity/plan, no session-A override text or audit event present |
| Tests | 110 unit/property/integration/security tests + 2 Playwright browser journeys pass locally and in CI |
| CI | GitHub Actions green on `bbc80b1` (run 44176412865): ruff, pytest, E2E, evaluation with live public data, pip-audit, Docker build + container health check |
| Evaluation | `docs/model-evaluation-report.md` (synthetic, held-out, Bitext real labelled, F-Droid real unlabelled, latency vs pre-set budgets) |
| Security | `docs/security-threat-model.md`; pip-audit: no known vulnerabilities |

## Partly met / not done
* Paraphrase recall (held-out false-split 0.80) and real multi-app review stability (ARI 0.13) are weak; documented, mitigated with human merge/split; sentence-embedding backend optional, not default.
* Durable persistence: none by design for the public demo (session memory + exports).
* PDF export not built (MD + DOCX are); GitHub import shares a 60 req/h rate limit on the hosted app.
* Full 25-step journey on the live URL was verified for the core steps above; downloads and malformed/empty uploads were verified by the automated E2E against a local server, not repeated on the live URL.
* Human-review agreement and product metrics (time-to-insight, correction rate) are planned measurements, not results.
