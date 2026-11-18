# BUILD_STATUS (2026-11-18)

**Status: COMPLETE for the Release 1 scope:** the P0 evidence-to-outcome vertical slice, built, tested and deployed.
P1/P2 items and measurement limits are listed below. They are not hidden as "done".

## Verified, with evidence
| Item | Evidence |
|---|---|
| Live public app | https://signalforge-decision-os.streamlit.app (Streamlit Community Cloud) |
| Live E2E | `live-e2e` workflow run 44402378436: **3/3 Playwright journeys passed against the live URL**. Covers full decision journey, downloads (MD/DOCX/PDF), save/restore, malformed + empty uploads, isolated second session, and the closed loop (decision → ticket + PRD → release → measured outcome) |
| Tests | 139 unit/property/integration/security tests + 3 browser journeys pass locally and in CI |
| CI | `ci` workflow green on `c0f1065` (run 44399991423): ruff, pytest, E2E, evaluation, pip-audit, Docker build + health check |
| Closed loop (P0 #1) | GitHub Issues ingest → validate/dedupe → evidence-linked themes → explicit-assumption prioritisation → capacity/dependency-feasible MILP roadmap → human approval → ticket + PRD → decision record → outcome vs baseline/target → forecast-vs-actual retrospective |
| Data model & persistence | `store.py`: SQLite workspaces, decisions, tickets, releases, outcomes, audit log; stable IDs, provenance, timestamps; workspace resume by key; cross-session isolation tested |
| GitHub integration | `github_sync.py`: public repos without credentials, pagination, retries/backoff, rate-limit handling, incremental `since` sync |
| Telemetry | Documented event schema (`telemetry.py`); upload your own CSV, or use a seeded simulated sample that is **labelled SIMULATED** everywhere it appears. No telemetry is inferred from issue text |
| Evidence quality | Completeness, source mix, freshness, contradiction and sample-bias checks shown per opportunity |
| UX | Bright gradient redesign, 7-step progress bar, page blurbs, "Next →" buttons, icons, friendly empty/error states |
| Evaluation / security | `docs/model-evaluation-report.md`, `docs/security-threat-model.md`; pip-audit clean |

## Known limits (honest, not blockers for Release 1)
* Paraphrase recall is still weak with the default lexical backend (held-out false-split 0.80). Semantic matching (Model2Vec) is opt-in, and human merge/split mitigates the gap.
* Hosted-demo SQLite persists across sessions and reloads but **not across a platform redeploy**. Use Postgres for real teams (Release 2).
* GitHub import shares the unauthenticated 60 requests/hour limit on the hosted app.
* Human-review agreement and real product metrics need real users, so they are planned measurements, not results.
* Release 2/3 (Jira/Linear, SSO/RBAC multi-tenancy, experimentation, predictive models) are not built.
