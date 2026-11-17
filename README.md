# SignalForge AI - From Customer Pain to Revenue-Optimized Product Decisions

> **Personal portfolio project.** Not built at or for any employer; all data is **synthetic**. Commercial figures are modelled hypotheses, not results.

Product teams collect thousands of tickets but rank work by volume. SignalForge turns unstructured feedback into **evidence-linked, financially prioritised roadmap decisions** using unique-account revenue exposure, effort, evidence quality and scenario analysis - and shows its reasoning.

## Demo scenario (reproduced by the code, not hard-coded)
In the synthetic company (150 accounts, ~1,000 records) a Salesforce-sync defect affecting **12 high-ACV accounts** ranks **#1**, while a popular dark-mode request from many small accounts ranks lower. A naive severity x frequency baseline ranks them the other way. Change the assumptions in the sidebar and rankings update live.

## Workflow
Upload CSV -> validate/dedupe -> cluster themes -> unique-account exposure -> ROI/priority -> explain -> override -> export PRD.

## Features
- CSV ingestion: schema validation, size/type limits, missing-value handling, duplicate detection, formula-injection-safe export
- TF-IDF clustering (deterministic, offline) with optional sentence-transformers; benchmark vs ground truth
- Revenue-at-Risk model: exposure, recoverable value, investment, net value, ROI, payback, cost of delay; no account double counting
- RICE and severity x frequency baselines side by side; "why A outranks B" explanations
- Human overrides (rename, merge, exclude evidence, rank override with rationale) persisted in SQLite with audit trail
- Scenario Lab: bear/base/bull, tornado, Monte Carlo (model-implied, not empirical)
- Downloadable PRD and executive memo with ticket-level evidence

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
Docker (not yet build-verified): `docker build -t signalforge . && docker run -p 8501:8501 signalforge`

## Tests and benchmark
```bash
pip install -r requirements-dev.txt
pytest -q                      # unit, integration, held-out wording, interactive app flows
python scripts/benchmark.py    # synthetic clustering benchmark
```
Last local run: all tests passed, ruff clean, pip-audit found no known vulnerabilities in `requirements.txt`.

| Clustering benchmark | ARI | Precision | Recall |
|---|---|---|---|
| Synthetic templates (inflated: shared keywords) | 0.999 | 1.000 | 0.998 |
| **Held-out paraphrases / typos (unseen wording)** | **0.30** | 0.95 | 0.21 |

On unseen wording, TF-IDF keeps clusters pure but **over-splits** one issue into several. The app offers a threshold slider, "Suggested merges", and manual merge/rename controls. Sentence-transformers is supported but could not be run in the build sandbox (see `docs/AI-evaluation.md`). Real-customer accuracy is unmeasured.

## Docs
[PRD](docs/PRD.md) - [Competitive gap analysis (hypotheses)](docs/competitive-gap-analysis.md) - [Personas](docs/customer-personas.md) - [Opportunity tree](docs/opportunity-solution-tree.md) - [Prioritisation](docs/prioritization-framework.md) - [Architecture](docs/architecture.md) - [Decision log](docs/decision-log.md) - [Unit economics & GTM](docs/go-to-market.md) - [Threat model](docs/security-threat-model.md) - [AI evaluation](docs/AI-evaluation.md) - [Executive memo](docs/executive-investment-memo.md) - [Case study](docs/principal-pm-case-study.md)

## Limitations
Not proven on real data; clustering recall on paraphrases is low (see above); Docker image and GitHub Actions not yet run; no screenshots yet; competitor claims unverified (research backlog); no auth/multi-tenancy; exposure is a scenario estimate, not churn prediction; no live demo URL yet.

## Roadmap
Validate with 5 interviews -> real-data pilot -> Zendesk/Salesforce import -> multi-tenant auth.
