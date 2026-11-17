# Technical & data architecture

```mermaid
flowchart LR
  subgraph Sources
    SYN[Synthetic demo<br/>labelled fictional]
    PUB[Public datasets<br/>fetched at runtime]
    GH[GitHub issues<br/>REST API]
    UP[Upload CSV/JSON/JSONL/XLSX]
  end
  SYN & PUB & GH --> REG[datasets.py<br/>registry, adapters, provenance]
  UP --> MAP[datasets.coerce_to_schema<br/>schema detection + column mapping]
  REG & MAP --> ING[ingestion.py<br/>validate, dedupe, quality]
  ING --> INT[intelligence.py<br/>TF-IDF word+char -> LSA<br/>agglomerative / K-Means auto]
  INT --> SIG[signals.py<br/>rule tags, trends, explanations]
  SIG --> PIPE[pipeline.py<br/>opportunities, evidence sufficiency, rank]
  ECO[economics.py<br/>exposure, ROI, payback, CoD, Monte Carlo] --> PIPE
  PIPE --> PRI[prioritization.py<br/>RICE ICE WSJF CoD V/E custom + sensitivity]
  PIPE --> OPT[optimizer.py<br/>0/1 MILP: capacity, budget, deps, mandatory]
  PIPE & PRI & OPT --> COP[copilot.py SIGNAL<br/>intent router -> analytical tools]
  COP -. optional, server-side key .-> LLM[(LLM narration)]
  PIPE & OPT --> EXP[exports.py<br/>memo, PRD, assessment, evidence, roadmap, experiments, log -> MD/DOCX]
  AUD[audit.py<br/>in-memory SQLite per session] <--> UI[app.py Streamlit, 6 pages + SIGNAL panel]
  PIPE & PRI & OPT & COP & EXP --> UI
```

**Boundaries.** Every module under `src/signalforge` is UI-free and unit-tested; `app.py` only wires widgets to functions.
**State.** Per-visitor `st.session_state` holds an in-memory SQLite `AuditStore` (overrides + audit trail). Nothing is
written to disk or shared between sessions; `st.cache_data` caches only pure functions of public/synthetic inputs and
clustering results keyed by the input frame (no per-user state in caches). This is *not* durable multi-user storage:
the upgrade path is swapping `AuditStore` for PostgreSQL with tenant keys.
**Determinism.** LSA uses `random_state=0`, K-Means `random_state=0`, Monte Carlo seeds are user-visible.
**Model choice.** `sentence-transformers` remains an optional backend (`prefer_transformer=True`); it is not the default
because model download + torch exceed the free hosting memory budget and could not be verified offline. See
`docs/model-evaluation-report.md` for the measured cost of that decision (paraphrase recall).

## Data model (per record)
`ticket_id, account_id, customer_segment, annual_contract_value, product_area, feedback_text, severity, frequency,
created_at, status, estimated_engineering_effort` (+ optional `source`, `ground_truth_issue`). Public adapters fill
`annual_contract_value` and `estimated_engineering_effort` with **missing**, never zero.
