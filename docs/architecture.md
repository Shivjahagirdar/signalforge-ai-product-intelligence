# Architecture
```mermaid
flowchart LR
  CSV[CSV / synthetic] --> ING[ingestion.py<br/>validate, dedupe]
  ING --> INT[intelligence.py<br/>TF-IDF/transformer, clustering]
  INT --> PIPE[pipeline.py<br/>opportunities, rank]
  ECO[economics.py<br/>exposure, ROI, Monte Carlo] --> PIPE
  PIPE --> EXP[exports.py<br/>PRD, memo]
  PIPE --> UI[app.py Streamlit]
  AUD[audit.py<br/>SQLite overrides + events] <--> UI
```
Domain logic is UI-free and unit-tested. SQLite for local MVP; PostgreSQL is the upgrade path (swap `AuditStore`). Embeddings: TF-IDF by default (deterministic, offline); `sentence-transformers` is used when requested and available. Deviation from the original spec: flat module layout (one file per service) instead of nested packages, since each service is small.
