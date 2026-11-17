# Data model
Implemented as pandas frames and dicts (MVP), with SQLite for audit/overrides.
- **Feedback** (ticket_id PK, account_id, customer_segment, annual_contract_value, product_area, feedback_text, severity 1-5, frequency, created_at, status, estimated_engineering_effort)
- **Cluster** (cluster_id, cluster_label, cohesion) 1-N Feedback
- **Opportunity** (cluster_id -> ticket_ids, unique accounts, acv_sum_unique, avg_severity, effort_weeks, confidence)
- **Assumptions** (exposure fraction, recovery probability, costs, horizon, strategic alignment per cluster)
- **PrioritizationResult** (priority, rice, baseline rank, model rank, final rank, override rationale)
- **Overrides** (rename, merge, irrelevant tickets, rank) and **AuditEvent** (ts, kind, detail) in SQLite, in-memory per visitor session.
Traceability: ticket -> cluster -> opportunity -> ranking row -> PRD "Evidence trail" lists ticket IDs.
Deferred: formal FeatureProposal/Experiment/DecisionEvidence tables and schema versioning.
