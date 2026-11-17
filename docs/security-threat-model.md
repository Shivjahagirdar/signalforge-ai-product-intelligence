# Security and threat model (MVP)
Implemented: 10 MB / .csv-only limit, schema validation, formula-injection-safe CSV export, no secrets in repo, safe error messages, local audit log, reset button, PII warning in UI.
Threats: malicious CSV (formula injection - mitigated; huge files - limited), PII upload (warned, no retention beyond local SQLite until reset), dependency CVEs (not scanned in MVP - add pip-audit).
Not implemented / not claimed: SSO/RBAC, tenant isolation, encryption at rest, SOC 2, GDPR/CCPA programme. No compliance certification is claimed.
