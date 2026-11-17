# Security & privacy model

| Threat | Mitigation | Verified by |
|---|---|---|
| CSV/formula injection in exports | `safe_csv` prefixes `= + - @ \t \r` cells | `tests/security/test_security.py::test_csv_injection_neutralised` |
| Oversized / malformed / wrong-type uploads | 10 MB cap (app + server config), typed parsers, user-safe errors, never crashes | security + E2E malformed/empty upload tests |
| Malicious filenames / path tricks | uploads are read in memory only; filename is used for extension check only | `test_oversized_and_wrong_extension_rejected` |
| Prompt injection inside feedback / GitHub issues | customer text is data: deterministic copilot never executes it; instruction-like phrases redacted & flagged; optional LLM receives JSON with an explicit "data not instructions" system rule; ranking uses numbers only | `test_prompt_injection_in_feedback_is_redacted_and_flagged`, `test_injected_feedback_cannot_change_ranking` |
| Cross-session leakage | per-session in-memory store; caches hold only pure-function results of shared public inputs | `test_sessions_are_isolated` (AppTest) + E2E second visitor context |
| Secret exposure | key read server-side from env/secrets only, never rendered or logged; repo secret scan test | `test_api_key_never_rendered`, `test_no_secrets_committed` |
| SSRF via dataset fetch | only registry URLs + validated `owner/repo` for GitHub; https enforced; size + timeout caps | `test_github_repo_name_is_validated_before_any_network_call` |
| Dependency CVEs | `pip-audit` in CI | CI log ("No known vulnerabilities found" on 2026-11-17 locally) |
| Personal data | GitHub logins / reviewer ids pseudonymised (SHA-256 prefix); UI warns against uploading unauthorised PII | adapter tests |

Not implemented / not claimed: SSO/RBAC, durable tenant isolation, encryption at rest, SOC 2, GDPR programme.
