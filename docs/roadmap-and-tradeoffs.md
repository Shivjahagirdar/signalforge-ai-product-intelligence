# Roadmap and tradeoffs
**Now (built):** CSV upload -> clustering -> unique-account economics -> ranking -> overrides -> PRD/memo.
**Next (validate):** 5 discovery interviews; pilot on one real export; sentence-transformer embeddings (needs HF access; unverified in the build sandbox); merge-suggestion acceptance tracking.
**Later:** Zendesk/Salesforce import, multi-tenant auth, outcome measurement after launch.
**Tradeoffs:** deterministic offline TF-IDF vs semantic quality (we chose reproducibility, accepted over-splitting, mitigated by merge suggestions); simple exposure formula vs predictive churn (chose transparency); in-memory per-visitor state vs persistence (chose privacy).
**Deliberately deferred:** SSO, billing, streaming, autonomous execution, churn ML.
