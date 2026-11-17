# Limitations & known issues (honest list)

1. **Paraphrase recall is low** without sentence embeddings (held-out false-split rate 0.80). Mitigated by merge
   suggestions and human merge/split; fix = enable a sentence-transformer backend on hosting with more memory.
2. **Real-data cluster stability is low** (subsample ARI ≈ 0.13 on multi-app reviews). Themes are triage aids.
3. **No durable storage.** Overrides/audit live in the visitor's session memory and vanish when the tab closes.
   Exports (decision log, audit CSV) are the persistence mechanism in the demo.
4. **Public data has no revenue.** $ figures on public datasets come from a user assumption and are badged as such.
5. Bitext is evaluated in CI only (unreachable from the build sandbox); GitHub import is rate-limited (60/h per server IP, shared by all visitors of the hosted demo).
6. **Evidence confidence is a heuristic**, not a calibrated probability; Monte Carlo is conditional on assumed ranges.
7. **Signal tags are keyword rules** (transparent, but miss implicit phrasing); "churn language" is not churn prediction.
8. **PDF export** is not built in; DOCX and Markdown are (DOCX → PDF via Word/Docs).
9. Single-process Streamlit app; not load-tested beyond the two-session isolation checks.
