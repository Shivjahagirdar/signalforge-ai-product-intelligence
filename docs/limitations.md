# Limitations & known issues (honest list)

1. **Paraphrase recall is low by default** (lexical backend: held-out false-split 0.80). Opt-in *Semantic matching (beta)*
   raises held-out ARI 0.30 -> 0.76 but lowers tightly-templated data (synthetic 0.999 -> 0.37): see the measured grid in
   `docs/model-evaluation-report.md`. No setting wins everywhere; human merge/split remains the safety net.
2. **Real-data cluster stability is low** (subsample ARI ≈ 0.13 on multi-app reviews). Themes are triage aids.
3. **No server-side storage, by design.** Overrides/audit live in the visitor's session memory. Persistence is
   client-held: *Save analysis (.json)* downloads settings, assumptions, overrides, constraints and the audit trail (never raw
   uploads) and *Load saved analysis* restores them in any later session. Cluster ids depend on data + settings, so a
   snapshot restored onto different data shows a warning.
4. **Public data has no revenue.** $ figures on public datasets come from a user assumption and are badged as such.
5. Bitext is evaluated in CI only (unreachable from the build sandbox); GitHub import is rate-limited (60/h per server IP, shared by all visitors of the hosted demo).
6. **Evidence confidence is a heuristic**, not a calibrated probability; Monte Carlo is conditional on assumed ranges.
7. **Signal tags are keyword rules** (transparent, but miss implicit phrasing); "churn language" is not churn prediction.
8. **PDF export** is built-in (fpdf2, core fonts): non-Latin text is transliterated/replaced, and tables are rendered as plain text rows. DOCX and Markdown are richer.
9. Single-process Streamlit app; not load-tested beyond the two-session isolation checks.
