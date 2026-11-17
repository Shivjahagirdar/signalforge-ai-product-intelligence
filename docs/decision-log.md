# Decision log
1. TF-IDF default over transformer: deterministic, offline, zero download; trade-off = weaker semantics on paraphrase.
2. Agglomerative (average, cosine, threshold 0.9) over k-means: no need to pick k; threshold tuned on synthetic data only (**risk: overfit to synthetic**).
3. ACV max per account, never summed per ticket: prevents duplicate inflation.
4. Missing ACV: fill from same account, else segment median (flagged) or exclude.
5. Exposure presented as scenario, never "churn".
6. Synthetic templates include anchor terms per issue; benchmark scores therefore show the pipeline works, **not** real-world accuracy.
7. Competitive research deferred to backlog rather than asserting unverified claims.
