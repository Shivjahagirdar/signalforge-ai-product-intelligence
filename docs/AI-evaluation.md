# AI evaluation - synthetic benchmark vs held-out wording
Reproduce: `python scripts/benchmark.py` (seeded). **None of this is validated real-world performance.**

| Data | Features | Clusters | ARI | Pairwise precision | Pairwise recall |
|---|---|---|---|---|---|
| Synthetic templates (996 rows) | word+char TF-IDF, threshold 0.9 | 12 | 0.999 | 1.000 | 0.998 |
| Held-out paraphrases/typos/caps (280 rows, wording unseen) | same | 36 | ~0.30 | ~0.95 | ~0.21 |

**Reading this honestly:** the synthetic score is inflated - those templates share keywords by construction. On unseen wording, clusters that form are mostly correct (high precision) but one issue is split into many clusters (low recall), because TF-IDF/char n-grams match words, not meaning. Sweeps up to threshold 0.96 reach ARI ~0.45 at precision ~0.79.
**Mitigations built:** adjustable threshold, "Suggested merges" panel, human merge/rename/exclude controls.
**Not verified here:** sentence-transformers (the build sandbox blocks huggingface.co, so the transformer path is implemented with a tested fallback but never run). Expect better recall with real embeddings - unmeasured.
**Not done:** multi-seed stability, noise-level sweeps, human-correction effort, real customer text.
