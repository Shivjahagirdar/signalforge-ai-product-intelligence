# Model evaluation report

Reproduce: `python scripts/evaluate.py` (writes `docs/research/evaluation-results.json`). All numbers below were
measured on 2026-11-17 in the build sandbox (Python 3.13, x86_64). Performance budgets were set **before** measuring.

## What changed and why
The previous build clustered full TF-IDF matrices with average-linkage at a fixed threshold. On real app reviews it
fragmented 1,500 reviews into **287 clusters** and took **19 s**. v2 projects word+char TF-IDF onto 100 LSA dimensions
and uses `method="auto"`: agglomerative first; if that yields one giant theme (>35% of records) or >40% of records in
clusters smaller than 3, it falls back to K-Means with k = clip(sqrt(n/2), 6, 30). The method used is shown in the UI.

## Labelled datasets
| Dataset | n | Method | Clusters | ARI | NMI | Purity | Pair precision | Pair recall | False-merge | False-split |
|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic templates | 1,025 | agglomerative | 12 | 0.999 | 0.992 | 1.000 | 1.000 | 0.998 | 0.000 | 0.002 |
| Held-out paraphrases + typos | 280 | agglomerative | 36 | 0.302 | 0.703 | 0.982 | 0.949 | 0.205 | 0.002 | 0.795 |

Reading: clusters that form are pure (few false merges), but lexical features **over-split true paraphrases** (false-split
0.80). The UI mitigates with merge suggestions + human merge/split; the real fix is a sentence-embedding backend, which is
implemented as optional but not enabled by default (memory budget, unverifiable offline).

## Real, unlabelled data (F-Droid Google Play reviews, 1-3★ sample)
| n | Method | Clusters | Largest share | Silhouette (cos) | Stability ARI (3x 80% subsamples) | Topic diversity | Time |
|---|---|---|---|---|---|---|---|
| 1,500 | kmeans(k=27) | 27 | 0.095 | 0.119 | 0.135 mean / 0.127 min | 0.533 | 1.05 s |

Reading: themes are usable and balanced, but **partition stability is low** on diverse multi-app review text. Treat
real-data themes as triage starting points, which is why every theme exposes evidence sufficiency and human controls.

## Not run
* Bitext labelled intents - host unreachable from the build sandbox (runs in CI / deployed app where reachable).
* Human-review agreement - requires reviewers; planned measurement (see experimentation plan).

## Robustness (automated tests, pass/fail)
Order invariance, punctuation noise, duplicate templates collapse, typos/case, empty/garbage input, prompt-injection
text cannot change ranking - see `tests/property/test_invariants.py` and `tests/unit`.

## Performance (budget set before benchmarking)
| Workload | Measured | Budget |
|---|---|---|
| ingest + cluster + rank, 976 records | 0.54 s | 5 s |
| ingest + cluster + rank, 4,857 records | 3.15 s | 30 s |
| peak RSS of full evaluation run | 511 MB | 1,536 MB |
