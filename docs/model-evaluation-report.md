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
| **Bitext customer-service (real/NLG, 27 intents)** - measured in CI run 44176412865 | 1,500 | agglomerative | 15 | **0.638** | **0.843** | 0.689 | 0.537 | 0.863 | 0.043 | 0.137 |

Reading: on templates and held-out paraphrases, clusters that form are pure but lexical features **over-split true
paraphrases** (false-split 0.80). On Bitext's 27 real intents the opposite trade-off appears: 15 themes, related intents
merged (e.g. order/refund families; purity 0.69), with high recall (0.86) and **stability ARI 0.93** across subsamples. The UI mitigates with merge suggestions + human merge/split; the real fix is a sentence-embedding backend, which is
implemented as optional but not enabled by default (memory budget, unverifiable offline).

## Semantic (Model2Vec) matching: measured trade-off (CI run 44179711099, branch `tune/semantic`)
Hybrid embedding = w x Model2Vec(potion-base-8M) + (1-w) x lexical LSA. Grid over w x cluster-breadth, ARI per labelled set
(`scripts/tune_semantic.py`, reproducible from the `semantic-eval` workflow):

| w | breadth | synthetic ARI | held-out paraphrase ARI | Bitext ARI |
|---|---|---|---|---|
| 0.0 (lexical default) | 0.90 | **0.999** (at 0.9 in the app default flow) | 0.302 | 0.638 |
| 0.85 | 0.80 | 0.368 | **0.759** | 0.644 |
| 0.85 | 0.90 | 0.368 | 0.631 | 0.644 |
| 0.70 | 0.70 | 0.999 | 0.335 | 0.518 |

Finding: **no single setting wins on every dataset.** Semantic matching more than doubles held-out paraphrase ARI
(0.30 -> 0.76) and holds Bitext, but loosens tightly-templated data (synthetic 0.999 -> 0.37 at the same breadth).
Decision: the deterministic lexical backend stays the default; semantic matching is an opt-in sidebar toggle
("Semantic matching (beta)") that applies the measured breadth 0.80 and falls back to lexical, with a visible warning, if
the model cannot be downloaded. The 0.759 figure is the best cell of a 20-cell grid evaluated on the same data it was
selected on, so treat it as optimistic.

**Negative result:** an unsupervised "adaptive cut" (choose the dendrogram threshold by best silhouette) was tried on the
same grid and did **not** help (held-out ARI 0.25 at every w; silhouette always picks the lowest threshold and fragments the
data). It was reverted; the code is in history (commit 7896b99).

## Real, unlabelled data (F-Droid Google Play reviews, 1-3★ sample)
| n | Method | Clusters | Largest share | Silhouette (cos) | Stability ARI (3x 80% subsamples) | Topic diversity | Time |
|---|---|---|---|---|---|---|---|
| 1,500 | kmeans(k=27) | 27 | 0.095 | 0.119 | 0.135 mean / 0.127 min | 0.533 | 1.05 s |

Reading: themes are usable and balanced, but **partition stability is low** on diverse multi-app review text. Treat
real-data themes as triage starting points, which is why every theme exposes evidence sufficiency and human controls.

Bitext unlabelled view (CI): silhouette 0.276, largest theme 16.8%, topic diversity 0.76, 0.31 s.

## Not run
* Bitext was unreachable from the build sandbox; its numbers above come from the GitHub Actions run, which fetches it live.
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
