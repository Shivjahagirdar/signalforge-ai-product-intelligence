# Dataset provenance registry

Source of truth in code: `src/signalforge/datasets.py::REGISTRY`. No third-party data is committed to this repository;
public datasets are fetched from their original host at runtime and cached in memory for one hour.

| Key | Kind | Source | License / terms | Size | Used for | Limitations |
|---|---|---|---|---|---|---|
| `synthetic` | Synthetic SaaS feedback | `src/signalforge/synthetic.py` (seeded) | Project MIT | 1,000 records, 119 accounts | Demo of economics end-to-end; labelled clustering regression test | **Fictional** accounts, ACV, effort. Template text is easier than real text. |
| `fdroid_reviews` | Real public app reviews | [sealuzh/user_quality](https://github.com/sealuzh/user_quality) `csv_files/reviews.csv` (Grano et al., Univ. of Zurich) | **No license file** in the repo - analysed at runtime, not redistributed | 288,065 reviews / 395 apps (we read the first ~8 MB, keep 1-3★, sample 1,500) | Real-text clustering, stability, latency; second-dataset user journey | End-user reviews of many apps, not one B2B product; no accounts/revenue; prefix sample, not random; star rating is only a severity proxy. |
| `bitext_support` | Real+NLG customer-service requests, 27 labelled intents | [bitext/customer-support-llm-chatbot-training-dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset) (data file on Hugging Face) | **CDLA-Sharing-1.0** (license file verified 2026-11-17) | 26,872 rows | Labelled clustering metrics (ARI/NMI) on non-template text | Partly machine-generated; B2C service intents. **Not evaluated in this build**: Hugging Face was unreachable from the build sandbox; the app fetches it at runtime where reachable. |
| `github_issues` | Real developer/user issues | GitHub REST API `GET /repos/{owner}/{repo}/issues` | GitHub Terms; content owned by authors - analysed, not stored | up to 300 issues per import | Live import of any public repo's issues | Unauthenticated rate limit 60 req/h per server IP; PRs excluded; reactions are a demand proxy; no revenue. |

Rules applied: provenance shown in-app (sidebar "Data provenance"), real vs synthetic badge on every page, missing ACV
never becomes zero, usernames pseudonymised, no scraping of authenticated content. Fewer than three *redistributable*
real datasets with verified licenses were found that were reachable from the build environment; this is recorded here
rather than substituting invented data.
