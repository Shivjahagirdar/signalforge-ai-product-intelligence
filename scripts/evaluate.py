"""Reproducible evaluation: clustering quality, stability, robustness and latency on synthetic + real public data.

Usage: python scripts/evaluate.py [--out docs/research/evaluation-results.json]
Performance budgets were fixed BEFORE benchmarking (docs/model-evaluation-report.md):
  cluster+rank 1,000 records <= 5 s; 5,000 records <= 30 s; peak RSS <= 1.5 GB (Streamlit Community Cloud limit ~2.7 GB).
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score  # noqa: E402

from signalforge.datasets import load_public  # noqa: E402
from signalforge.economics import Assumptions  # noqa: E402
from signalforge.heldout import generate_heldout  # noqa: E402
from signalforge.ingestion import validate  # noqa: E402
from signalforge.intelligence import cluster_feedback, embed, evaluate_clustering  # noqa: E402
from signalforge.pipeline import build_opportunities, rank_opportunities  # noqa: E402
from signalforge.synthetic import generate  # noqa: E402

BUDGET = {"1000": 5.0, "5000": 30.0, "peak_rss_mb": 1536}


def labelled_metrics(df: pd.DataFrame, name: str) -> dict:
    t = time.perf_counter()
    c = cluster_feedback(df)
    dt = time.perf_counter() - t
    m = evaluate_clustering(c["ground_truth_issue"], c["cluster_id"])
    truth, pred = c["ground_truth_issue"].to_numpy(), c["cluster_id"].to_numpy()
    # false merge = pair in same predicted cluster but different true issue; false split = converse
    same_t, same_p = truth[:, None] == truth[None, :], pred[:, None] == pred[None, :]
    iu = np.triu_indices(len(truth), 1)
    purity = float(pd.crosstab(pred, truth).max(axis=1).sum() / len(truth))
    return dict(dataset=name, n=len(df), method=c.attrs["method"], backend=c.attrs["backend"], n_clusters=m.n_clusters,
                ari=round(m.adjusted_rand_index, 3), nmi=round(float(normalized_mutual_info_score(truth, pred)), 3),
                purity=round(purity, 3), pairwise_precision=round(m.pairwise_precision, 3), pairwise_recall=round(m.pairwise_recall, 3),
                false_merge_rate=round(float((same_p & ~same_t)[iu].sum() / max((~same_t)[iu].sum(), 1)), 4),
                false_split_rate=round(float((~same_p & same_t)[iu].sum() / max(same_t[iu].sum(), 1)), 4), seconds=round(dt, 2))


def unlabelled_metrics(df: pd.DataFrame, name: str, seeds=(0, 1, 2)) -> dict:
    t = time.perf_counter()
    c = cluster_feedback(df)
    dt = time.perf_counter() - t
    x, _, _ = embed(df["feedback_text"].tolist())
    sizes = c["cluster_id"].value_counts()
    sil = float(silhouette_score(x, c["cluster_id"], metric="cosine")) if 1 < len(sizes) < len(c) else float("nan")
    # stability: re-cluster 80% subsamples, compare co-assignment on shared records (ARI)
    stab = []
    for s in seeds:
        sub = df.sample(frac=0.8, random_state=s)
        cs = cluster_feedback(sub.reset_index(drop=True))
        a = c.set_index("ticket_id").loc[cs["ticket_id"], "cluster_id"].to_numpy()
        stab.append(adjusted_rand_score(a, cs["cluster_id"].to_numpy()))
    terms = [set(w for w, _ in o["shared_terms"][:5]) for o in build_opportunities(c, Assumptions(assumed_acv_per_account=500))]
    diversity = len(set().union(*terms)) / max(sum(len(t) for t in terms), 1) if terms else 0.0
    return dict(dataset=name, n=len(df), method=c.attrs["method"], backend=c.attrs["backend"], n_clusters=int(len(sizes)),
                largest_cluster_share=round(float(sizes.iloc[0] / len(c)), 3), noise_rate=round(float(c["is_outlier"].mean()), 3),
                silhouette_cosine=round(sil, 3), stability_ari_mean=round(float(np.mean(stab)), 3),
                stability_ari_min=round(float(np.min(stab)), 3), topic_diversity=round(diversity, 3), seconds=round(dt, 2))


def latency(n: int) -> dict:
    df = pd.concat([generate(seed=s) for s in range(int(np.ceil(n / 1000)))]).head(n)
    df["ticket_id"] = [f"T{i}" for i in range(len(df))]
    t = time.perf_counter()
    ing = validate(df)
    c = cluster_feedback(ing.df)
    a = Assumptions()
    rank_opportunities(build_opportunities(c, a), a)
    return dict(records=len(ing.df), seconds=round(time.perf_counter() - t, 2), budget_s=BUDGET[str(n)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/research/evaluation-results.json"))
    args = ap.parse_args()
    res: dict = {"env": dict(python=platform.python_version(), machine=platform.machine()), "labelled": [], "unlabelled": [],
                 "not_run": [], "latency": []}
    res["labelled"].append(labelled_metrics(generate().rename(columns={}), "synthetic templates (labelled)"))
    res["labelled"].append(labelled_metrics(generate_heldout().assign(account_id="a", product_area="x"), "held-out paraphrases + typos (labelled)"))
    for key in ("bitext_support", "fdroid_reviews"):
        try:
            d = load_public(key)
        except Exception as exc:  # noqa: BLE001 - record, don't crash the whole evaluation
            res["not_run"].append(dict(dataset=key, reason=f"{type(exc).__name__}: {exc}"[:300]))
            continue
        if "ground_truth_issue" in d:
            res["labelled"].append(labelled_metrics(d, f"{key} (real, labelled intents)"))
        res["unlabelled"].append(unlabelled_metrics(d, f"{key} (real)"))
    for n in (1000, 5000):
        res["latency"].append(latency(n))
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    res["budget"] = BUDGET
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
