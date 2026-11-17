"""Reproducible benchmark on synthetic data. Run: python scripts/benchmark.py"""
import sys
import time

sys.path.insert(0, "src")
from signalforge.economics import Assumptions
from signalforge.heldout import generate_heldout
from signalforge.ingestion import validate
from signalforge.intelligence import cluster_feedback, evaluate_clustering
from signalforge.pipeline import build_opportunities, rank_opportunities
from signalforge.synthetic import generate

if __name__ == "__main__":
    df = generate()
    r = validate(df)
    t = time.time()
    c = cluster_feedback(r.df)
    lat = time.time() - t
    m = evaluate_clustering(c["ground_truth_issue"], c["cluster_id"])
    print(f"rows={len(c)} backend={c.attrs['backend']} clusters={m.n_clusters} latency={lat:.1f}s")
    print(f"ARI={m.adjusted_rand_index:.3f} pairwise_precision={m.pairwise_precision:.3f} recall={m.pairwise_recall:.3f} coverage={m.coverage:.3f}")
    h = cluster_feedback(generate_heldout().assign(account_id="a", product_area="x"))
    mh = evaluate_clustering(h["ground_truth_issue"], h["cluster_id"])
    print(f"HELD-OUT wording: clusters={mh.n_clusters} ARI={mh.adjusted_rand_index:.3f} precision={mh.pairwise_precision:.3f} recall={mh.pairwise_recall:.3f}")
    ranked = rank_opportunities(build_opportunities(c, Assumptions()), Assumptions())
    print(ranked[["final_rank", "label", "unique_accounts", "acv_sum_unique", "net_opportunity_value", "baseline_rank"]].head(8).to_string(index=False))
