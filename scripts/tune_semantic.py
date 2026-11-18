"""Grid-search semantic weight x cluster threshold on labelled data (needs the Model2Vec download; run in CI)."""
import itertools
import json
import os
import sys
from pathlib import Path

os.environ["SF_SEMANTIC"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

from signalforge import intelligence as I  # noqa: E402
from signalforge.datasets import load_public  # noqa: E402
from signalforge.heldout import generate_heldout  # noqa: E402
from signalforge.synthetic import generate  # noqa: E402

sets = {"synthetic": generate(), "heldout": generate_heldout().assign(account_id="a", product_area="x"),
        "bitext": load_public("bitext_support")}
fd = load_public("fdroid_reviews")
assert I._semantic_model() is not None, "semantic model failed to load"
rows = []
for w, thr in itertools.product([0.0, 0.5, 0.7, 0.85, 1.0], [0.6, 0.7, 0.8, 0.9]):
    I.SEMANTIC_WEIGHT = w
    r = {"w": w, "thr": thr}
    for k, d in sets.items():
        c = I.cluster_feedback(d, thr)
        r[k] = round(ari(c["ground_truth_issue"], c["cluster_id"]), 3)
        r[k + "_k"] = c["cluster_id"].nunique()
    c = I.cluster_feedback(fd, thr)
    r["fdroid_method"] = c.attrs["method"]
    r["fdroid_k"] = c["cluster_id"].nunique()
    rows.append(r)
    print(json.dumps(r), flush=True)
df = pd.DataFrame(rows)
df["min_ari"] = df[["synthetic", "heldout", "bitext"]].min(axis=1)
print("TUNE_RESULT_BEGIN")
print(df.sort_values("min_ari", ascending=False).to_string())
print("TUNE_RESULT_END")
