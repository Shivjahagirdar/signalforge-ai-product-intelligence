"""Semantic clustering with deterministic TF-IDF fallback; evidence-linked opportunities."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger("signalforge.intelligence")


def embed(texts: list[str], prefer_transformer: bool = False) -> tuple[np.ndarray, list[str], str]:
    """Return (matrix, vocabulary terms, backend). Falls back to TF-IDF if the model is unavailable."""
    if prefer_transformer:
        try:  # optional dependency; never required
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer("all-MiniLM-L6-v2")
            return np.asarray(m.encode(texts, normalize_embeddings=True)), [], "sentence-transformers"
        except Exception as exc:  # noqa: BLE001
            log.warning("transformer unavailable (%s); using TF-IDF", type(exc).__name__)
    for min_df in (2, 1):  # min_df=1 fallback avoids all-zero rows on tiny/very diverse inputs
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
        try:
            x = vec.fit_transform(texts)
        except ValueError:  # empty vocabulary
            continue
        if (x.getnnz(axis=1) > 0).all():
            return x.toarray(), list(vec.get_feature_names_out()), "tfidf"
    raise ValueError("Feedback text has no usable words after normalisation.")


def cluster_feedback(df: pd.DataFrame, distance_threshold: float = 0.9, prefer_transformer: bool = False) -> pd.DataFrame:
    """Add cluster_id, cluster_label, cohesion, is_outlier columns. Deterministic."""
    out = df.copy().reset_index(drop=True)
    x, terms, backend = embed(out["feedback_text"].tolist(), prefer_transformer)
    if len(out) < 2:
        out["cluster_id"] = 0
    else:
        cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=distance_threshold)
        out["cluster_id"] = cl.fit_predict(x)
    sims = cosine_similarity(x) if len(out) < 4000 else None
    labels, coh = {}, {}
    for cid, idx in out.groupby("cluster_id").indices.items():
        cent = x[idx].mean(axis=0)
        labels[cid] = ", ".join(terms[i] for i in np.argsort(cent)[::-1][:3]) if terms else f"cluster {cid}"
        if sims is not None and len(idx) > 1:
            sub = sims[np.ix_(idx, idx)]
            coh[cid] = float((sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
        else:
            coh[cid] = 0.0
    out["cluster_label"] = out["cluster_id"].map(labels)
    out["cohesion"] = out["cluster_id"].map(coh)
    sizes = out["cluster_id"].map(out["cluster_id"].value_counts())
    out["is_outlier"] = sizes < 3
    out.attrs["backend"] = backend
    return out


def near_duplicates(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Same-account ticket pairs with very similar wording (candidate duplicates; not auto-removed)."""
    x, _, _ = embed(df["feedback_text"].tolist())
    s = cosine_similarity(x)
    ids = df["ticket_id"].to_numpy()
    acc = df["account_id"].to_numpy()
    i, j = np.where(np.triu(s, 1) >= threshold)
    pairs = [(ids[a], ids[b], float(s[a, b])) for a, b in zip(i, j) if acc[a] == acc[b]]
    return pd.DataFrame(pairs, columns=["ticket_a", "ticket_b", "similarity"])


@dataclass
class ClusterMetrics:
    adjusted_rand_index: float
    pairwise_precision: float
    pairwise_recall: float
    coverage: float
    n_clusters: int


def evaluate_clustering(truth: pd.Series, pred: pd.Series, min_cluster: int = 3) -> ClusterMetrics:
    """Benchmark predicted clusters against hidden ground truth (clustering quality, not calibrated confidence)."""
    t, p = truth.to_numpy(), pred.to_numpy()
    same_t = t[:, None] == t[None, :]
    same_p = p[:, None] == p[None, :]
    iu = np.triu_indices(len(t), 1)
    tp = float((same_t & same_p)[iu].sum())
    prec = tp / max(float(same_p[iu].sum()), 1.0)
    rec = tp / max(float(same_t[iu].sum()), 1.0)
    sizes = pd.Series(p).value_counts()
    cov = float(pd.Series(p).map(sizes).ge(min_cluster).mean())
    return ClusterMetrics(float(adjusted_rand_score(t, p)), prec, rec, cov, len(sizes))
