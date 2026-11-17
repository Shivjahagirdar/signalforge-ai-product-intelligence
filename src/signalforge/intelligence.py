"""Semantic clustering with deterministic TF-IDF fallback; evidence-linked opportunities."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

log = logging.getLogger("signalforge.intelligence")
LSA_DIMS = 100
GIANT_SHARE = 0.35  # a 'theme' holding >35% of records is too broad to act on


def embed(texts: list[str], prefer_transformer: bool = False) -> tuple[np.ndarray, list[str], str]:
    """Return (matrix, vocabulary terms, backend).

    Order: sentence-transformers (optional, needs a model download) -> hybrid word+char TF-IDF (offline, deterministic).
    Char n-grams add robustness to typos and word forms; they do NOT understand true paraphrases (see docs/AI-evaluation.md).
    """
    if prefer_transformer:
        try:  # optional dependency; never required, never verified offline
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer("all-MiniLM-L6-v2")
            return np.asarray(m.encode(texts, normalize_embeddings=True)), [], "sentence-transformers"
        except Exception as exc:  # noqa: BLE001
            log.warning("transformer unavailable (%s); using TF-IDF fallback", type(exc).__name__)
    for min_df in (2, 1):  # min_df=1 fallback avoids all-zero rows on tiny/very diverse inputs
        try:
            wv = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
            cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True)
            w, c = wv.fit_transform(texts), cv.fit_transform(texts)
        except ValueError:  # empty vocabulary
            continue
        x = normalize(hstack([w, c]).tocsr())
        if (x.getnnz(axis=1) > 0).all():
            k = min(LSA_DIMS, x.shape[1] - 1, x.shape[0] - 1)
            if k >= 20:  # Latent Semantic Analysis: dense, fast, captures co-occurrence beyond exact tokens
                z = normalize(TruncatedSVD(k, random_state=0).fit_transform(x))
                return z, list(wv.get_feature_names_out()), "tfidf-word+char+lsa"
            return x.toarray(), list(wv.get_feature_names_out()), "tfidf-word+char"
    raise ValueError("Feedback text has no usable words after normalisation.")


def _doc_freq(df: pd.DataFrame) -> tuple[dict, int]:
    from collections import Counter

    stop = TfidfVectorizer(stop_words="english").get_stop_words()
    c = Counter(w for t in df["feedback_text"] for w in set(re.findall(r"[a-z]{4,}", t.lower())) if w not in stop)
    return dict(c), len(df)


def _label(df: pd.DataFrame, idx, cid: int, gdf: dict, n_all: int) -> str:
    """Label = 3 words most distinctive for this cluster (cluster share x inverse global frequency); no invented text."""
    from collections import Counter

    local = Counter(w for t in df["feedback_text"].iloc[idx] for w in set(re.findall(r"[a-z]{4,}", t.lower())) if w in gdf)
    scored = sorted(local, key=lambda w: -(local[w] / len(idx)) * np.log(1 + n_all / gdf[w]))
    return ", ".join(scored[:3]) if scored else f"cluster {cid}"


def _auto_k(n: int) -> int:
    return int(np.clip(round(np.sqrt(n / 2)), 6, 30))


def cluster_feedback(df: pd.DataFrame, distance_threshold: float = 0.9, prefer_transformer: bool = False,
                     method: str = "auto") -> pd.DataFrame:
    """Add cluster_id, cluster_label, cohesion, is_outlier columns. Deterministic.

    method='auto': average-linkage agglomerative first (best on templated/consistent feedback). If that produces an
    unusable partition on diverse real-world text - one giant theme (> GIANT_SHARE of records) or heavy fragmentation
    (>40% of records in clusters smaller than 3) - fall back to K-Means with k = clip(sqrt(n/2), 6, 30).
    The method actually used is recorded in out.attrs['method']."""
    out = df.copy().reset_index(drop=True)
    x, terms, backend = embed(out["feedback_text"].tolist(), prefer_transformer)
    used = "single"
    if len(out) < 2:
        out["cluster_id"] = 0
    else:
        p = None
        if method in ("auto", "agglomerative"):
            p = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                        distance_threshold=distance_threshold).fit_predict(x)
            used = "agglomerative"
            sizes = pd.Series(p).value_counts()
            frag = float(pd.Series(p).map(sizes).lt(3).mean())
            if method == "auto" and len(out) >= 60 and (sizes.iloc[0] / len(out) > GIANT_SHARE or frag > 0.4):
                p = None
        if p is None:
            k = min(_auto_k(len(out)), len(out))
            p = KMeans(k, n_init=4, random_state=0).fit_predict(x)
            used = f"kmeans(k={k})"
        out["cluster_id"] = p
    sims = cosine_similarity(x) if len(out) < 4000 else None
    labels, coh = {}, {}
    gdf, n_all = _doc_freq(out)
    for cid, idx in out.groupby("cluster_id").indices.items():
        labels[cid] = _label(out, idx, cid, gdf, n_all)
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
    out.attrs["method"] = used
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


def suggest_merges(clustered: pd.DataFrame, top_n: int = 5, min_similarity: float = 0.15) -> pd.DataFrame:
    """Suggest cluster pairs a human may want to merge (centroid similarity). Mitigates over-splitting on paraphrases."""
    x, _, _ = embed(clustered["feedback_text"].tolist())
    ids = sorted(clustered["cluster_id"].unique())
    cent = np.vstack([x[(clustered["cluster_id"] == c).to_numpy()].mean(axis=0) for c in ids])
    sim = cosine_similarity(cent)
    sizes = clustered["cluster_id"].value_counts()
    rows = [(ids[i], ids[j], float(sim[i, j])) for i in range(len(ids)) for j in range(i + 1, len(ids))
            if sizes[ids[i]] >= 3 and sizes[ids[j]] >= 3 and sim[i, j] >= min_similarity]
    out = pd.DataFrame(rows, columns=["cluster_a", "cluster_b", "similarity"])
    return out.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
