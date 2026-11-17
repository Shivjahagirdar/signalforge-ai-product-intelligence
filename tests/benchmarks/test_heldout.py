"""Held-out wording (paraphrases, typos, caps) - NOT the training templates. These are regression guards on a
deliberately modest bar: lexical TF-IDF over-splits paraphrases, so recall is low. See docs/AI-evaluation.md."""
import pandas as pd
import pytest

from signalforge.heldout import generate_heldout
from signalforge.ingestion import IngestionError, load_csv
from signalforge.intelligence import cluster_feedback, embed, evaluate_clustering, suggest_merges


@pytest.fixture(scope="module")
def held():
    return cluster_feedback(generate_heldout().assign(account_id="a", product_area="x"))


def test_heldout_clusters_are_pure_but_oversplit(held):
    m = evaluate_clustering(held["ground_truth_issue"], held["cluster_id"])
    assert m.pairwise_precision > 0.85          # clusters that form are mostly correct
    assert m.adjusted_rand_index > 0.2          # far below the synthetic-template score: documented limitation
    assert m.pairwise_recall < 0.6              # guard against silently over-claiming


def test_merge_suggestions_surface_paraphrase_splits(held):
    sm = suggest_merges(held, top_n=10, min_similarity=0.05)
    truth = held.groupby("cluster_id")["ground_truth_issue"].agg(lambda s: s.mode().iloc[0])
    same = [(truth[a] == truth[b]) for a, b in zip(sm.cluster_a, sm.cluster_b)]
    assert len(sm) > 0 and sum(same) >= 1       # at least one useful suggestion is a true same-issue pair


def test_transformer_failure_falls_back_to_tfidf():
    x, _, backend = embed(["app crashes on launch", "report export is slow"] * 3, prefer_transformer=True)
    assert backend in ("tfidf-word+char", "sentence-transformers") and len(x) == 6


def test_typos_and_case_do_not_crash():
    df = pd.DataFrame({"ticket_id": ["1", "2", "3"], "feedback_text": ["CRASH!!!", "crahs on startup", "???"],
                       "account_id": "a", "product_area": "x"})
    assert "cluster_id" in cluster_feedback(df)


def test_unusable_text_gives_clear_error():
    with pytest.raises(ValueError, match="no usable words"):
        embed(["!!!", "???"])


@pytest.mark.parametrize("payload", [b"a,b\n1,2\n", b"ticket_id\n", b"\xff\xfe\x00bad"])
def test_malformed_csv_raises_safe_error(payload):
    with pytest.raises(IngestionError):
        load_csv(payload, "x.csv")


def test_header_only_valid_schema_is_empty_dataset_error():
    from signalforge.ingestion import REQUIRED
    with pytest.raises(IngestionError):
        load_csv((",".join(REQUIRED) + "\n").encode(), "x.csv")
