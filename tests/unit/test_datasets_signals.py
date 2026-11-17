
import numpy as np
import pandas as pd
import pytest

from signalforge.datasets import (
    REGISTRY,
    adapt_app_reviews,
    adapt_bitext,
    adapt_github_issues,
    coerce_to_schema,
    data_quality,
    detect_mapping,
    load_public,
)
from signalforge.ingestion import IngestionError, validate
from signalforge.signals import is_actionable, tag_signals, trend


def test_registry_has_provenance_and_license_for_every_dataset():
    assert len([s for s in REGISTRY.values() if s.kind != "synthetic"]) >= 3
    for s in REGISTRY.values():
        assert s.url and s.license and s.provenance and s.limitations
        if s.kind != "synthetic":
            assert not s.redistributed_in_repo


def test_app_review_adapter_keeps_ids_and_never_invents_revenue():
    raw = pd.DataFrame({"id": ["a1", "a2", "a3"], "package_name": ["org.x.app"] * 3, "star": [1, 5, 3],
                        "review": ["Crashes every time I open the camera screen", "Love it, works perfectly well", "Sync to the server fails after update"],
                        "date": ["October 12 2016", "August 23 2016", "May 1 2016"]})
    out = adapt_app_reviews(raw)
    assert set(out.ticket_id) == {"a1", "a3"}  # 5-star praise filtered, ids preserved
    assert out.annual_contract_value.isna().all() and out.source.eq("fdroid_reviews").all()
    assert out.loc[out.ticket_id == "a1", "severity"].item() == 5
    assert validate(out).df.annual_contract_value.isna().all()


def test_bitext_adapter_keeps_ground_truth():
    raw = pd.DataFrame({"instruction": ["i want a refund", "where is my order"], "category": ["REFUND", "ORDER"],
                        "intent": ["get_refund", "track_order"], "response": ["", ""], "flags": ["", ""]})
    out = adapt_bitext(raw)
    assert list(out.ground_truth_issue) == ["get_refund", "track_order"]


def test_github_adapter_skips_prs_and_pseudonymises_users():
    items = [{"number": 1, "title": "Crash on save", "body": "Steps```code```", "labels": [{"name": "bug"}], "user": {"login": "alice"},
              "reactions": {"total_count": 4}, "created_at": "2025-01-01T00:00:00Z", "state": "open"},
             {"number": 2, "title": "PR", "pull_request": {}, "user": {"login": "bob"}}]
    out = adapt_github_issues(items, "o/r")
    assert list(out.ticket_id) == ["o/r#1"] and "alice" not in out.account_id.iloc[0]
    assert out.frequency.iloc[0] == 5 and out.severity.iloc[0] == 4 and "code" not in out.feedback_text.iloc[0]


def test_github_repo_name_is_validated_before_any_network_call():
    with pytest.raises(IngestionError):
        load_public("github_issues", "not a repo; rm -rf /")


def test_schema_detection_and_mapping_reports_assumptions():
    df = pd.DataFrame({"Review": ["Export to CSV is broken", "Please add dark mode"], "Rating": [1, 3], "Date": ["2024-01-01", "2024-02-01"]})
    m = detect_mapping(list(df.columns))
    assert m["feedback_text"] == "Review" and m["rating"] == "Rating"
    out, notes = coerce_to_schema(df)
    assert list(out.severity) == [5, 3] and out.annual_contract_value.isna().all()
    assert any("UNKNOWN" in n for n in notes)


def test_no_text_column_gives_actionable_error():
    with pytest.raises(IngestionError, match="text column"):
        coerce_to_schema(pd.DataFrame({"a": [1], "b": [2]}))


def test_data_quality_score_components():
    q = data_quality(pd.DataFrame({"feedback_text": ["a b c d e f", ""], "annual_contract_value": [1.0, np.nan],
                                   "created_at": ["2024-01-01", None], "customer_segment": ["SMB", "Unknown"],
                                   "estimated_engineering_effort": [np.nan, np.nan]}))
    assert 0 <= q["score"] <= 100 and q["components"]["acv_coverage"] == 0.5


@pytest.mark.parametrize("text,tag", [("The app crashes on startup", "reliability"), ("Salesforce sync fails", "integration"),
                                      ("Too expensive for our plan", "pricing"), ("We are switching to a competitor", "churn_language"),
                                      ("Would love a dark mode", "feature_request")])
def test_signal_rules(text, tag):
    assert tag in tag_signals(text)


def test_thanks_chatter_is_not_actionable():
    g = pd.DataFrame({"feedback_text": ["Thanks for the call", "Great product overall", "Thanks again"]})
    assert not is_actionable(g, {})


def test_trend_is_volume_normalised():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    all_df = pd.DataFrame({"created_at": list(dates) + list(dates[-100:]) * 3})  # total volume grows a lot recently
    g = pd.DataFrame({"created_at": list(dates[::10])})  # theme volume flat over time
    t = trend(g, all_df)
    assert t["status"] in ("stable", "falling")  # growth in total volume must not make a flat theme 'rising'
