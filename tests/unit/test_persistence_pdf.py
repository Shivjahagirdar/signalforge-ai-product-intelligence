import io
import json

import pytest
from tests.fixtures_data import demo

from signalforge.audit import AuditStore, Overrides
from signalforge.exports import generate_experiment_plan, markdown_to_pdf
from signalforge.persistence import restore, snapshot


def test_snapshot_roundtrip_restores_everything():
    ov = Overrides(rename={3: "Billing pain"}, rank={3: {"rank": 1, "rationale": "board ask"}}, rejected=[7])
    raw = snapshot({"dataset": "Synthetic demo (1,000 records)", "thr": 0.85, "rec": 0.5, "assumed": None}, ov,
                   {"cap": 12.0, "minconf": 0.2, "budget": None, "mandatory": [3], "deps": [[3, 4]]},
                   [("2026-01-01T00:00:00+00:00", "rank_override", "x")], ["synthetic", 996, 0.85, "T-00001"])
    s = restore(raw)
    assert s["widgets"] == {"dataset": "Synthetic demo (1,000 records)", "thr": 0.85, "rec": 0.5}
    assert s["overrides"].rename == {3: "Billing pain"} and s["overrides"].rank[3]["rationale"] == "board ask" and s["overrides"].rejected == [7]
    assert s["constraints"]["cap"] == 12.0 and s["constraints"]["deps"] == [(3, 4)]
    st = AuditStore(); st.import_events(s["audit"])
    assert st.events()[0][1] == "rank_override"


@pytest.mark.parametrize("raw", [b"not json", b"[]", json.dumps({"schema": 99}).encode(), b"x" * 1_000_001,
                                 json.dumps({"schema": 1, "widgets": {"thr": {"evil": 1}}}).encode()])
def test_restore_rejects_bad_files(raw):
    with pytest.raises(ValueError):
        restore(raw)


def test_snapshot_never_contains_raw_feedback():
    clustered, *_ = demo()
    raw = snapshot({}, Overrides(), {}, [], [])
    assert clustered.feedback_text.iloc[0][:30].encode() not in raw


def test_pdf_export_is_valid_and_contains_text():
    import pypdf

    _, _, _, ranked = demo()
    md = "# Title — with dash → arrow\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n" + generate_experiment_plan(ranked)
    pdf = pypdf.PdfReader(io.BytesIO(markdown_to_pdf(md)))
    text = "".join(p.extract_text() for p in pdf.pages)
    assert "Experiment Plan" in text and ranked.label.iloc[0] in text


def test_session_metrics_counts_corrections_and_usage():
    from signalforge.metrics import session_metrics

    ev = [("t", "copilot", "x"), ("t", "merge", "a"), ("t", "reject", "b"), ("t", "export", "pdf"), ("t", "dependency", "c")]
    m = session_metrics(ev, n_themes=10, seconds_to_first_insight=2.345)
    assert m["corrections"] == 2 and m["correction_rate"] == 0.2
    assert m["signal_questions"] == 1 and m["exports"] == 1 and m["seconds_to_first_insight"] == 2.3
    assert session_metrics([], 0, None)["correction_rate"] is None
