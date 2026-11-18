"""Streamlit AppTest: runs the real app script headlessly (widgets, reruns, session state)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")
PAGES = ["Executive Dashboard", "Customer Intelligence", "Opportunity Economics", "Roadmap Optimizer", "Scenario Lab", "Decision Memo & PRD"]

def df_with(at, *cols, exact=False):
    """Find a rendered dataframe by its columns (AppTest can't look dataframes up by key)."""
    for d in at.dataframe:
        c = list(d.value.columns)
        if (c == list(cols)) if exact else set(cols) <= set(c):
            return d.value
    raise AssertionError(f"no dataframe with {cols}")



@pytest.fixture
def at(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(APP, default_timeout=180).run()


def goto(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, at.exception
    return at


def test_all_six_pages_render(at):
    assert not at.exception
    for p in PAGES:
        goto(at, p)


def test_override_persists_across_navigation_and_is_audited(at):
    goto(at, "Roadmap Optimizer")
    last = at.selectbox(key="ov_theme").options[-1]
    at.selectbox(key="ov_theme").set_value(last)
    at.text_input(key="ov_why").set_value("Strategic priority for key account")
    at.button(key="b_override").click().run()
    df = df_with(at, "framework_rank")
    assert "Strategic priority for key account" in df["override_rationale"].tolist()
    goto(at, "Customer Intelligence"); goto(at, "Roadmap Optimizer")
    assert "Strategic priority for key account" in df_with(at, "framework_rank")["override_rationale"].tolist()
    goto(at, "Decision Memo & PRD")
    assert "rank_override" in df_with(at, "event")["event"].tolist()


def test_override_without_rationale_is_ignored(at):
    goto(at, "Roadmap Optimizer")
    at.button(key="b_override").click().run()
    assert df_with(at, "framework_rank")["override_rationale"].fillna("").eq("").all()


def test_assumption_change_recalculates_economics(at):
    goto(at, "Opportunity Economics")
    base = df_with(at, "accounts_observed")["net_opportunity_value"].max()
    at.slider(key="rec").set_value(0.05).run()
    assert df_with(at, "accounts_observed")["net_opportunity_value"].max() < base


def test_capacity_change_recalculates_roadmap(at):
    goto(at, "Roadmap Optimizer")
    n_big = len(df_with(at, "label", "effort_weeks", "net_opportunity_value", "confidence", exact=True))
    at.number_input(key="cap_in").set_value(3.0).run()
    plan = df_with(at, "label", "effort_weeks", "net_opportunity_value", "confidence", exact=True)
    assert plan["effort_weeks"].sum() <= 3.0 and len(plan) <= n_big


def test_human_review_merge_split_reject(at):
    goto(at, "Customer Intelligence")
    n0 = len(at.selectbox(key="theme").options)
    at.button(key="b_merge").click().run()
    assert len(at.selectbox(key="theme").options) == n0 - 1
    at.button(key="b_reject").click().run()
    assert len(at.selectbox(key="theme").options) == n0 - 2
    at.button(key="b_undo").click().run()
    assert len(at.selectbox(key="theme").options) == n0


def test_copilot_answers_from_state_on_every_page(at):
    for p in PAGES:
        goto(at, p)
        [b for b in at.button if b.label == "Challenge this roadmap."][0].click().run()
        assert not at.exception
        assert any("Counterargument" in m.value for m in at.markdown)


def test_exports_present(at):
    goto(at, "Decision Memo & PRD")
    labels = [d.label for d in at.get("download_button")]
    for name in ["Executive memo (.md)", "PRD (.md)", "PRD (.docx)", "PRD (.pdf)", "Evidence appendix (.docx)",
                 "Download full decision pack (.docx)", "Download full decision pack (.pdf)", "Save analysis (.json)"]:
        assert name in labels


def test_session_store_is_in_memory_not_disk(at, tmp_path):
    assert not list(tmp_path.glob("*.db"))


def test_semantic_toggle_falls_back_cleanly_offline(at):
    at.sidebar.toggle(key="semantic").set_value(True).run()   # model download is unavailable/disabled in tests
    assert not at.exception, at.exception
    assert at.session_state["thr"] == 0.8                      # recommended breadth for semantic matching is applied
    assert any("lexical backend" in w.value for w in at.sidebar.warning)
    at.sidebar.toggle(key="semantic").set_value(False).run()
    assert at.session_state["thr"] == 0.9 and not at.exception


def test_next_step_button_navigates_and_metrics_panel_renders(at):
    next(b for b in at.button if b.key == "next_0").click().run()
    assert at.sidebar.radio(key="page").value == "Customer Intelligence"
    goto(at, "Decision Memo & PRD")
    assert any("Time to first insight" in m.label for m in at.metric)
