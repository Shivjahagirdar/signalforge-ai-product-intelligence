from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")


def test_app_starts_and_pages_render(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    for page in ["Customer Intelligence", "Opportunity Economics", "Roadmap Optimizer", "Scenario Lab", "Decision Memo & PRD"]:
        at.sidebar.radio[1].set_value(page).run()
        assert not at.exception, page


def _goto(at, page):
    at.sidebar.radio[1].set_value(page).run()
    return at


def test_override_flow_persists_and_is_audited(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=120).run()
    _goto(at, "Roadmap Optimizer")
    last = at.selectbox[2].options[-1]
    at.selectbox[2].set_value(last)  # theme to move
    at.number_input[-1].set_value(1)
    at.text_input[-1].set_value("Strategic priority for key account")
    [b for b in at.button if b.label == "Apply override"][0].click().run()
    assert not at.exception
    df = at.dataframe[0].value
    assert df.iloc[0]["override_rationale"] == "Strategic priority for key account"
    _goto(at, "Decision Memo & PRD")
    events = at.dataframe[-1].value
    assert "rank_override" in events["event"].tolist()


def test_override_without_rationale_is_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=120).run()
    _goto(at, "Roadmap Optimizer")
    [b for b in at.button if b.label == "Apply override"][0].click().run()
    assert not at.exception and at.dataframe[0].value["override_rationale"].eq("").all()


def test_assumption_change_updates_rankings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=120).run()
    _goto(at, "Opportunity Economics")
    base = at.dataframe[0].value["net_opportunity_value"].max()
    at.sidebar.slider[1].set_value(0.05).run()  # recovery probability
    assert not at.exception
    assert at.dataframe[0].value["net_opportunity_value"].max() < base


def test_session_store_is_in_memory_not_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    AppTest.from_file(APP, default_timeout=120).run()
    assert not list(tmp_path.glob("*.db"))
