"""AppTest: the full evidence -> decision -> ticket -> release -> outcome journey through the real UI, plus isolation."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")


@pytest.fixture
def at(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIGNALFORGE_DB", str(tmp_path / "loop.db"))
    a = AppTest.from_file(APP, default_timeout=180).run()
    a.sidebar.radio(key="page").set_value("Decision Loop").run()
    assert not a.exception, a.exception
    return a


def _ok(a):
    assert not a.exception, a.exception
    assert not [e.value for e in a.error], [e.value for e in a.error]
    return a


def test_full_closed_loop_through_ui(at):
    at.button(key="b_propose").click().run()
    assert any("rationale" in e.value.lower() or "owner" in e.value.lower() for e in at.error)  # validation, nothing created
    at.text_input(key="loop_owner").set_value("Shivani")
    at.text_input(key="loop_why").set_value("Top risk-adjusted value with strong evidence")
    _ok(at.button(key="b_propose").click().run())
    assert any("proposed" in s.value for s in at.success)
    at.text_input(key="loop_note").set_value("Approved in roadmap review")
    _ok(at.button(key="b_approve").click().run())
    assert "approved" in at.selectbox(key="loop_dec").format_func(at.selectbox(key="loop_dec").value)
    _ok(at.button(key="b_ticket").click().run())
    assert any("Ticket #" in s.value for s in at.success)
    labels = [d.label for d in at.get("download_button")]
    assert {"Ticket (.md)", "PRD (.md)", "PRD (.pdf)"} <= set(labels)
    _ok(at.button(key="b_release").click().run())
    assert any("Released v1.0" in s.value for s in at.success)
    at.slider(key="tel_lift").set_value(0.25)
    _ok(at.button(key="b_measure").click().run())
    html = " ".join(m.value for m in at.markdown)
    assert "Verdict:" in html and "SIMULATED" in html and "Recommended at rank" in html
    assert "measured" in at.selectbox(key="loop_dec").format_func(at.selectbox(key="loop_dec").value)
    # SIGNAL answers from the recorded outcome
    at.button(key="q_Decision Loop_11").click().run()
    assert any("Adoption" in m.value for m in at.markdown)


def test_workspace_data_not_visible_to_another_session(at, tmp_path, monkeypatch):
    at.text_input(key="loop_owner").set_value("A-owner")
    at.text_input(key="loop_why").set_value("session A secret rationale")
    _ok(at.button(key="b_propose").click().run())
    b = AppTest.from_file(APP, default_timeout=180).run()
    b.sidebar.radio(key="page").set_value("Decision Loop").run()
    assert not b.exception
    assert not any("secret rationale" in m.value for m in b.markdown)
    assert any("No decisions yet" in i.value for i in b.info)


def test_resume_workspace_by_key(at):
    at.text_input(key="loop_owner").set_value("Owner")
    at.text_input(key="loop_why").set_value("resume me")
    _ok(at.button(key="b_propose").click().run())
    key = at.session_state["ws"]
    b = AppTest.from_file(APP, default_timeout=180).run()
    b.text_input(key="ws_resume").set_value(key)
    b.button(key="b_ws_open").click().run()
    b.sidebar.radio(key="page").set_value("Decision Loop").run()
    assert b.session_state["ws"] == key
    assert any("resume me" in m.value for m in b.markdown)
    c = AppTest.from_file(APP, default_timeout=180).run()
    c.text_input(key="ws_resume").set_value("ws_notarealkey00")
    c.button(key="b_ws_open").click().run()
    assert any("No workspace" in e.value for e in c.error)
