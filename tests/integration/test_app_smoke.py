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
