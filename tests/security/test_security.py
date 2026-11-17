import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from signalforge.ingestion import MAX_BYTES, IngestionError, load_csv, safe_csv

APP = str(Path(__file__).resolve().parents[2] / "app.py")
ROOT = Path(__file__).resolve().parents[2]

def df_with(at, *cols, exact=False):
    """Find a rendered dataframe by its columns (AppTest can't look dataframes up by key)."""
    for d in at.dataframe:
        c = list(d.value.columns)
        if (c == list(cols)) if exact else set(cols) <= set(c):
            return d.value
    raise AssertionError(f"no dataframe with {cols}")



@pytest.mark.parametrize("payload", ["=HYPERLINK(\"http://x\")", "+1+1", "-2+3", "@SUM(A1)", "\t=cmd"])
def test_csv_injection_neutralised(payload):
    out = pd.read_csv(io.BytesIO(safe_csv(pd.DataFrame({"t": [payload]}))))
    assert out.t.iloc[0].startswith("'")


def test_oversized_and_wrong_extension_rejected():
    with pytest.raises(IngestionError):
        load_csv(b"a" * (MAX_BYTES + 1), "x.csv")
    with pytest.raises(IngestionError):
        load_csv(b"a,b\n1,2", "../../etc/passwd")


def test_no_secrets_committed():
    import re
    import subprocess

    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    rx = re.compile(r"(sk-ant-[A-Za-z0-9-]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)")
    hits = [f for f in files if f != "tests/security/test_security.py" and (ROOT / f).is_file() and (ROOT / f).stat().st_size < 2_000_000
            and rx.search((ROOT / f).read_text(errors="ignore"))]
    assert not hits, hits


def test_api_key_never_rendered(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-" + "ant-test-SHOULD-NOT-LEAK-" + "0" * 12)
    import signalforge.copilot as cp

    monkeypatch.setattr(cp, "narrate_with_llm", lambda rec, **k: None)
    at = AppTest.from_file(APP, default_timeout=180).run()
    [b for b in at.button if b.label == "What should we build next?"][0].click().run()
    blob = " ".join(str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.caption) + list(at.text))
    assert "SHOULD-NOT-LEAK" not in blob


def test_sessions_are_isolated(tmp_path, monkeypatch):
    """Two independent visitors: overrides and copilot history in one never appear in the other."""
    monkeypatch.chdir(tmp_path)
    a = AppTest.from_file(APP, default_timeout=180).run()
    b = AppTest.from_file(APP, default_timeout=180).run()
    a.sidebar.radio(key="page").set_value("Roadmap Optimizer").run()
    a.text_input(key="ov_why").set_value("visitor A secret rationale")
    a.button(key="b_override").click().run()
    assert "visitor A secret rationale" in df_with(a, "framework_rank")["override_rationale"].tolist()
    b.sidebar.radio(key="page").set_value("Roadmap Optimizer").run()
    assert "visitor A secret rationale" not in df_with(b, "framework_rank")["override_rationale"].tolist()
    assert a.session_state["store"] is not b.session_state["store"]
