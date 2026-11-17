"""Real-browser E2E (Playwright + Chromium/Chrome) against a live `streamlit run` server.

Run:  pytest tests/e2e -m e2e   (set SF_E2E_URL to test a deployed URL instead of a local server)
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "docs" / "screenshots"


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def base_url():
    if os.environ.get("SF_E2E_URL"):
        yield os.environ["SF_E2E_URL"]; return
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless=true",
                             f"--server.port={port}", "--browser.gatherUsageStats=false"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            socket.create_connection(("127.0.0.1", port), 1).close(); break
        except OSError:
            time.sleep(1)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        kw = {"executable_path": os.environ["SF_CHROME"]} if os.environ.get("SF_CHROME") else {}
        b = p.chromium.launch(headless=True, **kw)
        yield b
        b.close()


def _wait(page, text, timeout=120_000, state="visible"):
    page.get_by_text(text, exact=False).first.wait_for(timeout=timeout, state=state)


def _nav(page, name):
    page.locator('[data-testid="stSidebar"]').get_by_text(name, exact=True).click()
    page.get_by_role("heading", name=name if name != "Decision Memo & PRD" else "Executive Decision Memo").wait_for(timeout=120_000)
    page.wait_for_timeout(800)


def _shot(page, name):
    if not os.environ.get("SF_E2E_URL"):
        ART.mkdir(parents=True, exist_ok=True)
        page.wait_for_timeout(3000)  # let plotly charts finish drawing
        page.screenshot(path=str(ART / f"{name}.png"), full_page=False)


def test_full_decision_journey(base_url, browser, tmp_path):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
    page = ctx.new_page()
    page.goto(base_url)
    _wait(page, "Executive Dashboard")                                  # 1-2 open, demo loads by default
    _wait(page, "Data quality")                                         # 3 data quality visible
    _shot(page, "01-executive-dashboard")
    _nav(page, "Customer Intelligence")                                 # 4-7 clusters, evidence, accounts
    _wait(page, "why these records were grouped")
    _wait(page, "Representative records")
    _shot(page, "02-customer-intelligence")
    _nav(page, "Opportunity Economics")                                 # 8-9 assumptions & exposure
    _wait(page, "Formulas")
    _shot(page, "03-opportunity-economics")
    _nav(page, "Roadmap Optimizer")                                     # 10-13 frameworks, capacity, override
    _wait(page, "Recommended roadmap")
    cap = page.get_by_label("Engineering capacity (person-weeks)")
    cap.fill("5"); cap.press("Enter"); page.wait_for_timeout(2500)
    _wait(page, "Binding constraints")
    page.get_by_label("Rationale (required)").fill("E2E: committed to strategic account"); page.keyboard.press("Enter")
    page.get_by_role("button", name="Apply override").click(); page.wait_for_timeout(2500)
    _wait(page, "E2E: committed to strategic account", state="attached")  # dataframe cells are canvas-backed
    _shot(page, "04-roadmap-optimizer")
    _nav(page, "Scenario Lab")                                          # 14 simulation
    _wait(page, "Monte Carlo")
    _shot(page, "05-scenario-lab")
    page.get_by_text("SIGNAL - your AI product strategist").click()     # 15 SIGNAL challenge
    page.get_by_role("button", name="Challenge this roadmap.").click()
    _wait(page, "Counterargument")
    page.get_by_text("Counterargument").first.scroll_into_view_if_needed()
    _shot(page, "06-signal-copilot")
    _nav(page, "Decision Memo & PRD")                                   # 16-19 generate + download PRD & memo
    for label, fname in [("PRD (.md)", "prd.md"), ("Executive memo (.docx)", "executive_memo.docx")]:
        with page.expect_download() as d:
            page.get_by_role("button", name=label).click()
        path = tmp_path / fname
        d.value.save_as(path)
        assert path.stat().st_size > 500
    assert "# PRD:" in (tmp_path / "prd.md").read_text()
    _shot(page, "07-decision-memo")
    _nav(page, "Roadmap Optimizer")                                     # 24 state persists across navigation
    _wait(page, "E2E: committed to strategic account", state="attached")  # dataframe cells are canvas-backed
    ctx.close()


def test_upload_malformed_empty_and_independent_session(base_url, browser, tmp_path):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(base_url)
    _wait(page, "Executive Dashboard")
    page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first.click()
    page.get_by_role("option", name="Upload file").click()
    _wait(page, "No data, no drama")
    bad = tmp_path / "bad.csv"; bad.write_bytes(b"\x00\x01\x02garbage")
    page.locator('input[type="file"]').set_input_files(str(bad))       # 22 malformed input
    _wait(page, "isn't cooperating")
    empty = tmp_path / "empty.csv"; empty.write_text("feedback_text\n")
    page.locator('input[type="file"]').set_input_files(str(empty))     # 23 empty dataset
    _wait(page, "isn't cooperating")
    good = tmp_path / "reviews.csv"
    good.write_text("review,rating\n" + "\n".join(
        [f"App crashes when I open the camera screen {i}" ,f"Sync with the server fails after update {i}", f"Please add an export to PDF option {i}"][i % 3] + f",{1 + i % 3}"
        for i in range(60)))
    page.locator('input[type="file"]').set_input_files(str(good))      # 20-21 independent upload, core analysis repeats
    _wait(page, "NO REVENUE DATA")
    _wait(page, "Actionable themes")
    other = browser.new_context().new_page()                          # 25 separate visitor session
    other.goto(base_url)
    _wait(other, "SYNTHETIC DEMO")
    assert other.get_by_text("NO REVENUE DATA").count() == 0
    other.context.close(); ctx.close()
