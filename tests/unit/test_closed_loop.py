"""Closed-loop vertical slice: store, GitHub connector, telemetry, evidence quality, lifecycle, SIGNAL tools, PDF."""
import json
import re

import pandas as pd
import pytest

from signalforge import evidence_quality as EQ
from signalforge import lifecycle as LC
from signalforge import telemetry as TM
from signalforge.copilot import CopilotState, answer
from signalforge.economics import Assumptions
from signalforge.exports import markdown_to_pdf
from signalforge.github_sync import HttpResponse, fetch_issues, next_link, sync_repo
from signalforge.ingestion import IngestionError, validate
from signalforge.intelligence import cluster_feedback
from signalforge.optimizer import Constraints
from signalforge.pipeline import build_opportunities, rank_opportunities
from signalforge.store import Store, StoreError, new_workspace_id
from signalforge.synthetic import generate


@pytest.fixture(scope="module")
def world():
    c = cluster_feedback(validate(generate()).df, 0.9)
    a = Assumptions().validated()
    opps = build_opportunities(c, a)
    return c, a, opps, rank_opportunities(opps, a), {o["cluster_id"]: o for o in opps}


@pytest.fixture
def db():
    s = Store(":memory:")
    return s, s.ensure_workspace(new_workspace_id())


def _issue(n, upd="2026-10-01T00:00:00Z", pr=False):
    it = {"number": n, "title": f"Crash when exporting report {n}", "body": "Export fails with error", "user": {"login": f"u{n % 7}"},
          "labels": [{"name": "bug"}], "reactions": {"total_count": n % 3}, "created_at": "2026-09-01T00:00:00Z", "updated_at": upd,
          "state": "open", "html_url": f"https://github.com/o/r/issues/{n}"}
    if pr:
        it["pull_request"] = {}
    return it


class FakeGitHub:
    def __init__(self, pages, fail_first=0, rate_limited=False):
        self.pages, self.fail_first, self.rate_limited, self.urls = pages, fail_first, rate_limited, []

    def __call__(self, url):
        self.urls.append(url)
        if self.rate_limited:
            return HttpResponse(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1999999999"}, b'{"message":"rate"}')
        if self.fail_first:
            self.fail_first -= 1
            return HttpResponse(502, {}, b"")
        m = re.search(r"[?&]page=(\d+)", url)
        i = int(m.group(1)) if m else 1
        hdr = {"x-ratelimit-remaining": "50"}
        if i < len(self.pages):
            hdr["link"] = f'<https://api.github.com/repos/o/r/issues?page={i + 1}>; rel="next", <x>; rel="last"'
        return HttpResponse(200, hdr, json.dumps(self.pages[i - 1]).encode())


# ---------------- store ----------------
def test_decision_lifecycle_and_illegal_transitions(db):
    from signalforge.store import Decision
    s, ws = db
    with pytest.raises(StoreError):
        s.create_decision(ws, Decision("t", "x", "owner", "", {}, [], {}, "f"))  # rationale required
    did = s.create_decision(ws, Decision("t", "x", "owner", "because", {}, ["a#1"], {}, "f"))
    with pytest.raises(StoreError):
        s.transition(ws, did, "released")  # can't skip approval
    with pytest.raises(StoreError):
        s.transition(ws, did, "approved", note="")  # approval note required
    with pytest.raises(StoreError):
        s.add_ticket(ws, did, "t", "b", "p")  # no ticket before approval
    s.transition(ws, did, "approved", "pm", "ok")
    assert s.previous_status(ws, did) == "proposed"
    s.transition(ws, did, "reverted", "pm", "changed mind")
    s.transition(ws, did, "proposed", "pm", "")
    assert s.get_decision(ws, did).status == "proposed"
    assert [e["action"] for e in s.audit_events(ws)][:3] == ["decision_proposed", "decision_reverted", "decision_approved"]


def test_workspace_isolation_enforced_in_queries(db):
    from signalforge.store import Decision
    s, ws = db
    other = s.ensure_workspace(new_workspace_id())
    did = s.create_decision(ws, Decision("t", "x", "o", "r", {}, [], {}, "f"))
    assert s.decisions(other) == []
    with pytest.raises(StoreError):
        s.get_decision(other, did)
    with pytest.raises(StoreError):
        s.transition(other, did, "approved", note="x")
    with pytest.raises(StoreError):
        s.decisions("ws_doesnotexist123")
    with pytest.raises(StoreError):
        s.ensure_workspace("'; DROP TABLE decision;--")


def test_durable_file_store_survives_reopen(tmp_path):
    from signalforge.store import Decision
    p = str(tmp_path / "sf.db")
    s = Store(p); ws = s.ensure_workspace(new_workspace_id())
    did = s.create_decision(ws, Decision("t", "x", "o", "r", {}, [], {}, "f"))
    s.db.close()
    s2 = Store(p)
    assert s2.get_decision(ws, did).title == "t"


# ---------------- GitHub connector ----------------
def test_github_pagination_skips_prs_and_sets_cursor():
    fake = FakeGitHub([[_issue(i, f"2026-10-{i:02d}T00:00:00Z") for i in range(1, 4)] + [_issue(99, pr=True)],
                       [_issue(i, f"2026-10-{i:02d}T00:00:00Z") for i in range(4, 6)]])
    items, rep = fetch_issues("o/r", http=fake, sleep=lambda _: None)
    assert rep.pages == 2 and len(items) == 5 and rep.pull_requests_skipped == 1
    assert rep.cursor == "2026-10-05T00:00:00Z" and rep.rate_remaining == 50


def test_github_retries_then_succeeds_and_rate_limit_is_friendly():
    fake = FakeGitHub([[_issue(1)]], fail_first=2)
    _, rep = fetch_issues("o/r", http=fake, sleep=lambda _: None)
    assert rep.retries == 2
    with pytest.raises(IngestionError, match="rate limit"):
        fetch_issues("o/r", http=FakeGitHub([[]], rate_limited=True), sleep=lambda _: None)
    with pytest.raises(IngestionError, match="after 4 attempts"):
        fetch_issues("o/r", http=FakeGitHub([[_issue(1)]], fail_first=9), sleep=lambda _: None)
    with pytest.raises(IngestionError):
        fetch_issues("../etc/passwd", http=fake)


def test_github_incremental_sync_is_idempotent(db):
    s, ws = db
    rep = sync_repo(s, ws, "o/r", http=FakeGitHub([[_issue(i) for i in range(1, 6)]]), sleep=lambda _: None)
    assert "5 new" in rep.status and len(s.feedback(ws, "github:o/r")) == 5
    fake = FakeGitHub([[_issue(2, "2026-11-01T00:00:00Z") | {"title": "Crash when exporting report 2 (edited)"}]])
    rep2 = sync_repo(s, ws, "o/r", http=fake, sleep=lambda _: None)
    assert "since=2026-10-01T00:00:00Z" in fake.urls[0] and rep2.incremental
    assert "0 new, 1 updated" in rep2.status and len(s.feedback(ws, "github:o/r")) == 5
    rep3 = sync_repo(s, ws, "o/r", http=FakeGitHub([[_issue(2, "2026-11-01T00:00:00Z") | {"title": "Crash when exporting report 2 (edited)"}]]),
                     sleep=lambda _: None)
    assert "1 unchanged" in rep3.status
    assert next_link('<https://a?page=2>; rel="next"') == "https://a?page=2" and next_link(None) is None


# ---------------- telemetry ----------------
def test_telemetry_schema_validation():
    with pytest.raises(TM.TelemetryError, match="missing columns"):
        TM.validate_events(pd.DataFrame({"ts": [1]}))
    df = pd.DataFrame({"event_id": ["a", "a", "b", "c"], "ts": ["2026-01-01", "2026-01-01", "garbage", "2026-01-02"],
                       "account_id": ["x"] * 4, "user_id": ["u"] * 4, "event_name": ["signup", "signup", "key_action", "bogus"],
                       "feature": [None] * 4})
    out = TM.validate_events(df)
    assert len(out) == 1 and out.attrs["dropped_bad_ts"] == 1 and out.attrs["dropped_unknown_event"] == 1


@pytest.mark.parametrize("lift,expect", [(0.2, {"worked", "improved_below_target"}), (0.0, {"inconclusive"}),
                                         (-0.15, {"regressed", "guardrail_breached"})])
def test_evaluator_recovers_known_simulated_effect(lift, expect):
    ev = TM.simulate_events(["f", "g"], "f", "2026-09-01", lift)
    res = TM.evaluate_release(ev, "f", "2026-09-01")
    assert res["verdict"] in expect
    if lift:
        assert abs(res["lift"] - lift) < 0.12  # recovers the injected ground truth within sampling noise


def test_too_early_when_no_post_release_data():
    ev = TM.simulate_events(["f"], "f", "2026-09-01", 0.1, days=40)
    assert TM.evaluate_release(ev, "f", "2026-12-30")["verdict"] == "too_early"


# ---------------- evidence quality ----------------
def test_evidence_quality_flags_gaps(world):
    c, _a, opps, _r, _b = world
    q = EQ.assess(opps[0], c)
    assert 0 <= q["score"] <= 1 and q["grade"] in {"strong", "moderate", "weak"}
    thin = {**opps[0], "tickets": 3, "unique_accounts": 1, "ticket_ids": opps[0]["ticket_ids"][:3], "effort_weeks": None,
            "accounts_observed": 0}
    qt = EQ.assess(thin, c)
    assert qt["grade"] == "weak" and any("effort" in m for m in qt["missing"])


# ---------------- full loop ----------------
def test_end_to_end_evidence_to_outcome(world, db):
    c, a, _o, ranked, by = world
    s, ws = db
    row = ranked.iloc[0]
    opp = by[int(row.cluster_id)]
    did = LC.propose(s, ws, row, opp, c, a, "PM", "Highest risk-adjusted value", 0.5)
    d = s.get_decision(ws, did)
    assert d.evidence_ids and set(d.evidence_ids) <= set(opp["ticket_ids"]) and d.metrics["evidence_grade"]
    s.transition(ws, did, "approved", "PM", "go")
    LC.create_ticket(s, ws, did, opp, row, a)
    tk = s.tickets(ws, did)[0]
    assert "Acceptance criteria" in tk["body"] and d.feature_key in tk["body"] and tk["prd"].startswith("# PRD")
    with pytest.raises(StoreError):
        s.add_release(ws, did, "bad version!", "2026-09-01")
    s.add_release(ws, did, "v2.1", "2026-09-01")
    ev = TM.simulate_events([d.feature_key, "x"], d.feature_key, "2026-09-01", 0.2)
    res = LC.measure(s, ws, did, ev, "SIMULATED")
    assert s.get_decision(ws, did).status == "measured"
    assert "Recommended at rank #1" in res["forecast"]["summary"] and s.outcomes(ws, did)[0]["data_label"] == "SIMULATED"
    st = CopilotState(ranked, by, a, Constraints(capacity_weeks=20), loop={"decisions": s.decisions(ws),
                      "outcomes": {did: s.outcomes(ws, did)}, "quality": {o: EQ.assess(by[o], c) for o in by}})
    rec = answer(st, "Did the release work?")
    assert rec.tool == "outcome_review" and rec.table and "SIMULATED" in " ".join(rec.supporting_evidence)
    assert answer(st, "Create an engineering ticket.").tool == "ticket_preview"
    assert answer(st, "How solid is the evidence?").tool == "evidence_quality"


def test_outcome_review_without_release_is_honest(world):
    _c, a, _o, ranked, by = world
    rec = answer(CopilotState(ranked, by, a, Constraints(capacity_weeks=20), loop={}), "did it work?")
    assert "Nothing has been released" in rec.recommendation


def test_pdf_export_is_valid_pdf():
    pdf = markdown_to_pdf("# Title\n\n- bullet — dash\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nText with **bold** → arrow")
    assert pdf[:5] == b"%PDF-" and len(pdf) > 800
