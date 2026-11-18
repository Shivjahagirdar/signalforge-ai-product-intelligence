"""Durable, workspace-scoped persistence for the closed-loop workflow (SQLite).

Canonical entities: Workspace, Feedback (ingested records with provenance), SyncState, Decision, Ticket, Release,
Outcome, AuditLog. Every read/write takes a ``workspace_id`` and every query filters on it: tenant isolation is enforced
in the data layer, not by hiding UI. Decisions are append-only versioned (status changes are new audit rows and the
previous status is kept) so human approvals/overrides are reversible.

Storage location: ``SIGNALFORGE_DB`` env var (default ``data/signalforge.db``). On the hosted public demo the disk is
the container's: it survives reruns, reloads and new browser sessions (resume with your workspace key) but NOT a
platform redeploy/restart. That limitation is surfaced in the UI; production would point this at managed Postgres.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
_WS_RX = re.compile(r"^ws_[A-Za-z0-9_-]{8,40}$")
DECISION_STATUSES = ("proposed", "approved", "rejected", "in_delivery", "released", "measured", "reverted")
_TRANSITIONS = {
    "proposed": {"approved", "rejected"},
    "approved": {"in_delivery", "rejected", "reverted"},
    "rejected": {"proposed"},
    "in_delivery": {"released", "reverted"},
    "released": {"measured", "reverted"},
    "measured": {"reverted"},
    "reverted": {"proposed"},
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StoreError(ValueError):
    """User-safe error (invalid id, illegal transition, cross-workspace access)."""


def new_workspace_id() -> str:
    return "ws_" + secrets.token_urlsafe(12).replace("-", "a").replace("_", "b")


def valid_workspace_id(ws: str) -> bool:
    return bool(ws and _WS_RX.fullmatch(ws))


@dataclass
class Decision:
    title: str
    cluster_label: str
    owner: str
    rationale: str
    assumptions: dict
    evidence_ids: list
    metrics: dict  # {"net_value":..., "roi":..., "effort_weeks":..., "evidence_score":...}
    feature_key: str
    target: dict = field(default_factory=dict)  # {"metric": "feature_adoption", "baseline": .., "target": ..}
    status: str = "proposed"
    id: int | None = None
    created_at: str = ""
    updated_at: str = ""
    approved_by: str = ""


DDL = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS workspace(id TEXT PRIMARY KEY, name TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS feedback(workspace_id TEXT NOT NULL, ticket_id TEXT NOT NULL, source TEXT, source_url TEXT,
  body TEXT, updated_at TEXT, ingested_at TEXT, PRIMARY KEY(workspace_id, ticket_id));
CREATE TABLE IF NOT EXISTS sync_state(workspace_id TEXT NOT NULL, source TEXT NOT NULL, cursor TEXT, last_run TEXT,
  last_status TEXT, fetched INTEGER, PRIMARY KEY(workspace_id, source));
CREATE TABLE IF NOT EXISTS decision(id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL, body TEXT NOT NULL,
  status TEXT NOT NULL, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS ticket(id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL, decision_id INTEGER NOT NULL,
  title TEXT, body TEXT, prd TEXT, created_at TEXT, external_ref TEXT);
CREATE TABLE IF NOT EXISTS release(id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL, decision_id INTEGER NOT NULL,
  version TEXT, released_at TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS outcome(id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL, decision_id INTEGER NOT NULL,
  body TEXT, data_label TEXT, measured_at TEXT);
CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL, ts TEXT, actor TEXT,
  action TEXT, entity TEXT, before TEXT, after TEXT);
CREATE INDEX IF NOT EXISTS ix_dec_ws ON decision(workspace_id);
CREATE INDEX IF NOT EXISTS ix_audit_ws ON audit_log(workspace_id);
"""


class Store:
    def __init__(self, path: str | None = None) -> None:
        path = path or os.environ.get("SIGNALFORGE_DB", "data/signalforge.db")
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.db.execute("PRAGMA foreign_keys=ON")
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(DDL)
        self.db.execute("INSERT OR IGNORE INTO meta(k,v) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        self.db.commit()

    @property
    def durable(self) -> bool:
        return self.path != ":memory:"

    # ---------- workspaces ----------
    def ensure_workspace(self, ws: str, name: str = "My workspace") -> str:
        if not valid_workspace_id(ws):
            raise StoreError("Workspace key is invalid. It looks like ws_ followed by 8-40 letters/digits.")
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO workspace(id,name,created_at) VALUES(?,?,?)", (ws, name[:80], now()))
            self.db.commit()
        return ws

    def workspace_exists(self, ws: str) -> bool:
        return valid_workspace_id(ws) and self.db.execute("SELECT 1 FROM workspace WHERE id=?", (ws,)).fetchone() is not None

    def _check(self, ws: str) -> None:
        if not self.workspace_exists(ws):
            raise StoreError("Unknown workspace.")

    # ---------- audit ----------
    def audit(self, ws: str, action: str, entity: str, before=None, after=None, actor: str = "user") -> None:
        self._check(ws)
        with self._lock:
            self.db.execute("INSERT INTO audit_log(workspace_id,ts,actor,action,entity,before,after) VALUES(?,?,?,?,?,?,?)",
                            (ws, now(), actor[:60], action[:60], entity[:120],
                             None if before is None else json.dumps(before, default=str)[:4000],
                             None if after is None else json.dumps(after, default=str)[:4000]))
            self.db.commit()

    def audit_events(self, ws: str, limit: int = 200) -> list[dict]:
        self._check(ws)
        rows = self.db.execute("SELECT ts,actor,action,entity,before,after FROM audit_log WHERE workspace_id=? ORDER BY id DESC LIMIT ?",
                               (ws, limit)).fetchall()
        return [dict(zip(("ts", "actor", "action", "entity", "before", "after"), r)) for r in rows]

    # ---------- feedback ingestion (idempotent upsert) ----------
    def upsert_feedback(self, ws: str, records: list[dict], source: str) -> dict:
        self._check(ws)
        ins = upd = same = 0
        with self._lock:
            for r in records:
                tid = str(r["ticket_id"])[:200]
                body = json.dumps(r, default=str, sort_keys=True)
                cur = self.db.execute("SELECT body FROM feedback WHERE workspace_id=? AND ticket_id=?", (ws, tid)).fetchone()
                if cur is None:
                    ins += 1
                elif cur[0] == body:
                    same += 1
                    continue
                else:
                    upd += 1
                self.db.execute("INSERT OR REPLACE INTO feedback(workspace_id,ticket_id,source,source_url,body,updated_at,ingested_at) "
                                "VALUES(?,?,?,?,?,?,?)", (ws, tid, source, str(r.get("source_url", ""))[:400], body,
                                                          str(r.get("updated_at", "")), now()))
            self.db.commit()
        return {"inserted": ins, "updated": upd, "unchanged": same}

    def feedback(self, ws: str, source: str | None = None) -> list[dict]:
        self._check(ws)
        q, p = "SELECT body FROM feedback WHERE workspace_id=?", [ws]
        if source:
            q += " AND source=?"; p.append(source)
        return [json.loads(b) for (b,) in self.db.execute(q + " ORDER BY ticket_id", p).fetchall()]

    def get_sync(self, ws: str, source: str) -> dict | None:
        self._check(ws)
        r = self.db.execute("SELECT cursor,last_run,last_status,fetched FROM sync_state WHERE workspace_id=? AND source=?", (ws, source)).fetchone()
        return dict(zip(("cursor", "last_run", "last_status", "fetched"), r)) if r else None

    def set_sync(self, ws: str, source: str, cursor: str | None, status: str, fetched: int) -> None:
        self._check(ws)
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO sync_state(workspace_id,source,cursor,last_run,last_status,fetched) VALUES(?,?,?,?,?,?)",
                            (ws, source[:200], cursor, now(), status[:200], fetched))
            self.db.commit()

    # ---------- decisions ----------
    def create_decision(self, ws: str, d: Decision, actor: str = "user") -> int:
        self._check(ws)
        if not d.title.strip() or not d.owner.strip():
            raise StoreError("A decision needs a title and an owner.")
        if not d.rationale.strip():
            raise StoreError("Write a rationale: decisions without reasons can't be reviewed later.")
        d.status, d.created_at = "proposed", now()
        d.updated_at = d.created_at
        body = asdict(d); body.pop("id")
        with self._lock:
            cur = self.db.execute("INSERT INTO decision(workspace_id,body,status,created_at,updated_at) VALUES(?,?,?,?,?)",
                                  (ws, json.dumps(body, default=str), d.status, d.created_at, d.updated_at))
            self.db.commit()
            did = int(cur.lastrowid)
        self.audit(ws, "decision_created", f"decision:{did}", None, {"title": d.title, "status": "proposed"}, actor)
        return did

    def get_decision(self, ws: str, did: int) -> Decision:
        self._check(ws)
        r = self.db.execute("SELECT id,body,status FROM decision WHERE workspace_id=? AND id=?", (ws, int(did))).fetchone()
        if r is None:
            raise StoreError("Decision not found in this workspace.")
        body = json.loads(r[1]); body["status"] = r[2]
        return Decision(id=r[0], **body)

    def decisions(self, ws: str) -> list[Decision]:
        self._check(ws)
        out = []
        for i, b, s in self.db.execute("SELECT id,body,status FROM decision WHERE workspace_id=? ORDER BY id DESC", (ws,)).fetchall():
            body = json.loads(b); body["status"] = s
            out.append(Decision(id=i, **body))
        return out

    def transition(self, ws: str, did: int, new: str, actor: str = "user", note: str = "") -> Decision:
        d = self.get_decision(ws, did)
        if new not in _TRANSITIONS.get(d.status, set()):
            raise StoreError(f"Can't move a decision from '{d.status}' to '{new}'.")
        if new in ("approved", "rejected", "reverted") and not note.strip():
            raise StoreError("Approvals, rejections and reversals need a note.")
        before = d.status
        d.status, d.updated_at = new, now()
        if new == "approved":
            d.approved_by = actor
        body = asdict(d); body.pop("id")
        with self._lock:
            self.db.execute("UPDATE decision SET body=?,status=?,updated_at=? WHERE workspace_id=? AND id=?",
                            (json.dumps(body, default=str), new, d.updated_at, ws, did))
            self.db.commit()
        self.audit(ws, f"decision_{new}", f"decision:{did}", {"status": before}, {"status": new, "note": note[:500]}, actor)
        return d

    def previous_status(self, ws: str, did: int) -> str | None:
        """Status before the last transition, for one-click undo."""
        self._check(ws)
        r = self.db.execute("SELECT before FROM audit_log WHERE workspace_id=? AND entity=? AND before IS NOT NULL ORDER BY id DESC LIMIT 1",
                            (ws, f"decision:{did}")).fetchone()
        return json.loads(r[0]).get("status") if r else None

    # ---------- ticket / release / outcome ----------
    def add_ticket(self, ws: str, did: int, title: str, body: str, prd: str, actor: str = "user") -> int:
        d = self.get_decision(ws, did)
        if d.status != "approved":
            raise StoreError("Tickets are only created for approved decisions (human approval first).")
        with self._lock:
            cur = self.db.execute("INSERT INTO ticket(workspace_id,decision_id,title,body,prd,created_at,external_ref) VALUES(?,?,?,?,?,?,?)",
                                  (ws, did, title[:200], body, prd, now(), None))
            self.db.commit()
            tid = int(cur.lastrowid)
        self.audit(ws, "ticket_created", f"ticket:{tid}", None, {"decision": did, "title": title[:200]}, actor)
        self.transition(ws, did, "in_delivery", actor, "ticket created")
        return tid

    def tickets(self, ws: str, did: int | None = None) -> list[dict]:
        self._check(ws)
        q, p = "SELECT id,decision_id,title,body,prd,created_at FROM ticket WHERE workspace_id=?", [ws]
        if did is not None:
            q += " AND decision_id=?"; p.append(int(did))
        return [dict(zip(("id", "decision_id", "title", "body", "prd", "created_at"), r)) for r in self.db.execute(q + " ORDER BY id DESC", p)]

    def add_release(self, ws: str, did: int, version: str, released_at: str, notes: str = "", actor: str = "user") -> int:
        d = self.get_decision(ws, did)
        if d.status != "in_delivery":
            raise StoreError("Only decisions in delivery can be released.")
        if not re.fullmatch(r"[\w.\-]{1,30}", version or ""):
            raise StoreError("Version must be 1-30 chars: letters, digits, dot, dash.")
        with self._lock:
            cur = self.db.execute("INSERT INTO release(workspace_id,decision_id,version,released_at,notes) VALUES(?,?,?,?,?)",
                                  (ws, did, version, released_at, notes[:1000]))
            self.db.commit()
            rid = int(cur.lastrowid)
        self.audit(ws, "release_recorded", f"release:{rid}", None, {"decision": did, "version": version, "date": released_at}, actor)
        self.transition(ws, did, "released", actor, f"released {version}")
        return rid

    def releases(self, ws: str, did: int) -> list[dict]:
        self._check(ws)
        return [dict(zip(("id", "version", "released_at", "notes"), r)) for r in self.db.execute(
            "SELECT id,version,released_at,notes FROM release WHERE workspace_id=? AND decision_id=? ORDER BY id DESC", (ws, int(did)))]

    def add_outcome(self, ws: str, did: int, result: dict, data_label: str, actor: str = "user") -> int:
        d = self.get_decision(ws, did)
        if d.status not in ("released", "measured"):
            raise StoreError("Measure outcomes after a release is recorded.")
        with self._lock:
            cur = self.db.execute("INSERT INTO outcome(workspace_id,decision_id,body,data_label,measured_at) VALUES(?,?,?,?,?)",
                                  (ws, did, json.dumps(result, default=str), data_label, now()))
            self.db.commit()
            oid = int(cur.lastrowid)
        self.audit(ws, "outcome_measured", f"outcome:{oid}", None, {"decision": did, "verdict": result.get("verdict"), "label": data_label}, actor)
        if d.status == "released":
            self.transition(ws, did, "measured", actor, "outcome measured")
        return oid

    def outcomes(self, ws: str, did: int) -> list[dict]:
        self._check(ws)
        return [{**json.loads(b), "data_label": lab, "measured_at": t} for b, lab, t in self.db.execute(
            "SELECT body,data_label,measured_at FROM outcome WHERE workspace_id=? AND decision_id=? ORDER BY id DESC", (ws, int(did)))]

    def wipe_workspace(self, ws: str) -> None:
        self._check(ws)
        with self._lock:
            for t in ("feedback", "sync_state", "decision", "ticket", "release", "outcome", "audit_log"):
                self.db.execute(f"DELETE FROM {t} WHERE workspace_id=?", (ws,))  # noqa: S608 - fixed table names
            self.db.commit()
        self.audit(ws, "workspace_reset", f"workspace:{ws[:10]}", None, None)
