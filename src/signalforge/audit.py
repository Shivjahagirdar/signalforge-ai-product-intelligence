"""SQLite-backed audit trail and persisted human overrides."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Overrides:
    rename: dict = field(default_factory=dict)  # cluster_id -> new label
    merge: list = field(default_factory=list)  # [[cid, cid, ...], ...]
    irrelevant: list = field(default_factory=list)  # ticket_ids excluded as evidence
    rank: dict = field(default_factory=dict)  # cluster_id -> {"rank": int, "rationale": str}

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(s: str) -> Overrides:
        d = json.loads(s)
        return Overrides(rename={int(k): v for k, v in d["rename"].items()}, merge=d["merge"],
                         irrelevant=d["irrelevant"], rank={int(k): v for k, v in d["rank"].items()})


class AuditStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, detail TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS overrides(id INTEGER PRIMARY KEY CHECK(id=1), body TEXT)")

    def log(self, kind: str, detail: str) -> None:
        self.db.execute("INSERT INTO audit(ts,kind,detail) VALUES(?,?,?)",
                        (datetime.now(UTC).isoformat(timespec="seconds"), kind, detail[:2000]))
        self.db.commit()

    def events(self) -> list[tuple]:
        return self.db.execute("SELECT ts,kind,detail FROM audit ORDER BY id DESC").fetchall()

    def save_overrides(self, o: Overrides) -> None:
        self.db.execute("INSERT OR REPLACE INTO overrides(id,body) VALUES(1,?)", (o.to_json(),))
        self.db.commit()

    def load_overrides(self) -> Overrides:
        r = self.db.execute("SELECT body FROM overrides WHERE id=1").fetchone()
        return Overrides.from_json(r[0]) if r else Overrides()

    def reset(self) -> None:
        self.db.execute("DELETE FROM audit")
        self.db.execute("DELETE FROM overrides")
        self.db.commit()
        self.log("reset", "All local data and overrides cleared.")
