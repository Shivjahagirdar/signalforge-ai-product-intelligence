"""Portable saved analyses: a versioned JSON snapshot the visitor downloads and can re-upload later.

The public demo deliberately keeps no server-side storage (no accounts, no cross-visitor data). Saving is therefore
client-held: the file contains settings, assumptions, overrides, constraints and the audit trail - never raw uploads.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from .audit import Overrides

SCHEMA_VERSION = 1
WIDGET_KEYS = {"dataset": str, "thr": float, "exp": float, "rec": float, "eng": float, "infra": float, "sup": float,
               "impute": bool, "assumed": float, "semantic": bool}
MAX_BYTES = 1_000_000


def snapshot(widgets: dict, overrides: Overrides, constraints: dict, events: list[tuple], data_signature: list) -> bytes:
    body = {"schema": SCHEMA_VERSION, "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "widgets": {k: widgets[k] for k in WIDGET_KEYS if k in widgets and widgets[k] is not None},
            "overrides": json.loads(overrides.to_json()), "constraints": constraints,
            "audit": [list(e) for e in events], "data_signature": data_signature}
    return json.dumps(body, indent=1, default=str).encode()


def restore(raw: bytes) -> dict:
    """Validate and parse a snapshot. Raises ValueError with a user-safe message."""
    if len(raw) > MAX_BYTES:
        raise ValueError("Saved analysis is larger than 1 MB.")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("That isn't a SignalForge saved analysis (invalid JSON).") from exc
    if not isinstance(body, dict) or body.get("schema") != SCHEMA_VERSION:
        raise ValueError("Unsupported saved-analysis version.")
    w = {}
    for k, typ in WIDGET_KEYS.items():
        if k in body.get("widgets", {}):
            v = body["widgets"][k]
            if not isinstance(v, (int, float, str, bool)):
                raise ValueError(f"Invalid value for {k}.")
            w[k] = typ(v)
    ov = Overrides.from_json(json.dumps({"rename": {}, "merge": [], "irrelevant": [], "rank": {}, **body.get("overrides", {})}))
    c = body.get("constraints", {})
    cons = {"cap": float(c["cap"]) if c.get("cap") is not None else None, "minconf": float(c.get("minconf", 0.0)),
            "budget": float(c["budget"]) if c.get("budget") else None,
            "mandatory": [int(x) for x in c.get("mandatory", [])], "deps": [(int(a), int(b)) for a, b in c.get("deps", [])]}
    audit = [tuple(str(x)[:2000] for x in e[:3]) for e in body.get("audit", []) if isinstance(e, list) and len(e) >= 3][:5000]
    return {"widgets": w, "overrides": ov, "constraints": cons, "audit": audit, "data_signature": body.get("data_signature")}
