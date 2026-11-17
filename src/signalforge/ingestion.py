"""CSV ingestion: schema validation, normalization, dedup, safe export."""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger("signalforge.ingestion")
REQUIRED = ["ticket_id", "account_id", "customer_segment", "annual_contract_value", "product_area",
            "feedback_text", "severity", "frequency", "created_at", "status", "estimated_engineering_effort"]
MAX_BYTES = 10 * 1024 * 1024
SEGMENTS = {"enterprise": "Enterprise", "mid-market": "Mid-Market", "midmarket": "Mid-Market", "smb": "SMB"}


class IngestionError(ValueError):
    """Raised for unrecoverable input problems (safe to show to users)."""


@dataclass
class IngestionResult:
    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    dropped_rows: int = 0
    duplicates_removed: int = 0
    acv_filled_from_account: int = 0


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def load_csv(data: bytes | str | io.BytesIO, filename: str = "upload.csv") -> IngestionResult:
    if not filename.lower().endswith(".csv"):
        raise IngestionError("Only .csv files are accepted.")
    raw = data.getvalue() if isinstance(data, io.BytesIO) else (data.encode() if isinstance(data, str) else data)
    if len(raw) > MAX_BYTES:
        raise IngestionError(f"File exceeds {MAX_BYTES // (1024 * 1024)} MB limit.")
    if not raw.strip():
        raise IngestionError("File is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # malformed CSV
        raise IngestionError(f"Could not parse CSV: {type(exc).__name__}") from exc
    return validate(df)


def validate(df: pd.DataFrame) -> IngestionResult:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise IngestionError(f"Missing required columns: {', '.join(missing)}")
    res = IngestionResult(df=df.copy())
    d = res.df
    n0 = len(d)
    d["feedback_text"] = d["feedback_text"].astype("string").str.strip()
    bad = d["feedback_text"].isna() | (d["feedback_text"] == "") | d["ticket_id"].isna() | d["account_id"].isna()
    d = d[~bad].copy()
    res.dropped_rows = int(bad.sum())
    if res.dropped_rows:
        res.warnings.append(f"Dropped {res.dropped_rows} rows missing ticket_id, account_id or feedback_text.")
    d["feedback_text"] = d["feedback_text"].astype(str)
    d["customer_segment"] = d["customer_segment"].astype(str).str.strip().str.lower().map(SEGMENTS).fillna("Unknown")
    for col, lo, hi, default in [("severity", 1, 5, None), ("frequency", 1, 10_000, 1)]:
        num = pd.to_numeric(d[col], errors="coerce")
        n_bad = int(num.isna().sum())
        fill = default if default is not None else (num.median() if num.notna().any() else 3)
        d[col] = num.fillna(fill).clip(lo, hi).round().astype(int)
        if n_bad:
            res.warnings.append(f"{n_bad} invalid/missing {col} values set to {fill:g}.")
    d["annual_contract_value"] = pd.to_numeric(d["annual_contract_value"], errors="coerce").where(lambda s: s >= 0)
    d["estimated_engineering_effort"] = pd.to_numeric(d["estimated_engineering_effort"], errors="coerce").where(lambda s: s > 0)
    d["created_at"] = pd.to_datetime(d["created_at"], errors="coerce")
    n_acv = int(d["annual_contract_value"].isna().sum())
    acct_acv = d.groupby("account_id")["annual_contract_value"].transform("max")
    d["annual_contract_value"] = d["annual_contract_value"].fillna(acct_acv)
    res.acv_filled_from_account = n_acv - int(d["annual_contract_value"].isna().sum())
    if res.acv_filled_from_account:
        res.warnings.append(f"{res.acv_filled_from_account} missing ACV values filled from the same account's other tickets.")
    still = d.loc[d["annual_contract_value"].isna(), "account_id"].nunique()
    if still:
        res.warnings.append(f"{still} accounts have no ACV; economics will impute segment median or exclude them.")
    d["_norm"] = d["feedback_text"].map(norm_text)
    dup = d.duplicated(subset=["account_id", "_norm"], keep="first")
    res.duplicates_removed = int(dup.sum())
    if res.duplicates_removed:
        res.warnings.append(f"Removed {res.duplicates_removed} exact duplicate tickets (same account + same text).")
    d = d[~dup].drop(columns="_norm").drop_duplicates(subset="ticket_id").reset_index(drop=True)
    if d.empty:
        raise IngestionError("No valid rows remain after validation.")
    if len(d) < n0 - res.dropped_rows - res.duplicates_removed:
        res.warnings.append("Duplicate ticket_id values were collapsed.")
    res.df = d
    log.info("ingested rows=%d dropped=%d dups=%d", len(d), res.dropped_rows, res.duplicates_removed)
    return res


def safe_csv(df: pd.DataFrame) -> bytes:
    """CSV export with spreadsheet formula-injection neutralised."""
    out = df.copy()
    for c in out.select_dtypes(include=["object", "string"]).columns:
        out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in "=+-@\t\r" and v else v)
    return out.to_csv(index=False).encode()
