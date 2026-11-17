
import pandas as pd
import pytest

from signalforge.ingestion import IngestionError, load_csv, safe_csv, validate
from signalforge.synthetic import generate


def test_synthetic_has_1000_records_and_dedups():
    df = generate()
    assert len(df) >= 1000
    r = validate(df)
    assert r.duplicates_removed >= 25
    assert r.acv_filled_from_account > 0


def test_missing_columns_rejected():
    with pytest.raises(IngestionError, match="Missing required columns"):
        validate(pd.DataFrame({"ticket_id": [1]}))


def test_rejects_non_csv_empty_and_garbage():
    with pytest.raises(IngestionError):
        load_csv(b"x", "a.xlsx")
    with pytest.raises(IngestionError):
        load_csv(b"  ", "a.csv")
    with pytest.raises(IngestionError):
        load_csv(b"a" * (11 * 1024 * 1024), "a.csv")


def test_invalid_values_are_coerced_with_warnings():
    df = generate(with_defects=False).head(20).astype({"severity": object, "frequency": object})
    df.loc[0, "severity"] = "abc"
    df.loc[1, "frequency"] = -5
    df.loc[2, "feedback_text"] = ""
    r = validate(df)
    assert r.dropped_rows == 1 and r.df["severity"].between(1, 5).all() and (r.df["frequency"] >= 1).all()
    assert any("severity" in w for w in r.warnings)


def test_csv_roundtrip_and_formula_injection_safe():
    csv = generate(with_defects=False).head(10).to_csv(index=False)
    assert len(load_csv(csv.encode(), "x.csv").df) == 10
    out = safe_csv(pd.DataFrame({"t": ["=SUM(A1)", "ok", "@cmd"]})).decode()
    assert "'=SUM" in out and "'@cmd" in out
