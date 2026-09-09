"""Dataset loading and schema validation."""

import pandas as pd
import pytest

from src.config import RAW_COLUMNS
from src.data_loader import load_transactions, validate_schema


def test_missing_file_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_transactions(tmp_path / "Online_Retail.xlsx")


def test_validate_schema_rejects_missing_columns(raw_transactions):
    bad = raw_transactions.drop(columns=["UnitPrice"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(bad)


def test_loads_csv_and_normalises_dtypes(tmp_path, raw_transactions):
    path = tmp_path / "retail.csv"
    raw_transactions.to_csv(path, index=False)
    df = load_transactions(path)
    assert all(col in df.columns for col in RAW_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])
    assert df["InvoiceNo"].dtype == "string"
