"""Load and validate the UCI Online Retail transaction dataset."""

from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH, RAW_COLUMNS


def load_transactions(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the raw Online Retail workbook and validate its schema.

    Raises FileNotFoundError with download instructions if the file is
    absent and ValueError if required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download the UCI Online Retail "
            "dataset (https://archive.ics.uci.edu/dataset/352/online+retail), "
            "extract it, and save the workbook as data/Online_Retail.xlsx. "
            "See data/README.md for full instructions."
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    validate_schema(df)
    # Normalise dtypes that matter downstream.
    df["InvoiceNo"] = df["InvoiceNo"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    df["Country"] = df["Country"].astype("string")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure every required raw column is present."""
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
