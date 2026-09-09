"""Shared synthetic fixtures matching the UCI Online Retail schema.

Tests never require the real raw dataset.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def raw_transactions() -> pd.DataFrame:
    """Synthetic transaction feed with known dirty rows.

    60 clean multi-row invoices from 20 customers plus, appended:
    1 exact duplicate row, 2 missing-CustomerID rows, 1 cancellation
    invoice row, 1 negative-quantity row, 1 zero-price row.
    """
    rng = np.random.default_rng(7)
    rows = []
    base = pd.Timestamp("2011-01-01 10:00:00")
    for invoice in range(60):
        customer = 12346 + int(rng.integers(0, 20))
        date = base + pd.Timedelta(days=int(rng.integers(0, 300)))
        for line in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "InvoiceNo": f"5{invoice:05d}",
                    "StockCode": f"P{line:03d}",
                    "Description": f"PRODUCT {line}",
                    "Quantity": int(rng.integers(1, 20)),
                    "InvoiceDate": date,
                    "UnitPrice": round(float(rng.uniform(0.5, 25.0)), 2),
                    "CustomerID": float(customer),
                    "Country": "United Kingdom" if customer % 3 else "France",
                }
            )
    df = pd.DataFrame(rows)
    dirty = pd.DataFrame(
        [
            df.iloc[0].to_dict(),  # exact duplicate
            {**df.iloc[1].to_dict(), "CustomerID": np.nan},
            {**df.iloc[2].to_dict(), "CustomerID": np.nan},
            {**df.iloc[3].to_dict(), "InvoiceNo": "C99999", "Quantity": -5},
            {**df.iloc[4].to_dict(), "Quantity": -2},
            {**df.iloc[5].to_dict(), "UnitPrice": 0.0},
        ]
    )
    out = pd.concat([df, dirty], ignore_index=True)
    for col in ("InvoiceNo", "StockCode", "Description", "Country"):
        out[col] = out[col].astype("string")
    return out


@pytest.fixture()
def clean_customers(raw_transactions) -> pd.DataFrame:
    """Customer-level features built from the synthetic transactions."""
    from src.features import build_customer_features
    from src.preprocessing import clean_transactions, compute_reference_date

    transactions, _ = clean_transactions(raw_transactions)
    reference = compute_reference_date(transactions)
    return build_customer_features(transactions, reference)
