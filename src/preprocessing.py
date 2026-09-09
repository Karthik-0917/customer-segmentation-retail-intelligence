"""Transaction cleaning, cancellation policy, and scaling utilities.

Transaction definition used throughout the project:
    A transaction/order is one unique InvoiceNo that survives the cleaning
    policy below. Multiple product rows on one invoice count as ONE order.

Cleaning policy (applied in this order, each step counted):
    1. Drop exact duplicate rows (repeated feed extracts, not real sales).
    2. Drop rows with missing CustomerID (cannot be attributed to a customer).
    3. Drop rows with unparseable InvoiceDate.
    4. Drop cancellation invoices (InvoiceNo starting with 'C', the
       documented UCI convention; 'A' rows are bad-debt adjustments and are
       removed by the price/quantity rule below).
    5. Drop rows with non-positive Quantity or non-positive UnitPrice
       (remaining returns, manual adjustments, zero-priced samples).

Legitimate high-value customers are retained: statistical outliers in spend
are meaningful business signal, not noise. Skew is handled later with a
log1p transform rather than record deletion.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CANCELLATION_PREFIX = "C"


def flag_cancellations(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of cancellation rows per the UCI InvoiceNo convention."""
    return df["InvoiceNo"].astype("string").str.upper().str.startswith(CANCELLATION_PREFIX)


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the documented cleaning policy and count every removal."""
    counts: dict[str, int] = {"rows_loaded": int(len(df))}

    step = df.drop_duplicates()
    counts["duplicate_rows_removed"] = counts["rows_loaded"] - len(step)

    before = len(step)
    step = step.dropna(subset=["CustomerID"])
    counts["missing_customer_id_removed"] = before - len(step)

    before = len(step)
    step = step.dropna(subset=["InvoiceDate"])
    counts["invalid_date_removed"] = before - len(step)

    cancels = flag_cancellations(step)
    counts["cancellation_rows_removed"] = int(cancels.sum())
    step = step[~cancels]

    invalid = (step["Quantity"] <= 0) | (step["UnitPrice"] <= 0)
    counts["non_positive_qty_or_price_removed"] = int(invalid.sum())
    step = step[~invalid]

    step = step.copy()
    step["CustomerID"] = step["CustomerID"].astype("int64").astype("string")
    step["Revenue"] = step["Quantity"] * step["UnitPrice"]
    counts["valid_rows"] = int(len(step))
    counts["valid_orders"] = int(step["InvoiceNo"].nunique())
    counts["customers"] = int(step["CustomerID"].nunique())
    return step.reset_index(drop=True), counts


def compute_reference_date(transactions: pd.DataFrame) -> pd.Timestamp:
    """Deterministic recency anchor: max(valid InvoiceDate) + 1 day.

    Using the dataset's own final purchase date (not the current system
    date) keeps Recency identical on every rerun of this retrospective
    analysis and prevents future-information leakage.
    """
    return transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def fit_scaler(matrix: np.ndarray | pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on the final (already transformed) feature matrix.

    K-Means is Euclidean: without standardisation, Monetary (range in the
    hundreds of thousands) would dominate Recency (range in the hundreds).
    """
    scaler = StandardScaler()
    scaler.fit(matrix)
    return scaler


def scale_matrix(scaler: StandardScaler, matrix: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Apply a previously fitted scaler (identical at train and inference)."""
    return scaler.transform(matrix)
