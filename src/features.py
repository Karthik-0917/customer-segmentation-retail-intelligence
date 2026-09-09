"""Customer-level feature engineering: RFM, behavioral features, selection.

Single source of truth for feature definitions used by training, inference,
the Streamlit app, and tests.

Candidate features (all computed from cleaned positive-purchase data):

    Recency          days since last valid purchase (vs reference date)
    Frequency        number of unique valid purchase invoices (orders)
    Monetary         total valid purchase revenue (sum of Quantity*UnitPrice)
    AvgOrderValue    Monetary / Frequency
    UniqueProducts   unique StockCode count per customer
    TotalQuantity    total units purchased
    ActiveDays       days between first and last valid purchase

Final clustering features: Recency, Frequency, Monetary (log1p-transformed,
then standardised). Selection evidence (computed on the actual dataset by
the pipeline and persisted to outputs/):

- Spearman correlations show heavy redundancy among the monetary-intensity
  candidates (e.g. Monetary vs TotalQuantity, Frequency vs ActiveDays,
  Frequency vs Monetary). Adding them would double-count the same
  purchasing-intensity behaviour.
- Candidate feature-set sweeps (K=2..8, identical preprocessing) showed the
  RFM set achieving the strongest silhouette profile; every augmented set
  lowered silhouette at every K.
- RFM is the standard, directly interpretable behavioural basis for retail
  segmentation.

Descriptive-only columns (AvgOrderValue, UniqueProducts, TotalQuantity,
ActiveDays, Country) are kept for profiling and the dashboard but are NOT
model inputs. CustomerID is an identifier only and is never a feature.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_SCHEMA_PATH

FEATURE_SCHEMA_VERSION = "1.0"
CLUSTERING_FEATURES = ["Recency", "Frequency", "Monetary"]
TRANSFORMATION = "log1p"
CANDIDATE_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "TotalQuantity",
    "ActiveDays",
]


def build_customer_features(
    transactions: pd.DataFrame, reference_date: pd.Timestamp
) -> pd.DataFrame:
    """Aggregate cleaned transactions into one row per customer."""
    grouped = transactions.groupby("CustomerID")
    customers = pd.DataFrame(
        {
            "Recency": (reference_date - grouped["InvoiceDate"].max()).dt.days,
            "Frequency": grouped["InvoiceNo"].nunique(),
            "Monetary": grouped["Revenue"].sum(),
            "UniqueProducts": grouped["StockCode"].nunique(),
            "TotalQuantity": grouped["Quantity"].sum(),
            "ActiveDays": (
                grouped["InvoiceDate"].max() - grouped["InvoiceDate"].min()
            ).dt.days,
            "Country": grouped["Country"].agg(lambda s: s.mode().iloc[0]),
        }
    )
    customers["AvgOrderValue"] = customers["Monetary"] / customers["Frequency"]
    return customers.reset_index()


def correlation_diagnostics(customers: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation matrix of candidate features (robust to skew)."""
    return customers[CANDIDATE_FEATURES].corr(method="spearman")


def skewness_diagnostics(customers: pd.DataFrame) -> pd.DataFrame:
    """Raw and log1p skewness for every candidate feature."""
    raw = customers[CANDIDATE_FEATURES].skew()
    logged = np.log1p(customers[CANDIDATE_FEATURES]).skew()
    return pd.DataFrame({"skew_raw": raw, "skew_log1p": logged}).round(3)


def transform_features(customers: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to the final clustering features, preserving order.

    Retail RFM features are heavily right-skewed; log1p compresses the long
    tails so K-Means centroids are not dictated by a handful of extreme
    customers, while retaining every customer record (no outlier deletion).
    """
    matrix = customers[CLUSTERING_FEATURES].astype("float64")
    if (matrix < 0).any().any():
        raise ValueError("Clustering features must be non-negative for log1p.")
    return np.log1p(matrix)


def compute_observed_ranges(customers: pd.DataFrame) -> dict[str, list[float]]:
    """Raw-scale [min, max] per clustering feature, for OOD warnings."""
    return {
        name: [float(customers[name].min()), float(customers[name].max())]
        for name in CLUSTERING_FEATURES
    }


def build_feature_schema(observed_ranges: dict[str, list[float]] | None = None) -> dict:
    """Machine-readable training/inference feature contract."""
    features = []
    for i, name in enumerate(CLUSTERING_FEATURES):
        entry = {
            "name": name,
            "order": i,
            "transformation": TRANSFORMATION,
            "dtype": "float64",
        }
        if observed_ranges and name in observed_ranges:
            lo, hi = observed_ranges[name]
            entry["observed_range"] = [float(lo), float(hi)]
        features.append(entry)
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "features": features,
        "feature_count": len(CLUSTERING_FEATURES),
        "scaler": "StandardScaler",
        "missing_value_policy": "reject",
        "definitions": {
            "Recency": "Days between the customer's latest valid purchase and the "
            "reference date (max valid InvoiceDate + 1 day).",
            "Frequency": "Number of unique valid purchase invoices (orders).",
            "Monetary": "Total valid purchase revenue (sum of Quantity x UnitPrice).",
        },
    }


def save_feature_schema(
    path: Path = FEATURE_SCHEMA_PATH,
    observed_ranges: dict[str, list[float]] | None = None,
) -> dict:
    """Persist the feature schema next to the model artifacts."""
    schema = build_feature_schema(observed_ranges)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2))
    return schema


def load_feature_schema(path: Path = FEATURE_SCHEMA_PATH) -> dict:
    """Load and sanity-check the persisted feature schema."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Feature schema not found: '{path}'. Train first: python -m src.pipeline"
        )
    schema = json.loads(Path(path).read_text())
    names = [f["name"] for f in sorted(schema["features"], key=lambda f: f["order"])]
    if len(names) != schema.get("feature_count"):
        raise ValueError("Corrupted feature schema: feature_count mismatch.")
    return schema
